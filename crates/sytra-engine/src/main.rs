use std::{env, path::PathBuf, process::ExitCode, sync::Arc, time::Duration};

use serde::Serialize;
use sytra_engine::{
    bf16_tile_cpu, bf16_tile_matmul_cpu, bf16_transpose_tile_cpu, bf16_transpose_tile_matmul_cpu,
    compiled_adapters, estimate_io_performance, expert_swiglu, floating_gated_expert_batch,
    floating_gated_expert_batch_with_cuda, int4_group_matvec, plan_memory_envelope,
    plan_memory_envelope_with_capabilities, validate_compiled_contract, validate_forward_contract,
    validate_model_config, verify_kimi_oracle, verify_mixtral_oracle, Accelerator, ActivationKind,
    AttentionKind, CudaAccelerator, DenseTensorStore, ExpertKey, ExpertStore, GenerationConfig,
    GenerationRuntime, IoPerformanceEstimate, KimiK27Config, KimiRuntime, KimiRuntimeOptions,
    MemoryEnvelope, MixtralRuntime, MixtralRuntimeOptions, ModelGenerator, ModelTokenizer,
    NoAccelerator, OpenAiDraftModel, OpenAiServer, OracleSuite, PackedInt4Bf16View,
    PackedInt4Matrix, ResidencyManager, ResidentTensor, RuntimeCompletionBackend, RuntimeManifest,
    WeightedMirror,
};

#[derive(Debug, Serialize)]
struct PlacementSummary {
    adapter: String,
    model_type: String,
    forward_verified: bool,
    dense_bytes: u64,
    expert_bytes: u64,
    ram_dense_budget_bytes: u64,
    accelerator_dense_budget_bytes: u64,
    ram_expert_budget_bytes: u64,
    accelerator_expert_budget_bytes: u64,
    storage_dense_bytes: u64,
    storage_expert_bytes: u64,
    memory_envelope: MemoryEnvelope,
    io_performance: IoPerformanceEstimate,
}

fn usage() {
    eprintln!(
        "sytra-engine doctor --model PATH [--deep]\n\
         sytra-engine plan --model PATH [--ram-limit-mb N] [--accelerator-limit-mb N]\n\
             [--context N] [--verification-positions N] [--dense-tile-mb N] [--kv-scalar-bytes N]\n\
             [--ram-dense-mb N] [--ram-expert-mb N]\n\
             [--accelerator-dense-mb N] [--accelerator-expert-mb N]\n\
             [--storage-bandwidth-mbps N] [--target-tps N]\n\
         sytra-engine self-test --model PATH [--cuda-device N]\n\
         sytra-engine cuda-check [--cuda-device N] [--bytes N]\n\
         sytra-engine kimi-k27-check --model PATH\n\
         sytra-engine fingerprint --model PATH\n\
         sytra-engine oracle-check --model PATH [--cuda-device N]\n\
         sytra-engine benchmark --model PATH [--prompt TEXT] [--max-tokens N]\n\
             [--iterations N] [--warmup-tokens N] [--target-tps N] [--cuda-device N]\n\
         sytra-engine kimi-k27-cuda-check [--cuda-device N]\n\
         sytra-engine list-adapters\n\
         sytra-engine serve --model PATH [--host HOST] [--port N] [--served-model-name NAME]\n\
             [--max-concurrent-requests N] [--cuda-device N]\n\
             [--draft-url http://127.0.0.1:PORT] [--draft-model NAME] [--draft-timeout-ms N]"
    );
}

fn value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn number(args: &[String], flag: &str, default: u64) -> Result<u64, String> {
    match value(args, flag) {
        Some(raw) => raw
            .parse()
            .map_err(|_| format!("{flag} must be a non-negative integer")),
        None => Ok(default),
    }
}

fn allocate_cache(total: u64, requested_dense: u64, requested_expert: u64) -> (u64, u64) {
    let requested_total = requested_dense.saturating_add(requested_expert);
    if requested_total <= total {
        return (requested_dense, requested_expert);
    }
    if requested_total == 0 {
        return (0, 0);
    }
    let dense =
        ((u128::from(total) * u128::from(requested_dense)) / u128::from(requested_total)) as u64;
    (dense, total.saturating_sub(dense))
}

fn load(args: &[String]) -> Result<(PathBuf, RuntimeManifest), String> {
    let root = PathBuf::from(value(args, "--model").ok_or("--model is required")?);
    let manifest = RuntimeManifest::load(&root).map_err(|error| error.to_string())?;
    manifest
        .validate(&root)
        .map_err(|error| error.to_string())?;
    Ok((root, manifest))
}

fn mirrors(args: &[String]) -> Vec<WeightedMirror> {
    value(args, "--mirror")
        .map(|raw| {
            raw.split(';')
                .filter(|path| !path.trim().is_empty())
                .map(|path| WeightedMirror {
                    root: path.trim().into(),
                    weight: 1,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn requested_memory_envelope(
    args: &[String],
    manifest: &RuntimeManifest,
) -> Result<MemoryEnvelope, String> {
    let mib = 1024 * 1024;
    let accelerator_limit = number(args, "--accelerator-limit-mb", 0)? * mib;
    let accelerator_kv_supported = cfg!(target_os = "windows")
        && accelerator_limit > 0
        && manifest.architecture.attention == AttentionKind::Standard;
    plan_memory_envelope_with_capabilities(
        manifest,
        number(args, "--context", 4096)?,
        number(args, "--ram-limit-mb", 8192)? * mib,
        accelerator_limit,
        number(args, "--kv-scalar-bytes", 2)?,
        number(args, "--dense-tile-mb", 64)? * mib,
        number(args, "--verification-positions", 8)?,
        accelerator_kv_supported,
    )
    .map_err(|error| error.to_string())
}

fn optional_mib(args: &[String], flag: &str) -> Result<Option<u64>, String> {
    value(args, flag)
        .map(|raw| {
            raw.parse::<u64>()
                .map_err(|_| format!("{flag} must be a non-negative integer"))?
                .checked_mul(1024 * 1024)
                .ok_or_else(|| format!("{flag} is too large"))
        })
        .transpose()
}

fn build_kimi_runtime(
    args: &[String],
    root: &PathBuf,
    manifest: &RuntimeManifest,
    memory: &MemoryEnvelope,
) -> Result<KimiRuntime, String> {
    let cuda_device = if memory.accelerator_limit_bytes > 0 {
        let ordinal = number(args, "--cuda-device", 0)?;
        Some(
            i32::try_from(ordinal)
                .map_err(|_| "--cuda-device is outside the i32 range".to_string())?,
        )
    } else {
        None
    };
    KimiRuntime::new(
        root,
        manifest.clone(),
        memory.clone(),
        mirrors(args),
        KimiRuntimeOptions {
            cuda_device,
            dense_tile_bytes: number(args, "--dense-tile-mb", 64)?
                .checked_mul(1024 * 1024)
                .ok_or("--dense-tile-mb is too large")?,
            ram_dense_cache_bytes: optional_mib(args, "--ram-dense-mb")?,
            ram_expert_cache_bytes: optional_mib(args, "--ram-expert-mb")?,
            accelerator_expert_cache_bytes: optional_mib(args, "--accelerator-expert-mb")?,
        },
    )
    .map_err(|error| error.to_string())
}

fn build_mixtral_runtime(
    args: &[String],
    root: &PathBuf,
    manifest: &RuntimeManifest,
    memory: &MemoryEnvelope,
) -> Result<MixtralRuntime, String> {
    let cuda_device = if memory.accelerator_limit_bytes > 0 {
        Some(
            i32::try_from(number(args, "--cuda-device", 0)?)
                .map_err(|_| "--cuda-device is outside the i32 range".to_string())?,
        )
    } else {
        None
    };
    MixtralRuntime::new(
        root,
        manifest.clone(),
        memory.clone(),
        mirrors(args),
        MixtralRuntimeOptions {
            cuda_device,
            dense_tile_bytes: number(args, "--dense-tile-mb", 64)?
                .checked_mul(1024 * 1024)
                .ok_or("--dense-tile-mb is too large")?,
            ram_dense_cache_bytes: optional_mib(args, "--ram-dense-mb")?,
            ram_expert_cache_bytes: optional_mib(args, "--ram-expert-mb")?,
            accelerator_expert_cache_bytes: optional_mib(args, "--accelerator-expert-mb")?,
        },
    )
    .map_err(|error| error.to_string())
}

enum VerifiedRuntime {
    Kimi(KimiRuntime, sytra_engine::OracleReport),
    Mixtral(MixtralRuntime, sytra_engine::OracleReport),
}

impl VerifiedRuntime {
    fn report(&self) -> &sytra_engine::OracleReport {
        match self {
            Self::Kimi(_, report) | Self::Mixtral(_, report) => report,
        }
    }

    fn diagnostics(&self) -> Result<serde_json::Value, String> {
        match self {
            Self::Kimi(runtime, _) => Ok(serde_json::json!({
                "placement": runtime.placement(),
                "runtime_metrics": runtime.metrics().map_err(|error| error.to_string())?,
            })),
            Self::Mixtral(runtime, _) => Ok(serde_json::json!({
                "placement": runtime.placement(),
                "runtime_metrics": runtime.metrics().map_err(|error| error.to_string())?,
            })),
        }
    }
}

fn verify_runtime_oracle(
    args: &[String],
    root: &PathBuf,
    manifest: &RuntimeManifest,
    memory: &MemoryEnvelope,
) -> Result<VerifiedRuntime, String> {
    let suite = OracleSuite::load(root).map_err(|error| error.to_string())?;
    match manifest.architecture.adapter.as_str() {
        sytra_engine::kimi_k27::ADAPTER_ID => {
            let runtime = build_kimi_runtime(args, root, manifest, memory)?;
            let report =
                verify_kimi_oracle(root, &runtime, &suite).map_err(|error| error.to_string())?;
            Ok(VerifiedRuntime::Kimi(runtime, report))
        }
        "sytra-mixtral" | "sytra-qwen3-moe" | "sytra-qwen2-moe" | "sytra-olmoe"
        | "sytra-granite-moe" => {
            let runtime = build_mixtral_runtime(args, root, manifest, memory)?;
            let report =
                verify_mixtral_oracle(root, &runtime, &suite).map_err(|error| error.to_string())?;
            Ok(VerifiedRuntime::Mixtral(runtime, report))
        }
        adapter => Err(format!(
            "{adapter} has no checkpoint oracle executor in this build"
        )),
    }
}

fn doctor(args: &[String]) -> Result<(), String> {
    let (root, manifest) = load(args)?;
    let descriptor = validate_compiled_contract(&manifest).map_err(|error| error.to_string())?;
    validate_model_config(&root, &manifest, descriptor).map_err(|error| error.to_string())?;
    let memory = requested_memory_envelope(args, &manifest)?;
    if descriptor.id == sytra_engine::kimi_k27::ADAPTER_ID {
        KimiK27Config::load(&root)
            .and_then(|config| config.validate_runtime_manifest(&manifest))
            .map_err(|error| error.to_string())?;
    }
    let configured_mirrors = mirrors(args);
    let store = ExpertStore::new(
        &root,
        configured_mirrors.clone(),
        manifest.storage.experts.iter().cloned(),
    );
    let dense_store = DenseTensorStore::new(
        &root,
        configured_mirrors,
        manifest.storage.dense_tensors.iter().cloned(),
    );
    let keys: Vec<_> = if args.iter().any(|arg| arg == "--deep") {
        manifest
            .storage
            .experts
            .iter()
            .map(|entry| entry.key())
            .collect()
    } else {
        let mut sampled = Vec::new();
        for layer in 0..manifest.architecture.num_layers {
            if let Some(first) = manifest
                .storage
                .experts
                .iter()
                .find(|entry| entry.layer == layer)
            {
                sampled.push(first.key());
            }
            if let Some(last) = manifest
                .storage
                .experts
                .iter()
                .rev()
                .find(|entry| entry.layer == layer)
            {
                if sampled.last() != Some(&last.key()) {
                    sampled.push(last.key());
                }
            }
        }
        sampled
    };
    for key in keys {
        store.read(key).map_err(|error| error.to_string())?;
    }
    if args.iter().any(|arg| arg == "--deep") {
        for tensor in &manifest.storage.dense_tensors {
            dense_store
                .read(&tensor.tensor)
                .map_err(|error| error.to_string())?;
        }
    } else {
        for tensor in manifest
            .storage
            .dense_tensors
            .first()
            .into_iter()
            .chain(manifest.storage.dense_tensors.last())
        {
            dense_store
                .read(&tensor.tensor)
                .map_err(|error| error.to_string())?;
        }
    }
    let forward_compiled = validate_forward_contract(&manifest).is_ok();
    let oracle = if forward_compiled {
        Some(
            verify_runtime_oracle(args, &root, &manifest, &memory)?
                .report()
                .clone(),
        )
    } else {
        None
    };
    let ready = forward_compiled && memory.feasible && oracle.is_some();
    println!(
        "{}",
        serde_json::json!({
            "ready": ready,
            "storage_valid": true,
            "memory": memory,
            "adapter": manifest.architecture.adapter,
            "compiled_forward_kernel": forward_compiled,
            "metadata_forward_verified_ignored": manifest.architecture.forward_verified,
            "checkpoint_oracle": oracle,
            "experts_indexed": manifest.storage.experts.len(),
            "dense_tensors_indexed": manifest.storage.dense_tensors.len(),
            "note": if ready {
                "compiled kernel passed the checkpoint-bound runtime oracle"
            } else {
                "storage is valid, but serving remains locked without a compiled kernel and passing runtime oracle"
            }
        })
    );
    if !ready {
        return Err("adapter is not runtime-oracle-verified and cannot serve tokens".into());
    }
    Ok(())
}

fn kimi_k27_check(args: &[String]) -> Result<(), String> {
    let root = PathBuf::from(value(args, "--model").ok_or("--model is required")?);
    let config = KimiK27Config::load(&root).map_err(|error| error.to_string())?;
    let fingerprint = sytra_engine::checkpoint_fingerprint(&root).ok();
    let oracle_present = root.join(sytra_engine::ORACLE_FILE).is_file();
    println!(
        "{}",
        serde_json::json!({
            "adapter": sytra_engine::kimi_k27::ADAPTER_ID,
            "config_exact": true,
            "text_only": true,
            "layers": config.num_hidden_layers,
            "routed_experts": config.n_routed_experts,
            "experts_per_token": config.num_experts_per_tok,
            "expert_format": "packed_int4_group32",
            "kv_cache": "compressed_mla",
            "kv_bytes_per_token_per_layer_bf16": config.compressed_kv_bytes_per_token(2),
            "materialized_kv_bytes_per_token_per_layer_bf16":
                config.materialized_kv_bytes_per_token(2),
            "compiled_forward_kernel": true,
            "checkpoint_fingerprint": fingerprint,
            "checkpoint_oracle_present": oracle_present,
            "checkpoint_oracle_passed": false,
            "serving_unlocked": false
        })
    );
    Ok(())
}

fn fingerprint(args: &[String]) -> Result<(), String> {
    let root = PathBuf::from(value(args, "--model").ok_or("--model is required")?);
    let fingerprint =
        sytra_engine::checkpoint_fingerprint(&root).map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "model": root,
            "model_fingerprint": fingerprint,
        })
    );
    Ok(())
}

fn kimi_k27_cuda_check(args: &[String]) -> Result<(), String> {
    let ordinal = number(args, "--cuda-device", 0)?;
    let ordinal =
        i32::try_from(ordinal).map_err(|_| "--cuda-device is outside the i32 range".to_string())?;
    let rows = 3;
    let cols = 64;
    let quantized: Vec<i8> = (-8..8).cycle().take(rows * cols).collect();
    let mut packed = vec![0_u32; rows * cols / 8];
    for (index, value) in quantized.iter().enumerate() {
        packed[index / 8] |= ((*value + 8) as u32) << ((index % 8) * 4);
    }
    let requested_scales: Vec<f32> = (0..rows * (cols / 32))
        .map(|index| 0.125 * (index + 1) as f32)
        .collect();
    let scales_bf16: Vec<u16> = requested_scales
        .iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();
    let scales: Vec<f32> = scales_bf16
        .iter()
        .map(|value| f32::from_bits((*value as u32) << 16))
        .collect();
    let input: Vec<f32> = (0..cols)
        .map(|index| (index as f32 - 17.0) / 13.0)
        .collect();
    let reference =
        int4_group_matvec(&packed, &scales, rows, cols, 32, &input).map_err(|e| e.to_string())?;
    let cuda = CudaAccelerator::new(ordinal)?;
    let actual = cuda.int4_group32_bf16_matvec(&packed, &scales_bf16, rows, cols, &input)?;
    let max_absolute_error = reference
        .iter()
        .zip(&actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if max_absolute_error > 1e-4 {
        return Err(format!(
            "Kimi INT4 CUDA kernel differs from the reference by {max_absolute_error}"
        ));
    }
    let mut resident_payload = Vec::new();
    resident_payload.extend(packed.iter().flat_map(|value| value.to_le_bytes()));
    let scale_offset = resident_payload.len();
    resident_payload.extend(scales_bf16.iter().flat_map(|value| value.to_le_bytes()));
    let resident = cuda.upload(ExpertKey::new(0, 99), &resident_payload)?;
    let resident_actual =
        cuda.resident_int4_group32_bf16_matvec(&resident, 0, scale_offset, rows, cols, &input);
    let batch_positions = 3;
    let batch_input: Vec<f32> = (0..batch_positions)
        .flat_map(|position| {
            input
                .iter()
                .map(move |value| *value + position as f32 * 0.125)
        })
        .collect();
    let batch_reference: Vec<f32> = batch_input
        .chunks_exact(cols)
        .map(|values| int4_group_matvec(&packed, &scales, rows, cols, 32, values))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?
        .into_iter()
        .flatten()
        .collect();
    let packed_bytes: Vec<u8> = packed
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let scale_bytes: Vec<u8> = scales_bf16
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let batch_actual = cuda.int4_group32_bf16_bytes_matmul(
        &packed_bytes,
        &scale_bytes,
        rows,
        cols,
        batch_positions,
        &batch_input,
    )?;
    let resident_batch_actual = cuda.resident_int4_group32_bf16_matmul(
        &resident,
        0,
        scale_offset,
        rows,
        cols,
        batch_positions,
        &batch_input,
    );
    cuda.release(&resident);
    let resident_actual = resident_actual?;
    let resident_max_absolute_error = reference
        .iter()
        .zip(&resident_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if resident_max_absolute_error > 1e-4 {
        return Err(format!(
            "resident Kimi INT4 CUDA kernel differs from the reference by {resident_max_absolute_error}"
        ));
    }
    let batch_max_absolute_error = batch_reference
        .iter()
        .zip(&batch_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    let resident_batch_actual = resident_batch_actual?;
    let resident_batch_max_absolute_error = batch_reference
        .iter()
        .zip(&resident_batch_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if batch_max_absolute_error > 1e-4 || resident_batch_max_absolute_error > 1e-4 {
        return Err(format!(
            "batched Kimi INT4 CUDA kernel differs from the reference by host={batch_max_absolute_error} resident={resident_batch_max_absolute_error}"
        ));
    }

    let expert_hidden = 32;
    let expert_intermediate = 32;
    let expert_output = 3;
    let make_matrix = |rows: usize, cols: usize, phase: usize| {
        let values: Vec<i8> = (0..rows * cols)
            .map(|index| ((index + phase) % 16) as i8 - 8)
            .collect();
        let mut packed = vec![0_u32; rows * cols / 8];
        for (index, value) in values.iter().enumerate() {
            packed[index / 8] |= ((*value + 8) as u32) << ((index % 8) * 4);
        }
        let scales = vec![((0.0625_f32).to_bits() >> 16) as u16; rows * (cols / 32)];
        (packed, scales)
    };
    let (gate_packed, gate_scales) = make_matrix(expert_intermediate, expert_hidden, 0);
    let (up_packed, up_scales) = make_matrix(expert_intermediate, expert_hidden, 3);
    let (down_packed, down_scales) = make_matrix(expert_output, expert_intermediate, 7);
    let expert_input: Vec<f32> = (0..expert_hidden)
        .map(|index| (index as f32 - 11.0) / 17.0)
        .collect();
    let as_f32_scales = |values: &[u16]| {
        values
            .iter()
            .map(|value| f32::from_bits((*value as u32) << 16))
            .collect::<Vec<_>>()
    };
    let gate_scales_f32 = as_f32_scales(&gate_scales);
    let up_scales_f32 = as_f32_scales(&up_scales);
    let down_scales_f32 = as_f32_scales(&down_scales);
    let expert_reference = expert_swiglu(
        &expert_input,
        PackedInt4Matrix {
            packed: &gate_packed,
            scales: &gate_scales_f32,
            rows: expert_intermediate,
            cols: expert_hidden,
            group_size: 32,
        },
        PackedInt4Matrix {
            packed: &up_packed,
            scales: &up_scales_f32,
            rows: expert_intermediate,
            cols: expert_hidden,
            group_size: 32,
        },
        PackedInt4Matrix {
            packed: &down_packed,
            scales: &down_scales_f32,
            rows: expert_output,
            cols: expert_intermediate,
            group_size: 32,
        },
    )
    .map_err(|error| error.to_string())?;
    let expert_actual = cuda.expert_swiglu_bf16(
        &expert_input,
        PackedInt4Bf16View {
            packed: &gate_packed,
            scales: &gate_scales,
            rows: expert_intermediate,
            cols: expert_hidden,
        },
        PackedInt4Bf16View {
            packed: &up_packed,
            scales: &up_scales,
            rows: expert_intermediate,
            cols: expert_hidden,
        },
        PackedInt4Bf16View {
            packed: &down_packed,
            scales: &down_scales,
            rows: expert_output,
            cols: expert_intermediate,
        },
    )?;
    let expert_max_absolute_error = expert_reference
        .iter()
        .zip(&expert_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if expert_max_absolute_error > 1e-4 {
        return Err(format!(
            "Kimi CUDA expert differs from the reference by {expert_max_absolute_error}"
        ));
    }
    println!(
        "{}",
        serde_json::json!({
            "adapter": sytra_engine::kimi_k27::ADAPTER_ID,
            "kernel": "packed_int4_group32_matvec",
            "scale_dtype": "bf16",
            "device": ordinal,
            "rows": rows,
            "cols": cols,
            "reference_match": true,
            "max_absolute_error": max_absolute_error,
            "resident_reference_match": true,
            "resident_max_absolute_error": resident_max_absolute_error,
            "batch_positions": batch_positions,
            "batch_reference_match": true,
            "batch_max_absolute_error": batch_max_absolute_error,
            "resident_batch_reference_match": true,
            "resident_batch_max_absolute_error": resident_batch_max_absolute_error,
            "expert_swiglu_reference_match": true,
            "expert_swiglu_max_absolute_error": expert_max_absolute_error
        })
    );
    Ok(())
}

fn oracle_check(args: &[String]) -> Result<(), String> {
    let (root, manifest) = load(args)?;
    let descriptor = validate_compiled_contract(&manifest).map_err(|error| error.to_string())?;
    validate_model_config(&root, &manifest, descriptor).map_err(|error| error.to_string())?;
    validate_forward_contract(&manifest).map_err(|error| error.to_string())?;
    let memory = requested_memory_envelope(args, &manifest)?;
    let runtime = verify_runtime_oracle(args, &root, &manifest, &memory)?;
    let diagnostics = runtime.diagnostics()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "oracle": runtime.report(),
            "memory": memory,
            "runtime": diagnostics,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn plan(args: &[String]) -> Result<(), String> {
    let (_, manifest) = load(args)?;
    let ram_expert = number(args, "--ram-expert-mb", 4096)? * 1024 * 1024;
    let accelerator_expert = number(args, "--accelerator-expert-mb", 0)? * 1024 * 1024;
    let ram_dense = number(args, "--ram-dense-mb", 4096)? * 1024 * 1024;
    let accelerator_dense = number(args, "--accelerator-dense-mb", 0)? * 1024 * 1024;
    let ram_limit = number(
        args,
        "--ram-limit-mb",
        (ram_dense + ram_expert) / (1024 * 1024),
    )? * 1024
        * 1024;
    let accelerator_limit = number(
        args,
        "--accelerator-limit-mb",
        (accelerator_dense + accelerator_expert) / (1024 * 1024),
    )? * 1024
        * 1024;
    let context = number(args, "--context", 4096)?;
    let dense_tile = number(args, "--dense-tile-mb", 64)? * 1024 * 1024;
    let kv_scalar_bytes = number(args, "--kv-scalar-bytes", 2)?;
    let verification_positions = number(args, "--verification-positions", 8)?;
    let memory_envelope = plan_memory_envelope(
        &manifest,
        context,
        ram_limit,
        accelerator_limit,
        kv_scalar_bytes,
        dense_tile,
        verification_positions,
    )
    .map_err(|error| error.to_string())?;
    let (ram_dense, ram_expert) =
        allocate_cache(memory_envelope.ram_cache_bytes, ram_dense, ram_expert);
    let (accelerator_dense, accelerator_expert) = allocate_cache(
        memory_envelope.accelerator_cache_bytes,
        accelerator_dense,
        accelerator_expert,
    );
    let expert_bytes = manifest.expert_bytes();
    // Host and accelerator expert caches commonly contain the same hot
    // experts. Counting only the larger tier avoids promising storage reuse
    // from two duplicate copies. Dense tiles currently have guaranteed LRU
    // reuse only in the host store; accelerator space is staging/residency.
    let guaranteed_cached_experts = ram_expert.max(accelerator_expert).min(expert_bytes);
    let guaranteed_cached_dense = ram_dense.min(manifest.dense_bytes);
    let storage_bandwidth = number(args, "--storage-bandwidth-mbps", 3500)?
        .checked_mul(1_000_000)
        .ok_or("--storage-bandwidth-mbps is too large")?;
    let target_tps = value(args, "--target-tps")
        .map(|raw| {
            raw.parse::<f64>()
                .map_err(|_| "--target-tps must be a positive number".to_string())
        })
        .transpose()?;
    let io_performance = estimate_io_performance(
        &manifest,
        memory_envelope.max_verification_positions,
        storage_bandwidth,
        ram_dense,
        ram_expert.max(accelerator_expert),
        target_tps,
    )
    .map_err(|error| error.to_string())?;
    let summary = PlacementSummary {
        adapter: manifest.architecture.adapter,
        model_type: manifest.architecture.model_type,
        forward_verified: manifest.architecture.forward_verified,
        dense_bytes: manifest.dense_bytes,
        expert_bytes,
        ram_dense_budget_bytes: ram_dense.min(manifest.dense_bytes),
        accelerator_dense_budget_bytes: accelerator_dense.min(manifest.dense_bytes),
        ram_expert_budget_bytes: ram_expert.min(expert_bytes),
        accelerator_expert_budget_bytes: accelerator_expert.min(expert_bytes),
        storage_dense_bytes: manifest.dense_bytes.saturating_sub(guaranteed_cached_dense),
        storage_expert_bytes: expert_bytes.saturating_sub(guaranteed_cached_experts),
        memory_envelope,
        io_performance,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn self_test(args: &[String]) -> Result<(), String> {
    let (root, manifest) = load(args)?;
    let first = manifest
        .storage
        .experts
        .first()
        .ok_or("expert index is empty")?;
    let store = Arc::new(ExpertStore::new(
        root,
        mirrors(args),
        manifest.storage.experts.iter().cloned(),
    ));
    let accelerator: Arc<dyn Accelerator> = match value(args, "--cuda-device") {
        Some(raw) => {
            let ordinal = raw
                .parse::<i32>()
                .map_err(|_| "--cuda-device must be an integer".to_string())?;
            Arc::new(CudaAccelerator::new(ordinal)?)
        }
        None => Arc::new(NoAccelerator),
    };
    let accelerator_budget = if value(args, "--cuda-device").is_some() {
        first.byte_len()
    } else {
        0
    };
    let cache = ResidencyManager::new(store, accelerator, first.byte_len(), accelerator_budget);
    let key = ExpertKey::new(first.layer, first.expert);
    let cold = cache.get(key).map_err(|error| error.to_string())?;
    let warm = cache.get(key).map_err(|error| error.to_string())?;
    if cold.host_bytes.as_deref() != warm.host_bytes.as_deref() {
        return Err("cache changed expert bytes".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "byte_exact": true,
            "accelerator_staged": cold.accelerator_buffer().is_some(),
            "cache": cache.metrics().map_err(|e| e.to_string())?
        })
    );
    Ok(())
}

fn serve(args: &[String]) -> Result<(), String> {
    let (root, manifest) = load(args)?;
    let descriptor = validate_compiled_contract(&manifest).map_err(|error| error.to_string())?;
    validate_model_config(&root, &manifest, descriptor).map_err(|error| error.to_string())?;
    let memory = requested_memory_envelope(args, &manifest)?;
    if !memory.feasible {
        return Err(format!(
            "the requested RAM/VRAM envelope is not feasible: {}",
            memory.notes.join("; ")
        ));
    }
    validate_forward_contract(&manifest).map_err(|error| error.to_string())?;
    let verified = verify_runtime_oracle(args, &root, &manifest, &memory)?;
    let tokenizer = ModelTokenizer::load(&root).map_err(|error| error.to_string())?;
    match verified {
        VerifiedRuntime::Kimi(runtime, oracle) => serve_verified(
            args,
            &root,
            Arc::new(runtime),
            oracle,
            tokenizer,
            manifest.architecture.adapter,
        ),
        VerifiedRuntime::Mixtral(runtime, oracle) => serve_verified(
            args,
            &root,
            Arc::new(runtime),
            oracle,
            tokenizer,
            manifest.architecture.adapter,
        ),
    }
}

fn benchmark(args: &[String]) -> Result<(), String> {
    let (root, manifest) = load(args)?;
    let descriptor = validate_compiled_contract(&manifest).map_err(|error| error.to_string())?;
    validate_model_config(&root, &manifest, descriptor).map_err(|error| error.to_string())?;
    validate_forward_contract(&manifest).map_err(|error| error.to_string())?;
    let memory = requested_memory_envelope(args, &manifest)?;
    if !memory.feasible {
        return Err(format!(
            "the requested benchmark memory envelope is infeasible: {}",
            memory.notes.join("; ")
        ));
    }
    let verified = verify_runtime_oracle(args, &root, &manifest, &memory)?;
    let tokenizer = ModelTokenizer::load(&root).map_err(|error| error.to_string())?;
    match verified {
        VerifiedRuntime::Kimi(runtime, oracle) => benchmark_verified(
            args,
            &runtime,
            &tokenizer,
            &manifest.architecture.adapter,
            &oracle,
        ),
        VerifiedRuntime::Mixtral(runtime, oracle) => benchmark_verified(
            args,
            &runtime,
            &tokenizer,
            &manifest.architecture.adapter,
            &oracle,
        ),
    }
}

fn benchmark_verified<R: GenerationRuntime>(
    args: &[String],
    runtime: &R,
    tokenizer: &ModelTokenizer,
    adapter: &str,
    oracle: &sytra_engine::OracleReport,
) -> Result<(), String> {
    if tokenizer.vocab_size() != runtime.generation_vocab_size() {
        return Err("tokenizer vocabulary does not match the benchmark runtime".into());
    }
    let prompt = value(args, "--prompt").unwrap_or_else(|| "Hello".into());
    let prompt_tokens = tokenizer
        .encode(&prompt, true)
        .map_err(|error| error.to_string())?;
    if prompt_tokens.is_empty() {
        return Err("benchmark prompt encoded to zero tokens".into());
    }
    let max_tokens = usize::try_from(number(args, "--max-tokens", 16)?)
        .map_err(|_| "--max-tokens is too large".to_string())?;
    let iterations = usize::try_from(number(args, "--iterations", 3)?)
        .map_err(|_| "--iterations is too large".to_string())?;
    let warmup_tokens = usize::try_from(number(args, "--warmup-tokens", 1)?)
        .map_err(|_| "--warmup-tokens is too large".to_string())?;
    if max_tokens == 0 || iterations == 0 || iterations > 100 {
        return Err("benchmark requires positive tokens and 1..=100 iterations".into());
    }
    let target_tps = value(args, "--target-tps")
        .unwrap_or_else(|| "5".into())
        .parse::<f64>()
        .map_err(|_| "--target-tps must be a positive number".to_string())?;
    if !target_tps.is_finite() || target_tps <= 0.0 {
        return Err("--target-tps must be a positive finite number".into());
    }
    let generator = ModelGenerator::new(runtime, tokenizer);
    if warmup_tokens > 0 {
        generator
            .generate(
                &prompt_tokens,
                &GenerationConfig {
                    max_tokens: warmup_tokens,
                    ..GenerationConfig::default()
                },
                |_| Ok(()),
            )
            .map_err(|error| error.to_string())?;
    }
    let config = GenerationConfig {
        max_tokens,
        ..GenerationConfig::default()
    };
    let mut total_tokens = 0_usize;
    let mut total_seconds = 0.0_f64;
    let mut runs = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let output = generator
            .generate(&prompt_tokens, &config, |_| Ok(()))
            .map_err(|error| error.to_string())?;
        total_tokens = total_tokens.saturating_add(output.completion_tokens);
        total_seconds += output.elapsed_seconds;
        runs.push(serde_json::json!({
            "iteration": iteration + 1,
            "completion_tokens": output.completion_tokens,
            "elapsed_seconds": output.elapsed_seconds,
            "tokens_per_second": output.tokens_per_second,
            "finish_reason": output.finish_reason,
            "step_metrics": output.metrics,
        }));
    }
    let measured_tps = if total_seconds > 0.0 {
        total_tokens as f64 / total_seconds
    } else {
        0.0
    };
    let health = runtime
        .generation_health()
        .map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "adapter": adapter,
            "oracle": oracle,
            "prompt_tokens": prompt_tokens.len(),
            "requested_completion_tokens_per_iteration": max_tokens,
            "warmup_tokens": warmup_tokens,
            "iterations": iterations,
            "total_completion_tokens": total_tokens,
            "total_elapsed_seconds": total_seconds,
            "measured_tokens_per_second": measured_tps,
            "target_tokens_per_second": target_tps,
            "target_met": measured_tps >= target_tps,
            "runs": runs,
            "bounded_runtime": health,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn serve_verified<R: GenerationRuntime + Send + Sync + 'static>(
    args: &[String],
    root: &PathBuf,
    runtime: Arc<R>,
    oracle: sytra_engine::OracleReport,
    tokenizer: ModelTokenizer,
    adapter: String,
) -> Result<(), String> {
    if tokenizer.vocab_size() != runtime.generation_vocab_size() {
        return Err(format!(
            "tokenizer vocabulary size {} does not match model vocabulary size {}",
            tokenizer.vocab_size(),
            runtime.generation_vocab_size()
        ));
    }
    let host = value(args, "--host").unwrap_or_else(|| "127.0.0.1".into());
    let port = u16::try_from(number(args, "--port", 8080)?)
        .map_err(|_| "--port must fit in the range 0..=65535".to_string())?;
    let max_concurrent_requests = usize::try_from(number(args, "--max-concurrent-requests", 4)?)
        .map_err(|_| "--max-concurrent-requests is too large".to_string())?;
    let model_id = value(args, "--served-model-name").unwrap_or_else(|| {
        root.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("sytra-model")
            .to_owned()
    });
    let bind = format!("{host}:{port}");
    let tokenizer = Arc::new(tokenizer);
    let mut backend =
        RuntimeCompletionBackend::new(model_id.clone(), runtime.clone(), tokenizer.clone());
    let draft_enabled = if let Some(draft_url) = value(args, "--draft-url") {
        let draft_model = value(args, "--draft-model").unwrap_or_else(|| "draft-model".into());
        let timeout = Duration::from_millis(number(args, "--draft-timeout-ms", 30_000)?);
        let draft = OpenAiDraftModel::new(draft_url, draft_model, tokenizer, timeout)
            .map_err(|error| error.to_string())?;
        let target_tps = value(args, "--target-tps")
            .unwrap_or_else(|| "5".into())
            .parse::<f32>()
            .map_err(|_| "--target-tps must be a positive number".to_string())?;
        backend = backend.with_draft(Arc::new(draft), target_tps);
        true
    } else {
        false
    };
    let backend = Arc::new(backend);
    let health = runtime
        .generation_health()
        .map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "ready": true,
            "bind": bind,
            "model": model_id,
            "adapter": adapter,
            "oracle": oracle,
            "runtime": health,
            "max_concurrent_requests": max_concurrent_requests,
            "speculative_draft_enabled": draft_enabled,
        }))
        .map_err(|error| error.to_string())?
    );
    OpenAiServer::new(bind, backend, max_concurrent_requests)
        .map_err(|error| error.to_string())?
        .run()
        .map_err(|error| error.to_string())
}

fn cuda_check(args: &[String]) -> Result<(), String> {
    let ordinal = number(args, "--cuda-device", 0)?;
    let ordinal =
        i32::try_from(ordinal).map_err(|_| "--cuda-device is outside the i32 range".to_string())?;
    let byte_count = number(args, "--bytes", 1024 * 1024)?;
    let byte_count = usize::try_from(byte_count).map_err(|_| "--bytes is too large".to_string())?;
    let accelerator = CudaAccelerator::new(ordinal)?;
    let payload = vec![0xA5; byte_count];
    let buffer = accelerator.upload(ExpertKey::new(0, 0), &payload)?;
    let id = buffer.id;
    accelerator.release(&buffer);
    let rows = 3_usize;
    let cols = 64_usize;
    let bf16_weights: Vec<u8> = (0..rows * cols)
        .flat_map(|index| {
            let value = ((index % 17) as f32 - 8.0) / 8.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let input: Vec<f32> = (0..cols)
        .map(|index| (index as f32 - 13.0) / 19.0)
        .collect();
    let bf16_reference = bf16_tile_cpu(&bf16_weights, rows, cols, &input)?;
    let bf16_actual = accelerator.bf16_matvec_bytes(&bf16_weights, rows, cols, &input)?;
    let bf16_max_absolute_error = bf16_reference
        .iter()
        .zip(&bf16_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if bf16_max_absolute_error > 1e-4 {
        return Err(format!(
            "CUDA BF16 dense kernel differs from the CPU oracle by {bf16_max_absolute_error}"
        ));
    }
    let transpose_input: Vec<f32> = (0..rows).map(|index| (index as f32 + 1.0) / 3.0).collect();
    let transpose_reference = bf16_transpose_tile_cpu(&bf16_weights, rows, cols, &transpose_input)?;
    let transpose_actual =
        accelerator.bf16_transpose_matvec_bytes(&bf16_weights, rows, cols, &transpose_input)?;
    let transpose_max_absolute_error = transpose_reference
        .iter()
        .zip(&transpose_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if transpose_max_absolute_error > 1e-4 {
        return Err(format!(
            "CUDA BF16 transpose kernel differs from the CPU oracle by {transpose_max_absolute_error}"
        ));
    }
    let batch_positions = 3_usize;
    let batch_input: Vec<f32> = (0..batch_positions * cols)
        .map(|index| (index as f32 - 29.0) / 31.0)
        .collect();
    let batch_reference =
        bf16_tile_matmul_cpu(&bf16_weights, rows, cols, batch_positions, &batch_input)?;
    let batch_actual =
        accelerator.bf16_matmul_bytes(&bf16_weights, rows, cols, batch_positions, &batch_input)?;
    let resident_prefix_bytes = 32_usize;
    let mut resident_payload = vec![0x5A; resident_prefix_bytes];
    resident_payload.extend_from_slice(&bf16_weights);
    let resident_bf16 = accelerator.upload(ExpertKey::new(0, 2), &resident_payload)?;
    let resident_batch_actual = accelerator.resident_bf16_matmul(
        &resident_bf16,
        resident_prefix_bytes,
        rows,
        cols,
        batch_positions,
        &batch_input,
    );
    accelerator.release(&resident_bf16);
    let resident_batch_actual = resident_batch_actual?;
    let batch_max_absolute_error = batch_reference
        .iter()
        .zip(&batch_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if batch_max_absolute_error > 1e-4 {
        return Err(format!(
            "CUDA BF16 batched kernel differs from the CPU oracle by {batch_max_absolute_error}"
        ));
    }
    let resident_batch_max_absolute_error = batch_reference
        .iter()
        .zip(&resident_batch_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if resident_batch_max_absolute_error > 1e-4 {
        return Err(format!(
            "resident CUDA BF16 batched kernel differs from the CPU oracle by {resident_batch_max_absolute_error}"
        ));
    }
    let transpose_batch_input: Vec<f32> = (0..batch_positions * rows)
        .map(|index| (index as f32 + 2.0) / 7.0)
        .collect();
    let transpose_batch_reference = bf16_transpose_tile_matmul_cpu(
        &bf16_weights,
        rows,
        cols,
        batch_positions,
        &transpose_batch_input,
    )?;
    let transpose_batch_actual = accelerator.bf16_transpose_matmul_bytes(
        &bf16_weights,
        rows,
        cols,
        batch_positions,
        &transpose_batch_input,
    )?;
    let transpose_batch_max_absolute_error = transpose_batch_reference
        .iter()
        .zip(&transpose_batch_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if transpose_batch_max_absolute_error > 1e-4 {
        return Err(format!(
            "CUDA BF16 transpose batched kernel differs from the CPU oracle by {transpose_batch_max_absolute_error}"
        ));
    }
    let expert_hidden = 4_usize;
    let expert_intermediate = 8_usize;
    let expert_positions = 3_usize;
    let mut expert_bytes: Vec<u8> = (0..2 * expert_intermediate * expert_hidden)
        .flat_map(|index| {
            let value = ((index % 11) as f32 - 5.0) / 16.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let expert_down_offset = expert_bytes.len();
    expert_bytes.extend((0..expert_hidden * expert_intermediate).flat_map(|index| {
        let value = ((index % 7) as f32 - 3.0) / 12.0;
        ((value.to_bits() >> 16) as u16).to_le_bytes()
    }));
    let expert_tensors = vec![
        ResidentTensor {
            name: "expert.gate_up_proj.weight".into(),
            dtype: Some("BF16".into()),
            shape: vec![(2 * expert_intermediate) as u64, expert_hidden as u64],
            offset: 0,
            length: expert_down_offset,
        },
        ResidentTensor {
            name: "expert.down_proj.weight".into(),
            dtype: Some("BF16".into()),
            shape: vec![expert_hidden as u64, expert_intermediate as u64],
            offset: expert_down_offset,
            length: expert_bytes.len() - expert_down_offset,
        },
    ];
    let expert_input: Vec<f32> = (0..expert_positions * expert_hidden)
        .map(|index| (index as f32 - 4.0) / 9.0)
        .collect();
    let expert_reference = floating_gated_expert_batch(
        &expert_bytes,
        &expert_tensors,
        expert_hidden,
        expert_intermediate,
        expert_positions,
        &expert_input,
        ActivationKind::Silu,
    )
    .map_err(|error| error.to_string())?;
    let expert_actual = floating_gated_expert_batch_with_cuda(
        &expert_bytes,
        &expert_tensors,
        expert_hidden,
        expert_intermediate,
        expert_positions,
        &expert_input,
        ActivationKind::Silu,
        Some(&accelerator),
    )
    .map_err(|error| error.to_string())?;
    let expert_max_absolute_error = expert_reference
        .iter()
        .zip(&expert_actual)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0_f32, f32::max);
    if expert_max_absolute_error > 1e-4 {
        return Err(format!(
            "CUDA BF16 gated expert differs from the CPU oracle by {expert_max_absolute_error}"
        ));
    }
    println!(
        "{}",
        serde_json::json!({
            "backend": accelerator.name(),
            "device": ordinal,
            "bytes_staged": byte_count,
            "device_pointer": id,
            "released": true,
            "bf16_dense_reference_match": true,
            "bf16_dense_max_absolute_error": bf16_max_absolute_error,
            "bf16_transpose_reference_match": true,
            "bf16_transpose_max_absolute_error": transpose_max_absolute_error,
            "bf16_batch_reference_match": true,
            "bf16_batch_positions": batch_positions,
            "bf16_batch_max_absolute_error": batch_max_absolute_error,
            "resident_bf16_batch_reference_match": true,
            "resident_bf16_batch_max_absolute_error": resident_batch_max_absolute_error,
            "bf16_transpose_batch_reference_match": true,
            "bf16_transpose_batch_max_absolute_error": transpose_batch_max_absolute_error,
            "bf16_gated_expert_reference_match": true,
            "bf16_gated_expert_max_absolute_error": expert_max_absolute_error
        })
    );
    Ok(())
}

fn list_adapters() -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(&compiled_adapters()).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("doctor") => doctor(&args[1..]),
        Some("plan") => plan(&args[1..]),
        Some("self-test") => self_test(&args[1..]),
        Some("cuda-check") => cuda_check(&args[1..]),
        Some("kimi-k27-check") => kimi_k27_check(&args[1..]),
        Some("kimi-k27-cuda-check") => kimi_k27_cuda_check(&args[1..]),
        Some("fingerprint") => fingerprint(&args[1..]),
        Some("oracle-check") => oracle_check(&args[1..]),
        Some("benchmark") => benchmark(&args[1..]),
        Some("list-adapters") => list_adapters(),
        Some("serve") => serve(&args[1..]),
        _ => {
            usage();
            Err("unknown or missing command".into())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sytra-engine: {error}");
            ExitCode::from(2)
        }
    }
}
