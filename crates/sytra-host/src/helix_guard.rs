//! Conservative estimates for the experimental local Mixtral training path.
use std::{collections::{BTreeSet, HashMap, HashSet}, fs::File, io::Read, path::{Path, PathBuf}};
use serde_json::Value;
use sytra_contracts::operation::TrainSpec;

const MIB: u128 = 1024 * 1024;

fn positive(config: &Value, key: &str) -> Result<u128, String> {
    config.get(key).and_then(Value::as_u64).filter(|n| *n > 0 && *n <= 1_000_000)
        .map(u128::from).ok_or_else(|| format!("Invalid Mixtral {key}"))
}

fn shards(root: &Path) -> Result<Vec<PathBuf>, String> {
    let index = root.join("model.safetensors.index.json");
    if !index.is_file() {
        let file = root.join("model.safetensors").canonicalize().map_err(|e| e.to_string())?;
        if !file.starts_with(root) { return Err("Checkpoint file escapes model directory".into()); }
        return Ok(vec![file]);
    }
    let meta = std::fs::metadata(&index).map_err(|e| e.to_string())?;
    if meta.len() > 64 * 1024 * 1024 { return Err("Oversized checkpoint index".into()); }
    let data: Value = serde_json::from_reader(File::open(index).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let map = data.get("weight_map").and_then(Value::as_object).filter(|m| !m.is_empty()).ok_or("Invalid safetensors index")?;
    let mut files = BTreeSet::new();
    for name in map.values() {
        let name = name.as_str().ok_or("Invalid shard name")?;
        let file = root.join(name).canonicalize().map_err(|e| e.to_string())?;
        if !file.starts_with(root) || file.extension().and_then(|x| x.to_str()) != Some("safetensors") || !file.is_file() {
            return Err("Invalid safetensors shard path".into());
        }
        files.insert(file);
    }
    Ok(files.into_iter().collect())
}

/// Returns (estimated VRAM MB, estimated system RAM MB), not a fit guarantee.
pub fn estimate(spec: &TrainSpec) -> Result<(u64, u64), String> {
    let root = Path::new(&spec.config.model).canonicalize().map_err(|e| e.to_string())?;
    let config: Value = serde_json::from_reader(File::open(root.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let model_type = config.get("model_type").and_then(Value::as_str).unwrap_or("");
    if !matches!(model_type, "mixtral" | "qwen2_moe" | "qwen3_moe" | "deepseek_v2" | "deepseek_v3") ||
       config.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true) {
        return Err("Helix requires an untied local Mixtral, Qwen MoE, or DeepSeek MoE checkpoint".into());
    }
    if let Some(qcfg) = config.get("quantization_config").filter(|v| !v.is_null()) {
        let method = qcfg.get("quant_method").and_then(Value::as_str).unwrap_or("bitsandbytes");
        if method != "bitsandbytes" {
            return Err(format!("Helix does not support quantization_config method '{method}'"));
        }
    }
    let hidden = positive(&config, "hidden_size")?;
    let layers = positive(&config, "num_hidden_layers")?;
    let heads = positive(&config, "num_attention_heads")?;
    let intermediate = positive(&config, "moe_intermediate_size")
        .or_else(|_| positive(&config, "intermediate_size"))?;
    let top_k = positive(&config, "num_experts_per_tok")?;
    let vocab = positive(&config, "vocab_size")?;
    let quant_bits = if spec.config.adapter.kind == sytra_contracts::run_config::AdapterType::Qlora {
        spec.config.adapter.quant_bits
    } else {
        None
    };
    let nf4 = quant_bits == Some(4);
    let q8 = quant_bits == Some(8);
    let prequantized = config.get("quantization_config").is_some_and(|v| !v.is_null());
    let experts = if nf4 || q8 {
        positive(&config, "num_local_experts")
            .or_else(|_| positive(&config, "num_experts"))
            .or_else(|_| positive(&config, "n_routed_experts"))
            .unwrap_or(0)
    } else {
        0
    };
    let mut layer_sizes: HashMap<usize, u128> = HashMap::new();
    let mut head = 0u128;
    let mut adapter_params = 0u128;
    let mut quant_attention = 0u128;
    let mut quant_experts = 0u128;
    let mut seen = HashSet::new();
    for path in shards(&root)? {
        let mut file = File::open(&path).map_err(|e| e.to_string())?;
        let file_size = file.metadata().map_err(|e| e.to_string())?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
        let length = u64::from_le_bytes(prefix);
        if length == 0 || length > 64 * 1024 * 1024 || length > file_size.saturating_sub(8) {
            return Err("Invalid safetensors header length".into());
        }
        let mut bytes = vec![0u8; length as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let header: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        for (name, item) in header.as_object().ok_or("Invalid safetensors header")? {
            if name == "__metadata__" { continue; }
            if !seen.insert(name.clone()) { return Err("Duplicate checkpoint tensor".into()); }
            let dtype = item.get("dtype").and_then(Value::as_str).ok_or("Missing tensor dtype")?;
            let element_bytes = match dtype {
                "F16" | "BF16" => 2u128,
                "F32" | "I32" => 4,
                "F64" | "I64" => 8,
                "U8" | "I8" | "BOOL" => 1,
                _ => return Err(format!("Unsupported tensor dtype {dtype}")),
            };
            let dims = item.get("shape").and_then(Value::as_array).ok_or("Missing tensor shape")?;
            let mut elements = 1u128;
            for dim in dims { elements = elements.checked_mul(u128::from(dim.as_u64().ok_or("Invalid tensor shape")?)).ok_or("Tensor shape overflow")?; }
            let offsets = item.get("data_offsets").and_then(Value::as_array).filter(|v| v.len() == 2).ok_or("Invalid tensor offsets")?;
            let start = offsets[0].as_u64().ok_or("Invalid tensor offset")?;
            let end = offsets[1].as_u64().ok_or("Invalid tensor offset")?;
            if end < start || end > file_size - 8 - length || u128::from(end - start) != elements.saturating_mul(element_bytes) {
                return Err("Tensor bounds or shape mismatch".into());
            }
            let attention_projection = name.starts_with("model.layers.") && name.contains(".self_attn.") &&
                [".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
                 ".q_a_proj.weight", ".q_b_proj.weight", ".kv_a_proj_with_mqa.weight", ".kv_b_proj.weight"]
                    .iter().any(|suffix| name.ends_with(suffix));
            let expert_projection = name.starts_with("model.layers.") && (
                (name.contains(".experts.") && [".w1.weight", ".w2.weight", ".w3.weight", ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight", ".gate_up_proj", ".down_proj"].iter().any(|suffix| name.ends_with(suffix))) ||
                ((name.contains(".shared_expert.") || name.contains(".shared_experts.")) && [".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"].iter().any(|suffix| name.ends_with(suffix)))
            );
            if (nf4 || q8) && attention_projection { quant_attention += 1; }
            if (nf4 || q8) && expert_projection { quant_experts += 1; }
            let size = if !prequantized && (attention_projection || expert_projection || name == "lm_head.weight") {
                if nf4 {
                    (elements + 1) / 2 + ((elements + 63) / 64) * 4
                } else if q8 {
                    elements + ((elements + 63) / 64) * 4
                } else {
                    u128::from(end - start)
                }
            } else { u128::from(end - start) };
            if let Some(rest) = name.strip_prefix("model.layers.") {
                let index = rest.split('.').next().and_then(|v| v.parse::<usize>().ok()).ok_or("Invalid layer name")?;
                *layer_sizes.entry(index).or_default() += size;
            } else if name == "lm_head.weight" { head = size; }
            let topology = spec.config.adapter.moe_topology.as_str();
            let is_router = name.ends_with(".block_sparse_moe.gate.weight") || name.ends_with(".mlp.gate.weight");
            let is_shared_gate = name.ends_with(".mlp.shared_expert_gate.weight");
            let is_shared_expert = (name.contains(".shared_expert.") || name.contains(".shared_experts.")) &&
                [".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"].iter().any(|suffix| name.ends_with(suffix));
            let matched_target = spec.config.adapter.target_modules.iter().any(|target|
                name == &format!("{target}.weight") || name.ends_with(&format!(".{target}.weight")) ||
                (target == "router_projection" && is_router)) ||
                (matches!(topology, "moe_hybrid" | "moe_full") && (is_router || is_shared_gate || is_shared_expert)) ||
                (topology == "moe_full" && expert_projection);
            if dims.len() == 2 && matched_target {
                adapter_params += u128::from(spec.config.adapter.rank) *
                    (u128::from(dims[0].as_u64().unwrap()) + u128::from(dims[1].as_u64().unwrap()));
            }
        }
    }
    if layer_sizes.len() != layers as usize || (0..layers as usize).any(|i| !layer_sizes.contains_key(&i)) || head == 0 || adapter_params == 0 {
        return Err("Incomplete MoE layers, head or LoRA targets".into());
    }
    if (nf4 || q8) && !prequantized && (quant_attention == 0 || (experts > 0 && quant_experts == 0)) {
        return Err("Quantized Helix training requires complete attention and expert projections".into());
    }
    let window = *layer_sizes.values().max().unwrap_or(&0).max(&head);
    let batch = u128::from(spec.config.train.batch_size);
    let seq = u128::from(spec.config.train.max_seq_len);
    let activations = batch.saturating_mul(seq).saturating_mul(hidden).saturating_mul(4);
    let attention = batch.saturating_mul(heads).saturating_mul(seq).saturating_mul(seq).saturating_mul(12);
    let routed = batch.saturating_mul(seq).saturating_mul(top_k).saturating_mul(intermediate).saturating_mul(32);
    let loss = 256u128.saturating_mul(vocab).saturating_mul(12);
    let nf4_workspace = if nf4 { hidden.max(intermediate).saturating_mul(32 * 64) } else { 0 };
    let prefetch = match spec.config.performance.get("helix_prefetch") {
        None => false,
        Some(value) => value.as_bool().ok_or("Invalid Helix prefetch setting")?,
    };
    if let Some(value) = spec.config.performance.get("prefetch_pinned_memory_mb") {
        if !prefetch || value.as_u64().filter(|v| *v <= 4096).is_none() {
            return Err("Invalid pinned prefetch memory budget".into());
        }
    }
    if prefetch && spec.config.backend.kind != sytra_contracts::run_config::BackendKind::Cuda {
        return Err("Helix layer prefetch requires CUDA".into());
    }
    let gpu = 1536 * MIB + window + (if prefetch { window } else { 0 }) + activations.saturating_mul(32) + attention + routed + loss + adapter_params.saturating_mul(16) + nf4_workspace;
    let memory_mb = match spec.config.performance.get("activation_memory_mb") {
        None => 512,
        Some(value) => value.as_u64().filter(|v| *v <= 65536).ok_or("Invalid activation memory budget")?,
    };
    let stash = activations.saturating_mul(layers + 1).min(u128::from(memory_mb) * MIB);
    let cpu_mask = batch.saturating_mul(seq).saturating_mul(seq).saturating_mul(4);
    // Current and next collated batches each hold input IDs, attention mask,
    // and labels as int64 tensors. Helix prefetch keeps at most one ahead.
    let batch_buffers = batch.saturating_mul(seq).saturating_mul(8 * 3 * 2);
    let ram = 2048 * MIB + window + (if prefetch { window } else { 0 }) + stash + activations.saturating_mul(4) + cpu_mask + batch_buffers + adapter_params.saturating_mul(16);
    let mb = |value: u128| -> u64 { ((value.saturating_add(MIB - 1) / MIB).min(u128::from(u64::MAX))) as u64 };
    Ok((mb(gpu), mb(ram)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_guard::{ResourceError, ResourceGuard};
    use serde_json::json;
    use sytra_contracts::run_config::{BackendKind, RunConfig};

    fn fixture() -> (PathBuf, TrainSpec) {
        let root = std::env::temp_dir().join(format!("sytra-helix-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), json!({"model_type":"mixtral", "hidden_size":4,
            "num_hidden_layers":2, "num_attention_heads":2, "intermediate_size":8,
            "num_local_experts":2, "num_experts_per_tok":1, "vocab_size":16}).to_string()).unwrap();
        let mut header = json!({
            "model.layers.0.self_attn.q_proj.weight": {"dtype":"F32", "shape":[4,4], "data_offsets":[0,64]},
            "model.layers.1.self_attn.q_proj.weight": {"dtype":"F32", "shape":[4,4], "data_offsets":[64,128]},
            "lm_head.weight": {"dtype":"F32", "shape":[16,4], "data_offsets":[128,384]}
        });
        let mut offset = 384u64;
        for layer in 0..2 {
            for projection in ["k_proj", "v_proj", "o_proj"] {
                header[format!("model.layers.{layer}.self_attn.{projection}.weight")] =
                    json!({"dtype":"F32", "shape":[4,4], "data_offsets":[offset,offset + 64]});
                offset += 64;
            }
            for expert in 0..2 {
                for projection in ["w1", "w2", "w3"] {
                    header[format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight")] =
                        json!({"dtype":"F32", "shape":[4,4], "data_offsets":[offset,offset + 64]});
                    offset += 64;
                }
            }
        }
        let header = header.to_string().into_bytes();
        let mut data = (header.len() as u64).to_le_bytes().to_vec();
        data.extend_from_slice(&header);
        data.resize(data.len() + offset as usize, 0);
        std::fs::write(root.join("model.safetensors"), data).unwrap();
        let mut config = RunConfig::from_yaml_str(include_str!("../../../fixtures/run.golden.yaml")).unwrap();
        config.model = root.to_string_lossy().into_owned();
        config.backend.kind = BackendKind::Cuda;
        (root, TrainSpec { config, config_path: "run.yaml".into() })
    }

    #[test]
    fn local_window_estimate_checks_hardware_and_rejects_bad_header() {
        let (root, mut spec) = fixture();
        spec.config.performance.insert("training_engine".into(), json!("helix"));
        assert!(crate::validate_before_spawn(&sytra_contracts::Operation::Train(spec.clone())).is_ok());
        let mut unsupported = spec.clone();
        unsupported.config.adapter.kind = sytra_contracts::run_config::AdapterType::Qlora;
        assert!(crate::validate_before_spawn(&sytra_contracts::Operation::Train(unsupported)).is_err());
        let (vram, ram) = estimate(&spec).unwrap();
        let mut prefetched = spec.clone();
        prefetched.config.performance.insert("helix_prefetch".into(), json!(true));
        let (prefetch_vram, prefetch_ram) = estimate(&prefetched).unwrap();
        assert!(prefetch_vram >= vram && prefetch_ram >= ram);
        prefetched.config.performance.insert("prefetch_pinned_memory_mb".into(), json!(4097));
        assert!(estimate(&prefetched).unwrap_err().contains("pinned prefetch"));
        prefetched.config.performance.insert("prefetch_pinned_memory_mb".into(), json!(0));
        assert!(estimate(&prefetched).is_ok());
        prefetched.config.backend.kind = BackendKind::Cpu;
        assert!(estimate(&prefetched).unwrap_err().contains("requires CUDA"));
        let mut quantized = spec.clone();
        quantized.config.adapter.kind = sytra_contracts::run_config::AdapterType::Qlora;
        quantized.config.adapter.quant_bits = Some(4);
        assert!(crate::validate_before_spawn(&sytra_contracts::Operation::Train(quantized.clone())).is_ok());
        assert!(estimate(&quantized).is_ok());
        assert_eq!(crate::commands::estimate_memory(spec.config.model.clone(), spec.clone(),
                   &ResourceGuard::new(vram, ram, 10000), &sytra_contracts::guider::Guider::new()).unwrap(), vram);
        assert!(vram >= 1536 && ram >= 2048);
        assert!(ResourceGuard::new(vram, ram, 10000).check_helix_train(&spec).is_ok());
        assert!(matches!(ResourceGuard::new(vram - 1, ram, 10000).check_helix_train(&spec), Err(ResourceError::VramOverflow { .. })));
        assert!(matches!(ResourceGuard::new(vram, ram - 1, 10000).check_helix_train(&spec), Err(ResourceError::RamOverflow { .. })));
        assert_eq!(ResourceGuard::new(0, ram, 10000).check_helix_train(&spec), Err(ResourceError::UnknownHardware));
        spec.config.backend.kind = BackendKind::Cpu;
        assert!(ResourceGuard::new(0, ram, 10000).check_helix_train(&spec).is_ok());
        spec.config.train.max_seq_len = 32768;
        let (_, resident_ram) = estimate(&spec).unwrap();
        spec.config.performance.insert("activation_memory_mb".into(), json!(0));
        let (_, spilled_ram) = estimate(&spec).unwrap();
        assert!(spilled_ram < resident_ram);
        spec.config.performance.insert("activation_memory_mb".into(), json!(-1));
        assert!(estimate(&spec).is_err());
        spec.config.performance.remove("activation_memory_mb");
        std::fs::write(root.join("model.safetensors"), 500u64.to_le_bytes()).unwrap();
        assert!(matches!(ResourceGuard::new(vram, ram, 10000).check_helix_train(&spec), Err(ResourceError::InvalidHelix(_))));
        std::fs::remove_dir_all(root).unwrap();
    }
}
