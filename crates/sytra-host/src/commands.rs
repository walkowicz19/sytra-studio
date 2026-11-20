use crate::{
    backend_resolver::BackendResolver, job_runner::JobRunner, resource_guard::ResourceGuard,
    run_archive::RunArchive,
};
use sytra_contracts::{
    guider::{Compatibility, Guider, HardwareCapabilities, ModelCatalogEntry, TrainRecipe},
    merge_config::{MergeMethod, Verdict},
    operation::{OpRecord, OpStatus, Operation, TrainSpec},
    TelemetryLine,
};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Starts a training or merge operation.
/// Returns the op UUID and the telemetry receiver for streaming.
pub fn start_op(
    op: Operation,
    runner: &JobRunner,
    archive: &RunArchive,
    guard: &ResourceGuard,
    guider: &Guider,
) -> Result<(Uuid, mpsc::Receiver<TelemetryLine>), String> {
    // 1. Resolve and validate backend accelerator
    let resolved_op = match op.clone() {
        Operation::Train(mut spec) => {
            // Contract 1: the host assigns run_id and rewrites the YAML so
            // the runner and the archive agree on the id. Without this,
            // every hand-off with a null run_id archives under the nil
            // UUID and overwrites the previous record.
            if spec.config.run_id.is_none() {
                spec.config.run_id = Some(Uuid::new_v4());
                if let Ok(yaml) = spec.config.to_yaml_string() {
                    let _ = std::fs::write(&spec.config_path, yaml);
                }
            }

            let base_model = crate::training_source::resolve(&mut spec.config)?;
            spec.config.backend.kind = BackendResolver::resolve(spec.config.backend.kind);

            // Check hardware resource availability
            if spec.config.performance.get("training_engine").and_then(serde_json::Value::as_str) == Some("helix") {
                crate::validate_before_spawn(&Operation::Train(spec.clone())).map_err(|e| e.to_string())?;
                guard.check_helix_train(&spec).map_err(|e| e.to_string())?;
            } else {
                let model_entry = guider.resolve_model(&base_model)
                    .ok_or_else(|| format!("Model '{}' not found in catalog", spec.config.model))?;
                guard.check_train(model_entry, &spec).map_err(|e| e.to_string())?;
            }

            let yaml = spec.config.to_yaml_string().map_err(|e| e.to_string())?;
            std::fs::write(&spec.config_path, yaml).map_err(|e| format!("Cannot persist resolved training config: {e}"))?;
            Operation::Train(spec)
        }
        Operation::Merge(mut spec) => {
            if spec.config.op_id.is_none() {
                spec.config.op_id = Some(Uuid::new_v4());
                if let Ok(yaml) = spec.config.to_yaml_string() {
                    let _ = std::fs::write(&spec.config_path, yaml);
                }
            }

            let model_refs: Vec<String> =
                spec.config.models.iter().map(|m| m.model.clone()).collect();
            let compat = guider.merge_check_with_base(
                spec.config.base_model.as_deref(),
                &model_refs,
                spec.config.merge_method,
            );
            if compat.verdict == Verdict::Red {
                return Err(format!("Cannot merge: {}", compat.reason));
            }

            // Resource guard check for merge
            let mut resolved_models = Vec::new();
            let mut seen_refs = std::collections::HashSet::new();
            for m_ref in &model_refs {
                if seen_refs.insert(m_ref.clone()) {
                    if let Some(entry) = resolve_model_with_fallback(guider, m_ref) {
                        resolved_models.push(entry);
                    }
                }
            }
            if let Some(base) = spec.config.base_model.as_deref() {
                if !base.trim().is_empty() && seen_refs.insert(base.to_string()) {
                    if let Some(entry) = resolve_model_with_fallback(guider, base) {
                        resolved_models.push(entry);
                    }
                }
            }
            if spec.config.merge_method == MergeMethod::Moe {
                if let Some(options) = &spec.config.moe {
                    for expert in &options.shared_experts {
                        if seen_refs.insert(expert.source_model.clone()) {
                            if let Some(entry) = resolve_model_with_fallback(guider, &expert.source_model) {
                                resolved_models.push(entry);
                            }
                        }
                    }
                }
            }
            let resolved_refs: Vec<&ModelCatalogEntry> = resolved_models.iter().collect();
            guard
                .check_merge(&resolved_refs, spec.config.merge_method, true)
                .map_err(|e| e.to_string())?;

            Operation::Merge(spec)
        }
        Operation::Publish(spec) => Operation::Publish(spec),
    };

    crate::validate_before_spawn(&resolved_op).map_err(|e| e.to_string())?;
    let op_id = resolved_op.op_id();

    // 2. Initialize initial running archive record
    let record = OpRecord {
        op_id,
        kind: resolved_op.kind().to_string(),
        config: match &resolved_op {
            Operation::Train(s) => {
                serde_json::to_value(&s.config).unwrap_or(serde_json::Value::Null)
            }
            Operation::Merge(s) => {
                serde_json::to_value(&s.config).unwrap_or(serde_json::Value::Null)
            }
            Operation::Publish(s) => serde_json::to_value(s).unwrap_or(serde_json::Value::Null),
        },
        artifact_path: match &resolved_op {
            Operation::Train(s) => s.config.output.adapter_path.clone(),
            Operation::Merge(s) => s.config.output.model_path.clone(),
            Operation::Publish(s) => s.artifact_path.clone(),
        },
        status: OpStatus::Running,
        provenance: None,
    };
    archive.store(&record).map_err(|e| e.to_string())?;

    // 3. Spawn process runner and return the telemetry receiver
    let rx = runner.start(&resolved_op).map_err(|e| e.to_string())?;

    Ok((op_id, rx))
}

/// Stops the currently running operation.
///
/// The Stopped status is persisted BEFORE the kill: killing closes the
/// telemetry channel, and the stream-drain task settles the final status
/// the moment that happens — writing first is what guarantees it sees
/// Stopped instead of racing us and recording Done.
pub fn stop_op(op_id: Uuid, runner: &JobRunner, archive: &RunArchive) -> Result<(), String> {
    // Only a still-running record becomes Stopped; a stop after natural
    // completion must not overwrite Done/Error.
    if let Ok(mut record) = archive.load(op_id) {
        if record.status == OpStatus::Running {
            record.status = OpStatus::Stopped;
            let _ = archive.store(&record);
        }
    }

    runner.stop().map_err(|e| e.to_string())
}

/// Resumes a paused/stopped training operation.
pub fn resume_op(
    mut op: Operation,
    runner: &JobRunner,
    archive: &RunArchive,
    guard: &ResourceGuard,
    guider: &Guider,
) -> Result<(Uuid, mpsc::Receiver<TelemetryLine>), String> {
    if let Operation::Train(ref mut spec) = op {
        // Set resume path to latest checkpoint folder
        let adapter_path = &spec.config.output.adapter_path;
        spec.config.output.resume_from = Some(crate::training_source::latest_checkpoint(adapter_path)?);
        spec.config.init_adapter = None;
        start_op(op, runner, archive, guard, guider)
    } else {
        Err("Only training operations can be resumed".to_string())
    }
}

/// Lists all run history records.
pub fn list_ops(archive: &RunArchive) -> Result<Vec<OpRecord>, String> {
    archive.list().map_err(|e| e.to_string())
}

/// Fetches details of a single operation run.
pub fn get_op(op_id: Uuid, archive: &RunArchive) -> Result<OpRecord, String> {
    archive.load(op_id).map_err(|e| e.to_string())
}

/// Deletes an operation history entry.
pub fn delete_op(op_id: Uuid, archive: &RunArchive) -> Result<(), String> {
    archive.delete(op_id).map_err(|e| e.to_string())
}

/// Estimates training VRAM usage.
pub fn estimate_memory(
    model_ref: String,
    spec: TrainSpec,
    _guard: &ResourceGuard,
    guider: &Guider,
) -> Result<u64, String> {
    if spec.config.performance.get("training_engine").and_then(serde_json::Value::as_str) == Some("helix") {
        if spec.config.model != model_ref { return Err("Model reference does not match the Helix configuration".into()); }
        return crate::helix_guard::estimate(&spec).map(|(vram, _)| vram);
    }
    let model = guider
        .resolve_model(&model_ref)
        .ok_or_else(|| format!("Model '{}' not found in catalog", model_ref))?;
    Ok(ResourceGuard::estimate_train_vram(model, &spec))
}

/// Provides model recommendations.
pub fn guider_recommend(
    hardware: HardwareCapabilities,
    guider: &Guider,
) -> Result<Vec<TrainRecipe>, String> {
    Ok(guider.recommend(&hardware))
}

/// Checks model merge compatibility (incl. lineage heuristic when a base
/// model is provided for task-vector methods).
pub fn merge_check(
    model_refs: Vec<String>,
    method: MergeMethod,
    base_model: Option<String>,
    guider: &Guider,
) -> Result<Compatibility, String> {
    Ok(guider.merge_check_with_base(base_model.as_deref(), &model_refs, method))
}

fn extract_billions_param_count(model_ref: &str) -> Option<u64> {
    let lower = model_ref.to_lowercase();
    let bytes = lower.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        if b != b'b' || idx == 0 {
            continue;
        }
        // Require non-alphanumeric boundaries so path segments like "base", "hub", or
        // "build" are never misread as parameter sizes.
        if idx + 1 < bytes.len() && bytes[idx + 1].is_ascii_alphanumeric() {
            continue;
        }
        if !bytes[idx - 1].is_ascii_digit() {
            continue;
        }
        let mut start = idx;
        while start > 0 && (bytes[start - 1].is_ascii_digit() || bytes[start - 1] == b'.') {
            start -= 1;
        }
        let is_moe_product = start >= 2
            && bytes[start - 1] == b'x'
            && bytes[start - 2].is_ascii_digit();
        if start > 0 && bytes[start - 1].is_ascii_alphabetic() && !is_moe_product {
            continue;
        }
        if let Ok(num) = lower[start..idx].parse::<f64>() {
            if num.is_finite() && num > 0.0 {
                return Some((num * 1_000_000_000.0) as u64);
            }
        }
    }
    None
}

fn estimate_params_from_local_dir(model_ref: &str) -> Option<u64> {
    let metadata = std::fs::metadata(model_ref).ok()?;
    if !metadata.is_dir() {
        return None;
    }
    let mut total_bytes: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(model_ref) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    let name = entry.file_name().to_string_lossy().to_lowercase();
                    if name.ends_with(".safetensors")
                        || name.ends_with(".bin")
                        || name.ends_with(".pt")
                    {
                        total_bytes = total_bytes.saturating_add(meta.len());
                    }
                }
            }
        }
    }
    if total_bytes > 0 {
        Some((total_bytes / 2).max(1))
    } else {
        None
    }
}

fn resolve_model_with_fallback(guider: &Guider, m_ref: &str) -> Option<ModelCatalogEntry> {
    if let Some(entry) = guider.resolve_model(m_ref) {
        return Some(entry.clone());
    }

    let param_count = estimate_params_from_local_dir(m_ref)
        .or_else(|| extract_billions_param_count(m_ref))
        .unwrap_or(7_000_000_000);

    Some(ModelCatalogEntry {
        model_id: m_ref.to_string(),
        name: m_ref.to_string(),
        param_count,
        architecture: "unknown".to_string(),
        dtype: "bfloat16".to_string(),
        moe_active_params: None,
        license: "unknown".to_string(),
        default_target_modules: vec![],
        tokenizer_id: m_ref.to_string(),
        use_case_tags: vec![],
        benchmark_hint: String::new(),
        format: String::new(),
        downloadable: false,
        workflows: vec![],
        gated: false,
        approx_download_gb: None,
        explicit_risks: vec!["Not in the pinned Sytra catalog; architecture is unknown.".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_measures_local_weights_even_when_path_contains_b() {
        let root = std::env::temp_dir().join(format!("sytra_cmd_test_{}", std::process::id()));
        let model_dir = root.join("hub").join("v2_base_model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.safetensors"), vec![0u8; 2_000_000]).unwrap();
        let guider = Guider::new();
        let entry = resolve_model_with_fallback(&guider, model_dir.to_str().unwrap()).unwrap();
        assert_eq!(entry.param_count, 1_000_000);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn fallback_parses_size_suffixes_without_matching_words() {
        assert_eq!(extract_billions_param_count("Qwen/Qwen2.5-0.5B-Instruct"), Some(500_000_000));
        assert_eq!(extract_billions_param_count("D:\\cache\\hub\\my2base"), None);
        assert_eq!(extract_billions_param_count("mistralai/Mixtral-8x7B-v0.1"), Some(7_000_000_000));
    }
}
