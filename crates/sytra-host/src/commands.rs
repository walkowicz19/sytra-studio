use crate::{
    backend_resolver::BackendResolver, job_runner::JobRunner, resource_guard::ResourceGuard,
    run_archive::RunArchive,
};
use sytra_contracts::{
    guider::{Compatibility, Guider, HardwareCapabilities, TrainRecipe, ModelCatalogEntry},
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

            let model_entry = guider
                .resolve_model(&spec.config.model)
                .ok_or_else(|| format!("Model '{}' not found in catalog", spec.config.model))?;

            // Check hardware resource availability
            guard
                .check_train(model_entry, &spec)
                .map_err(|e| e.to_string())?;

            spec.config.backend.kind = BackendResolver::resolve(spec.config.backend.kind);
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
            for m_ref in &model_refs {
                if let Some(entry) = resolve_model_with_fallback(guider, m_ref) {
                    resolved_models.push(entry);
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
        spec.config.output.resume_from = Some(adapter_path.join("checkpoint-latest"));
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

fn resolve_model_with_fallback(guider: &Guider, m_ref: &str) -> Option<ModelCatalogEntry> {
    if let Some(entry) = guider.resolve_model(m_ref) {
        return Some(entry.clone());
    }

    let lower = m_ref.to_lowercase();
    let mut param_count = 7_000_000_000; // default to 7B

    // Extract parameters from name like -7b, -70b
    if let Some(pos) = lower.find('b') {
        let chars: Vec<char> = lower[..pos].chars().collect();
        let mut num_str = String::new();
        for c in chars.into_iter().rev() {
            if c.is_ascii_digit() || c == '.' {
                num_str.insert(0, c);
            } else if !num_str.is_empty() {
                break;
            }
        }
        if let Ok(num) = num_str.parse::<f64>() {
            param_count = (num * 1_000_000_000.0) as u64;
        }
    } else {
        // Fallback: estimate from local directory if present
        if let Ok(metadata) = std::fs::metadata(m_ref) {
            if metadata.is_dir() {
                let mut total_bytes = 0;
                if let Ok(entries) = std::fs::read_dir(m_ref) {
                    for entry in entries.flatten() {
                        if let Ok(meta) = entry.metadata() {
                            if meta.is_file() {
                                let name = entry.file_name().to_string_lossy().to_lowercase();
                                if name.ends_with(".safetensors") || name.ends_with(".bin") || name.ends_with(".pt") {
                                    total_bytes += meta.len();
                                }
                            }
                        }
                    }
                }
                if total_bytes > 0 {
                    param_count = total_bytes / 2;
                }
            }
        }
    }

    Some(ModelCatalogEntry {
        model_id: m_ref.to_string(),
        name: m_ref.to_string(),
        param_count,
        architecture: "Qwen2".to_string(),
        dtype: "bfloat16".to_string(),
        moe_active_params: None,
        license: "apache-2.0".to_string(),
        default_target_modules: vec![],
        tokenizer_id: m_ref.to_string(),
        use_case_tags: vec![],
        benchmark_hint: String::new(),
    })
}
