use std::path::PathBuf;

use serde_json::Value;
use tauri::{AppHandle, State};
use uuid::Uuid;

use sytra_contracts::{
    merge_config::MergeConfig,
    merge_config::MergeMethod,
    operation::{MergeSpec, Operation, TrainSpec},
    run_config::RunConfig,
    guider::Compatibility,
};
use sytra_host::{commands, materialize::materialize_dataset_for_config};

use crate::helpers::{spawn_telemetry_stream, write_temp_yaml};
use crate::state::AppState;

#[tauri::command]
pub async fn start_train(
    state: State<'_, AppState>,
    app: AppHandle,
    config: Value,
) -> Result<String, String> {
    let mut run_config: RunConfig =
        serde_json::from_value(config.clone()).map_err(|e| format!("Bad train config: {e}"))?;
    let ws = &state.workspace;
    let dataset_dir = ws.join("runs").join("dataset_materialized");
    materialize_dataset_for_config(&mut run_config.data, run_config.train_mode, &dataset_dir)
        .await?;
    let config_path = {
        std::fs::create_dir_all(ws.join("runs")).map_err(|e| e.to_string())?;
        let config_val = serde_json::to_value(&run_config).map_err(|e| e.to_string())?;
        write_temp_yaml(&ws.join("runs"), "last_run.yaml", &config_val)?
    };
    let spec = TrainSpec {
        config: run_config,
        config_path,
    };
    let op = Operation::Train(spec);
    let runner = state.runner.lock().map_err(|_| "lock")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    let guard = state.guard.lock().map_err(|_| "lock")?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    let (op_id, rx) = commands::start_op(op, &runner, &archive, &guard, &guider)?;
    spawn_telemetry_stream(app, op_id, rx);
    Ok(op_id.to_string())
}

#[tauri::command]
pub async fn start_merge(
    state: State<'_, AppState>,
    app: AppHandle,
    config: Value,
) -> Result<String, String> {
    let merge_config: MergeConfig =
        serde_json::from_value(config.clone()).map_err(|e| format!("Bad merge config: {e}"))?;
    let config_path = {
        let ws = &state.workspace;
        std::fs::create_dir_all(ws.join("runs")).map_err(|e| e.to_string())?;
        write_temp_yaml(&ws.join("runs"), "last_merge.yaml", &config)?
    };
    let spec = MergeSpec {
        config: merge_config,
        config_path,
    };
    let op = Operation::Merge(spec);
    let runner = state.runner.lock().map_err(|_| "lock")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    let guard = state.guard.lock().map_err(|_| "lock")?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    let (op_id, rx) = commands::start_op(op, &runner, &archive, &guard, &guider)?;
    spawn_telemetry_stream(app, op_id, rx);
    Ok(op_id.to_string())
}

#[tauri::command]
pub fn stop_op(state: State<'_, AppState>, op_id: String) -> Result<(), String> {
    let id = Uuid::parse_str(&op_id).map_err(|_| "invalid uuid")?;
    let runner = state.runner.lock().map_err(|_| "lock")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::stop_op(id, &runner, &archive)
}

#[tauri::command]
pub fn list_runs(state: State<'_, AppState>) -> Result<Vec<sytra_contracts::OpRecord>, String> {
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::list_ops(&archive)
}

#[tauri::command]
pub fn delete_run(state: State<'_, AppState>, op_id: String) -> Result<(), String> {
    let id = Uuid::parse_str(&op_id).map_err(|_| "invalid uuid")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::delete_op(id, &archive)
}

#[tauri::command]
pub fn estimate_memory(
    state: State<'_, AppState>,
    model_ref: String,
    spec: Value,
) -> Result<u64, String> {
    let rc: RunConfig = serde_json::from_value(spec).map_err(|e| e.to_string())?;
    let train_spec = TrainSpec {
        config: rc,
        config_path: PathBuf::from("estimate"),
    };
    let guard = state.guard.lock().map_err(|_| "lock")?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    commands::estimate_memory(model_ref, train_spec, &guard, &guider)
}

#[tauri::command]
pub fn merge_check(
    state: State<'_, AppState>,
    model_refs: Vec<String>,
    method: String,
    base_model: Option<String>,
) -> Result<Compatibility, String> {
    let method: MergeMethod =
        serde_json::from_str(&format!("\"{}\"", method)).map_err(|e| e.to_string())?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    commands::merge_check(model_refs, method, base_model, &guider)
}

#[tauri::command]
pub async fn publish_run(
    state: State<'_, AppState>,
    app: AppHandle,
    run_op_id: String,
    repo_id: String,
    private: bool,
    token: String,
    license: Option<String>,
) -> Result<String, String> {
    let run_id = Uuid::parse_str(&run_op_id).map_err(|_| "Invalid run UUID")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    let run_record = archive
        .load(run_id)
        .map_err(|e| format!("Failed to find run: {e}"))?;
    if let Some(license) = license {
        if !license.trim().is_empty() {
            let license_path = run_record.artifact_path.join("LICENSE");
            if let Err(err) = std::fs::write(&license_path, format!("{license}\n")) {
                eprintln!("failed to write LICENSE: {err}");
            }
        }
    }
    let publish_op_id = Uuid::new_v4();
    let spec = sytra_contracts::operation::PublishSpec {
        op_id: publish_op_id,
        artifact_path: run_record.artifact_path,
        repo_id,
        private,
        token,
    };
    let op = Operation::Publish(spec);
    let runner = state.runner.lock().map_err(|_| "lock")?;
    let guard = state.guard.lock().map_err(|_| "lock")?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    let (op_id, rx) = commands::start_op(op, &runner, &archive, &guard, &guider)?;
    spawn_telemetry_stream(app, op_id, rx);
    Ok(op_id.to_string())
}

#[tauri::command]
pub async fn preview_dataset(source: Value, rows: usize) -> Result<Vec<Vec<String>>, String> {
    let data_spec: sytra_contracts::run_config::DataSpec = if source.get("source").is_some() {
        serde_json::from_value(source).map_err(|e| e.to_string())?
    } else {
        let hf_params: sytra_contracts::run_config::HfParams =
            serde_json::from_value(source).map_err(|e| e.to_string())?;
        sytra_contracts::run_config::DataSpec::Hf {
            jsonl_path: None,
            fingerprint: None,
            hf: hf_params,
        }
    };
    let mut current_spec = &data_spec;
    while let sytra_contracts::run_config::DataSpec::Multi { datasets, .. } = current_spec {
        if datasets.is_empty() {
            return Err("Empty multi-dataset list".to_string());
        }
        current_spec = &datasets[0];
    }
    let (kind, params) = match current_spec {
        sytra_contracts::run_config::DataSpec::Hf { hf, .. } => (
            sytra_host::SourceKind::Hf,
            serde_json::to_value(hf).map_err(|e| e.to_string())?,
        ),
        sytra_contracts::run_config::DataSpec::Local { local, .. } => (
            sytra_host::SourceKind::Local,
            serde_json::to_value(local).map_err(|e| e.to_string())?,
        ),
        sytra_contracts::run_config::DataSpec::Synthetic { synthetic, .. } => (
            sytra_host::SourceKind::Synthetic,
            serde_json::to_value(synthetic).map_err(|e| e.to_string())?,
        ),
        sytra_contracts::run_config::DataSpec::Klayer { klayer, .. } => (
            sytra_host::SourceKind::Klayer,
            serde_json::to_value(klayer).map_err(|e| e.to_string())?,
        ),
        sytra_contracts::run_config::DataSpec::Multi { .. } => unreachable!(),
    };
    let spec = sytra_host::DatasetSpec {
        source: kind,
        train_mode: sytra_contracts::run_config::TrainMode::Sft,
        params,
    };
    let provider = sytra_host::get_datasource(kind);
    let preview = provider
        .preview(&spec, rows)
        .await
        .map_err(|e| format!("preview error: {e}"))?;
    let mut result = Vec::new();
    result.push(vec!["prompt".to_string(), "completion".to_string()]);
    for row in preview.rows {
        let prompt = row
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let completion = row
            .get("completion")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        result.push(vec![prompt, completion]);
    }
    Ok(result)
}
