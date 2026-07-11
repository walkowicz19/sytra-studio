// ─── Sytra Studio — Tauri 2 Library ──────────────────────────────────────────
// Rebuild trigger for new icon set
//
// AppState  — Mutex-wrapped host singletons
// Command Handlers — #[tauri::command] handlers wired directly to frontend actions
// run()     — Tauri builder

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{path::PathBuf, sync::Mutex};

use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

use sytra_contracts::{
    guider::{Compatibility, Guider, HardwareCapabilities, TrainRecipe},
    merge_config::MergeConfig,
    merge_config::MergeMethod,
    operation::{MergeSpec, OpRecord, Operation, TrainSpec},
    run_config::RunConfig,
};
use sytra_host::{
    backend_resolver::BackendResolver, commands, job_runner::JobRunner,
    resource_guard::ResourceGuard, run_archive::RunArchive,
};

// ─── Shared state ─────────────────────────────────────────────────────────────

pub struct AppState {
    pub archive: Mutex<RunArchive>,
    pub runner: Mutex<JobRunner>,
    pub guard: Mutex<ResourceGuard>,
    pub guider: Mutex<Guider>,
    pub workspace: PathBuf,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn write_temp_yaml(dir: &PathBuf, name: &str, value: &Value) -> Result<PathBuf, String> {
    let path = dir.join(name);
    let yaml = serde_yaml::to_string(value).map_err(|e| e.to_string())?;
    std::fs::write(&path, yaml).map_err(|e| e.to_string())?;
    Ok(path)
}

fn spawn_telemetry_stream(
    app: AppHandle,
    op_id: Uuid,
    mut rx: tokio::sync::mpsc::Receiver<sytra_contracts::TelemetryLine>,
) {
    let ev = format!("telemetry:{}", op_id);
    tauri::async_runtime::spawn(async move {
        let mut final_status = sytra_contracts::OpStatus::Done;
        while let Some(line) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&line) {
                let _ = app.emit(&ev, json);
            }
            if let sytra_contracts::TelemetryLine::Event { event, .. } = &line {
                if event == "error" {
                    final_status = sytra_contracts::OpStatus::Error;
                }
            }
        }
        // Persist the final status, but only if the record is still
        // Running — a user cancel (stop_op) already wrote Stopped, and the
        // channel closing afterwards must not clobber it back to Done.
        let mut was_stopped = false;
        if let Some(state) = app.try_state::<AppState>() {
            if let Ok(archive) = state.archive.lock() {
                let res: Result<OpRecord, _> = archive.load(op_id);
                if let Ok(mut record) = res {
                    if record.status == sytra_contracts::OpStatus::Running {
                        record.status = final_status;
                        let _ = archive.store(&record);
                    } else {
                        was_stopped = record.status == sytra_contracts::OpStatus::Stopped;
                    }
                }
            }
        }
        let terminal = if was_stopped {
            r#"{"type":"event","event":"stopped"}"#
        } else {
            r#"{"type":"event","event":"done"}"#
        };
        let _ = app.emit(&ev, terminal);
    });
}

use sytra_host::materialize::materialize_dataset_for_config;

// ─── Commands ─────────────────────────────────────────────────────────────────

#[tauri::command]
async fn start_train(
    state: State<'_, AppState>,
    app: AppHandle,
    config: Value,
) -> Result<String, String> {
    let mut run_config: RunConfig =
        serde_json::from_value(config.clone()).map_err(|e| format!("Bad train config: {e}"))?;

    // Materialize dataset in runs folder before starting train process
    let ws = &state.workspace;
    let dataset_dir = ws.join("runs").join("dataset_materialized");
    materialize_dataset_for_config(&mut run_config.data, run_config.train_mode, &dataset_dir)
        .await?;

    let config_path = {
        std::fs::create_dir_all(ws.join("runs")).map_err(|e| e.to_string())?;
        let config_val = serde_json::to_value(&run_config).unwrap();
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
async fn start_merge(
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
fn stop_op(state: State<'_, AppState>, op_id: String) -> Result<(), String> {
    let id = Uuid::parse_str(&op_id).map_err(|_| "invalid uuid")?;
    let runner = state.runner.lock().map_err(|_| "lock")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::stop_op(id, &runner, &archive)
}

#[tauri::command]
fn list_runs(state: State<'_, AppState>) -> Result<Vec<OpRecord>, String> {
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::list_ops(&archive)
}

#[tauri::command]
fn delete_run(state: State<'_, AppState>, op_id: String) -> Result<(), String> {
    let id = Uuid::parse_str(&op_id).map_err(|_| "invalid uuid")?;
    let archive = state.archive.lock().map_err(|_| "lock")?;
    commands::delete_op(id, &archive)
}

#[tauri::command]
fn estimate_memory(
    state: State<'_, AppState>,
    model_ref: String,
    spec: Value,
) -> Result<u64, String> {
    let rc: RunConfig = serde_json::from_value(spec).map_err(|e| e.to_string())?;
    let train_spec = TrainSpec {
        config: rc,
        config_path: PathBuf::from("dummy"),
    };
    let guard = state.guard.lock().map_err(|_| "lock")?;
    let guider = state.guider.lock().map_err(|_| "lock")?;
    commands::estimate_memory(model_ref, train_spec, &guard, &guider)
}

#[tauri::command]
fn get_settings(state: State<'_, AppState>) -> Result<Value, String> {
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let detected_ram_mb = BackendResolver::detect_system_ram_mb();
    Ok(serde_json::json!({
        "hf_cache_dir": settings.effective_hf_cache(&state.workspace).display().to_string(),
        "is_custom": settings.hf_cache_dir.is_some(),
        "main_memory_limit_mb": settings.main_memory_limit_mb,
        "effective_main_memory_mb": settings.effective_main_memory_mb(detected_ram_mb),
        "detected_ram_mb": detected_ram_mb,
    }))
}

#[tauri::command]
fn set_cache_dir(state: State<'_, AppState>, path: Option<String>) -> Result<Value, String> {
    let mut settings = sytra_host::settings::AppSettings::load(&state.workspace);
    settings.hf_cache_dir = path.map(PathBuf::from);
    settings.save(&state.workspace)?;
    Ok(serde_json::json!({
        "hf_cache_dir": settings.effective_hf_cache(&state.workspace).display().to_string(),
        "is_custom": settings.hf_cache_dir.is_some(),
    }))
}

#[tauri::command]
fn set_main_memory_limit(
    state: State<'_, AppState>,
    limit_mb: Option<u64>,
) -> Result<Value, String> {
    let detected_ram_mb = BackendResolver::detect_system_ram_mb();
    if let Some(limit) = limit_mb {
        if limit < 2048 || limit > detected_ram_mb {
            return Err(format!(
                "Main memory limit must be between 2048 and {detected_ram_mb} MB"
            ));
        }
    }
    let mut settings = sytra_host::settings::AppSettings::load(&state.workspace);
    settings.main_memory_limit_mb = limit_mb;
    settings.save(&state.workspace)?;
    let effective = settings.effective_main_memory_mb(detected_ram_mb);
    state.guard.lock().map_err(|_| "lock")?.total_ram_mb = effective;
    Ok(serde_json::json!({
        "main_memory_limit_mb": settings.main_memory_limit_mb,
        "effective_main_memory_mb": effective,
        "detected_ram_mb": detected_ram_mb,
    }))
}

#[tauri::command]
fn get_hardware_info() -> Result<Value, String> {
    use sytra_contracts::run_config::BackendKind;
    let backend = BackendResolver::resolve(BackendKind::Auto);
    let vram_mb = sytra_host::BackendResolver::detect_system_vram_mb();
    let ram_mb = sytra_host::BackendResolver::detect_system_ram_mb();
    Ok(serde_json::json!({
        "backend":  format!("{:?}", backend).to_lowercase(),
        "vram_mb":  vram_mb,
        "ram_mb":   ram_mb,
    }))
}

#[tauri::command]
fn guider_recommend(
    state: State<'_, AppState>,
    hardware: Option<Value>,
) -> Result<Vec<TrainRecipe>, String> {
    let hw = match hardware {
        Some(val) => serde_json::from_value(val).map_err(|e| e.to_string())?,
        None => HardwareCapabilities {
            accelerator: "cuda".to_string(),
            total_vram_mb: 24576,
            total_ram_mb: 65536,
        },
    };
    let guider = state.guider.lock().map_err(|_| "lock")?;
    commands::guider_recommend(hw, &guider)
}

#[tauri::command]
fn merge_check(
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
async fn publish_run(
    state: State<'_, AppState>,
    app: AppHandle,
    run_op_id: String,
    repo_id: String,
    private: bool,
    token: String,
) -> Result<String, String> {
    let run_id = Uuid::parse_str(&run_op_id).map_err(|_| "Invalid run UUID")?;

    // Resolve run from archive to get the artifact path
    let archive = state.archive.lock().map_err(|_| "lock")?;
    let run_record = archive
        .load(run_id)
        .map_err(|e| format!("Failed to find run: {e}"))?;

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
async fn preview_dataset(source: Value, rows: usize) -> Result<Vec<Vec<String>>, String> {
    // Determine the data spec
    let data_spec: sytra_contracts::run_config::DataSpec = if source.get("source").is_some() {
        serde_json::from_value(source).map_err(|e| e.to_string())?
    } else {
        // Fallback to HF if the raw parameters are passed directly
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
            serde_json::to_value(hf).unwrap(),
        ),
        sytra_contracts::run_config::DataSpec::Local { local, .. } => (
            sytra_host::SourceKind::Local,
            serde_json::to_value(local).unwrap(),
        ),
        sytra_contracts::run_config::DataSpec::Synthetic { synthetic, .. } => (
            sytra_host::SourceKind::Synthetic,
            serde_json::to_value(synthetic).unwrap(),
        ),
        sytra_contracts::run_config::DataSpec::Klayer { klayer, .. } => (
            sytra_host::SourceKind::Klayer,
            serde_json::to_value(klayer).unwrap(),
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

    // Convert preview rows to string[][] format (header + row values)
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

fn find_project_root() -> Option<PathBuf> {
    if let Ok(mut dir) = std::env::current_exe() {
        while dir.pop() {
            if dir.join("runner").join("sytra_runner").exists() {
                return Some(dir);
            }
        }
    }
    None
}

// ─── App entry ────────────────────────────────────────────────────────────────

pub fn run() {
    let workspace = std::env::var("SYTRA_WORKSPACE")
        .map(PathBuf::from)
        .ok()
        .or_else(find_project_root)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let runs_dir = workspace.join("runs");
    std::fs::create_dir_all(&runs_dir).ok();

    // Start background provisioning of virtual environments (automatically installs mergekit)
    let env_provisioner = sytra_host::EnvProvisioner::new(&workspace);
    std::thread::spawn(move || {
        let _ = env_provisioner.provision_merge();
        let _ = env_provisioner.provision_train();
    });

    let detected_vram_mb = BackendResolver::detect_system_vram_mb();
    let detected_ram_mb = BackendResolver::detect_system_ram_mb();
    let memory_limit_mb = sytra_host::settings::AppSettings::load(&workspace)
        .effective_main_memory_mb(detected_ram_mb);
    let state = AppState {
        archive: Mutex::new(RunArchive::new(&runs_dir)),
        runner: Mutex::new(JobRunner::new(&workspace)),
        guard: Mutex::new(ResourceGuard::new(
            detected_vram_mb,
            memory_limit_mb,
            500 * 1024,
        )),
        guider: Mutex::new(Guider::new()),
        workspace,
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            start_train,
            start_merge,
            stop_op,
            list_runs,
            delete_run,
            estimate_memory,
            get_hardware_info,
            get_settings,
            set_cache_dir,
            set_main_memory_limit,
            guider_recommend,
            merge_check,
            preview_dataset,
            publish_run,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Sytra Studio");
}
