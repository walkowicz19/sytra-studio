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
    resource_guard::ResourceGuard, run_archive::RunArchive, settings::AppSettings,
};

// ─── Shared state ─────────────────────────────────────────────────────────────

use std::sync::Arc;

pub struct AppState {
    pub archive: Mutex<RunArchive>,
    pub runner: Mutex<JobRunner>,
    pub guard: Mutex<ResourceGuard>,
    pub guider: Mutex<Guider>,
    pub workspace: PathBuf,
    pub active_download_pid: Arc<Mutex<Option<u32>>>,
    pub active_server_pid: Arc<Mutex<Option<u32>>>,
}

fn get_python_executable() -> &'static str {
    if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    }
}

fn get_user_home_dir() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("USERPROFILE") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    None
}

fn get_default_model_dir() -> PathBuf {
    if let Some(home) = get_user_home_dir() {
        home.join("lm-studio models")
    } else {
        PathBuf::from("./lm-studio models")
    }
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

#[derive(serde::Serialize)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub size_gb: f64,
    pub format: String,
    pub tags: Vec<String>,
    pub recommended: bool,
}

#[tauri::command]
async fn list_catalog(state: State<'_, AppState>) -> Result<Vec<CatalogEntry>, String> {
    let vram_mb = state
        .guard
        .lock()
        .map(|g| g.total_vram_mb)
        .unwrap_or(12288);
    let vram_gb = (vram_mb as f64) / 1024.0;

    let entries = vec![
        CatalogEntry {
            id: "ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF".into(),
            name: "Kimi VL A3B Thinking".into(),
            size_gb: 9.81,
            format: "gguf".into(),
            tags: vec!["vision".into(), "coding".into(), "thinking".into()],
            recommended: vram_gb >= 8.0,
        },
        CatalogEntry {
            id: "unsloth/Kimi-K2.7-Code-GGUF".into(),
            name: "Kimi K2.7 Coder (MoE)".into(),
            size_gb: 295.0,
            format: "gguf".into(),
            tags: vec!["coding".into(), "moe".into(), "large".into()],
            recommended: false,
        },
        CatalogEntry {
            id: "unsloth/GLM-5.2-GGUF".into(),
            name: "GLM-5.2 744B MoE (GGUF)".into(),
            size_gb: 370.0,
            format: "gguf".into(),
            tags: vec!["moe".into(), "frontier".into(), "large".into()],
            recommended: false,
        },
        CatalogEntry {
            id: "unsloth/DeepSeek-V3-GGUF".into(),
            name: "DeepSeek V3 671B MoE".into(),
            size_gb: 330.0,
            format: "gguf".into(),
            tags: vec!["moe".into(), "coding".into(), "large".into()],
            recommended: false,
        },
        CatalogEntry {
            id: "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF".into(),
            name: "DeepSeek R1 Distill Qwen 14B".into(),
            size_gb: 9.0,
            format: "gguf".into(),
            tags: vec!["reasoning".into(), "quant".into(), "fast".into()],
            recommended: vram_gb >= 8.0,
        },
        CatalogEntry {
            id: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".into(),
            name: "Qwen2.5 Coder 7B Instruct".into(),
            size_gb: 4.7,
            format: "gguf".into(),
            tags: vec!["coding".into(), "instruct".into(), "fast".into()],
            recommended: vram_gb >= 6.0,
        },
        CatalogEntry {
            id: "Qwen/Qwen2.5-3B-Instruct-GGUF".into(),
            name: "Qwen2.5 3B Instruct".into(),
            size_gb: 2.1,
            format: "gguf".into(),
            tags: vec!["small".into(), "fast".into(), "lightweight".into()],
            recommended: true,
        },
        CatalogEntry {
            id: "microsoft/phi-4-gguf".into(),
            name: "Phi-4 14B Instruct".into(),
            size_gb: 9.1,
            format: "gguf".into(),
            tags: vec!["reasoning".into(), "small".into(), "math".into()],
            recommended: vram_gb >= 8.0,
        },
        CatalogEntry {
            id: "bartowski/gemma-2-9b-it-GGUF".into(),
            name: "Gemma-2 9B IT".into(),
            size_gb: 5.8,
            format: "gguf".into(),
            tags: vec!["chat".into(), "general".into()],
            recommended: vram_gb >= 6.0,
        },
        CatalogEntry {
            id: "mradermacher/Mixtral-8x22B-Instruct-v0.1-GGUF".into(),
            name: "Mixtral 8x22B Instruct".into(),
            size_gb: 141.0,
            format: "gguf".into(),
            tags: vec!["moe".into(), "multilingual".into()],
            recommended: false,
        },
        CatalogEntry {
            id: "unsloth/mistral-7b-v0.3-bnb-4bit".into(),
            name: "Mistral 7B v0.3 (4-bit)".into(),
            size_gb: 4.1,
            format: "safetensors".into(),
            tags: vec!["finetune".into(), "4-bit".into(), "unsloth".into()],
            recommended: true,
        },
        CatalogEntry {
            id: "bartowski/Llama-3.3-70B-Instruct-GGUF".into(),
            name: "Llama 3.3 70B Instruct (Q4)".into(),
            size_gb: 42.5,
            format: "gguf".into(),
            tags: vec!["quant".into(), "frontier".into()],
            recommended: false,
        },
        CatalogEntry {
            id: "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF".into(),
            name: "SmolLM2 1.7B Instruct".into(),
            size_gb: 1.1,
            format: "gguf".into(),
            tags: vec!["small".into(), "edge".into(), "fast".into()],
            recommended: true,
        },
    ];

    Ok(entries)
}

#[derive(serde::Serialize)]
pub struct DownloadResponse {
    pub op_id: String,
}

#[tauri::command]
async fn download_model(
    state: State<'_, AppState>,
    repo_id: String,
    purpose: String,
    dest_dir: Option<String>,
    quant: Option<String>,
) -> Result<DownloadResponse, String> {
    let op_id = Uuid::new_v4();
    let workspace = state.workspace.clone();
    let runner_dir = workspace.join("runner");
    let script_path = runner_dir.join("scripts").join("download_gguf_model.py");

    let repo_id_clone = repo_id.clone();
    let dest_dir_clone = dest_dir.clone();
    let quant_clone = quant.unwrap_or_else(|| "auto".into());
    let _purpose = purpose;
    let pid_ref = state.active_download_pid.clone();

    std::thread::spawn(move || {
        let python_exe = "python";
        let mut cmd = std::process::Command::new(python_exe);
        cmd.arg(&script_path)
            .arg("--model")
            .arg(&repo_id_clone)
            .arg("--quant")
            .arg(&quant_clone);
        if let Some(dest) = dest_dir_clone {
            cmd.arg("--dest").arg(dest);
        }
        cmd.current_dir(&workspace);

        if let Ok(mut child) = cmd.spawn() {
            let pid = child.id();
            if let Ok(mut g) = pid_ref.lock() {
                *g = Some(pid);
            }
            let _ = child.wait();
            if let Ok(mut g) = pid_ref.lock() {
                if *g == Some(pid) {
                    *g = None;
                }
            }
        }
    });

    Ok(DownloadResponse {
        op_id: op_id.to_string(),
    })
}

#[tauri::command]
async fn cancel_download(
    state: State<'_, AppState>,
    dest_dir: Option<String>,
) -> Result<bool, String> {
    if let Ok(mut pid_guard) = state.active_download_pid.lock() {
        if let Some(pid) = *pid_guard {
            #[cfg(target_os = "windows")]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(&["/F", "/T", "/PID", &pid.to_string()])
                    .status();
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = std::process::Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .status();
            }
            *pid_guard = None;
        }
    }

    let target = dest_dir.map(PathBuf::from).unwrap_or_else(get_default_model_dir);
    let status_file = target.join(".download_status.json");
    if status_file.exists() {
        let _ = std::fs::remove_file(status_file);
    }
    Ok(true)
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct LocalModelItem {
    pub id: String,
    pub name: String,
    pub category: String,
    pub path: String,
    pub size_gb: f64,
    pub format: String,
}

#[tauri::command]
async fn list_local_models(state: State<'_, AppState>, custom_dir: Option<String>) -> Result<Vec<LocalModelItem>, String> {
    let mut items = Vec::new();
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let mut search_dirs = vec![
        ("downloaded", get_default_model_dir()),
        ("downloaded", settings.effective_hf_cache(&state.workspace)),
        ("downloaded", state.workspace.join("runner").join(".hf-cache")),
        ("finetuned", state.workspace.join("runs")),
        ("merged", state.workspace.join("runs").join("merged")),
    ];

    if let Some(c_dir) = custom_dir {
        if !c_dir.trim().is_empty() {
            search_dirs.insert(0, ("custom", PathBuf::from(c_dir)));
        }
    }

    let mut seen = std::collections::HashSet::new();

    for (cat, base_dir) in search_dirs {
        if !base_dir.exists() {
            continue;
        }
        // Recursively walk directory up to 4 levels deep
        let mut stack = vec![base_dir.clone()];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        // Limit depth to avoid scanning huge system drives
                        if path.components().count() <= base_dir.components().count() + 4 {
                            stack.push(path);
                        }
                    } else if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            let ext_lower = ext.to_lowercase();
                            if ext_lower == "gguf" || ext_lower == "safetensors" || ext_lower == "bin" || ext_lower == "pth" {
                                let p_str = path.to_string_lossy().to_string();
                                if !seen.contains(&p_str) {
                                    seen.insert(p_str.clone());
                                    let size_gb = (path.metadata().map(|m| m.len()).unwrap_or(0) as f64) / (1024.0 * 1024.0 * 1024.0);
                                    items.push(LocalModelItem {
                                        id: path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                                        name: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                                        category: cat.to_string(),
                                        path: p_str,
                                        size_gb: (size_gb * 100.0).round() / 100.0,
                                        format: ext.to_string(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(items)
}

#[tauri::command]
async fn start_chat_server(
    state: State<'_, AppState>,
    model_path: String,
    context: Option<usize>,
    vram_limit: Option<usize>,
    cpu_kv_cache: Option<bool>,
) -> Result<bool, String> {
    let workspace = state.workspace.clone();
    let runner_dir = workspace.join("runner");
    let script_path = runner_dir.join("scripts").join("serve_moe.py");
    let pid_ref = state.active_server_pid.clone();

    let settings = AppSettings::load(&workspace);
    let ctx_val = context.unwrap_or(settings.default_context_window);
    let vram_val = vram_limit.unwrap_or(settings.vram_limit_mb.unwrap_or(8192) as usize);
    let use_cpu_kv = cpu_kv_cache.unwrap_or(settings.cpu_kv_cache);

    // Stop existing server if running
    if let Ok(mut g) = pid_ref.lock() {
        if let Some(pid) = *g {
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("taskkill").args(&["/F", "/T", "/PID", &pid.to_string()]).status();
            *g = None;
        }
    }

    std::thread::spawn(move || {
        let mut cmd = std::process::Command::new("python");
        cmd.arg(&script_path)
            .arg("--model").arg(&model_path)
            .arg("--context").arg(ctx_val.to_string())
            .arg("--vram-limit").arg(vram_val.to_string())
            .arg("--port").arg("8080");
        if use_cpu_kv {
            cmd.arg("--cpu-kv-cache");
        }
        cmd.current_dir(&workspace);

        if let Ok(mut child) = cmd.spawn() {
            let pid = child.id();
            if let Ok(mut g) = pid_ref.lock() {
                *g = Some(pid);
            }
            let _ = child.wait();
            if let Ok(mut g) = pid_ref.lock() {
                if *g == Some(pid) {
                    *g = None;
                }
            }
        }
    });

    Ok(true)
}

#[tauri::command]
async fn stop_chat_server(state: State<'_, AppState>) -> Result<bool, String> {
    if let Ok(mut pid_guard) = state.active_server_pid.lock() {
        if let Some(pid) = *pid_guard {
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("taskkill").args(&["/F", "/T", "/PID", &pid.to_string()]).status();
            *pid_guard = None;
        }
    }
    Ok(true)
}

#[tauri::command]
async fn convert_model(
    state: State<'_, AppState>,
    model: String,
    outtype: Option<String>,
    outfile: Option<String>,
) -> Result<String, String> {
    let op_id = Uuid::new_v4();
    let workspace = state.workspace.clone();
    let runner_dir = workspace.join("runner");
    let script_path = runner_dir.join("scripts").join("convert_hf_to_gguf.py");

    let model_clone = model.clone();
    let outtype_val = outtype.unwrap_or_else(|| "auto".into());
    let outfile_clone = outfile.clone();

    std::thread::spawn(move || {
        let python_exe = "python";
        let mut cmd = std::process::Command::new(python_exe);
        cmd.arg(&script_path)
            .arg(&model_clone)
            .arg("--outtype")
            .arg(&outtype_val);
        if let Some(out) = outfile_clone {
            cmd.arg("--outfile").arg(out);
        }
        cmd.current_dir(&workspace);
        let _ = cmd.status();
    });

    Ok(op_id.to_string())
}

#[tauri::command]
async fn export_model(
    state: State<'_, AppState>,
    model: String,
    name: Option<String>,
    context: Option<usize>,
) -> Result<String, String> {
    let op_id = Uuid::new_v4();
    let workspace = state.workspace.clone();
    let runner_dir = workspace.join("runner");
    let script_path = runner_dir.join("scripts").join("export_model.py");

    let model_clone = model.clone();
    let name_clone = name.clone();
    let settings = AppSettings::load(&workspace);
    let ctx_val = context.unwrap_or(settings.default_context_window);

    std::thread::spawn(move || {
        let python_exe = "python";
        let mut cmd = std::process::Command::new(python_exe);
        cmd.arg(&script_path)
            .arg("--model")
            .arg(&model_clone)
            .arg("--context")
            .arg(ctx_val.to_string());
        if let Some(n) = name_clone {
            cmd.arg("--name").arg(n);
        }
        cmd.current_dir(&workspace);
        let _ = cmd.status();
    });

    Ok(op_id.to_string())
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct ModelDownloadStatus {
    pub repo_id: String,
    pub status: String,
    pub downloaded_gb: f64,
    pub total_gb: f64,
    pub pct: f64,
    pub speed_mbps: f64,
    pub eta_seconds: u64,
    pub eta_formatted: String,
    pub current_file: String,
    pub shard_index: usize,
    pub total_shards: usize,
    pub timestamp: f64,
}

#[tauri::command]
async fn get_download_status(
    dest_dir: Option<String>,
) -> Result<Option<ModelDownloadStatus>, String> {
    let target = dest_dir.unwrap_or_else(|| "D:\\lm-studio models".into());
    let status_file = PathBuf::from(target).join(".download_status.json");
    if status_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&status_file) {
            if let Ok(parsed) = serde_json::from_str::<ModelDownloadStatus>(&content) {
                return Ok(Some(parsed));
            }
        }
    }
    Ok(None)
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
        active_download_pid: Arc::new(Mutex::new(None)),
        active_server_pid: Arc::new(Mutex::new(None)),
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
            list_catalog,
            download_model,
            cancel_download,
            convert_model,
            export_model,
            get_download_status,
            list_local_models,
            start_chat_server,
            stop_chat_server,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Sytra Studio");
}
