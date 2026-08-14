use std::path::PathBuf;

use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

use sytra_contracts::{OpStatus, TelemetryLine};

use crate::state::AppState;

pub fn write_temp_yaml(dir: &PathBuf, name: &str, value: &Value) -> Result<PathBuf, String> {
    let path = dir.join(name);
    let yaml = serde_yaml::to_string(value).map_err(|e| e.to_string())?;
    std::fs::write(&path, yaml).map_err(|e| e.to_string())?;
    Ok(path)
}

pub fn spawn_telemetry_stream(
    app: AppHandle,
    op_id: Uuid,
    mut rx: tokio::sync::mpsc::Receiver<TelemetryLine>,
) {
    let ev = format!("telemetry:{}", op_id);
    tauri::async_runtime::spawn(async move {
        let mut final_status = OpStatus::Done;
        while let Some(line) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&line) {
                if let Err(err) = app.emit(&ev, json) {
                    eprintln!("failed to emit telemetry: {err}");
                }
            }
            if let TelemetryLine::Event { event, .. } = &line {
                if event == "error" {
                    final_status = OpStatus::Error;
                }
            }
        }
        let mut was_stopped = false;
        if let Some(state) = app.try_state::<AppState>() {
            if let Ok(archive) = state.archive.lock() {
                if let Ok(mut record) = archive.load(op_id) {
                    if record.status == OpStatus::Running {
                        record.status = final_status;
                        if let Err(err) = archive.store(&record) {
                            eprintln!("failed to store run status: {err}");
                        }
                    } else {
                        was_stopped = record.status == OpStatus::Stopped;
                    }
                }
            }
        }
        let terminal = if was_stopped {
            r#"{"type":"event","event":"stopped"}"#
        } else {
            r#"{"type":"event","event":"done"}"#
        };
        if let Err(err) = app.emit(&ev, terminal) {
            eprintln!("failed to emit terminal telemetry: {err}");
        }
    });
}

pub fn detected_ram_or_err() -> Result<u64, String> {
    sytra_host::BackendResolver::detect_system_ram_mb().ok_or_else(|| {
        "Could not detect system RAM; refusing to estimate. Check OS permissions and try again."
            .into()
    })
}
