use std::path::PathBuf;

use serde_json::Value;
use tauri::State;

use sytra_host::catalog::HubCatalogEntry;
use sytra_host::download::{DownloadService, DownloadStatus};
use sytra_host::workspace::default_model_dir;
use sytra_host::{catalog, convert};

use crate::state::AppState;

#[tauri::command]
pub async fn list_catalog(state: State<'_, AppState>) -> Result<Vec<HubCatalogEntry>, String> {
    let vram_mb = state.guard.lock().ok().map(|g| g.total_vram_mb).filter(|v| *v > 0);
    let ram_mb = state.guard.lock().ok().map(|g| g.total_ram_mb).filter(|v| *v > 0);
    let guider = state.guider.lock().map_err(|_| "lock")?;
    Ok(catalog::hub_entries(&guider, vram_mb, ram_mb))
}

#[derive(serde::Serialize)]
pub struct DownloadResponse {
    pub op_id: String,
}

#[tauri::command]
pub async fn download_model(
    state: State<'_, AppState>,
    repo_id: String,
    purpose: String,
    dest_dir: Option<String>,
    quant: Option<String>,
) -> Result<DownloadResponse, String> {
    {
        let guider = state.guider.lock().map_err(|_| "lock")?;
        let entry = catalog::require_catalog_download(&guider, &repo_id)?;
        let vram_mb = state.guard.lock().ok().map(|g| g.total_vram_mb).filter(|v| *v > 0);
        let ram_mb = state.guard.lock().ok().map(|g| g.total_ram_mb).filter(|v| *v > 0);
        let alerts = sytra_contracts::alerts_for(entry, vram_mb, ram_mb);
        if let Some(blocked) = alerts.iter().find(|a| a.blocks_download) {
            return Err(format!("{} ({})", blocked.message, blocked.code));
        }
    }
    let started = state.downloads.start(
        &repo_id,
        &purpose,
        dest_dir.as_deref(),
        quant.as_deref(),
        None,
    )?;
    Ok(DownloadResponse {
        op_id: started.op_id,
    })
}

#[tauri::command]
pub async fn cancel_download(
    state: State<'_, AppState>,
    dest_dir: Option<String>,
) -> Result<bool, String> {
    state.downloads.cancel(dest_dir.as_deref())?;
    Ok(true)
}

#[tauri::command]
pub async fn get_download_status(
    dest_dir: Option<String>,
) -> Result<Option<DownloadStatus>, String> {
    Ok(DownloadService::read_status(dest_dir.as_deref()))
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
pub async fn list_local_models(
    state: State<'_, AppState>,
    custom_dir: Option<String>,
) -> Result<Vec<LocalModelItem>, String> {
    let mut items = Vec::new();
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let mut search_dirs = vec![
        ("downloaded", default_model_dir()),
        ("downloaded", settings.effective_hf_cache(&state.workspace)),
        (
            "downloaded",
            state.workspace.join("runner").join(".hf-cache"),
        ),
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
        let mut stack = vec![base_dir.clone()];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if path.components().count() <= base_dir.components().count() + 4 {
                            stack.push(path);
                        }
                    } else if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            let ext_lower = ext.to_lowercase();
                            if matches!(ext_lower.as_str(), "gguf" | "safetensors" | "bin" | "pth")
                            {
                                let p_str = path.to_string_lossy().to_string();
                                if seen.insert(p_str.clone()) {
                                    let size_gb = (path.metadata().map(|m| m.len()).unwrap_or(0)
                                        as f64)
                                        / (1024.0 * 1024.0 * 1024.0);
                                    items.push(LocalModelItem {
                                        id: path
                                            .file_stem()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .to_string(),
                                        name: path
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .to_string(),
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
pub async fn convert_model(
    state: State<'_, AppState>,
    model: String,
    outtype: Option<String>,
    outfile: Option<String>,
) -> Result<String, String> {
    convert::convert_model(
        &state.workspace,
        &model,
        outtype.as_deref(),
        outfile.as_deref(),
    )
}

#[tauri::command]
pub async fn export_model(
    state: State<'_, AppState>,
    model: String,
    name: Option<String>,
    context: Option<usize>,
) -> Result<String, String> {
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let ctx_val = context.unwrap_or(settings.default_context_window);
    convert::export_model(&state.workspace, &model, name.as_deref(), ctx_val)
}

#[tauri::command]
pub async fn build_moe_index(
    state: State<'_, AppState>,
    model_path: String,
    adapter: String,
    expert_format: String,
    expert_regex: Option<String>,
) -> Result<Value, String> {
    convert::build_moe_index(
        &state.workspace,
        &model_path,
        &adapter,
        &expert_format,
        expert_regex.as_deref(),
    )
}
