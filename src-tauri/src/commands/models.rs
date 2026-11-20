use serde_json::Value;
use tauri::State;

use sytra_host::catalog::HubCatalogEntry;
use sytra_host::download::{DownloadService, DownloadStatus};
use sytra_host::{catalog, convert};
pub use sytra_host::models::{scan_local_models, LocalModelItem};

use crate::state::AppState;

#[tauri::command]
pub async fn list_catalog(state: State<'_, AppState>) -> Result<Vec<HubCatalogEntry>, String> {
    let vram_mb = state.guard.lock().unwrap_or_else(|p| p.into_inner()).total_vram_mb;
    let vram_mb = (vram_mb > 0).then_some(vram_mb);
    let ram_mb = state.guard.lock().unwrap_or_else(|p| p.into_inner()).total_ram_mb;
    let ram_mb = (ram_mb > 0).then_some(ram_mb);
    let guider = state.guider.lock().unwrap_or_else(|p| p.into_inner());
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
        let guider = state.guider.lock().unwrap_or_else(|p| p.into_inner());
        let entry = catalog::require_catalog_download(&guider, &repo_id)?;
        let vram_mb = state.guard.lock().unwrap_or_else(|p| p.into_inner()).total_vram_mb;
        let vram_mb = (vram_mb > 0).then_some(vram_mb);
        let ram_mb = state.guard.lock().unwrap_or_else(|p| p.into_inner()).total_ram_mb;
        let ram_mb = (ram_mb > 0).then_some(ram_mb);
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

#[tauri::command]
pub async fn list_local_models(
    state: State<'_, AppState>,
    custom_dir: Option<String>,
) -> Result<Vec<LocalModelItem>, String> {
    let workspace = state.workspace.clone();
    tauri::async_runtime::spawn_blocking(move || scan_local_models(&workspace, custom_dir.as_deref()))
        .await
        .map_err(|e| e.to_string())
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
    let workspace = state.workspace.clone();
    tauri::async_runtime::spawn_blocking(move || {
        convert::build_moe_index(
            &workspace,
            &model_path,
            &adapter,
            &expert_format,
            expert_regex.as_deref(),
        )
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn generate_moe_oracle(
    state: State<'_, AppState>,
    model_path: String,
    trust_remote_code: Option<bool>,
    device_map: Option<String>,
) -> Result<Value, String> {
    let workspace = state.workspace.clone();
    let trust = trust_remote_code.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || {
        convert::generate_moe_oracle(&workspace, &model_path, trust, device_map.as_deref())
    })
    .await
    .map_err(|e| e.to_string())?
}
