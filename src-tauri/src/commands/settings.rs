use serde_json::Value;
use tauri::State;

use sytra_contracts::guider::{HardwareCapabilities, TrainRecipe};
use sytra_host::{commands, settings::InferenceSettingsInput, BackendResolver};

use crate::helpers::detected_ram_or_err;
use crate::state::AppState;

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Result<Value, String> {
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let detected_ram_mb = detected_ram_or_err()?;
    Ok(sytra_host::settings::settings_payload(
        &settings,
        &state.workspace,
        detected_ram_mb,
    ))
}

#[tauri::command]
pub fn set_inference_settings(
    state: State<'_, AppState>,
    settings: InferenceSettingsInput,
) -> Result<Value, String> {
    let detected_ram_mb = detected_ram_or_err()?;
    sytra_host::settings::apply_inference_settings(&state.workspace, detected_ram_mb, settings)
}

#[tauri::command]
pub fn set_cache_dir(state: State<'_, AppState>, path: Option<String>) -> Result<Value, String> {
    let mut settings = sytra_host::settings::AppSettings::load(&state.workspace);
    settings.hf_cache_dir = path.map(std::path::PathBuf::from);
    settings.save(&state.workspace)?;
    Ok(serde_json::json!({
        "hf_cache_dir": settings.effective_hf_cache(&state.workspace).display().to_string(),
        "is_custom": settings.hf_cache_dir.is_some(),
    }))
}

#[tauri::command]
pub fn set_main_memory_limit(
    state: State<'_, AppState>,
    limit_mb: Option<u64>,
) -> Result<Value, String> {
    let detected_ram_mb = detected_ram_or_err()?;
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
    state.guard.lock().unwrap_or_else(|p| p.into_inner()).total_ram_mb = effective;
    Ok(serde_json::json!({
        "main_memory_limit_mb": settings.main_memory_limit_mb,
        "effective_main_memory_mb": effective,
        "detected_ram_mb": detected_ram_mb,
    }))
}

#[tauri::command]
pub fn get_hardware_info() -> Result<Value, String> {
    use sytra_contracts::run_config::BackendKind;
    let backend = BackendResolver::resolve(BackendKind::Auto);
    Ok(serde_json::json!({
        "backend": format!("{:?}", backend).to_lowercase(),
        "vram_mb": BackendResolver::detect_system_vram_mb(),
        "ram_mb": BackendResolver::detect_system_ram_mb(),
    }))
}

#[tauri::command]
pub fn guider_recommend(
    state: State<'_, AppState>,
    hardware: Option<Value>,
) -> Result<Vec<TrainRecipe>, String> {
    let hw = match hardware {
        Some(val) => serde_json::from_value(val).map_err(|e| e.to_string())?,
        None => {
            let vram = BackendResolver::detect_system_vram_mb().ok_or_else(|| {
                "Could not detect VRAM; pass hardware explicitly or fix detection".to_string()
            })?;
            let ram = detected_ram_or_err()?;
            HardwareCapabilities {
                accelerator: format!("{:?}", BackendResolver::detect_best_backend()).to_lowercase(),
                total_vram_mb: vram,
                total_ram_mb: ram,
            }
        }
    };
    let guider = state.guider.lock().unwrap_or_else(|p| p.into_inner());
    commands::guider_recommend(hw, &guider)
}

#[tauri::command]
pub fn get_mcp_status(state: State<'_, AppState>) -> Result<Value, String> {
    let report = sytra_host::mcp_install::status_report(&state.workspace, None);
    serde_json::to_value(report).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn install_mcp_clients(
    state: State<'_, AppState>,
    clients: Option<Vec<String>>,
) -> Result<Value, String> {
    if let Err(err) = sytra_host::ensure_runner_deployed(&state.workspace) {
        eprintln!("failed to deploy embedded runner: {err}");
    }
    let selection = clients.filter(|v| !v.is_empty());
    let outcomes = sytra_host::mcp_install::install_mcp_clients(
        &state.workspace,
        selection.as_deref(),
        None,
        false,
    )
    .map_err(|e| e.to_string())?;
    let report = sytra_host::mcp_install::status_report(&state.workspace, None);
    Ok(serde_json::json!({
        "outcomes": outcomes,
        "status": report,
    }))
}

#[tauri::command]
pub fn uninstall_mcp_clients(
    state: State<'_, AppState>,
    clients: Option<Vec<String>>,
) -> Result<Value, String> {
    let selection = clients.filter(|v| !v.is_empty());
    let outcomes = sytra_host::mcp_install::uninstall_mcp_clients(
        &state.workspace,
        selection.as_deref(),
        false,
    )
    .map_err(|e| e.to_string())?;
    let report = sytra_host::mcp_install::status_report(&state.workspace, None);
    Ok(serde_json::json!({
        "outcomes": outcomes,
        "status": report,
    }))
}
