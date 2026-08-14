use serde_json::Value;
use tauri::State;

use sytra_contracts::guider::{HardwareCapabilities, TrainRecipe};
use sytra_host::{commands, BackendResolver};

use crate::helpers::detected_ram_or_err;
use crate::state::AppState;

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> Result<Value, String> {
    let settings = sytra_host::settings::AppSettings::load(&state.workspace);
    let detected_ram_mb = detected_ram_or_err()?;
    Ok(serde_json::json!({
        "hf_cache_dir": settings.effective_hf_cache(&state.workspace).display().to_string(),
        "is_custom": settings.hf_cache_dir.is_some(),
        "main_memory_limit_mb": settings.main_memory_limit_mb,
        "effective_main_memory_mb": settings.effective_main_memory_mb(detected_ram_mb),
        "detected_ram_mb": detected_ram_mb,
        "default_context_window": settings.default_context_window,
        "default_temperature": settings.default_temperature,
        "enable_flash_attention": settings.enable_flash_attention,
        "kv_cache_quant": settings.kv_cache_quant,
        "vram_limit_mb": settings.vram_limit_mb,
        "cpu_kv_cache": settings.cpu_kv_cache,
        "vram_expert_cache_mb": settings.vram_expert_cache_mb,
        "memory_limit_note": "main_memory_limit_mb gates preflight estimates only; it does not cap process RSS.",
    }))
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
    state.guard.lock().map_err(|_| "lock")?.total_ram_mb = effective;
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
    let guider = state.guider.lock().map_err(|_| "lock")?;
    commands::guider_recommend(hw, &guider)
}
