use tauri::State;

use crate::state::AppState;

#[tauri::command]
pub async fn start_chat_server(
    state: State<'_, AppState>,
    model_path: String,
    context: Option<usize>,
    vram_limit: Option<usize>,
    cpu_kv_cache: Option<bool>,
) -> Result<bool, String> {
    state
        .chat
        .start(&model_path, context, vram_limit, cpu_kv_cache)?;
    Ok(true)
}

#[tauri::command]
pub async fn stop_chat_server(state: State<'_, AppState>) -> Result<bool, String> {
    state.chat.stop()?;
    Ok(true)
}

#[tauri::command]
pub async fn plan_inference(
    state: State<'_, AppState>,
    model_path: String,
    context: Option<usize>,
    export_runtimes: Option<bool>,
) -> Result<serde_json::Value, String> {
    let workspace = state.workspace.clone();
    let (vram, ram) = {
        let guard = state.guard.lock().map_err(|_| "lock")?;
        (guard.total_vram_mb, guard.total_ram_mb)
    };
    if vram == 0 || ram == 0 {
        return Err("Hardware memory could not be detected; refusing to plan inference".into());
    }
    let export = export_runtimes.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::plan_inference(
            &workspace,
            &model_path,
            vram,
            Some(ram),
            context,
            export,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}
