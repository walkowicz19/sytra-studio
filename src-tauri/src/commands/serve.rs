use tauri::State;

use crate::state::AppState;

#[tauri::command]
pub fn get_model_settings(state: State<'_, AppState>, model_path: String) -> Result<sytra_host::model_settings::ModelSettings, String> {
    sytra_host::model_settings::load(&state.workspace, std::path::Path::new(&model_path))
}

#[tauri::command]
pub fn set_model_settings(state: State<'_, AppState>, model_path: String, settings: sytra_host::model_settings::ModelSettings) -> Result<(), String> {
    sytra_host::model_settings::save(&state.workspace, std::path::Path::new(&model_path), &settings)
}

#[tauri::command]
pub async fn start_chat_server(
    state: State<'_, AppState>,
    model_path: String,
    context: Option<usize>,
    vram_limit: Option<usize>,
    cpu_kv_cache: Option<bool>,
    allow_unverified: Option<bool>,
) -> Result<bool, String> {
    let chat = state.chat.clone();
    tauri::async_runtime::spawn_blocking(move || {
        chat.start_with_options(&model_path, context, vram_limit, cpu_kv_cache, allow_unverified)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(true)
}

#[tauri::command]
pub async fn stop_chat_server(state: State<'_, AppState>) -> Result<bool, String> {
    state.chat.stop()?;
    Ok(true)
}

#[tauri::command]
pub fn chat_server_running(state: State<'_, AppState>) -> bool {
    state.chat.is_running()
}

#[tauri::command]
pub fn chat_mcp_token(state: State<'_, AppState>) -> String {
    state.chat.mcp_token()
}

#[tauri::command]
pub fn list_chat_skills(state: State<'_, AppState>) -> Vec<sytra_host::chat_skills::ChatSkill> {
    sytra_host::chat_skills::list(&state.workspace)
}

#[tauri::command]
pub fn read_chat_skill(state: State<'_, AppState>, id: String) -> Result<String, String> {
    sytra_host::chat_skills::read(&state.workspace, &id)
}

#[tauri::command]
pub fn install_chat_skill(
    state: State<'_, AppState>,
    name: String,
    content: String,
) -> Result<sytra_host::chat_skills::ChatSkill, String> {
    sytra_host::chat_skills::install(&state.workspace, &name, &content)
}

#[tauri::command]
pub fn install_chat_skill_file(
    state: State<'_, AppState>,
    path: String,
    name: Option<String>,
) -> Result<sytra_host::chat_skills::ChatSkill, String> {
    sytra_host::chat_skills::import_file(
        &state.workspace,
        std::path::Path::new(&path),
        name.as_deref(),
    )
}

#[tauri::command]
pub fn remove_chat_skill(state: State<'_, AppState>, id: String) -> Result<(), String> {
    sytra_host::chat_skills::remove(&state.workspace, &id)
}

#[tauri::command]
pub async fn plan_inference(
    state: State<'_, AppState>,
    model_path: String,
    context: Option<usize>,
    export_runtimes: Option<bool>,
    allow_unverified: Option<bool>,
) -> Result<serde_json::Value, String> {
    let workspace = state.workspace.clone();
    let (vram, ram) = {
        let guard = state.guard.lock().unwrap_or_else(|p| p.into_inner());
        (guard.total_vram_mb, guard.total_ram_mb)
    };
    if vram == 0 || ram == 0 {
        return Err("Hardware memory could not be detected; refusing to plan inference".into());
    }
    let export = export_runtimes.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::inference::plan_inference_with_options(
            &workspace,
            &model_path,
            vram,
            Some(ram),
            context,
            export,
            allow_unverified,
        )
    })
    .await
    .map_err(|e| e.to_string())?
}


#[derive(serde::Serialize)]
pub struct ChatTargetsReport {
    pub providers: sytra_host::chat_agents::LiveProvidersStatus,
    pub local_models: Vec<crate::commands::LocalModelItem>,
}

#[tauri::command]
pub async fn discover_chat_targets(state: State<'_, AppState>) -> Result<ChatTargetsReport, String> {
    let workspace = state.workspace.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let providers = sytra_host::chat_agents::probe_chat_endpoints();
        let local_models = crate::commands::scan_local_models(&workspace, None);
        ChatTargetsReport {
            providers,
            local_models,
        }
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn start_ollama_daemon() -> Result<bool, String> {
    tauri::async_runtime::spawn_blocking(sytra_host::chat_agents::start_ollama_serve)
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn get_ollama_context(base_url: String, model: String) -> Result<Option<usize>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::chat_agents::ollama_model_context(&base_url, &model)
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn chat_http_relay_stream(
    url: String,
    body: serde_json::Value,
    channel: tauri::ipc::Channel<String>,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::chat_agents::stream_http_relay(&url, &body, |line| {
            channel.send(line.to_string()).is_ok()
        })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn list_chat_agents(state: State<'_, AppState>) -> Result<Vec<sytra_host::chat_agents::AgentProfile>, String> {
    sytra_host::chat_agents::list_agents(&state.workspace)
}

#[tauri::command]
pub fn save_chat_agent(
    state: State<'_, AppState>,
    agent: sytra_host::chat_agents::AgentProfile,
) -> Result<sytra_host::chat_agents::AgentProfile, String> {
    sytra_host::chat_agents::save_agent(&state.workspace, agent)
}

#[tauri::command]
pub fn delete_chat_agent(state: State<'_, AppState>, id: String) -> Result<(), String> {
    sytra_host::chat_agents::delete_agent(&state.workspace, &id)
}

#[tauri::command]
pub fn list_agent_memories(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<Vec<sytra_host::chat_agents::MemoryEntry>, String> {
    sytra_host::chat_agents::list_memories(&state.workspace, agent_id.as_deref())
}

#[tauri::command]
pub fn save_agent_memory(
    state: State<'_, AppState>,
    agent_id: String,
    session_id: String,
    session_title: String,
    kind: String,
    summary: String,
    tags: Vec<String>,
) -> Result<sytra_host::chat_agents::MemoryEntry, String> {
    sytra_host::chat_agents::save_memory(
        &state.workspace,
        &agent_id,
        &session_id,
        &session_title,
        &kind,
        &summary,
        tags,
    )
}

#[tauri::command]
pub fn recall_agent_memories(
    state: State<'_, AppState>,
    agent_id: Option<String>,
    query: String,
    limit: Option<usize>,
) -> Result<Vec<sytra_host::chat_agents::MemoryEntry>, String> {
    sytra_host::chat_agents::recall_memories(
        &state.workspace,
        agent_id.as_deref(),
        &query,
        limit.unwrap_or(6),
    )
}

#[tauri::command]
pub fn delete_agent_memory(state: State<'_, AppState>, id: String) -> Result<(), String> {
    sytra_host::chat_agents::delete_memory(&state.workspace, &id)
}

#[tauri::command]
pub async fn execute_agent_tool(
    state: State<'_, AppState>,
    agent_id: String,
    tool_name: String,
    arguments: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let workspace = state.workspace.clone();
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::chat_agents::execute_agent_tool(&workspace, &agent_id, &tool_name, &arguments)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn search_web_direct(
    query: String,
    max_results: Option<usize>,
) -> Result<sytra_host::chat_agents::WebSearchResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::chat_agents::search_web_native(&query, max_results.unwrap_or(5))
    })
    .await
    .map_err(|e| e.to_string())
}


#[tauri::command]
pub async fn unload_chat_model(
    state: State<'_, AppState>,
    provider: String,
    base_url: String,
    model: String,
) -> Result<String, String> {
    let chat = state.chat.clone();
    tauri::async_runtime::spawn_blocking(move || {
        sytra_host::chat_agents::unload_chat_model(&chat, &provider, &base_url, &model)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn voice_status(state: State<'_, AppState>) -> sytra_host::VoiceRuntimeStatus {
    state.voice.status()
}

#[tauri::command]
pub fn setup_kokoro_voice(state: State<'_, AppState>) -> sytra_host::VoiceRuntimeStatus {
    state.voice.setup_kokoro_async()
}

#[tauri::command]
pub async fn voice_synthesize(
    state: State<'_, AppState>,
    text: String,
    voice: Option<String>,
    speed: Option<f32>,
    lang: Option<String>,
) -> Result<sytra_host::VoiceAudioResponse, String> {
    let svc = state.voice.clone();
    tauri::async_runtime::spawn_blocking(move || {
        svc.synthesize(&text, voice.as_deref(), speed, lang.as_deref())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn voice_transcribe(
    state: State<'_, AppState>,
    audio_base64: String,
    language: Option<String>,
) -> Result<sytra_host::VoiceTranscriptResponse, String> {
    let svc = state.voice.clone();
    tauri::async_runtime::spawn_blocking(move || svc.transcribe(&audio_base64, language.as_deref()))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn stop_voice_engine(state: State<'_, AppState>) -> Result<(), String> {
    state.voice.stop()
}
