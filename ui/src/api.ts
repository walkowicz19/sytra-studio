/**
 * Tauri bridge — wraps invoke() calls with typed payloads.
 * Browser-only mocks below run exclusively when `__TAURI_INTERNALS__` is
 * absent (Vite without the desktop shell). They never execute inside Tauri.
 */
import type {
  RunConfig, MergeConfig, OpRecord, GuiderRecipe,
  CompatResult, MergeMethod, HfParams, CatalogEntry,
  ModelDownloadStatus, LocalModelItem, MoeIndexResult,
  ChatTargetsReport, AgentProfile, MemoryEntry,
  VoiceAudioResponse, VoiceRuntimeStatus, VoiceTranscriptResponse,
} from './types'

export interface AppSettings {
  hf_cache_dir: string
  is_custom: boolean
  main_memory_limit_mb: number | null
  effective_main_memory_mb: number
  detected_ram_mb: number
  default_context_window?: number
  default_temperature?: number
  enable_flash_attention?: boolean
  kv_cache_quant?: string
  vram_limit_mb?: number | null
  cpu_kv_cache?: boolean
  vram_expert_cache_mb?: number | null
}

export interface InferenceSettingsUpdate {
  default_context_window?: number
  default_temperature?: number
  enable_flash_attention?: boolean
  kv_cache_quant?: 'fp16' | 'q8_0' | 'q4_0'
  vram_limit_mb?: number | null
  cpu_kv_cache?: boolean
  vram_expert_cache_mb?: number | null
}

export interface ModelSettings {
  context_length: number | null
  temperature: number | null
  vram_limit_mb: number | null
  vram_guardrail_mb: number | null
  kv_cache_quant: 'fp16' | 'q8_0' | 'q4_0' | null
  cpu_kv_cache: boolean | null
  flash_attention: boolean | null
  idle_ttl_minutes: number | null
  draft_model_path: string | null
  allow_unverified?: boolean | null
}

export interface ChatSkill { id: string; name: string; scope: string }

export interface McpClientStatus {
  id: string
  label: string
  config_path: string
  detected: boolean
  configured: boolean
}

export interface McpStatusReport {
  mcp_binary_path: string
  mcp_binary_exists: boolean
  workspace: string
  runner_ready: boolean
  clients: McpClientStatus[]
  generic_json: string
  vscode_json: string
  codex_toml: string
}

export interface McpActionOutcome {
  id: string
  label: string
  config_path: string
  action: string
}

export interface McpInstallResult {
  outcomes: McpActionOutcome[]
  status: McpStatusReport
}

// Detect if we are inside the Tauri runtime
const isTauri = typeof (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ !== 'undefined'

async function invoke<T>(cmd: string, args?: unknown): Promise<T> {
  if (isTauri) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core')
    return tauriInvoke<T>(cmd, args as Record<string, unknown>)
  }
  // ── Mock responses for browser-only dev ──────────────────────────────
  return mockInvoke<T>(cmd, args)
}

// ─── Commands ─────────────────────────────────────────────────────────────────
export const api = {
  startTrain: (config: RunConfig) =>
    invoke<string>('start_train', { config }),

  startMerge: (config: MergeConfig) =>
    invoke<string>('start_merge', { config }),

  stopOp: (opId: string) =>
    invoke<void>('stop_op', { opId }),

  listRuns: () =>
    invoke<OpRecord[]>('list_runs'),

  deleteRun: (opId: string) =>
    invoke<void>('delete_run', { opId }),

  getHardwareInfo: () =>
    invoke<{ backend: string; vram_mb: number | null; ram_mb: number | null }>('get_hardware_info'),

  getSettings: () =>
    invoke<AppSettings>('get_settings'),

  setCacheDir: (path: string | null) =>
    invoke<AppSettings>('set_cache_dir', { path }),

  setMainMemoryLimit: (limitMb: number | null) =>
    invoke<AppSettings>('set_main_memory_limit', { limitMb }),

  setInferenceSettings: (settings: InferenceSettingsUpdate) =>
    invoke<AppSettings>('set_inference_settings', { settings }),

  getMcpStatus: () =>
    invoke<McpStatusReport>('get_mcp_status'),

  installMcpClients: (clients?: string[]) =>
    invoke<McpInstallResult>('install_mcp_clients', { clients: clients ?? null }),

  uninstallMcpClients: (clients?: string[]) =>
    invoke<McpInstallResult>('uninstall_mcp_clients', { clients: clients ?? null }),

  guiderRecommend: (hardware?: { accelerator: string; total_vram_mb: number; total_ram_mb: number }) =>
    invoke<GuiderRecipe[]>('guider_recommend', { hardware }),

  mergeCheck: (models: string[], method: MergeMethod, baseModel?: string | null) =>
    invoke<CompatResult>('merge_check', { models, method, baseModel: baseModel ?? null }),

  previewDataset: (source: HfParams, rows: number) =>
    invoke<string[][]>('preview_dataset', { source, rows }),

  prepareDataset: (operation: 'merge' | 'split', options: Record<string, unknown>) =>
    invoke<{ output_dir: string; rows?: number; counts?: number[] | Record<string, number>; shards?: { path: string; rows: number }[]; curation?: { removed: Record<string, number>; token_counting?: string } }>('prepare_dataset', { operation, options }),

  publishRun: (runOpId: string, repoId: string, isPrivate: boolean, token: string, license?: string) =>
    invoke<string>('publish_run', { runOpId, repoId, private: isPrivate, token, license: license ?? null }),

  downloadModel: (repoId: string, purpose: 'inference' | 'finetune' | 'merge', destDir?: string, quant?: string) =>
    invoke<{ op_id: string }>('download_model', { repoId, purpose, destDir: destDir ?? null, quant: quant ?? null }),

  cancelDownload: (destDir?: string) =>
    invoke<boolean>('cancel_download', { destDir: destDir ?? null }),

  listCatalog: () =>
    invoke<CatalogEntry[]>('list_catalog'),

  convertModel: (model: string, outtype?: string, outfile?: string) =>
    invoke<string>('convert_model', { model, outtype: outtype ?? 'auto', outfile: outfile ?? null }),

  exportModel: (model: string, name?: string, context?: number) =>
    invoke<string>('export_model', { model, name: name ?? null, context: context ?? 4096 }),

  getDownloadStatus: (destDir?: string) =>
    invoke<ModelDownloadStatus | null>('get_download_status', { destDir: destDir ?? null }),

  listLocalModels: (customDir?: string) =>
    invoke<LocalModelItem[]>('list_local_models', { customDir: customDir ?? null }),

  buildMoeIndex: (
    modelPath: string,
    adapter: string,
    expertFormat: string,
    expertRegex?: string,
  ) =>
    invoke<MoeIndexResult>('build_moe_index', {
      modelPath,
      adapter,
      expertFormat,
      expertRegex: expertRegex?.trim() || null,
    }),

  generateMoeOracle: (
    modelPath: string,
    trustRemoteCode?: boolean,
    deviceMap?: string,
  ) =>
    invoke<{ oracle: string; next?: string }>('generate_moe_oracle', {
      modelPath,
      trustRemoteCode: trustRemoteCode ?? false,
      deviceMap: deviceMap ?? null,
    }),

  startChatServer: (
    modelPath: string,
    context?: number,
    vramLimit?: number,
    cpuKvCache?: boolean,
    allowUnverified?: boolean,
  ) =>
    invoke<boolean>('start_chat_server', {
      modelPath,
      context: context ?? null,
      vramLimit: vramLimit ?? null,
      cpuKvCache: cpuKvCache ?? null,
      allowUnverified: allowUnverified ?? null,
    }),

  stopChatServer: () =>
    invoke<boolean>('stop_chat_server'),

  chatServerRunning: () =>
    invoke<boolean>('chat_server_running'),

  chatMcpToken: () => invoke<string>('chat_mcp_token'),

  listChatSkills: () => invoke<ChatSkill[]>('list_chat_skills'),
  readChatSkill: (id: string) => invoke<string>('read_chat_skill', { id }),
  installChatSkill: (name: string, content: string) =>
    invoke<ChatSkill>('install_chat_skill', { name, content }),
  installChatSkillFromFile: (path: string, name?: string) =>
    invoke<ChatSkill>('install_chat_skill_file', { path, name: name ?? null }),
  removeChatSkill: (id: string) => invoke<void>('remove_chat_skill', { id }),

  getModelSettings: (modelPath: string) =>
    invoke<ModelSettings>('get_model_settings', { modelPath }),

  setModelSettings: (modelPath: string, settings: ModelSettings) =>
    invoke<void>('set_model_settings', { modelPath, settings }),

  planInference: (
    modelPath: string,
    context?: number,
    exportRuntimes?: boolean,
    allowUnverified?: boolean,
  ) =>
    invoke<Record<string, unknown>>('plan_inference', {
      modelPath,
      context: context ?? null,
      exportRuntimes: exportRuntimes ?? false,
      allowUnverified: allowUnverified ?? null,
    }),

  discoverChatTargets: () => invoke<ChatTargetsReport>('discover_chat_targets'),
  startOllamaDaemon: () => invoke<boolean>('start_ollama_daemon'),
  getOllamaContext: (baseUrl: string, model: string) =>
    invoke<number | null>('get_ollama_context', { baseUrl, model }),
  chatHttpRelayStream: async (
    url: string,
    body: Record<string, unknown>,
    onLine: (line: string) => void,
  ): Promise<void> => {
    const tauriPresent = typeof (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ !== 'undefined'
    if (!tauriPresent) throw new Error('Relay requires the desktop app.')
    const { invoke: tauriInvoke, Channel } = await import('@tauri-apps/api/core')
    const channel = new Channel<string>()
    channel.onmessage = onLine
    await tauriInvoke('chat_http_relay_stream', { url, body, channel })
  },

  listChatAgents: () => invoke<AgentProfile[]>('list_chat_agents'),
  saveChatAgent: (agent: AgentProfile) => invoke<AgentProfile>('save_chat_agent', { agent }),
  deleteChatAgent: (id: string) => invoke<void>('delete_chat_agent', { id }),

  listAgentMemories: (agentId?: string) =>
    invoke<MemoryEntry[]>('list_agent_memories', { agentId: agentId ?? null }),
  saveAgentMemory: (entry: {
    agentId: string
    sessionId: string
    sessionTitle: string
    kind: 'fact' | 'compaction' | 'routine'
    summary: string
    tags: string[]
  }) => invoke<MemoryEntry>('save_agent_memory', entry),
  recallAgentMemories: (query: string, agentId?: string, limit?: number) =>
    invoke<MemoryEntry[]>('recall_agent_memories', {
      agentId: agentId ?? null,
      query,
      limit: limit ?? 6,
    }),
  deleteAgentMemory: (id: string) => invoke<void>('delete_agent_memory', { id }),

  executeAgentTool: (agentId: string, toolName: string, argumentsObj: Record<string, unknown>) =>
    invoke<Record<string, unknown>>('execute_agent_tool', {
      agentId,
      toolName,
      arguments: argumentsObj,
    }),

  searchWebDirect: (query: string, maxResults?: number) =>
    invoke<{ query: string; results: { title: string; url: string; snippet: string }[]; error?: string }>(
      'search_web_direct',
      { query, maxResults: maxResults ?? 5 },
    ),

  unloadChatModel: (provider: string, baseUrl: string, model: string) =>
    invoke<string>('unload_chat_model', { provider, baseUrl, model }),

  voiceStatus: () => invoke<VoiceRuntimeStatus>('voice_status'),
  setupKokoroVoice: () => invoke<VoiceRuntimeStatus>('setup_kokoro_voice'),
  voiceSynthesize: (text: string, voice?: string, speed?: number, lang?: string) =>
    invoke<VoiceAudioResponse>('voice_synthesize', {
      text,
      voice: voice ?? 'auto',
      speed: speed ?? 1.0,
      lang: lang ?? 'auto',
    }),
  voiceTranscribe: (audioBase64: string, language?: string) =>
    invoke<VoiceTranscriptResponse>('voice_transcribe', {
      audioBase64,
      language: language ?? 'auto',
    }),
  stopVoiceEngine: () => invoke<void>('stop_voice_engine'),
}

// ─── Mock ─────────────────────────────────────────────────────────────────────
const mockSkillsKey = 'sytra-mock-skills-v1'
function loadMockSkills(): { meta: ChatSkill; content: string }[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(mockSkillsKey) || '[]')
    return Array.isArray(parsed) ? (parsed as { meta: ChatSkill; content: string }[]) : []
  } catch {
    return []
  }
}
function saveMockSkills(items: { meta: ChatSkill; content: string }[]) {
  try {
    localStorage.setItem(mockSkillsKey, JSON.stringify(items))
  } catch { /* ignore */ }
}

function mockInvoke<T>(cmd: string, args?: unknown): Promise<T> {
  const payload = (args ?? {}) as Record<string, unknown>
  if (cmd === 'read_chat_skill') {
    const hit = loadMockSkills().find((s) => s.meta.id === payload.id)
    if (hit) return Promise.resolve(hit.content as T)
    return Promise.reject(new Error('Skills require the desktop app.'))
  }
  if (cmd === 'install_chat_skill') {
    const rawName = String(payload.name ?? '').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'custom-skill'
    const meta: ChatSkill = { id: `workspace:${rawName}`, name: rawName, scope: 'workspace' }
    const content = String(payload.content ?? '')
    const items = loadMockSkills()
    const existingIdx = items.findIndex((s) => s.meta.id === meta.id)
    if (existingIdx >= 0) items[existingIdx] = { meta, content }
    else items.push({ meta, content })
    saveMockSkills(items)
    return Promise.resolve(meta as T)
  }
  if (cmd === 'remove_chat_skill') {
    saveMockSkills(loadMockSkills().filter((s) => s.meta.id !== payload.id))
    return Promise.resolve(undefined as T)
  }
  if (cmd === 'list_chat_skills') {
    return Promise.resolve(loadMockSkills().map((s) => s.meta) as T)
  }
  if (cmd === 'discover_chat_targets') {
    return Promise.resolve({
      providers: {
        sytra_running: false,
        sytra_models: [],
        ollama_running: false,
        ollama_installed: false,
        ollama_models: [],
        lmstudio_running: false,
        lmstudio_models: [],
      },
      local_models: [],
    } as T)
  }
  if (cmd === 'list_chat_agents') {
    try {
      return Promise.resolve((JSON.parse(localStorage.getItem('sytra-mock-agents-v1') || '[]') as AgentProfile[]) as T)
    } catch { return Promise.resolve([] as T) }
  }
  if (cmd === 'save_chat_agent') {
    const incoming = payload.agent as AgentProfile
    const list: AgentProfile[] = JSON.parse(localStorage.getItem('sytra-mock-agents-v1') || '[]')
    const idx = list.findIndex((a) => a.id === incoming.id)
    if (idx >= 0) list[idx] = incoming
    else list.push(incoming)
    localStorage.setItem('sytra-mock-agents-v1', JSON.stringify(list))
    return Promise.resolve(incoming as T)
  }
  if (cmd === 'delete_chat_agent') {
    const list: AgentProfile[] = JSON.parse(localStorage.getItem('sytra-mock-agents-v1') || '[]')
    localStorage.setItem('sytra-mock-agents-v1', JSON.stringify(list.filter((a) => a.id !== payload.id)))
    return Promise.resolve(undefined as T)
  }
  if (cmd === 'list_agent_memories') {
    try {
      const list: MemoryEntry[] = JSON.parse(localStorage.getItem('sytra-mock-memories-v1') || '[]')
      const aid = payload.agentId as string | null
      return Promise.resolve((aid ? list.filter((m) => m.agent_id === aid || m.agent_id === 'default') : list) as T)
    } catch { return Promise.resolve([] as T) }
  }
  if (cmd === 'save_agent_memory') {
    const list: MemoryEntry[] = JSON.parse(localStorage.getItem('sytra-mock-memories-v1') || '[]')
    const entry: MemoryEntry = {
      id: crypto.randomUUID(),
      agent_id: String(payload.agentId || 'default'),
      session_id: String(payload.sessionId || ''),
      session_title: String(payload.sessionTitle || ''),
      kind: (payload.kind as MemoryEntry['kind']) || 'fact',
      summary: String(payload.summary || ''),
      tags: Array.isArray(payload.tags) ? payload.tags.map(String) : [],
      created_at: Math.floor(Date.now() / 1000),
    }
    list.unshift(entry)
    localStorage.setItem('sytra-mock-memories-v1', JSON.stringify(list.slice(0, 200)))
    return Promise.resolve(entry as T)
  }
  if (cmd === 'recall_agent_memories') {
    const list: MemoryEntry[] = JSON.parse(localStorage.getItem('sytra-mock-memories-v1') || '[]')
    const q = String(payload.query || '').toLowerCase().split(/\W+/).filter((w) => w.length >= 2)
    const aid = payload.agentId as string | null
    const filtered = aid ? list.filter((m) => m.agent_id === aid || m.agent_id === 'default') : list
    const limit = Number(payload.limit || 6)
    const scored = filtered
      .map((m) => {
        const hay = `${m.summary} ${m.session_title} ${m.tags.join(' ')}`.toLowerCase()
        const score = q.reduce((s, tok) => s + (hay.includes(tok) ? 1 : 0), 0)
        return { m, score }
      })
      .sort((a, b) => b.score - a.score || b.m.created_at - a.m.created_at)
      .slice(0, limit)
      .map((x) => x.m)
    return Promise.resolve(scored as T)
  }
  if (cmd === 'delete_agent_memory') {
    const list: MemoryEntry[] = JSON.parse(localStorage.getItem('sytra-mock-memories-v1') || '[]')
    localStorage.setItem('sytra-mock-memories-v1', JSON.stringify(list.filter((m) => m.id !== payload.id)))
    return Promise.resolve(undefined as T)
  }
  if (cmd === 'execute_agent_tool') {
    const key = `sytra-mock-ws-${String(payload.agentId || 'default')}`
    const files: Record<string, string> = JSON.parse(localStorage.getItem(key) || '{}')
    const tool = String(payload.toolName || '')
    const args = (payload.arguments ?? {}) as Record<string, unknown>
    if (tool === 'workspace_write') {
      files[String(args.path || 'note.txt')] = String(args.content || '')
      localStorage.setItem(key, JSON.stringify(files))
      return Promise.resolve({ written: args.path, bytes: String(args.content || '').length } as T)
    }
    if (tool === 'workspace_read') {
      return Promise.resolve({ path: args.path, content: files[String(args.path || '')] ?? '' } as T)
    }
    if (tool === 'workspace_list') {
      return Promise.resolve({ workspace: 'sandbox', entries: Object.keys(files).map((name) => ({ name, is_dir: false, bytes: files[name].length })) } as T)
    }
    return Promise.reject(new Error('Host PC commands require the desktop runtime with permission enabled.'))
  }
  if (cmd === 'search_web_direct') {
    return Promise.resolve({ query: String(payload.query || ''), results: [], error: 'Live web search in browser mock uses MCP proxy.' } as T)
  }
  if (cmd === 'unload_chat_model') {
    return Promise.resolve(`Unloaded ${String(payload.model || payload.provider || 'model')} from GPU/RAM.` as T)
  }
  if (cmd === 'voice_status' || cmd === 'setup_kokoro_voice') {
    return Promise.resolve({
      env_ready: false,
      kokoro_model_ready: false,
      server_running: false,
      installing: cmd === 'setup_kokoro_voice',
      stage: cmd === 'setup_kokoro_voice' ? 'installing_env' : 'idle',
      message: cmd === 'setup_kokoro_voice' ? 'Mock setup running...' : '',
      active_port: null,
      model_dir: 'D:\\.sytra\\voice-models\\kokoro-82m',
      voices: [
        { id: 'auto', label: 'Auto (EN / PT-BR)', lang: 'auto' },
        { id: 'af_heart', label: 'Kokoro Heart (US F)', lang: 'en-us' },
        { id: 'pf_dora', label: 'Kokoro Dora (PT-BR F)', lang: 'pt-br' },
      ],
    } as T)
  }
  if (cmd === 'voice_synthesize') {
    // Valid tiny silent WAV header + 16 samples
    return Promise.resolve({
      audio_base64: 'UklGRjIAAABXQVZFZm10IBAAAAABAAEAQBYAAIA+AAACABAAZGF0YQ4AAAAAAAAAAAAAAAAAAAAAAA==',
      mime: 'audio/wav',
    } as T)
  }
  if (cmd === 'voice_transcribe') {
    return Promise.resolve({ text: 'Voice mock transcript', language: 'en', duration_sec: 1.0 } as T)
  }
  if (cmd === 'stop_voice_engine') {
    return Promise.resolve(undefined as T)
  }
  if (cmd === 'chat_mcp_token') return Promise.reject(new Error('MCP requires the desktop app.'))
  if (cmd === 'prepare_dataset' || cmd === 'get_model_settings' || cmd === 'set_model_settings') return Promise.reject(new Error('This operation requires the desktop app.'))
  const mocks: Record<string, unknown> = {
    get_hardware_info: { backend: 'cuda', vram_mb: 24576, ram_mb: 65536 },
    get_settings: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: false, main_memory_limit_mb: null, effective_main_memory_mb: 65536, detected_ram_mb: 65536, default_context_window: 4096, default_temperature: 0.7, enable_flash_attention: true, kv_cache_quant: 'q8_0', vram_limit_mb: 8192, cpu_kv_cache: false, vram_expert_cache_mb: 4096 },
    set_cache_dir: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: true, main_memory_limit_mb: null, effective_main_memory_mb: 65536, detected_ram_mb: 65536 },
    set_main_memory_limit: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: false, main_memory_limit_mb: 49152, effective_main_memory_mb: 49152, detected_ram_mb: 65536 },
    set_inference_settings: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: false, main_memory_limit_mb: null, effective_main_memory_mb: 65536, detected_ram_mb: 65536, default_context_window: 4096, default_temperature: 0.7, enable_flash_attention: true, kv_cache_quant: 'q8_0', vram_limit_mb: 8192, cpu_kv_cache: false, vram_expert_cache_mb: 4096 },
    get_mcp_status: {
      mcp_binary_path: 'C:\\Users\\demo\\.sytra\\bin\\sytra-mcp.exe',
      mcp_binary_exists: true,
      workspace: 'C:\\Users\\demo\\.sytra',
      runner_ready: true,
      clients: [
        { id: 'claude-desktop', label: 'Claude Desktop', config_path: 'C:\\Users\\demo\\AppData\\Roaming\\Claude\\claude_desktop_config.json', detected: true, configured: false },
        { id: 'cursor', label: 'Cursor', config_path: 'C:\\Users\\demo\\.cursor\\mcp.json', detected: true, configured: false },
        { id: 'codex', label: 'Codex CLI', config_path: 'C:\\Users\\demo\\.codex\\config.toml', detected: false, configured: false },
      ],
      generic_json: '{\n  "mcpServers": {\n    "sytra-studio": {\n      "args": [],\n      "command": "C:\\\\Users\\\\demo\\\\.sytra\\\\bin\\\\sytra-mcp.exe",\n      "env": {\n        "SYTRA_WORKSPACE": "C:\\\\Users\\\\demo\\\\.sytra"\n      }\n    }\n  }\n}',
      vscode_json: '{\n  "servers": {\n    "sytra-studio": {\n      "args": [],\n      "command": "C:\\\\Users\\\\demo\\\\.sytra\\\\bin\\\\sytra-mcp.exe",\n      "env": {\n        "SYTRA_WORKSPACE": "C:\\\\Users\\\\demo\\\\.sytra"\n      },\n      "type": "stdio"\n    }\n  }\n}',
      codex_toml: '[mcp_servers.sytra-studio]\ncommand = "C:\\\\Users\\\\demo\\\\.sytra\\\\bin\\\\sytra-mcp.exe"\nargs = []\n\n[mcp_servers.sytra-studio.env]\nSYTRA_WORKSPACE = "C:\\\\Users\\\\demo\\\\.sytra"\n',
    },
    install_mcp_clients: {
      outcomes: [{ id: 'cursor', label: 'Cursor', config_path: 'C:\\Users\\demo\\.cursor\\mcp.json', action: 'configured' }],
      status: {
        mcp_binary_path: 'C:\\Users\\demo\\.sytra\\bin\\sytra-mcp.exe',
        mcp_binary_exists: true,
        workspace: 'C:\\Users\\demo\\.sytra',
        runner_ready: true,
        clients: [
          { id: 'claude-desktop', label: 'Claude Desktop', config_path: 'C:\\Users\\demo\\AppData\\Roaming\\Claude\\claude_desktop_config.json', detected: true, configured: true },
          { id: 'cursor', label: 'Cursor', config_path: 'C:\\Users\\demo\\.cursor\\mcp.json', detected: true, configured: true },
          { id: 'codex', label: 'Codex CLI', config_path: 'C:\\Users\\demo\\.codex\\config.toml', detected: false, configured: false },
        ],
        generic_json: '{\n  "mcpServers": {\n    "sytra-studio": {\n      "args": [],\n      "command": "C:\\\\Users\\\\demo\\\\.sytra\\\\bin\\\\sytra-mcp.exe"\n    }\n  }\n}',
        vscode_json: '',
        codex_toml: '',
      },
    },
    uninstall_mcp_clients: { outcomes: [], status: { mcp_binary_path: 'C:\\Users\\demo\\.sytra\\bin\\sytra-mcp.exe', mcp_binary_exists: true, workspace: 'C:\\Users\\demo\\.sytra', runner_ready: true, clients: [], generic_json: '', vscode_json: '', codex_toml: '' } },
    list_runs: [
      {
        op_id: 'abc-001', kind: 'train', status: 'done',
        artifact_path: 'runs/adapter-mistral-7b',
        config: { model: 'mistralai/Mistral-7B-v0.1' }, provenance: null,
      },
      {
        op_id: 'abc-002', kind: 'merge', status: 'error',
        artifact_path: 'runs/merged-llama',
        config: { merge_method: 'slerp' }, provenance: null,
      },
    ],
    start_train: 'mock-run-' + Math.random().toString(36).slice(2),
    start_merge: 'mock-merge-' + Math.random().toString(36).slice(2),
    stop_op: undefined,
    start_chat_server: true,
    stop_chat_server: true,
    chat_server_running: false,
    list_chat_skills: [],
    plan_inference: {
      compatible: true,
      backend: 'llama_cpp',
      reasons: ['GPU-first hybrid plan (browser mock)'],
      warnings: [],
      estimates: {
        architecture: 'qwen3moe',
        is_moe: true,
        gpu_layers: 18,
        peak_vram_mb: 9800,
        peak_ram_mb: 4200,
        quantization: 'MOSTLY_Q4_K_M',
      },
      command: ['llama-server', '-m', 'model.gguf', '-ngl', '18'],
    },
    build_moe_index: {
      runtime_manifest: '.sytra-runtime.json',
      experts_indexed: 1024,
      dense_bytes: 1024,
      forward_verified: false,
    },
    generate_moe_oracle: {
      oracle: '.sytra-oracle.json',
      next: 'sytra-engine oracle-check',
    },
    delete_run: undefined,
    guider_recommend: [
      {
        model: { name: 'Mistral 7B', family: 'mistral', param_count: 7, arch: 'mistral', dtype: 'bfloat16', vram_fp16_mb: 14336, license: 'apache-2.0', hf_id: 'mistralai/Mistral-7B-v0.1' },
        adapter: { kind: 'qlora', rank: 16, alpha: 32, dropout: 0.05, target_modules: [], quant_bits: 4 },
        reason: 'Fits in 24 GB VRAM with 4-bit QLoRA — excellent instruction-following at 7B scale',
        fits_vram: true,
      },
      {
        model: { name: 'LLaMA-3 8B', family: 'llama', param_count: 8, arch: 'llama', dtype: 'bfloat16', vram_fp16_mb: 16384, license: 'llama3', hf_id: 'meta-llama/Meta-Llama-3-8B' },
        adapter: { kind: 'lora', rank: 32, alpha: 64, dropout: 0.05, target_modules: [], quant_bits: null },
        reason: 'Full 16-bit LoRA fits at 24 GB — best benchmark performance in the 8B class',
        fits_vram: true,
      },
      {
        model: { name: 'Mistral 22B', family: 'mistral', param_count: 22, arch: 'mistral', dtype: 'bfloat16', vram_fp16_mb: 44032, license: 'apache-2.0', hf_id: 'mistralai/Mistral-22B-v0.1' },
        adapter: { kind: 'qlora', rank: 8, alpha: 16, dropout: 0.05, target_modules: [], quant_bits: 4 },
        reason: 'Requires 4-bit quant — exceeds available VRAM at fp16, but fits with 4-bit NF4',
        fits_vram: false,
      },
    ],
    merge_check: { verdict: 'green', reason: 'Compatible architectures' },
    preview_dataset: [
      ['instruction', 'response'],
      ['What is 2+2?', '4'],
      ['Summarise this text.', 'Summary here.'],
    ],
    publish_run: 'mock-publish-' + Math.random().toString(36).slice(2),
    download_model: { op_id: 'mock-dl-' + Math.random().toString(36).slice(2) },
    cancel_download: true,
    list_local_models: [],
    get_download_status: null,
    list_catalog: [
      { id: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', name: 'Qwen2.5 0.5B Instruct GGUF', size_gb: 0.47, format: 'gguf', tags: ['small', 'fast'], recommended: true, downloadable: true, architecture: 'Qwen2ForCausalLM', alert_level: 'none', alerts: [] },
      { id: 'Qwen/Qwen3.5-9B-Base', name: 'Qwen3.5 9B Base', size_gb: 20, format: 'safetensors', tags: ['multimodal'], recommended: false, downloadable: true, architecture: 'Qwen3_5ForConditionalGeneration', alert_level: 'danger', alerts: [{ level: 'danger', code: 'never_qwen2', message: 'This is Qwen3.5, not Qwen2.', blocks_download: false }] },
      { id: 'unsloth/Kimi-K2.7-Code-GGUF', name: 'Kimi K2.7 Coder (MoE)', size_gb: 295, format: 'gguf', tags: ['coding', 'moe', 'large'], recommended: false, downloadable: true, alert_level: 'danger', alerts: [{ level: 'danger', code: 'exceeds_hybrid_envelope', message: 'Does not fit a GPU-first hybrid plan on this machine.', blocks_download: false }] },
    ],
  }
  return Promise.resolve((mocks[cmd] ?? null) as T)
}
