<script lang="ts">
  import { onDestroy, onMount } from 'svelte'
  import {
    chatStore,
    deleteConversation,
    lastCheckpointTargetIndex,
    newConversation,
    pushCheckpoint,
    recordWorkspaceDelta,
    rollbackLastCheckpoint,
    saveConversations,
    type ChatAttachment,
    type ChatEndpoint,
    type ToolEvent,
  } from './chatStore.svelte'
  type McpPermission = 'ask' | 'auto' | 'off'
  import { readChatStream } from './stream'
  import { buildSkillPrompt } from './skills'
  import { mcpRequest, type IndexedTool } from './mcp'
  import { parseMcpConfigJson } from './mcpConfig'
  import { parseToolArguments } from './toolArguments'
  import { chatServerPreferences } from './sampling'
  import { buildCompactionDigest, buildLightweightSystemPrompt, extractInlineToolCalls } from './promptBuilder'
  import { chatAppearance, isSafeMediaUrl, readImageFileAsDataUrl } from './chatCustomizer.svelte'
  import RichMessage from './RichMessage.svelte'
  import AgentPet, { type PetMood } from './AgentPet.svelte'
  import ModelPickerBar from './ModelPickerBar.svelte'
  import ChatCustomizerModal from './ChatCustomizerModal.svelte'
  import AgentStudioDrawer from './AgentStudioDrawer.svelte'
  import { inferLanguageScope, routeBestModel } from './modelRouter'
  import { chatPower, noteChatActivity, setChatBusyState } from './chatPower.svelte'
  import { buildProactiveBriefingPrompt } from './voiceHelpers'
  import {
    flushStreamingSpeechText,
    interruptSpeech,
    pushStreamingSpeechText,
    refreshKokoroStatus,
    resetStreamingSpeechTracker,
    saveVoicePrefs,
    speakFullText,
    startKokoroSetup,
    startLiveVoiceChat,
    stopLiveVoiceChat,
    stopMicCapture,
    toggleDictation,
    voicePrefs,
    voiceRuntime,
  } from './voiceController.svelte'
  import { hwStore, setTab } from '../../store.svelte'
  import { t } from '../../i18n.svelte'
  import { api, type ChatSkill } from '../../api'
  import type { AgentProfile, ChatTargetsReport, MemoryEntry } from '../../types'

  let prompt = $state('')
  let busy = $state(false)
  let notice = $state('')
  let controller: AbortController | null = null
  let steering = ''
  let skills = $state<ChatSkill[]>([])
  let skillsError = $state('')
  let skillInstallName = $state('')
  let skillInstallContent = $state('')
  let skillBusy = $state(false)
  let skillFileInput: HTMLInputElement
  let query = $state<string | null>(null)
  let triggerStart = 0
  let textarea: HTMLTextAreaElement
  let serverName = $state('')
  let serverCommand = $state('')
  let mcpJsonText = $state('')
  let mcpFileInput: HTMLInputElement
  let mediaFileInput = $state<HTMLInputElement | undefined>(undefined)
  let servers = $state<string[]>([])
  let sandboxEnabled = $state(false)
  let mcpBusy = $state(false)
  let considered = $state<IndexedTool[]>([])
  let approval = $state<{ server: string; name: string; args: string; repaired: boolean; resolve: (approved: boolean) => void } | null>(null)
  let contextWindow = $state<number | undefined>(undefined)
  let serverModelName = $state<string>('')
  let liveTkPerSec = $state<number | undefined>(undefined)
  let reportedPromptTokens = $state<number | undefined>(undefined)

  let pendingAttachments = $state<ChatAttachment[]>([])
  let showCustomizer = $state(false)
  let showAgentStudio = $state(false)
  let showVoicePanel = $state(false)
  let agents = $state<AgentProfile[]>([])
  let memories = $state<MemoryEntry[]>([])
  let isCompacting = $state(false)
  let isToolRunning = $state(false)
  let routineTimer: ReturnType<typeof setInterval> | null = null
  let routineInFlight = false
  let chatDestroyed = false
  let lastProactiveAt = 0
  let pickerRef: { refreshTargets: () => Promise<ChatTargetsReport | null> } | undefined = $state(undefined)
  let latestTargets = $state<ChatTargetsReport | null>(null)
  let routeDecisionNote = $state('')
  const modelLanguage = $derived(inferLanguageScope(serverModelName).scope)
  const activeModelLang = $derived(t(
    modelLanguage === 'english' ? 'chat.langEnglish'
      : modelLanguage === 'multilingual' ? 'chat.langMultilingual' : 'chat.langUnknown',
  ))
  const chatAccent = $derived(chatAppearance.accentHex || 'var(--color-brand)')
  const chatRadius = $derived(chatAppearance.bubbleRadiusPx === null ? 'var(--radius-card)' : `${chatAppearance.bubbleRadiusPx}px`)

  function withRequestTimeout(signal: AbortSignal, timeoutMs = 45_000): AbortSignal {
    const timeoutFn = (AbortSignal as unknown as { timeout?: (ms: number) => AbortSignal }).timeout
    const anyFn = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any
    if (typeof timeoutFn === 'function' && typeof anyFn === 'function') {
      return anyFn([signal, timeoutFn(timeoutMs)])
    }
    return signal
  }

  function cleanBaseUrl(raw?: string): string {
    return (raw || 'http://127.0.0.1:8080').trim().replace(/\/+$/, '').replace(/\/v1$/, '')
  }

  function safeCssBgUrl(url: string): string {
    if (!url || !isSafeMediaUrl(url)) return 'none'
    const sanitized = url.replace(/["'()\\\n\r]/g, '')
    return `url("${sanitized}")`
  }

  async function refreshServerTelemetry(signal?: AbortSignal) {
    const reqSignal = signal ?? new AbortController().signal
    const baseUrl = cleanBaseUrl(current?.endpoint?.baseUrl)
    try {
      const prefs = await chatServerPreferences(reqSignal, fetch, baseUrl)
      if (prefs.contextWindow) contextWindow = prefs.contextWindow
      if (current?.endpoint?.model) serverModelName = current.endpoint.model
      else if (prefs.model) serverModelName = prefs.model
    } catch {
      // Server may not be started yet.
    }
  }
  const current = $derived(chatStore.conversations.find(item => item.id === chatStore.currentId)!)
  const suggestions = $derived(query === null ? [] : skills.filter(skill => skill.name.toLowerCase().includes(query!.toLowerCase())).slice(0, 8))

  let activeWebSearch = $derived(current?.webSearch ?? false)
  let activeMcpPermission = $derived<McpPermission>(current?.mcpPermission ?? 'ask')
  let activeSkillsEnabled = $derived(current?.skillsEnabled ?? true)
  let activeAgent = $derived(agents.find(a => a.id === current?.agentId) ?? null)
  let undoTargetIndex = $derived(lastCheckpointTargetIndex(current))
  let activePetSpecies = $derived(activeAgent?.avatar_kind ?? chatAppearance.petSpecies)
  let activePetCustomUrl = $derived(activeAgent?.avatar_url || chatAppearance.assistantAvatarUrl || null)
  let activePetName = $derived(activeAgent?.name || 'Lil Sytra')

  let usedContextTokens = $derived.by(() => {
    if (reportedPromptTokens !== undefined && reportedPromptTokens > 0) return reportedPromptTokens
    const msgs = current?.messages ?? []
    const chars = msgs.reduce((sum, m) => sum + m.content.length + (m.reasoning?.length ?? 0), 0)
    return chars ? Math.max(1, Math.ceil(chars / 3.8)) : 0
  })

  let lastMessageTkPerSec = $derived.by(() => {
    if (liveTkPerSec !== undefined) return liveTkPerSec
    const msgs = current?.messages ?? []
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'assistant' && msgs[i].metrics?.tkPerSec !== undefined) {
        return msgs[i].metrics?.tkPerSec
      }
    }
    return undefined
  })

  let effectiveContextWindow = $derived.by(() => {
    if (contextWindow && contextWindow > 0) return contextWindow
    const msgs = current?.messages ?? []
    for (let i = msgs.length - 1; i >= 0; i--) {
      const win = msgs[i].metrics?.contextWindow
      if (win && win > 0) return win
    }
    return undefined
  })

  let contextPct = $derived(
    effectiveContextWindow && effectiveContextWindow > 0
      ? Math.min(100, Math.round((usedContextTokens / effectiveContextWindow) * 100))
      : 0,
  )

  let showCompactAlert = $derived(
    contextPct >= 75
    && contextPct < 85
    && (current?.messages.length ?? 0) > 4
    && current?.compactionDismissedAtPct !== contextPct,
  )

  let petMood = $derived<PetMood>(
    isCompacting ? 'compacting'
      : isToolRunning ? 'working'
      : voiceRuntime.speaking ? 'speaking'
      : busy && (current?.messages.at(-1)?.reasoning && !current?.messages.at(-1)?.content) ? 'thinking'
      : busy ? 'speaking'
      : 'idle',
  )

  async function refreshAgents() {
    try {
      agents = (await api.listChatAgents()) ?? []
    } catch {
      agents = []
    }
  }

  async function refreshMemories() {
    try {
      memories = (await api.listAgentMemories(current?.agentId ?? undefined)) ?? []
    } catch {
      memories = []
    }
  }

  function selectEndpoint(ep: ChatEndpoint) {
    if (!current) return
    current.endpoint = ep
    serverModelName = ep.model
    saveConversations()
    void refreshServerTelemetry()
  }

  function selectAgent(agentId: string | undefined) {
    if (!current) return
    current.agentId = agentId
    const chosen = agents.find(a => a.id === agentId)
    if (chosen?.skill_ids?.length) {
      current.skillIds = [...new Set([...(current.skillIds ?? []), ...chosen.skill_ids])].slice(0, 8)
    }
    if (chosen?.provider && chosen?.model_id) {
      current.endpoint = {
        provider: chosen.provider as ChatEndpoint['provider'],
        baseUrl: chosen.endpoint_url || (chosen.provider === 'ollama' ? 'http://127.0.0.1:11434' : chosen.provider === 'lmstudio' ? 'http://127.0.0.1:1234' : 'http://127.0.0.1:8080'),
        model: chosen.model_id,
      }
    }
    saveConversations()
    void refreshMemories()
  }

  async function runCompaction(session: typeof current, autoTriggered: boolean) {
    if (!session || session.messages.length <= 4) return
    isCompacting = true
    try {
      const splitIdx = session.messages.length - 4
      const head = session.messages.slice(0, splitIdx)
      const tail = session.messages.slice(splitIdx)
      const digest = buildCompactionDigest(head, session.lastCompactedSummary)
      session.lastCompactedSummary = digest
      session.compactionCount = (session.compactionCount ?? 0) + 1
      session.compactionDismissedAtPct = undefined
      session.messages = [
        { role: 'assistant', content: `Compacted memory summary (${head.length} earlier turns):\n\n${digest}`, compacted: true },
        ...tail,
      ]
      reportedPromptTokens = Math.max(1, Math.ceil(JSON.stringify(session.messages).length / 3.8))
      saveConversations()
      await api.saveAgentMemory({
        agentId: session.agentId || 'default',
        sessionId: session.id,
        sessionTitle: session.title,
        kind: 'compaction',
        summary: digest,
        tags: [autoTriggered ? 'auto-compact' : 'manual-compact'],
      })
      await refreshMemories()
      notice = t(autoTriggered ? 'chat.compactedAutomatic' : 'chat.compactedManual')
    } catch {
      // Compaction failure stays non-fatal.
    } finally {
      setTimeout(() => { isCompacting = false }, 700)
    }
  }

  async function handleAttachmentFiles(files: FileList | File[] | null) {
    if (!files) return
    for (const file of Array.from(files)) {
      try {
        const { dataUrl, isGif } = await readImageFileAsDataUrl(file)
        pendingAttachments = [
          ...pendingAttachments.slice(0, 5),
          { id: crypto.randomUUID(), name: file.name, mime: file.type, dataUrl, isGif },
        ]
      } catch (err) {
        notice = String(err)
      }
    }
  }

  function handleComposerPaste(event: ClipboardEvent) {
    const items = event.clipboardData?.files
    if (items && items.length > 0) {
      const images = Array.from(items).filter(f => f.type.startsWith('image/'))
      if (images.length) {
        event.preventDefault()
        void handleAttachmentFiles(images)
      }
    }
  }

  function handleComposerDrop(event: DragEvent) {
    const files = event.dataTransfer?.files
    if (files && files.length > 0) {
      const images = Array.from(files).filter(f => f.type.startsWith('image/'))
      if (images.length) {
        event.preventDefault()
        void handleAttachmentFiles(images)
      }
    }
  }

  async function openChatCompletionStream(
    chatUrl: string,
    payload: Record<string, unknown>,
    signal: AbortSignal,
  ): Promise<{ ok: boolean; status: number; body: ReadableStream<Uint8Array> | null; errorText?: string }> {
    try {
      const response = await fetch(chatUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal,
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        return { ok: false, status: response.status, body: null, errorText: (await response.text()).slice(0, 500) }
      }
      return { ok: true, status: response.status, body: response.body }
    } catch (err) {
      if (signal.aborted) throw err
      const isTauriEnv = typeof (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ !== 'undefined'
      if (!isTauriEnv) throw err
      // CORS fallback for Ollama / LM Studio inside desktop WebView2 via Rust reqwest relay.
      const encoder = new TextEncoder()
      let aborted = false
      const onAbort = () => { aborted = true }
      signal.addEventListener('abort', onAbort, { once: true })
      const stream = new ReadableStream<Uint8Array>({
        start(streamCtrl) {
          void api.chatHttpRelayStream(chatUrl, payload, (line) => {
            if (aborted) return
            streamCtrl.enqueue(encoder.encode(line))
          }).then(() => {
            signal.removeEventListener('abort', onAbort)
            try { streamCtrl.close() } catch { /* already closed */ }
          }).catch((relayErr) => {
            signal.removeEventListener('abort', onAbort)
            streamCtrl.error(relayErr instanceof Error ? relayErr : new Error(String(relayErr)))
          })
        },
      })
      return { ok: true, status: 200, body: stream }
    }
  }

  function toggleWebSearch() {
    if (!current) return
    current.webSearch = !activeWebSearch
    saveConversations()
  }

  function setMcpPermission(perm: McpPermission) {
    if (!current) return
    current.mcpPermission = perm
    saveConversations()
  }

  function toggleSkillsEnabled() {
    if (!current) return
    current.skillsEnabled = !activeSkillsEnabled
    saveConversations()
  }

  function handleVoiceTranscript(text: string, autoSend: boolean) {
    const clean = text.trim()
    if (!clean) return
    if (autoSend) {
      prompt = clean
      if (!busy) {
        void send()
      }
      return
    }
    prompt = prompt.trim() ? `${prompt.trim()} ${clean}` : clean
  }

  function toggleVoiceEngine(mode: 'browser' | 'kokoro') {
    voicePrefs.engine = mode
    saveVoicePrefs()
    if (mode === 'kokoro') {
      void refreshKokoroStatus().then(st => {
        if (st && !st.kokoro_model_ready && !st.installing) {
          notice = t('chat.kokoroFallback')
        }
      })
    }
  }

  function toggleProactiveWatch() {
    voicePrefs.proactiveEnabled = !voicePrefs.proactiveEnabled
    saveVoicePrefs()
    if (voicePrefs.proactiveEnabled) {
      notice = t('chat.watchActive')
      if (!busy) {
        void runProactiveBriefing(true)
      }
    } else {
      notice = t('chat.watchPaused')
    }
  }

  async function runProactiveBriefing(manual = false) {
    if (busy || !current) return
    const targets = latestTargets ?? (await pickerRef?.refreshTargets() ?? null)
    const briefingPrompt = buildProactiveBriefingPrompt({
      agent: activeAgent,
      hw: hwStore.info,
      targets,
      memories,
      webSearchEnabled: Boolean(current.webSearch),
      contextPct,
    })
    lastProactiveAt = Math.floor(Date.now() / 1000)
    await send(false, {
      label: `${manual ? 'Briefing' : 'Auto-Watch'} · ${activePetName}`,
      userInstruction: briefingPrompt,
      agentOverride: activeAgent,
    })
  }

  async function runAgentRoutine(agent: AgentProfile) {
    if (busy || routineInFlight || chatDestroyed) return
    const promptText = agent.routine_prompt.trim()
    if (!promptText) return
    routineInFlight = true
    try {
      agent.last_routine_at = Math.floor(Date.now() / 1000)
      await api.saveChatAgent(agent).catch(() => {})
      if (!chatDestroyed) {
        await send(false, {
          label: `${t('chat.backgroundRoutine')} · ${agent.name}`,
          userInstruction: `[Scheduled agent routine]\n${promptText}`,
          agentOverride: agent,
        })
        if (!chatDestroyed) await refreshAgents()
      }
    } finally {
      routineInFlight = false
    }
  }

  onMount(() => {
    api.listChatSkills().then(found => { skills = found }).catch(error => { skillsError = String(error) })
    void refreshServers(false)
    void refreshServerTelemetry()
    void refreshAgents()
    void refreshMemories()
    void refreshKokoroStatus()
    routineTimer = setInterval(() => {
      if (busy || routineInFlight) return
      const nowSec = Math.floor(Date.now() / 1000)
      for (const ag of agents) {
        if (!ag.routine_enabled || !ag.routine_prompt.trim()) continue
        const due = !ag.last_routine_at || (nowSec - ag.last_routine_at) >= ag.routine_interval_mins * 60
        if (due) {
          void runAgentRoutine(ag)
          return
        }
      }
      if (voicePrefs.proactiveEnabled) {
        const dueProactive = !lastProactiveAt || (nowSec - lastProactiveAt) >= voicePrefs.proactiveIntervalMins * 60
        if (dueProactive) {
          void runProactiveBriefing(false)
        }
      }
    }, 30_000)
  })
  onDestroy(() => {
    chatDestroyed = true
    steering = ''
    controller?.abort()
    stopLiveVoiceChat()
    stopMicCapture(false)
    interruptSpeech()
    if (routineTimer) clearInterval(routineTimer)
  })

  function updateSkillQuery() {
    const val = textarea?.value ?? prompt
    const rawCursor = textarea?.selectionStart ?? 0
    const cursor = rawCursor > 0 ? rawCursor : val.length
    const before = val.slice(0, cursor)
    const match = /(?:^|\s)@([\w-]*)$/.exec(before)
    query = match ? match[1] : null
    if (match) triggerStart = cursor - match[1].length - 1
  }

  function selectSkill(skill: ChatSkill) {
    if (!(current.skillIds ?? []).includes(skill.id) && (current.skillIds?.length ?? 0) >= 8) {
      notice = t('chat.skillsLimit')
      return
    }
    const end = textarea.selectionStart
    prompt = `${prompt.slice(0, triggerStart)}@${skill.name} ${prompt.slice(end)}`
    current.skillIds = [...new Set([...(current.skillIds ?? []), skill.id])]
    saveConversations()
    query = null
    textarea.focus()
  }

  function toggleSkill(skill: ChatSkill) {
    const active = current.skillIds ?? []
    if (active.includes(skill.id)) {
      removeSkill(skill.id)
      return
    }
    if (active.length >= 8) {
      notice = t('chat.skillsLimit')
      return
    }
    current.skillIds = [...active, skill.id]
    saveConversations()
  }

  function removeSkill(id: string) {
    current.skillIds = (current.skillIds ?? []).filter(item => item !== id)
    saveConversations()
  }

  async function refreshSkills() {
    try {
      skills = await api.listChatSkills()
      skillsError = ''
    } catch (error) {
      skillsError = String(error)
    }
  }

  async function installSkill(nameArg?: string, contentArg?: string) {
    const name = (nameArg ?? skillInstallName).trim()
    const content = (contentArg ?? skillInstallContent).trim()
    if (!name || !content) {
      notice = t('chat.skillRequired')
      return
    }
    skillBusy = true
    try {
      const created = await api.installChatSkill(name, content)
      await refreshSkills()
      if (!(current.skillIds ?? []).includes(created.id) && (current.skillIds?.length ?? 0) < 8) {
        current.skillIds = [...(current.skillIds ?? []), created.id]
        saveConversations()
      }
      if (!nameArg) skillInstallName = ''
      if (!contentArg) skillInstallContent = ''
      notice = t('chat.skillActivated', { name: created.name })
    } catch (error) {
      notice = t('chat.skillInstallError', { error: error instanceof Error ? error.message : String(error) })
    } finally {
      skillBusy = false
    }
  }

  async function handleSkillFilePicked(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    try {
      const content = await file.text()
      const relParts = (file.webkitRelativePath || '').split('/').filter(Boolean)
      const fallbackName = file.name.toLowerCase() === 'skill.md'
        ? (relParts.length >= 2 ? relParts[relParts.length - 2] : skillInstallName.trim() || 'custom-skill')
        : file.name.replace(/\.[^.]+$/, '')
      const chosen = skillInstallName.trim() || fallbackName
      skillInstallName = chosen
      skillInstallContent = content
      await installSkill(chosen, content)
    } catch (error) {
      notice = t('chat.skillImportError', { error: error instanceof Error ? error.message : String(error) })
    }
  }

  async function removeInstalledSkill(id: string) {
    skillBusy = true
    try {
      await api.removeChatSkill(id)
      removeSkill(id)
      await refreshSkills()
      notice = t('chat.skillRemoved')
    } catch (error) {
      notice = String(error)
    } finally {
      skillBusy = false
    }
  }

  async function refreshServers(report = true) {
    try { const status = await mcpRequest<{ servers: string[]; sandbox: boolean }>('status', {}); servers = status.servers; sandboxEnabled = status.sandbox }
    catch (error) { servers = []; sandboxEnabled = false; if (report) notice = t('chat.error', { error: String(error) }) }
  }

  async function toggleSandbox() {
    mcpBusy = true
    try { await mcpRequest(sandboxEnabled ? 'disable-sandbox' : 'enable-sandbox', {}); await refreshServers() }
    catch (error) { notice = t('chat.error', { error: String(error) }) }
    finally { mcpBusy = false }
  }

  async function connectServer() {
    mcpBusy = true
    try {
      const argv: unknown = JSON.parse(serverCommand)
      if (!Array.isArray(argv) || !argv.length || !argv.every(arg => typeof arg === 'string' && arg.length > 0)) throw new Error(t('chat.invalidCommand'))
      await mcpRequest('connect', { server: serverName.trim(), argv })
      await refreshServers()
      notice = t('chat.serverConnected')
    } catch (error) { notice = t('chat.error', { error: String(error) }) }
    finally { mcpBusy = false }
  }

  async function applyMcpJson() {
    mcpBusy = true
    try {
      const entries = parseMcpConfigJson(mcpJsonText)
      for (const entry of entries) {
        await mcpRequest('connect', entry)
      }
      await refreshServers()
      notice = t('chat.mcpLoaded', { n: entries.length, names: entries.map((entry) => entry.server).join(', ') })
    } catch (error) {
      notice = t('chat.mcpJsonError', { error: error instanceof Error ? error.message : String(error) })
    } finally {
      mcpBusy = false
    }
  }

  async function handleMcpJsonFilePicked(event: Event) {
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    try {
      mcpJsonText = await file.text()
      notice = t('chat.mcpFileLoaded', { name: file.name, action: t('chat.applyJson') })
    } catch (error) {
      notice = t('chat.readJsonError', { error: error instanceof Error ? error.message : String(error) })
    }
  }

  async function disconnectServer(server: string) {
    mcpBusy = true
    try { await mcpRequest('disconnect', { server }); await refreshServers() }
    catch (error) { notice = t('chat.error', { error: String(error) }) }
    finally { mcpBusy = false }
  }

  function approveTool(server: string, name: string, args: Record<string, unknown>, repaired: boolean, signal: AbortSignal): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const cancel = () => { approval = null; reject(new DOMException('Aborted', 'AbortError')) }
      if (signal.aborted) { cancel(); return }
      signal.addEventListener('abort', cancel, { once: true })
      approval = { server, name, args: JSON.stringify(args, null, 2), repaired, resolve: value => {
        signal.removeEventListener('abort', cancel)
        approval = null
        resolve(value)
      } }
    })
  }

  async function handleUndoLastCheckpoint() {
    if (!current) return
    if (busy) controller?.abort()
    interruptSpeech()
    const undone = rollbackLastCheckpoint(current)
    if (!undone) return
    for (const delta of undone.workspaceDeltas) {
      try {
        if (delta.existed && delta.previousContent !== undefined) {
          await api.executeAgentTool(delta.agentId, 'workspace_write', {
            path: delta.path,
            content: delta.previousContent,
          })
        } else {
          await api.executeAgentTool(delta.agentId, 'workspace_delete', { path: delta.path })
        }
      } catch {
        // Non-fatal if sandbox file or agent was deleted independently.
      }
    }
    if (undone.promptText && !prompt.trim() && !undone.promptText.includes('Proactive Autonomous Update')) {
      prompt = undone.promptText
    }
    reportedPromptTokens = undefined
    liveTkPerSec = undefined
    saveConversations()
    notice = t('chat.reverted')
  }

  async function send(
    resume = false,
    proactiveOpts?: { label: string; userInstruction: string; agentOverride?: AgentProfile | null },
  ) {
    const rawText = prompt.trim()
    const hasPendingMedia = !resume && !proactiveOpts && pendingAttachments.length > 0
    if (
      busy
      || (proactiveOpts
        ? !proactiveOpts.userInstruction.trim()
        : (resume ? current.messages.at(-1)?.role !== 'user' : (!rawText && !hasPendingMedia)))
    ) return
    const session = current
    const text = proactiveOpts
      ? proactiveOpts.userInstruction.trim()
      : (rawText || (hasPendingMedia ? 'Analyze the attached image(s).' : ''))

    if (!resume && !proactiveOpts) {
      pushCheckpoint(session, text)
      const attachCopy = pendingAttachments.length ? pendingAttachments.map(a => ({ ...a })) : undefined
      const previous = session.messages.at(-1)
      if (previous?.role === 'user' && !attachCopy) {
        previous.content += `\n\n${text}`
        previous.skillIds = [...(session.skillIds ?? [])]
      } else {
        session.messages.push({
          role: 'user',
          content: text,
          skillIds: [...(session.skillIds ?? [])],
          ...(attachCopy ? { attachments: attachCopy } : {}),
        })
      }
      if (session.messages.length <= 2 && session.title === t('chat.newTitle')) {
        session.title = text.slice(0, 48)
      }
      prompt = ''
      pendingAttachments = []
    } else if (proactiveOpts) {
      pushCheckpoint(session, proactiveOpts.userInstruction)
    }

    if (effectiveContextWindow && effectiveContextWindow > 0 && session.messages.length > 4) {
      const estChars = session.messages.reduce((s, m) => s + m.content.length, 0)
      const estPct = Math.round((Math.ceil(estChars / 3.8) / effectiveContextWindow) * 100)
      if (estPct >= 85) {
        await runCompaction(session, true)
      }
    }

    notice = ''
    considered = []
    liveTkPerSec = undefined
    resetStreamingSpeechTracker()

    const serializeMessages = (allowImages: boolean): Record<string, unknown>[] => {
      const out: Record<string, unknown>[] = []
      for (const msg of session.messages) {
        if (!msg.content.trim() && !msg.attachments?.length) continue
        if (msg.role === 'user' && allowImages && msg.attachments?.length) {
          out.push({
            role: 'user',
            content: [
              { type: 'text', text: msg.content || 'Inspect the attached image(s).' },
              ...msg.attachments.map(a => ({ type: 'image_url', image_url: { url: a.dataUrl } })),
            ],
          })
          continue
        }
        const textBody = msg.attachments?.length
          ? `${msg.content}\n[Attached image/GIF files: ${msg.attachments.map(a => a.name).join(', ')}]`
          : msg.content
        const previous = out.at(-1)
        if (msg.role === 'user' && previous?.role === 'user' && typeof previous.content === 'string') {
          previous.content = `${previous.content}\n\n${textBody}`
        } else {
          out.push({ role: msg.role, content: textBody })
        }
      }
      if (proactiveOpts) {
        out.push({ role: 'user', content: proactiveOpts.userInstruction })
      }
      return out
    }

    let allowVisionPayload = true
    let messages = serializeMessages(allowVisionPayload)
    const index = session.messages.length
    session.messages.push({
      role: 'assistant',
      content: '',
      ...(proactiveOpts ? { proactiveLabel: proactiveOpts.label } : {}),
    })
    busy = true
    isToolRunning = false
    const abort = new AbortController()
    controller = abort

    try {
      const skillsOn = session.skillsEnabled ?? true
      const agent = proactiveOpts?.agentOverride ?? agents.find(a => a.id === session.agentId) ?? null
      const activeSkills = skillsOn
        ? [...new Set([...(session.skillIds ?? []), ...(agent?.skill_ids ?? [])])].slice(0, 8)
        : []
      let skillsPrompt = ''
      let skillsUsed: string[] = []
      if (activeSkills.length) {
        if (activeSkills.length > 8) throw new Error(t('chat.skillsLimit'))
        const contents = await Promise.all(activeSkills.map(id => api.readChatSkill(id)))
        skillsPrompt = buildSkillPrompt(activeSkills, contents)
        skillsUsed = activeSkills.map(id => skills.find(s => s.id === id)?.name ?? id)
      }
      if (abort.signal.aborted) throw new DOMException('Aborted', 'AbortError')

      const skillLoadedOk = activeSkills.length > 0
      void skillLoadedOk
      setChatBusyState(true)

      let endpointProvider = session.endpoint?.provider ?? 'sytra'
      let endpointBase = cleanBaseUrl(session.endpoint?.baseUrl)
      let model = session.endpoint?.provider === 'router' ? '' : (session.endpoint?.model || '')

      if (session.endpoint?.provider === 'router') {
        const targets = latestTargets ?? (await pickerRef?.refreshTargets() ?? null)
        const routed = routeBestModel({
          prompt: text,
          hasImages: hasPendingMedia || session.messages.some(m => Boolean(m.attachments?.length)),
          agentRole: agent?.role,
          agentPrompt: agent?.system_prompt,
          targets,
          vramMb: hwStore.info?.vram_mb,
          fallbackEndpoint: chatPower.activeTarget ?? undefined,
          translateReason: t,
        })
        if (!routed) {
          throw new Error(t('chat.noRouteModel'))
        }
        if (routed.localFilePath && targets && !targets.providers.sytra_running) {
          await api.startChatServer(routed.localFilePath)
          await pickerRef?.refreshTargets()
        } else if (routed.endpoint.provider === 'ollama' && targets && !targets.providers.ollama_running) {
          await api.startOllamaDaemon()
          await pickerRef?.refreshTargets()
        }
        endpointProvider = routed.endpoint.provider
        endpointBase = cleanBaseUrl(routed.endpoint.baseUrl)
        model = routed.endpoint.model
        routeDecisionNote = t('chat.routeDecision', { model, reason: routed.reason })
      } else {
        routeDecisionNote = ''
      }

      const perm = session.mcpPermission ?? 'ask'
      const searchSeed = text || session.messages[index - 1]?.content || ''
      const wantsMemoryRecall = memories.length > 0 && Boolean(
        proactiveOpts
        || /\b(remember|recal|memor|session|previous|history|earlier|last time|passad|lembr|mem[oó]ri|sess[aã]o|hist[oó]ri|antes)\b/i.test(text),
      )
      const wantsMemorySave = Boolean(agent) && Boolean(
        proactiveOpts
        || /\b(remember|save|memor|note|prefer|lembr|salv|guard|anot|prefer)\b/i.test(text),
      )
      const wantsSkillCreate = perm !== 'off' && Boolean(
        proactiveOpts
        || /\b(skill|skills|skill\.md|habilidad|rotina|workflow|instru|create|criar|salvar|automat)\b/i.test(text),
      )
      const wantsWorkspaceTools = Boolean(agent) && Boolean(
        proactiveOpts
        || /\b(file|dir|folder|workspace|pc|computer|desktop|command|cmd|shell|script|log|code|repo|read|write|list|arquiv|past|computador|coman|ler|escrev|list)\b/i.test(text),
      )

      const [prefs, recalled, foundMcp] = await Promise.all([
        chatServerPreferences(abort.signal, fetch, endpointBase, {
          provider: endpointProvider,
          knownModel: model || undefined,
        }).catch(() => ({} as { temperature?: number; contextWindow?: number; model?: string })),
        memories.length > 0
          ? api.recallAgentMemories(searchSeed, agent?.id, 5).catch(() => [])
          : Promise.resolve([]),
        perm !== 'off' && (servers.length || sandboxEnabled)
          ? mcpRequest<{ tools: IndexedTool[] }>('search', { query: searchSeed }, abort.signal).catch(() => ({ tools: [] as IndexedTool[] }))
          : Promise.resolve({ tools: [] as IndexedTool[] }),
      ])

      if (!model) {
        model = prefs.model || ''
      }
      if (!model) {
        const modelsResponse = await fetch(`${endpointBase}/v1/models`, {
          signal: withRequestTimeout(abort.signal, 20_000),
        })
        if (!modelsResponse.ok) throw new Error(`Could not list local models (${modelsResponse.status}).`)
        const models = await modelsResponse.json()
        model = models?.data?.[0]?.id || ''
      }
      if (typeof model !== 'string' || !model) {
        throw new Error('Select a model in the Chat Model Picker or start a server in Models.')
      }

      noteChatActivity({
        provider: endpointProvider === 'router' ? 'sytra' : endpointProvider,
        baseUrl: endpointBase,
        model,
      })
      serverModelName = model
      if (prefs.contextWindow) contextWindow = prefs.contextWindow
      const temperature = prefs.temperature

      const exposedTools: { exposedName: string; item: IndexedTool }[] = []

      if (session.webSearch) {
        exposedTools.push({
          exposedName: 'web_search',
          item: {
            server: 'web',
            tool: {
              name: 'web_search',
              description: 'Search the live web for current facts, documentation, or news.',
              inputSchema: {
                type: 'object',
                properties: { query: { type: 'string', description: 'Search query' } },
                required: ['query'],
              },
            },
          },
        })
      }

      if (wantsMemoryRecall) {
        exposedTools.push({
          exposedName: 'recall_memory',
          item: {
            server: 'memory',
            tool: {
              name: 'recall_memory',
              description: 'Recall facts or compacted summaries from past sessions.',
              inputSchema: {
                type: 'object',
                properties: { query: { type: 'string', description: 'Search topic or keywords' } },
                required: ['query'],
              },
            },
          },
        })
      }
      if (wantsMemorySave) {
        exposedTools.push({
          exposedName: 'save_memory',
          item: {
            server: 'memory',
            tool: {
              name: 'save_memory',
              description: 'Persist an important fact or preference across future sessions.',
              inputSchema: {
                type: 'object',
                properties: {
                  summary: { type: 'string' },
                  tags: { type: 'array', items: { type: 'string' } },
                },
                required: ['summary'],
              },
            },
          },
        })
      }

      if (wantsSkillCreate) {
        exposedTools.push({
          exposedName: 'create_skill',
          item: {
            server: 'skills',
            tool: {
              name: 'create_skill',
              description: 'Save a reusable workflow as a local SKILL.md instruction file.',
              inputSchema: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  instructions: { type: 'string' },
                },
                required: ['name', 'instructions'],
              },
            },
          },
        })
      }

      if (wantsWorkspaceTools && agent) {
        exposedTools.push(
          {
            exposedName: 'workspace_list',
            item: {
              server: 'agent-ws',
              tool: {
                name: 'workspace_list',
                description: 'List files inside this agent workspace.',
                inputSchema: { type: 'object', properties: { path: { type: 'string' } } },
              },
            },
          },
          {
            exposedName: 'workspace_read',
            item: {
              server: 'agent-ws',
              tool: {
                name: 'workspace_read',
                description: 'Read a text file inside this agent workspace.',
                inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
              },
            },
          },
          {
            exposedName: 'workspace_write',
            item: {
              server: 'agent-ws',
              tool: {
                name: 'workspace_write',
                description: 'Write a text file inside this agent workspace.',
                inputSchema: {
                  type: 'object',
                  properties: { path: { type: 'string' }, content: { type: 'string' } },
                  required: ['path', 'content'],
                },
              },
            },
          },
        )

        if (agent.computer_mode !== 'sandbox') {
          exposedTools.push(
            {
              exposedName: 'pc_list_dir',
              item: {
                server: 'host-pc',
                tool: {
                  name: 'pc_list_dir',
                  description: 'List files in a folder on the host PC (requires permission).',
                  inputSchema: { type: 'object', properties: { path: { type: 'string' } } },
                },
              },
            },
            {
              exposedName: 'pc_read_file',
              item: {
                server: 'host-pc',
                tool: {
                  name: 'pc_read_file',
                  description: 'Read a text file on the host PC (requires permission).',
                  inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
                },
              },
            },
            {
              exposedName: 'pc_run_command',
              item: {
                server: 'host-pc',
                tool: {
                  name: 'pc_run_command',
                  description: 'Run a shell command on the host PC (requires permission).',
                  inputSchema: {
                    type: 'object',
                    properties: { command: { type: 'string' }, cwd: { type: 'string' } },
                    required: ['command'],
                  },
                },
              },
            },
          )
        }
      }

      if (foundMcp.tools.length) {
        considered = foundMcp.tools
        considered.forEach((item, i) => {
          exposedTools.push({ exposedName: `mcp_${i}`, item })
        })
      }

      const systemPrompt = buildLightweightSystemPrompt({
        skillsPrompt,
        skillsCatalog: skills.map(s => s.name),
        mcpServers: servers,
        mcpPermission: perm,
        tools: exposedTools,
        agentPrompt: agent?.system_prompt,
        computerMode: agent?.computer_mode,
        recalledMemories: recalled ?? [],
      })
      if (systemPrompt) messages.unshift({ role: 'system', content: systemPrompt })

      const promptChars = JSON.stringify(messages).length
      reportedPromptTokens = Math.max(1, Math.ceil(promptChars / 3.8))
      if (skillsUsed.length) session.messages[index].skillsUsed = skillsUsed

      const openaiTools = exposedTools.map(({ exposedName, item }) => ({
        type: 'function',
        function: {
          name: exposedName,
          description: item.tool.description || item.tool.name,
          parameters: item.tool.inputSchema ?? { type: 'object', properties: {} },
        },
      }))

      let nativeToolsUnsupported = false
      let prevCallSig = ''
      let duplicateRounds = 0

      for (let round = 0; round < 6; round++) {
        const chatUrl = `${endpointBase}/v1/chat/completions`
        const buildPayload = (withNativeTools: boolean) => ({
          model,
          messages,
          stream: true,
          ...(temperature !== undefined ? { temperature } : {}),
          ...(withNativeTools && openaiTools.length
            ? { tools: openaiTools, stream_options: { include_usage: true } }
            : {}),
        })

        let streamRes = await openChatCompletionStream(
          chatUrl,
          buildPayload(!nativeToolsUnsupported),
          withRequestTimeout(abort.signal, 360_000),
        )
        if (!streamRes.ok && (streamRes.status === 400 || streamRes.status === 422)) {
          if (allowVisionPayload && session.messages.some(m => m.attachments?.length)) {
            allowVisionPayload = false
            messages = serializeMessages(false)
            if (systemPrompt) messages.unshift({ role: 'system', content: systemPrompt })
            streamRes = await openChatCompletionStream(
              chatUrl,
              buildPayload(!nativeToolsUnsupported),
              withRequestTimeout(abort.signal, 360_000),
            )
          }
          if (!streamRes.ok && !nativeToolsUnsupported && openaiTools.length && (streamRes.status === 400 || streamRes.status === 422)) {
            nativeToolsUnsupported = true
            streamRes = await openChatCompletionStream(
              chatUrl,
              buildPayload(false),
              withRequestTimeout(abort.signal, 360_000),
            )
          }
        }
        if (!streamRes.ok) {
          throw new Error(`Server returned ${streamRes.status}: ${streamRes.errorText || ''}`)
        }
        if (!streamRes.body) throw new Error('The server did not provide a streaming response.')

        const basePrefix = session.messages[index].content
        let roundRawText = ''
        let nativeCalls: Awaited<ReturnType<typeof readChatStream>> = []

        try {
          nativeCalls = await readChatStream(streamRes.body, {
            firstTokenTimeoutMs: 360_000,
            idleTimeoutMs: 120_000,
            onText: (chunk) => {
              roundRawText += chunk
              const mightHaveTool = exposedTools.length > 0 && (roundRawText.includes('<tool_call') || roundRawText.includes('```'))
              const cleaned = mightHaveTool
                ? extractInlineToolCalls(roundRawText, exposedTools).cleanText.replace(/<tool_call>[\s\S]*$/i, '').trimEnd()
                : roundRawText
              session.messages[index].content = basePrefix
                ? (cleaned ? `${basePrefix}\n\n${cleaned}` : basePrefix)
                : cleaned
              pushStreamingSpeechText(session.messages[index].content)
            },
            onReasoning: (chunk) => {
              session.messages[index].reasoning = `${session.messages[index].reasoning ?? ''}${chunk}`
            },
            onMetrics: (m) => {
              if (m.tkPerSec !== undefined) liveTkPerSec = m.tkPerSec
              if (m.promptTokens !== undefined) reportedPromptTokens = m.promptTokens
              const existing = session.messages[index].metrics ?? {}
              session.messages[index].metrics = {
                ...existing,
                ...(m.tkPerSec !== undefined ? { tkPerSec: m.tkPerSec } : {}),
                ...(m.completionTokens !== undefined ? { completionTokens: m.completionTokens } : {}),
                ...(m.promptTokens !== undefined ? { promptTokens: m.promptTokens } : {}),
                ...(contextWindow ? { contextWindow } : {}),
              }
            },
          })
        } catch (streamErr) {
          const msgStr = String(streamErr)
          if (!nativeToolsUnsupported && openaiTools.length && /HTTP (?:400|422)/i.test(msgStr)) {
            nativeToolsUnsupported = true
            const retryRes = await openChatCompletionStream(
              chatUrl,
              buildPayload(false),
              withRequestTimeout(abort.signal, 360_000),
            )
            if (!retryRes.ok || !retryRes.body) throw streamErr
            nativeCalls = await readChatStream(retryRes.body, {
              firstTokenTimeoutMs: 360_000,
              idleTimeoutMs: 120_000,
              onText: (chunk) => {
                roundRawText += chunk
                const cleaned = extractInlineToolCalls(roundRawText, exposedTools).cleanText.replace(/<tool_call>[\s\S]*$/i, '').trimEnd()
                session.messages[index].content = basePrefix ? (cleaned ? `${basePrefix}\n\n${cleaned}` : basePrefix) : cleaned
              },
              onReasoning: (chunk) => {
                session.messages[index].reasoning = `${session.messages[index].reasoning ?? ''}${chunk}`
              },
            })
          } else {
            throw streamErr
          }
        }

        let usedInlineFallback = false
        let calls = nativeCalls
        if (!calls.length && exposedTools.length) {
          const inline = extractInlineToolCalls(roundRawText, exposedTools)
          if (inline.calls.length) {
            calls = inline.calls
            usedInlineFallback = true
            session.messages[index].content = basePrefix
              ? (inline.cleanText ? `${basePrefix}\n\n${inline.cleanText}` : basePrefix)
              : inline.cleanText
          }
        }

        if (!calls.length) break
        if (round === 5) throw new Error(t('chat.toolLimit'))

        const callSig = calls.map(c => `${c.function.name}:${c.function.arguments}`).join('|')
        if (callSig === prevCallSig) {
          duplicateRounds += 1
          if (duplicateRounds >= 2) {
            notice = t('chat.loopStopped')
            break
          }
        } else {
          prevCallSig = callSig
          duplicateRounds = 0
        }

        if (usedInlineFallback) {
          messages.push({ role: 'assistant', content: roundRawText })
        } else {
          messages.push({ role: 'assistant', content: roundRawText || null, tool_calls: calls })
        }

        isToolRunning = true
        for (const call of calls) {
          const entry = exposedTools.find(e => e.exposedName === call.function.name || e.item.tool.name === call.function.name)
          if (!entry) throw new Error(t('chat.unknownTool'))
          const item = entry.item
          const parsed = parseToolArguments(call.function.arguments)

          let approved = false
          if (item.server === 'web') {
            approved = Boolean(session.webSearch)
          } else if (item.server === 'memory' || item.server === 'agent-ws') {
            approved = true
          } else if (item.server === 'host-pc') {
            approved = agent?.computer_mode === 'pc_auto'
              ? true
              : await approveTool('host-pc', item.tool.name, parsed.value, parsed.repaired, abort.signal)
          } else if (perm === 'auto') {
            approved = true
          } else if (perm === 'ask') {
            approved = await approveTool(item.server, item.tool.name, parsed.value, parsed.repaired, abort.signal)
          }

          let result: unknown = { error: perm === 'off' ? 'Tools are disabled by user settings.' : 'User declined this tool call.' }
          if (approved) {
            if (item.server === 'web' && item.tool.name === 'web_search') {
              const q = String(parsed.value.query ?? parsed.value.q ?? searchSeed).trim()
              if (endpointBase === 'http://127.0.0.1:8080') {
                try {
                  result = await mcpRequest('web-search', { query: q }, abort.signal)
                } catch {
                  result = await api.searchWebDirect(q)
                }
              } else {
                const direct = await api.searchWebDirect(q)
                result = direct ?? await mcpRequest('web-search', { query: q }, abort.signal)
              }
            } else if (item.server === 'memory') {
              if (item.tool.name === 'recall_memory') {
                result = await api.recallAgentMemories(String(parsed.value.query ?? searchSeed), agent?.id, 6)
              } else {
                const summary = String(parsed.value.summary ?? '').trim()
                const tags = Array.isArray(parsed.value.tags) ? parsed.value.tags.map(String) : []
                result = await api.saveAgentMemory({
                  agentId: agent?.id || 'default',
                  sessionId: session.id,
                  sessionTitle: session.title,
                  kind: 'fact',
                  summary,
                  tags,
                })
                await refreshMemories()
              }
            } else if (item.server === 'skills' && item.tool.name === 'create_skill') {
              const skName = String(parsed.value.name ?? 'agent-skill')
              const skBody = String(parsed.value.instructions ?? '')
              const created = await api.installChatSkill(skName, `# ${skName}\n\n${skBody}\n`)
              await refreshSkills()
              if (!(session.skillIds ?? []).includes(created.id) && (session.skillIds?.length ?? 0) < 8) {
                session.skillIds = [...(session.skillIds ?? []), created.id]
              }
              result = { installed: created }
            } else if ((item.server === 'agent-ws' || item.server === 'host-pc') && agent) {
              if (item.server === 'agent-ws' && item.tool.name === 'workspace_write' && typeof parsed.value.path === 'string') {
                const filePath = String(parsed.value.path)
                try {
                  const prev = await api.executeAgentTool(agent.id, 'workspace_read', { path: filePath }) as { content?: string }
                  recordWorkspaceDelta(session, { agentId: agent.id, path: filePath, existed: true, previousContent: String(prev?.content ?? '') })
                } catch {
                  recordWorkspaceDelta(session, { agentId: agent.id, path: filePath, existed: false })
                }
              }
              result = await api.executeAgentTool(agent.id, item.tool.name, parsed.value)
            } else {
              const requestId = crypto.randomUUID()
              const cancel = () => { void mcpRequest('cancel', { request_id: requestId }).catch(() => {}) }
              abort.signal.addEventListener('abort', cancel, { once: true })
              try { result = await mcpRequest('call', { server: item.server, name: item.tool.name, arguments: parsed.value, request_id: requestId }, abort.signal) }
              finally { abort.signal.removeEventListener('abort', cancel); void refreshServers(false) }
            }
          }

          const resultJson = JSON.stringify(result)
          const assistant = session.messages[index]
          const eventRecord: ToolEvent = {
            server: item.server,
            name: item.tool.name,
            approved,
            args: JSON.stringify(parsed.value).slice(0, 140),
            resultPreview: resultJson.slice(0, 220),
          }
          assistant.toolEvents = [...(assistant.toolEvents ?? []), eventRecord]

          if (usedInlineFallback) {
            messages.push({
              role: 'user',
              content: `[Tool result for ${item.tool.name}]:\n${resultJson}\nAnswer concisely in the same language as the user.`,
            })
          } else {
            messages.push({ role: 'tool', tool_call_id: call.id, content: resultJson })
          }
        }
        isToolRunning = false
      }
    } catch (error) {
      notice = abort.signal.aborted ? t('chat.stopped') : t('chat.error', { error: String(error) })
    } finally {
      isToolRunning = false
      setChatBusyState(false)
      const msg = session.messages[index]
      if (msg && !msg.content && !msg.reasoning && !msg.toolEvents?.length && !msg.skillsUsed?.length) {
        session.messages.splice(index, 1)
      } else if (msg?.content) {
        flushStreamingSpeechText()
        if (proactiveOpts) {
          await api.saveAgentMemory({
            agentId: (proactiveOpts.agentOverride?.id ?? session.agentId) || 'default',
            sessionId: session.id,
            sessionTitle: proactiveOpts.label,
            kind: 'routine',
            summary: msg.content.trim().slice(0, 1200),
            tags: ['proactive', 'autonomous'],
          }).catch(() => {})
          await refreshMemories()
        }
      }
      if (effectiveContextWindow && effectiveContextWindow > 0 && contextPct >= 85 && session.messages.length > 4) {
        await runCompaction(session, true)
      }
      try { saveConversations() } catch { notice = t('chat.saveError') }
      if (controller === abort) controller = null
      liveTkPerSec = undefined
      busy = false
      if (steering) {
        prompt = steering
        steering = ''
        void send()
      }
    }
  }
</script>

<div class="chat-layout card">
  <aside class="sessions" aria-label={t('chat.conversations')}>
    <div class="sessions-head"><h2>{t('chat.conversations')}</h2><button class="btn btn-secondary btn-sm" onclick={newConversation} disabled={busy}>{t('chat.new')}</button></div>
    {#each chatStore.conversations as session (session.id)}
      {@const title = session.messages.length ? session.title : t('chat.newTitle')}
      <div class="session-row" class:selected={session.id === current.id}>
        <button class="session-select" onclick={() => { chatStore.currentId = session.id; notice = ''; reportedPromptTokens = undefined }} disabled={busy} aria-current={session.id === current.id ? 'true' : undefined}>{title}</button>
        <button class="delete" onclick={() => deleteConversation(session.id)} disabled={busy} aria-label={t('chat.delete', { title })}>×</button>
      </div>
    {/each}
  </aside>
  <section
    class="conversation"
    aria-label={t('chat.localChat')}
    style:--chat-accent={chatAccent}
    style:--chat-radius={chatRadius}
    style:--chat-max-w={`${chatAppearance.maxWidthPx}px`}
    style:--chat-font-size={`${chatAppearance.fontSizePx}px`}
  >
    <header>
      <div class="header-top">
        <div class="header-brand">
          <div class="title-row">
            <h1>{t('chat.title')}</h1>
          </div>
          <p>{t('chat.intro')}</p>
        </div>
        <div class="telemetry-bar" aria-label={t('chat.telemetry')}>
          {#if serverModelName}
            <span class="telemetry-pill model-pill" title={serverModelName}>
              <i class="bi bi-cpu" aria-hidden="true"></i>
              <span>{serverModelName}</span>
            </span>
            <span class="telemetry-pill" data-testid="chat-lang-pill" title={t('chat.languageSupport', { language: activeModelLang })}>
              <i class="bi bi-translate" aria-hidden="true"></i>
              <span>{activeModelLang}</span>
            </span>
          {/if}
          {#if routeDecisionNote}
            <span class="telemetry-pill" data-testid="chat-route-pill" title={routeDecisionNote}>
              <i class="bi bi-signpost-split" aria-hidden="true"></i>
              <span>{routeDecisionNote}</span>
            </span>
          {/if}
          {#if chatPower.sleeping}
            <span class="telemetry-pill sleep-pill" data-testid="chat-sleep-pill">
              <i class="bi bi-moon-stars" aria-hidden="true"></i>
              <span>{t('chat.gpuIdle')}</span>
            </span>
          {/if}
          <span class="telemetry-pill" data-testid="chat-context-pill">
            <i class="bi bi-layers" aria-hidden="true"></i>
            <span>
              {t('chat.contextLabel')}:
              {#if effectiveContextWindow}
                {usedContextTokens.toLocaleString()} / {effectiveContextWindow.toLocaleString()} tok ({contextPct}%)
              {:else}
                ~{usedContextTokens.toLocaleString()} tok
              {/if}
            </span>
            {#if effectiveContextWindow}
              <span class="ctx-bar" aria-hidden="true"><span class="ctx-fill" style={`width:${contextPct}%`}></span></span>
            {/if}
          </span>
          <span class="telemetry-pill" data-testid="chat-speed-pill">
            <i class="bi bi-speedometer2" aria-hidden="true"></i>
            <span>{lastMessageTkPerSec !== undefined ? `${lastMessageTkPerSec.toFixed(1)} tk/s` : '— tk/s'}</span>
          </span>
          <div class="telemetry-actions">
            <button
              type="button"
              id="btn-open-agents"
              class="telemetry-pill customize-pill"
              class:active={showAgentStudio}
              onclick={() => { showAgentStudio = !showAgentStudio; if (showAgentStudio) showCustomizer = false }}
              title={t('chat.agentsHint')}
            >
              <i class="bi bi-robot" aria-hidden="true"></i>
              <span>{activeAgent ? activeAgent.name : t('chat.agents')}</span>
            </button>
            <button
              type="button"
              id="btn-open-customizer"
              class="telemetry-pill customize-pill"
              class:active={showCustomizer}
              onclick={() => { showCustomizer = !showCustomizer; if (showCustomizer) showAgentStudio = false }}
              title={t('chat.customizeHint')}
            >
              <i class="bi bi-palette2" aria-hidden="true"></i>
              <span>{t('chat.customize')}</span>
            </button>
          </div>
        </div>
      </div>

      <ModelPickerBar
        bind:this={pickerRef}
        selected={current?.endpoint}
        onSelect={selectEndpoint}
        onContextDetected={(ctx) => { if (ctx) contextWindow = ctx }}
        onReportUpdated={(rep) => { latestTargets = rep }}
        onUnloadNotice={(msg) => { notice = msg }}
      />

      <div class="chat-tools-bar">
        <details class="mcp-settings">
          <summary>{t('chat.mcpServers')} ({servers.length}) · mcp_config.json</summary>
          <p>{t('chat.mcpHint')}</p>
          <div class="panel-block">
            <div class="panel-block-head">
              <label class="field-label" for="mcp-json-input">{t('chat.importMcp')}</label>
              <div style="display:flex;gap:6px;flex-wrap:wrap">
                <input bind:this={mcpFileInput} type="file" accept=".json,application/json" style="display:none" onchange={handleMcpJsonFilePicked} />
                <button type="button" id="btn-import-mcp-json" class="btn btn-secondary btn-sm" disabled={busy || mcpBusy} onclick={() => mcpFileInput.click()}>{t('chat.loadJson')}</button>
                <button type="button" class="btn btn-secondary btn-sm" onclick={() => setTab('settings')}>{t('chat.globalMcp')}</button>
              </div>
            </div>
            <textarea
              id="mcp-json-input"
              class="input input-mono json-textarea"
              rows="4"
              bind:value={mcpJsonText}
              disabled={busy || mcpBusy}
              placeholder={'{"mcpServers": {"tools": {"command": "python", "args": ["server.py"]}}}'}
            ></textarea>
            <div>
              <button type="button" id="btn-apply-mcp-json" class="btn btn-secondary btn-sm" disabled={busy || mcpBusy || !mcpJsonText.trim()} onclick={applyMcpJson}>{t('chat.applyJson')}</button>
            </div>
          </div>
          <div class="mcp-form">
            <label for="mcp-name">{t('chat.serverName')}</label><input id="mcp-name" class="input" bind:value={serverName} placeholder="local-tools" disabled={busy || mcpBusy} />
            <label for="mcp-command">{t('chat.serverCommand')}</label><input id="mcp-command" class="input input-mono" bind:value={serverCommand} placeholder='["python", "server.py"]' disabled={busy || mcpBusy} />
            <button type="button" class="btn btn-secondary btn-sm" onclick={connectServer} disabled={busy || mcpBusy || !serverName.trim() || !serverCommand.trim()}>{t('chat.connect')}</button>
          </div>
          {#each servers as server}<div class="server-row"><span>{server}</span><button type="button" class="btn btn-secondary btn-sm" disabled={busy || mcpBusy} onclick={() => disconnectServer(server)}>{t('chat.disconnect')}</button></div>{/each}
          <div class="server-row"><span>{t('chat.sandboxHint')}</span><button type="button" class="btn btn-secondary btn-sm" disabled={busy || mcpBusy} onclick={toggleSandbox}>{t(sandboxEnabled ? 'chat.disableSandbox' : 'chat.enableSandbox')}</button></div>
        </details>

        <details class="skills-panel">
          <summary>Skills · SKILL.md ({skills.length}{#if current.skillIds?.length}, {current.skillIds.length} active{/if})</summary>
          <p>{t('chat.installSkillsHint', { file: 'SKILL.md', mention: '@skill' })}</p>
          {#if skills.length}
            <div class="installed-skills">
              {#each skills as skill (skill.id)}
                {@const isActive = (current.skillIds ?? []).includes(skill.id)}
                <div class="skill-item" class:active={isActive}>
                  <button type="button" class="btn btn-secondary btn-sm" onclick={() => toggleSkill(skill)} disabled={busy}>
                    {isActive ? '✓ Active' : 'Use'} · @{skill.name}
                  </button>
                  <span class="badge badge-neutral">{skill.scope}</span>
                  {#if skill.scope === 'workspace'}
                    <button type="button" class="btn btn-ghost btn-sm" disabled={busy || skillBusy} onclick={() => removeInstalledSkill(skill.id)}>{t('chat.remove')}</button>
                  {/if}
                </div>
              {/each}
            </div>
          {:else}
            <p>{skillsError || 'No skills installed yet. Import a SKILL.md file or create one below.'}</p>
          {/if}
          <div class="panel-block">
            <div class="panel-block-head">
              <span class="field-label">{t('chat.installSkill')}</span>
              <div>
                <input bind:this={skillFileInput} type="file" accept=".md,.markdown,text/markdown" style="display:none" onchange={handleSkillFilePicked} />
                <button type="button" id="btn-import-skill-file" class="btn btn-secondary btn-sm" disabled={busy || skillBusy} onclick={() => skillFileInput.click()}>{t('chat.importSkill')}</button>
              </div>
            </div>
            <div class="mcp-form">
              <label for="skill-install-name">{t('chat.skillName')}</label>
              <input id="skill-install-name" class="input input-mono" bind:value={skillInstallName} placeholder="code-reviewer" disabled={busy || skillBusy} />
            </div>
            <textarea
              id="skill-install-content"
              class="input json-textarea"
              rows="3"
              bind:value={skillInstallContent}
              disabled={busy || skillBusy}
              placeholder="# Code Reviewer&#10;Inspect diffs carefully and report bugs concisely."
            ></textarea>
            <div>
              <button type="button" id="btn-install-skill" class="btn btn-secondary btn-sm" disabled={busy || skillBusy || !skillInstallName.trim() || !skillInstallContent.trim()} onclick={() => installSkill()}>{t('chat.installUseSkill')}</button>
            </div>
          </div>
        </details>
      </div>
    </header>

    <ChatCustomizerModal open={showCustomizer} onClose={() => { showCustomizer = false }} />
    <AgentStudioDrawer
      open={showAgentStudio}
      {agents}
      activeAgentId={current?.agentId}
      {skills}
      {memories}
      {contextPct}
      onClose={() => { showAgentStudio = false }}
      onSelectAgent={selectAgent}
      onAgentsChanged={refreshAgents}
      onMemoriesChanged={refreshMemories}
      onSkillsChanged={refreshSkills}
      onRunRoutineNow={(ag) => { void runAgentRoutine(ag) }}
      onCompactNow={() => { void runCompaction(current, false) }}
    />

    <div
      class="chat-stage"
      role="region"
      aria-label={t('chat.workspace')}
      class:composer-top={chatAppearance.composerDock === 'top'}
      class:has-pet={chatAppearance.showFloatingPet}
      ondragover={(e) => e.preventDefault()}
      ondrop={handleComposerDrop}
    >
      {#if chatAppearance.wallpaperUrl}
        <div
          class="chat-wallpaper"
          aria-hidden="true"
          style={`background-image:${safeCssBgUrl(chatAppearance.wallpaperUrl)}; opacity:${chatAppearance.wallpaperOpacity}; filter:blur(${chatAppearance.wallpaperBlur}px);`}
        ></div>
      {/if}

      {#if chatAppearance.showFloatingPet}
        <div class="floating-pet-dock">
          <AgentPet
            species={activePetSpecies}
            customUrl={activePetCustomUrl}
            mood={petMood}
            name={activePetName}
            accent={chatAccent}
            size={36}
            onclick={() => { showAgentStudio = !showAgentStudio; showCustomizer = false }}
          />
          <span class="pet-label">
            {activePetName}
          </span>
        </div>
      {/if}

      <div class="messages" aria-live="polite" aria-relevant="additions">
        <div class="messages-track">
        {#if !current.messages.length}
          <div class="empty">
            <AgentPet
              species={activePetSpecies}
              customUrl={activePetCustomUrl}
              mood={petMood}
              name={activePetName}
              accent={chatAccent}
              size={48}
            />
            <strong>{t('chat.ready')}</strong>
            <span>{t('chat.hint')}</span>
          </div>
        {/if}
        {#each current.messages as message, index}
          {@const align = message.role === 'user' ? chatAppearance.userAlign : chatAppearance.assistantAlign}
          <article class="message" class:user={message.role === 'user'} class:compacted-msg={message.compacted} data-align={align}>
            <div class="message-head">
              <div class="sender-id">
                {#if message.role === 'user' && chatAppearance.userAvatarUrl}
                  <img class="msg-avatar" src={chatAppearance.userAvatarUrl} alt={t('chat.you')} />
                {:else if message.role === 'assistant' && activePetCustomUrl}
                  <img class="msg-avatar" src={activePetCustomUrl} alt={activePetName} />
                {:else if message.role === 'assistant'}
                  <AgentPet
                    species={activePetSpecies}
                    mood={busy && index === current.messages.length - 1 ? petMood : 'idle'}
                    name={activePetName}
                    size={22}
                    interactive={false}
                    accent={chatAccent}
                  />
                {/if}
                <strong>{message.role === 'user' ? t('chat.you') : activePetName}</strong>
                {#if message.proactiveLabel}
                  <span class="proactive-badge">{message.proactiveLabel}</span>
                {/if}
              </div>
              <div class="msg-head-actions">
                {#if message.role === 'assistant' && message.content.trim()}
                  <button
                    type="button"
                    class="msg-speak-btn"
                    title={t('chat.speakMessage')}
                    onclick={() => {
                      if (voiceRuntime.speaking) interruptSpeech()
                      else speakFullText(message.content)
                    }}
                  >
                    <i class={`bi ${voiceRuntime.speaking ? 'bi-stop-circle' : 'bi-volume-up'}`} aria-hidden="true"></i>
                    <span>{voiceRuntime.speaking ? 'Stop' : 'Listen'}</span>
                  </button>
                {/if}
                {#if index === undoTargetIndex}
                  <button
                    type="button"
                    class="checkpoint-undo-btn"
                    title={t('chat.undo')}
                    aria-label={t('chat.undo')}
                    onclick={() => void handleUndoLastCheckpoint()}
                  >
                    <i class="bi bi-arrow-counterclockwise" aria-hidden="true"></i>
                    <span class="undo-hover-label" aria-hidden="true">{t('chat.undo')}</span>
                  </button>
                {/if}
              </div>
            </div>
          {#if message.role === 'assistant' && (message.reasoning || message.toolEvents?.length || message.skillsUsed?.length)}
            <details class="activity-trace">
              <summary class="activity-summary">
                {#if message.reasoning}
                  <span class="trace-chip chip-think" title={t('chat.reasoningLabel')}>
                    <i class="bi bi-lightbulb" aria-hidden="true"></i>
                    <span>{t('chat.reasoningLabel')}</span>
                  </span>
                {/if}
                {#if message.toolEvents?.length}
                  <span class="trace-chip chip-tool">
                    <i class="bi bi-hammer" aria-hidden="true"></i>
                    <span>{t('chat.toolsUsed', { n: message.toolEvents.length })}</span>
                  </span>
                {/if}
                {#if message.skillsUsed?.length}
                  <span class="trace-chip chip-skill">
                    <i class="bi bi-list-ul" aria-hidden="true"></i>
                    <span>{t('chat.skillsUsed', { n: message.skillsUsed.length })}</span>
                  </span>
                {/if}
              </summary>
              <div class="activity-body">
                {#if message.skillsUsed?.length}
                  <div class="activity-section">
                    <div class="activity-label"><i class="bi bi-list-ul" aria-hidden="true"></i> {t('chat.skillsToggle')}</div>
                    <div class="activity-chips">
                      {#each message.skillsUsed as sk}
                        <span class="activity-pill">@{sk}</span>
                      {/each}
                    </div>
                  </div>
                {/if}
                {#if message.toolEvents?.length}
                  <div class="activity-section">
                    <div class="activity-label"><i class="bi bi-hammer" aria-hidden="true"></i> {t('chat.mcpPerm')}</div>
                    <div class="tool-history">
                      {#each message.toolEvents as event}
                        <div class="tool-event-row">
                          <span class="tool-event-title"><i class="bi bi-hammer" aria-hidden="true"></i> {event.server === 'web' ? t('chat.webSearchServer') : event.server}/{event.name} · {t(event.approved ? 'chat.toolApproved' : 'chat.toolDenied')}</span>
                          {#if event.args}
                            <code class="tool-event-args">{event.args}</code>
                          {/if}
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}
                {#if message.reasoning}
                  <div class="activity-section">
                    <div class="activity-label"><i class="bi bi-lightbulb" aria-hidden="true"></i> {t('chat.reasoningLabel')}</div>
                    <pre class="reasoning-pre">{message.reasoning}</pre>
                  </div>
                {/if}
              </div>
            </details>
          {/if}
          {#if message.attachments?.length}
            <div class="msg-attachments">
              {#each message.attachments as att (att.id)}
                <figure class="msg-media-card">
                  <img src={att.dataUrl} alt={att.name} />
                  <figcaption>{att.name}{att.isGif ? ' · GIF' : ''}</figcaption>
                </figure>
              {/each}
            </div>
          {/if}
          {#if message.role === 'assistant'}
            <div class="message-body rich-body">
              <RichMessage content={message.content} fallback={busy && index === current.messages.length - 1 ? t('chat.thinking') : ''} />
            </div>
            {#if message.metrics?.tkPerSec !== undefined || message.metrics?.completionTokens !== undefined}
              <div class="message-meta">
                {#if message.metrics?.tkPerSec !== undefined}
                  <span><i class="bi bi-speedometer2" aria-hidden="true"></i> {message.metrics.tkPerSec.toFixed(1)} tk/s</span>
                {/if}
                {#if message.metrics?.completionTokens !== undefined}
                  <span>{message.metrics.completionTokens} tok</span>
                {/if}
              </div>
            {/if}
          {:else}
            <div class="message-body">{message.content}</div>
          {/if}
          </article>
        {/each}
        </div>
      </div>

      {#if considered.length}<div class="tool-summary">{t('chat.toolsConsidered')}: {considered.map(item => `${item.server}/${item.tool.name}`).join(', ')}</div>{/if}
      {#if showCompactAlert}
        <div class="compaction-alert" role="alert">
          <span>
            <i class="bi bi-stars" aria-hidden="true"></i>
            {t('chat.compactQuestion', { n: contextPct })}
          </span>
          <div class="action-buttons">
            <button type="button" id="btn-dismiss-compact" class="btn btn-secondary btn-sm" onclick={() => { current.compactionDismissedAtPct = contextPct; saveConversations() }}>{t('chat.continueAction')}</button>
            <button type="button" id="btn-approve-compact" class="btn btn-primary btn-sm" onclick={() => void runCompaction(current, false)}>{t('chat.approveCompact')}</button>
          </div>
        </div>
      {/if}
      {#if approval}
        <div class="tool-approval" role="group" aria-live="assertive" aria-label={t('chat.reviewTool')}>
          <strong>{t('chat.reviewTool')}: {approval.server}/{approval.name}</strong>
          {#if approval.repaired}<span>{t('chat.repairedArgs')}</span>{/if}
          <pre>{approval.args}</pre>
          <div class="action-buttons"><button class="btn btn-secondary" type="button" onclick={() => approval?.resolve(false)}>{t('chat.deny')}</button><button class="btn btn-primary" type="button" onclick={() => approval?.resolve(true)}>{t('chat.approve')}</button></div>
        </div>
      {/if}
      {#if notice}<p class="notice" role="status">{notice}</p>{/if}

      <form class="composer" onsubmit={(event) => { event.preventDefault(); void send() }}>
        {#if showVoicePanel || voiceRuntime.liveMode}
          <section id="voice-kokoro-panel" class="voice-panel" aria-label={t('chat.voiceControls')}>
            <div class="voice-row">
              <span class="voice-title"><i class="bi bi-soundwave" aria-hidden="true"></i> TTS</span>
              <div class="seg-mini">
                <button
                  type="button"
                  id="btn-engine-browser"
                  class:active={voicePrefs.engine === 'browser'}
                  onclick={() => toggleVoiceEngine('browser')}
                >{t('chat.browser')}</button>
                <button
                  type="button"
                  id="btn-engine-kokoro"
                  class:active={voicePrefs.engine === 'kokoro'}
                  onclick={() => toggleVoiceEngine('kokoro')}
                >Kokoro 82M</button>
              </div>

              {#if voiceRuntime.kokoroStatus?.kokoro_model_ready}
                <span class="voice-badge ok" data-testid="kokoro-ready-badge">
                  <i class="bi bi-check-circle-fill" aria-hidden="true"></i> {t('chat.kokoroReady')}
                </span>
              {:else}
                <button
                  type="button"
                  id="btn-setup-kokoro"
                  class="btn btn-secondary btn-sm"
                  title={t('chat.kokoroDownloadHint')}
                  disabled={Boolean(voiceRuntime.kokoroStatus?.installing)}
                  onclick={() => void startKokoroSetup()}
                >
                  {#if voiceRuntime.kokoroStatus?.installing}
                    {voiceRuntime.kokoroStatus.stage === 'downloading_model'
                      ? t('chat.downloading')
                      : voiceRuntime.kokoroStatus.stage === 'starting'
                        ? t('chat.starting')
                        : t('chat.installing')}
                  {:else}
                    {t('chat.setupKokoro')}
                  {/if}
                </button>
              {/if}

              <select id="select-kokoro-voice" class="input voice-select" aria-label={t('chat.kokoroVoice')} bind:value={voicePrefs.voiceId} onchange={saveVoicePrefs}>
                {#each voiceRuntime.kokoroStatus?.voices ?? [
                  { id: 'auto', label: 'Auto (EN / PT-BR)' },
                  { id: 'af_heart', label: 'Kokoro Heart (US F)' },
                  { id: 'af_bella', label: 'Kokoro Bella (US F)' },
                  { id: 'am_adam', label: 'Kokoro Adam (US M)' },
                  { id: 'bf_emma', label: 'Kokoro Emma (UK F)' },
                  { id: 'pf_dora', label: 'Kokoro Dora (PT-BR F)' },
                  { id: 'pm_alex', label: 'Kokoro Alex (PT-BR M)' },
                ] as v (v.id)}
                  <option value={v.id}>{v.label}</option>
                {/each}
              </select>

              <label class="voice-field" title={t('chat.speechSpeed')}>
                <span>{voicePrefs.speed.toFixed(1)}x</span>
                <input type="range" min="0.75" max="1.35" step="0.05" bind:value={voicePrefs.speed} onchange={saveVoicePrefs} />
              </label>

              <label class="voice-check">
                <input type="checkbox" bind:checked={voicePrefs.autoSpeakReplies} onchange={saveVoicePrefs} />
                <span>{t('chat.speakReplies')}</span>
              </label>

              <div class="proactive-mini">
                <span>{t('chat.every')}</span>
                <input type="number" class="input mins-mini" min="1" max="120" bind:value={voicePrefs.proactiveIntervalMins} onchange={saveVoicePrefs} />
                <span>{t('chat.minutes')}</span>
                <button
                  type="button"
                  id="btn-proactive-run-now"
                  class="btn btn-secondary btn-sm"
                  disabled={busy}
                  onclick={() => void runProactiveBriefing(true)}
                >{t('chat.briefingNow')}</button>
              </div>
            </div>
            {#if voiceRuntime.statusNote}
              <div class="voice-note" role="status">{voiceRuntime.statusNote}</div>
            {/if}
          </section>
        {/if}

        <div class="composer-shell">
          {#if voiceRuntime.liveMode || voiceRuntime.micState !== 'idle' || voiceRuntime.speaking || voiceRuntime.interimText}
            <div class="voice-live-ribbon" data-testid="voice-live-ribbon">
              <span class="live-dot" class:pulsing={voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording' || voiceRuntime.speaking}></span>
              <span class="voice-ribbon-label">
                {#if voiceRuntime.speaking}
                  {t('chat.speakingEngine', { engine: voicePrefs.engine === 'kokoro' ? 'Kokoro 82M' : t('chat.browser') })}
                {:else if voiceRuntime.micState === 'transcribing'}
                  {t('chat.transcribing')}
                {:else if voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording'}
                  {voiceRuntime.interimText || voiceRuntime.statusNote || t('chat.listening')}
                {:else}
                  {voiceRuntime.statusNote}
                {/if}
              </span>
              {#if voiceRuntime.micLevel > 0}
                <span class="mic-meter" aria-hidden="true"><span class="mic-fill" style={`width:${voiceRuntime.micLevel}%`}></span></span>
              {/if}
              {#if voiceRuntime.speaking}
                <button type="button" class="btn btn-secondary btn-sm" onclick={interruptSpeech}>{t('chat.stop')}</button>
              {/if}
            </div>
          {/if}

          {#if pendingAttachments.length}
            <div class="attach-preview">
              {#each pendingAttachments as att (att.id)}
                <div class="attach-chip">
                  <img src={att.dataUrl} alt={att.name} />
                  <span>{att.name}{att.isGif ? ' (GIF)' : ''}</span>
                  <button type="button" aria-label={t('chat.removeAttachment', { name: att.name })} onclick={() => { pendingAttachments = pendingAttachments.filter(x => x.id !== att.id) }}>×</button>
                </div>
              {/each}
            </div>
          {/if}

          <textarea
            id="chat-prompt"
            aria-label={t('chat.message')}
            bind:this={textarea}
            bind:value={prompt}
            oninput={updateSkillQuery}
            onpaste={handleComposerPaste}
            onkeydown={(event) => { if (event.key === 'Escape') query = null }}
            rows="2"
            placeholder={t(busy ? 'chat.steerPlaceholder' : 'chat.placeholder')}
          ></textarea>

          {#if query !== null && !busy}
            <div class="skill-suggestions" role="listbox" aria-label={t('chat.skills')}>
              {#each suggestions as skill (skill.id)}<button type="button" role="option" aria-selected="false" onclick={() => selectSkill(skill)}>@{skill.name} <span>{skill.scope}</span></button>{:else}<span>{skillsError || t('chat.noSkills')}</span>{/each}
            </div>
          {/if}
          {#if current.skillIds?.length}<div class="skill-tags" aria-label={t('chat.activeSkills')}>{#each current.skillIds as id (id)}<button type="button" disabled={busy} onclick={() => removeSkill(id)} aria-label={t('chat.removeSkill', { name: id })}>{id} ×</button>{/each}</div>{/if}

          <div class="composer-footer">
            <div class="composer-controls" role="toolbar" aria-label={t('chat.capabilities')}>
              <input
                bind:this={mediaFileInput}
                type="file"
                accept="image/*,.gif,.webp,.png,.jpg,.jpeg"
                multiple
                style="display:none"
                onchange={(e) => {
                  const inp = e.currentTarget as HTMLInputElement
                  void handleAttachmentFiles(inp.files)
                  inp.value = ''
                }}
              />
              <button
                type="button"
                id="btn-attach-media"
                class="ctrl-pill"
                disabled={busy}
                title={t('chat.attachMedia')}
                onclick={() => mediaFileInput?.click()}
              >
                <i class="bi bi-image" aria-hidden="true"></i>
                <span>IMG / GIF</span>
              </button>

              <button
                type="button"
                id="btn-toggle-web-search"
                class="ctrl-pill"
                class:active={activeWebSearch}
                aria-pressed={activeWebSearch}
                disabled={busy}
                onclick={toggleWebSearch}
              >
                <i class="bi bi-globe2" aria-hidden="true"></i>
                <span>{t('chat.webSearch')}</span>
              </button>

              <div class="perm-group" role="group" aria-label={t('chat.mcpPerm')}>
                <span class="perm-label"><i class="bi bi-hammer" aria-hidden="true"></i> {t('chat.mcpPerm')}</span>
                <button type="button" class="perm-btn" class:active={activeMcpPermission === 'ask'} disabled={busy} onclick={() => setMcpPermission('ask')}>{t('chat.mcpAsk')}</button>
                <button type="button" class="perm-btn" class:active={activeMcpPermission === 'auto'} disabled={busy} onclick={() => setMcpPermission('auto')}>{t('chat.mcpAuto')}</button>
                <button type="button" class="perm-btn" class:active={activeMcpPermission === 'off'} disabled={busy} onclick={() => setMcpPermission('off')}>{t('chat.mcpOff')}</button>
              </div>

              <button
                type="button"
                id="btn-toggle-skills"
                class="ctrl-pill"
                class:active={activeSkillsEnabled}
                aria-pressed={activeSkillsEnabled}
                disabled={busy}
                onclick={toggleSkillsEnabled}
              >
                <i class="bi bi-list-check" aria-hidden="true"></i>
                <span>{t('chat.skillsToggle')}</span>
              </button>

              <button
                type="button"
                id="btn-proactive-watch"
                class="ctrl-pill"
                class:active={voicePrefs.proactiveEnabled}
                aria-pressed={voicePrefs.proactiveEnabled}
                title={t('chat.autoWatchHint')}
                onclick={toggleProactiveWatch}
              >
                <i class="bi bi-lightning-charge" aria-hidden="true"></i>
                <span>{t('chat.autoWatch')}</span>
              </button>
            </div>

            <div class="composer-right">
              <button
                type="button"
                id="btn-voice-mic"
                class="voice-icon-btn"
                class:recording={voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording'}
                aria-pressed={voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording'}
                aria-label={voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording' ? t('chat.stopDictation') : t('chat.dictation')}
                title={t('chat.micHint')}
                disabled={busy && !voiceRuntime.liveMode}
                onclick={() => toggleDictation(handleVoiceTranscript)}
              >
                <i class={`bi ${voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording' ? 'bi-mic-fill' : 'bi-mic'}`} aria-hidden="true"></i>
              </button>

              <button
                type="button"
                id="btn-live-voice"
                class="voice-orb-btn"
                class:live-active={voiceRuntime.liveMode}
                aria-pressed={voiceRuntime.liveMode}
                aria-label={voiceRuntime.liveMode ? t('chat.stopLive') : t('chat.liveChat')}
                title={t('chat.liveHint')}
                onclick={() => {
                  if (voiceRuntime.liveMode) stopLiveVoiceChat()
                  else startLiveVoiceChat(handleVoiceTranscript)
                }}
              >
                <i class={`bi ${voiceRuntime.liveMode ? 'bi-broadcast' : 'bi-soundwave'}`} aria-hidden="true"></i>
                {#if voiceRuntime.liveMode}
                  <span>{t('chat.live')}</span>
                {/if}
              </button>

              <button
                type="button"
                id="btn-voice-panel"
                class="voice-icon-btn"
                class:active={showVoicePanel}
                aria-pressed={showVoicePanel}
                aria-label={t('chat.voiceSettingsA11y')}
                title={voicePrefs.engine === 'kokoro' ? t('chat.kokoroSettings') : t('chat.voiceSettings')}
                onclick={() => { showVoicePanel = !showVoicePanel }}
              >
                <i class="bi bi-sliders2" aria-hidden="true"></i>
                {#if voicePrefs.engine === 'kokoro'}
                  <span class="kokoro-dot" aria-hidden="true"></span>
                {/if}
              </button>

              {#if busy}
                <button type="button" class="composer-action-btn secondary" onclick={() => controller?.abort()}><span>{t('chat.stop')}</span></button>
                <button type="button" class="composer-action-btn primary" disabled={!prompt.trim()} onclick={() => { steering = prompt.trim(); prompt = ''; controller?.abort() }}><span>{t('chat.steer')}</span></button>
              {:else}
                {#if current.messages.at(-1)?.role === 'user'}
                  <button type="button" class="composer-action-btn secondary" onclick={() => void send(true)}><span>{t('chat.continue')}</span></button>
                {/if}
                <button class="composer-action-btn primary" type="submit" disabled={!prompt.trim() && !pendingAttachments.length}><span>{t('chat.send')}</span></button>
              {/if}
            </div>
          </div>
        </div>
      </form>
    </div>
  </section>
</div>

<style>
  .chat-layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); min-height: 0; height: 100%; overflow: hidden; }
  .sessions { overflow: auto; border-right: 1px solid var(--color-border); padding: 16px 10px; }
  .sessions-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 14px; }
  h2 { margin: 0; font-size: 13px; }
  .session-row { display: flex; align-items: center; border-left: 2px solid transparent; }
  .session-row.selected { background: var(--color-brand-subtle); border-color: var(--color-brand); }
  .session-select, .delete { border: 0; background: transparent; color: var(--color-ink-subtle); cursor: pointer; }
  .session-select { min-width: 0; flex: 1; padding: 10px 8px; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .delete { font-size: 20px; padding: 4px 8px; }
  .conversation { min-width: 0; min-height: 0; display: flex; flex-direction: column; }
  header { padding: 14px 22px 10px; border-bottom: 1px solid var(--color-border); max-height: 44vh; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; }
  .header-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  .header-brand { display: flex; flex-direction: column; gap: 3px; flex: 1 1 240px; min-width: 180px; }
  .title-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .telemetry-bar { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; justify-content: flex-end; margin-left: auto; }
  .telemetry-actions { display: inline-flex; align-items: center; gap: 5px; flex-shrink: 0; }
  .telemetry-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 26px;
    min-height: 26px;
    box-sizing: border-box;
    padding: 0 8px;
    font-size: 11px;
    line-height: 1;
    font-family: var(--font-mono);
    border: 1px solid var(--color-border);
    background: var(--color-surface-muted);
    color: var(--color-ink-subtle);
    border-radius: var(--radius-md);
    white-space: nowrap;
  }
  .customize-pill {
    cursor: pointer;
    color: var(--color-ink);
  }
  .customize-pill:hover {
    border-color: var(--chat-accent, var(--color-brand));
  }
  .customize-pill.active {
    border-color: var(--chat-accent, var(--color-brand));
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .voice-panel {
    max-width: var(--chat-max-w, 860px);
    width: 100%;
    margin: 0 auto 8px;
    padding: 7px 12px;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    background: var(--color-surface-muted);
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 11px;
  }
  .voice-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .voice-title { font-weight: 600; color: var(--color-ink); display: inline-flex; align-items: center; gap: 4px; }
  .seg-mini { display: inline-flex; border: 1px solid var(--color-border); border-radius: 999px; overflow: hidden; }
  .seg-mini button {
    border: 0;
    background: var(--color-surface);
    color: var(--color-ink-subtle);
    font-size: 11px;
    height: 24px;
    padding: 0 9px;
    cursor: pointer;
  }
  .seg-mini button.active {
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .voice-badge.ok {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--color-success);
    background: var(--color-success-bg);
    padding: 2px 8px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 11px;
  }
  .voice-field, .voice-check, .proactive-mini {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: var(--color-ink-subtle);
  }
  .voice-select { height: 26px; font-size: 11px; padding: 0 8px; width: auto; min-width: 140px; }
  .voice-check { cursor: pointer; height: 26px; }
  .voice-check input[type="checkbox"] {
    width: 14px;
    height: 14px;
    margin: 0;
    accent-color: var(--chat-accent, var(--color-brand));
    flex-shrink: 0;
  }
  .mins-mini { width: 46px; height: 26px; padding: 0 5px; text-align: center; font-size: 11px; }
  .voice-note { font-size: 11px; color: var(--chat-accent, var(--color-brand)); }
  .model-pill span { max-width: 190px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ctx-bar { width: 42px; height: 5px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 999px; overflow: hidden; display: inline-block; }
  .ctx-fill { display: block; height: 100%; background: var(--chat-accent, var(--color-brand)); border-radius: 999px; }
  h1 { margin: 0; font-size: 22px; }
  header p { margin: 0; color: var(--color-ink-subtle); font-size: 12px; }
  .chat-tools-bar { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
  .mcp-settings, .skills-panel { font-size: 12px; border: 1px solid var(--color-border); background: var(--color-surface-muted); padding: 8px 10px; border-radius: var(--radius-sm); }
  .mcp-settings summary, .skills-panel summary { cursor: pointer; font-weight: 600; }
  .mcp-settings p, .skills-panel p { margin: 8px 0; }
  .panel-block { display: flex; flex-direction: column; gap: 6px; padding: 8px; border: 1px solid var(--color-border); background: var(--color-surface); border-radius: var(--radius-sm); margin: 8px 0; }
  .panel-block-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
  .json-textarea { height: auto; min-height: 70px; padding: 8px; font-size: 12px; resize: vertical; }
  .installed-skills { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
  .skill-item { display: inline-flex; align-items: center; gap: 4px; padding: 2px; border: 1px solid transparent; border-radius: var(--radius-sm); }
  .skill-item.active { border-color: var(--chat-accent, var(--color-brand)); background: var(--color-brand-subtle); }
  .mcp-form { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .mcp-form input { width: auto; min-width: 130px; flex: 1; }
  .server-row { display: flex; justify-content: space-between; align-items: center; padding-top: 5px; }

  .chat-stage {
    position: relative;
    flex: 1;
    min-height: 220px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .chat-stage.composer-top .composer {
    order: -1;
    border-top: 0;
    border-bottom: 1px solid var(--color-border);
  }
  .chat-stage .messages { order: 2; }
  .chat-stage .composer { order: 3; }
  .chat-wallpaper {
    position: absolute;
    inset: -12px;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    pointer-events: none;
    z-index: 0;
  }
  .floating-pet-dock {
    position: relative;
    order: 1;
    align-self: flex-end;
    flex-shrink: 0;
    margin: 8px 16px 0 0;
    z-index: 4;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 1px;
    min-width: 60px;
    background: color-mix(in srgb, var(--color-surface) 86%, transparent);
    border: 1px solid var(--color-border);
    border-radius: 10px;
    padding: 5px 8px 4px;
    backdrop-filter: blur(6px);
    box-shadow: var(--shadow-sm);
  }
  .pet-label {
    display: block;
    width: 100%;
    text-align: center;
    font-size: 9.5px;
    line-height: 1.15;
    color: var(--color-ink-subtle);
    font-weight: 600;
    max-width: 66px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tool-summary, .tool-approval, .notice, .compaction-alert, .composer, .messages { position: relative; z-index: 1; }
  .tool-summary, .tool-approval, .notice, .compaction-alert { order: 2; }
  .tool-summary { padding: 4px 24px; font-size: 11px; color: var(--color-ink-subtle); }
  .compaction-alert {
    padding: 8px 20px;
    border-top: 1px solid var(--color-border);
    background: var(--color-warn-bg);
    color: var(--color-ink);
    font-size: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
  }
  .compaction-alert span { display: inline-flex; align-items: center; gap: 6px; }
  .tool-approval { padding: 12px 24px; display: grid; gap: 8px; border-top: 1px solid var(--color-border); font-size: 12px; background: var(--color-surface); }
  .tool-approval pre { max-height: 140px; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; }
  .messages {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
    padding: 24px 18px 32px 24px;
    display: flex;
    flex-direction: column;
  }
  .chat-stage.has-pet .messages {
    padding-top: 10px;
  }
  .messages::-webkit-scrollbar { width: 6px; }
  .messages::-webkit-scrollbar-track { background: transparent; }
  .messages::-webkit-scrollbar-thumb { background: color-mix(in srgb, var(--color-border-strong) 55%, transparent); border-radius: 999px; }
  .messages-track {
    max-width: var(--chat-max-w, 860px);
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
    flex: 1;
  }
  .empty { margin: auto; text-align: center; color: var(--color-ink-subtle); display: grid; gap: 10px; justify-items: center; }
  .message {
    max-width: 100%;
    width: 100%;
    align-self: stretch;
    padding: 6px 0 14px;
    background: transparent;
    border: 0;
    border-radius: 0;
  }
  .message:not(.user) {
    margin-bottom: 14px;
  }
  .message[data-align="left"] { align-self: flex-start; width: 100%; }
  .message[data-align="right"] { align-self: flex-end; width: min(84%, 100%); }
  .message[data-align="center"] { align-self: center; width: 100%; }
  .message.user {
    width: fit-content;
    min-width: 140px;
    max-width: min(74%, 640px);
    padding: 14px 20px;
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: color-mix(in srgb, var(--color-surface-muted) 92%, transparent);
    border: 1px solid color-mix(in srgb, var(--color-border) 82%, transparent);
    border-radius: var(--chat-radius);
    box-shadow: var(--card-shadow);
  }
  .message.user[data-align="right"] {
    align-self: flex-end;
    margin-left: auto;
  }
  .message.user[data-align="left"] {
    align-self: flex-start;
    margin-right: auto;
  }
  .message.user[data-align="center"] {
    align-self: center;
    margin-inline: auto;
  }
  .message.user .message-body {
    margin-top: 0;
    text-align: left;
    line-height: 1.65;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    word-break: normal;
  }
  .message.compacted-msg {
    padding: 10px 14px;
    background: color-mix(in srgb, var(--color-surface-muted) 55%, transparent);
    border: 1px dashed var(--color-border);
    border-radius: 10px;
    opacity: 0.9;
    font-size: 0.92em;
  }
  .message-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    min-height: 22px;
    font-size: 12.5px;
    color: var(--color-ink-subtle);
  }
  .sender-id { display: inline-flex; align-items: center; gap: 7px; }
  .sender-id strong { font-weight: 600; color: var(--color-ink); }
  .msg-avatar { width: 22px; height: 22px; border-radius: 999px; object-fit: cover; border: 1px solid var(--color-border-strong); }
  .msg-head-actions { display: inline-flex; align-items: center; gap: 8px; }
  .checkpoint-undo-btn {
    position: relative;
    width: 22px;
    height: 22px;
    padding: 0;
    border: 1px solid transparent;
    border-radius: 999px;
    background: transparent;
    color: var(--color-ink-subtle);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    opacity: 0.45;
    transition: opacity var(--anim-duration), background var(--anim-duration), border-color var(--anim-duration), color var(--anim-duration);
  }
  .message:hover .checkpoint-undo-btn,
  .checkpoint-undo-btn:hover,
  .checkpoint-undo-btn:focus-visible {
    opacity: 1;
  }
  .checkpoint-undo-btn:hover,
  .checkpoint-undo-btn:focus-visible {
    background: var(--color-surface);
    border-color: var(--color-border-strong);
    color: var(--color-ink);
  }
  .undo-hover-label {
    position: absolute;
    bottom: calc(100% + 6px);
    right: 0;
    white-space: nowrap;
    padding: 3px 8px;
    border-radius: 6px;
    background: var(--color-surface);
    border: 1px solid var(--color-border-strong);
    color: var(--color-ink);
    font-size: 11px;
    font-weight: 500;
    line-height: 1.2;
    box-shadow: var(--shadow-sm);
    opacity: 0;
    transform: translateY(2px);
    pointer-events: none;
    transition: opacity var(--anim-duration) var(--ease), transform var(--anim-duration) var(--ease);
    z-index: 6;
  }
  .checkpoint-undo-btn:hover .undo-hover-label,
  .checkpoint-undo-btn:focus-visible .undo-hover-label {
    opacity: 1;
    transform: translateY(0);
  }
  .msg-speak-btn {
    border: 1px solid transparent;
    background: transparent;
    color: var(--color-ink-subtle);
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 11px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    opacity: 0.78;
  }
  .msg-speak-btn:hover {
    border-color: var(--color-border);
    background: var(--color-surface-muted);
    color: var(--color-ink);
    opacity: 1;
  }
  .proactive-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 999px;
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .voice-live-ribbon {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-surface-muted);
    font-size: 11.5px;
  }
  .live-dot { width: 8px; height: 8px; border-radius: 999px; background: var(--color-success); flex-shrink: 0; }
  .live-dot.pulsing { animation: pulseLive calc(1s * var(--anim-scale)) ease-in-out infinite; }
  @keyframes pulseLive { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.25); } }
  .voice-ribbon-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .mic-meter { width: 70px; height: 6px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 999px; overflow: hidden; }
  .mic-fill { display: block; height: 100%; background: var(--color-success); transition: width calc(0.08s * var(--anim-scale)) linear; }
  .message-body { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; margin-top: 8px; font-size: var(--chat-font-size, 15px); }
  .message-body.rich-body { white-space: normal; margin-top: 10px; }
  .msg-attachments { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
  .msg-media-card {
    margin: 0;
    border: 1px solid var(--color-border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--color-surface-muted);
    max-width: 260px;
  }
  .msg-media-card img { display: block; max-width: 260px; max-height: 200px; width: 100%; height: auto; object-fit: cover; }
  .msg-media-card figcaption { padding: 4px 8px; font-size: 11px; color: var(--color-ink-subtle); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .activity-trace {
    margin-top: 8px;
    border: 0;
    background: transparent;
    font-size: 12px;
  }
  .activity-summary {
    list-style: none;
    cursor: pointer;
    padding: 2px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    user-select: none;
  }
  .activity-summary::-webkit-details-marker { display: none; }
  .trace-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    border: 1px solid color-mix(in srgb, var(--color-border) 80%, transparent);
    background: color-mix(in srgb, var(--color-surface-muted) 70%, transparent);
    color: var(--color-ink-subtle);
  }
  .trace-chip:hover { color: var(--color-ink); border-color: var(--color-border-strong); }
  .trace-chip.chip-think i { color: #d97706; }
  .trace-chip.chip-tool i { color: var(--chat-accent, var(--color-brand)); }
  .trace-chip.chip-skill i { color: var(--color-info); }
  .activity-body {
    margin-top: 8px;
    padding: 10px 12px;
    border: 1px solid var(--color-border);
    border-radius: 10px;
    background: color-mix(in srgb, var(--color-surface-muted) 75%, transparent);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .activity-section { display: flex; flex-direction: column; gap: 4px; }
  .activity-label { font-size: 11px; font-weight: 600; color: var(--color-ink-subtle); display: flex; align-items: center; gap: 5px; }
  .activity-chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .activity-pill { font-family: var(--font-mono); font-size: 11px; padding: 2px 7px; border-radius: var(--radius-sm); background: var(--color-surface-muted); border: 1px solid var(--color-border); }
  .tool-history { display: grid; gap: 6px; font-size: 11px; color: var(--color-ink-subtle); }
  .tool-event-row { display: flex; flex-direction: column; gap: 2px; padding: 5px 8px; border-radius: var(--radius-sm); background: var(--color-surface-muted); border: 1px solid var(--color-border); }
  .tool-event-title { display: flex; align-items: center; gap: 5px; color: var(--color-ink); font-weight: 500; }
  .tool-event-args { font-family: var(--font-mono); font-size: 11px; color: var(--color-ink-subtle); overflow-wrap: anywhere; }
  .reasoning-pre {
    margin: 0;
    padding: 8px 10px;
    max-height: 220px;
    overflow: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.45;
    border-radius: var(--radius-sm);
    background: var(--color-surface-muted);
    border: 1px solid var(--color-border);
    color: var(--color-ink-subtle);
  }
  .message-meta {
    margin-top: 10px;
    padding-top: 2px;
    border-top: 0;
    display: flex;
    gap: 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-ink-subtle);
    opacity: 0.72;
  }
  .message-meta span { display: inline-flex; align-items: center; gap: 4px; }
  .composer {
    border-top: 0;
    padding: 6px 24px 14px;
    background: transparent;
  }
  .composer-shell {
    max-width: var(--chat-max-w, 860px);
    width: 100%;
    margin: 0 auto;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card);
    background: color-mix(in srgb, var(--color-surface-muted) 92%, transparent);
    box-shadow: var(--card-shadow);
    padding: 6px 10px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: border-color var(--anim-duration) var(--ease);
  }
  .composer-shell:focus-within {
    border-color: var(--color-border-strong);
  }
  .composer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }
  .composer-controls {
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
  }
  .composer-right {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    margin-left: auto;
  }
  .attach-preview { display: flex; gap: 8px; flex-wrap: wrap; padding: 2px 4px 0; }
  .attach-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px 3px 4px;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    background: var(--color-surface);
    font-size: 11px;
  }
  .attach-chip img { width: 24px; height: 24px; border-radius: 4px; object-fit: cover; }
  .attach-chip button { border: 0; background: transparent; cursor: pointer; font-size: 14px; color: var(--color-ink-subtle); }
  .ctrl-pill, .perm-group {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 26px;
    min-height: 26px;
    box-sizing: border-box;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-ink-subtle);
    font-size: 11px;
    line-height: 1;
    border-radius: 999px;
    padding: 0 9px;
    white-space: nowrap;
  }
  .ctrl-pill { cursor: pointer; }
  .ctrl-pill:hover { border-color: var(--color-border-strong); color: var(--color-ink); }
  .ctrl-pill.active {
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
    border-color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .perm-group { padding: 0 3px 0 9px; gap: 2px; }
  .perm-label { display: inline-flex; align-items: center; gap: 4px; margin-right: 3px; font-size: 11px; }
  .perm-btn {
    border: 0;
    background: transparent;
    color: var(--color-ink-subtle);
    font-size: 10.5px;
    line-height: 1;
    height: 20px;
    min-height: 20px;
    padding: 0 6px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }
  .perm-btn.active {
    background: var(--color-surface-muted);
    color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .voice-icon-btn, .voice-orb-btn {
    position: relative;
    height: 28px;
    min-height: 28px;
    box-sizing: border-box;
    border-radius: 999px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-ink-subtle);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    font-size: 13px;
    line-height: 1;
    cursor: pointer;
  }
  .voice-icon-btn { width: 28px; padding: 0; }
  .voice-orb-btn { min-width: 28px; padding: 0 7px; }
  .voice-icon-btn:hover, .voice-orb-btn:hover {
    border-color: var(--color-border-strong);
    color: var(--color-ink);
  }
  .voice-icon-btn.recording {
    border-color: var(--chat-accent, var(--color-brand));
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
  }
  .voice-icon-btn.active {
    border-color: var(--chat-accent, var(--color-brand));
    color: var(--chat-accent, var(--color-brand));
  }
  .voice-orb-btn.live-active {
    padding: 0 10px;
    font-size: 11px;
    font-weight: 600;
    border-color: var(--color-success);
    background: var(--color-success-bg);
    color: var(--color-success);
  }
  .kokoro-dot {
    position: absolute;
    top: 3px;
    right: 3px;
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: var(--color-success);
  }
  .composer-action-btn {
    height: 28px;
    min-height: 28px;
    box-sizing: border-box;
    padding: 0 14px;
    margin: 0;
    border-radius: 999px;
    font-family: var(--font-ui, var(--font-sans));
    font-size: 11.5px;
    font-weight: 600;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    white-space: nowrap;
    cursor: pointer;
  }
  .composer-action-btn span {
    display: inline-block;
    line-height: 1;
    transform: translateY(-0.5px);
  }
  .composer-action-btn.primary {
    border: 0;
    background: var(--chat-accent, var(--color-brand));
    color: #fff;
  }
  .composer-action-btn.primary:disabled {
    opacity: 0.42;
    cursor: not-allowed;
  }
  .composer-action-btn.secondary {
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    color: var(--color-ink);
  }
  textarea {
    width: 100%;
    resize: none;
    min-height: 44px;
    max-height: 180px;
    padding: 6px 6px 2px;
    border: 0;
    background: transparent;
    color: var(--color-ink);
    font: inherit;
    font-size: 14px;
    line-height: 1.5;
    outline: none;
    box-sizing: border-box;
  }
  .skill-suggestions, .skill-tags { display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px; }
  .skill-suggestions button, .skill-tags button { padding: 4px 8px; border-radius: 999px; border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-ink); cursor: pointer; }
  .skill-suggestions span { color: var(--color-ink-subtle); }
  .notice { margin: 0 auto 4px; max-width: var(--chat-max-w, 860px); width: 100%; padding: 6px 14px; color: var(--color-ink-subtle); font-size: 12.5px; }
  @media (max-width: 760px) { .chat-layout { grid-template-columns: 150px minmax(0, 1fr); } .messages, header { padding: 14px; } .composer { padding: 8px 12px 12px; } }
</style>
