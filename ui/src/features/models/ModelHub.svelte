<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../../api'
  import { hwStore, uiMode, toggleUiMode, setTab } from '../../store.svelte'
  import { t } from '../../i18n.svelte'
  import type { CatalogEntry, LocalModelItem, ModelDownloadStatus } from '../../types'

  // ─── State ───────────────────────────────────────────────────────────────────
  let catalog = $state<CatalogEntry[]>([])
  let loading = $state(true)
  let error   = $state<string | null>(null)

  // Download state per model
  let downloads = $state<Record<string, { status: 'idle' | 'downloading' | 'done' | 'error'; msg: string }>>({})

  // Custom download fields
  let customRepo = $state('')
  let customPurpose = $state<'inference' | 'finetune' | 'merge'>('inference')
  let customDestDir = $state('')
  let serveModelPath = $state('')
  let serverState = $state<'idle' | 'starting' | 'running' | 'error'>('idle')
  let serverMessage = $state('')
  let nativeAdapter = $state('auto')
  let expertFormat = $state('auto')
  let indexingNative = $state(false)
  let generatingOracle = $state(false)
  let modelAllowUnverified = $state(false)
  let inferencePlan = $state<Record<string, unknown> | null>(null)
  let planning = $state(false)
  let settingsPath = $state('')
  let modelContext = $state<number | undefined>(undefined)
  let modelTemperature = $state<number | undefined>(undefined)
  let modelVram = $state<number | undefined>(undefined)
  let modelGuardrail = $state<number | undefined>(undefined)
  let modelCache = $state<'inherit' | 'fp16' | 'q8_0' | 'q4_0'>('inherit')
  let modelCpuKv = $state<'inherit' | 'on' | 'off'>('inherit')
  let modelFlash = $state<'inherit' | 'on' | 'off'>('inherit')
  let modelIdleTtl = $state<number | undefined>(undefined)
  let modelDraftPath = $state('')
  let settingsBusy = $state(false)
  let localModels = $state<LocalModelItem[]>([])
  let serveAttempt = 0

  // Filter & Pagination
  let filterTag = $state('all')
  let catalogQuery = $state('')
  let showAllTags = $state(false)
  let currentPage = $state(1)
  let pageSize = $state(12)

  // Storage Location & Quantization
  let defaultDestDir = $state('')
  let selectedDestDir = $state('')
  let selectedQuant = $state('auto')

  const activeDestDir = $derived(selectedDestDir || defaultDestDir)

  let liveStatus = $state<ModelDownloadStatus | null>(null)

  // ─── Load catalog & settings on mount ──────────────────────────────────────
  onMount(() => {
    api.getSettings().then(s => {
      if (s.hf_cache_dir) defaultDestDir = s.hf_cache_dir
    }).catch((e) => {
      console.error('Failed to load settings', e)
    })

    api.listLocalModels().then(items => {
      localModels = items ?? []
    }).catch(() => {
      localModels = []
    })

    const interval = setInterval(async () => {
      try {
        const st = await api.getDownloadStatus(activeDestDir)
        if (st) {
          liveStatus = st
          if (st.status === 'completed') {
            downloads[st.repo_id] = { status: 'done', msg: `Verified download completed in ${activeDestDir}` }
          } else if (st.status === 'error') {
            downloads[st.repo_id] = { status: 'error', msg: st.error || 'Download verification failed' }
          } else {
            downloads[st.repo_id] = { status: 'downloading', msg: `Downloading and verifying in ${activeDestDir}` }
          }
        }
      } catch (e) {
        console.error('Failed to poll download status', e)
      }
    }, 1500)

    const serverInterval = setInterval(async () => {
      if (serverState !== 'running') return
      const attempt = serveAttempt
      try {
        const running = await api.chatServerRunning()
        if (!running && attempt === serveAttempt && serverState === 'running') {
          serverState = 'idle'
          serverMessage = 'Model server stopped. Start it again when needed.'
        }
      } catch (e) { console.error('Failed to check model server', e) }
    }, 5000)

    return () => { clearInterval(interval); clearInterval(serverInterval) }
  })

  $effect(() => {
    api.listCatalog()
      .then(c => { catalog = c; loading = false })
      .catch(e => { error = String(e); loading = false })
  })

  async function pickStorageDir() {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({ directory: true, title: t('models.downloadFolderPickerTitle') })
      if (typeof picked === 'string') {
        selectedDestDir = picked
      }
    } catch (e) {
      console.error('Folder picker failed', e)
    }
  }

  async function pickServeModel(kind: 'file' | 'directory') {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        directory: kind === 'directory',
        multiple: false,
        title: kind === 'directory' ? t('models.serveFolderPickerTitle') : t('models.serveFilePickerTitle'),
        filters: kind === 'file' ? [{ name: 'GGUF model', extensions: ['gguf'] }] : undefined,
      })
      if (typeof picked === 'string') {
        serveModelPath = picked
        await loadModelSettings()
      }
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  async function pickDraftModel() {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({ multiple: false, title: 'Select draft GGUF model', filters: [{ name: 'GGUF model', extensions: ['gguf'] }] })
      if (typeof picked === 'string') modelDraftPath = picked
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  async function inspectPlan(exportRuntimes = false) {
    if (!serveModelPath.trim()) return
    planning = true
    try {
      inferencePlan = await api.planInference(serveModelPath.trim(), undefined, exportRuntimes, modelAllowUnverified)
      const compatible = inferencePlan.compatible === true
      serverState = compatible ? 'idle' : 'error'
      const reasons = Array.isArray(inferencePlan.reasons) ? inferencePlan.reasons.join(' ') : ''
      serverMessage = compatible ? t('models.planCompatible') : `${t('models.planRejected')} ${reasons}`
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
      inferencePlan = null
    } finally {
      planning = false
    }
  }

  async function startServer() {
    if (!serveModelPath.trim()) return
    const target = serveModelPath.trim()
    serverState = 'starting'
    const attempt = ++serveAttempt
    serverMessage = t('models.serverPreflight')
    try {
      await api.startChatServer(target, undefined, undefined, undefined, modelAllowUnverified)
      if (attempt !== serveAttempt) return
      serverState = 'running'
      serverMessage = t('models.serverRunning')
      try {
        const plan = await api.planInference(target, undefined, false, modelAllowUnverified)
        if (attempt === serveAttempt) inferencePlan = plan
      } catch {
        // Keep a running server clean even if plan introspection is unavailable.
      }
    } catch (e) {
      if (attempt !== serveAttempt) return
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  async function generateNativeOracle() {
    if (!serveModelPath.trim()) return
    generatingOracle = true
    serverState = 'starting'
    serverMessage = t('models.generatingOracle')
    try {
      const result = await api.generateMoeOracle(serveModelPath.trim())
      serverState = 'idle'
      serverMessage = `${t('models.oracleReady')} ${result.oracle}`
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    } finally {
      generatingOracle = false
    }
  }

  async function buildNativeIndex() {
    if (!serveModelPath.trim()) return
    indexingNative = true
    serverState = 'starting'
    serverMessage = t('models.nativeIndexing')
    try {
      const result = await api.buildMoeIndex(
        serveModelPath.trim(),
        nativeAdapter,
        expertFormat,
      )
      serverState = 'idle'
      serverMessage = `${t('models.nativeIndexReady')} ${result.experts_indexed}`
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    } finally {
      indexingNative = false
    }
  }

  async function stopServer() {
    ++serveAttempt
    try {
      await api.stopChatServer()
      serverState = 'idle'
      serverMessage = t('models.serverStopped')
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  async function loadModelSettings() {
    const path = serveModelPath.trim()
    if (!path || settingsBusy) return
    settingsBusy = true
    settingsPath = ''
    try {
      const saved = await api.getModelSettings(path)
      if (serveModelPath.trim() !== path) return
      modelContext = saved.context_length ?? undefined
      modelTemperature = saved.temperature ?? undefined
      modelVram = saved.vram_limit_mb ?? undefined
      modelGuardrail = saved.vram_guardrail_mb ?? undefined
      modelCache = saved.kv_cache_quant ?? 'inherit'
      modelCpuKv = saved.cpu_kv_cache === null ? 'inherit' : saved.cpu_kv_cache ? 'on' : 'off'
      modelFlash = saved.flash_attention === null ? 'inherit' : saved.flash_attention ? 'on' : 'off'
      modelIdleTtl = saved.idle_ttl_minutes ?? undefined
      modelDraftPath = saved.draft_model_path ?? ''
      modelAllowUnverified = Boolean(saved.allow_unverified)
      settingsPath = path
    } catch (error) {
      serverState = 'error'
      serverMessage = String(error)
    } finally { settingsBusy = false }
  }

  async function saveModelSettings() {
    const path = serveModelPath.trim()
    if (!path || settingsPath !== path || settingsBusy) return
    settingsBusy = true
    try {
      await api.setModelSettings(path, {
        context_length: modelContext ?? null, temperature: modelTemperature ?? null, vram_limit_mb: modelVram ?? null, vram_guardrail_mb: modelGuardrail ?? null,
        kv_cache_quant: modelCache === 'inherit' ? null : modelCache,
        cpu_kv_cache: modelCpuKv === 'inherit' ? null : modelCpuKv === 'on',
        flash_attention: modelFlash === 'inherit' ? null : modelFlash === 'on',
        idle_ttl_minutes: modelIdleTtl ?? null,
        draft_model_path: modelDraftPath.trim() || null,
        allow_unverified: modelAllowUnverified,
      })
      if (serverState !== 'running') serverState = 'idle'
      serverMessage = 'Model settings saved. They apply the next time the server starts.'
    } catch (error) {
      serverState = 'error'
      serverMessage = String(error)
    } finally { settingsBusy = false }
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────────
  const usableCatalog = $derived(
    catalog.filter(c => c.downloadable !== false && !c.id.startsWith('org/') && c.format !== 'mlx')
  )

  const allTags = $derived(
    ['all', ...Array.from(new Set(usableCatalog.flatMap(c => c.tags)))]
  )

  const guidedCategoryTags: { id: string; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'fits-pc', label: 'Fits my PC' },
    { id: 'recommended', label: 'Recommended' },
    { id: 'small', label: 'Small & Fast' },
    { id: 'chat', label: 'Chat' },
    { id: 'coding', label: 'Coding' },
    { id: 'reasoning', label: 'Reasoning' },
    { id: 'moe', label: 'MoE' },
    { id: 'multimodal', label: 'Vision' },
  ]

  const visibleGuidedTags = $derived(
    guidedCategoryTags.filter(item =>
      item.id === 'all' ||
      item.id === 'fits-pc' ||
      (item.id === 'recommended' ? usableCatalog.some(c => c.recommended) : usableCatalog.some(c => c.tags.includes(item.id)))
    )
  )

  const visibleTags = $derived(showAllTags ? allTags : allTags.slice(0, 12))

  const filtered = $derived(
    usableCatalog
      .filter(c => {
        if (filterTag === 'all') return true
        if (filterTag === 'fits-pc') return fitsVRAM(c) !== 'large'
        if (filterTag === 'recommended') return Boolean(c.recommended)
        return c.tags.includes(filterTag)
      })
      .filter(c => {
        const q = catalogQuery.trim().toLowerCase()
        if (!q) return true
        return c.id.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.tags.some(t => t.toLowerCase().includes(q))
      })
  )

  const totalPages = $derived(Math.max(1, Math.ceil(filtered.length / pageSize)))
  const safePage = $derived(Math.min(Math.max(1, currentPage), totalPages))
  const paged = $derived(filtered.slice((safePage - 1) * pageSize, safePage * pageSize))
  const pageNumbers = $derived.by(() => {
    const pages: (number | 'ellipsis')[] = []
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i)
      return pages
    }
    pages.push(1)
    if (safePage > 3) pages.push('ellipsis')
    for (let i = Math.max(2, safePage - 1); i <= Math.min(totalPages - 1, safePage + 1); i++) pages.push(i)
    if (safePage < totalPages - 2) pages.push('ellipsis')
    pages.push(totalPages)
    return pages
  })

  function shortArch(arch: string | undefined): string {
    if (!arch) return ''
    return arch.replace(/(ForCausalLM|ForConditionalGeneration)$/i, '')
  }

  function resetCatalogPage() {
    currentPage = 1
  }

  const vramGb = $derived(
    hwStore.info?.vram_mb != null ? Math.round(hwStore.info.vram_mb / 1024) : 0
  )

  const moeCapacity = $derived.by(() => {
    const ramMb = hwStore.info?.ram_mb
    const vramMb = hwStore.info?.vram_mb ?? 0
    const known = ramMb != null && ramMb > 0
    const vram = Math.round(vramMb / 1024)
    const ram = ramMb != null ? Math.round(ramMb / 1024) : 0
    if (!known) {
      return {
        known: false,
        vramGb: vram,
        ramGb: 0,
        maxVramOnly4BitB: 0,
        maxHybrid4BitB: 0,
        helixMaxTierLabel: 'Unknown hardware',
        helixEnvelopeNote: 'Hardware RAM could not be detected; check Settings or switch to Advanced mode.',
      }
    }
    const maxVramOnly4BitB = vram >= 3
      ? Math.max(1, Math.round(Math.max(vram - 1.5, vram * 0.8) / 0.56))
      : 0
    const hybridBudgetGb = Math.max(0, vram * 0.8 + ram * 0.7 - 1.5)
    const maxHybrid4BitB = Math.max(1, Math.round(hybridBudgetGb / 0.56))

    let helixMaxTierLabel = 'Up to <30B MoE'
    let helixEnvelopeNote = '8 GB+ RAM supports compact MoE models; >=16 GB RAM is needed for 30B–99B Helix SSD streaming.'
    if (ram >= 60 && vram >= 4) {
      helixMaxTierLabel = 'Up to 500B–700B+ MoE'
      helixEnvelopeNote = 'Meets 64 GB RAM frontier tier; requires fast PCIe Gen4/Gen5 NVMe SSD (>=5000 MB/s) for streaming.'
    } else if (ram >= 31 && vram >= 4) {
      helixMaxTierLabel = 'Up to ~480B MoE'
      helixEnvelopeNote = 'Meets 32 GB RAM large-MoE tier (100B–480B) with fast NVMe SSD; 500B+ recommends 64 GB RAM.'
    } else if (ram >= 15 && vram >= 4) {
      helixMaxTierLabel = 'Up to ~99B MoE'
      helixEnvelopeNote = 'Meets 16 GB RAM standard-MoE tier (30B–99B) with NVMe SSD; 100B+ MoE requires >=32 GB RAM.'
    } else if (ram < 8) {
      helixMaxTierLabel = 'Small quantized models (<8B)'
      helixEnvelopeNote = 'Under 8 GB RAM detected; large disk-streamed MoE models are not advised.'
    }

    return {
      known: true,
      vramGb: vram,
      ramGb: ram,
      maxVramOnly4BitB,
      maxHybrid4BitB,
      helixMaxTierLabel,
      helixEnvelopeNote,
    }
  })

  function formatBillions(count: unknown): string {
    if (typeof count !== 'number' || !Number.isFinite(count) || count <= 0) return ''
    const b = count / 1e9
    return b >= 100 ? `${b.toFixed(0)}B` : `${b.toFixed(1)}B`
  }

  function formatGiB(bytes: unknown): string {
    if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes < 0) return '0 GB'
    return `${(bytes / (1024 ** 3)).toFixed(1)} GB`
  }

  function simpleFitsBadge(entry: CatalogEntry): { label: string; cls: string } {
    const fit = fitsVRAM(entry)
    if (fit === 'yes') return { label: 'Fits GPU', cls: 'badge-green' }
    if (fit === 'streaming') return { label: 'Fits w/ RAM or 4-bit', cls: 'badge-amber' }
    return { label: 'Too large for this PC', cls: 'badge-red' }
  }

  function fitsVRAM(entry: CatalogEntry): 'yes' | 'streaming' | 'large' {
    const ramGb = hwStore.info?.ram_mb != null ? Math.round(hwStore.info.ram_mb / 1024) : 0
    if (vramGb <= 0 && ramGb <= 0) return 'large'
    const gpuCeiling = vramGb > 0 ? Math.max(1, vramGb * 0.85) : 0
    const hybridCeiling = vramGb * 0.8 + ramGb * 0.7
    const isMoe = entry.tags?.includes('moe') || (entry.moe_active_params != null && entry.moe_active_params > 0)

    if (entry.format === 'gguf') {
      if (vramGb > 0 && entry.size_gb <= Math.max(0.5, vramGb - 1.5)) return 'yes'
      if (entry.size_gb <= hybridCeiling) return 'streaming'
      return 'large'
    }

    if (vramGb > 0 && entry.size_gb <= gpuCeiling) return 'yes'
    const est4BitGb = entry.size_gb * 0.30
    if (entry.size_gb <= hybridCeiling || (ramGb > 0 && est4BitGb <= hybridCeiling)) {
      return 'streaming'
    }
    if (isMoe && vramGb >= 4 && ramGb >= 16) {
      const totalB = entry.param_count ? entry.param_count / 1e9 : entry.size_gb / 2
      if (ramGb >= 32 || totalB < 100) return 'streaming'
    }
    return 'large'
  }

  function fitsBadge(entry: CatalogEntry): { label: string; cls: string } {
    const fit = fitsVRAM(entry)
    if (fit === 'yes')        return { label: t('models.badgeFitsVram'),     cls: 'badge-green' }
    if (fit === 'streaming')  return { label: t('models.badgeExpertStream'), cls: 'badge-amber' }
    return                          { label: t('models.badgeLargeModel'),    cls: 'badge-red'   }
  }

  async function handleCancelDownload() {
    try {
      await api.cancelDownload(activeDestDir)
      liveStatus = null
      downloads = {}
    } catch (e) {
      console.error('Error cancelling download:', e)
    }
  }

  async function downloadCatalog(entry: CatalogEntry) {
    if (entry.downloadable === false || (entry.alerts ?? []).some(a => a.blocks_download)) {
      downloads[entry.id] = { status: 'error', msg: t('models.notDownloadable') }
      return
    }
    const dest = activeDestDir
    downloads[entry.id] = { status: 'downloading', msg: `Queueing download (${selectedQuant}) to ${dest}…` }
    try {
      await api.downloadModel(entry.id, 'inference', dest, selectedQuant)
      downloads[entry.id] = { status: 'downloading', msg: `Download and verification started in ${dest}` }
    } catch (e) {
      downloads[entry.id] = { status: 'error', msg: String(e) }
    }
  }

  async function downloadCustom() {
    if (!customRepo.trim()) return
    const id = customRepo.trim()
    const match = catalog.find(c => c.id === id)
    if (!match) {
      downloads[id] = { status: 'error', msg: t('models.catalogOnly') }
      return
    }
    if (match.downloadable === false || (match.alerts ?? []).some(a => a.blocks_download)) {
      downloads[id] = { status: 'error', msg: t('models.notDownloadable') }
      return
    }
    const dest = customDestDir || activeDestDir
    downloads[id] = { status: 'downloading', msg: `Queueing download (${selectedQuant}) to ${dest}…` }
    try {
      await api.downloadModel(id, customPurpose, dest, selectedQuant)
      downloads[id] = { status: 'downloading', msg: `Download and verification started in ${dest}` }
      customRepo = ''
    } catch (e) {
      downloads[id] = { status: 'error', msg: String(e) }
    }
  }
</script>

<div class="hub">
  <!-- Header -->
  <div class="hub-header">
    <div class="hub-title-row">
      <span class="hub-icon"><i class="bi bi-download"></i></span>
      <div>
        <h1 class="hub-title">{uiMode.advanced ? t('models.title') : 'Models'}</h1>
        <p class="hub-sub">
          {uiMode.advanced ? t('models.subtitle') : 'Download a model that fits your computer or start one to chat.'}
        </p>
      </div>
    </div>
    {#if hwStore.info}
      <div class="hw-chips">
        <span class="chip"><i class="bi bi-gpu-card"></i> {vramGb} GB {uiMode.advanced ? 'VRAM' : 'GPU'}</span>
        <span class="chip"><i class="bi bi-memory"></i> {hwStore.info.ram_mb != null ? Math.round(hwStore.info.ram_mb / 1024) + ' GB RAM' : 'RAM unknown'}</span>
        {#if moeCapacity.known}
          <span class="chip chip-brand" title={moeCapacity.helixEnvelopeNote}>
            <i class="bi bi-cpu"></i> MoE limit: ~{moeCapacity.maxHybrid4BitB}B fast · {moeCapacity.helixMaxTierLabel}
          </span>
        {:else if uiMode.advanced}
          <span class="chip chip-brand"><i class="bi bi-shield-check"></i> {t('models.expertStreamingOn')}</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- Hardware Capacity & Max MoE Size Summary -->
  {#if moeCapacity.known}
    <div class="card capacity-card" id="moe-capacity-card">
      <div class="capacity-grid">
        <div class="capacity-stat">
          <span class="capacity-label">{uiMode.advanced ? 'GPU-only (4-bit)' : 'Fastest (GPU memory)'}</span>
          <span class="capacity-value">Up to ~{moeCapacity.maxVramOnly4BitB}B</span>
        </div>
        <div class="capacity-stat">
          <span class="capacity-label">{uiMode.advanced ? 'VRAM + RAM hybrid (4-bit)' : 'In memory (GPU + RAM)'}</span>
          <span class="capacity-value">Up to ~{moeCapacity.maxHybrid4BitB}B</span>
        </div>
        <div class="capacity-stat">
          <span class="capacity-label">Helix Engine MoE (SSD streaming)</span>
          <span class="capacity-value">{moeCapacity.helixMaxTierLabel}</span>
        </div>
      </div>
      <p class="capacity-note">{moeCapacity.helixEnvelopeNote} Estimates assume ~4-bit weights; actual fit and tokens/sec depend on quantization, context length, and NVMe SSD speed.</p>
    </div>
  {/if}

  {#if !uiMode.advanced}
    <!-- ── Guided & simple: Start a local model for Chat ─────────────────── -->
    <div class="card serving-section" id="simple-serving-card">
      <div class="simple-serving-head">
        <div>
          <div class="card-title" style="margin-bottom:4px"><i class="bi bi-play-circle"></i> Run a model for Chat</div>
          <p class="serving-copy">Pick a downloaded model on your computer and start it so you can talk to it in Chat.</p>
        </div>
        <button type="button" class="btn btn-ghost btn-sm" onclick={toggleUiMode}>
          Need precision quant, custom IDs, or expert index? Switch to Advanced mode
        </button>
      </div>

      {#if localModels.length > 0}
        <select
          class="select"
          aria-label="Choose a model on this computer"
          value={localModels.some(lm => lm.path === serveModelPath) ? serveModelPath : ''}
          onchange={(e) => {
            const val = (e.currentTarget as HTMLSelectElement).value
            if (val) { serveModelPath = val; void loadModelSettings() }
          }}
        >
          <option value="">-- Choose from downloaded models on this PC --</option>
          {#each localModels as lm}
            <option value={lm.path}>{lm.name} ({lm.size_gb} GB)</option>
          {/each}
        </select>
      {/if}

      <div class="serving-row">
        <input
          id="serve-model-path"
          class="input serving-path"
          placeholder="Choose a model file or folder on your computer…"
          bind:value={serveModelPath}
        />
        <button class="btn btn-secondary btn-sm" onclick={() => pickServeModel('file')}>
          <i class="bi bi-file-earmark"></i> Pick file
        </button>
        <button class="btn btn-secondary btn-sm" onclick={() => pickServeModel('directory')}>
          <i class="bi bi-folder2-open"></i> Pick folder
        </button>
        <button
          id="start-model-server"
          class="btn btn-primary"
          onclick={startServer}
          disabled={!serveModelPath.trim() || serverState === 'starting'}
        >
          <i class="bi bi-play-fill"></i>
          {serverState === 'starting' ? 'Starting…' : 'Start model'}
        </button>
        {#if serverState === 'running' || serverState === 'starting'}
          <button class="btn btn-secondary" onclick={stopServer}>
            <i class="bi bi-stop-fill"></i> Stop
          </button>
        {/if}
      </div>

      {#if inferencePlan}
        {@const estimates = (inferencePlan.estimates ?? {}) as Record<string, unknown>}
        {@const helix = (estimates.helix_resource_plan ?? null) as Record<string, unknown> | null}
        {@const totalB = formatBillions(estimates.total_params)}
        {@const activeB = formatBillions(estimates.active_params_per_token)}
        <div class="plan-card" id="active-runtime-summary">
          <div class="plan-row">
            <strong>{inferencePlan.backend === 'sytra_moe' ? 'Helix Engine MoE Server' : String(inferencePlan.backend ?? 'Local Server')}</strong>
            {#if totalB}<span>Size: {totalB}{activeB ? ` (${activeB} active per token)` : ''}</span>{/if}
            {#if estimates.is_moe}<span class="badge badge-amber">MoE</span>{/if}
          </div>
          {#if helix}
            <div class="plan-row">
              <span>GPU cache: {Number(helix.tier0_vram_attention_mb ?? 0) + Number(helix.tier0_vram_hot_expert_cache_mb ?? 0)} MB</span>
              <span>RAM pool: {String(helix.tier1_pinned_ram_pool_mb ?? '—')} MB</span>
              <span>SSD streaming: {formatGiB(helix.tier2_nvme_weight_bytes)}</span>
              {#if helix.effective_estimated_tps}<span>Est. speed: ~{String(helix.effective_estimated_tps)} tok/s (hardware-dependent)</span>{/if}
            </div>
          {:else}
            <div class="plan-row">
              <span>Peak GPU memory: {String(estimates.peak_vram_mb ?? '—')} MB</span>
              <span>Peak RAM: {String(estimates.peak_ram_mb ?? '—')} MB</span>
            </div>
          {/if}
        </div>
      {/if}

      {#if serverMessage}
        <div class="server-status" class:server-ok={serverState === 'running'} class:server-error={serverState === 'error'}>
          <i class={serverState === 'running' ? 'bi bi-check-circle-fill' : serverState === 'error' ? 'bi bi-exclamation-triangle-fill' : 'bi bi-info-circle'}></i>
          <span>{serverState === 'running' ? 'Model is running and ready for Chat.' : serverMessage}</span>
          {#if serverState === 'running'}
            <button type="button" class="btn btn-primary btn-sm" style="margin-left:auto" onclick={() => setTab('chat')}>
              <i class="bi bi-chat-dots"></i> Open Chat
            </button>
          {/if}
        </div>
      {/if}
    </div>
  {:else}
    <!-- ── Advanced: Storage, Quantization, Local Serving, MoE Index & Custom ID ── -->
    <div class="card storage-section">
      <div class="storage-row">
        <div class="storage-info">
          <span class="storage-label"><i class="bi bi-folder2"></i> {t('models.storageDestination')}</span>
          <span class="storage-path">{activeDestDir}</span>
        </div>
        <div class="storage-controls">
          <div class="quant-select-group">
            <label for="quant-select-hdr" class="quant-select-label"><i class="bi bi-sliders"></i> {t('models.precisionQuant')}</label>
            <select id="quant-select-hdr" class="select select-sm quant-dropdown" bind:value={selectedQuant}>
              <option value="auto">Auto (Best for {vramGb} GB VRAM)</option>
              <option value="Q4_K_M">Q4_K_M (Balanced 4-bit - Recommended)</option>
              <option value="Q5_K_M">Q5_K_M (High Precision 5-bit)</option>
              <option value="Q8_0">Q8_0 (8-bit High Quality)</option>
              <option value="FP16">FP16 / BF16 (Full Unquantized Weights)</option>
            </select>
          </div>
          <button class="btn btn-secondary btn-sm" onclick={pickStorageDir}>
            <i class="bi bi-folder2-open"></i> {t('models.browseFolder')}
          </button>
        </div>
      </div>
    </div>

    <div class="card serving-section">
      <div class="card-title"><i class="bi bi-gpu-card"></i> {t('models.localServing')}</div>
      <p class="serving-copy">{t('models.localServingDesc')}</p>
      <div class="serving-row">
        <input
          id="serve-model-path"
          class="input serving-path"
          placeholder={t('models.servePathPlaceholder')}
          bind:value={serveModelPath}
        />
        <button class="btn btn-secondary btn-sm" onclick={() => pickServeModel('file')}>
          <i class="bi bi-file-earmark-binary"></i> GGUF
        </button>
        <button class="btn btn-secondary btn-sm" onclick={() => pickServeModel('directory')}>
          <i class="bi bi-folder2-open"></i> SafeTensors / Sytra
        </button>
        <button
          class="btn btn-secondary"
          onclick={() => inspectPlan(false)}
          disabled={!serveModelPath.trim() || planning}
        >
          <i class="bi bi-clipboard-data"></i> {planning ? '…' : t('models.inspectPlan')}
        </button>
        <button
          class="btn btn-secondary"
          onclick={() => inspectPlan(true)}
          disabled={!serveModelPath.trim() || planning}
        >
          <i class="bi bi-box-arrow-up"></i> {t('models.exportRuntimes')}
        </button>
        <button
          id="start-model-server"
          class="btn btn-primary"
          onclick={startServer}
          disabled={!serveModelPath.trim() || serverState === 'starting'}
        >
          <i class="bi bi-play-fill"></i>
          {serverState === 'starting' ? t('models.serverStarting') : t('models.startServer')}
        </button>
        <button
          class="btn btn-secondary"
          onclick={stopServer}
          disabled={serverState !== 'running' && serverState !== 'starting'}
        >
          <i class="bi bi-stop-fill"></i> {t('models.stopServer')}
        </button>
      </div>
      <details class="model-preferences">
        <summary>Per-model inference settings</summary>
        <p class="serving-copy">Load a model's settings, then save changes before starting its server. Empty fields inherit global settings.</p>
        <button class="btn btn-secondary btn-sm" onclick={loadModelSettings} disabled={!serveModelPath.trim() || settingsBusy}>Load settings</button>
        {#if settingsPath && settingsPath === serveModelPath.trim()}
          <div class="preference-grid">
            <div class="field"><label class="field-label" for="model-temperature">Chat temperature</label><input id="model-temperature" class="input" type="number" min="0" max="2" step="0.05" bind:value={modelTemperature} placeholder="Inherit global" /><span class="field-hint">Applied to new messages in the built-in chat after the server starts.</span></div>
            <div class="field"><label class="field-label" for="model-context">Context tokens</label><input id="model-context" class="input" type="number" min="1" max="2097152" bind:value={modelContext} /></div>
            <div class="field"><label class="field-label" for="model-vram">VRAM limit (MB)</label><input id="model-vram" class="input" type="number" min="1" bind:value={modelVram} /></div>
            <div class="field"><label class="field-label" for="model-guardrail">Free VRAM reserve (MB)</label><input id="model-guardrail" class="input" type="number" min="0" bind:value={modelGuardrail} /><span class="field-hint">Optional; NVIDIA llama.cpp only.</span></div>
            <div class="field"><label class="field-label" for="model-kv">KV cache</label><select id="model-kv" class="select" bind:value={modelCache}><option value="inherit">Global default</option><option value="fp16">FP16</option><option value="q8_0">Q8_0</option><option value="q4_0">Q4_0</option></select></div>
            <div class="field"><label class="field-label" for="model-cpu-kv">CPU KV cache</label><select id="model-cpu-kv" class="select" bind:value={modelCpuKv}><option value="inherit">Global default</option><option value="on">On</option><option value="off">Off</option></select></div>
            <div class="field"><label class="field-label" for="model-flash">Flash attention</label><select id="model-flash" class="select" bind:value={modelFlash}><option value="inherit">Global default</option><option value="on">On</option><option value="off">Off</option></select></div>
            <div class="field"><label class="field-label" for="model-idle">Unload after idle (minutes)</label><input id="model-idle" class="input" type="number" min="0" max="10080" step="1" bind:value={modelIdleTtl} placeholder="30" /><span class="field-hint">Default 30; 0 keeps the server running.</span></div>
            <div class="field"><label class="field-label" for="model-draft">Draft GGUF path</label><input id="model-draft" class="input input-mono" bind:value={modelDraftPath} placeholder="Optional local model path" /><button type="button" class="btn btn-secondary btn-sm" onclick={pickDraftModel}>Choose draft model</button><span class="field-hint">Requires a matching tokenizer; draft runs on CPU and may not improve speed.</span></div>
          </div>
          <button class="btn btn-secondary btn-sm" onclick={saveModelSettings} disabled={settingsBusy}>Save model settings</button>
        {/if}
      </details>
      <div class="native-index-row">
        <span class="native-index-label">{t('models.nativeMoeIndex')}</span>
        <select class="select" bind:value={nativeAdapter} aria-label={t('models.nativeAdapter')}>
          <option value="auto">Auto-detect</option>
          <option value="sytra-glm52">GLM-5.2</option>
          <option value="sytra-kimi-k2.7-code">Kimi K2.7 Code</option>
          <option value="sytra-kimi-k3">Kimi K3</option>
          <option value="sytra-inkling">Inkling</option>
          <option value="sytra-deepseek-v3">DeepSeek V2/V3</option>
          <option value="sytra-qwen3-moe">Qwen3 MoE</option>
          <option value="sytra-qwen2-moe">Qwen2 MoE</option>
          <option value="sytra-mixtral">Mixtral</option>
          <option value="sytra-olmoe">OLMoE</option>
          <option value="sytra-dbrx">DBRX</option>
          <option value="sytra-granite-moe">Granite MoE</option>
          <option value="sytra-arctic">Arctic</option>
          <option value="sytra-minimax-moe">MiniMax MoE</option>
          <option value="sytra-generic-moe">Generic storage-only</option>
        </select>
        <select class="select" bind:value={expertFormat} aria-label={t('models.expertFormat')}>
          <option value="auto">Auto-detect</option>
          <option value="int4_group">INT4 group</option>
          <option value="packed_int4_group32">Packed INT4 group-32</option>
          <option value="packed_int4_group">Packed INT4 group-128 / auto</option>
          <option value="fp8_e4m3">FP8 E4M3</option>
          <option value="nvfp4">NVFP4</option>
          <option value="mxfp4">MXFP4</option>
          <option value="int8">INT8</option>
          <option value="bf16">BF16</option>
          <option value="f16">FP16</option>
          <option value="custom">Custom</option>
        </select>
        <button
          class="btn btn-secondary"
          onclick={buildNativeIndex}
          disabled={!serveModelPath.trim() || indexingNative}
        >
          <i class="bi bi-diagram-3"></i>
          {indexingNative ? t('models.nativeIndexing') : t('models.buildNativeIndex')}
        </button>
        <button
          class="btn btn-secondary"
          onclick={generateNativeOracle}
          disabled={!serveModelPath.trim() || generatingOracle}
        >
          <i class="bi bi-shield-check"></i>
          {generatingOracle ? t('models.generatingOracle') : t('models.generateOracle')}
        </button>
        <label class="native-unverified-toggle" title={t('models.allowUnverifiedHint')}>
          <input id="native-allow-unverified" type="checkbox" bind:checked={modelAllowUnverified} />
          <span>{t('models.allowUnverified')}</span>
        </label>
        <span class="native-index-hint">{t('models.nativeIndexHint')}</span>
      </div>
      {#if inferencePlan}
        {@const estimates = (inferencePlan.estimates ?? {}) as Record<string, unknown>}
        {@const helix = (estimates.helix_resource_plan ?? null) as Record<string, unknown> | null}
        {@const totalB = formatBillions(estimates.total_params)}
        {@const activeB = formatBillions(estimates.active_params_per_token)}
        <div class="plan-card" id="active-runtime-summary">
          <div class="plan-row">
            <span>{String(estimates.architecture ?? inferencePlan.backend)}</span>
            <span>{String(estimates.quantization ?? '')}</span>
            {#if totalB}<span>Total: {totalB}{activeB ? ` · Active: ${activeB}/tok` : ''}</span>{/if}
            {#if estimates.is_moe}<span class="badge badge-amber">MoE</span>{/if}
            {#if moeCapacity.known}
              <span class="badge badge-neutral">Host ceiling: ~{moeCapacity.maxHybrid4BitB}B memory · {moeCapacity.helixMaxTierLabel}</span>
            {/if}
          </div>
          <div class="plan-row">
            <span>{t('models.gpuLayers')}: {String(estimates.gpu_layers ?? '—')}</span>
            <span>{t('models.peakVram')}: {String(estimates.peak_vram_mb ?? '—')} MB</span>
            <span>{t('models.peakRam')}: {String(estimates.peak_ram_mb ?? '—')} MB</span>
          </div>
          {#if helix}
            <div class="plan-row">
              <span>Helix Engine: {String(helix.execution_strategy ?? helix.profile ?? 'sytra-helix')}</span>
              <span>VRAM attn+hot: {Number(helix.tier0_vram_attention_mb ?? 0) + Number(helix.tier0_vram_hot_expert_cache_mb ?? 0)} MB</span>
              <span>Pinned RAM: {String(helix.tier1_pinned_ram_pool_mb ?? '—')} MB</span>
              <span>NVMe streamed: {formatGiB(helix.tier2_nvme_weight_bytes)}</span>
              {#if helix.effective_estimated_tps}<span>Est. decode: ~{String(helix.effective_estimated_tps)} tok/s</span>{/if}
              <span class="badge {helix.meets_minimum_envelope ? 'badge-green' : 'badge-red'}">
                {helix.meets_minimum_envelope ? 'Within host MoE envelope' : 'Exceeds host minimum RAM/VRAM'}
              </span>
            </div>
          {/if}
          {#if Array.isArray(inferencePlan.command) && inferencePlan.command.length}
            <code class="plan-cmd">{(inferencePlan.command as string[]).join(' ')}</code>
          {/if}
          {#if Array.isArray(inferencePlan.warnings) && inferencePlan.warnings.length}
            <div class="risk-list">
              <div class="risk-title">{t('models.planWarnings')}</div>
              {#each inferencePlan.warnings as warning}
                <div class="risk-item risk-warning">{String(warning)}</div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
      {#if serverMessage}
        <div class="server-status" class:server-ok={serverState === 'running'} class:server-error={serverState === 'error'}>
          <i class={serverState === 'running' ? 'bi bi-check-circle-fill' : serverState === 'error' ? 'bi bi-exclamation-triangle-fill' : 'bi bi-info-circle'}></i>
          <span>{serverMessage}</span>
          {#if serverState === 'running'}
            <code>http://127.0.0.1:8080/v1</code>
          {/if}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Live Download Progress Banner -->
  {#if liveStatus && ['resolving', 'downloading', 'completed', 'error'].includes(liveStatus.status)}
    <div class="card live-progress-card" class:download-error={liveStatus.status === 'error'}>
      <div class="live-progress-header">
        <div class="live-model-info">
          <span class="spinner"></span>
          <div>
            <div class="live-model-name">
              {liveStatus.status === 'completed' ? t('models.downloaded') : liveStatus.status === 'error' ? t('live.error') : t('models.downloading')}
              {liveStatus.repo_id}
            </div>
            <div class="live-file-sub">{t('models.file')} {liveStatus.shard_index}/{liveStatus.total_shards}: {liveStatus.current_file}</div>
          </div>
        </div>
        <div class="live-stats">
          <span class="stat-badge"><i class="bi bi-speedometer2"></i> {liveStatus.speed_mbps} MB/s</span>
          <span class="stat-badge"><i class="bi bi-clock-history"></i> ETA: {liveStatus.eta_formatted}</span>
          <span class="pct-badge">{liveStatus.pct}%</span>
          {#if liveStatus.status === 'resolving' || liveStatus.status === 'downloading'}
            <button class="btn btn-red btn-sm" onclick={handleCancelDownload}>
              <i class="bi bi-x-circle-fill"></i> {t('models.cancelDownload')}
            </button>
          {/if}
        </div>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: {Math.max(liveStatus.pct, 2)}%"></div>
      </div>
      <div class="live-progress-footer">
        <span>{liveStatus.error || `${liveStatus.downloaded_gb} GB / ${liveStatus.total_gb} GB ${t('models.downloaded')}`}</span>
        <span>{t('models.destination')} {activeDestDir}</span>
      </div>
    </div>
  {/if}

  {#if uiMode.advanced}
    <!-- Custom download -->
    <div class="card custom-section">
      <div class="card-title"><i class="bi bi-cloud-arrow-down"></i> {t('models.customDownload')}</div>
      <p class="catalog-policy">{t('models.catalogOnly')}</p>
      <div class="custom-row">
        <input
          id="custom-repo-input"
          class="input"
          placeholder={t('models.repoPlaceholder')}
          bind:value={customRepo}
        />
        <select id="custom-purpose-select" class="select" bind:value={customPurpose}>
          <option value="inference">{t('models.inferenceGguf')}</option>
          <option value="finetune">{t('models.finetuneSafetensors')}</option>
          <option value="merge">{t('models.mergeSafetensors')}</option>
        </select>
        <input
          id="custom-dest-input"
          class="input input-sm"
          placeholder={t('models.destDirPlaceholder')}
          bind:value={customDestDir}
        />
        <button id="custom-download-btn" class="btn btn-primary" onclick={downloadCustom} disabled={!customRepo.trim()}>
          <i class="bi bi-cloud-arrow-down"></i> {t('models.download')}
        </button>
      </div>
      {#if downloads[customRepo.trim()]}
        <div class="dl-status" class:dl-ok={downloads[customRepo.trim()].status === 'done'} class:dl-err={downloads[customRepo.trim()].status === 'error'}>
          {downloads[customRepo.trim()].msg}
        </div>
      {/if}
    </div>
  {/if}

  <!-- Filter & pagination bar -->
  <div class="catalog-toolbar">
    <div class="filter-row">
      <input
        class="input catalog-search"
        placeholder={uiMode.advanced ? t('models.searchCatalog') : 'Search models by name or use (coding, chat, small)…'}
        bind:value={catalogQuery}
        oninput={resetCatalogPage}
      />
      {#if uiMode.advanced}
        {#each visibleTags as tag}
          <button
            id="filter-{tag}"
            class="chip"
            class:chip-active={filterTag === tag}
            onclick={() => { filterTag = tag; resetCatalogPage() }}
          >{tag}</button>
        {/each}
        {#if allTags.length > 12}
          <button type="button" class="chip" onclick={() => showAllTags = !showAllTags}>
            {showAllTags ? 'Fewer tags' : `+${allTags.length - 12} tags`}
          </button>
        {/if}
      {:else}
        {#each visibleGuidedTags as item}
          <button
            id="filter-{item.id}"
            class="chip"
            class:chip-active={filterTag === item.id}
            onclick={() => { filterTag = item.id; resetCatalogPage() }}
          >{item.label}</button>
        {/each}
      {/if}
    </div>
    <div class="pagination-bar" aria-label="Catalog pagination top">
      <span class="pagination-summary">
        {#if filtered.length === 0}
          0 models
        {:else}
          Showing {(safePage - 1) * pageSize + 1}–{Math.min(safePage * pageSize, filtered.length)} of {filtered.length} models
        {/if}
      </span>
      <div class="pagination-controls">
        <label class="page-size-label">
          <span>Per page</span>
          <select class="select quant-dropdown" aria-label="Models per page" bind:value={pageSize} onchange={resetCatalogPage}>
            <option value={12}>12</option>
            <option value={24}>24</option>
            <option value={48}>48</option>
          </select>
        </label>
      </div>
    </div>
  </div>

  <!-- Catalog grid -->
  {#if loading}
    <div class="loading-state"><span class="spinner"></span> {t('models.loadingCatalog')}</div>
  {:else if error}
    <div class="error-state"><i class="bi bi-exclamation-triangle"></i> {error}</div>
  {:else if filtered.length === 0}
    <div class="loading-state">No matching downloadable models found.</div>
  {:else}
    <div class="catalog-grid">
      {#each paged as entry (entry.id)}
        {@const dl = downloads[entry.id]}
        {@const badge = uiMode.advanced ? fitsBadge(entry) : simpleFitsBadge(entry)}
        {@const blocked = entry.downloadable === false || (entry.alerts ?? []).some(a => a.blocks_download)}
        {@const archLabel = shortArch(entry.architecture)}
        <div
          class="model-card"
          class:card-recommended={entry.recommended}
          class:card-danger={entry.alert_level === 'danger'}
          class:card-warning={entry.alert_level === 'warning'}
        >
          <div class="model-top">
            <div class="model-title-line">
              <div class="model-name" title={entry.name}>{entry.name}</div>
              <span class="size-chip">{entry.size_gb >= 100 ? entry.size_gb.toFixed(0) : entry.size_gb.toFixed(1)} GB</span>
            </div>
            {#if uiMode.advanced}
              <div class="model-id" title={entry.id}>{entry.id}</div>
            {/if}
            <div class="model-meta">
              {#if uiMode.advanced}
                <span class="badge badge-neutral">{entry.format.toUpperCase()}</span>
                {#if archLabel}
                  <span class="badge badge-neutral" title={entry.architecture}>{archLabel}</span>
                {/if}
              {:else if entry.tags.includes('moe')}
                <span class="badge badge-neutral">MoE</span>
              {/if}
              <span class="badge {badge.cls}">{badge.label}</span>
              {#if entry.recommended}
                <span class="badge badge-brand-pill"><i class="bi bi-star-fill"></i> {t('models.badgeRecommended')}</span>
              {/if}
              {#if entry.gated}
                <span class="badge badge-amber">{uiMode.advanced ? t('models.gatedLicense') : 'Requires sign-in'}</span>
              {/if}
            </div>
          </div>

          {#if entry.tags.length}
            <div class="tag-row">
              {#each entry.tags.slice(0, 3) as tag}
                <span class="tag">{tag}</span>
              {/each}
              {#if entry.tags.length > 3}
                <span class="tag">+{entry.tags.length - 3}</span>
              {/if}
            </div>
          {/if}

          {#if entry.alerts && entry.alerts.length}
            <details class="risk-drawer">
              <summary class="risk-summary risk-{entry.alert_level}">
                <i class="bi bi-exclamation-triangle-fill"></i>
                <span>
                  {uiMode.advanced
                    ? `${entry.alerts.length} compatibility ${entry.alerts.length === 1 ? 'note' : 'notes'}`
                    : 'Before you download'}
                </span>
                <span class="risk-toggle-hint">details</span>
              </summary>
              <div class="risk-list">
                {#each entry.alerts as alert}
                  <div class="risk-item risk-{alert.level}">{alert.message}</div>
                {/each}
              </div>
            </details>
          {:else if fitsVRAM(entry) === 'streaming'}
            <div class="stream-note">
              <i class="bi bi-lightning-charge-fill"></i>
              <span>{uiMode.advanced ? t('models.expertStreamWarning') : 'Larger than GPU memory; uses system RAM too (runs slower).'}</span>
            </div>
          {/if}

          <div class="model-actions">
            <button
              id="dl-{entry.id.replace(/\//g, '-')}"
              class="btn btn-primary btn-sm"
              onclick={() => downloadCatalog(entry)}
              disabled={blocked || dl?.status === 'downloading'}
            >
              {#if dl?.status === 'downloading'}
                <span class="spinner-sm"></span> {t('models.downloading')}…
              {:else if dl?.status === 'done'}
                <i class="bi bi-check-circle-fill"></i> {t('models.downloaded')}
              {:else}
                <i class="bi bi-cloud-arrow-down"></i> {t('models.download')}
              {/if}
            </button>
            {#if dl?.status === 'downloading' || (liveStatus && liveStatus.repo_id === entry.id && liveStatus.status === 'downloading')}
              <button class="btn btn-red btn-sm" onclick={handleCancelDownload}>
                <i class="bi bi-x-circle"></i> Cancel
              </button>
            {/if}
            <a
              class="btn btn-ghost btn-sm"
              href="https://huggingface.co/{entry.id}"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i class="bi bi-box-arrow-up-right"></i> HuggingFace
            </a>
          </div>

          {#if dl && dl.status !== 'idle'}
            <div class="dl-status" class:dl-ok={dl.status === 'done'} class:dl-err={dl.status === 'error'}>
              {dl.msg}
            </div>
          {/if}
        </div>
      {/each}
    </div>

    {#if totalPages > 1}
      <nav class="pagination-footer" aria-label="Catalog pages">
        <button type="button" class="btn btn-ghost btn-sm" disabled={safePage <= 1} onclick={() => currentPage = safePage - 1}>Previous</button>
        {#each pageNumbers as num}
          {#if num === 'ellipsis'}
            <span class="pagination-ellipsis">…</span>
          {:else}
            <button
              type="button"
              class="btn btn-sm {num === safePage ? 'btn-primary' : 'btn-ghost'}"
              aria-current={num === safePage ? 'page' : undefined}
              onclick={() => currentPage = num}
            >{num}</button>
          {/if}
        {/each}
        <button type="button" class="btn btn-ghost btn-sm" disabled={safePage >= totalPages} onclick={() => currentPage = safePage + 1}>Next</button>
      </nav>
    {/if}
  {/if}
</div>

<style>
  .model-preferences { margin-top: var(--space-3); }
  .model-preferences summary { cursor: pointer; font-weight: 600; }
  .preference-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: var(--space-3); margin: var(--space-3) 0; }
  .hub {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding: var(--space-5);
    overflow-y: auto;
    height: 100%;
  }

  .hub-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-4);
    flex-wrap: wrap;
  }
  .hub-title-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .hub-icon {
    font-size: 2rem;
    color: var(--color-brand);
  }
  .hub-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
  }
  .hub-sub {
    font-size: 0.82rem;
    color: var(--color-text-muted);
    margin: 2px 0 0;
  }

  .hw-chips {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    align-items: center;
  }
  .chip {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    color: var(--color-text);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .chip:hover, .chip-active {
    background: var(--color-brand-subtle);
    border-color: var(--color-brand);
    color: var(--color-brand);
  }
  .chip-brand {
    background: var(--color-brand-subtle);
    border-color: var(--color-brand);
    color: var(--color-brand);
    cursor: default;
  }

  .model-hub {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding-bottom: var(--space-4);
    box-sizing: border-box;
  }

  .card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card, 8px);
    box-shadow: var(--card-shadow, none);
    padding: var(--space-4);
  }
  .card-title {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: var(--space-3);
    color: var(--color-text-muted);
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }

  .storage-section {
    padding: var(--space-3) var(--space-4);
  }
  .storage-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
    min-height: 34px;
  }
  .storage-info {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    font-size: 0.85rem;
  }
  .storage-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    color: var(--color-text-muted);
  }
  .storage-path {
    display: inline-flex;
    align-items: center;
    font-family: var(--font-mono);
    font-size: 0.80rem;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 3px 10px;
    color: var(--color-brand);
    line-height: 1.2;
  }

  .storage-controls {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .quant-select-group {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
  }
  .quant-select-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--color-text-muted);
    white-space: nowrap;
  }
  .quant-dropdown {
    font-size: 0.80rem;
    padding: 3px 8px;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    color: var(--color-text);
  }

  .live-progress-card {
    border: 1px solid var(--color-brand);
    background: linear-gradient(135deg, var(--color-surface), var(--color-brand-subtle));
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    padding: var(--space-4);
  }
  .live-progress-card.download-error {
    border-color: #ef4444;
    background: linear-gradient(135deg, var(--color-surface), #ef444411);
  }
  .live-progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
  .live-model-info {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .live-model-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--color-text);
  }
  .live-file-sub {
    font-size: 0.76rem;
    color: var(--color-text-muted);
    font-family: var(--font-mono);
  }
  .live-stats {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .stat-badge {
    font-size: 0.76rem;
    padding: 3px 10px;
    border-radius: 12px;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    color: var(--color-text-muted);
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .pct-badge {
    font-size: 0.85rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 12px;
    background: var(--color-brand);
    color: #fff;
  }
  .progress-bar-bg {
    width: 100%;
    height: 8px;
    background: var(--color-surface-raised);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--color-border);
  }
  .progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-brand), #10b981);
    border-radius: 4px;
    transition: width 0.4s ease;
  }
  .live-progress-footer {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--color-text-muted);
    font-family: var(--font-mono);
  }

  .serving-section {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .serving-copy {
    margin: 0;
    color: var(--color-text-muted);
    font-size: 0.82rem;
    line-height: 1.5;
  }
  .serving-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .native-index-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
    padding-top: var(--space-2);
    border-top: 1px solid var(--color-border);
  }
  .native-index-label {
    color: var(--color-text);
    font-size: 0.8rem;
    font-weight: 600;
  }
  .native-unverified-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--color-text);
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 5px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
  }
  .native-unverified-toggle input {
    accent-color: var(--color-brand);
    cursor: pointer;
  }
  .native-index-hint {
    flex: 1 1 260px;
    color: var(--color-text-muted);
    font-size: 0.72rem;
    line-height: 1.4;
  }
  .serving-path {
    flex: 1 1 320px;
  }
  .server-status {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    border: 1px solid var(--color-border);
    background: var(--color-surface-raised);
    border-radius: var(--radius-1);
    padding: 8px 10px;
    color: var(--color-text-muted);
    font-size: 0.78rem;
  }
  .server-status code {
    margin-left: auto;
    color: var(--color-text);
  }
  .server-ok { border-color: #22c55e66; color: #22c55e; }
  .server-error { border-color: #ef444466; color: #ef4444; }
  .plan-card {
    border: 1px solid var(--color-border);
    background: var(--color-surface-raised);
    border-radius: var(--radius-1);
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.78rem;
  }
  .plan-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .plan-cmd {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .custom-section {}
  .custom-row {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .custom-row .input { flex: 2; min-width: 200px; }
  .custom-row .select { flex: 1; min-width: 140px; }
  .custom-row .input-sm { flex: 1; min-width: 140px; }

  .catalog-toolbar {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .filter-row {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    align-items: center;
  }
  .pagination-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
    font-size: 0.78rem;
    color: var(--color-text-muted);
    padding: 4px 0;
  }
  .pagination-summary { font-weight: 500; }
  .pagination-controls {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    flex-wrap: wrap;
  }
  .page-size-label {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.76rem;
    color: var(--color-text-muted);
    white-space: nowrap;
  }
  .pagination-page-label {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    padding: 0 4px;
  }
  .pagination-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    flex-wrap: wrap;
    padding: var(--space-3) 0 var(--space-5);
  }
  .pagination-ellipsis {
    color: var(--color-text-muted);
    padding: 0 4px;
  }

  .catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: var(--space-3);
  }

  .model-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-2);
    padding: var(--space-3) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    position: relative;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .model-card:hover {
    border-color: var(--color-brand);
    box-shadow: 0 0 0 2px var(--color-brand-subtle);
  }
  .card-recommended {
    border-color: var(--color-brand);
    background: linear-gradient(135deg, var(--color-surface), var(--color-brand-subtle));
  }
  .card-warning { border-color: #f59e0b66; }
  .card-danger { border-color: #ef444488; }
  .model-title-line {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-2);
  }
  .model-id {
    font-size: 0.71rem;
    font-family: var(--font-mono);
    color: var(--color-text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .catalog-policy {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    margin: 0 0 var(--space-3);
    line-height: 1.4;
  }
  .catalog-search { min-width: 220px; flex: 1; }
  .risk-drawer { display: flex; flex-direction: column; gap: 6px; }
  .risk-summary {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 5px 8px;
    border-radius: 5px;
    list-style: none;
  }
  .risk-summary::-webkit-details-marker { display: none; }
  .risk-toggle-hint {
    margin-left: auto;
    font-weight: 500;
    opacity: 0.8;
    text-decoration: underline;
  }
  .risk-list { display: flex; flex-direction: column; gap: 5px; }
  .risk-item {
    font-size: 0.73rem;
    line-height: 1.38;
    border-radius: 5px;
    padding: 5px 8px;
  }
  .risk-info { background: #38bdf822; border: 1px solid #38bdf844; color: var(--color-text); }
  .risk-warning { background: #f59e0b11; border: 1px solid #f59e0b44; color: #f59e0b; }
  .risk-danger { background: #ef444411; border: 1px solid #ef444466; color: #ef4444; }

  .model-top { display: flex; flex-direction: column; gap: 3px; }
  .model-name { font-size: 0.92rem; font-weight: 700; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .model-meta { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }

  .badge { font-size: 0.68rem; padding: 1px 7px; border-radius: 10px; font-weight: 500; }
  .badge-neutral { background: var(--color-surface-raised); border: 1px solid var(--color-border); color: var(--color-text-muted); }
  .badge-green   { background: #22c55e22; border: 1px solid #22c55e; color: #22c55e; }
  .badge-amber   { background: #f59e0b22; border: 1px solid #f59e0b; color: #f59e0b; }
  .badge-red     { background: #ef444422; border: 1px solid #ef4444; color: #ef4444; }
  .badge-brand-pill { background: var(--color-brand); color: #fff; font-weight: 600; display: inline-flex; align-items: center; gap: 3px; }

  .size-chip {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--color-text-muted);
    flex-shrink: 0;
  }

  .tag-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .tag {
    font-size: 0.66rem;
    padding: 1px 7px;
    border-radius: 10px;
    background: var(--color-surface-raised);
    color: var(--color-text-muted);
    border: 1px solid var(--color-border);
  }

  .stream-note {
    font-size: 0.72rem;
    color: #f59e0b;
    background: #f59e0b11;
    border: 1px solid #f59e0b44;
    border-radius: 5px;
    padding: 5px 8px;
    display: flex;
    gap: 6px;
    align-items: center;
    line-height: 1.35;
  }
  .stream-note i { flex-shrink: 0; }

  .model-actions {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
    margin-top: auto;
    padding-top: var(--space-2);
  }
  .btn { display: inline-flex; align-items: center; gap: 6px; border-radius: var(--radius-1); font-size: 0.82rem; font-weight: 500; padding: 6px 14px; cursor: pointer; border: 1px solid transparent; transition: background 0.15s, opacity 0.15s; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary { background: var(--color-brand); color: #fff; border-color: var(--color-brand); }
  .btn-primary:hover:not(:disabled) { opacity: 0.85; }
  .btn-ghost { background: transparent; color: var(--color-text-muted); border-color: var(--color-border); }
  .btn-ghost:hover:not(:disabled) { background: var(--color-surface-raised); }
  .btn-sm { padding: 4px 10px; font-size: 0.78rem; }

  .dl-status { font-size: 0.75rem; padding: 4px 8px; border-radius: 4px; background: var(--color-surface-raised); }
  .dl-ok { background: #22c55e22; color: #22c55e; }
  .dl-err { background: #ef444422; color: #ef4444; }

  .loading-state, .error-state {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    font-size: 0.9rem;
    color: var(--color-text-muted);
    padding: var(--space-6);
    justify-content: center;
  }

  .spinner {
    width: 20px; height: 20px;
    border: 2px solid var(--color-border);
    border-top-color: var(--color-brand);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  .spinner-sm {
    width: 12px; height: 12px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .input, .select {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-1);
    padding: 6px 10px;
    font-size: 0.83rem;
    color: var(--color-text);
    outline: none;
    transition: border-color 0.15s;
  }
  .input:focus, .select:focus {
    border-color: var(--color-brand);
  }

  .capacity-card {
    padding: var(--space-3) var(--space-4);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .capacity-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--space-3);
  }
  @media (max-width: 820px) { .capacity-grid { grid-template-columns: 1fr; } }
  .capacity-stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 10px;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: 5px;
  }
  .capacity-label {
    font-size: 0.72rem;
    color: var(--color-text-muted);
    font-weight: 500;
  }
  .capacity-value {
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--color-ink);
  }
  .capacity-note {
    margin: 0;
    font-size: 0.72rem;
    color: var(--color-text-muted);
    line-height: 1.4;
  }
  .simple-serving-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-3);
    flex-wrap: wrap;
  }
</style>
