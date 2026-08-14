<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../../api'
  import { hwStore } from '../../store.svelte'
  import { t } from '../../i18n.svelte'
  import type { CatalogEntry } from '../../types'

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
  let inferencePlan = $state<Record<string, unknown> | null>(null)
  let planning = $state(false)

  // Filter
  let filterTag = $state('all')
  let catalogQuery = $state('')

  // Storage Location & Quantization
  let defaultDestDir = $state('')
  let selectedDestDir = $state('')
  let selectedQuant = $state('auto')

  const activeDestDir = $derived(selectedDestDir || defaultDestDir)

  import type { ModelDownloadStatus } from '../../types'

  let liveStatus = $state<ModelDownloadStatus | null>(null)

  // ─── Load catalog & settings on mount ──────────────────────────────────────
  onMount(() => {
    api.getSettings().then(s => {
      if (s.hf_cache_dir) defaultDestDir = s.hf_cache_dir
    }).catch((e) => {
      console.error('Failed to load settings', e)
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

    return () => clearInterval(interval)
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
      if (typeof picked === 'string') serveModelPath = picked
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  async function inspectPlan(exportRuntimes = false) {
    if (!serveModelPath.trim()) return
    planning = true
    try {
      inferencePlan = await api.planInference(serveModelPath.trim(), 4096, exportRuntimes)
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
    serverState = 'starting'
    serverMessage = t('models.serverPreflight')
    try {
      await api.startChatServer(serveModelPath.trim())
      serverState = 'running'
      serverMessage = t('models.serverRunning')
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
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
    try {
      await api.stopChatServer()
      serverState = 'idle'
      serverMessage = t('models.serverStopped')
    } catch (e) {
      serverState = 'error'
      serverMessage = String(e)
    }
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────────
  const allTags = $derived(
    ['all', ...Array.from(new Set(catalog.flatMap(c => c.tags)))]
  )

  const filtered = $derived(
    (filterTag === 'all' ? catalog : catalog.filter(c => c.tags.includes(filterTag)))
      .filter(c => {
        const q = catalogQuery.trim().toLowerCase()
        if (!q) return true
        return c.id.toLowerCase().includes(q) || c.name.toLowerCase().includes(q) || c.tags.some(t => t.toLowerCase().includes(q))
      })
  )

  const vramGb = $derived(
    hwStore.info?.vram_mb != null ? Math.round(hwStore.info.vram_mb / 1024) : 0
  )

  function fitsVRAM(entry: CatalogEntry): 'yes' | 'streaming' | 'large' {
    if (entry.format === 'gguf') {
      if (entry.size_gb <= vramGb - 2) return 'yes'
      if (entry.size_gb <= 400) return 'streaming'
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
        <h1 class="hub-title">{t('models.title')}</h1>
        <p class="hub-sub">{t('models.subtitle')}</p>
      </div>
    </div>
    {#if hwStore.info}
      <div class="hw-chips">
        <span class="chip"><i class="bi bi-gpu-card"></i> {vramGb} GB VRAM</span>
        <span class="chip"><i class="bi bi-memory"></i> {hwStore.info.ram_mb != null ? Math.round(hwStore.info.ram_mb / 1024) + ' GB RAM' : 'RAM unknown'}</span>
        <span class="chip chip-brand"><i class="bi bi-shield-check"></i> {t('models.expertStreamingOn')}</span>
      </div>
    {/if}
  </div>

  <!-- Storage Location & Quantization Picker -->
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
        disabled={serverState !== 'running'}
      >
        <i class="bi bi-stop-fill"></i> {t('models.stopServer')}
      </button>
    </div>
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
      <span class="native-index-hint">{t('models.nativeIndexHint')}</span>
    </div>
    {#if inferencePlan}
      {@const estimates = (inferencePlan.estimates ?? {}) as Record<string, unknown>}
      <div class="plan-card">
        <div class="plan-row">
          <span>{String(estimates.architecture ?? inferencePlan.backend)}</span>
          <span>{String(estimates.quantization ?? '')}</span>
          {#if estimates.is_moe}<span class="badge badge-amber">MoE</span>{/if}
        </div>
        <div class="plan-row">
          <span>{t('models.gpuLayers')}: {String(estimates.gpu_layers ?? '—')}</span>
          <span>{t('models.peakVram')}: {String(estimates.peak_vram_mb ?? '—')} MB</span>
          <span>{t('models.peakRam')}: {String(estimates.peak_ram_mb ?? '—')} MB</span>
        </div>
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

  <!-- Filter chips -->
  <div class="filter-row">
    <input
      class="input catalog-search"
      placeholder={t('models.searchCatalog')}
      bind:value={catalogQuery}
    />
    {#each allTags as tag}
      <button
        id="filter-{tag}"
        class="chip"
        class:chip-active={filterTag === tag}
        onclick={() => filterTag = tag}
      >{tag}</button>
    {/each}
  </div>

  <!-- Catalog grid -->
  {#if loading}
    <div class="loading-state"><span class="spinner"></span> {t('models.loadingCatalog')}</div>
  {:else if error}
    <div class="error-state"><i class="bi bi-exclamation-triangle"></i> {error}</div>
  {:else}
    <div class="catalog-grid">
      {#each filtered as entry (entry.id)}
        {@const dl = downloads[entry.id]}
        {@const badge = fitsBadge(entry)}
        {@const blocked = entry.downloadable === false || (entry.alerts ?? []).some(a => a.blocks_download)}
        <div
          class="model-card"
          class:card-recommended={entry.recommended}
          class:card-danger={entry.alert_level === 'danger'}
          class:card-warning={entry.alert_level === 'warning'}
        >
          {#if entry.recommended}
            <div class="recommended-ribbon"><i class="bi bi-star-fill"></i> {t('models.badgeRecommended')}</div>
          {/if}
          <div class="model-top">
            <div class="model-name">{entry.name}</div>
            <div class="model-id">{entry.id}</div>
            <div class="model-meta">
              <span class="badge badge-neutral">{entry.format.toUpperCase()}</span>
              {#if entry.architecture}
                <span class="badge badge-neutral">{entry.architecture}</span>
              {/if}
              <span class="badge {badge.cls}">{badge.label}</span>
              {#if entry.gated}
                <span class="badge badge-amber">{t('models.gatedLicense')}</span>
              {/if}
              {#if blocked}
                <span class="badge badge-red">{t('models.notDownloadable')}</span>
              {/if}
              <span class="size-chip">{entry.size_gb >= 100 ? entry.size_gb.toFixed(0) : entry.size_gb.toFixed(1)} GB</span>
            </div>
          </div>

          <div class="tag-row">
            {#each entry.tags as tag}
              <span class="tag">{tag}</span>
            {/each}
          </div>

          {#if entry.alerts && entry.alerts.length}
            <div class="risk-list">
              <div class="risk-title"><i class="bi bi-exclamation-triangle-fill"></i> {t('models.riskTitle')}</div>
              {#each entry.alerts as alert}
                <div class="risk-item risk-{alert.level}">{alert.message}</div>
              {/each}
            </div>
          {:else if fitsVRAM(entry) === 'streaming'}
            <div class="stream-note">
              <i class="bi bi-lightning-charge-fill"></i>
              {t('models.expertStreamWarning')}
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
  {/if}
</div>

<style>
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

  .filter-row {
    display: flex;
    gap: var(--space-2);
    flex-wrap: wrap;
  }

  .catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: var(--space-4);
  }

  .model-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-2);
    padding: var(--space-4);
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
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
  .card-warning { border-color: #f59e0b88; }
  .card-danger { border-color: #ef4444; }
  .model-id {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--color-text-muted);
    word-break: break-all;
  }
  .catalog-policy {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    margin: 0 0 var(--space-3);
    line-height: 1.4;
  }
  .catalog-search { min-width: 220px; flex: 1; }
  .risk-list { display: flex; flex-direction: column; gap: 6px; }
  .risk-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--color-text-muted);
  }
  .risk-item {
    font-size: 0.75rem;
    line-height: 1.4;
    border-radius: 6px;
    padding: 6px 10px;
  }
  .risk-info { background: #38bdf822; border: 1px solid #38bdf844; color: var(--color-text); }
  .risk-warning { background: #f59e0b11; border: 1px solid #f59e0b44; color: #f59e0b; }
  .risk-danger { background: #ef444411; border: 1px solid #ef444466; color: #ef4444; }
  .recommended-ribbon {
    position: absolute;
    top: -1px;
    right: 12px;
    background: var(--color-brand);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 0 0 8px 8px;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .model-top { display: flex; flex-direction: column; gap: 4px; }
  .model-name { font-size: 0.95rem; font-weight: 700; }
  .model-meta { display: flex; gap: 4px; flex-wrap: wrap; align-items: center; }

  .badge { font-size: 0.7rem; padding: 1px 8px; border-radius: 10px; font-weight: 500; }
  .badge-neutral { background: var(--color-surface-raised); border: 1px solid var(--color-border); color: var(--color-text-muted); }
  .badge-green   { background: #22c55e22; border: 1px solid #22c55e; color: #22c55e; }
  .badge-amber   { background: #f59e0b22; border: 1px solid #f59e0b; color: #f59e0b; }
  .badge-red     { background: #ef444422; border: 1px solid #ef4444; color: #ef4444; }

  .size-chip {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    margin-left: auto;
  }

  .tag-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .tag {
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 10px;
    background: var(--color-surface-raised);
    color: var(--color-text-muted);
    border: 1px solid var(--color-border);
  }

  .stream-note {
    font-size: 0.75rem;
    color: #f59e0b;
    background: #f59e0b11;
    border: 1px solid #f59e0b44;
    border-radius: 6px;
    padding: 6px 10px;
    display: flex;
    gap: 6px;
    align-items: flex-start;
    line-height: 1.4;
  }
  .stream-note i { flex-shrink: 0; margin-top: 1px; }

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
</style>
