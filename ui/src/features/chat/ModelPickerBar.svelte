<script lang="ts">
  import { t } from '../../i18n.svelte'
  import { api } from '../../api'
  import type { ChatTargetsReport, LocalModelItem } from '../../types'
  import type { ChatEndpoint } from './chatStore.svelte'
  import { inferLanguageScope } from './modelRouter'
  import { chatPower, noteChatActivity, releaseChatGpu, savePowerPrefs } from './chatPower.svelte'

  interface Props {
    selected?: ChatEndpoint
    onSelect: (endpoint: ChatEndpoint) => void
    onContextDetected?: (ctx: number | undefined) => void
    onReportUpdated?: (report: ChatTargetsReport) => void
    onUnloadNotice?: (message: string) => void
  }

  let { selected, onSelect, onContextDetected, onReportUpdated, onUnloadNotice }: Props = $props()

  let report = $state<ChatTargetsReport | null>(null)
  let scanning = $state(false)
  let unloading = $state(false)
  let startingModel = $state<string | null>(null)
  let statusMsg = $state('')
  let showCustom = $state(false)
  let customUrl = $state('http://127.0.0.1:8080')
  let customModel = $state('')

  function langShort(modelName: string, reportedScope?: string): string {
    const info = inferLanguageScope(modelName, reportedScope)
    if (info.scope === 'multilingual') return t('chat.langShortMulti')
    if (info.scope === 'english') return t('chat.langShortEnglish')
    return t('chat.langShortUnknown')
  }

  export async function refreshTargets(): Promise<ChatTargetsReport | null> {
    scanning = true
    statusMsg = ''
    try {
      const next = await api.discoverChatTargets()
      report = next
      onReportUpdated?.(next)
      if (!selected?.model && selected?.provider !== 'router') {
        if (next.providers.sytra_running && next.providers.sytra_models.length) {
          onSelect({ provider: 'sytra', baseUrl: 'http://127.0.0.1:8080', model: next.providers.sytra_models[0] })
        } else if (next.providers.ollama_running && next.providers.ollama_models.length) {
          onSelect({ provider: 'ollama', baseUrl: 'http://127.0.0.1:11434', model: next.providers.ollama_models[0].name })
        } else if (next.providers.lmstudio_running && next.providers.lmstudio_models.length) {
          onSelect({ provider: 'lmstudio', baseUrl: 'http://127.0.0.1:1234', model: next.providers.lmstudio_models[0] })
        }
      }
      return next
    } catch (err) {
      statusMsg = String(err)
      return null
    } finally {
      scanning = false
    }
  }

  $effect(() => {
    void refreshTargets()
  })

  function buildChoiceValue(ep?: ChatEndpoint): string {
    if (ep?.provider === 'router') return 'auto-route'
    if (!ep?.model) return ''
    return `${ep.provider}::${ep.baseUrl}::${ep.model}`
  }

  async function handleSelectChange(event: Event) {
    const val = (event.currentTarget as HTMLSelectElement).value
    if (!val) return
    if (val === 'auto-route') {
      onSelect({ provider: 'router', baseUrl: '', model: 'auto-route' })
      statusMsg = t('chat.routeActive')
      return
    }
    if (val === '__custom__') {
      customUrl = selected?.baseUrl || customUrl || 'http://127.0.0.1:8080'
      customModel = selected?.model || customModel || ''
      showCustom = true
      return
    }
    if (val.startsWith('local::')) {
      const path = val.slice('local::'.length)
      await launchLocalModel(path)
      return
    }
    const [provider, baseUrl, ...rest] = val.split('::')
    const model = rest.join('::')
    if (provider === 'ollama' && report && !report.providers.ollama_running) {
      startingModel = model
      statusMsg = t('chat.startOllama')
      try {
        await api.startOllamaDaemon()
        await refreshTargets()
      } catch (err) {
        statusMsg = String(err)
      } finally {
        startingModel = null
      }
    }
    const chosen: ChatEndpoint = {
      provider: provider as ChatEndpoint['provider'],
      baseUrl,
      model,
    }
    noteChatActivity(chosen)
    onSelect(chosen)
    if (provider === 'ollama') {
      api.getOllamaContext(baseUrl, model).then(ctx => onContextDetected?.(ctx ?? undefined)).catch(() => {})
    }
  }

  async function launchLocalModel(path: string) {
    startingModel = path
    statusMsg = t('chat.startLocal')
    try {
      await api.startChatServer(path)
      await refreshTargets()
      const activeId = report?.providers.sytra_models[0] || path.split(/[\\/]/).pop() || path
      const chosen: ChatEndpoint = { provider: 'sytra', baseUrl: 'http://127.0.0.1:8080', model: activeId }
      noteChatActivity(chosen)
      onSelect(chosen)
      statusMsg = t('chat.serving', { model: activeId })
    } catch (err) {
      statusMsg = String(err)
    } finally {
      startingModel = null
    }
  }

  async function handleUnloadGpu() {
    unloading = true
    try {
      if (selected && selected.provider !== 'router' && selected.model) {
        chatPower.activeTarget = { ...selected }
      }
      const msg = await releaseChatGpu('manual')
      statusMsg = msg || t('chat.released')
      onUnloadNotice?.(statusMsg)
      await refreshTargets()
    } finally {
      unloading = false
    }
  }

  function applyCustomEndpoint() {
    if (!customUrl.trim() || !customModel.trim()) return
    const chosen: ChatEndpoint = {
      provider: 'custom',
      baseUrl: customUrl.trim().replace(/\/+$/, ''),
      model: customModel.trim(),
    }
    noteChatActivity(chosen)
    onSelect(chosen)
    showCustom = false
  }

  function localCategoryTag(item: LocalModelItem): string {
    if (item.category === 'lm-studio') return 'LM Studio cache'
    if (item.category === 'hf-cache' || item.category === 'downloaded') return 'HF Cache'
    if (item.category === 'finetuned') return 'Finetuned'
    if (item.category === 'merged') return 'Merged'
    return item.category
  }
</script>

<div class="model-picker-bar">
  <div class="picker-main">
    <span class="picker-label"><i class="bi bi-hdd-rack" aria-hidden="true"></i> {t('chat.model')}</span>
    <select
      id="chat-model-select"
      class="input picker-select"
      aria-label={t('chat.modelSelect')}
      value={buildChoiceValue(selected)}
      disabled={startingModel !== null}
      onchange={handleSelectChange}
    >
      <option value="">{t('chat.autoDetect')}</option>
      <option value="auto-route">{t('chat.autoRoute')}</option>

      {#if report?.providers.sytra_models.length}
        <optgroup label={t('chat.sytraServer')}>
          {#each report.providers.sytra_models as m}
            <option value={`sytra::http://127.0.0.1:8080::${m}`}>{m} [{langShort(m)}]</option>
          {/each}
        </optgroup>
      {/if}

      {#if report?.providers.ollama_models.length}
        <optgroup label={`Ollama (127.0.0.1:11434 - ${t(report.providers.ollama_running ? 'chat.running' : 'chat.clickStart')})`}>
          {#each report.providers.ollama_models as om}
            <option value={`ollama::http://127.0.0.1:11434::${om.name}`}>
              {om.name}{om.size_gb ? ` (${om.size_gb} GB)` : ''}{om.parameter_size ? ` - ${om.parameter_size}` : ''} [{langShort(om.name, om.language_scope)}]
            </option>
          {/each}
        </optgroup>
      {/if}

      {#if report?.providers.lmstudio_models.length}
        <optgroup label={t('chat.lmServer')}>
          {#each report.providers.lmstudio_models as lm}
            <option value={`lmstudio::http://127.0.0.1:1234::${lm}`}>{lm} [{langShort(lm)}]</option>
          {/each}
        </optgroup>
      {/if}

      {#if report?.local_models.length}
        <optgroup label={t('chat.localFiles')}>
          {#each report.local_models as item}
            <option value={`local::${item.path}`}>
              [{localCategoryTag(item)}] {item.name} ({item.size_gb} GB - {langShort(item.name)})
            </option>
          {/each}
        </optgroup>
      {/if}

      <option value="__custom__">{t('chat.customEndpoint')}</option>
    </select>

    <button
      type="button"
      id="btn-refresh-chat-models"
      class="btn btn-secondary btn-sm"
      disabled={scanning || startingModel !== null}
      title={t('chat.scanHint')}
      onclick={() => void refreshTargets()}
    >
      <i class={`bi ${scanning ? 'bi-hourglass-split' : 'bi-arrow-clockwise'}`} aria-hidden="true"></i>
      {scanning ? t('chat.scanning') : t('chat.scan')}
    </button>

    <button
      type="button"
      id="btn-unload-model"
      class="btn btn-secondary btn-sm"
      disabled={unloading || chatPower.isBusy}
      title={t('chat.unloadHint')}
      onclick={() => void handleUnloadGpu()}
    >
      <i class="bi bi-power" aria-hidden="true"></i>
      {unloading ? t('chat.releasing') : t('chat.unload')}
    </button>

    <label class="power-check" title={t('chat.idleOutsideHint')}>
      <input type="checkbox" bind:checked={chatPower.releaseOnLeaveChat} onchange={savePowerPrefs} />
      <span>{t('chat.idleOutside')}</span>
    </label>

    <label class="idle-select-wrap" title={t('chat.idleHint')}>
      <span>{t('chat.idle')}</span>
      <select
        id="chat-idle-timeout"
        class="input idle-select"
        bind:value={chatPower.idleMinutes}
        onchange={savePowerPrefs}
      >
        <option value={0}>{t('chat.mcpOff')}</option>
        <option value={2}>2m</option>
        <option value={5}>5m</option>
        <option value={15}>15m</option>
      </select>
    </label>
  </div>

  {#if showCustom}
    <div class="custom-ep-row">
      <input class="input" placeholder="http://127.0.0.1:1234" bind:value={customUrl} />
      <input class="input" placeholder={t('chat.modelIdExample')} bind:value={customModel} />
      <button type="button" class="btn btn-primary btn-sm" onclick={applyCustomEndpoint}>{t('chat.useEndpoint')}</button>
      <button type="button" class="btn btn-ghost btn-sm" onclick={() => { showCustom = false }}>{t('chat.cancel')}</button>
    </div>
  {/if}

  {#if statusMsg}
    <span class="picker-status">{statusMsg}</span>
  {/if}
</div>

<style>
  .model-picker-bar { display: flex; flex-direction: column; gap: 6px; }
  .picker-main { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .picker-label {
    font-size: 11.5px;
    font-weight: 600;
    color: var(--color-ink-subtle);
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .picker-select {
    flex: 1;
    min-width: 220px;
    max-width: 440px;
    font-size: 12px;
    padding: 0 8px;
    height: 30px;
    box-sizing: border-box;
  }
  .picker-main .btn-sm {
    height: 30px;
    padding: 0 10px;
    box-sizing: border-box;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    line-height: 1;
  }
  .power-check, .idle-select-wrap {
    height: 30px;
    box-sizing: border-box;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    line-height: 1;
    color: var(--color-ink-subtle);
    padding: 0 9px;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface-muted);
    cursor: pointer;
  }
  .power-check input[type="checkbox"] {
    width: 14px;
    height: 14px;
    margin: 0;
    accent-color: var(--color-brand);
    flex-shrink: 0;
  }
  .idle-select {
    font-size: 11px;
    padding: 0 4px;
    height: 22px;
    min-height: 0;
    line-height: 1;
    width: auto;
    box-sizing: border-box;
  }
  .custom-ep-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .custom-ep-row .input { flex: 1; min-width: 150px; font-size: 12px; padding: 4px 8px; }
  .picker-status { font-size: 11px; color: var(--color-ink-subtle); }
</style>
