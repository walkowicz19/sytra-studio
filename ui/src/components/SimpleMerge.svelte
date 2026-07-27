<script lang="ts">
  /**
   * SimpleMerge — guided model merging for non-technical users.
   * Pick 2-3 models, say what you want, and the right method, weights and
   * base model are chosen automatically. Compatibility is checked before
   * anything starts.
   */
  import { onMount } from 'svelte'
  import { run, resetRun, pushToast, watchTelemetry, hwStore } from '../store.svelte'
  import { t } from '../i18n.svelte'
  import { api } from '../api'
  import type { CompatResult, LocalModelItem } from '../types'
  import catalogData from '../../../crates/sytra-contracts/src/catalog.json'

  const catalogIds: string[] = (catalogData as { model_id: string }[]).map(m => m.model_id)

  let localModels = $state<LocalModelItem[]>([])
  let models = $state<string[]>(['', ''])

  onMount(async () => {
    try {
      localModels = await api.listLocalModels()
    } catch {}
  })

  async function pickLocalModel(index: number) {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        directory: false,
        multiple: false,
        title: 'Select local model file or folder',
        filters: [{ name: 'Model Files', extensions: ['gguf', 'safetensors', 'bin', 'pth'] }]
      })
      if (typeof picked === 'string' && picked) {
        models[index] = picked
      }
    } catch {}
  }

  const vramMb = $derived(hwStore.info ? hwStore.info.vram_mb : 0)
  const totalVramGb = $derived(vramMb / 1024)

  // Combined size of all chosen local models
  const totalSelectedGb = $derived(
    models
      .map(p => localModels.find(m => m.path === p || m.id === p || m.name === p))
      .filter(Boolean)
      .reduce((sum, m) => sum + (m?.size_gb ?? 0), 0)
  )

  // Whether the VRAM check blocks the merge (known models that collectively exceed 2× VRAM)
  // MoE streaming handles up to 2× VRAM, beyond that we block.
  const vramBlocked = $derived(
    totalSelectedGb > 0 && vramMb > 0 && totalSelectedGb > totalVramGb * 2
  )

  function getModelVramStatus(modelPath: string): { label: string; status: 'ok' | 'streaming' | 'too-large' | 'unknown' } {
    if (!modelPath) return { label: '', status: 'unknown' }
    const match = localModels.find(m => m.path === modelPath || m.id === modelPath || m.name === modelPath)
    if (match) {
      const neededMb = match.size_gb * 1024
      if (neededMb <= vramMb) {
        return { label: `Fits VRAM (${match.size_gb} GB)`, status: 'ok' }
      } else if (match.size_gb <= totalVramGb * 2) {
        return { label: `Expert Streaming Ready (${match.size_gb} GB)`, status: 'streaming' }
      } else {
        return { label: `Too large for Expert Streaming (${match.size_gb} GB — needs ${(match.size_gb / 2).toFixed(1)} GB VRAM)`, status: 'too-large' }
      }
    }
    return { label: 'External / HF Model', status: 'unknown' }
  }

  // ── Step 1: models ────────────────────────────────────────────────────
  function addModel() {
    if (models.length < 3) models = [...models, '']
  }
  function removeModel(i: number) {
    if (models.length > 2) models = models.filter((_, idx) => idx !== i)
  }

  const filled = $derived(models.map(m => m.trim()).filter(Boolean))
  const modelsReady = $derived(filled.length >= 2)

  // ── Step 2: goal ──────────────────────────────────────────────────────
  type Goal = 'combine' | 'blend' | 'specialize'
  let goal = $state<Goal>('combine')

  const goals: Goal[] = ['combine', 'blend', 'specialize']

  const method = $derived.by(() => {
    if (goal === 'blend' && filled.length === 2) return 'slerp'
    if (goal === 'specialize') return 'ties'
    return 'dare_ties'
  })

  // ── Compat + start ────────────────────────────────────────────────────
  let compat = $state<CompatResult | null>(null)
  let checking = $state(false)
  let submitting = $state(false)

  async function checkCompat(): Promise<CompatResult | null> {
    checking = true
    try {
      const base = method === 'ties' || method === 'dare_ties' ? filled[0] : null
      compat = await api.mergeCheck(filled, method as never, base)
      return compat
    } catch (e) {
      pushToast('error', `${t('combine.checkFailed')}: ${e instanceof Error ? e.message : String(e)}`)
      return null
    } finally {
      checking = false
    }
  }

  async function start() {
    if (submitting || run.status === 'running' || !modelsReady) return
    submitting = true

    // VRAM check: block if total local model size exceeds 2× available VRAM
    if (vramBlocked) {
      pushToast('error', `Cannot merge: combined model size (${totalSelectedGb.toFixed(1)} GB) exceeds Expert Streaming capacity (${(totalVramGb * 2).toFixed(1)} GB). Select smaller models or use HF model IDs instead.`)
      submitting = false
      return
    }

    const verdict = await checkCompat()
    if (!verdict) { submitting = false; return }
    if (verdict.verdict === 'red') {
      pushToast('error', `${t('combine.cantMerge')}: ${verdict.reason}`)
      submitting = false
      return
    }

    resetRun()
    const needsBase = method === 'ties' || method === 'dare_ties'
    const weight = 1 / filled.length
    const config = {
      version: 1,
      merge_method: method,
      base_model: needsBase ? filled[0] : null,
      dtype: 'bfloat16',
      models: filled.map(m => ({ model: m, parameters: { weight, density: 0.53 } })),
      tokenizer: { source: 'base' },
      compat: { verdict: verdict.verdict, fingerprint: null },
      output: { model_path: `runs/merged-${Date.now()}` },
    }

    try {
      const opId = await api.startMerge(config as never)
      run.opId = opId; run.kind = 'merge'; run.status = 'running'; run.startedAt = Date.now()
      pushToast('success', t('combine.started'))
      watchTelemetry(opId)
    } catch (e) {
      run.status = 'error'
      pushToast('error', `${t('teach.couldNotStart')}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      submitting = false
    }
  }
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">{t('combine.title')}</h1>
      <p class="text-small">{t('combine.subtitle')}</p>
    </div>
  </div>

  <div class="simple-scroll">
    <div class="simple-flow">

      <!-- 01 · Models -->
      <section class="step">
        <div class="step-head">
          <span class="step-num">01</span>
          <div>
            <div class="step-title">{t('combine.step1.title')}</div>
            <div class="step-sub">{t('combine.step1.sub')}</div>
          </div>
        </div>

        <div class="model-list">
          {#each models as _, i}
            {@const vStat = getModelVramStatus(models[i])}
            <div class="model-row-group">
              <div class="model-row">
                <span class="model-index">{i + 1}</span>
                <input
                  class="input input-mono"
                  placeholder={i === 0 ? 'e.g. mistralai/Mistral-7B-v0.1 or C:\\models\\model.gguf' : 'e.g. org/knowledge-ft'}
                  bind:value={models[i]}
                  list="catalog-models"
                />

                <button class="btn btn-secondary btn-sm" onclick={() => pickLocalModel(i)} title="Browse OS File Explorer for local model file or folder">
                  <i class="bi bi-folder2-open"></i> Browse
                </button>

                {#if models.length > 2}
                  <button class="btn btn-ghost btn-icon" onclick={() => removeModel(i)} aria-label="Remove model">
                    <i class="bi bi-x-lg"></i>
                  </button>
                {/if}
              </div>

              <!-- Local Model Select & VRAM compatibility status -->
              <div class="model-meta-row">
                {#if localModels.length > 0}
                  <select class="select select-sm local-model-select" onchange={(e) => models[i] = (e.target as HTMLSelectElement).value}>
                    <option value="">-- Choose from Scanned Local Models --</option>
                    {#each localModels as lm}
                      <option value={lm.path}>[{lm.category.toUpperCase()}] {lm.name} ({lm.size_gb} GB)</option>
                    {/each}
                  </select>
                {/if}

                {#if vStat.status === 'ok'}
                  <span class="badge badge-green"><i class="bi bi-check-circle-fill"></i> {vStat.label}</span>
                {:else if vStat.status === 'streaming'}
                  <span class="badge badge-warning"><i class="bi bi-lightning-fill"></i> {vStat.label}</span>
                {:else if vStat.status === 'too-large'}
                  <span class="badge badge-red"><i class="bi bi-x-circle-fill"></i> {vStat.label}</span>
                {/if}
              </div>
            </div>
          {/each}
          <datalist id="catalog-models">
            {#each catalogIds as id}<option value={id}></option>{/each}
          </datalist>
        </div>

        <!-- Global VRAM block warning -->
        {#if vramBlocked}
          <div class="vram-block-alert">
            <i class="bi bi-exclamation-triangle-fill"></i>
            <div>
              <strong>Cannot Merge — Models Too Large for This GPU</strong>
              <p>Combined local model size <strong>{totalSelectedGb.toFixed(1)} GB</strong> exceeds Expert Streaming capacity of <strong>{(totalVramGb * 2).toFixed(1)} GB</strong> (2× your {totalVramGb.toFixed(1)} GB VRAM). Use smaller models or supply HF model IDs to download on-demand instead.</p>
            </div>
          </div>
        {/if}

        {#if models.length < 3}
          <button class="btn btn-ghost btn-sm" onclick={addModel} style="align-self:flex-start">
            <i class="bi bi-plus-lg"></i> {t('combine.addThird')}
          </button>
        {/if}
      </section>

      <!-- 02 · Goal -->
      <section class="step">
        <div class="step-head">
          <span class="step-num">02</span>
          <div>
            <div class="step-title">{t('combine.step2.title')}</div>
            <div class="step-sub">{t('combine.step2.sub')}</div>
          </div>
        </div>

        <div class="choice-col">
          {#each goals as g}
            <button
              class="choice"
              class:selected={goal === g}
              class:disabled-choice={g === 'blend' && filled.length !== 2}
              disabled={g === 'blend' && filled.length > 2}
              onclick={() => (goal = g)}
            >
              <div>
                <span class="choice-label">{t(`combine.goal.${g}`)}</span>
                <span class="choice-hint">{t(`combine.goal.${g}Hint`)}</span>
              </div>
            </button>
          {/each}
        </div>
      </section>

      <!-- Compat verdict -->
      {#if compat}
        <div class="verdict verdict-{compat.verdict}">
          {#if compat.verdict === 'green'}
            <i class="bi bi-check-circle-fill"></i> {t('combine.compatible')}
          {:else if compat.verdict === 'amber'}
            <i class="bi bi-exclamation-triangle-fill"></i> {compat.reason} — {t('combine.mayWork')}
          {:else}
            <i class="bi bi-x-octagon-fill"></i> {compat.reason}
          {/if}
        </div>
      {/if}

      <!-- Start -->
      <div class="start-row">
        <button
          class="btn btn-primary btn-lg start-btn"
          onclick={start}
          disabled={submitting || checking || !modelsReady || run.status === 'running' || vramBlocked}
        >
          {#if submitting || checking}<span class="spinner"></span>{/if}
          {t('combine.start')}
        </button>
        {#if vramBlocked}
          <span class="text-small" style="color: var(--color-error, #e05c5c)"><i class="bi bi-x-circle-fill"></i> Models too large for your GPU — merge blocked</span>
        {:else if run.status === 'running'}
          <span class="text-small">{t('teach.runInProgress')}</span>
        {:else if !modelsReady}
          <span class="text-small">{t('combine.enterTwo')}</span>
        {/if}
      </div>

    </div>
  </div>
</div>

<style>
  .simple-scroll { flex: 1; overflow-y: auto; }
  .simple-flow {
    width: 100%;
    max-width: 1280px;
    padding: var(--space-10) var(--space-10) 64px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  /* Two-column grid: number in a fixed left rail; title and content share
     the second column so they align to one vertical line. */
  .step {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr);
    row-gap: var(--space-5);
    padding: var(--space-10) 0;
    border-bottom: 1px solid var(--color-border);
  }
  .step:first-child { padding-top: 0; }
  .step > :global(*) { grid-column: 2; }
  .step > :global(button) { justify-self: start; }
  .step-head { display: contents; }
  .step-num {
    grid-column: 1;
    grid-row: 1;
    font-family: var(--font-display);
    font-size: 40px;
    font-weight: 700;
    color: var(--color-brand);
    line-height: 1;
    letter-spacing: -0.02em;
    padding-top: 2px;
  }
  .step-head > div { grid-column: 2; grid-row: 1; }
  .step-title { font-family: var(--font-display); font-size: 24px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; }
  .step-sub { font-size: 15px; color: var(--color-ink-subtle); margin-top: 4px; }

  .model-list { display: flex; flex-direction: column; gap: var(--space-3); }
  .model-row-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: var(--color-surface-raised);
    padding: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card, 8px);
  }
  .model-row { display: flex; align-items: center; gap: var(--space-4); }
  .model-meta-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding-left: 28px;
    flex-wrap: wrap;
  }
  .local-model-select {
    flex: 1;
    max-width: 360px;
    font-size: 0.8rem;
  }
  .model-row :global(.input) { height: 46px; font-size: 14px; padding: 0 var(--space-4); }
  .model-index {
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 17px;
    color: var(--color-ink-ghost);
    width: 20px;
    text-align: center;
    flex-shrink: 0;
  }

  .choice-col { display: flex; flex-direction: column; gap: var(--space-3); }

  .vram-block-alert {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    background: color-mix(in srgb, #e05c5c 10%, transparent);
    border: 1px solid color-mix(in srgb, #e05c5c 40%, transparent);
    border-radius: var(--radius-card, 8px);
    padding: var(--space-3) var(--space-4);
    color: var(--color-text);
    font-size: 0.85rem;
  }
  .vram-block-alert > i {
    font-size: 1.1rem;
    color: #e05c5c;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .vram-block-alert p {
    margin: 4px 0 0 0;
    color: var(--color-text-muted);
    font-size: 0.8rem;
  }
  .choice {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-6);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-ink);
    font-family: var(--font-sans);
    cursor: pointer;
    transition: border-color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
    text-align: left;
  }
  .choice:hover:not(:disabled) { border-color: var(--color-border-strong); }
  .choice.selected { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .choice:disabled { opacity: 0.4; cursor: not-allowed; }
  .choice-label { display: block; font-weight: 600; font-size: 17px; }
  .choice-hint { display: block; font-size: 13px; color: var(--color-ink-subtle); margin-top: 3px; line-height: 1.45; }

  .verdict {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 500;
    margin-left: 88px; /* aligns with the step content column */
    margin-top: var(--space-6);
  }
  .verdict-green { background: var(--color-success-bg); color: var(--color-success); }
  .verdict-amber { background: var(--color-warn-bg); color: var(--color-warn); }
  .verdict-red   { background: var(--color-error-bg); color: var(--color-error); }

  .start-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding-top: var(--space-8);
    padding-left: 88px; /* aligns with the step content column */
  }
  .start-btn { min-width: 280px; padding: 15px var(--space-8); font-size: 13px; }
</style>
