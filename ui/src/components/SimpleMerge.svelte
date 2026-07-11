<script lang="ts">
  /**
   * SimpleMerge — guided model merging for non-technical users.
   * Pick 2-3 models, say what you want, and the right method, weights and
   * base model are chosen automatically. Compatibility is checked before
   * anything starts.
   */
  import { run, resetRun, pushToast, watchTelemetry } from '../store.svelte'
  import { t } from '../i18n.svelte'
  import { api } from '../api'
  import type { CompatResult } from '../types'
  import catalogData from '../../../crates/sytra-contracts/src/catalog.json'

  const catalogIds: string[] = (catalogData as { model_id: string }[]).map(m => m.model_id)

  // ── Step 1: models ────────────────────────────────────────────────────
  let models = $state<string[]>(['', ''])

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
            <div class="model-row">
              <span class="model-index">{i + 1}</span>
              <input
                class="input input-mono"
                placeholder={i === 0 ? 'e.g. mistralai/Mistral-7B-v0.1' : 'e.g. org/knowledge-ft'}
                bind:value={models[i]}
                list="catalog-models"
              />
              {#if models.length > 2}
                <button class="btn btn-ghost btn-icon" onclick={() => removeModel(i)} aria-label="Remove model">
                  <i class="bi bi-x-lg"></i>
                </button>
              {/if}
            </div>
          {/each}
          <datalist id="catalog-models">
            {#each catalogIds as id}<option value={id}></option>{/each}
          </datalist>
        </div>
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
          disabled={submitting || checking || !modelsReady || run.status === 'running'}
        >
          {#if submitting || checking}<span class="spinner"></span>{/if}
          {t('combine.start')}
        </button>
        {#if run.status === 'running'}
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
  .model-row { display: flex; align-items: center; gap: var(--space-4); }
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
