<script lang="ts">
  /**
   * SimpleTrain — guided fine-tuning for non-technical users.
   * Three decisions: your data, a model (auto-recommended for the
   * hardware), and a quality preset. Everything else is derived.
   */
  import { onMount } from 'svelte'
  import { run, resetRun, pushToast, watchTelemetry, hwStore } from '../store.svelte'
  import { t } from '../i18n.svelte'
  import { api } from '../api'

  // ── Step 1: data ──────────────────────────────────────────────────────
  let dataKind = $state<'file' | 'hf'>('file')
  let filePath = $state('')
  let hfRepoId = $state('')
  let previewRows = $state<string[][] | null>(null)
  let previewLoading = $state(false)

  async function browseFile() {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        multiple: false,
        filters: [{ name: 'Datasets', extensions: ['jsonl', 'json', 'csv', 'parquet'] }],
      })
      if (typeof picked === 'string') {
        filePath = picked
        previewRows = null
      }
    } catch {
      pushToast('error', t('teach.pickerFailed'))
    }
  }

  function fileFormat(path: string): 'jsonl' | 'csv' | 'parquet' {
    const ext = path.split('.').pop()?.toLowerCase()
    if (ext === 'csv') return 'csv'
    if (ext === 'parquet') return 'parquet'
    return 'jsonl'
  }

  function dataSpec() {
    if (dataKind === 'file') {
      return {
        source: 'local',
        jsonl_path: null,
        fingerprint: null,
        local: {
          path: filePath,
          format: fileFormat(filePath),
          mapping: { prompt: 'prompt', completion: 'completion' },
        },
      }
    }
    return {
      source: 'hf',
      jsonl_path: null,
      fingerprint: null,
      hf: { repo_id: hfRepoId, split: 'train', revision: null },
    }
  }

  const dataReady = $derived(dataKind === 'file' ? filePath.length > 0 : hfRepoId.length > 2)

  async function previewData() {
    previewLoading = true
    previewRows = null
    try {
      previewRows = await api.previewDataset(dataSpec() as never, 3)
    } catch (e) {
      pushToast('error', `${t('teach.couldNotReadData')}: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      previewLoading = false
    }
  }

  // ── Step 2: model (auto-recommended) ──────────────────────────────────
  interface Recipe {
    model_id: string
    adapter_type: string
    quant_bits: number | null
    quality_tier: string
    estimated_vram_mb: number
    reason: string
  }
  let recipes = $state<Recipe[]>([])
  let selectedRecipe = $state<Recipe | null>(null)
  let loadingRecipes = $state(true)

  onMount(async () => {
    try {
      const hw = hwStore.info
        ? { accelerator: hwStore.info.backend, total_vram_mb: hwStore.info.vram_mb, total_ram_mb: hwStore.info.ram_mb }
        : undefined
      const raw = (await api.guiderRecommend(hw)) as unknown as Recipe[]
      recipes = raw.filter(r => r.model_id)
      selectedRecipe = recipes[0] ?? null
    } catch {
      pushToast('error', t('teach.recsFailed'))
    } finally {
      loadingRecipes = false
    }
  })

  function shortName(id: string) {
    return id.split('/').pop() ?? id
  }

  // ── Step 3: preset ────────────────────────────────────────────────────
  type Preset = 'fast' | 'balanced' | 'best'
  let preset = $state<Preset>('balanced')

  const presets: Record<Preset, { steps: number; rank: number; save: number }> = {
    fast:     { steps: 100, rank: 8,  save: 50  },
    balanced: { steps: 300, rank: 16, save: 100 },
    best:     { steps: 800, rank: 32, save: 200 },
  }

  // ── Start ─────────────────────────────────────────────────────────────
  let submitting = $state(false)

  async function start() {
    if (submitting || run.status === 'running' || !selectedRecipe) return
    submitting = true
    resetRun()

    const p = presets[preset]
    const config = {
      version: 1,
      train_mode: 'sft',
      model: selectedRecipe.model_id,
      backend: { kind: 'auto' },
      data: dataSpec(),
      adapter: {
        type: selectedRecipe.adapter_type || 'lora',
        rank: p.rank,
        alpha: p.rank * 2,
        dropout: 0.05,
        target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        quant_bits: selectedRecipe.quant_bits ?? null,
      },
      optim: { learning_rate: 2e-4, schedule: 'cosine', warmup_steps: 20, weight_decay: 0.0, grad_accumulation_steps: 8 },
      train: { max_steps: p.steps, batch_size: 2, max_seq_len: 2048, save_every: p.save, packing: false },
      output: { adapter_path: `runs/adapter-${Date.now()}` },
    }

    try {
      const opId = await api.startTrain(config as never)
      run.opId = opId; run.kind = 'train'; run.status = 'running'; run.startedAt = Date.now()
      pushToast('success', t('teach.started'))
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
      <h1 class="text-display">{t('teach.title')}</h1>
      <p class="text-small">{t('teach.subtitle')}</p>
    </div>
  </div>

  <div class="simple-scroll">
    <div class="simple-flow">

      <!-- 01 · Data -->
      <section class="step">
        <div class="step-head">
          <span class="step-num">01</span>
          <div>
            <div class="step-title">{t('teach.step1.title')}</div>
            <div class="step-sub">{t('teach.step1.sub')}</div>
          </div>
        </div>

        <div class="choice-row">
          <button class="choice" class:selected={dataKind === 'file'} onclick={() => (dataKind = 'file')}>
            <i class="bi bi-file-earmark-text"></i>
            <span>{t('teach.fileChoice')}</span>
          </button>
          <button class="choice" class:selected={dataKind === 'hf'} onclick={() => (dataKind = 'hf')}>
            <i class="bi bi-cloud-download"></i>
            <span>{t('teach.hfChoice')}</span>
          </button>
        </div>

        {#if dataKind === 'file'}
          <div class="file-row">
            <button class="btn btn-secondary" onclick={browseFile}>{t('teach.chooseFile')}</button>
            {#if filePath}
              <span class="file-name text-mono">{filePath.split(/[\\/]/).pop()}</span>
            {:else}
              <span class="text-small">{t('teach.fileHint')}</span>
            {/if}
          </div>
        {:else}
          <input class="input" placeholder={t('teach.hfPlaceholder')} bind:value={hfRepoId} />
        {/if}

        {#if dataReady}
          <button class="btn btn-ghost btn-sm" onclick={previewData} disabled={previewLoading}>
            {#if previewLoading}<span class="spinner"></span>{/if}
            {t('teach.checkData')}
          </button>
        {/if}
        {#if previewRows && previewRows.length > 1}
          <div class="preview">
            {#each previewRows.slice(1) as row}
              <div class="preview-row">
                <div class="preview-q">{row[0]}</div>
                <div class="preview-a">{row[1]}</div>
              </div>
            {/each}
          </div>
        {/if}
      </section>

      <!-- 02 · Model -->
      <section class="step">
        <div class="step-head">
          <span class="step-num">02</span>
          <div>
            <div class="step-title">{t('teach.step2.title')}</div>
            <div class="step-sub">{t('teach.step2.sub')}</div>
          </div>
        </div>

        {#if loadingRecipes}
          <div class="empty-state" style="min-height:60px"><span class="spinner"></span></div>
        {:else if recipes.length === 0}
          <p class="text-small">{t('teach.noModelFits')}</p>
        {:else}
          <div class="model-grid">
            {#each recipes.slice(0, 3) as recipe, i}
              <button
                class="model-card"
                class:selected={selectedRecipe === recipe}
                onclick={() => (selectedRecipe = recipe)}
              >
                {#if i === 0}<span class="badge badge-brand model-badge">{t('teach.recommended')}</span>{/if}
                <div class="model-name">{shortName(recipe.model_id)}</div>
                <div class="model-hint">{recipe.reason}</div>
              </button>
            {/each}
          </div>
        {/if}
      </section>

      <!-- 03 · Quality -->
      <section class="step">
        <div class="step-head">
          <span class="step-num">03</span>
          <div>
            <div class="step-title">{t('teach.step3.title')}</div>
            <div class="step-sub">{t('teach.step3.sub')}</div>
          </div>
        </div>

        <div class="choice-row">
          {#each Object.keys(presets) as key}
            <button class="choice choice-tall" class:selected={preset === key} onclick={() => (preset = key as never)}>
              <span class="choice-label">{t(`teach.preset.${key}`)}</span>
              <span class="choice-hint">{t(`teach.preset.${key}Hint`)}</span>
            </button>
          {/each}
        </div>
      </section>

      <!-- Start -->
      <div class="start-row">
        <button
          class="btn btn-primary btn-lg start-btn"
          onclick={start}
          disabled={submitting || !dataReady || !selectedRecipe || run.status === 'running'}
        >
          {#if submitting}<span class="spinner"></span>{/if}
          {t('teach.start')}
        </button>
        {#if run.status === 'running'}
          <span class="text-small">{t('teach.runInProgress')}</span>
        {:else if !dataReady}
          <span class="text-small">{t('teach.pickExamplesFirst')}</span>
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

  /* Two-column grid: the number lives in a fixed left rail; the title and
     every content block share the second column, so they all align to one
     vertical line. */
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

  .choice-row { display: flex; gap: var(--space-4); flex-wrap: wrap; }
  .choice {
    flex: 1;
    min-width: 220px;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-6);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-ink);
    font-family: var(--font-sans);
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: border-color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
    text-align: left;
  }
  .choice:hover { border-color: var(--color-border-strong); }
  .choice.selected { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .choice i { font-size: 22px; color: var(--color-ink-subtle); }
  .choice.selected i { color: var(--color-brand); }

  .choice-tall { flex-direction: column; align-items: flex-start; gap: var(--space-2); }
  .choice-label { font-weight: 600; font-size: 17px; }
  .choice-hint { font-size: 13px; color: var(--color-ink-subtle); font-weight: 400; line-height: 1.45; }

  .file-row { display: flex; align-items: center; gap: var(--space-4); flex-wrap: wrap; }
  .file-name { font-size: 13px; color: var(--color-ink); }
  .step :global(.input) { height: 46px; font-size: 14px; padding: 0 var(--space-4); max-width: 560px; }

  .preview { display: flex; flex-direction: column; gap: var(--space-2); }
  .preview-row {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    background: var(--color-surface);
  }
  .preview-q { font-size: 13px; font-weight: 600; margin-bottom: 2px; }
  .preview-a { font-size: 13px; color: var(--color-ink-subtle); }

  .model-grid { display: flex; gap: var(--space-4); flex-wrap: wrap; }
  .model-card {
    flex: 1;
    min-width: 260px;
    position: relative;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-6);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    cursor: pointer;
    text-align: left;
    font-family: var(--font-sans);
    color: var(--color-ink);
    transition: border-color var(--dur-fast) var(--ease), background var(--dur-fast) var(--ease);
  }
  .model-card:hover { border-color: var(--color-border-strong); }
  .model-card.selected { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .model-badge { position: absolute; top: var(--space-3); right: var(--space-3); }
  .model-name { font-family: var(--font-display); font-size: 18px; font-weight: 600; padding-right: 100px; }
  .model-hint { font-size: 13px; color: var(--color-ink-subtle); line-height: 1.45; }

  .start-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding-top: var(--space-8);
    padding-left: 88px; /* aligns with the step content column */
  }
  .start-btn { min-width: 280px; padding: 15px var(--space-8); font-size: 13px; }
</style>
