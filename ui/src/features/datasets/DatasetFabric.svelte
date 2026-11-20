<script lang="ts">
  import { api } from '../../api'
  import { t } from '../../i18n.svelte'
  import { pushToast, setTab, trainFormStore, uiMode, toggleUiMode } from '../../store.svelte'

  type Source = { id: number; path: string; weight: number }
  type Result = Awaited<ReturnType<typeof api.prepareDataset>>
  let mode = $state<'merge' | 'split'>('merge')
  let sources = $state<Source[]>([{ id: 1, path: '', weight: 1 }, { id: 2, path: '', weight: 1 }])
  let nextId = 3
  let splitInput = $state('')
  let outputDir = $state('')
  let weighted = $state(false)
  let totalSamples = $state(1000)
  let shardRows = $state(0)
  let exactDuplicates = $state(false)
  let nearDuplicates = $state(false)
  let repetitive = $state(false)
  let dropInvalid = $state(false)
  let sanitizeSyntax = $state(false)
  let semanticEnabled = $state(false)
  let semanticModel = $state('')
  let semanticDensity = $state(0.5)
  let limitContext = $state(false)
  let tokenizerPath = $state('')
  let maxContextTokens = $state(2048)
  let minTokens = $state(0)
  let similarityPercent = $state(85)
  let benchmarkPaths = $state('')
  let validationPercent = $state(10)
  let seed = $state(3407)
  let busy = $state(false)
  let result = $state<Result | null>(null)
  let completedMode = $state<'merge' | 'split'>('merge')

  const splitPresets: { pct: number; label: string; hint: string }[] = [
    { pct: 5, label: 'datasets.minimal', hint: 'datasets.minimalHint' },
    { pct: 10, label: 'datasets.balanced', hint: 'datasets.balancedHint' },
    { pct: 20, label: 'datasets.thorough', hint: 'datasets.thoroughHint' },
  ]

  async function chooseInput(source?: Source) {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({ multiple: false, filters: [{ name: t('nav.datasets'), extensions: ['jsonl', 'parquet'] }] })
      if (typeof picked === 'string') {
        if (source) source.path = picked
        else splitInput = picked
      }
    } catch (error) {
      pushToast('error', t('datasets.chooseError', { error: error instanceof Error ? error.message : String(error) }))
    }
  }

  async function prepare() {
    if (busy) return
    const advanced = uiMode.advanced
    const paths = sources.map(source => source.path.trim())
    const targetOutputDir = outputDir.trim() || (advanced ? '' : `prepared-dataset-${Date.now()}`)
    if (!targetOutputDir || (mode === 'merge' ? paths.some(path => !path) : !splitInput.trim())) {
      pushToast('error', advanced ? t('datasets.requiredAdvanced') : t('datasets.required'))
      return
    }
    if (advanced && weighted && mode === 'merge' && (!Number.isInteger(totalSamples) || totalSamples <= 0 || sources.some(source => !Number.isFinite(source.weight) || source.weight <= 0))) {
      pushToast('error', t('datasets.weightsError'))
      return
    }
    if (advanced && (!Number.isSafeInteger(seed) || !Number.isSafeInteger(shardRows) || shardRows < 0 || !Number.isFinite(validationPercent) || validationPercent <= 0 || validationPercent >= 100)) {
      pushToast('error', t('datasets.splitError'))
      return
    }
    if (advanced && mode === 'merge' && (!Number.isSafeInteger(minTokens) || minTokens < 0 || !Number.isFinite(similarityPercent) || similarityPercent <= 0 || similarityPercent > 100)) {
      pushToast('error', t('datasets.curationError'))
      return
    }
    if (advanced && mode === 'merge' && semanticEnabled && (!semanticModel.trim() || !Number.isFinite(semanticDensity) || semanticDensity < 0 || semanticDensity > 1)) {
      pushToast('error', t('datasets.semanticError'))
      return
    }
    if (advanced && mode === 'merge' && limitContext && (!tokenizerPath.trim() || !Number.isInteger(maxContextTokens) || maxContextTokens < 2 || maxContextTokens > 1048576)) {
      pushToast('error', t('datasets.tokenizerError'))
      return
    }
    const operation = mode
    const benchmarks = advanced ? benchmarkPaths.split(/\r?\n/).map(path => path.trim()).filter(Boolean) : []
    if (benchmarks.length > 50) {
      pushToast('error', t('datasets.benchmarkError'))
      return
    }
    const options: Record<string, unknown> = { output_dir: targetOutputDir, seed: advanced ? seed : 3407 }
    if (operation === 'merge') {
      options.inputs = paths
      options.shard_rows = advanced ? shardRows : 0
      if (advanced && weighted) {
        options.weights = sources.map(source => source.weight)
        options.total_samples = totalSamples
      }
      if (advanced) {
        if (exactDuplicates || nearDuplicates || repetitive || dropInvalid || sanitizeSyntax || semanticEnabled || limitContext || minTokens > 0 || benchmarks.length > 0) {
          const useNativeContext = limitContext || (minTokens > 0 && Boolean(tokenizerPath.trim()))
          options.curation = {
            exact_duplicates: exactDuplicates, near_duplicates: nearDuplicates,
            repetitive, drop_invalid: dropInvalid, sanitize_syntax: sanitizeSyntax,
            similarity: similarityPercent / 100, min_tokens: minTokens,
            benchmark_files: benchmarks,
            ...(semanticEnabled ? { semantic: { model_path: semanticModel.trim(), min_density: semanticDensity } } : {}),
            ...(useNativeContext ? {
              context: {
                tokenizer_path: tokenizerPath.trim(),
                ...(limitContext ? { max_seq_length: maxContextTokens } : {}),
                ...(minTokens > 0 ? { min_tokens: minTokens } : {}),
              },
            } : {}),
          }
        }
      } else if (exactDuplicates || nearDuplicates || repetitive || dropInvalid) {
        options.curation = {
          exact_duplicates: exactDuplicates,
          near_duplicates: nearDuplicates,
          repetitive,
          drop_invalid: dropInvalid,
          similarity: 0.85,
          min_tokens: 0,
          benchmark_files: [],
        }
      }
    } else {
      options.input = splitInput.trim()
      options.validation_fraction = validationPercent / 100
    }
    busy = true
    result = null
    try {
      result = await api.prepareDataset(operation, options)
      completedMode = operation
      pushToast('success', t('datasets.success'))
    } catch (error) {
      pushToast('error', t('datasets.failure', { error: error instanceof Error ? error.message : String(error) }))
    } finally {
      busy = false
    }
  }

  function guidedRemovedCount(res: Result): number {
    if (!res.curation) return 0
    const r = res.curation.removed
    return (r.exact_duplicate ?? 0) + (r.near_duplicate ?? 0) + (r.repetitive ?? 0) + (r.invalid_structure ?? 0)
  }

  function useForTraining() {
    if (!result) return
    const file = completedMode === 'split' ? 'train.jsonl' : result.shards?.length === 1 ? result.shards[0].path : 'manifest.json'
    trainFormStore.dataSplitSource = 'local'
    trainFormStore.localPath = `${result.output_dir}/${file}`
    trainFormStore.localFormat = file.endsWith('.json') ? 'manifest' : 'jsonl'
    trainFormStore.validationPath = completedMode === 'split' ? `${result.output_dir}/validation.jsonl` : ''
    setTab('train')
  }
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">{t('nav.datasets')}</h1>
      {#if uiMode.advanced}
        <p class="text-small">{t('datasets.advancedIntro')}</p>
      {:else}
        <p class="text-small">{t('datasets.intro')}</p>
      {/if}
    </div>
    {#if uiMode.advanced}
      <div class="segmented-control" role="group" aria-label={t('datasets.operation')}>
        <button type="button" class:active={mode === 'merge'} aria-pressed={mode === 'merge'} disabled={busy} onclick={() => { mode = 'merge'; result = null }}>{t('datasets.combine')}</button>
        <button type="button" class:active={mode === 'split'} aria-pressed={mode === 'split'} disabled={busy} onclick={() => { mode = 'split'; result = null }}>{t('datasets.split')}</button>
      </div>
    {/if}
  </div>

  {#if !uiMode.advanced}
    <!-- ── Guided & simple mode (for non-technical users) ───────────────── -->
    <div class="simple-scroll">
      <div class="simple-flow">
        <!-- 01 · Goal -->
        <section class="step">
          <div class="step-head">
            <span class="step-num">01</span>
            <div>
              <div class="step-title">{t('datasets.goal')}</div>
              <div class="step-sub">{t('datasets.goalHint')}</div>
            </div>
          </div>

          <div class="choice-row">
            <button
              type="button"
              class="choice card"
              class:selected={mode === 'merge'}
              disabled={busy}
              onclick={() => { mode = 'merge'; result = null }}
            >
              <i class="bi bi-layers"></i>
              <div>
                <span class="choice-label">{t('datasets.clean')}</span>
                <span class="choice-hint">{t('datasets.cleanHint')}</span>
              </div>
            </button>
            <button
              type="button"
              class="choice card"
              class:selected={mode === 'split'}
              disabled={busy}
              onclick={() => { mode = 'split'; result = null }}
            >
              <i class="bi bi-scissors"></i>
              <div>
                <span class="choice-label">{t('datasets.splitForTest')}</span>
                <span class="choice-hint">{t('datasets.splitForTestHint')}</span>
              </div>
            </button>
          </div>
        </section>

        <!-- 02 · Your files -->
        <section class="step">
          <div class="step-head">
            <span class="step-num">02</span>
            <div>
              <div class="step-title">{mode === 'merge' ? t('datasets.yourFiles') : t('datasets.yourFile')}</div>
              <div class="step-sub">
                {#if mode === 'merge'}
                  {t('datasets.chooseMerge')}
                {:else}
                  {t('datasets.chooseSplit')}
                {/if}
              </div>
            </div>
          </div>

          {#if mode === 'merge'}
            <div class="guided-file-list">
              {#each sources as source, index (source.id)}
                <div class="guided-file-row">
                  <span class="guided-file-num">{index + 1}</span>
                  <input
                    id="guided-source-{source.id}"
                    class="input input-mono"
                    bind:value={source.path}
                    disabled={busy}
                    placeholder={t('datasets.workspaceFile')}
                    aria-label={t('datasets.exampleFile', { number: index + 1 })}
                  />
                  <button type="button" class="btn btn-secondary" disabled={busy} onclick={() => chooseInput(source)}>{t('datasets.chooseFile')}</button>
                  {#if sources.length > 1}
                    <button
                      type="button"
                      class="btn btn-ghost btn-icon"
                      aria-label={t('datasets.removeFile', { number: index + 1 })}
                      disabled={busy}
                      onclick={() => sources = sources.filter(item => item.id !== source.id)}
                    >
                      <i class="bi bi-x-lg"></i>
                    </button>
                  {/if}
                </div>
              {/each}
            </div>
            <div class="guided-file-footer">
              <button type="button" class="btn btn-ghost btn-sm" disabled={busy || sources.length >= 150} onclick={() => sources = [...sources, { id: nextId++, path: '', weight: 1 }]}>
                <i class="bi bi-plus-lg"></i> {t('datasets.addFile')}
              </button>
            </div>
          {:else}
            <div class="guided-file-row">
              <input
                id="guided-split-input"
                class="input input-mono"
                bind:value={splitInput}
                disabled={busy}
                placeholder={t('datasets.workspaceFile')}
                aria-label={t('datasets.splitFile')}
              />
              <button type="button" class="btn btn-secondary" disabled={busy} onclick={() => chooseInput()}>{t('datasets.chooseFile')}</button>
            </div>
          {/if}
        </section>

        <!-- 03 · Cleanup or Split percentage -->
        <section class="step">
          <div class="step-head">
            <span class="step-num">03</span>
            <div>
              {#if mode === 'merge'}
                <div class="step-title">{t('datasets.cleanup')}</div>
                <div class="step-sub">{t('datasets.cleanupHint')}</div>
              {:else}
                <div class="step-title">{t('datasets.testSize')}</div>
                <div class="step-sub">{t('datasets.testSizeHint')}</div>
              {/if}
            </div>
          </div>

          {#if mode === 'merge'}
            <div class="guided-check-grid">
              <label class="guided-check-card card" class:checked={exactDuplicates}>
                <input type="checkbox" bind:checked={exactDuplicates} disabled={busy} />
                <div>
                  <span class="guided-check-title">{t('datasets.identical')}</span>
                  <span class="guided-check-hint">{t('datasets.identicalHint')}</span>
                </div>
              </label>
              <label class="guided-check-card card" class:checked={nearDuplicates}>
                <input type="checkbox" bind:checked={nearDuplicates} disabled={busy} />
                <div>
                  <span class="guided-check-title">{t('datasets.similar')}</span>
                  <span class="guided-check-hint">{t('datasets.similarHint')}</span>
                </div>
              </label>
              <label class="guided-check-card card" class:checked={repetitive}>
                <input type="checkbox" bind:checked={repetitive} disabled={busy} />
                <div>
                  <span class="guided-check-title">{t('datasets.repetitive')}</span>
                  <span class="guided-check-hint">{t('datasets.repetitiveHint')}</span>
                </div>
              </label>
              <label class="guided-check-card card" class:checked={dropInvalid}>
                <input type="checkbox" bind:checked={dropInvalid} disabled={busy} />
                <div>
                  <span class="guided-check-title">{t('datasets.incomplete')}</span>
                  <span class="guided-check-hint">{t('datasets.incompleteHint')}</span>
                </div>
              </label>
            </div>
          {:else}
            <div class="choice-row">
              {#each splitPresets as preset}
                <button
                  type="button"
                  class="choice choice-tall card"
                  class:selected={validationPercent === preset.pct}
                  disabled={busy}
                  onclick={() => (validationPercent = preset.pct)}
                >
                  <span class="choice-label">{t(preset.label)}</span>
                  <span class="choice-hint">{t(preset.hint)}</span>
                </button>
              {/each}
            </div>
          {/if}

          <div class="guided-output-field">
            <label class="field-label" for="guided-dataset-output">{t('datasets.outputFolder')}</label>
            <input
              id="guided-dataset-output"
              class="input input-mono"
              bind:value={outputDir}
              disabled={busy}
              placeholder={t('datasets.autoOutput')}
            />
          </div>
        </section>

        <!-- Start & Result -->
        <div class="guided-start-block">
          <div class="start-row">
            <button type="button" class="btn btn-primary btn-lg start-btn" disabled={busy} onclick={prepare}>
              {#if busy}<span class="spinner"></span>{/if}
              {busy ? t('datasets.preparing') : mode === 'merge' ? t('datasets.prepare') : t('datasets.splitDataset')}
            </button>
            <button type="button" class="btn btn-ghost btn-sm" disabled={busy} onclick={toggleUiMode}>
              {t('datasets.switchAdvanced')}
            </button>
          </div>

          {#if busy}
            <p class="text-small" role="status">{t('datasets.preparingHint')}</p>
          {/if}

          {#if result}
            <div class="result" role="status">
              <strong>{t('datasets.ready')}</strong>
              {#if completedMode === 'merge'}
                <p>{t('datasets.saved', { rows: result.rows ?? 0, path: result.output_dir })}</p>
                {#if guidedRemovedCount(result) > 0}
                  <p>{t('datasets.cleaned', { rows: guidedRemovedCount(result) })}</p>
                {/if}
              {:else}
                <p>{t('datasets.splitSaved', { path: result.output_dir })}</p>
              {/if}
              <button type="button" class="btn btn-secondary" onclick={useForTraining}>{t('datasets.teach')}</button>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {:else}
    <!-- ── Advanced mode (full Dataset Fabric controls) ──────────────────── -->
    <div class="dataset-scroll">
      <div class="dataset-content">
        <div class="dataset-grid" class:split-grid={mode === 'split'}>
          <div class="dataset-col">
            {#if mode === 'merge'}
              <section class="card">
                <div class="card-header"><span class="text-label">{t('datasets.inputs')}</span></div>
                <div class="card-body fields">
                  <p class="text-small">{t('datasets.inputsHint')}</p>
                  {#each sources as source, index (source.id)}
                    <div class="source-row" class:has-weight={weighted}>
                      <div class="field source-path">
                        <label class="field-label" for="dataset-source-{source.id}">{t('datasets.inputNumber', { number: index + 1 })}</label>
                        <input id="dataset-source-{source.id}" class="input input-mono" bind:value={source.path} disabled={busy} placeholder={t('datasets.workspaceDataset')} />
                      </div>
                      {#if weighted}
                        <div class="field weight">
                          <label class="field-label" for="dataset-weight-{source.id}">{t('datasets.weight')}</label>
                          <input id="dataset-weight-{source.id}" class="input" type="number" min="0.01" step="0.1" bind:value={source.weight} disabled={busy} />
                        </div>
                      {/if}
                      <div class="source-actions">
                        <button type="button" class="btn btn-secondary" disabled={busy} onclick={() => chooseInput(source)}>{t('datasets.browse')}</button>
                        <button type="button" class="btn btn-ghost" aria-label={t('datasets.removeDataset', { number: index + 1 })} disabled={busy || sources.length === 1} onclick={() => sources = sources.filter(item => item.id !== source.id)}>{t('datasets.remove')}</button>
                      </div>
                    </div>
                  {/each}
                  <div class="inline-actions">
                    <button type="button" class="btn btn-secondary" disabled={busy || sources.length >= 150} onclick={() => sources = [...sources, { id: nextId++, path: '', weight: 1 }]}>{t('datasets.addDataset')}</button>
                    <label class="check-row"><input type="checkbox" bind:checked={weighted} disabled={busy} /> {t('datasets.weighted')}</label>
                  </div>
                  <div class="two-col-fields">
                    {#if weighted}
                      <div class="field">
                        <label class="field-label" for="dataset-total">{t('datasets.total')}</label>
                        <input id="dataset-total" class="input" type="number" min="1" step="1" bind:value={totalSamples} disabled={busy} />
                        <span class="field-hint">{t('datasets.totalHint')}</span>
                      </div>
                    {/if}
                    <div class="field">
                      <label class="field-label" for="dataset-shards">{t('datasets.shards')}</label>
                      <input id="dataset-shards" class="input" type="number" min="0" step="1" bind:value={shardRows} disabled={busy} />
                      <span class="field-hint">{t('datasets.shardsHint')}</span>
                    </div>
                  </div>
                </div>
              </section>
            {:else}
              <section class="card">
                <div class="card-header"><span class="text-label">{t('datasets.splitDataset')}</span></div>
                <div class="card-body fields">
                  <div class="field">
                    <label class="field-label" for="split-input">{t('datasets.splitInput')}</label>
                    <div class="source-row">
                      <input id="split-input" class="input input-mono source-path" bind:value={splitInput} disabled={busy} placeholder={t('datasets.workspaceDataset')} />
                      <button type="button" class="btn btn-secondary" disabled={busy} onclick={() => chooseInput()}>{t('datasets.browse')}</button>
                    </div>
                  </div>
                  <div class="field">
                    <label class="field-label" for="split-percent">{t('datasets.validation')}</label>
                    <input id="split-percent" class="input" type="number" min="1" max="99" step="1" bind:value={validationPercent} disabled={busy} />
                    <span class="field-hint">{t('datasets.validationHint')}</span>
                  </div>
                </div>
              </section>
            {/if}

            <section class="card">
              <div class="card-header"><span class="text-label">{t('datasets.output')}</span></div>
              <div class="card-body fields">
                <div class="two-col-fields">
                  <div class="field">
                    <label class="field-label" for="dataset-output">{t('datasets.newDirectory')}</label>
                    <input id="dataset-output" class="input input-mono" bind:value={outputDir} disabled={busy} placeholder={t('datasets.directoryExample')} />
                    <span class="field-hint">{t('datasets.directoryHint')}</span>
                  </div>
                  <div class="field">
                    <label class="field-label" for="dataset-seed">{t('datasets.seed')}</label>
                    <input id="dataset-seed" class="input" type="number" step="1" bind:value={seed} disabled={busy} />
                  </div>
                </div>
                <button type="button" class="btn btn-primary" disabled={busy} onclick={prepare}>{busy ? t('datasets.preparing') : mode === 'merge' ? t('datasets.combineDatasets') : t('datasets.createFiles')}</button>
                {#if busy}<p class="text-small" role="status">{t('datasets.preparingAdvanced')}</p>{/if}
                {#if result}
                  <div class="result" role="status">
                    <strong>{t('datasets.readyPath', { path: result.output_dir })}</strong>
                    {#if completedMode === 'merge'}
                      <p>{t('datasets.rowCount', { rows: result.rows ?? 0, files: result.shards?.length ?? 0 })}</p>
                      {#if result.curation}
                        <p>{t('datasets.removed', { exact: result.curation.removed.exact_duplicate, near: result.curation.removed.near_duplicate, repetitive: result.curation.removed.repetitive, short: result.curation.removed.too_short, counting: t(result.curation.token_counting === 'native_tokenizer' ? 'datasets.nativeTokenizer' : 'datasets.lexicalCount'), overlap: result.curation.removed.benchmark_overlap })}</p>
                        {#if result.curation.removed.syntax_sanitized !== undefined}<p>{t('datasets.syntaxCount', { rows: result.curation.removed.syntax_sanitized })}</p>{/if}
                        {#if result.curation.removed.invalid_structure !== undefined}<p>{t('datasets.invalidCount', { rows: result.curation.removed.invalid_structure })}</p>{/if}
                        {#if result.curation.removed.too_long !== undefined}<p>{t('datasets.longCount', { rows: result.curation.removed.too_long })}</p>{/if}
                        {#if result.curation.removed.semantic_rejected !== undefined}<p>{t('datasets.semanticCount', { rows: result.curation.removed.semantic_rejected })}</p>{/if}
                      {/if}
                      {#if (result.shards?.length ?? 0) > 1}<p>{t('datasets.manifest')}</p>{/if}
                    {:else}
                      <p>{t('datasets.filesCreated')}</p>
                    {/if}
                    <button type="button" class="btn btn-secondary" onclick={useForTraining}>{t('datasets.use')}</button>
                  </div>
                {/if}
              </div>
            </section>
          </div>

          {#if mode === 'merge'}
            <div class="dataset-col">
              <section class="card">
                <div class="card-header"><span class="text-label">{t('datasets.filters')}</span></div>
                <div class="card-body fields">
                  <div class="filter-list">
                    <label class="check-row"><input type="checkbox" bind:checked={exactDuplicates} disabled={busy} /> <span>{t('datasets.exact')}</span></label>
                    <label class="check-row"><input type="checkbox" bind:checked={nearDuplicates} disabled={busy} /> <span>{t('datasets.near')}</span></label>
                    {#if nearDuplicates}
                      <div class="field nested-field">
                        <label class="field-label" for="dataset-similarity">{t('datasets.similarity')}</label>
                        <input id="dataset-similarity" class="input" type="number" min="1" max="100" step="1" bind:value={similarityPercent} disabled={busy} />
                        <span class="field-hint">{t('datasets.similarityHint')}</span>
                      </div>
                    {/if}
                    <label class="check-row"><input type="checkbox" bind:checked={repetitive} disabled={busy} /> <span>{t('datasets.filterRepetitive')}</span></label>
                    <label class="check-row"><input type="checkbox" bind:checked={dropInvalid} disabled={busy} /> <span>{t('datasets.filterInvalid')}</span></label>
                    {#if dropInvalid}<span class="field-hint nested-hint">{t('datasets.invalidHint')}</span>{/if}
                    <label class="check-row"><input type="checkbox" bind:checked={sanitizeSyntax} disabled={busy} /> <span>{t('datasets.sanitize')}</span></label>
                    {#if sanitizeSyntax}<span class="field-hint nested-hint">{t('datasets.sanitizeHint')}</span>{/if}
                    <label class="check-row"><input type="checkbox" bind:checked={limitContext} disabled={busy} /> <span>{t('datasets.limitContext')}</span></label>
                    {#if limitContext || minTokens > 0}
                      <div class="two-col-fields nested-field">
                        <div class="field">
                          <label class="field-label" for="context-tokenizer">{t(limitContext ? 'datasets.tokenizer' : 'datasets.optionalTokenizer')}</label>
                          <input id="context-tokenizer" class="input input-mono" bind:value={tokenizerPath} disabled={busy} placeholder={t('datasets.workspacePath')} />
                          <span class="field-hint">{t('datasets.tokenizerHint')}</span>
                        </div>
                        {#if limitContext}
                          <div class="field">
                            <label class="field-label" for="context-limit">{t('datasets.maximumTokens')}</label>
                            <input id="context-limit" class="input" type="number" min="2" max="1048576" step="1" bind:value={maxContextTokens} disabled={busy} />
                          </div>
                        {/if}
                      </div>
                    {/if}
                    <label class="check-row"><input type="checkbox" bind:checked={semanticEnabled} disabled={busy} /> <span>{t('datasets.semantic')}</span></label>
                    {#if semanticEnabled}
                      <div class="two-col-fields nested-field">
                        <div class="field">
                          <label class="field-label" for="semantic-model">{t('datasets.semanticModel')}</label>
                          <input id="semantic-model" class="input input-mono" bind:value={semanticModel} disabled={busy} placeholder={t('datasets.workspacePath')} />
                          <span class="field-hint">{t('datasets.semanticHint')}</span>
                        </div>
                        <div class="field">
                          <label class="field-label" for="semantic-density">{t('datasets.density')}</label>
                          <input id="semantic-density" class="input" type="number" min="0" max="1" step="0.05" bind:value={semanticDensity} disabled={busy} />
                        </div>
                      </div>
                    {/if}
                  </div>
                  <div class="field">
                    <label class="field-label" for="dataset-min-tokens">{t('datasets.minimumTokens', { counting: t(tokenizerPath.trim() ? 'datasets.nativeTokenizer' : 'datasets.lexicalUnits') })}</label>
                    <input id="dataset-min-tokens" class="input" type="number" min="0" step="1" bind:value={minTokens} disabled={busy} />
                    <span class="field-hint">{t(tokenizerPath.trim() ? 'datasets.minimumNativeHint' : 'datasets.minimumLexicalHint')}</span>
                  </div>
                  <div class="field">
                    <label class="field-label" for="dataset-benchmarks">{t('datasets.benchmarks')}</label>
                    <textarea id="dataset-benchmarks" class="input input-mono benchmark-area" rows="3" bind:value={benchmarkPaths} disabled={busy} placeholder={t('datasets.benchmarkPlaceholder')}></textarea>
                    <span class="field-hint">{t('datasets.benchmarkHint')}</span>
                  </div>
                </div>
              </section>
            </div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  /* ── Guided & simple styles (aligned with Teach & Combine) ─────────────── */
  .simple-scroll { flex: 1; overflow-y: auto; min-height: 0; }
  .simple-flow {
    width: 100%;
    max-width: 1280px;
    padding: var(--space-10) var(--space-10) 64px;
    display: flex;
    flex-direction: column;
    gap: 0;
  }
  .step {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr);
    row-gap: var(--space-5);
    padding: var(--space-10) 0;
    border-bottom: 1px solid var(--color-border);
  }
  .step:first-child { padding-top: 0; }
  .step > :global(*) { grid-column: 2; }
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
    min-width: 240px;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-6);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    color: var(--color-ink);
    font-family: inherit;
    cursor: pointer;
    transition: border-color var(--anim-duration) var(--ease), background var(--anim-duration) var(--ease);
    text-align: left;
  }
  .choice:hover:not(:disabled) { border-color: var(--color-border-strong); }
  .choice.selected { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .choice:disabled { opacity: 0.45; cursor: not-allowed; }
  .choice i { font-size: 22px; color: var(--color-ink-subtle); flex-shrink: 0; }
  .choice.selected i { color: var(--color-brand); }
  .choice-tall { flex-direction: column; align-items: flex-start; gap: var(--space-1); min-width: 190px; }
  .choice-label { display: block; font-weight: 600; font-size: 17px; }
  .choice-hint { display: block; font-size: 13px; color: var(--color-ink-subtle); margin-top: 3px; line-height: 1.45; }

  .guided-file-list { display: flex; flex-direction: column; gap: var(--space-3); }
  .guided-file-row { display: flex; align-items: center; gap: var(--space-3); }
  .guided-file-row .input { flex: 1; min-width: 0; height: 42px; font-size: 13px; }
  .guided-file-row .btn { height: 42px; flex-shrink: 0; }
  .guided-file-num {
    width: 24px;
    text-align: center;
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 16px;
    color: var(--color-ink-ghost);
    flex-shrink: 0;
  }
  .guided-file-footer { display: flex; align-items: center; }

  .guided-check-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  @media (max-width: 720px) { .guided-check-grid { grid-template-columns: 1fr; } }
  .guided-check-card {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-4);
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    cursor: pointer;
    transition: border-color var(--anim-duration) var(--ease), background var(--anim-duration) var(--ease);
  }
  .guided-check-card:hover { border-color: var(--color-border-strong); }
  .guided-check-card.checked { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .guided-check-card input[type="checkbox"] {
    accent-color: var(--color-brand);
    width: 16px;
    height: 16px;
    margin: 0;
    flex-shrink: 0;
    cursor: pointer;
  }
  .guided-check-title { display: block; font-size: 14px; font-weight: 600; color: var(--color-ink); line-height: 1.3; }
  .guided-check-hint { display: block; font-size: 12px; color: var(--color-ink-subtle); margin-top: 3px; line-height: 1.35; }

  .guided-output-field {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    max-width: 520px;
    padding-top: var(--space-2);
  }
  .guided-output-field .input { height: 42px; }

  .guided-start-block {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    padding-top: var(--space-8);
    padding-left: 88px;
  }
  .start-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--space-4);
  }
  .start-btn { min-width: 240px; padding: 14px var(--space-8); font-size: 13px; }
  @media (max-width: 720px) {
    .step { grid-template-columns: 1fr; }
    .step > :global(*), .step-head > div { grid-column: 1; }
    .step-num { grid-row: auto; }
    .guided-start-block { padding-left: 0; }
  }

  /* ── Advanced Dataset Fabric styles ──────────────────────────────────── */
  .dataset-scroll { flex: 1; overflow-y: auto; min-height: 0; }
  .dataset-content { width: 100%; padding: var(--space-5) var(--space-6); display: flex; flex-direction: column; gap: var(--space-4); }
  .dataset-grid { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 0.92fr); gap: var(--space-4); align-items: start; width: 100%; }
  .dataset-grid.split-grid { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  @media (max-width: 1040px) { .dataset-grid, .dataset-grid.split-grid { grid-template-columns: 1fr; } }
  .dataset-col { display: flex; flex-direction: column; gap: var(--space-4); min-width: 0; }
  .fields { display: flex; flex-direction: column; gap: var(--space-3); }
  .segmented-control {
    display: inline-flex;
    background: var(--color-surface-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card);
    padding: 3px;
    gap: 3px;
  }
  .segmented-control button {
    border: none;
    background: transparent;
    color: var(--color-ink-subtle);
    font-family: inherit;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--anim-duration) var(--ease), color var(--anim-duration) var(--ease);
  }
  .segmented-control button:hover:not(:disabled) { color: var(--color-ink); background: var(--color-surface-hover); }
  .segmented-control button.active { background: var(--color-brand); color: var(--color-brand-contrast); }
  .source-row { display: flex; flex-wrap: wrap; align-items: flex-end; gap: var(--space-2); }
  .source-path { flex: 1 1 240px; min-width: 0; }
  .source-actions { display: flex; align-items: center; gap: var(--space-2); flex-shrink: 0; height: 34px; }
  .source-row .btn, .source-actions .btn { height: 34px; }
  .inline-actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-start; gap: var(--space-4); padding-top: var(--space-1); }
  .two-col-fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-3); align-items: start; }
  @media (max-width: 680px) { .two-col-fields { grid-template-columns: 1fr; } }
  .weight { width: 100px; flex-shrink: 0; }
  .filter-list { display: flex; flex-direction: column; gap: var(--space-2); }
  .check-row { display: flex; align-items: center; gap: var(--space-2); font-size: 13px; color: var(--color-ink); cursor: pointer; line-height: 1.4; }
  .check-row input[type="checkbox"] { accent-color: var(--color-brand); width: 15px; height: 15px; cursor: pointer; flex-shrink: 0; margin: 0; }
  .nested-field, .nested-hint { margin-left: 23px; }
  .benchmark-area { min-height: 82px; height: auto; padding-top: 8px; padding-bottom: 8px; resize: vertical; }
  .result { display: flex; flex-direction: column; align-items: flex-start; gap: var(--space-2); overflow-wrap: anywhere; padding: var(--space-3); border: 1px solid var(--color-border); background: var(--color-surface-muted); border-radius: var(--radius-card); }
  .result p { margin: 0; font-size: 13px; }
</style>
