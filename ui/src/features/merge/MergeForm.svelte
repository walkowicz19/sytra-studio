<script lang="ts">
  import { onMount } from 'svelte'
  import { api } from '../../api'
  import { run, resetRun, applyTelemetry, pushToast, mergeFormStore } from '../../store.svelte'
  import { startWatchedOp } from '../shared/startWatchedOp'
  import type { LocalModelItem, MergeConfig, MergeMethod } from '../../types'

  let localModels = $state<LocalModelItem[]>([])
  onMount(async () => {
    try {
      localModels = (await api.listLocalModels()) ?? []
    } catch {
      localModels = []
    }
  })

  async function browseModel(index?: number) {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({
        directory: false,
        multiple: false,
        title: 'Select local model file',
        filters: [{ name: 'Model Files', extensions: ['gguf', 'safetensors', 'bin', 'pth'] }]
      })
      if (typeof picked === 'string' && picked) {
        if (index === undefined) mergeFormStore.baseModel = picked
        else mergeFormStore.models[index].model = picked
        void checkCompat()
      }
    } catch (e) {
      pushToast('error', String(e))
    }
  }

  let submitting = $state(false)
  let moeArchitecture = $state('mixtral')
  let expertsPerToken = $state(2)
  let positivePrompts = $state<string[]>(['', ''])
  let negativePrompts = $state<string[]>(['', ''])
  let sharedExperts = $state('')
  let sharedPositivePrompts = $state('')
  let sharedNegativePrompts = $state('')

  function lines(value: string | undefined): string[] {
    return (value ?? '').split(/\r?\n/).map(item => item.trim()).filter(Boolean)
  }

  function addModel() {
    if (mergeFormStore.models.length < (mergeFormStore.mergeMethod === 'moe' ? 16 : 3)) {
      mergeFormStore.models = [...mergeFormStore.models, { model: '', weight: 0.3 }]
      positivePrompts = [...positivePrompts, '']
      negativePrompts = [...negativePrompts, '']
    }
  }
  
  function removeModel(i: number) {
    if (mergeFormStore.models.length > 2) {
      mergeFormStore.models = mergeFormStore.models.filter((_, idx) => idx !== i)
      positivePrompts = positivePrompts.filter((_, idx) => idx !== i)
      negativePrompts = negativePrompts.filter((_, idx) => idx !== i)
    }
  }

  const requiresBase = ['task_arithmetic', 'dare_ties', 'dare_linear', 'ties', 'moe']
  const supportsOptionalBase = ['multi_slerp', 'nuslerp']
  $effect(() => { 
    if (mergeFormStore.mergeMethod === 'slerp' && mergeFormStore.models.length > 2) {
      mergeFormStore.models = mergeFormStore.models.slice(0, 2)
      positivePrompts = positivePrompts.slice(0, 2)
      negativePrompts = negativePrompts.slice(0, 2)
    } 
  })

  let compat = $state<import('../../types').CompatResult | null>(null)
  async function checkCompat() {
    const filled = mergeFormStore.models.filter(m => m.model)
    if (filled.length < 2) return
    const base = requiresBase.includes(mergeFormStore.mergeMethod) || supportsOptionalBase.includes(mergeFormStore.mergeMethod)
      ? (mergeFormStore.baseModel || null)
      : null
    compat = await api.mergeCheck(filled.map(m => m.model), mergeFormStore.mergeMethod, base)
  }

  async function startMerge() {
    if (submitting) return
    submitting = true; resetRun()
    const isMoe = mergeFormStore.mergeMethod === 'moe'
    if (isMoe && (!mergeFormStore.baseModel.trim() || mergeFormStore.models.some((m, i) => !m.model.trim() || !lines(positivePrompts[i]).length)
      || !Number.isInteger(expertsPerToken) || expertsPerToken < 1 || expertsPerToken > mergeFormStore.models.length)) {
      pushToast('error', 'MoE requires a base model, at least two experts, routing prompts and a valid experts-per-token count.')
      submitting = false
      return
    }
    const shared = moeArchitecture === 'mixtral' ? [] : lines(sharedExperts)
    if (isMoe && ((moeArchitecture === 'mixtral' && shared.length > 0) ||
      (moeArchitecture === 'deepseek' && shared.length > 1) ||
      (moeArchitecture === 'qwen' && (shared.length !== 1 || !lines(sharedPositivePrompts).length)))) {
      pushToast('error', 'Mixtral has no shared expert; DeepSeek accepts at most one; Qwen requires one with positive routing prompts.')
      submitting = false
      return
    }
    const slerpWeights = mergeFormStore.models.filter(m => m.model).map(m => m.weight)
    const slerpT = slerpWeights.length === 2 && (slerpWeights[0] + slerpWeights[1]) > 0
      ? slerpWeights[1] / (slerpWeights[0] + slerpWeights[1])
      : 0.5
    const config: any = {
      performance: { mode: mergeFormStore.performanceMode },
      version: 1,
      merge_method: mergeFormStore.mergeMethod,
      base_model: mergeFormStore.baseModel || null,
      dtype: 'float16',
      ...(mergeFormStore.mergeMethod === 'slerp' ? { parameters: { t: slerpT } } : {}),
      models: mergeFormStore.models.filter(m => m.model).map((m, i) => ({
        model: m.model,
        ...(isMoe ? { positive_prompts: lines(positivePrompts[i]), negative_prompts: lines(negativePrompts[i]) } : { parameters: { weight: m.weight } })
      })),
      moe: isMoe ? { architecture: moeArchitecture, experts_per_token: expertsPerToken,
        shared_experts: shared.map(source_model => ({ source_model, ...(moeArchitecture === 'qwen' ? {
          positive_prompts: lines(sharedPositivePrompts), negative_prompts: lines(sharedNegativePrompts) } : {}) })) } : undefined,
      tokenizer: { source: 'base' },
      compat: { verdict: compat?.verdict || 'green', fingerprint: compat?.fingerprint || null },
      output: { model_path: mergeFormStore.outputPath },
    }
    try {
      await startWatchedOp('merge', () => api.startMerge(config))
      if (!('__TAURI_INTERNALS__' in window)) simulateMerge()
    } catch (e: unknown) {
      run.status = 'error'
      pushToast('error', `Merge failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally { submitting = false }
  }

  function simulateMerge() {
    let p = 0
    const iv = setInterval(() => {
      p += 0.04 + Math.random() * 0.02
      applyTelemetry({ type: 'metric', step: Math.round(p * 100), progress: Math.min(p, 1) })
      if (p >= 1) { clearInterval(iv); applyTelemetry({ type: 'event', event: 'done' }); pushToast('success', 'Merge complete!') }
    }, 200)
  }

  const verdictClass: Record<string, string> = {
    green: 'badge-success', amber: 'badge-warn', red: 'badge-error',
  }

  let totalWeight = $derived(mergeFormStore.models.reduce((s, m) => s + m.weight, 0))
</script>

<div class="page-layout">
  <!-- Header (Header buttons removed as they are redundant now) -->
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">Merge</h1>
      <p class="text-small">Combine model weights using SLERP, Multi-SLERP, NuSLERP, DARE, or TIES</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area">
      <!-- Performance -->
      <section class="card" id="section-merge-performance">
        <div class="card-header"><span class="text-label">Performance</span></div>
        <div class="card-body">
          <div class="field">
            <label class="field-label" for="advanced-merge-performance">Execution profile</label>
            <select id="advanced-merge-performance" class="select" bind:value={mergeFormStore.performanceMode}>
              <option value="auto">Auto</option><option value="safe">Safe - low memory</option><option value="fast">Fast</option>
            </select>
          </div>
        </div>
      </section>

      <!-- Method -->
      <section class="card" id="section-merge-method">
        <div class="card-header"><span class="text-label">Merge Algorithm</span></div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
          <div class="field">
            <label class="field-label" for="select-method">Algorithm</label>
            <select id="select-method" class="select" bind:value={mergeFormStore.mergeMethod} onchange={checkCompat}>
              <option value="slerp">SLERP — Spherical interpolation (2 models)</option>
              <option value="multi_slerp">Multi-SLERP — Barycentric spherical mean (2–3 models)</option>
              <option value="nuslerp">NuSLERP — Normalized uniform SLERP (2–3 models)</option>
              <option value="ties">TIES — Task-Informed Ensemble</option>
              <option value="dare_ties">DARE-TIES — Drop And REscale + TIES</option>
              <option value="dare_linear">DARE-Linear — Drop And REscale</option>
              <option value="task_arithmetic">Task Vector</option>
              <option value="linear">Linear (weighted average)</option>
              <option value="passthrough">Passthrough (slice/concatenate layers)</option>
              <option value="moe">MoE builder (hidden-state routing)</option>
            </select>
          </div>
          {#if requiresBase.includes(mergeFormStore.mergeMethod) || supportsOptionalBase.includes(mergeFormStore.mergeMethod)}
            <div class="field">
              <label class="field-label" for="input-base-model">Base Model ({requiresBase.includes(mergeFormStore.mergeMethod) ? `required for ${mergeFormStore.mergeMethod}` : `optional for ${mergeFormStore.mergeMethod}`})</label>
              <input id="input-base-model" class="input input-mono" bind:value={mergeFormStore.baseModel} placeholder="org/base-model" />
            </div>
          {/if}
          {#if mergeFormStore.mergeMethod === 'moe'}
            <div class="grid-2">
              <div class="field"><label class="field-label" for="moe-architecture">Output architecture</label><select id="moe-architecture" class="select" bind:value={moeArchitecture}><option value="mixtral">Mixtral</option><option value="deepseek">DeepSeek</option><option value="qwen">Qwen MoE</option></select></div>
              <div class="field"><label class="field-label" for="moe-topk">Experts per token</label><input id="moe-topk" class="input" type="number" min="1" max={mergeFormStore.models.length} bind:value={expertsPerToken} /></div>
            </div>
            {#if moeArchitecture !== 'mixtral'}
              <div class="field"><label class="field-label" for="moe-shared">Shared expert model</label><textarea id="moe-shared" class="input input-mono" rows="2" bind:value={sharedExperts} placeholder="One model path or ID per line"></textarea><span class="field-hint">DeepSeek accepts at most one; Qwen requires one. Support depends on the installed MergeKit version.</span></div>
              {#if moeArchitecture === 'qwen'}
                <div class="field"><label class="field-label" for="moe-shared-positive">Shared expert positive prompts</label><textarea id="moe-shared-positive" class="input" rows="2" bind:value={sharedPositivePrompts} placeholder="One routing example per line"></textarea></div>
                <div class="field"><label class="field-label" for="moe-shared-negative">Shared expert negative prompts (optional)</label><textarea id="moe-shared-negative" class="input" rows="2" bind:value={sharedNegativePrompts} placeholder="One counterexample per line"></textarea></div>
              {/if}
            {/if}
          {/if}
        </div>
      </section>

      <!-- Models -->
      <section class="card" id="section-merge-models">
        <div class="card-header">
          <span class="text-label">Models to Merge</span>
          <button
            class="btn btn-secondary btn-sm"
            onclick={addModel}
            disabled={mergeFormStore.models.length >= (mergeFormStore.mergeMethod === 'slerp' ? 2 : mergeFormStore.mergeMethod === 'moe' ? 16 : 3)}
            id="btn-add-model"
            style="display: flex; align-items: center; gap: var(--space-1)"
          >
            <i class="bi bi-plus-lg" style="font-size: 10px"></i> Add
          </button>
        </div>
        <div class="card-body model-list">
          {#each mergeFormStore.models as m, i (i)}
            <div class="model-entry">
              <div class="model-entry-header">
                <span class="field-label">Model {i + 1}</span>
                {#if mergeFormStore.models.length > 2}
                  <button
                    class="btn btn-ghost btn-icon"
                    onclick={() => removeModel(i)}
                    id="btn-remove-{i}"
                    aria-label="Remove model {i + 1}"
                    style="display: flex; align-items: center; justify-content: center; width: 24px; height: 24px"
                  >
                    <i class="bi bi-x-lg" style="font-size: 11px"></i>
                  </button>
                {/if}
              </div>
              <div class="picker-inline-row">
                <input
                  id="input-merge-model-{i}"
                  class="input input-mono"
                  bind:value={m.model}
                  onblur={checkCompat}
                  placeholder="org/model-name"
                />
                <button type="button" class="btn btn-secondary" onclick={() => browseModel(i)}>Browse</button>
              </div>
              {#if localModels.length > 0}
                <select
                  class="select"
                  aria-label="Scanned local model {i + 1}"
                  value={localModels.some(lm => lm.path === m.model) ? m.model : ''}
                  onchange={(e) => {
                    const val = (e.currentTarget as HTMLSelectElement).value
                    if (val) { m.model = val; void checkCompat() }
                  }}
                >
                  <option value="">-- Choose from Scanned Local Models --</option>
                  {#each localModels as lm}
                    <option value={lm.path}>[{lm.category.toUpperCase()}] {lm.name} ({lm.size_gb} GB)</option>
                  {/each}
                </select>
              {/if}
              {#if mergeFormStore.mergeMethod === 'moe'}
                <div class="field"><label class="field-label" for="moe-positive-{i}">Positive routing prompts (one per line)</label><textarea id="moe-positive-{i}" class="input" rows="2" bind:value={positivePrompts[i]} placeholder="Example requests for this expert"></textarea></div>
                <div class="field"><label class="field-label" for="moe-negative-{i}">Negative routing prompts (optional)</label><textarea id="moe-negative-{i}" class="input" rows="2" bind:value={negativePrompts[i]} placeholder="One per line"></textarea></div>
              {:else}
              <div class="weight-row">
                <label class="field-label" for="slider-weight-{i}" style="white-space:nowrap">
                  Weight — {m.weight.toFixed(2)}
                </label>
                <input id="slider-weight-{i}" type="range" class="slider" min="0" max="1" step="0.01" bind:value={m.weight} style="flex:1" />
              </div>
              {/if}
            </div>
          {/each}

          {#if compat}
            <div class="compat-row animate-in">
              <span class="badge {verdictClass[compat.verdict] ?? 'badge-neutral'}">
                {compat.verdict.toUpperCase()}
              </span>
              <span class="text-small" style="flex:1">{compat.reason}</span>
            </div>
          {/if}
        </div>
      </section>

      <!-- Output -->
      <section class="card" id="section-merge-output">
        <div class="card-header"><span class="text-label">Output</span></div>
        <div class="card-body">
          <div class="field">
            <label class="field-label" for="input-merge-output">Output Path</label>
            <input id="input-merge-output" class="input input-mono" bind:value={mergeFormStore.outputPath} />
          </div>
        </div>
      </section>

      <!-- Repositioned start button at bottom of forms -->
      <div style="margin-top:var(--space-4); display:flex">
        <button
          class="btn btn-primary btn-lg"
          onclick={startMerge}
          disabled={submitting || run.status === 'running'}
          id="btn-start-merge-bottom"
          style="flex:1; justify-content:center; display: flex; align-items: center; gap: var(--space-2)"
        >
          {#if submitting}<span class="spinner"></span>{/if}
          <i class="bi bi-lightning-charge" style="font-size: 16px"></i>
          Merge Models and Export Checkpoint
        </button>
      </div>

    </div>

    <!-- Summary panel -->
    <div class="page-side-panel">
      <div class="summary-section">
        <div class="summary-title">Merge Config</div>
        <div class="summary-row"><span class="summary-key">Algorithm</span><span class="summary-val">{mergeFormStore.mergeMethod.toUpperCase()}</span></div>
        <div class="summary-row"><span class="summary-key">Models</span><span class="summary-val">{mergeFormStore.models.filter(m=>m.model).length}</span></div>
        <div class="summary-row"><span class="summary-key">Σ Weights</span>
          <span class="summary-val" style="color:{Math.abs(totalWeight - 1) < 0.01 ? 'var(--color-success)' : 'var(--color-warn)'}">
            {totalWeight.toFixed(2)}
          </span>
        </div>
        {#if (requiresBase.includes(mergeFormStore.mergeMethod) || supportsOptionalBase.includes(mergeFormStore.mergeMethod)) && mergeFormStore.baseModel}
          <div class="summary-row"><span class="summary-key">Base</span><span class="summary-val">{mergeFormStore.baseModel.split('/').pop()}</span></div>
        {/if}
      </div>

      {#if compat}
        <div class="summary-divider"></div>
        <div class="summary-section">
          <div class="summary-title">Compatibility</div>
          <div class="compat-panel">
            <span class="badge {verdictClass[compat.verdict] ?? 'badge-neutral'}">{compat.verdict.toUpperCase()}</span>
            <p class="tip-text">{compat.reason}</p>
          </div>
        </div>
      {/if}

      <div class="summary-divider"></div>
      <div class="summary-section">
        <div class="summary-title">Method Guide</div>
        <div class="tip-text">
          {#if mergeFormStore.mergeMethod === 'slerp'}
            SLERP interpolates on the unit hypersphere — ideal for blending two related checkpoints.
          {:else}
            Combine model deltas. Requires base model reference for task vector generation.
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .model-list {
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
  }
  .model-entry {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    padding: var(--space-3);
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
  }
  .model-entry-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .picker-inline-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
  }
  .picker-inline-row :global(.input) { flex: 1; min-width: 0; }
  .weight-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
  }
  .compat-row {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
  }
  .compat-panel {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .tip-text {
    font-size: 12px;
    color: var(--color-ink-subtle);
    line-height: 1.6;
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    padding: var(--space-3);
    border: 1px solid var(--color-border);
  }
</style>
