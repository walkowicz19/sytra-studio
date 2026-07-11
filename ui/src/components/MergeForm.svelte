<script lang="ts">
  import { api } from '../api'
  import { run, resetRun, applyTelemetry, pushToast, mergeFormStore, watchTelemetry } from '../store.svelte'
  import type { MergeConfig, MergeMethod } from '../types'

  let submitting = $state(false)

  function addModel() {
    if (mergeFormStore.models.length < 3) {
      mergeFormStore.models = [...mergeFormStore.models, { model: '', weight: 0.3 }]
    }
  }
  
  function removeModel(i: number) {
    if (mergeFormStore.models.length > 2) {
      mergeFormStore.models = mergeFormStore.models.filter((_, idx) => idx !== i)
    }
  }

  const requiresBase = ['task_vector', 'dare_ties', 'ties']
  $effect(() => { 
    if (mergeFormStore.mergeMethod === 'slerp' && mergeFormStore.models.length > 2) {
      mergeFormStore.models = mergeFormStore.models.slice(0, 2)
    } 
  })

  let compat = $state<import('../types').CompatResult | null>(null)
  async function checkCompat() {
    const filled = mergeFormStore.models.filter(m => m.model)
    if (filled.length < 2) return
    compat = await api.mergeCheck(filled.map(m => m.model), mergeFormStore.mergeMethod, mergeFormStore.baseModel || null)
  }

  async function startMerge() {
    if (submitting) return
    submitting = true; resetRun()
    const config: any = {
      version: 1,
      merge_method: mergeFormStore.mergeMethod,
      base_model: mergeFormStore.baseModel || null,
      dtype: 'float16',
      models: mergeFormStore.models.filter(m => m.model).map(m => ({
        model: m.model,
        parameters: { weight: m.weight }
      })),
      tokenizer: { source: 'base' },
      compat: { verdict: compat?.verdict || 'green', fingerprint: compat?.fingerprint || null },
      output: { model_path: mergeFormStore.outputPath },
    }
    try {
      const opId = await api.startMerge(config)
      run.opId = opId; run.kind = 'merge'; run.status = 'running'; run.startedAt = Date.now()
      pushToast('success', `Merge started — ${opId.slice(0, 8)}`)
      if (!('__TAURI_INTERNALS__' in window)) {
        simulateMerge()
      } else {
        watchTelemetry(opId)
      }
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
      <p class="text-small">Combine model weights using SLERP, TIES, or DARE-TIES</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area">

      <!-- Method -->
      <section class="card" id="section-merge-method">
        <div class="card-header"><span class="text-label">Merge Algorithm</span></div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
          <div class="field">
            <label class="field-label" for="select-method">Algorithm</label>
            <select id="select-method" class="select" bind:value={mergeFormStore.mergeMethod} onchange={checkCompat}>
              <option value="slerp">SLERP — Spherical interpolation (2 models)</option>
              <option value="ties">TIES — Task-Informed Ensemble</option>
              <option value="dare_ties">DARE-TIES — Drop And REscale + TIES</option>
              <option value="task_vector">Task Vector</option>
              <option value="linear">Linear (weighted average)</option>
              <option value="passthrough">Passthrough (slice/concatenate layers)</option>
              <option value="moe">FrankenMoE (Mixture of Experts)</option>
            </select>
          </div>
          {#if requiresBase.includes(mergeFormStore.mergeMethod)}
            <div class="field">
              <label class="field-label" for="input-base-model">Base Model (required for {mergeFormStore.mergeMethod})</label>
              <input id="input-base-model" class="input input-mono" bind:value={mergeFormStore.baseModel} placeholder="org/base-model" />
            </div>
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
            disabled={mergeFormStore.models.length >= (mergeFormStore.mergeMethod === 'slerp' ? 2 : 3)}
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
              <input
                id="input-merge-model-{i}"
                class="input input-mono"
                bind:value={m.model}
                onblur={checkCompat}
                placeholder="org/model-name"
              />
              <div class="weight-row">
                <label class="field-label" for="slider-weight-{i}" style="white-space:nowrap">
                  Weight — {m.weight.toFixed(2)}
                </label>
                <input id="slider-weight-{i}" type="range" class="slider" min="0" max="1" step="0.01" bind:value={m.weight} style="flex:1" />
              </div>
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
        {#if requiresBase.includes(mergeFormStore.mergeMethod) && mergeFormStore.baseModel}
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
