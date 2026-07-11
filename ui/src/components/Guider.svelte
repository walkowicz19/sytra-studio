<script lang="ts">
  import { onMount } from 'svelte'
  import { hwStore, pushToast, setTab, trainFormStore, mergeFormStore } from '../store.svelte'
  import { api } from '../api'
  import type { GuiderRecipe, CompatResult, MergeMethod } from '../types'
  import catalogData from '../../../crates/sytra-contracts/src/catalog.json'

  interface RustRecipe {
    model_id: string
    adapter_type: string
    quant_bits: number | null
    quality_tier: string
    estimated_vram_mb: number
    estimated_step_time_ms: number
    reason: string
  }

  let recipes   = $state<RustRecipe[]>([])
  let loading   = $state(true)
  
  // Hardware overrides
  let vramOverride = $state(24)
  let ramOverride = $state(64)
  let allowQualityLoss = $state(false)

  // Merge advisor inputs
  let advisorModels = $state('org/knowledge-ft, org/toolcalling-ft')
  let advisorGoal = $state<'knowledge' | 'behavior' | 'interpolation' | 'architecture'>('knowledge')
  let checking = $state(false)
  let compatResult = $state<CompatResult | null>(null)

  onMount(async () => {
    // Sync defaults with detected hardware if available
    if (hwStore.info) {
      vramOverride = Math.round(hwStore.info.vram_mb / 1024)
      ramOverride = Math.round(hwStore.info.ram_mb / 1024)
    }
    await updateRecommendations()
  })

  async function updateRecommendations() {
    loading = true
    try {
      const hw = {
        accelerator: hwStore.info?.backend || 'cuda',
        total_vram_mb: vramOverride * 1024,
        total_ram_mb: ramOverride * 1024
      }
      const raw = await api.guiderRecommend(hw)
      // Map backend-specific fields to RustRecipe
      recipes = (raw as any[]).map(r => ({
        model_id: r.model_id,
        adapter_type: r.adapter_type || r.adapter_kind || 'lora',
        quant_bits: r.quant_bits,
        quality_tier: r.quality_tier || 'A',
        estimated_vram_mb: r.estimated_vram_mb || 14336,
        estimated_step_time_ms: r.estimated_step_time_ms || 120,
        reason: r.reason || ''
      }))
    } catch {
      pushToast('error', 'Could not load recommendations')
    } finally {
      loading = false
    }
  }

  function getModelDetails(modelId: string) {
    const entry = catalogData.find(m => m.model_id === modelId)
    return entry || {
      name: modelId.split('/').pop() || modelId,
      param_count: 7000000000,
      dtype: 'bfloat16'
    }
  }

  // Filtered recipes: Tier-C hidden unless allowQualityLoss checked
  let filteredRecipes = $derived(
    recipes.map(r => {
      const model = getModelDetails(r.model_id)
      const fits = r.estimated_vram_mb <= vramOverride * 1024
      return {
        ...r,
        model_name: model.name,
        param_count_b: (model.param_count / 1e9).toFixed(1),
        fits_vram: fits
      }
    }).filter(r => allowQualityLoss || (r.quality_tier !== 'C' && r.quality_tier !== 'c'))
  )

  // Recommend merge method based on goal
  let recommendedMethod = $derived.by<MergeMethod>(() => {
    const modelsCount = advisorModels.split(',').filter(Boolean).length
    if (advisorGoal === 'interpolation' && modelsCount === 2) return 'slerp'
    if (advisorGoal === 'behavior') return 'ties'
    return 'dare_ties'
  })

  async function checkAdvisorMerge() {
    const models = advisorModels.split(',').map(s => s.trim()).filter(Boolean)
    if (models.length < 2) {
      pushToast('error', 'Enter at least 2 comma-separated models')
      return
    }
    checking = true
    try {
      compatResult = await api.mergeCheck(models, recommendedMethod)
      pushToast('success', 'Compatibility checked')
    } catch (e: any) {
      pushToast('error', `Failed to check compatibility: ${e.message || String(e)}`)
    } finally {
      checking = false
    }
  }

  function applyTrainRecipe(recipe: any) {
    trainFormStore.model = recipe.model_id
    trainFormStore.adapterKind = recipe.adapter_type
    trainFormStore.quantBits = recipe.quant_bits === null ? 0 : recipe.quant_bits as 0 | 4 | 8
    
    // Switch to train tab
    setTab('train')
    pushToast('success', `Applied training recipe for ${recipe.model_name}`)
  }

  function applyMergeConfig() {
    const models = advisorModels.split(',').map(s => s.trim()).filter(Boolean)
    if (models.length < 2) {
      pushToast('error', 'Enter at least 2 models to merge')
      return
    }

    mergeFormStore.mergeMethod = recommendedMethod
    mergeFormStore.models = models.map(m => ({ model: m, weight: 1.0 / models.length }))
    mergeFormStore.baseModel = models[0] // Default base model to first selection
    mergeFormStore.outputPath = 'runs/merged-model'

    setTab('merge')
    pushToast('success', `Configured merge using ${recommendedMethod.toUpperCase()}`)
  }

  const verdictClass: Record<string, string> = {
    green: 'badge-success', amber: 'badge-warn', red: 'badge-error',
  }
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">Guider</h1>
      <p class="text-small">Hardware-aware model recommendations and merge compatibility</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area">

      <!-- Hardware Simulator -->
      <section class="card" id="section-hardware">
        <div class="card-header"><span class="text-label">What-If Hardware Simulator</span></div>
        <div class="card-body">
          <div class="grid-2">
            <div class="field">
              <label class="field-label" for="slider-vram">VRAM Override — {vramOverride} GB</label>
              <input id="slider-vram" type="range" class="slider" min="4" max="80" step="4" bind:value={vramOverride} onchange={updateRecommendations} />
            </div>
            <div class="field">
              <label class="field-label" for="slider-ram">System RAM Override — {ramOverride} GB</label>
              <input id="slider-ram" type="range" class="slider" min="8" max="256" step="8" bind:value={ramOverride} onchange={updateRecommendations} />
            </div>
            <div class="toggle-row col-2">
              <div class="toggle-row-text">
                <span class="field-label">Allow Quality Loss</span>
                <span class="field-hint">Show low-quality recipes (Tier-C)</span>
              </div>
              <label class="toggle" id="toggle-quality">
                <input type="checkbox" bind:checked={allowQualityLoss} />
                <div class="toggle-track"><div class="toggle-thumb"></div></div>
              </label>
            </div>
          </div>
        </div>
      </section>

      <!-- Recommendations -->
      <section class="card" id="section-recommendations">
        <div class="card-header"><span class="text-label">Recommended Configurations</span></div>
        <div class="card-body recipe-body">
          {#if loading}
            <div class="empty-state" style="min-height:100px"><span class="spinner spinner-lg"></span></div>
          {:else if filteredRecipes.length === 0}
            <div class="empty-state" style="min-height:100px">
              <div class="empty-icon" style="color: var(--color-ink-ghost); font-size: 24px">
                <i class="bi bi-exclamation-triangle"></i>
              </div>
              <p class="text-small" style="text-align:center;max-width:260px;margin-top:var(--space-2)">No recommendations match your hardware settings.</p>
            </div>
          {:else}
            {#each filteredRecipes as recipe}
              <div class="recipe-card">
                <div class="recipe-header">
                  <div>
                    <div style="display:flex; align-items:center; gap:var(--space-2)">
                      <div class="text-title" style="font-size:14px">{recipe.model_name}</div>
                      <span class="badge badge-brand" style="font-size:10px; font-weight:600">Tier {recipe.quality_tier}</span>
                    </div>
                    <div class="text-small" style="margin-top:2px; color:var(--color-ink-subtle)">{recipe.reason}</div>
                  </div>
                  <div style="display:flex; align-items:center; gap:var(--space-2)">
                    <span class="badge {recipe.fits_vram ? 'badge-success' : 'badge-error'}">
                      {recipe.fits_vram ? '✓ fits' : '✕ too large'}
                    </span>
                    <button class="btn btn-primary btn-sm" onclick={() => applyTrainRecipe(recipe)} style="display: flex; align-items: center; gap: var(--space-1)">
                      <i class="bi bi-fire" style="font-size: 11px"></i>
                      Train This
                    </button>
                  </div>
                </div>
                <div class="recipe-chips">
                  <span class="meta-chip">{recipe.adapter_type.toUpperCase()}</span>
                  <span class="meta-chip">{recipe.param_count_b}B params</span>
                  <span class="meta-chip">{(recipe.estimated_vram_mb / 1024).toFixed(1)} GB est. VRAM</span>
                  {#if recipe.quant_bits}
                    <span class="meta-chip">{recipe.quant_bits}-bit quant</span>
                  {/if}
                </div>
              </div>
            {/each}
          {/if}
        </div>
      </section>

      <!-- Merge Advisor -->
      <section class="card" id="section-compat-checker">
        <div class="card-header"><span class="text-label">Merge Advisor Panel</span></div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-4)">
          <div class="field">
            <label class="field-label" for="input-compat-models">Models to Merge (comma-separated HF IDs)</label>
            <input id="input-compat-models" class="input input-mono" bind:value={advisorModels} placeholder="org/model-1, org/model-2" />
          </div>
          <div class="field">
            <label class="field-label" for="select-advisor-goal">Stated Goal</label>
            <select id="select-advisor-goal" class="select" bind:value={advisorGoal}>
              <option value="knowledge">Knowledge Blending (e.g. math + science)</option>
              <option value="behavior">Behavioral Alignment (instruction following)</option>
              <option value="interpolation">Simple Interpolation (average/blend two weights)</option>
              <option value="architecture">Architecture Transplant (advanced layer merging)</option>
            </select>
          </div>
          
          <div class="advice-card border-surface">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:var(--space-3)">
              <div style="flex:1; min-width:280px">
                <span class="text-label" style="font-size:10px; color:var(--color-ink-subtle)">RECOMMENDED METHOD</span>
                <div class="text-title" style="font-size:15px; font-weight:600; color:var(--color-brand); margin-top:2px">{recommendedMethod.toUpperCase()}</div>
                <p style="font-size:12px; color:var(--color-ink-subtle); margin-top:6px; line-height:1.5">
                  {#if recommendedMethod === 'slerp'}
                    SLERP provides spherical interpolation between two checkpoints for seamless capability blending.
                  {:else}
                    DARE-TIES/TIES resolves task vector conflicts, making it ideal for behavioral alignment across up to 3 models.
                  {/if}
                </p>
              </div>
              <button class="btn btn-secondary" onclick={checkAdvisorMerge} disabled={checking} style="display: flex; align-items: center; gap: var(--space-2); align-self:center">
                {#if checking}<span class="spinner"></span>{/if}
                <i class="bi bi-shield-check" style="font-size: 12px"></i>
                Validate Compatibility
              </button>
            </div>
          </div>

          {#if compatResult}
            <div class="compat-result animate-in">
              <div style="display:flex; align-items:center; gap:var(--space-3); flex:1; min-width:280px">
                <span class="badge {verdictClass[compatResult.verdict]}" style="display: inline-flex; align-items: center; gap: 4px; padding: 4px 8px">
                  {#if compatResult.verdict === 'green'}
                    <i class="bi bi-check-circle-fill" style="font-size: 10px"></i>
                  {:else if compatResult.verdict === 'amber'}
                    <i class="bi bi-exclamation-triangle-fill" style="font-size: 10px"></i>
                  {:else}
                    <i class="bi bi-x-circle-fill" style="font-size: 10px"></i>
                  {/if}
                  {compatResult.verdict.toUpperCase()}
                </span>
                <span class="text-body" style="font-size:13px; line-height:1.4">{compatResult.reason}</span>
              </div>
              {#if compatResult.verdict !== 'red'}
                <button class="btn btn-primary btn-sm" onclick={applyMergeConfig} style="display: flex; align-items: center; gap: var(--space-1); align-self: center">
                  <i class="bi bi-lightning-charge" style="font-size: 11px"></i>
                  Merge This
                </button>
              {/if}
            </div>
          {/if}
        </div>
      </section>

    </div>

    <!-- Side panel -->
    <div class="page-side-panel">
      <div class="summary-section">
        <div class="summary-title">Hardware Overrides</div>
        <div class="summary-row"><span class="summary-key">VRAM Limit</span><span class="summary-val">{vramOverride} GB</span></div>
        <div class="summary-row"><span class="summary-key">RAM Limit</span><span class="summary-val">{ramOverride} GB</span></div>
        <div class="summary-row"><span class="summary-key">Show low quality</span><span class="summary-val">{allowQualityLoss ? 'Yes' : 'No'}</span></div>
      </div>

      <div class="summary-divider"></div>

      <div class="summary-section">
        <div class="summary-title">Advisor Tips</div>
        <div class="tip-list">
          <div class="tip-item">
            <span class="tip-dot" style="color: var(--color-brand)">●</span>
            <span>Always run SFT fine-tuning (Heal) after combining models with passthrough method.</span>
          </div>
          <div class="tip-item">
            <span class="tip-dot" style="color: var(--color-brand)">●</span>
            <span>Check the compatibility banner before launching a merge operation to avoid run errors.</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* Recipes */
  .recipe-body { display: flex; flex-direction: column; gap: var(--space-3); }
  .recipe-card {
    background: var(--color-surface-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: var(--space-3) var(--space-4);
    display: flex; flex-direction: column; gap: var(--space-3);
  }
  .recipe-header { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-3); }
  .recipe-chips  { display: flex; flex-wrap: wrap; gap: var(--space-2); }

  /* Advice Card */
  .advice-card {
    padding: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface-muted);
  }

  /* Compat result */
  .compat-result {
    display: flex; align-items: center; justify-content: space-between;
    gap: var(--space-3); flex-wrap: wrap;
    padding: var(--space-3) var(--space-4);
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
  }

  /* Tips */
  .tip-list { display: flex; flex-direction: column; gap: var(--space-2); }
  .tip-item { display: flex; align-items: flex-start; gap: var(--space-2); font-size: 12px; color: var(--color-ink-subtle); line-height: 1.5; }
</style>
