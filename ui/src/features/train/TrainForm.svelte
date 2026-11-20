<script lang="ts">
  import { api } from '../../api'
  import { run, resetRun, applyTelemetry, pushToast, trainFormStore } from '../../store.svelte'
  import { startWatchedOp } from '../shared/startWatchedOp'
  import type { RunConfig, TrainMode, AdapterType, Schedule, BackendKind } from '../../types'

  let submitting  = $state(false)

  // Derived VRAM estimate
  const baseVram: Record<string, number> = {
    'mistralai/Mistral-7B-v0.1': 14336,
    'meta-llama/Llama-2-7b-hf': 14336,
    'meta-llama/Meta-Llama-3-8B': 16384,
  }
  let estimatedVram = $derived.by(() => {
    const base = baseVram[trainFormStore.model] ?? 14336
    const quant = trainFormStore.quantBits === 4 ? 0.35 : trainFormStore.quantBits === 8 ? 0.6 : 1.0
    return Math.round(base * quant + trainFormStore.rank * trainFormStore.alpha * 0.1)
  })

  async function previewDataset() {
    previewLoading = true
    previewRows = null
    
    let sourceObj: any
      sourceObj = { source: trainFormStore.dataSplitSource }
      if (trainFormStore.dataSplitSource === 'hf') {
        sourceObj.hf = {
          repo_id: trainFormStore.hfRepoId,
          split: trainFormStore.hfSplit,
          revision: trainFormStore.hfRevision || null,
          config: trainFormStore.hfConfig || null
        }
      } else if (trainFormStore.dataSplitSource === 'local') {
        sourceObj.local = {
          path: trainFormStore.localPath,
          format: trainFormStore.localFormat,
          mapping: {
            prompt: trainFormStore.localPromptCol,
            completion: trainFormStore.localCompletionCol
          }
        }
      } else if (trainFormStore.dataSplitSource === 'synthetic') {
        sourceObj.synthetic = {
          generator_model: trainFormStore.synthGenerator,
          judge_model: trainFormStore.synthJudge,
          mode: trainFormStore.synthMode,
          count: 5,
          topic: trainFormStore.synthTopic
        }
      } else if (trainFormStore.dataSplitSource === 'klayer') {
        sourceObj.klayer = trainFormStore.klayerDomain.trim()
          ? { domain: trainFormStore.klayerDomain.trim() }
          : {}
      }

    try {
      const res = await api.previewDataset(sourceObj, 5)
      previewRows = res
      pushToast('success', 'Preview loaded')
    } catch (e: any) {
      pushToast('error', `Failed to preview: ${e.message || String(e)}`)
    } finally {
      previewLoading = false
    }
  }

  // Preview state
  let previewLoading = $state(false)
  let previewRows = $state<string[][] | null>(null)

  async function startTrain() {
    if (submitting || run.status === 'running') return
    if (trainFormStore.trainingEngine === 'helix') {
      const validAdapter = (trainFormStore.adapterKind === 'lora' && trainFormStore.quantBits === 0) ||
        (trainFormStore.adapterKind === 'qlora' && (trainFormStore.quantBits === 4 || trainFormStore.quantBits === 8))
      if (trainFormStore.trainMode !== 'sft' || !validAdapter) {
        pushToast('error', 'Helix requires SFT with unquantized LoRA or 4-bit/8-bit QLoRA.')
        return
      }
      if (!Number.isInteger(trainFormStore.activationMemoryMb) || trainFormStore.activationMemoryMb < 0 || trainFormStore.activationMemoryMb > 65536) {
        pushToast('error', 'Helix activation memory must be between 0 and 65536 MB.')
        return
      }
      if (trainFormStore.helixPrefetch && trainFormStore.backend === 'cpu') {
        pushToast('error', 'Helix layer prefetch requires CUDA.')
        return
      }
      if (trainFormStore.helixPrefetch && (!Number.isInteger(trainFormStore.helixPrefetchMemoryMb) || trainFormStore.helixPrefetchMemoryMb < 0 || trainFormStore.helixPrefetchMemoryMb > 4096)) {
        pushToast('error', 'Pinned prefetch memory must be between 0 and 4096 MB.')
        return
      }
      if (trainFormStore.packing === true && (trainFormStore.batchSize !== 1 || trainFormStore.backend === 'cpu')) {
        pushToast('error', 'Helix packing requires CUDA and batch size 1.')
        return
      }
    }
    if (trainFormStore.workflow !== 'fresh' && !trainFormStore.sourceAdapter.trim()) {
      pushToast('error', 'Choose a source adapter or checkpoint directory')
      return
    }
    submitting = true; resetRun()

    let dataSpec: any = {}

      dataSpec = {
        source: trainFormStore.dataSplitSource,
        jsonl_path: null,
        fingerprint: null
      }
      if (trainFormStore.dataSplitSource === 'hf') {
        dataSpec.hf = {
          repo_id: trainFormStore.hfRepoId,
          split: trainFormStore.hfSplit,
          revision: trainFormStore.hfRevision || null,
          config: trainFormStore.hfConfig || null
        }
      } else if (trainFormStore.dataSplitSource === 'local') {
        dataSpec.local = {
          path: trainFormStore.localPath,
          format: trainFormStore.localFormat,
          mapping: {
            prompt: trainFormStore.localPromptCol,
            completion: trainFormStore.localCompletionCol
          }
        }
      } else if (trainFormStore.dataSplitSource === 'synthetic') {
        dataSpec.synthetic = {
          generator_model: trainFormStore.synthGenerator,
          judge_model: trainFormStore.synthJudge,
          mode: trainFormStore.synthMode,
          count: trainFormStore.synthCount,
          topic: trainFormStore.synthTopic
        }
      } else if (trainFormStore.dataSplitSource === 'klayer') {
        dataSpec.klayer = trainFormStore.klayerDomain.trim()
          ? { domain: trainFormStore.klayerDomain.trim() }
          : {}
      }

    const config: any = {
      version: 1, train_mode: trainFormStore.trainMode, model: trainFormStore.model,
      backend: { kind: trainFormStore.backend },
      data: dataSpec,
      val_dataset: trainFormStore.validationPath.trim() || undefined,
      adapter: { type: trainFormStore.adapterKind, rank: trainFormStore.rank, alpha: trainFormStore.alpha, dropout: trainFormStore.dropout, target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], quant_bits: trainFormStore.quantBits === 0 ? null : trainFormStore.quantBits, use_rslora: trainFormStore.useRsLora, moe_topology: trainFormStore.moeTopology },
      optim: { learning_rate: trainFormStore.lr, schedule: trainFormStore.schedule, warmup_steps: trainFormStore.warmupSteps, weight_decay: trainFormStore.weightDecay, grad_accumulation_steps: trainFormStore.gradAcc, router_lr_ratio: trainFormStore.workflow === 'fresh' && trainFormStore.moeTopology !== 'attention' ? 0.1 : null, router_aux_coef: trainFormStore.workflow === 'fresh' && trainFormStore.moeTopology !== 'attention' && trainFormStore.trainMode === 'sft' ? trainFormStore.routerAuxCoef : null },
      train: {
        ...(trainFormStore.useEpochs ? { epochs: Math.max(1, trainFormStore.epochs) } : { max_steps: trainFormStore.maxSteps }),
        batch_size: trainFormStore.batchSize, max_seq_len: trainFormStore.maxSeqLen, save_every: trainFormStore.saveEvery, packing: trainFormStore.packing, neftune_noise_alpha: trainFormStore.trainMode === 'sft' ? trainFormStore.neftuneAlpha : 0
      },
      ...(trainFormStore.trainMode === 'sft' ? { algo: { completion_only_loss: trainFormStore.completionOnlyLoss } } : {}),
      init_adapter: trainFormStore.workflow === 'continue' ? trainFormStore.sourceAdapter.trim() : undefined,
      performance: { mode: trainFormStore.performanceMode, training_engine: trainFormStore.trainingEngine,
        ...(trainFormStore.trainingEngine === 'helix' ? { activation_memory_mb: trainFormStore.activationMemoryMb, helix_prefetch: trainFormStore.helixPrefetch,
          ...(trainFormStore.helixPrefetch ? { prefetch_pinned_memory_mb: trainFormStore.helixPrefetchMemoryMb } : {}) } : {}) },
      output: {
        adapter_path: trainFormStore.outputPath,
        resume_from_checkpoint: trainFormStore.workflow === 'resume' ? trainFormStore.sourceAdapter.trim() : undefined,
      },
    }
    try {
      await startWatchedOp('train', () => api.startTrain(config))
      if (!('__TAURI_INTERNALS__' in window)) simulateTelemetry()
    } catch (e: unknown) {
      run.status = 'error'
      pushToast('error', `Failed: ${e instanceof Error ? e.message : String(e)}`)
    } finally { submitting = false }
  }

  async function stopTrain() {
    if (!run.opId) return
    try { await api.stopOp(run.opId); run.status = 'stopped'; pushToast('info', 'Training stopped') }
    catch (e: unknown) { pushToast('error', String(e)) }
  }

  function simulateTelemetry() {
    let s = 0
    const timer = setInterval(() => {
      if (run.status !== 'running' || s >= trainFormStore.maxSteps) {
        clearInterval(timer)
        if (run.status === 'running') { applyTelemetry({ type: 'event', event: 'done' }); pushToast('success', 'Training complete!') }
        return
      }
      s++
      applyTelemetry({
        type: 'metric', step: s, epoch: Math.floor(s / 50) + 1,
        loss: 2.4 * Math.exp(-s * 0.018) + 0.3 + (Math.random() - 0.5) * 0.15,
        lr: trainFormStore.lr * Math.cos((s / trainFormStore.maxSteps) * Math.PI / 2),
        grad_norm: 0.8 + Math.random() * 0.5,
        progress: s / trainFormStore.maxSteps,
      })
    }, 80)
  }
</script>

<div class="page-layout">
  <!-- Header (Header buttons removed as they are redundant now) -->
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">Fine-Tune</h1>
      <p class="text-small">Configure and launch an SFT / DPO training run</p>
    </div>
  </div>

  <!-- Body -->
  <div class="page-content">
    <div class="page-form-area">

      <!-- Model & Backend -->
      <section class="card" id="section-model">
        <div class="card-header"><span class="text-label">Model & Backend</span></div>
        <div class="card-body grid-2">
          <div class="field col-2">
            <label class="field-label" for="training-workflow">Starting point</label>
            <select id="training-workflow" class="select" bind:value={trainFormStore.workflow}>
              <option value="fresh">New adapter</option>
              <option value="continue">Post-train from adapter</option>
              <option value="resume">Resume interrupted run</option>
            </select>
          </div>
          {#if trainFormStore.workflow !== 'fresh'}
            <div class="field col-2">
              <label class="field-label" for="source-adapter">{trainFormStore.workflow === 'resume' ? 'Trainer checkpoint directory' : 'Source adapter directory'}</label>
              <input id="source-adapter" class="input input-mono" bind:value={trainFormStore.sourceAdapter} placeholder="D:/models/my-adapter/checkpoint-400" />
              <span class="field-hint">{trainFormStore.workflow === 'resume' ? 'Restores weights, optimizer, scheduler, and step counter. Set steps to the total target.' : 'Keeps the saved adapter weights and structure. Starts a new optimizer and step counter; choose a separate output directory.'}</span>
              {#if trainFormStore.workflow === 'continue' && trainFormStore.trainingEngine !== 'helix'}
                <button class="btn btn-ghost btn-sm" onclick={() => { trainFormStore.quantBits = 4; trainFormStore.adapterKind = 'qlora'; trainFormStore.lr = 1.5e-5; trainFormStore.maxSteps = 300; trainFormStore.outputPath = `runs/continued-${Date.now()}` }}>Apply continued SFT preset (4-bit, LR 1.5e-5, 300 steps)</button>
              {/if}
            </div>
          {/if}
          <div class="field col-2">
            <label class="field-label" for="input-model">Base Model</label>
            <input id="input-model" class="input input-mono" bind:value={trainFormStore.model} placeholder="org/model-name" />
            <span class="field-hint">{trainFormStore.trainingEngine === 'helix' ? 'Local directory of a Mixtral, Qwen MoE, or DeepSeek MoE safetensors checkpoint (unquantized or bitsandbytes 4/8-bit) and tokenizer.' : 'Catalog base model ID, or local adapter directory (its base must be in the catalog).'}</span>
          </div>
          <div class="field">
            <label class="field-label" for="select-train-mode">Training Mode</label>
            <select id="select-train-mode" class="select" bind:value={trainFormStore.trainMode}>
              <option value="sft">SFT – Supervised Fine-tuning</option>
              <option value="dpo">DPO – Direct Preference Optimization</option>
              <option value="orpo">ORPO</option>
              <option value="cpo">CPO</option>
            </select>
          </div>
          <div class="field">
            <label class="field-label" for="select-backend">Accelerator</label>
            <select id="select-backend" class="select" bind:value={trainFormStore.backend}>
              <option value="auto">Auto-detect</option>
              <option value="cuda">CUDA (NVIDIA)</option>
              {#if trainFormStore.trainingEngine === 'helix'}<option value="cpu">CPU</option>{/if}

            </select>
          </div>
        </div>
      </section>

      <div class="field">
        <label class="field-label" for="training-engine">Training engine</label>
        <select id="training-engine" class="select" bind:value={trainFormStore.trainingEngine}>
          <option value="auto">Standard</option>
          <option value="helix">Helix (layer-streamed MoE)</option>
        </select>
        <span class="field-hint">Helix streams local Mixtral, Qwen2/3 MoE, and DeepSeek V2/V3 checkpoints by layer. 4/8-bit QLoRA reuses compatible bitsandbytes weights or builds a quantized cache in the output directory. Packing requires CUDA, batch size 1, and compatible FlexAttention.</span>
      </div>
      {#if trainFormStore.trainingEngine === 'helix'}
        <div class="field">
          <label class="field-label" for="helix-activation-memory">Activation memory limit (MB)</label>
          <input id="helix-activation-memory" class="input" type="number" min="0" max="65536" step="64" bind:value={trainFormStore.activationMemoryMb} />
          <span class="field-hint">Activations above this limit are temporarily stored in the output directory. 0 uses disk for all stored activations.</span>
        </div>
        <div class="field">
          <label for="helix-prefetch"><input id="helix-prefetch" type="checkbox" bind:checked={trainFormStore.helixPrefetch} /> Prefetch the next layer on CUDA</label>
          <span class="field-hint">Uses additional VRAM. Transfer overlap depends on the system; experts are loaded after routing.</span>
        </div>
        {#if trainFormStore.helixPrefetch}
          <div class="field">
            <label class="field-label" for="helix-prefetch-memory">Pinned prefetch memory limit (MB)</label>
            <input id="helix-prefetch-memory" class="input" type="number" min="0" max="4096" step="64" bind:value={trainFormStore.helixPrefetchMemoryMb} />
            <span class="field-hint">Limits pinned staging for future layer weights. 0 uses pageable transfers.</span>
          </div>
        {/if}
      {/if}
      <div class="field">
        <label class="field-label" for="performance-mode">Performance</label>
        <select id="performance-mode" class="select" bind:value={trainFormStore.performanceMode}>
          <option value="auto">Auto</option>
          <option value="safe">Safe - low memory</option>
          <option value="fast">Fast</option>
        </select>
        <span class="field-hint">Plans workers and memory use while preserving your training settings. Resolved settings appear in the run log.</span>
      </div>

      <!-- Dataset Picker -->
      <section class="card" id="section-dataset">
        <div class="card-header" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:var(--space-2)">
          <div style="display:flex; align-items:center; gap:var(--space-4)">
            <span class="text-label">Dataset Source</span>
          </div>
            <div class="segmented-control">
              <button class:active={trainFormStore.dataSplitSource === 'hf'} onclick={() => trainFormStore.dataSplitSource = 'hf'}>HF</button>
              <button class:active={trainFormStore.dataSplitSource === 'local'} onclick={() => trainFormStore.dataSplitSource = 'local'}>Local</button>
              <button class:active={trainFormStore.dataSplitSource === 'synthetic'} onclick={() => trainFormStore.dataSplitSource = 'synthetic'}>Synthetic</button>
              <button class:active={trainFormStore.dataSplitSource === 'klayer'} onclick={() => trainFormStore.dataSplitSource = 'klayer'}>Klayer</button>
            </div>
        </div>
        <div class="card-body">
            {#if trainFormStore.dataSplitSource === 'hf'}
              <div class="grid-2">
                <div class="field">
                  <label class="field-label" for="input-hf-repo">HF Dataset Repo ID</label>
                  <input id="input-hf-repo" class="input input-mono" bind:value={trainFormStore.hfRepoId} placeholder="org/dataset" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-hf-config">Subset / Config (Optional)</label>
                  <input id="input-hf-config" class="input input-mono" bind:value={trainFormStore.hfConfig} placeholder="default" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-hf-split">Split</label>
                  <input id="input-hf-split" class="input" bind:value={trainFormStore.hfSplit} placeholder="train" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-hf-revision">Revision (Optional)</label>
                  <input id="input-hf-revision" class="input input-mono" bind:value={trainFormStore.hfRevision} placeholder="main" />
                </div>
              </div>
            {:else if trainFormStore.dataSplitSource === 'local'}
              <div class="grid-2">
                <div class="field col-2">
                  <label class="field-label" for="input-local-path">Local File Path</label>
                  <input id="input-local-path" class="input input-mono" bind:value={trainFormStore.localPath} placeholder="C:/path/to/dataset.csv" />
                </div>
                <div class="field">
                  <label class="field-label" for="select-local-format">Format</label>
                  <select id="select-local-format" class="select" bind:value={trainFormStore.localFormat}>
                    <option value="jsonl">JSONL (.jsonl)</option>
                    <option value="csv">CSV (.csv)</option>
                    <option value="parquet">Parquet (.parquet)</option>
                    <option value="manifest">Dataset manifest (.json)</option>
                  </select>
                </div>
                {#if trainFormStore.localFormat === 'csv' || trainFormStore.localFormat === 'parquet'}
                  <div class="field col-2 mapping-builder border-surface">
                    <div class="text-title" style="font-size:12px; margin-bottom:var(--space-2)">Column Mapping</div>
                    <div class="grid-2">
                      <div class="field">
                        <label class="field-label" for="input-local-prompt">Prompt Column</label>
                        <input id="input-local-prompt" class="input input-mono" bind:value={trainFormStore.localPromptCol} placeholder="prompt" />
                      </div>
                      <div class="field">
                        <label class="field-label" for="input-local-completion">Completion Column</label>
                        <input id="input-local-completion" class="input input-mono" bind:value={trainFormStore.localCompletionCol} placeholder="completion" />
                      </div>
                    </div>
                  </div>
                {/if}
              </div>
            {:else if trainFormStore.dataSplitSource === 'synthetic'}
              <div class="grid-2">
                <div class="field">
                  <label class="field-label" for="input-synth-generator">Generator Model</label>
                  <input id="input-synth-generator" class="input input-mono" bind:value={trainFormStore.synthGenerator} />
                </div>
                <div class="field">
                  <label class="field-label" for="input-synth-judge">Judge Model</label>
                  <input id="input-synth-judge" class="input input-mono" bind:value={trainFormStore.synthJudge} />
                </div>
                <div class="field">
                  <label class="field-label" for="select-synth-mode">Mode</label>
                  <select id="select-synth-mode" class="select" bind:value={trainFormStore.synthMode}>
                    <option value="prompts">Prompts Only</option>
                    <option value="sft">SFT (Prompt + Response)</option>
                    <option value="dpo">DPO (Prompt + Chosen + Rejected)</option>
                  </select>
                </div>
                <div class="field">
                  <label class="field-label" for="input-synth-count">Samples to Generate</label>
                  <input id="input-synth-count" type="number" class="input" bind:value={trainFormStore.synthCount} />
                </div>
                <div class="field col-2">
                  <label class="field-label" for="input-synth-topic">Topic</label>
                  <input id="input-synth-topic" class="input" bind:value={trainFormStore.synthTopic} placeholder="machine learning safety" />
                </div>
              </div>
            {:else if trainFormStore.dataSplitSource === 'klayer'}
              <div class="grid-2">
                <div class="field col-2">
                  <label class="field-label" for="input-klayer-domain">Klayer domain (MCP export_dataset)</label>
                  <input id="input-klayer-domain" class="input input-mono" bind:value={trainFormStore.klayerDomain} placeholder="Leave empty to export every reviewed+user domain" />
                </div>
              </div>
            {/if}

          <div class="field" style="margin-top:var(--space-4)">
            <label class="field-label" for="validation-path">Validation dataset (optional)</label>
            <input id="validation-path" class="input input-mono" bind:value={trainFormStore.validationPath} placeholder="Workspace path to .jsonl, .parquet, or .json manifest" />
            <span class="field-hint">Use a separate file within the workspace. Evaluation runs during training.</span>
          </div>

          <!-- Preview section -->
          <div style="margin-top:var(--space-4); display:flex; flex-direction:column; gap:var(--space-2)">
            <button class="btn btn-secondary" onclick={previewDataset} disabled={previewLoading} style="align-self:flex-start; display: flex; align-items: center; gap: var(--space-2)">
              {#if previewLoading}<span class="spinner"></span>{/if}
              <i class="bi bi-search" style="font-size: 11px"></i>
              Preview Dataset
            </button>

            {#if previewRows}
              <div class="preview-table-container">
                <table class="preview-table">
                  <thead>
                    <tr>
                      {#each previewRows[0] as col}
                        <th>{col}</th>
                      {/each}
                    </tr>
                  </thead>
                  <tbody>
                    {#each previewRows.slice(1) as row}
                      <tr>
                        {#each row as cell}
                          <td title={cell}>{cell.length > 120 ? cell.slice(0, 120) + '...' : cell}</td>
                        {/each}
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {/if}
          </div>
        </div>
      </section>

      <!-- Adapter -->
      <section class="card" id="section-adapter">
        <div class="card-header"><span class="text-label">Adapter</span></div>
        {#if trainFormStore.workflow !== 'fresh'}<p class="field-hint">Rank, alpha, dropout, and targets are inherited from the source adapter. Continuing a router adapter uses one tenth of the learning rate; a full resume preserves the saved optimizer groups.</p>{/if}
        <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-4)">
          <div class="grid-2">
            <div class="field">
              <label class="field-label" for="select-adapter-kind">Kind</label>
              <select id="select-adapter-kind" class="select" bind:value={trainFormStore.adapterKind}>
                <option value="lora">LoRA</option>
                <option value="qlora">QLoRA (quantized)</option>
                <option value="dora">DoRA</option>
              </select>
            </div>
            <div class="field">
              <label class="field-label" for="select-quant-bits">Quantization</label>
              <select id="select-quant-bits" class="select" bind:value={trainFormStore.quantBits}>
                <option value={0}>None</option>
                <option value={8}>8-bit</option>
                <option value={4}>4-bit NF4</option>
              </select>
            </div>
          </div>
          <div class="grid-3">
            <div class="field">
              <label class="field-label" for="slider-rank">Rank — {trainFormStore.rank}</label>
              <input id="slider-rank" type="range" class="slider" min="4" max="256" step="4" bind:value={trainFormStore.rank} />
            </div>
            <div class="field">
              <label class="field-label" for="slider-alpha">Alpha — {trainFormStore.alpha}</label>
              <input id="slider-alpha" type="range" class="slider" min="4" max="512" step="4" bind:value={trainFormStore.alpha} />
            </div>
            <div class="field">
              <label class="field-label" for="slider-dropout">Dropout — {trainFormStore.dropout.toFixed(2)}</label>
              <input id="slider-dropout" type="range" class="slider" min="0" max="0.5" step="0.01" bind:value={trainFormStore.dropout} />
            </div>
          </div>
          {#if trainFormStore.workflow === 'fresh'}
            <div class="field">
              <label for="rslora"><input id="rslora" type="checkbox" bind:checked={trainFormStore.useRsLora} /> Use rank-stabilized LoRA (rsLoRA)</label>
              <span class="field-hint">Uses alpha / √rank.</span>
            </div>
            <div class="field">
              <label class="field-label" for="moe-lora-topology">LoRA targets</label>
              <select id="moe-lora-topology" class="select" bind:value={trainFormStore.moeTopology}>
                <option value="attention">Attention projections</option>
                <option value="moe_hybrid">MoE: attention, router and shared experts</option>
                <option value="moe_full">MoE: attention, router and all experts</option>
              </select>
              <span class="field-hint">{trainFormStore.trainingEngine === 'helix' ? 'Router adapters use one tenth of the learning rate. Shared experts (Qwen/DeepSeek) are included when present; full mode pages selected experts. Reload router and expert adapters with Helix; other loaders have not been verified.' : 'MoE options require separate supported Linear modules. Router adapters use one tenth of the learning rate; expert paging is not enabled.'}</span>
            </div>
            {#if trainFormStore.moeTopology !== 'attention' && trainFormStore.trainMode === 'sft'}
              <div class="field"><label class="field-label" for="router-aux">Router balance coefficient</label><input id="router-aux" class="input" type="number" min="0" max="1" step="0.001" bind:value={trainFormStore.routerAuxCoef} /><span class="field-hint">0 disables the auxiliary loss. Requires compatible softmax routing and a padding mask.</span></div>
            {/if}
          {/if}
        </div>
      </section>

      <!-- Optimizer + Training in one row -->
      <div class="grid-2">
        <section class="card" id="section-optim">
          <div class="card-header"><span class="text-label">Optimizer</span></div>
          <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
            <div class="field">
              <label class="field-label" for="input-lr">Learning Rate</label>
              <input id="input-lr" class="input input-mono" type="number" step="0.00001" bind:value={trainFormStore.lr} />
            </div>
            <div class="field">
              <label class="field-label" for="select-schedule">LR Schedule</label>
              <select id="select-schedule" class="select" bind:value={trainFormStore.schedule}>
                <option value="cosine">Cosine</option>
                <option value="linear">Linear</option>
                <option value="constant">Constant</option>
              </select>
            </div>
            <div class="field">
              <label class="field-label" for="input-warmup">Warmup Steps</label>
              <input id="input-warmup" class="input" type="number" bind:value={trainFormStore.warmupSteps} />
            </div>
            <div class="field">
              <label class="field-label" for="input-grad-acc">Grad Accumulation</label>
              <input id="input-grad-acc" class="input" type="number" bind:value={trainFormStore.gradAcc} />
            </div>
            <div class="field">
              <label class="field-label" for="input-weight-decay">Weight Decay</label>
              <input id="input-weight-decay" class="input" type="number" step="0.001" bind:value={trainFormStore.weightDecay} />
            </div>
          </div>
        </section>

        <section class="card" id="section-train-params">
          <div class="card-header"><span class="text-label">Training</span></div>
          <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
            {#if trainFormStore.trainMode === 'sft'}
              <div class="field"><label class="field-label" for="neftune-alpha">NEFTune noise alpha</label><input id="neftune-alpha" class="input" type="number" min="0" max="100" step="0.1" bind:value={trainFormStore.neftuneAlpha} /><span class="field-hint">0 disables embedding noise.</span></div>
              <div class="field">
                <label for="completion-only-loss"><input id="completion-only-loss" type="checkbox" bind:checked={trainFormStore.completionOnlyLoss} /> Completion-only loss (mask prompt tokens)</label>
              </div>
            {/if}
            <div class="field">
              <label for="toggle-use-epochs"><input id="toggle-use-epochs" type="checkbox" bind:checked={trainFormStore.useEpochs} /> Schedule by Epochs (instead of Max Steps)</label>
            </div>
            {#if trainFormStore.useEpochs}
              <div class="field">
                <label class="field-label" for="input-epochs">Epochs</label>
                <input id="input-epochs" class="input" type="number" min="1" step="1" bind:value={trainFormStore.epochs} />
              </div>
            {:else}
              <div class="field">
                <label class="field-label" for="input-max-steps">Max Steps</label>
                <input id="input-max-steps" class="input" type="number" bind:value={trainFormStore.maxSteps} />
              </div>
            {/if}
            <div class="field">
              <label class="field-label" for="input-batch-size">Batch Size</label>
              <input id="input-batch-size" class="input" type="number" bind:value={trainFormStore.batchSize} />
            </div>
            <div class="field">
              <label class="field-label" for="input-seq-len">Max Seq Length</label>
              <input id="input-seq-len" class="input" type="number" bind:value={trainFormStore.maxSeqLen} />
            </div>
            <div class="field">
              <label class="field-label" for="input-save-every">Save Every N Steps</label>
              <input id="input-save-every" class="input" type="number" bind:value={trainFormStore.saveEvery} />
            </div>
            <div class="field">
              <label class="field-label" for="input-output-path">Output Path</label>
              <input id="input-output-path" class="input input-mono" bind:value={trainFormStore.outputPath} />
            </div>
            <div class="field">
                <label class="field-label" for="sequence-packing">Sequence packing</label>
                <select id="sequence-packing" class="select" bind:value={trainFormStore.packing}>
                  <option value="auto">Auto - when safe and useful</option>
                  <option value={false}>Off</option>
                  <option value={true}>On - requires compatible runtime</option>
                </select>
                {#if trainFormStore.trainingEngine === 'helix'}<span class="field-hint">For Helix, choose On explicitly to enable packing. Auto leaves it off.</span>{/if}
              </div>
          </div>
        </section>
      </div>

      <!-- Repositioned start button at bottom of forms -->
      <div style="margin-top:var(--space-4); display:flex">
        {#if run.status === 'running'}
          <button class="btn btn-secondary btn-lg" onclick={stopTrain} style="flex:1; justify-content:center; display: flex; align-items: center; gap: var(--space-2)">
            <i class="bi bi-stop-fill" style="font-size: 16px"></i>
            Stop Training
          </button>
        {:else}
          <button class="btn btn-primary btn-lg" onclick={startTrain} disabled={submitting} id="btn-start-train-bottom" style="flex:1; justify-content:center; display: flex; align-items: center; gap: var(--space-2)">
            {#if submitting}<span class="spinner"></span>{/if}
            <i class="bi bi-fire" style="font-size: 16px"></i>
            Start SFT / DPO Training
          </button>
        {/if}
      </div>

    </div>

    <!-- Summary panel -->
    <div class="page-side-panel">
      <div class="summary-section">
        <div class="summary-title">Config Summary</div>
        <div class="summary-row"><span class="summary-key">Model</span><span class="summary-val">{trainFormStore.model.split('/').pop()}</span></div>
        <div class="summary-row"><span class="summary-key">Mode</span><span class="summary-val">{trainFormStore.trainMode.toUpperCase()}</span></div>
        <div class="summary-row"><span class="summary-key">Engine</span><span class="summary-val">{trainFormStore.trainingEngine === 'helix' ? 'Helix' : 'Standard'}</span></div>
        <div class="summary-row"><span class="summary-key">Adapter</span><span class="summary-val">{trainFormStore.workflow === 'fresh' ? `${trainFormStore.adapterKind.toUpperCase()} r=${trainFormStore.rank}` : 'Source adapter'}</span></div>
        <div class="summary-row"><span class="summary-key">Quant</span><span class="summary-val">{trainFormStore.quantBits === 0 ? 'None' : trainFormStore.quantBits + '-bit'}</span></div>
        <div class="summary-row"><span class="summary-key">Horizon</span><span class="summary-val">{trainFormStore.useEpochs ? `${trainFormStore.epochs} epoch(s)` : `${trainFormStore.maxSteps} steps`}</span></div>
        <div class="summary-row"><span class="summary-key">LR</span><span class="summary-val">{trainFormStore.lr.toExponential(1)}</span></div>
        <div class="summary-row"><span class="summary-key">Batch×GradAcc</span><span class="summary-val">{trainFormStore.batchSize}×{trainFormStore.gradAcc}={trainFormStore.batchSize*trainFormStore.gradAcc}</span></div>
      </div>

      <div class="summary-divider"></div>

      <div class="summary-section">
        <div class="summary-title">VRAM Estimate</div>
        {#if trainFormStore.trainingEngine === 'helix'}
          <div class="field-hint">Calculated from the local checkpoint when training starts.</div>
        {:else}
        <div class="vram-gauge">
          <div class="vram-val">{(estimatedVram / 1024).toFixed(1)} GB</div>
          <div class="vram-bar">
            <div class="progress-bar" style="margin-top:var(--space-1)">
              <div class="progress-fill" style="width:{Math.min(estimatedVram / 245.76, 100).toFixed(0)}%"></div>
            </div>
          </div>
          <div class="field-hint" style="margin-top:4px">Estimate for 24 GB GPU</div>
        </div>
        {/if}
      </div>

      <div class="summary-divider"></div>

      <div class="summary-section">
        <div class="summary-title">Tips</div>
        <div class="tip-card">
          {#if trainFormStore.trainingEngine === 'helix'}
            <p>Helix requires a local Mixtral, Qwen MoE, or DeepSeek MoE checkpoint and a separate output directory.</p>
          {:else if trainFormStore.quantBits === 0 && trainFormStore.adapterKind === 'lora'}
            <p>Use 4-bit QLoRA to cut VRAM by ~65% for large models.</p>
          {:else if trainFormStore.quantBits === 4}
            <p>4-bit NF4 requires bitsandbytes ≥ 0.41. Unsloth handles this automatically.</p>
          {:else}
            <p>Lower rank (8–16) is often sufficient for instruction fine-tuning.</p>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .vram-gauge { display: flex; flex-direction: column; gap: 2px; }
  .vram-val { font-size: 22px; font-weight: 500; letter-spacing: -0.04em; color: var(--color-ink); }
  .tip-card {
    font-size: 12px; color: var(--color-ink-subtle); line-height: 1.6;
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm); padding: var(--space-3);
    border: 1px solid var(--color-border);
  }

  /* Segmented control for tabs */
  .segmented-control {
    display: flex;
    background: var(--color-surface-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 2px;
  }
  .segmented-control button {
    background: none;
    border: none;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    color: var(--color-ink-subtle);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .segmented-control button.active {
    background: var(--color-surface);
    color: var(--color-ink);
    box-shadow: var(--shadow-sm);
  }

  .mapping-builder {
    padding: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    background: var(--color-surface-muted);
  }

  .preview-table-container {
    margin-top: var(--space-2);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    overflow: auto;
    max-height: 200px;
    background: var(--color-surface);
  }
  .preview-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    text-align: left;
  }
  .preview-table th, .preview-table td {
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--color-border);
    max-width: 300px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .preview-table th {
    background: var(--color-surface-muted);
    font-weight: 500;
    color: var(--color-ink-subtle);
  }
</style>
