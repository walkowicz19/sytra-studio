<script lang="ts">
  import { api } from '../api'
  import { run, resetRun, applyTelemetry, pushToast, trainFormStore, createDefaultDatasetItem, watchTelemetry } from '../store.svelte'
  import type { RunConfig, TrainMode, AdapterType, Schedule, BackendKind } from '../types'

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
    if (trainFormStore.useMultiple) {
      if (trainFormStore.datasets.length === 0) {
        pushToast('error', 'Add at least one dataset to preview')
        previewLoading = false
        return
      }
      const first = trainFormStore.datasets[0]
      sourceObj = { source: first.source }
      if (first.source === 'hf') {
        sourceObj.hf = {
          repo_id: first.hfRepoId,
          split: first.hfSplit,
          revision: first.hfRevision || null,
          config: first.hfConfig || null
        }
      } else if (first.source === 'local') {
        sourceObj.local = {
          path: first.localPath,
          format: first.localFormat,
          mapping: {
            prompt: first.localPromptCol,
            completion: first.localCompletionCol
          }
        }
      } else if (first.source === 'synthetic') {
        sourceObj.synthetic = {
          generator_model: first.synthGenerator,
          judge_model: first.synthJudge,
          mode: first.synthMode,
          count: 5,
          topic: first.synthTopic
        }
      } else if (first.source === 'klayer') {
        sourceObj.klayer = {
          query: first.klayerQuery,
          min_trust_tier: first.klayerMinTrust,
          snapshot: first.klayerSnapshot
        }
      }
    } else {
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
        sourceObj.klayer = {
          query: trainFormStore.klayerQuery,
          min_trust_tier: trainFormStore.klayerMinTrust,
          snapshot: trainFormStore.klayerSnapshot
        }
      }
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
    submitting = true; resetRun()

    let dataSpec: any = {}

    if (trainFormStore.useMultiple) {
      if (trainFormStore.datasets.length === 0) {
        pushToast('error', 'Must specify at least one dataset')
        submitting = false
        return
      }
      if (trainFormStore.datasets.length > 5) {
        pushToast('error', 'Maximum limit of datasets is 5')
        submitting = false
        return
      }
      dataSpec = {
        source: 'multi',
        jsonl_path: null,
        fingerprint: null,
        datasets: trainFormStore.datasets.map(ds => {
          const itemSpec: any = {
            source: ds.source,
            jsonl_path: null,
            fingerprint: null
          }
          if (ds.source === 'hf') {
            itemSpec.hf = {
              repo_id: ds.hfRepoId,
              split: ds.hfSplit,
              revision: ds.hfRevision || null,
              config: ds.hfConfig || null
            }
          } else if (ds.source === 'local') {
            itemSpec.local = {
              path: ds.localPath,
              format: ds.localFormat,
              mapping: {
                prompt: ds.localPromptCol,
                completion: ds.localCompletionCol
              }
            }
          } else if (ds.source === 'synthetic') {
            itemSpec.synthetic = {
              generator_model: ds.synthGenerator,
              judge_model: ds.synthJudge,
              mode: ds.synthMode,
              count: ds.synthCount,
              topic: ds.synthTopic
            }
          } else if (ds.source === 'klayer') {
            itemSpec.klayer = {
              query: ds.klayerQuery,
              min_trust_tier: ds.klayerMinTrust,
              snapshot: ds.klayerSnapshot
            }
          }
          return itemSpec
        })
      }
    } else {
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
        dataSpec.klayer = {
          query: trainFormStore.klayerQuery,
          min_trust_tier: trainFormStore.klayerMinTrust,
          snapshot: trainFormStore.klayerSnapshot
        }
      }
    }

    const config: any = {
      version: 1, train_mode: trainFormStore.trainMode, model: trainFormStore.model,
      backend: { kind: trainFormStore.backend },
      data: dataSpec,
      adapter: { type: trainFormStore.adapterKind, rank: trainFormStore.rank, alpha: trainFormStore.alpha, dropout: trainFormStore.dropout, target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], quant_bits: trainFormStore.quantBits === 0 ? null : trainFormStore.quantBits },
      optim: { learning_rate: trainFormStore.lr, schedule: trainFormStore.schedule, warmup_steps: trainFormStore.warmupSteps, weight_decay: trainFormStore.weightDecay, grad_accumulation_steps: trainFormStore.gradAcc },
      train: { max_steps: trainFormStore.maxSteps, batch_size: trainFormStore.batchSize, max_seq_len: trainFormStore.maxSeqLen, save_every: trainFormStore.saveEvery, packing: trainFormStore.packing },
      output: { adapter_path: trainFormStore.outputPath },
    }
    try {
      const opId = await api.startTrain(config)
      run.opId = opId; run.kind = 'train'; run.status = 'running'; run.startedAt = Date.now()
      pushToast('success', `Training started — ${opId.slice(0, 8)}`)
      if (!('__TAURI_INTERNALS__' in window)) {
        simulateTelemetry()
      } else {
        watchTelemetry(opId)
      }
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
            <label class="field-label" for="input-model">Base Model</label>
            <input id="input-model" class="input input-mono" bind:value={trainFormStore.model} placeholder="org/model-name" />
            <span class="field-hint">Hugging Face model ID or local path</span>
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
              <option value="mps">MPS (Apple Silicon)</option>
              <option value="cpu">CPU (slow)</option>
            </select>
          </div>
        </div>
      </section>

      <!-- Dataset Picker -->
      <section class="card" id="section-dataset">
        <div class="card-header" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:var(--space-2)">
          <div style="display:flex; align-items:center; gap:var(--space-4)">
            <span class="text-label">Dataset Source</span>
            <label style="display:flex; align-items:center; gap:6px; font-size:12px; font-weight:500; cursor:pointer; user-select:none; color:var(--color-ink-subtle)">
              <input type="checkbox" bind:checked={trainFormStore.useMultiple} onchange={() => {
                if (trainFormStore.useMultiple && trainFormStore.datasets.length === 0) {
                  trainFormStore.datasets = [createDefaultDatasetItem(trainFormStore.dataSplitSource)];
                }
              }} style="width:14px; height:14px; accent-color:var(--color-brand)" />
              Mix Multiple Datasets (Max 5)
            </label>
          </div>
          {#if !trainFormStore.useMultiple}
            <div class="segmented-control">
              <button class:active={trainFormStore.dataSplitSource === 'hf'} onclick={() => trainFormStore.dataSplitSource = 'hf'}>HF</button>
              <button class:active={trainFormStore.dataSplitSource === 'local'} onclick={() => trainFormStore.dataSplitSource = 'local'}>Local</button>
              <button class:active={trainFormStore.dataSplitSource === 'synthetic'} onclick={() => trainFormStore.dataSplitSource = 'synthetic'}>Synthetic</button>
              <button class:active={trainFormStore.dataSplitSource === 'klayer'} onclick={() => trainFormStore.dataSplitSource = 'klayer'}>Klayer</button>
            </div>
          {/if}
        </div>
        <div class="card-body">
          {#if trainFormStore.useMultiple}
            <div style="display:flex; flex-direction:column; gap:var(--space-4)">
              {#each trainFormStore.datasets as ds, idx (ds.id)}
                <div class="card border-surface animate-in" style="padding:var(--space-3); background:var(--color-surface-muted)">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:var(--space-3)">
                    <div style="display:flex; align-items:center; gap:var(--space-3)">
                      <span class="badge badge-info" style="font-size:10px; font-weight:600">Dataset #{idx + 1}</span>
                      <select class="select" bind:value={ds.source} style="padding: 2px 6px; font-size:11px; height:auto; width:auto">
                        <option value="hf">Hugging Face</option>
                        <option value="local">Local File</option>
                        <option value="synthetic">Synthetic</option>
                        <option value="klayer">Klayer</option>
                      </select>
                    </div>
                    <button class="btn btn-ghost btn-icon text-error" onclick={() => {
                      trainFormStore.datasets = trainFormStore.datasets.filter(item => item.id !== ds.id)
                    }} style="width:24px; height:24px; font-size:12px; display:flex; align-items:center; justify-content:center" title="Remove dataset">
                      <i class="bi bi-trash"></i>
                    </button>
                  </div>

                  {#if ds.source === 'hf'}
                    <div class="grid-2">
                      <div class="field">
                        <label class="field-label" for="hf-repo-{ds.id}">HF Dataset Repo ID</label>
                        <input id="hf-repo-{ds.id}" class="input input-mono" bind:value={ds.hfRepoId} placeholder="org/dataset" />
                      </div>
                      <div class="field">
                        <label class="field-label" for="hf-config-{ds.id}">Subset / Config (Optional)</label>
                        <input id="hf-config-{ds.id}" class="input input-mono" bind:value={ds.hfConfig} placeholder="default" />
                      </div>
                      <div class="field">
                        <label class="field-label" for="hf-split-{ds.id}">Split</label>
                        <input id="hf-split-{ds.id}" class="input" bind:value={ds.hfSplit} placeholder="train" />
                      </div>
                      <div class="field">
                        <label class="field-label" for="hf-rev-{ds.id}">Revision (Optional)</label>
                        <input id="hf-rev-{ds.id}" class="input input-mono" bind:value={ds.hfRevision} placeholder="main" />
                      </div>
                    </div>
                  {:else if ds.source === 'local'}
                    <div class="grid-2">
                      <div class="field col-2">
                        <label class="field-label" for="local-path-{ds.id}">Local File Path</label>
                        <input id="local-path-{ds.id}" class="input input-mono" bind:value={ds.localPath} placeholder="C:/path/to/dataset.csv" />
                      </div>
                      <div class="field">
                        <label class="field-label" for="local-format-{ds.id}">Format</label>
                        <select id="local-format-{ds.id}" class="select" bind:value={ds.localFormat}>
                          <option value="jsonl">JSONL (.jsonl)</option>
                          <option value="csv">CSV (.csv)</option>
                          <option value="parquet">Parquet (.parquet)</option>
                        </select>
                      </div>
                      {#if ds.localFormat === 'csv' || ds.localFormat === 'parquet'}
                        <div class="field col-2 mapping-builder border-surface" style="margin-top:var(--space-2)">
                          <div class="text-title" style="font-size:12px; margin-bottom:var(--space-2)">Column Mapping</div>
                          <div class="grid-2">
                            <div class="field">
                              <label class="field-label" for="local-prompt-{ds.id}">Prompt Column</label>
                              <input id="local-prompt-{ds.id}" class="input input-mono" bind:value={ds.localPromptCol} placeholder="prompt" />
                            </div>
                            <div class="field">
                              <label class="field-label" for="local-completion-{ds.id}">Completion Column</label>
                              <input id="local-completion-{ds.id}" class="input input-mono" bind:value={ds.localCompletionCol} placeholder="completion" />
                            </div>
                          </div>
                        </div>
                      {/if}
                    </div>
                  {:else if ds.source === 'synthetic'}
                    <div class="grid-2">
                      <div class="field">
                        <label class="field-label" for="synth-gen-{ds.id}">Generator Model</label>
                        <input id="synth-gen-{ds.id}" class="input input-mono" bind:value={ds.synthGenerator} />
                      </div>
                      <div class="field">
                        <label class="field-label" for="synth-judge-{ds.id}">Judge Model</label>
                        <input id="synth-judge-{ds.id}" class="input input-mono" bind:value={ds.synthJudge} />
                      </div>
                      <div class="field">
                        <label class="field-label" for="synth-mode-{ds.id}">Mode</label>
                        <select id="synth-mode-{ds.id}" class="select" bind:value={ds.synthMode}>
                          <option value="prompts">Prompts Only</option>
                          <option value="sft">SFT (Prompt + Response)</option>
                          <option value="dpo">DPO (Prompt + Chosen + Rejected)</option>
                        </select>
                      </div>
                      <div class="field">
                        <label class="field-label" for="synth-count-{ds.id}">Sample Count</label>
                        <input id="synth-count-{ds.id}" type="number" class="input" bind:value={ds.synthCount} min="5" max="1000" />
                      </div>
                      <div class="field col-2">
                        <label class="field-label" for="synth-topic-{ds.id}">Seed Topic</label>
                        <input id="synth-topic-{ds.id}" class="input" bind:value={ds.synthTopic} />
                      </div>
                    </div>
                  {:else if ds.source === 'klayer'}
                    <div class="grid-2">
                      <div class="field col-2">
                        <label class="field-label" for="klayer-query-{ds.id}">Klayer SQL Query</label>
                        <input id="klayer-query-{ds.id}" class="input input-mono" bind:value={ds.klayerQuery} />
                      </div>
                      <div class="field">
                        <label class="field-label" for="klayer-trust-{ds.id}">Min Trust Tier</label>
                        <select id="klayer-trust-{ds.id}" class="select" bind:value={ds.klayerMinTrust}>
                          <option value="tier-1">Tier 1 (highest)</option>
                          <option value="tier-2">Tier 2</option>
                          <option value="tier-3">Tier 3</option>
                        </select>
                      </div>
                      <div class="field">
                        <label class="field-label" for="klayer-snapshot-{ds.id}">Snapshot</label>
                        <input id="klayer-snapshot-{ds.id}" class="input input-mono" bind:value={ds.klayerSnapshot} />
                      </div>
                    </div>
                  {/if}
                </div>
              {/each}

              <button
                class="btn btn-secondary btn-sm"
                onclick={() => {
                  if (trainFormStore.datasets.length < 5) {
                    trainFormStore.datasets = [...trainFormStore.datasets, createDefaultDatasetItem()];
                  }
                }}
                disabled={trainFormStore.datasets.length >= 5}
                style="align-self:flex-start"
              >
                <i class="bi bi-plus" style="margin-right:4px"></i> Add Dataset ({trainFormStore.datasets.length}/5)
              </button>
            </div>
          {:else}
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
                  <label class="field-label" for="input-klayer-query">Klayer Query</label>
                  <input id="input-klayer-query" class="input input-mono" bind:value={trainFormStore.klayerQuery} />
                </div>
                <div class="field">
                  <label class="field-label" for="input-klayer-trust">Minimum Trust Tier</label>
                  <input id="input-klayer-trust" class="input" bind:value={trainFormStore.klayerMinTrust} placeholder="tier-2" />
                </div>
                <div class="field">
                  <label class="field-label" for="input-klayer-snapshot">Snapshot Hash</label>
                  <input id="input-klayer-snapshot" class="input input-mono" bind:value={trainFormStore.klayerSnapshot} />
                </div>
              </div>
            {/if}
          {/if}

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
            <div class="field">
              <label class="field-label" for="input-max-steps">Max Steps</label>
              <input id="input-max-steps" class="input" type="number" bind:value={trainFormStore.maxSteps} />
            </div>
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
            <div class="toggle-row">
              <div class="toggle-row-text">
                <span class="field-label">Sequence Packing</span>
                <span class="field-hint">Pack short sequences together</span>
              </div>
              <label class="toggle" id="toggle-packing">
                <input type="checkbox" bind:checked={trainFormStore.packing} />
                <div class="toggle-track"><div class="toggle-thumb"></div></div>
              </label>
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
        <div class="summary-row"><span class="summary-key">Adapter</span><span class="summary-val">{trainFormStore.adapterKind.toUpperCase()} r={trainFormStore.rank}</span></div>
        <div class="summary-row"><span class="summary-key">Quant</span><span class="summary-val">{trainFormStore.quantBits === 0 ? 'None' : trainFormStore.quantBits + '-bit'}</span></div>
        <div class="summary-row"><span class="summary-key">Steps</span><span class="summary-val">{trainFormStore.maxSteps}</span></div>
        <div class="summary-row"><span class="summary-key">LR</span><span class="summary-val">{trainFormStore.lr.toExponential(1)}</span></div>
        <div class="summary-row"><span class="summary-key">Batch×GradAcc</span><span class="summary-val">{trainFormStore.batchSize}×{trainFormStore.gradAcc}={trainFormStore.batchSize*trainFormStore.gradAcc}</span></div>
      </div>

      <div class="summary-divider"></div>

      <div class="summary-section">
        <div class="summary-title">VRAM Estimate</div>
        <div class="vram-gauge">
          <div class="vram-val">{(estimatedVram / 1024).toFixed(1)} GB</div>
          <div class="vram-bar">
            <div class="progress-bar" style="margin-top:var(--space-1)">
              <div class="progress-fill" style="width:{Math.min(estimatedVram / 245.76, 100).toFixed(0)}%"></div>
            </div>
          </div>
          <div class="field-hint" style="margin-top:4px">Estimate for 24 GB GPU</div>
        </div>
      </div>

      <div class="summary-divider"></div>

      <div class="summary-section">
        <div class="summary-title">Tips</div>
        <div class="tip-card">
          {#if trainFormStore.quantBits === 0 && trainFormStore.adapterKind === 'lora'}
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
