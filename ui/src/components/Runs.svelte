<script lang="ts">
  import { onMount } from 'svelte'
  import { historyStore, setRunHistory, pushToast, trainFormStore, setTab } from '../store.svelte'
  import { t } from '../i18n.svelte'
  import { api } from '../api'

  let loading  = $state(true)
  let deleting = $state<string | null>(null)
  let effectiveRamMb = $state(0)
  
  // Publish state
  let activePublishRun = $state<any | null>(null)
  let activeExportRun = $state<any | null>(null)
  let repoId = $state('')
  let isPrivate = $state(false)
  let license = $state('apache-2.0')
  let useEnvToken = $state(true)
  let pasteToken = $state('')
  let publishing = $state(false)
  
  // Telemetry logs for publish
  let publishProgress = $state(0)
  let publishLogs = $state<string[]>([])

  // Export guide: commands are built here because the Modelfile template
  // uses {{ }} which Svelte markup would parse as expressions.
  function exportMergedDir(r: any): string {
    return r.kind === 'train' ? `${r.artifact_path}-merged` : r.artifact_path
  }
  function exportAdapterScript(r: any): string {
    return `# save as merge_adapter.py, then run:
#   .sytra-envs\\train-env\\Scripts\\python merge_adapter.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import gc

BASE = "<base model id, e.g. Qwen/Qwen2.5-Coder-7B-Instruct>"
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")
gc.collect()

print("Loading adapter and merging...")
model = PeftModel.from_pretrained(model, r"${r.artifact_path}").merge_and_unload()
gc.collect()

print("Saving merged model...")
model.save_pretrained(r"${exportMergedDir(r)}", safe_serialization=True, max_shard_size="2GB")
AutoTokenizer.from_pretrained(r"${r.artifact_path}").save_pretrained(r"${exportMergedDir(r)}")
print("Merge complete!")`
  }
  // Use disk-backed temp files when effective RAM <= 20 GB to prevent OOM crashes
  // during GGUF quantization. The threshold is detected from AppSettings automatically.
  const LOW_RAM_THRESHOLD_MB = 20_480
  function exportConvertCmd(r: any): string {
    const useTempFile = effectiveRamMb > 0 && effectiveRamMb <= LOW_RAM_THRESHOLD_MB
    const tempFlag = useTempFile ? ' --use-temp-file' : ''
    return `.sytra-envs\\merge-env\\Scripts\\python -u .tools\\llama.cpp\\convert_hf_to_gguf.py "${exportMergedDir(r)}" --outtype q8_0 --outfile model.q8_0.gguf${tempFlag}`
  }
  const exportModelfile = `FROM ./model.q8_0.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER num_ctx 8192`
  const exportOllamaCmd = `ollama create my-sytra-model -f Modelfile
ollama run my-sytra-model`

  onMount(async () => {
    try {
      const settings = await api.getSettings()
      effectiveRamMb = settings.effective_main_memory_mb
    } catch {}
    await refresh()
  })

  async function refresh() {
    loading = true
    try { setRunHistory(await api.listRuns()) }
    catch { pushToast('error', t('runs.loadFailed')) }
    finally { loading = false }
  }

  async function deleteRun(opId: string) {
    deleting = opId
    try {
      await api.deleteRun(opId)
      setRunHistory(historyStore.records.filter(r => r.op_id !== opId))
      pushToast('info', t('runs.deleted'))
    } catch { pushToast('error', t('runs.deleteFailed')) }
    finally { deleting = null }
  }

  const statusBadge: Record<string, string> = {
    done: 'badge-success', running: 'badge-info', error: 'badge-error', stopped: 'badge-neutral',
  }
  const statusIcon: Record<string, string> = {
    done: 'bi-check-circle-fill', running: 'bi-arrow-repeat', error: 'bi-exclamation-octagon-fill', stopped: 'bi-stop-circle-fill',
  }

  let doneCount    = $derived(historyStore.records.filter(r => r.status === 'done').length)
  let errorCount   = $derived(historyStore.records.filter(r => r.status === 'error').length)
  let trainCount   = $derived(historyStore.records.filter(r => r.kind === 'train').length)
  let mergeCount   = $derived(historyStore.records.filter(r => r.kind === 'merge').length)

  // Pagination — at most 7 runs per page
  const PAGE_SIZE = 7
  let page = $state(0)
  let pageCount = $derived(Math.max(1, Math.ceil(historyStore.records.length / PAGE_SIZE)))
  let pagedRecords = $derived(historyStore.records.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE))
  // Deleting the last item of the last page must not strand the view.
  $effect(() => { if (page >= pageCount) page = pageCount - 1 })

  function healModel(record: any) {
    trainFormStore.model = record.artifact_path
    setTab('train')
    pushToast('success', 'Heal Mode: trainer pre-pointed to merged checkpoint.')
  }

  async function startPublish() {
    if (!repoId) {
      pushToast('error', 'Please enter a repository ID')
      return
    }
    publishing = true
    publishProgress = 0
    publishLogs = ['Initializing publication...']
    
    const token = useEnvToken ? '' : pasteToken
    try {
      const opId = await api.publishRun(activePublishRun.op_id, repoId, isPrivate, token)
      pushToast('success', 'Publish process launched')
      
      if (!('__TAURI_INTERNALS__' in window)) {
        simulatePublish()
      } else {
        // In Tauri, the telemetry events will stream back via Tauri event listener
        // We listen for publish progress updates or logs.
        // For simple UI, we can poll status or listen to sytra event line telemetry.
      }
    } catch (e: any) {
      pushToast('error', `Publish failed: ${e.message || String(e)}`)
      publishing = false
    }
  }

  function simulatePublish() {
    let p = 0
    const iv = setInterval(() => {
      p += 0.1
      publishProgress = Math.min(p * 100, 100)
      publishLogs = [...publishLogs, `Uploading chunk file index: ${Math.round(p * 10)}...`]
      
      if (p >= 1) {
        clearInterval(iv)
        publishing = false
        const targetUrl = `https://huggingface.co/${repoId}`
        pushToast('success', 'Model published successfully!')
        
        // Update local provenance link in runs history list
        historyStore.records = historyStore.records.map(r => 
          r.op_id === activePublishRun.op_id ? { ...r, provenance: targetUrl } : r
        )
        activePublishRun = null
      }
    }, 450)
  }
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">{t('runs.title')}</h1>
      <p class="text-small">{historyStore.records.length === 1 ? t('runs.countOne') : t('runs.count', { n: historyStore.records.length })}</p>
    </div>
    <button class="btn btn-secondary" onclick={refresh} disabled={loading} id="btn-refresh-runs" style="display: flex; align-items: center; gap: var(--space-1)">
      {#if loading}<span class="spinner"></span>{:else}<i class="bi bi-arrow-clockwise"></i>{/if}
      {t('runs.refresh')}
    </button>
  </div>

  <div class="page-content">
    <div class="page-form-area">
      {#if loading}
        <div class="empty-state"><span class="spinner spinner-lg"></span></div>
      {:else if historyStore.records.length === 0}
        <div class="empty-state">
          <div class="empty-icon" style="color: var(--color-ink-ghost); font-size: 24px"><i class="bi bi-clipboard-data"></i></div>
          <div class="text-title" style="margin-top:var(--space-2)">{t('runs.empty')}</div>
          <div class="text-small" style="text-align:center;max-width:260px;margin-top:var(--space-1)">{t('runs.emptySub')}</div>
        </div>
      {:else}
        <div class="run-list">
          {#each pagedRecords as record (record.op_id)}
            <div class="run-card card animate-in" id="run-{record.op_id}">
              <div class="run-header">
                <div class="run-meta">
                  <span class="badge badge-brand" style="font-size:10px">{record.kind.toUpperCase()}</span>
                  <span class="badge {statusBadge[record.status] ?? 'badge-neutral'}" style="display: inline-flex; align-items: center; gap: 4px">
                    <i class="bi {statusIcon[record.status]}"></i> {record.status}
                  </span>
                  <code class="run-id">{record.op_id.slice(0, 12)}…</code>
                </div>
                <button
                  class="btn btn-ghost btn-icon"
                  onclick={() => deleteRun(record.op_id)}
                  disabled={deleting === record.op_id}
                  id="btn-delete-{record.op_id}"
                  data-tooltip="Delete run"
                  aria-label="Delete run"
                  style="display: flex; align-items: center; justify-content: center; width: 24px; height: 24px"
                >
                  {#if deleting === record.op_id}<span class="spinner"></span>{:else}<i class="bi bi-trash"></i>{/if}
                </button>
              </div>
              <div class="run-body">
                <div class="run-path">
                  <span class="field-label">{t('runs.output')}</span>
                  <code class="run-path-val">{record.artifact_path}</code>
                </div>
                {#if record.provenance}
                  <div class="provenance-row" style="margin-top: var(--space-2); display: flex; align-items: center; gap: var(--space-2)">
                    <span class="badge badge-success" style="font-size: 10px"><i class="bi bi-cloud-check"></i> Published</span>
                    <a href={record.provenance} target="_blank" rel="noopener noreferrer" class="link" style="font-size: 12px">{record.provenance}</a>
                  </div>
                {/if}
                
                {#if record.status === 'done'}
                  <div class="run-actions" style="margin-top: var(--space-3); display: flex; gap: var(--space-2)">
                    <button class="btn btn-secondary btn-sm" onclick={() => { activePublishRun = record; repoId = '' }} style="display: flex; align-items: center; gap: 4px">
                      <i class="bi bi-cloud-arrow-up"></i> {t('runs.publish')}
                    </button>
                    <button class="btn btn-secondary btn-sm" onclick={() => { activeExportRun = record }} style="display: flex; align-items: center; gap: 4px">
                      <i class="bi bi-box-arrow-up-right"></i> {t('runs.export')}
                    </button>
                    {#if record.kind === 'merge'}
                      <button class="btn btn-secondary btn-sm" onclick={() => healModel(record)} style="display: flex; align-items: center; gap: 4px">
                        <i class="bi bi-heart-pulse"></i> {t('runs.heal')}
                      </button>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
          {/each}
        </div>

        {#if pageCount > 1}
          <div class="pagination">
            <button
              class="btn btn-ghost btn-sm"
              onclick={() => (page = Math.max(0, page - 1))}
              disabled={page === 0}
              id="btn-runs-prev"
            >
              <i class="bi bi-chevron-left"></i> {t('runs.prev')}
            </button>
            {#each Array(pageCount) as _, i}
              <button
                class="page-dot"
                class:active={page === i}
                onclick={() => (page = i)}
                aria-label="Page {i + 1}"
              >{i + 1}</button>
            {/each}
            <button
              class="btn btn-ghost btn-sm"
              onclick={() => (page = Math.min(pageCount - 1, page + 1))}
              disabled={page === pageCount - 1}
              id="btn-runs-next"
            >
              {t('runs.next')} <i class="bi bi-chevron-right"></i>
            </button>
          </div>
        {/if}
      {/if}
    </div>

    <!-- Stats panel -->
    <div class="page-side-panel">
      <div class="summary-section">
        <div class="summary-title">{t('runs.stats')}</div>
        <div class="summary-row"><span class="summary-key">{t('runs.total')}</span><span class="summary-val">{historyStore.records.length}</span></div>
        <div class="summary-row"><span class="summary-key">{t('runs.completed')}</span><span class="summary-val" style="color:var(--color-success)">{doneCount}</span></div>
        <div class="summary-row"><span class="summary-key">{t('runs.errors')}</span><span class="summary-val" style="color:var(--color-error)">{errorCount}</span></div>
        <div class="summary-row"><span class="summary-key">{t('runs.trainJobs')}</span><span class="summary-val">{trainCount}</span></div>
        <div class="summary-row"><span class="summary-key">{t('runs.mergeJobs')}</span><span class="summary-val">{mergeCount}</span></div>
      </div>

      <div class="summary-divider"></div>
      <div class="summary-section">
        <div class="summary-title">About</div>
        <p style="font-size:12px;color:var(--color-ink-subtle);line-height:1.6">
          Runs are stored as JSON files in <code>runs/</code>. Adapter checkpoints are saved to the configured output path.
        </p>
      </div>
    </div>
  </div>
</div>

<!-- Modal Publish Panel -->
{#if activePublishRun}
  <div class="modal-overlay animate-in">
    <div class="modal-card card">
      <div class="modal-header">
        <div class="text-title">Publish to Hugging Face Hub</div>
        <button class="btn btn-ghost btn-icon" onclick={() => activePublishRun = null} disabled={publishing} aria-label="Close" style="display: flex; align-items: center; justify-content: center; width: 28px; height: 28px">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <div class="modal-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
        <div class="field">
          <span class="field-label">Target Artifact</span>
          <code style="font-size: 11px; display:block; padding:var(--space-2); background:var(--color-surface-muted); border-radius:var(--radius-sm)">{activePublishRun.artifact_path}</code>
        </div>
        
        <div class="field">
          <label class="field-label" for="input-repo-id">Repository ID</label>
          <input id="input-repo-id" class="input input-mono" bind:value={repoId} placeholder="username/model-name" disabled={publishing} />
          <span class="field-hint">e.g. your-username/my-mistral-adapter</span>
        </div>

        <div class="grid-2">
          <div class="field">
            <label class="field-label" for="select-license">License</label>
            <select id="select-license" class="select" bind:value={license} disabled={publishing}>
              <option value="apache-2.0">Apache 2.0</option>
              <option value="mit">MIT</option>
              <option value="openrail">OpenRAIL</option>
              <option value="gpl-3.0">GPL 3.0</option>
              <option value="llama3">Llama 3 Community License</option>
            </select>
          </div>
          <div class="field" style="display:flex; flex-direction:column; justify-content:center">
            <div class="toggle-row" style="margin:0; padding:0; border:none">
              <span class="field-label" style="font-size:12px">Private Repository</span>
              <label class="toggle" id="toggle-repo-private">
                <input type="checkbox" bind:checked={isPrivate} disabled={publishing} />
                <div class="toggle-track"><div class="toggle-thumb"></div></div>
              </label>
            </div>
          </div>
        </div>

        <div class="toggle-row" style="margin:0; padding:var(--space-2) 0; border:none; border-top:1px solid var(--color-border)">
          <div class="toggle-row-text">
            <span class="field-label">Use HF_TOKEN Environment Token</span>
            <span class="field-hint">Reads local environment variables</span>
          </div>
          <label class="toggle" id="toggle-env-token">
            <input type="checkbox" bind:checked={useEnvToken} disabled={publishing} />
            <div class="toggle-track"><div class="toggle-thumb"></div></div>
          </label>
        </div>

        {#if !useEnvToken}
          <div class="field animate-in">
            <label class="field-label" for="input-publish-token">HF User Access Token</label>
            <input id="input-publish-token" type="password" class="input input-mono" bind:value={pasteToken} placeholder="hf_..." disabled={publishing} />
          </div>
        {/if}

        {#if publishing}
          <div class="publish-progress-section animate-in" style="margin-top:var(--space-2)">
            <span class="field-label">Publishing Progress — {publishProgress.toFixed(0)}%</span>
            <div class="progress-bar" style="margin-top:var(--space-1)">
              <div class="progress-fill" style="width: {publishProgress}%"></div>
            </div>
            <div class="console-box" style="height:100px; font-size:10px; margin-top:var(--space-3)">
              {#each publishLogs as log}
                <div class="log-line">{log}</div>
              {/each}
            </div>
          </div>
        {/if}
      </div>
      <div class="modal-footer" style="display:flex; gap:var(--space-2); margin-top:var(--space-2)">
        <button class="btn btn-secondary" onclick={() => activePublishRun = null} disabled={publishing} style="flex:1">Cancel</button>
        <button class="btn btn-primary" onclick={startPublish} disabled={publishing} style="flex:1; display:flex; align-items:center; justify-content:center; gap:var(--space-2)">
          {#if publishing}<span class="spinner"></span>{/if}
          <i class="bi bi-cloud-arrow-up-fill"></i> Upload Checkpoint
        </button>
      </div>
    </div>
  </div>
{/if}

<!-- Modal Export Panel -->
{#if activeExportRun}
  <div class="modal-overlay animate-in">
    <div class="modal-card card" style="max-width: 600px">
      <div class="modal-header">
        <div class="text-title" style="font-weight:600">{t('export.title')}</div>
        <button class="btn btn-ghost btn-icon" onclick={() => activeExportRun = null} aria-label="Close" style="display: flex; align-items: center; justify-content: center; width: 28px; height: 28px">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
      <div class="modal-body" style="display:flex; flex-direction:column; gap:var(--space-3); max-height:70vh; overflow-y:auto">
        <p class="text-small" style="color:var(--color-ink-subtle)">
          {t('export.intro', { path: activeExportRun.artifact_path })}
        </p>

        <div class="info-block" style="padding:var(--space-3); background:var(--color-surface-muted); border-radius:var(--radius-sm); border:1px solid var(--color-border); display:flex; flex-direction:column; gap:6px">
          <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">{t('export.reqTitle')}</span>
          <span class="text-small">— {t('export.req1')}</span>
          <span class="text-small">— {t('export.req2')}</span>
          <span class="text-small">— {t('export.req3')}</span>
        </div>

        {#if activeExportRun.kind === 'train'}
          <div class="info-block" style="padding:var(--space-3); background:var(--color-surface-muted); border-radius:var(--radius-sm); border:1px solid var(--color-border); display:flex; flex-direction:column; gap:6px">
            <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">1. {t('export.stepAdapter')}</span>
            <span class="text-small">{t('export.stepAdapterBody')}</span>
            <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow-x:auto; margin:0"><code style="font-family:var(--font-mono)">{exportAdapterScript(activeExportRun)}</code></pre>
          </div>
        {/if}

        <div class="info-block" style="padding:var(--space-3); background:var(--color-surface-muted); border-radius:var(--radius-sm); border:1px solid var(--color-border); display:flex; flex-direction:column; gap:6px">
          <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">{activeExportRun.kind === 'train' ? 2 : 1}. {t('export.stepConvert')}</span>
          <span class="text-small">{t('export.stepConvertBody')}</span>
          <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow-x:auto; margin:0"><code style="font-family:var(--font-mono)">{exportConvertCmd(activeExportRun)}</code></pre>
        </div>

        <div class="info-block" style="padding:var(--space-3); background:var(--color-surface-muted); border-radius:var(--radius-sm); border:1px solid var(--color-border); display:flex; flex-direction:column; gap:6px">
          <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">{activeExportRun.kind === 'train' ? 3 : 2}. {t('export.stepModelfile')}</span>
          <span class="text-small">{t('export.stepModelfileBody')}</span>
          <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow-x:auto; margin:0"><code style="font-family:var(--font-mono)">{exportModelfile}</code></pre>
        </div>

        <div class="info-block" style="padding:var(--space-3); background:var(--color-surface-muted); border-radius:var(--radius-sm); border:1px solid var(--color-border); display:flex; flex-direction:column; gap:6px">
          <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">{activeExportRun.kind === 'train' ? 4 : 3}. {t('export.stepRun')}</span>
          <span class="text-small">{t('export.stepRunBody')}</span>
          <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow-x:auto; margin:0"><code style="font-family:var(--font-mono)">{exportOllamaCmd}</code></pre>
        </div>
      </div>
      <div class="modal-footer" style="display:flex; justify-content:flex-end">
        <button class="btn btn-secondary" onclick={() => activeExportRun = null} style="min-width:100px">{t('export.close')}</button>
      </div>
    </div>
  </div>
{/if}

<style>
  .run-list { display: flex; flex-direction: column; gap: var(--space-3); }

  .pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    padding: var(--space-4) 0 var(--space-2);
  }
  .page-dot {
    min-width: 28px;
    height: 28px;
    padding: 0 var(--space-2);
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    color: var(--color-ink-subtle);
    font-family: var(--font-mono);
    font-size: 12px;
    cursor: pointer;
    transition: border-color var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .page-dot:hover { border-color: var(--color-border-strong); color: var(--color-ink); }
  .page-dot.active {
    border-color: var(--color-brand);
    color: var(--color-brand);
    font-weight: 600;
  }
  .run-card { overflow: hidden; }
  .run-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: var(--space-2) var(--space-3) var(--space-2) var(--space-4);
    background: var(--color-surface-muted);
    border-bottom: 1px solid var(--color-border);
  }
  .run-meta { display: flex; align-items: center; gap: var(--space-2); flex-wrap: wrap; }
  .run-id { font-family: var(--font-mono); font-size: 11px; color: var(--color-ink-ghost); }
  .run-body { padding: var(--space-3) var(--space-4); display: flex; flex-direction: column; gap: var(--space-2); }
  .run-path { display: flex; align-items: baseline; gap: var(--space-3); }
  .run-path-val { font-family: var(--font-mono); font-size: 12px; color: var(--color-ink-subtle); }

  /* Modal styling */
  .modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    backdrop-filter: blur(2px);
  }
  .modal-card {
    width: 100%;
    max-width: 500px;
    padding: var(--space-4);
    background: var(--color-surface);
    box-shadow: var(--shadow-lg);
  }
  .modal-header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--color-border);
    padding-bottom: var(--space-2);
    margin-bottom: var(--space-3);
  }
  .modal-footer {
    border-top: 1px solid var(--color-border);
    padding-top: var(--space-3);
    margin-top: var(--space-3);
  }
</style>
