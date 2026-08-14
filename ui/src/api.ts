/**
 * Tauri bridge — wraps invoke() calls with typed payloads.
 * Browser-only mocks below run exclusively when `__TAURI_INTERNALS__` is
 * absent (Vite without the desktop shell). They never execute inside Tauri.
 */
import type {
  RunConfig, MergeConfig, OpRecord, GuiderRecipe,
  CompatResult, MergeMethod, HfParams, CatalogEntry,
  ModelDownloadStatus, LocalModelItem, MoeIndexResult,
} from './types'

export interface AppSettings {
  hf_cache_dir: string
  is_custom: boolean
  main_memory_limit_mb: number | null
  effective_main_memory_mb: number
  detected_ram_mb: number
  default_context_window?: number
  default_temperature?: number
  enable_flash_attention?: boolean
  kv_cache_quant?: string
  vram_limit_mb?: number | null
  cpu_kv_cache?: boolean
}

// Detect if we are inside the Tauri runtime
const isTauri = typeof (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ !== 'undefined'

async function invoke<T>(cmd: string, args?: unknown): Promise<T> {
  if (isTauri) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core')
    return tauriInvoke<T>(cmd, args as Record<string, unknown>)
  }
  // ── Mock responses for browser-only dev ──────────────────────────────
  return mockInvoke<T>(cmd, args)
}

// ─── Commands ─────────────────────────────────────────────────────────────────
export const api = {
  startTrain: (config: RunConfig) =>
    invoke<string>('start_train', { config }),

  startMerge: (config: MergeConfig) =>
    invoke<string>('start_merge', { config }),

  stopOp: (opId: string) =>
    invoke<void>('stop_op', { opId }),

  listRuns: () =>
    invoke<OpRecord[]>('list_runs'),

  deleteRun: (opId: string) =>
    invoke<void>('delete_run', { opId }),

  getHardwareInfo: () =>
    invoke<{ backend: string; vram_mb: number | null; ram_mb: number | null }>('get_hardware_info'),

  getSettings: () =>
    invoke<AppSettings>('get_settings'),

  setCacheDir: (path: string | null) =>
    invoke<AppSettings>('set_cache_dir', { path }),

  setMainMemoryLimit: (limitMb: number | null) =>
    invoke<AppSettings>('set_main_memory_limit', { limitMb }),

  guiderRecommend: (hardware?: { accelerator: string; total_vram_mb: number; total_ram_mb: number }) =>
    invoke<GuiderRecipe[]>('guider_recommend', { hardware }),

  mergeCheck: (models: string[], method: MergeMethod, baseModel?: string | null) =>
    invoke<CompatResult>('merge_check', { models, method, baseModel: baseModel ?? null }),

  previewDataset: (source: HfParams, rows: number) =>
    invoke<string[][]>('preview_dataset', { source, rows }),

  publishRun: (runOpId: string, repoId: string, isPrivate: boolean, token: string, license?: string) =>
    invoke<string>('publish_run', { runOpId, repoId, private: isPrivate, token, license: license ?? null }),

  downloadModel: (repoId: string, purpose: 'inference' | 'finetune' | 'merge', destDir?: string, quant?: string) =>
    invoke<{ op_id: string }>('download_model', { repoId, purpose, destDir: destDir ?? null, quant: quant ?? null }),

  cancelDownload: (destDir?: string) =>
    invoke<boolean>('cancel_download', { destDir: destDir ?? null }),

  listCatalog: () =>
    invoke<CatalogEntry[]>('list_catalog'),

  convertModel: (model: string, outtype?: string, outfile?: string) =>
    invoke<string>('convert_model', { model, outtype: outtype ?? 'auto', outfile: outfile ?? null }),

  exportModel: (model: string, name?: string, context?: number) =>
    invoke<string>('export_model', { model, name: name ?? null, context: context ?? 4096 }),

  getDownloadStatus: (destDir?: string) =>
    invoke<ModelDownloadStatus | null>('get_download_status', { destDir: destDir ?? null }),

  listLocalModels: (customDir?: string) =>
    invoke<LocalModelItem[]>('list_local_models', { customDir: customDir ?? null }),

  buildMoeIndex: (
    modelPath: string,
    adapter: string,
    expertFormat: string,
    expertRegex?: string,
  ) =>
    invoke<MoeIndexResult>('build_moe_index', {
      modelPath,
      adapter,
      expertFormat,
      expertRegex: expertRegex?.trim() || null,
    }),

  startChatServer: (
    modelPath: string,
    context?: number,
    vramLimit?: number,
    cpuKvCache?: boolean,
  ) =>
    invoke<boolean>('start_chat_server', {
      modelPath,
      context: context ?? null,
      vramLimit: vramLimit ?? null,
      cpuKvCache: cpuKvCache ?? null,
    }),

  stopChatServer: () =>
    invoke<boolean>('stop_chat_server'),

  planInference: (modelPath: string, context?: number, exportRuntimes?: boolean) =>
    invoke<Record<string, unknown>>('plan_inference', {
      modelPath,
      context: context ?? null,
      exportRuntimes: exportRuntimes ?? false,
    }),
}

// ─── Mock ─────────────────────────────────────────────────────────────────────
function mockInvoke<T>(cmd: string, _args?: unknown): Promise<T> {
  const mocks: Record<string, unknown> = {
    get_hardware_info: { backend: 'cuda', vram_mb: 24576, ram_mb: 65536 },
    get_settings: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: false, main_memory_limit_mb: null, effective_main_memory_mb: 65536, detected_ram_mb: 65536 },
    set_cache_dir: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: true, main_memory_limit_mb: null, effective_main_memory_mb: 65536, detected_ram_mb: 65536 },
    set_main_memory_limit: { hf_cache_dir: 'D:\\models\\.hf-cache', is_custom: false, main_memory_limit_mb: 49152, effective_main_memory_mb: 49152, detected_ram_mb: 65536 },
    list_runs: [
      {
        op_id: 'abc-001', kind: 'train', status: 'done',
        artifact_path: 'runs/adapter-mistral-7b',
        config: { model: 'mistralai/Mistral-7B-v0.1' }, provenance: null,
      },
      {
        op_id: 'abc-002', kind: 'merge', status: 'error',
        artifact_path: 'runs/merged-llama',
        config: { merge_method: 'slerp' }, provenance: null,
      },
    ],
    start_train: 'mock-run-' + Math.random().toString(36).slice(2),
    start_merge: 'mock-merge-' + Math.random().toString(36).slice(2),
    stop_op: undefined,
    start_chat_server: true,
    stop_chat_server: true,
    plan_inference: {
      compatible: true,
      backend: 'llama_cpp',
      reasons: ['GPU-first hybrid plan (browser mock)'],
      warnings: [],
      estimates: {
        architecture: 'qwen3moe',
        is_moe: true,
        gpu_layers: 18,
        peak_vram_mb: 9800,
        peak_ram_mb: 4200,
        quantization: 'MOSTLY_Q4_K_M',
      },
      command: ['llama-server', '-m', 'model.gguf', '-ngl', '18'],
    },
    build_moe_index: {
      runtime_manifest: '.sytra-runtime.json',
      experts_indexed: 1024,
      dense_bytes: 1024,
      forward_verified: false,
    },
    delete_run: undefined,
    guider_recommend: [
      {
        model: { name: 'Mistral 7B', family: 'mistral', param_count: 7, arch: 'mistral', dtype: 'bfloat16', vram_fp16_mb: 14336, license: 'apache-2.0', hf_id: 'mistralai/Mistral-7B-v0.1' },
        adapter: { kind: 'qlora', rank: 16, alpha: 32, dropout: 0.05, target_modules: [], quant_bits: 4 },
        reason: 'Fits in 24 GB VRAM with 4-bit QLoRA — excellent instruction-following at 7B scale',
        fits_vram: true,
      },
      {
        model: { name: 'LLaMA-3 8B', family: 'llama', param_count: 8, arch: 'llama', dtype: 'bfloat16', vram_fp16_mb: 16384, license: 'llama3', hf_id: 'meta-llama/Meta-Llama-3-8B' },
        adapter: { kind: 'lora', rank: 32, alpha: 64, dropout: 0.05, target_modules: [], quant_bits: null },
        reason: 'Full 16-bit LoRA fits at 24 GB — best benchmark performance in the 8B class',
        fits_vram: true,
      },
      {
        model: { name: 'Mistral 22B', family: 'mistral', param_count: 22, arch: 'mistral', dtype: 'bfloat16', vram_fp16_mb: 44032, license: 'apache-2.0', hf_id: 'mistralai/Mistral-22B-v0.1' },
        adapter: { kind: 'qlora', rank: 8, alpha: 16, dropout: 0.05, target_modules: [], quant_bits: 4 },
        reason: 'Requires 4-bit quant — exceeds available VRAM at fp16, but fits with 4-bit NF4',
        fits_vram: false,
      },
    ],
    merge_check: { verdict: 'green', reason: 'Compatible architectures' },
    preview_dataset: [
      ['instruction', 'response'],
      ['What is 2+2?', '4'],
      ['Summarise this text.', 'Summary here.'],
    ],
    publish_run: 'mock-publish-' + Math.random().toString(36).slice(2),
    download_model: { op_id: 'mock-dl-' + Math.random().toString(36).slice(2) },
    cancel_download: true,
    list_local_models: [],
    get_download_status: null,
    list_catalog: [
      { id: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', name: 'Qwen2.5 0.5B Instruct GGUF', size_gb: 0.47, format: 'gguf', tags: ['small', 'fast'], recommended: true, downloadable: true, architecture: 'Qwen2ForCausalLM', alert_level: 'none', alerts: [] },
      { id: 'Qwen/Qwen3.5-9B-Base', name: 'Qwen3.5 9B Base', size_gb: 20, format: 'safetensors', tags: ['multimodal'], recommended: false, downloadable: true, architecture: 'Qwen3_5ForConditionalGeneration', alert_level: 'danger', alerts: [{ level: 'danger', code: 'never_qwen2', message: 'This is Qwen3.5, not Qwen2.', blocks_download: false }] },
      { id: 'unsloth/Kimi-K2.7-Code-GGUF', name: 'Kimi K2.7 Coder (MoE)', size_gb: 295, format: 'gguf', tags: ['coding', 'moe', 'large'], recommended: false, downloadable: true, alert_level: 'danger', alerts: [{ level: 'danger', code: 'exceeds_hybrid_envelope', message: 'Does not fit a GPU-first hybrid plan on this machine.', blocks_download: false }] },
    ],
  }
  return Promise.resolve((mocks[cmd] ?? null) as T)
}
