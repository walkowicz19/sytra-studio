/**
 * Tauri bridge — wraps invoke() calls with typed payloads.
 * Falls back to mock data when running in a plain browser (dev mode without Tauri).
 */
import type {
  RunConfig, MergeConfig, OpRecord, GuiderRecipe,
  CompatResult, MergeMethod, HfParams,
} from './types'

export interface AppSettings {
  hf_cache_dir: string
  is_custom: boolean
  main_memory_limit_mb: number | null
  effective_main_memory_mb: number
  detected_ram_mb: number
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
    invoke<{ backend: string; vram_mb: number; ram_mb: number }>('get_hardware_info'),

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

  publishRun: (runOpId: string, repoId: string, isPrivate: boolean, token: string) =>
    invoke<string>('publish_run', { runOpId, repoId, private: isPrivate, token }),
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
  }
  return Promise.resolve((mocks[cmd] ?? null) as T)
}
