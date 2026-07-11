/**
 * Sytra Studio – shared TypeScript types mirroring Rust contracts
 */

export type BackendKind = 'cuda' | 'mps' | 'cpu' | 'auto'
export type TrainMode   = 'sft' | 'dpo' | 'orpo' | 'cpo'
export type AdapterType = 'lora' | 'qlora' | 'dora'
export type Schedule    = 'cosine' | 'linear' | 'constant'
export type MergeMethod = 'slerp' | 'ties' | 'dare_ties' | 'task_vector' | 'linear'
export type OpStatus    = 'running' | 'done' | 'error' | 'stopped'
export type Verdict     = 'green' | 'amber' | 'red'

export interface HfParams {
  repo_id:  string
  split:    string
  revision?: string
  jsonl_path?: string
}

export interface AdapterConfig {
  kind:           AdapterType
  rank:           number
  alpha:          number
  dropout:        number
  target_modules: string[]
  quant_bits?:    4 | 8 | null
}

export interface OptimConfig {
  learning_rate:         number
  schedule:              Schedule
  warmup_steps:          number
  weight_decay:          number
  grad_accumulation_steps: number
}

export interface TrainParams {
  max_steps?:  number
  epochs?:     number
  batch_size:  number
  max_seq_len: number
  save_every:  number
  packing:     boolean
}

export interface RunConfig {
  version:   number
  run_id?:   string
  train_mode: TrainMode
  model:     string
  backend:   { kind: BackendKind; judge_model?: string }
  data:      HfParams
  adapter:   AdapterConfig
  optim:     OptimConfig
  train:     TrainParams
  output:    { adapter_path: string; resume_from?: string }
}

export interface ModelEntry {
  name:       string
  family:     string
  param_count: number
  arch:       string
  dtype:      string
  vram_fp16_mb: number
  license:    string
  hf_id:      string
}

export interface MergeModel {
  model:  string
  weight: number
}

export interface MergeConfig {
  merge_method: MergeMethod
  models:       MergeModel[]
  base_model?:  string
  output:       { model_path: string }
}

export interface TelemetryLine {
  type:    'metric' | 'event' | 'log'
  epoch?:  number
  step?:   number
  loss?:   number
  lr?:     number
  grad_norm?: number
  progress?: number
  stage?:  string
  heartbeat?: boolean
  event?:  string
  payload?: Record<string, unknown>
  message?: string
  level?:  string
  line?:   string
  stream?: string
  ts?:     string | number
}

export interface OpRecord {
  op_id:        string
  kind:         'train' | 'merge'
  status:       OpStatus
  artifact_path: string
  config:       unknown
  provenance?:  string
}

export interface GuiderRecipe {
  model:      ModelEntry
  adapter:    AdapterConfig
  reason:     string
  fits_vram:  boolean
}

export interface CompatResult {
  verdict: Verdict
  reason:  string
  fingerprint?: string | null
}

/** Live telemetry chart data point */
export interface DataPoint { step: number; value: number }
