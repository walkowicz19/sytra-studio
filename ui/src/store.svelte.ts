import type { OpRecord, TelemetryLine, DataPoint, TrainMode, AdapterType, Schedule, MergeMethod, BackendKind } from './types'

// ─── Theme ───────────────────────────────────────────────────────────────────
export const themeStore = $state({ dark: false })

export interface CustomTheme {
  accentColor: string
  bgType: 'default' | 'solid' | 'gradient' | 'mesh' | 'aurora' | 'image' | 'gif'
  bgUrl: string
  bgOpacity: number
  effect: 'none' | 'glassmorphism' | 'glow' | 'scanlines' | 'cyberpunk' | 'frosted' | 'matrix' | 'holographic'
  glassBlur: number
  glowIntensity: number
  gradientDir: 'top' | 'top-right' | 'right' | 'bottom-right' | 'bottom' | 'radial'
  gradientSecond: string
  fontFamily: 'system' | 'inter' | 'jetbrains' | 'outfit' | 'geist'
  animSpeed: 'off' | 'slow' | 'normal' | 'fast'
  borderRadius: 'sharp' | 'default' | 'rounded'
  sidebarBlur: boolean
  cardShadow: 'none' | 'subtle' | 'elevated' | 'neon'
}

const defaultCustomTheme: CustomTheme = {
  accentColor: '#e03535',
  bgType: 'default',
  bgUrl: '',
  bgOpacity: 0.15,
  effect: 'none',
  glassBlur: 16,
  glowIntensity: 60,
  gradientDir: 'radial',
  gradientSecond: '#1e1b4b',
  fontFamily: 'system',
  animSpeed: 'normal',
  borderRadius: 'default',
  sidebarBlur: false,
  cardShadow: 'subtle',
}

function loadSavedCustomTheme(): CustomTheme {
  try {
    const raw = localStorage.getItem('sytra-custom-theme')
    if (raw) return { ...defaultCustomTheme, ...JSON.parse(raw) }
  } catch (e) {
    console.warn('Failed to parse saved custom theme', e)
  }
  return defaultCustomTheme
}

export const customThemeStore = $state<CustomTheme>(loadSavedCustomTheme())

export function applyCustomTheme() {
  const root = document.documentElement
  const hex = customThemeStore.accentColor || '#e03535'
  root.style.setProperty('--color-brand', hex)
  root.style.setProperty('--color-brand-hover', hex)
  root.style.setProperty('--color-brand-light', hexToRgba(hex, 0.15))
  root.style.setProperty('--color-brand-subtle', hexToRgba(hex, 0.08))

  // Background
  if (customThemeStore.bgType === 'image' || customThemeStore.bgType === 'gif') {
    root.style.setProperty('--custom-bg-image', customThemeStore.bgUrl ? `url("${customThemeStore.bgUrl}")` : 'none')
  } else if (customThemeStore.bgType === 'gradient') {
    const dir = customThemeStore.gradientDir === 'radial'
      ? `radial-gradient(ellipse at 50% 0%, ${hexToRgba(hex, 0.30)}, ${customThemeStore.gradientSecond} 60%, transparent 100%)`
      : `linear-gradient(to ${customThemeStore.gradientDir}, ${hexToRgba(hex, 0.28)}, ${customThemeStore.gradientSecond})`
    root.style.setProperty('--custom-bg-image', dir)
  } else if (customThemeStore.bgType === 'mesh') {
    root.style.setProperty('--custom-bg-image',
      `radial-gradient(at 40% 20%, ${hexToRgba(hex, 0.35)} 0%, transparent 50%),` +
      `radial-gradient(at 80% 0%, ${hexToRgba(customThemeStore.gradientSecond, 0.3)} 0%, transparent 50%),` +
      `radial-gradient(at 0% 50%, ${hexToRgba(hex, 0.15)} 0%, transparent 50%)`)
  } else if (customThemeStore.bgType === 'aurora') {
    root.style.setProperty('--custom-bg-image',
      `linear-gradient(125deg, ${hexToRgba(hex, 0.25)} 0%, ${hexToRgba(customThemeStore.gradientSecond, 0.2)} 40%, ${hexToRgba('#06b6d4', 0.15)} 80%)`)
  } else {
    root.style.setProperty('--custom-bg-image', 'none')
  }
  root.style.setProperty('--custom-bg-opacity', customThemeStore.bgOpacity.toString())

  // Glass / glow
  root.style.setProperty('--glass-blur', `${customThemeStore.glassBlur ?? 16}px`)
  const glow = (customThemeStore.glowIntensity ?? 60) / 100
  root.style.setProperty('--glow-strength', `${Math.round(glow * 24)}px`)
  root.style.setProperty('--glow-color', hexToRgba(hex, glow * 0.8))

  // Border radius
  const radii: Record<string, string> = { sharp: '2px', default: '6px', rounded: '12px' }
  root.style.setProperty('--radius-card', radii[customThemeStore.borderRadius ?? 'default'] ?? '6px')

  // Font family
  const fonts: Record<string, string> = {
    system: 'system-ui, sans-serif',
    inter: '"Inter", system-ui, sans-serif',
    jetbrains: '"JetBrains Mono", monospace',
    outfit: '"Outfit", system-ui, sans-serif',
    geist: '"Geist", system-ui, sans-serif',
  }
  root.style.setProperty('--font-ui', fonts[customThemeStore.fontFamily ?? 'system'] ?? 'system-ui, sans-serif')

  // Anim speed
  const speeds: Record<string, string> = { off: '0s', slow: '0.5s', normal: '0.2s', fast: '0.08s' }
  root.style.setProperty('--anim-duration', speeds[customThemeStore.animSpeed ?? 'normal'] ?? '0.2s')

  // Card shadow
  const shadows: Record<string, string> = {
    none: 'none',
    subtle: '0 1px 4px rgba(0,0,0,0.18)',
    elevated: '0 4px 24px rgba(0,0,0,0.35)',
    neon: `0 0 12px ${hexToRgba(hex, 0.4)}, 0 0 24px ${hexToRgba(hex, 0.2)}`,
  }
  root.style.setProperty('--card-shadow', shadows[customThemeStore.cardShadow ?? 'subtle'] ?? shadows.subtle)

  // Sidebar blur
  root.style.setProperty('--sidebar-blur', customThemeStore.sidebarBlur ? `blur(${customThemeStore.glassBlur ?? 16}px)` : 'none')

  root.dataset.effect = customThemeStore.effect
  localStorage.setItem('sytra-custom-theme', JSON.stringify(customThemeStore))
}

function hexToRgba(hex: string, alpha: number): string {
  try {
    let clean = hex.replace('#', '')
    if (clean.length === 3) clean = clean.split('').map(c => c + c).join('')
    let num = parseInt(clean, 16)
    let r = (num >> 16) & 255
    let g = (num >> 8) & 255
    let b = num & 255
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  } catch {
    return `rgba(224, 53, 53, ${alpha})`
  }
}

export function toggleTheme() {
  themeStore.dark = !themeStore.dark
  document.documentElement.dataset.theme = themeStore.dark ? 'dark' : 'light'
  localStorage.setItem('sytra-theme', themeStore.dark ? 'dark' : 'light')
}

export function initTheme() {
  // Dark is the brand default; light is an explicit opt-in.
  const saved = localStorage.getItem('sytra-theme')
  const dark = saved !== 'light'
  themeStore.dark = dark
  document.documentElement.dataset.theme = dark ? 'dark' : 'light'
  applyCustomTheme()
}

// ─── Active tab ───────────────────────────────────────────────────────────────
export type Tab = 'models' | 'train' | 'merge' | 'runs' | 'guider' | 'help' | 'settings'
export const tabStore = $state({ active: 'models' as Tab })
export function setTab(t: Tab) { tabStore.active = t }

// ─── UI mode ──────────────────────────────────────────────────────────────────
// Simple is the default: a guided, no-jargon flow for non-technical users.
// Advanced exposes the full run.yaml / merge.yaml surface.
export const uiMode = $state({ advanced: localStorage.getItem('sytra-ui-mode') === 'advanced' })

export function toggleUiMode() {
  uiMode.advanced = !uiMode.advanced
  localStorage.setItem('sytra-ui-mode', uiMode.advanced ? 'advanced' : 'simple')
  // Guider only exists in advanced mode; land somewhere valid.
  if (!uiMode.advanced && tabStore.active === 'guider') tabStore.active = 'train'
}

// ─── Run state ────────────────────────────────────────────────────────────────
export interface RunState {
  opId:      string | null
  kind:      'train' | 'merge' | 'publish' | null
  status:    'idle' | 'running' | 'done' | 'error' | 'stopped'
  startedAt: number | null
  progress:  number
  step:      number
  epoch:     number
  totalSteps: number
  stage:     string
  loss:      DataPoint[]
  lr:        DataPoint[]
  gradNorm:  DataPoint[]
  logLines:  TelemetryLine[]
}

export const run = $state<RunState>({
  opId: null, kind: null, status: 'idle', startedAt: null,
  progress: 0, step: 0, epoch: 0, totalSteps: 0, stage: '',
  loss: [], lr: [], gradNorm: [], logLines: [],
})

export function resetRun() {
  run.opId = null; run.kind = null; run.status = 'idle'; run.startedAt = null
  run.progress = 0; run.step = 0; run.epoch = 0; run.totalSteps = 0; run.stage = ''
  run.loss = []; run.lr = []; run.gradNorm = []; run.logLines = []
  if (unlistenTelemetry) {
    unlistenTelemetry()
    unlistenTelemetry = null
  }
}

export function applyTelemetry(line: TelemetryLine) {
  run.logLines = [...run.logLines.slice(-500), line]
  if (line.type === 'metric') {
    if (line.step     !== undefined) run.step     = line.step
    if (line.epoch    !== undefined) run.epoch    = line.epoch
    if (line.stage    !== undefined) run.stage    = line.stage
    if (line.progress !== undefined && line.progress !== null) {
      run.progress = line.progress
    } else if (line.step !== undefined && run.totalSteps > 0) {
      // Train metrics may carry only a step — derive the fraction from
      // total_steps announced in the starting event.
      run.progress = Math.min(line.step / run.totalSteps, 1)
    }
    if (line.loss     !== undefined && line.loss     !== null) run.loss     = [...run.loss,     { step: line.step ?? run.step, value: line.loss }]
    if (line.lr       !== undefined && line.lr       !== null) run.lr       = [...run.lr,       { step: line.step ?? run.step, value: line.lr }]
    if (line.grad_norm!== undefined && line.grad_norm!== null) run.gradNorm = [...run.gradNorm, { step: line.step ?? run.step, value: line.grad_norm }]
  }
  if (line.type === 'event') {
    if (line.event === 'starting') {
      const total = (line.payload as Record<string, unknown> | undefined)?.total_steps
      if (typeof total === 'number' && total > 0) run.totalSteps = total
    }
    if (line.event === 'stage') {
      const stage = (line.payload as Record<string, unknown> | undefined)?.stage
      if (typeof stage === 'string') run.stage = stage
    }
    if (line.event === 'done' && run.status !== 'error' && run.status !== 'stopped') {
      run.status = 'done'
      run.progress = 1
    }
    if (line.event === 'stopped') {
      run.status = 'stopped'
    }
    if (line.event === 'error') {
      run.status = 'error'
    }
  }
}

let unlistenTelemetry: (() => void) | null = null

export async function watchTelemetry(opId: string) {
  if (unlistenTelemetry) {
    unlistenTelemetry()
    unlistenTelemetry = null
  }
  const isTauri = typeof (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ !== 'undefined'
  if (isTauri) {
    try {
      const { listen } = await import('@tauri-apps/api/event')
      const eventName = `telemetry:${opId}`
      const unlisten = await listen<string>(eventName, (event) => {
        try {
          const line: TelemetryLine = JSON.parse(event.payload)
          applyTelemetry(line)
        } catch (err) {
          console.error("Failed to parse telemetry line:", err)
        }
      })
      unlistenTelemetry = unlisten
    } catch (e) {
      console.error("Failed to setup Tauri telemetry listener:", e)
    }
  }
}

// ─── Run history ──────────────────────────────────────────────────────────────
export const historyStore = $state<{ records: OpRecord[] }>({ records: [] })
export function setRunHistory(records: OpRecord[]) { historyStore.records = records }

// ─── Toast ────────────────────────────────────────────────────────────────────
export interface Toast { id: number; kind: 'success' | 'error' | 'info'; message: string }
export const toastStore = $state<{ items: Toast[] }>({ items: [] })
let _toastId = 0
export function pushToast(kind: Toast['kind'], message: string, ms = 4000) {
  const id = ++_toastId
  toastStore.items = [...toastStore.items, { id, kind, message }]
  setTimeout(() => { toastStore.items = toastStore.items.filter(t => t.id !== id) }, ms)
}

// ─── Hardware ─────────────────────────────────────────────────────────────────
export interface HwInfo { backend: string; vram_mb: number | null; ram_mb: number | null }
export const hwStore = $state<{ info: HwInfo | null }>({ info: null })
export function setHwInfo(info: HwInfo) { hwStore.info = info }

export interface DatasetItem {
  id: string;
  source: 'hf' | 'local' | 'synthetic' | 'klayer';
  hfRepoId: string;
  hfSplit: string;
  hfRevision: string;
  hfConfig: string; // new!
  localPath: string;
  localFormat: 'jsonl' | 'csv' | 'parquet';
  localPromptCol: string;
  localCompletionCol: string;
  synthGenerator: string;
  synthJudge: string;
  synthMode: 'prompts' | 'sft' | 'dpo';
  synthCount: number;
  synthTopic: string;
  klayerDomain: string;
}

export function createDefaultDatasetItem(source: 'hf' | 'local' | 'synthetic' | 'klayer' = 'hf'): DatasetItem {
  return {
    id: Math.random().toString(36).substring(2, 11),
    source,
    hfRepoId: 'org/my-dataset',
    hfSplit: 'train',
    hfRevision: '',
    hfConfig: '', // new!
    localPath: '',
    localFormat: 'jsonl',
    localPromptCol: 'prompt',
    localCompletionCol: 'completion',
    synthGenerator: 'mistralai/Mistral-7B-v0.1',
    synthJudge: 'meta-llama/Llama-2-7b-hf',
    synthMode: 'sft',
    synthCount: 100,
    synthTopic: 'machine learning',
    klayerDomain: '',
  };
}

// ─── Forms Shared States (Phase 4) ─────────────────────────────────────────────
export const trainFormStore = $state({
  model: 'mistralai/Mistral-7B-v0.1',
  trainMode: 'sft' as TrainMode,
  backend: 'auto' as BackendKind | 'auto',
  dataSplitSource: 'hf' as 'hf' | 'local' | 'synthetic' | 'klayer',
  
  // Multiple datasets mix support
  useMultiple: false,
  datasets: [] as DatasetItem[],
  
  // HF Params
  hfRepoId: 'org/my-dataset',
  hfSplit: 'train',
  hfRevision: '',
  hfConfig: '', // new!

  // Local Params
  localPath: '',
  localFormat: 'jsonl' as 'jsonl' | 'csv' | 'parquet',
  localPromptCol: 'prompt',
  localCompletionCol: 'completion',

  // Synthetic Params
  synthGenerator: 'mistralai/Mistral-7B-v0.1',
  synthJudge: 'meta-llama/Llama-2-7b-hf',
  synthMode: 'sft' as 'prompts' | 'sft' | 'dpo',
  synthCount: 100,
  synthTopic: 'machine learning',

  // Klayer Params
  klayerDomain: '',

  // Adapter params
  adapterKind: 'lora' as AdapterType,
  rank: 16,
  alpha: 32,
  dropout: 0.05,
  quantBits: 0 as 0 | 4 | 8,
  
  // Optim & Train params
  lr: 2e-4,
  schedule: 'cosine' as Schedule,
  warmupSteps: 20,
  weightDecay: 0.0,
  gradAcc: 8,
  maxSteps: 200,
  batchSize: 2,
  maxSeqLen: 2048,
  saveEvery: 50,
  packing: false,
  outputPath: 'runs/adapter',
})

export const mergeFormStore = $state({
  mergeMethod: 'slerp' as MergeMethod,
  models: [
    { model: 'mistralai/Mistral-7B-v0.1', weight: 0.5 },
    { model: 'meta-llama/Llama-2-7b-hf', weight: 0.5 },
  ] as { model: string; weight: number }[],
  baseModel: 'mistralai/Mistral-7B-v0.1',
  outputPath: 'runs/merged',
})
