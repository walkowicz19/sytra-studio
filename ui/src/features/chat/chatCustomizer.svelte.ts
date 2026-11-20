export type ChatLayoutPreset = 'standard' | 'messenger' | 'left' | 'compact'
export type BubbleAlign = 'left' | 'center' | 'right'
export type ComposerDock = 'bottom' | 'top'

export interface ChatAppearance {
  schemaVersion?: number
  layoutPreset: ChatLayoutPreset
  userAlign: BubbleAlign
  assistantAlign: BubbleAlign
  composerDock: ComposerDock
  maxWidthPx: number
  bubbleRadiusPx: number | null
  fontSizePx: number
  wallpaperUrl: string
  wallpaperOpacity: number
  wallpaperBlur: number
  accentHex: string
  userAvatarUrl: string
  assistantAvatarUrl: string
  showFloatingPet: boolean
  petSpecies: 'pochi' | 'kitsune' | 'mochi' | 'owl'
}

const STORAGE_KEY = 'sytra-chat-appearance-v1'
const SCHEMA_VERSION = 3
const HEX_RE = /^#[0-9a-fA-F]{6}$/

export const DEFAULT_CHAT_APPEARANCE: ChatAppearance = {
  schemaVersion: SCHEMA_VERSION,
  layoutPreset: 'standard',
  userAlign: 'right',
  assistantAlign: 'left',
  composerDock: 'bottom',
  maxWidthPx: 860,
  bubbleRadiusPx: null,
  fontSizePx: 15,
  wallpaperUrl: '',
  wallpaperOpacity: 0.24,
  wallpaperBlur: 0,
  accentHex: '',
  userAvatarUrl: '',
  assistantAvatarUrl: '',
  showFloatingPet: true,
  petSpecies: 'pochi',
}

export function isSafeMediaUrl(raw: string): boolean {
  const trimmed = raw.trim()
  if (!trimmed) return true
  if (trimmed.startsWith('data:image/') || trimmed.startsWith('blob:')) return true
  try {
    const u = new URL(trimmed)
    return u.protocol === 'https:' || u.protocol === 'http:'
  } catch {
    return false
  }
}

export function readImageFileAsDataUrl(file: File, maxBytes = 2_600_000): Promise<{ dataUrl: string; isGif: boolean }> {
  return new Promise((resolve, reject) => {
    if (!file.type.startsWith('image/')) {
      reject(new Error('Only image and GIF files are supported.'))
      return
    }
    if (file.size > maxBytes) {
      reject(new Error(`Image/GIF exceeds ${(maxBytes / 1_000_000).toFixed(1)} MB limit.`))
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      const result = typeof reader.result === 'string' ? reader.result : ''
      if (!result.startsWith('data:image/')) {
        reject(new Error('Invalid image payload.'))
        return
      }
      resolve({ dataUrl: result, isGif: file.type === 'image/gif' || file.name.toLowerCase().endsWith('.gif') })
    }
    reader.onerror = () => reject(new Error('Could not read selected file.'))
    reader.readAsDataURL(file)
  })
}

function restoreAppearance(): ChatAppearance {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
    if (!raw || typeof raw !== 'object') return { ...DEFAULT_CHAT_APPEARANCE }
    const obj = raw as Partial<ChatAppearance>
    const preset: ChatLayoutPreset =
      obj.layoutPreset === 'messenger' || obj.layoutPreset === 'left' || obj.layoutPreset === 'compact'
        ? obj.layoutPreset
        : 'standard'
    const needsV2Migration = !obj.schemaVersion && preset === 'standard' && obj.userAlign === 'center' && obj.assistantAlign === 'center'
    const usesLegacyDefaults = (obj.schemaVersion ?? 0) < SCHEMA_VERSION
    return {
      schemaVersion: SCHEMA_VERSION,
      layoutPreset: preset,
      userAlign: needsV2Migration
        ? 'right'
        : obj.userAlign === 'left' || obj.userAlign === 'right' || obj.userAlign === 'center'
          ? obj.userAlign
          : 'right',
      assistantAlign: needsV2Migration
        ? 'left'
        : obj.assistantAlign === 'left' || obj.assistantAlign === 'right' || obj.assistantAlign === 'center'
          ? obj.assistantAlign
          : 'left',
      composerDock: obj.composerDock === 'top' ? 'top' : 'bottom',
      maxWidthPx: typeof obj.maxWidthPx === 'number' ? Math.min(1140, Math.max(520, obj.maxWidthPx)) : 860,
      bubbleRadiusPx: typeof obj.bubbleRadiusPx === 'number' && !(usesLegacyDefaults && preset === 'standard' && obj.bubbleRadiusPx === 16)
        ? Math.min(22, Math.max(0, obj.bubbleRadiusPx))
        : null,
      fontSizePx: typeof obj.fontSizePx === 'number' ? Math.min(20, Math.max(12, obj.fontSizePx)) : 15,
      wallpaperUrl: typeof obj.wallpaperUrl === 'string' && isSafeMediaUrl(obj.wallpaperUrl) ? obj.wallpaperUrl : '',
      wallpaperOpacity: typeof obj.wallpaperOpacity === 'number' ? Math.min(0.85, Math.max(0.05, obj.wallpaperOpacity)) : 0.24,
      wallpaperBlur: typeof obj.wallpaperBlur === 'number' ? Math.min(16, Math.max(0, obj.wallpaperBlur)) : 0,
      accentHex: typeof obj.accentHex === 'string' && HEX_RE.test(obj.accentHex) && !(usesLegacyDefaults && obj.accentHex === '#e03535')
        ? obj.accentHex
        : '',
      userAvatarUrl: typeof obj.userAvatarUrl === 'string' && isSafeMediaUrl(obj.userAvatarUrl) ? obj.userAvatarUrl : '',
      assistantAvatarUrl: typeof obj.assistantAvatarUrl === 'string' && isSafeMediaUrl(obj.assistantAvatarUrl) ? obj.assistantAvatarUrl : '',
      showFloatingPet: obj.showFloatingPet !== false,
      petSpecies:
        obj.petSpecies === 'kitsune' || obj.petSpecies === 'mochi' || obj.petSpecies === 'owl'
          ? obj.petSpecies
          : 'pochi',
    }
  } catch {
    return { ...DEFAULT_CHAT_APPEARANCE }
  }
}

export const chatAppearance = $state<ChatAppearance>(restoreAppearance())

export function saveChatAppearance(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chatAppearance))
  } catch {
    // Ignore storage quota overflow gracefully.
  }
}

export function applyLayoutPreset(preset: ChatLayoutPreset): void {
  chatAppearance.layoutPreset = preset
  if (preset === 'messenger') {
    chatAppearance.userAlign = 'right'
    chatAppearance.assistantAlign = 'left'
    chatAppearance.bubbleRadiusPx = 14
    chatAppearance.maxWidthPx = 760
  } else if (preset === 'left') {
    chatAppearance.userAlign = 'left'
    chatAppearance.assistantAlign = 'left'
    chatAppearance.bubbleRadiusPx = 8
    chatAppearance.maxWidthPx = 880
  } else if (preset === 'compact') {
    chatAppearance.userAlign = 'center'
    chatAppearance.assistantAlign = 'center'
    chatAppearance.bubbleRadiusPx = 4
    chatAppearance.maxWidthPx = 980
    chatAppearance.fontSizePx = 13
  } else {
    chatAppearance.userAlign = 'right'
    chatAppearance.assistantAlign = 'left'
    chatAppearance.bubbleRadiusPx = null
    chatAppearance.maxWidthPx = 860
    chatAppearance.fontSizePx = 15
  }
  chatAppearance.schemaVersion = SCHEMA_VERSION
  saveChatAppearance()
}

export function resetChatAppearance(): void {
  Object.assign(chatAppearance, DEFAULT_CHAT_APPEARANCE)
  saveChatAppearance()
}
