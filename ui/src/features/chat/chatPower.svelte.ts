import { api } from '../../api'
import type { ChatEndpoint } from './chatStore.svelte'

const POWER_KEY = 'sytra-chat-power-v1'

export interface ChatPowerPreferences {
  releaseOnLeaveChat: boolean
  idleMinutes: 0 | 2 | 5 | 15
}

function loadPrefs(): ChatPowerPreferences {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(POWER_KEY) || 'null')
    if (raw && typeof raw === 'object') {
      const obj = raw as Partial<ChatPowerPreferences>
      const mins = obj.idleMinutes === 0 || obj.idleMinutes === 2 || obj.idleMinutes === 15 ? obj.idleMinutes : 5
      return {
        releaseOnLeaveChat: obj.releaseOnLeaveChat !== false,
        idleMinutes: mins,
      }
    }
  } catch {
    // Use defaults
  }
  return { releaseOnLeaveChat: true, idleMinutes: 5 }
}

const initialPrefs = loadPrefs()

export const chatPower = $state({
  releaseOnLeaveChat: initialPrefs.releaseOnLeaveChat,
  idleMinutes: initialPrefs.idleMinutes as 0 | 2 | 5 | 15,
  activeTarget: null as ChatEndpoint | null,
  isBusy: false,
  sleeping: false,
  lastActivityAt: Date.now(),
  statusNote: '',
})

export function savePowerPrefs(): void {
  try {
    localStorage.setItem(POWER_KEY, JSON.stringify({
      releaseOnLeaveChat: chatPower.releaseOnLeaveChat,
      idleMinutes: chatPower.idleMinutes,
    }))
  } catch {
    // Ignore storage error
  }
}

export function noteChatActivity(target?: ChatEndpoint | null): void {
  chatPower.lastActivityAt = Date.now()
  if (target && target.provider !== 'router' && target.model) {
    chatPower.activeTarget = { ...target }
    chatPower.sleeping = false
    chatPower.statusNote = ''
  }
}

export function setChatBusyState(busy: boolean): void {
  chatPower.isBusy = busy
  if (!busy) chatPower.lastActivityAt = Date.now()
}

export async function releaseChatGpu(reason: 'manual' | 'tab-leave' | 'idle'): Promise<string> {
  if (chatPower.isBusy) return ''
  const target = chatPower.activeTarget
  if (!target && reason !== 'manual') return ''
  try {
    const prov = target?.provider ?? 'sytra'
    const base = target?.baseUrl ?? 'http://127.0.0.1:8080'
    const model = target?.model ?? ''
    const msg = await api.unloadChatModel(prov, base, model)
    chatPower.sleeping = true
    chatPower.statusNote =
      reason === 'manual'
        ? msg
        : reason === 'tab-leave'
          ? `Released GPU outside Chat (${msg})`
          : `Idle sleep: ${msg}`
    return chatPower.statusNote
  } catch (err) {
    const fallback = String(err)
    chatPower.statusNote = fallback
    return fallback
  }
}
