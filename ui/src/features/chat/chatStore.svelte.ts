import { t } from '../../i18n.svelte'
import type { ChatCheckpoint } from './checkpoint'
export * from './checkpoint'

export interface ToolEvent {
  server: string
  name: string
  approved: boolean
  args?: string
  resultPreview?: string
}

export interface MessageMetrics {
  tkPerSec?: number
  completionTokens?: number
  promptTokens?: number
  contextWindow?: number
  durationMs?: number
}

export interface ChatAttachment {
  id: string
  name: string
  mime: string
  dataUrl: string
  isGif?: boolean
}

export interface ChatEndpoint {
  provider: 'router' | 'sytra' | 'ollama' | 'lmstudio' | 'custom'
  baseUrl: string
  model: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  skillIds?: string[]
  skillsUsed?: string[]
  toolEvents?: ToolEvent[]
  attachments?: ChatAttachment[]
  metrics?: MessageMetrics
  compacted?: boolean
  proactiveLabel?: string
}

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  checkpoints?: ChatCheckpoint[]
  skillIds?: string[]
  webSearch?: boolean
  mcpPermission?: 'ask' | 'auto' | 'off'
  skillsEnabled?: boolean
  agentId?: string
  endpoint?: ChatEndpoint
  compactionCount?: number
  lastCompactedSummary?: string
  compactionDismissedAtPct?: number
}

const key = 'sytra-chat-conversations-v1'
const fresh = (): Conversation => ({
  id: crypto.randomUUID(),
  title: t('chat.newTitle'),
  messages: [],
  webSearch: false,
  mcpPermission: 'ask',
  skillsEnabled: true,
})

function restore(): Conversation[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || '[]')
    if (!Array.isArray(value)) return [fresh()]
    const valid = value.filter((item): item is Conversation =>
      typeof item?.id === 'string' && typeof item?.title === 'string' && Array.isArray(item?.messages) &&
      (item.skillIds === undefined || (Array.isArray(item.skillIds) && item.skillIds.every((id: unknown) => typeof id === 'string'))) &&
      item.messages.every((message: ChatMessage) => ['user', 'assistant'].includes(message?.role) && typeof message?.content === 'string' &&
        (message.skillIds === undefined || (Array.isArray(message.skillIds) && message.skillIds.every(id => typeof id === 'string')))))
    return valid.length ? valid : [fresh()]
  } catch { return [fresh()] }
}

const initial = restore()
export const chatStore = $state({ conversations: initial, currentId: initial[0].id })

export function saveConversations(): void {
  localStorage.setItem(key, JSON.stringify(chatStore.conversations))
}

export function newConversation(): void {
  const session = fresh()
  chatStore.conversations.push(session)
  chatStore.currentId = session.id
  saveConversations()
}

export function forkConversation(source: Conversation, index: number): void {
  const fork: Conversation = {
    id: crypto.randomUUID(), title: t('chat.branchTitle', { title: source.title }),
    messages: source.messages.slice(0, index + 1).map(message => ({
      ...message,
      ...(message.skillIds ? { skillIds: [...message.skillIds] } : {}),
      ...(message.skillsUsed ? { skillsUsed: [...message.skillsUsed] } : {}),
      ...(message.toolEvents ? { toolEvents: message.toolEvents.map(event => ({ ...event })) } : {}),
      ...(message.attachments ? { attachments: message.attachments.map(a => ({ ...a })) } : {}),
      ...(message.proactiveLabel ? { proactiveLabel: message.proactiveLabel } : {}),
      ...(message.metrics ? { metrics: { ...message.metrics } } : {}),
    })),
    skillIds: [...(source.messages.slice(0, index + 1).findLast(message => message.role === 'user')?.skillIds ?? [])],
    webSearch: source.webSearch ?? false,
    mcpPermission: source.mcpPermission ?? 'ask',
    skillsEnabled: source.skillsEnabled ?? true,
    agentId: source.agentId,
    endpoint: source.endpoint ? { ...source.endpoint } : undefined,
  }
  chatStore.conversations.push(fork)
  chatStore.currentId = fork.id
  saveConversations()
}

export function deleteConversation(id: string): void {
  chatStore.conversations = chatStore.conversations.filter(item => item.id !== id)
  if (!chatStore.conversations.length) chatStore.conversations.push(fresh())
  if (chatStore.currentId === id) chatStore.currentId = chatStore.conversations[0].id
  saveConversations()
}
