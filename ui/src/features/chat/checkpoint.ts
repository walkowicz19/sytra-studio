export interface WorkspaceFileDelta {
  agentId: string
  path: string
  existed: boolean
  previousContent?: string
}

export interface ChatCheckpoint {
  id: string
  createdAt: number
  messageCount: number
  promptText: string
  lastCompactedSummary?: string
  compactionCount?: number
  workspaceDeltas?: WorkspaceFileDelta[]
}

export interface CheckpointSessionLike {
  messages: { role: string; content: string }[]
  checkpoints?: ChatCheckpoint[]
  lastCompactedSummary?: string
  compactionCount?: number
  compactionDismissedAtPct?: number
}

const MAX_CHECKPOINTS = 20

/**
 * Record an ultra-lightweight turn checkpoint before sending a prompt or mutating files.
 * Kept purely in session state so models never see checkpoint tokens or tool overhead.
 */
export function pushCheckpoint(source: CheckpointSessionLike, promptText: string): ChatCheckpoint {
  const checkpoint: ChatCheckpoint = {
    id: crypto.randomUUID(),
    createdAt: Date.now(),
    messageCount: source.messages.length,
    promptText,
    ...(source.lastCompactedSummary !== undefined ? { lastCompactedSummary: source.lastCompactedSummary } : {}),
    ...(source.compactionCount !== undefined ? { compactionCount: source.compactionCount } : {}),
  }
  const next = [...(source.checkpoints ?? []), checkpoint]
  source.checkpoints = next.length > MAX_CHECKPOINTS ? next.slice(-MAX_CHECKPOINTS) : next
  return checkpoint
}

/** Record earliest pre-edit sandbox file state for the active turn checkpoint. */
export function recordWorkspaceDelta(source: CheckpointSessionLike, delta: WorkspaceFileDelta): void {
  const latest = source.checkpoints?.at(-1)
  if (!latest) return
  const deltas = latest.workspaceDeltas ?? []
  if (deltas.some((d) => d.agentId === delta.agentId && d.path === delta.path)) return
  latest.workspaceDeltas = [...deltas, delta]
}

/** Return the message index carrying the single "Undo changes" icon for the latest checkpoint. */
export function lastCheckpointTargetIndex(source?: CheckpointSessionLike): number {
  if (!source?.messages.length) return -1
  const cps = source.checkpoints
  if (cps?.length) {
    const latest = cps[cps.length - 1]
    if (latest.messageCount < source.messages.length) {
      return latest.messageCount
    }
  }
  for (let i = source.messages.length - 1; i >= 0; i--) {
    if (source.messages[i].role === 'user') return i
  }
  return source.messages.length - 1
}

/** Roll conversation state back to the latest checkpoint (or prior user turn as safe fallback). */
export function rollbackLastCheckpoint(
  source: CheckpointSessionLike,
): { promptText: string; workspaceDeltas: WorkspaceFileDelta[] } | null {
  if (!source.messages.length) return null
  const cps = source.checkpoints
  while (cps?.length) {
    const cp = cps.pop()!
    if (cp.messageCount <= source.messages.length) {
      source.messages = source.messages.slice(0, cp.messageCount)
      source.lastCompactedSummary = cp.lastCompactedSummary
      source.compactionCount = cp.compactionCount
      source.compactionDismissedAtPct = undefined
      return {
        promptText: cp.promptText,
        workspaceDeltas: cp.workspaceDeltas ?? [],
      }
    }
  }
  const lastUserIdx = source.messages.findLastIndex((m) => m.role === 'user')
  if (lastUserIdx >= 0) {
    const promptText = source.messages[lastUserIdx].content
    source.messages = source.messages.slice(0, lastUserIdx)
    source.compactionDismissedAtPct = undefined
    return { promptText, workspaceDeltas: [] }
  }
  source.messages = source.messages.slice(0, -1)
  return { promptText: '', workspaceDeltas: [] }
}
