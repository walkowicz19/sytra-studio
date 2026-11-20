export interface ToolCall { id: string; type: 'function'; function: { name: string; arguments: string } }

export interface StreamMetrics {
  tkPerSec?: number
  completionTokens?: number
  promptTokens?: number
  durationMs: number
}

export interface StreamCallbacks {
  onText: (text: string) => void
  onReasoning?: (text: string) => void
  onMetrics?: (metrics: StreamMetrics) => void
  firstTokenTimeoutMs?: number
  idleTimeoutMs?: number
}

interface TagRule {
  open: string
  close: string[]
}

const THINK_TAGS: TagRule[] = [
  { open: '<think>', close: ['</think>'] },
  { open: '<thought>', close: ['</thought>'] },
  { open: '<reasoning>', close: ['</reasoning>'] },
  {
    open: '<|channel|>analysis<|message|>',
    close: ['<|end|><|start|>assistant<|channel|>final<|message|>', '<|channel|>final<|message|>', '<|end|>'],
  },
]

function trailingPrefixLen(text: string, candidates: string[]): number {
  let maxMatch = 0
  for (const tag of candidates) {
    const limit = Math.min(text.length, tag.length - 1)
    for (let len = limit; len >= 1; len--) {
      if (tag.startsWith(text.slice(text.length - len))) {
        if (len > maxMatch) maxMatch = len
        break
      }
    }
  }
  return maxMatch
}

function createThinkSplitter(onText: (t: string) => void, onReasoning?: (t: string) => void) {
  let pending = ''
  let activeRule: TagRule | null = null
  const openTags = THINK_TAGS.map(r => r.open)

  function push(chunk: string) {
    pending += chunk
    while (pending.length) {
      if (!activeRule) {
        let bestIdx = -1
        let matched: TagRule | null = null
        for (const rule of THINK_TAGS) {
          const idx = pending.indexOf(rule.open)
          if (idx >= 0 && (bestIdx === -1 || idx < bestIdx)) {
            bestIdx = idx
            matched = rule
          }
        }
        if (matched && bestIdx >= 0) {
          if (bestIdx > 0) onText(pending.slice(0, bestIdx))
          pending = pending.slice(bestIdx + matched.open.length)
          activeRule = matched
        } else {
          const keep = trailingPrefixLen(pending, openTags)
          const safeLen = pending.length - keep
          if (safeLen > 0) {
            onText(pending.slice(0, safeLen))
            pending = pending.slice(safeLen)
          }
          break
        }
      } else {
        let bestIdx = -1
        let closeLen = 0
        for (const closeTag of activeRule.close) {
          const idx = pending.indexOf(closeTag)
          if (idx >= 0 && (bestIdx === -1 || idx < bestIdx)) {
            bestIdx = idx
            closeLen = closeTag.length
          }
        }
        if (bestIdx >= 0) {
          if (bestIdx > 0) onReasoning?.(pending.slice(0, bestIdx))
          pending = pending.slice(bestIdx + closeLen)
          activeRule = null
        } else {
          const keep = trailingPrefixLen(pending, activeRule.close)
          const safeLen = pending.length - keep
          if (safeLen > 0) {
            onReasoning?.(pending.slice(0, safeLen))
            pending = pending.slice(safeLen)
          }
          break
        }
      }
    }
  }

  function flush() {
    if (!pending) return
    if (activeRule) onReasoning?.(pending)
    else onText(pending)
    pending = ''
  }

  return { push, flush }
}

/** Decode OpenAI-compatible SSE, preserving UTF-8, reasoning blocks, and live timing telemetry. */
export async function readChatStream(
  body: ReadableStream<Uint8Array>,
  target: ((text: string) => void) | StreamCallbacks,
): Promise<ToolCall[]> {
  const callbacks: StreamCallbacks = typeof target === 'function' ? { onText: target } : target
  const firstTokenTimeoutMs = callbacks.firstTokenTimeoutMs ?? 360_000
  const idleTimeoutMs = callbacks.idleTimeoutMs ?? 120_000
  const splitter = createThinkSplitter(callbacks.onText, callbacks.onReasoning)
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let lines: string[] = []
  let finished = false
  let ended = false
  let finishReason: string | null = null
  const toolCalls = new Map<number, ToolCall>()

  const startedAt = performance.now()
  let firstTokenAt: number | null = null
  let streamedChars = 0
  let approxChunks = 0
  let serverTkPerSec: number | undefined
  let completionTokens: number | undefined
  let promptTokens: number | undefined

  function emitMetrics() {
    if (!callbacks.onMetrics) return
    const now = performance.now()
    const durationMs = Math.max(1, now - startedAt)
    const decodeSec = firstTokenAt != null ? Math.max((now - firstTokenAt) / 1000, 0.05) : durationMs / 1000
    const estTokens = completionTokens ?? Math.max(approxChunks, Math.round(streamedChars / 3.8))
    const liveTk = serverTkPerSec ?? (estTokens > 0 ? estTokens / decodeSec : undefined)
    callbacks.onMetrics({
      tkPerSec: liveTk !== undefined ? Math.round(liveTk * 10) / 10 : undefined,
      completionTokens: estTokens > 0 ? estTokens : undefined,
      promptTokens,
      durationMs: Math.round(durationMs),
    })
  }

  function noteOutput(chunkText: string) {
    if (!chunkText) return
    if (firstTokenAt === null) firstTokenAt = performance.now()
    streamedChars += chunkText.length
    approxChunks += 1
  }

  function dispatch() {
    if (!lines.length) return
    const data = lines.join('\n')
    lines = []
    if (data === '[DONE]') { finished = true; return }
    const event = JSON.parse(data)
    if (event.error) throw new Error(String(event.error.message || event.error))

    if (event.usage && typeof event.usage === 'object') {
      if (typeof event.usage.prompt_tokens === 'number') promptTokens = event.usage.prompt_tokens
      if (typeof event.usage.completion_tokens === 'number') completionTokens = event.usage.completion_tokens
    }
    if (event.timings && typeof event.timings === 'object') {
      if (typeof event.timings.predicted_per_second === 'number' && Number.isFinite(event.timings.predicted_per_second)) {
        serverTkPerSec = event.timings.predicted_per_second
      }
      if (typeof event.timings.predicted_n === 'number') completionTokens = event.timings.predicted_n
      if (typeof event.timings.prompt_n === 'number') promptTokens = event.timings.prompt_n
    }

    const choice = event.choices?.[0]
    if (Array.isArray(choice?.delta?.tool_calls)) {
      for (const fragment of choice.delta.tool_calls) {
        const index = fragment?.index
        if (!Number.isInteger(index) || index < 0 || index >= 8) throw new Error('Invalid tool call index')
        const call = toolCalls.get(index) ?? { id: '', type: 'function' as const, function: { name: '', arguments: '' } }
        if (typeof fragment.id === 'string') call.id += fragment.id
        if (typeof fragment.function?.name === 'string') call.function.name += fragment.function.name
        if (typeof fragment.function?.arguments === 'string') call.function.arguments += fragment.function.arguments
        if (call.function.arguments.length > 256_000 || call.function.name.length > 256 || call.id.length > 256) throw new Error('Tool call is too large')
        toolCalls.set(index, call)
      }
    }

    const reasoning = choice?.delta?.reasoning_content ?? choice?.delta?.reasoning
    if (typeof reasoning === 'string' && reasoning.length) {
      noteOutput(reasoning)
      callbacks.onReasoning?.(reasoning)
      emitMetrics()
    }
    const content = choice?.delta?.content
    if (typeof content === 'string' && content.length) {
      noteOutput(content)
      splitter.push(content)
      emitMetrics()
    }
    if (choice?.finish_reason) {
      finished = true
      finishReason = choice.finish_reason
    }
  }

  async function readWithTimeout() {
    const activeTimeoutMs = firstTokenAt === null ? firstTokenTimeoutMs : idleTimeoutMs
    if (activeTimeoutMs <= 0) return reader.read()
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      return await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error(
            firstTokenAt === null
              ? 'Model took too long to load or start responding (first-token timeout).'
              : 'Model stopped responding (stream timed out).',
          )), activeTimeoutMs)
        }),
      ])
    } finally {
      if (timer !== undefined) clearTimeout(timer)
    }
  }

  try {
    while (true) {
      const { value, done } = await readWithTimeout()
      buffer += decoder.decode(value, { stream: !done })
      if (buffer.length > 16 * 1024 * 1024) throw new Error('The server sent an oversized event.')
      let newline: number
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, '')
        buffer = buffer.slice(newline + 1)
        if (!line) dispatch()
        else if (line.startsWith('data:')) lines.push(line.slice(5).replace(/^ /, ''))
      }
      if (finished && !done) break
      if (done) {
        ended = true
        if (buffer.startsWith('data:')) lines.push(buffer.slice(5).replace(/^ /, '').replace(/\r$/, ''))
        dispatch()
        break
      }
    }
    splitter.flush()
    emitMetrics()
    if (!finished) throw new Error('The server closed the response before completion.')
    const calls = [...toolCalls.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call)
    if (finishReason === 'tool_calls' && !calls.length) throw new Error('The server returned an incomplete tool call.')
    if (calls.some(call => !call.id || !call.function.name)) throw new Error('The server returned an incomplete tool call.')
    return calls
  } finally {
    if (!ended) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
