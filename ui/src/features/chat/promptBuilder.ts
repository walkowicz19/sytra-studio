import type { IndexedTool } from './mcp'
import type { ToolCall } from './stream'

const MAX_SKILL_PROMPT_CHARS = 4200

function tryParseArgString(raw: string): unknown {
  const trimmed = raw.trim()
  if (!trimmed) return {}
  try {
    return JSON.parse(trimmed)
  } catch {
    try {
      return JSON.parse(trimmed.replace(/,\s*([}\]])/g, '$1'))
    } catch {
      return null
    }
  }
}

function compactSchemaParams(schema: Record<string, unknown> | undefined): string {
  const props = schema?.properties
  if (!props || typeof props !== 'object') return ''
  const req = new Set(Array.isArray(schema?.required) ? schema.required.map(String) : [])
  const keys = Object.keys(props as Record<string, unknown>).slice(0, 5)
  return keys
    .map((k) => {
      const p = (props as Record<string, Record<string, unknown>>)[k]
      const t = typeof p?.type === 'string' ? p.type : 'any'
      return `${k}${req.has(k) ? '' : '?'}:${t}`
    })
    .join(', ')
}

export interface PromptMemorySnippet {
  kind: string
  session_title: string
  summary: string
}

/**
 * Extremely lightweight system prompt tuned so small or older models without
 * native OpenAI `tool_calls` support can still invoke tools reliably without
 * inflating prefill latency.
 */
export function buildLightweightSystemPrompt(opts: {
  skillsPrompt?: string
  skillsCatalog?: string[]
  mcpServers?: string[]
  mcpPermission?: 'ask' | 'auto' | 'off'
  tools: { exposedName: string; item: IndexedTool }[]
  agentPrompt?: string
  computerMode?: 'sandbox' | 'pc_ask' | 'pc_auto'
  recalledMemories?: PromptMemorySnippet[]
}): string {
  const sections: string[] = [
    "Always reply in the same language as the user's latest message unless asked to translate.",
    'Be concise, accurate, and fast. Keep any <think> reasoning brief (1-2 short sentences) unless solving a multi-step task.',
    'For charts or graphs, output a Markdown table or ```chart {"type":"bar"|"line"|"area"|"pie","title":"...","labels":[...],"values":[...],"unit":""} ``` (the UI draws interactive SVGs; never output plain JSON as code or claim you cannot draw charts). Use ```kpi {"items":[{"label":"...","value":"...","delta":"..."}]} for highlight metrics.',
  ]

  const skillList = (opts.skillsCatalog ?? []).filter(Boolean)
  const mcpList = (opts.mcpServers ?? []).filter(Boolean)
  const skillsLine = skillList.length
    ? `Skills (SKILL.md instruction files invoked via @name or the Skills button; ${skillList.length} installed: ${skillList.slice(0, 10).join(', ')})`
    : 'Skills (reusable SKILL.md instruction files invoked via @name or created with create_skill)'
  const mcpState = opts.mcpPermission === 'off'
    ? 'disabled in toolbar'
    : mcpList.length
      ? `${mcpList.length} connected (${mcpList.join(', ')})`
      : '0 connected right now (configurable in mcp_config.json)'
  sections.push(`Sytra Studio runtime: ${skillsLine}. MCP (Model Context Protocol external tool servers via mcp_config.json): ${mcpState}.`)

  if (opts.agentPrompt?.trim()) {
    sections.push(`Agent persona:\n${opts.agentPrompt.trim().slice(0, 2400)}`)
  }

  if (opts.computerMode) {
    sections.push(
      opts.computerMode === 'sandbox'
        ? 'Environment: Isolated 24/7 agent workspace (host PC untouched).'
        : 'Environment: Host PC access enabled with user approval.',
    )
  }

  if (opts.recalledMemories?.length) {
    const memLines = opts.recalledMemories
      .slice(0, 5)
      .map(m => `- [${m.kind}${m.session_title ? ` · ${m.session_title}` : ''}] ${m.summary.replace(/\s+/g, ' ').slice(0, 220)}`)
    sections.push(`Recalled session memory:\n${memLines.join('\n')}`)
  }

  if (opts.tools.length) {
    const lines = opts.tools.map(({ exposedName, item }) => {
      const sig = compactSchemaParams(item.tool.inputSchema)
      const desc = (item.tool.description || item.tool.name).replace(/\s+/g, ' ').trim().slice(0, 90)
      return `- ${exposedName}(${sig}): ${desc}`
    })
    sections.push(
      `Tools (only call when needed; answer directly if you already know, and do NOT call recall_memory unless asked about past conversations):\n${lines.join('\n')}\nInvoke via native tool calling or:\n<tool_call>{"name":"TOOL_NAME","arguments":{...}}</tool_call>`,
    )
  }

  if (opts.skillsPrompt?.trim()) {
    const trimmed = opts.skillsPrompt.trim()
    const bounded = trimmed.length > MAX_SKILL_PROMPT_CHARS
      ? `${trimmed.slice(0, MAX_SKILL_PROMPT_CHARS)}\n...[skill truncated for context budget]`
      : trimmed
    sections.push(`Follow active skills:\n${bounded}`)
  }

  return sections.join('\n\n')
}

/**
 * Deterministic lightweight compaction digest for older conversation turns so
 * sessions shrink cleanly under context pressure and remain recallable later.
 */
export function buildCompactionDigest(
  messages: { role: string; content: string }[],
  priorSummary?: string,
): string {
  const parts: string[] = []
  if (priorSummary?.trim()) {
    parts.push(`Prior summary: ${priorSummary.trim().slice(0, 600)}`)
  }
  for (const msg of messages) {
    const clean = msg.content.replace(/\s+/g, ' ').trim()
    if (!clean) continue
    const label = msg.role === 'user' ? 'User' : 'Assistant'
    parts.push(`- ${label}: ${clean.slice(0, 220)}${clean.length > 220 ? '…' : ''}`)
  }
  const combined = parts.join('\n')
  return combined.length > 1800 ? `${combined.slice(0, 1790)}\n...[digest compacted]` : combined
}

/**
 * Extract inline `<tool_call>` or fenced tool JSON emitted by models lacking native
 * OpenAI tool-calling support, and strip those raw tags from user-visible text.
 */
export function extractInlineToolCalls(
  rawText: string,
  tools: { exposedName: string; item: IndexedTool }[],
): { calls: ToolCall[]; cleanText: string } {
  if (!rawText || !tools.length) return { calls: [], cleanText: rawText }
  const calls: ToolCall[] = []
  let clean = rawText

  const resolveToolName = (rawName: string): string | null => {
    const target = rawName.trim()
    if (!target) return null
    const exact = tools.find(t => t.exposedName === target || t.item.tool.name === target || `${t.item.server}/${t.item.tool.name}` === target)
    return exact ? exact.exposedName : null
  }

  const pushJsonCandidate = (jsonBody: string) => {
    try {
      const parsed: unknown = JSON.parse(jsonBody.trim())
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return false
      const obj = parsed as Record<string, unknown>
      const rawName = typeof obj.name === 'string' ? obj.name : typeof obj.tool === 'string' ? obj.tool : ''
      const exposed = resolveToolName(rawName)
      if (!exposed) return false
      let rawArgs: unknown = obj.arguments ?? obj.args ?? obj.input ?? {}
      if (typeof rawArgs === 'string') {
        rawArgs = tryParseArgString(rawArgs)
      }
      if (!rawArgs || typeof rawArgs !== 'object' || Array.isArray(rawArgs)) return false
      calls.push({
        id: `inline_${calls.length}_${Math.random().toString(36).slice(2, 8)}`,
        type: 'function',
        function: { name: exposed, arguments: JSON.stringify(rawArgs) },
      })
      return true
    } catch {
      return false
    }
  }

  // 1. Explicit <tool_call>...</tool_call> or <tool_call>...EOF tags
  clean = clean.replace(/<tool_call>\s*([\s\S]*?)(?:<\/tool_call>|$)/gi, (full, inner) => {
    return pushJsonCandidate(inner) ? '' : full
  })

  // 2. ```tool_call or ```json blocks targeting known tool names when nothing found yet
  if (!calls.length) {
    clean = clean.replace(/```(?:tool_call|tool|json)?\s*\n([\s\S]*?)```/gi, (full, inner) => {
      return pushJsonCandidate(inner) ? '' : full
    })
  }

  // 3. Bare single-line JSON object calling a known tool
  if (!calls.length) {
    const bareMatch = /^\s*(\{"name"\s*:\s*"[^"]+"[\s\S]*\})\s*$/m.exec(clean)
    if (bareMatch && pushJsonCandidate(bareMatch[1])) {
      clean = clean.replace(bareMatch[0], '')
    }
  }

  return { calls, cleanText: clean.replace(/\n{3,}/g, '\n\n').trim() }
}
