import type { AgentProfile, ChatTargetsReport, MemoryEntry } from '../../types'

const CODE_FENCE_RE = /```[\s\S]*?```/g
const MD_LINK_RE = /\[([^\]]+)\]\([^)]+\)/g
const URL_RE = /https?:\/\/[^\s)]+/g
const TABLE_SEP_RE = /^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$/gm
const MD_SYNTAX_RE = /(^#+\s*|`+|[*_~]{1,3}|^>\s*)/gm
const PT_HINT_RE = /(\b(ol[aá]|voc[eê]s?|n[aã]o|sim|obrigad[oa]|tudo|bem|como|para|com|isso|aqui|agora|sistema|modelo|relat[oó]rio|arquivo|ajuda)\b|[ãõç])/i

export function cleanSpeechText(raw: string, maxChars = 2400): string {
  let text = (raw || '').replace(/\r\n/g, '\n')
  text = text.replace(CODE_FENCE_RE, ' Code block shown on screen. ')
  text = text.replace(TABLE_SEP_RE, ' ')
  text = text.replace(MD_LINK_RE, '$1')
  text = text.replace(URL_RE, ' link ')
  text = text.replace(MD_SYNTAX_RE, ' ')
  text = text.replace(/\|/g, ', ')
  text = text.replace(/[ \t\n]+/g, ' ').trim()
  if (text.length > maxChars) {
    return `${text.slice(0, maxChars - 3).trimEnd()}...`
  }
  return text
}

/**
 * Extract complete spoken sentences incrementally from a live LLM token stream
 * so Kokoro-82M / TTS speaks low-latency sentences while tokens continue streaming.
 */
export function extractReadySentences(
  buffer: string,
  flush = false,
): { sentences: string[]; remainder: string } {
  if (!buffer) return { sentences: [], remainder: '' }
  const sentences: string[] = []
  let rest = buffer

  // Avoid breaking inside an open fenced code block while streaming
  if (!flush) {
    const fenceCount = (rest.match(/```/g) ?? []).length
    if (fenceCount % 2 === 1) {
      return { sentences: [], remainder: rest }
    }
  }

  const re = /([^.!?\n]*[.!?]+)(?:\s+|\n+|$)/y
  let lastEnd = 0
  while (true) {
    re.lastIndex = lastEnd
    const m = re.exec(rest)
    if (!m) break
    const matchEnd = re.lastIndex
    // Don't split prematurely at EOF without whitespace unless flushing
    if (!flush && matchEnd === rest.length && !/\s$/.test(rest)) {
      break
    }
    const candidate = cleanSpeechText(m[1], 400)
    if (candidate.length >= 2) {
      sentences.push(candidate)
    }
    lastEnd = matchEnd
  }

  rest = rest.slice(lastEnd)
  if (flush) {
    const tail = cleanSpeechText(rest, 800)
    if (tail.length >= 2) {
      sentences.push(tail)
    }
    rest = ''
  }

  return { sentences, remainder: rest }
}

export function detectSpeechLangTag(text: string): 'pt-BR' | 'en-US' {
  return PT_HINT_RE.test(text || '') ? 'pt-BR' : 'en-US'
}

export function computeRmsLevel(samples: Float32Array): number {
  if (!samples.length) return 0
  let sumSq = 0
  for (let i = 0; i < samples.length; i += 1) {
    const v = samples[i]
    sumSq += v * v
  }
  return Math.sqrt(sumSq / samples.length)
}

export function downsampleFloat32(
  samples: Float32Array,
  inRate: number,
  outRate = 16000,
): Float32Array {
  if (outRate >= inRate || inRate <= 0) return samples
  const ratio = inRate / outRate
  const newLen = Math.max(1, Math.round(samples.length / ratio))
  const out = new Float32Array(newLen)
  let offsetResult = 0
  let offsetBuffer = 0
  while (offsetResult < out.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < samples.length; i += 1) {
      accum += samples[i]
      count += 1
    }
    out[offsetResult] = count ? accum / count : 0
    offsetResult += 1
    offsetBuffer = nextOffsetBuffer
  }
  return out
}

/**
 * Encode normalized mono [-1..1] float samples as a base64 PCM-16 WAV string.
 */
export function encodePcm16WavBase64(samples: Float32Array, sampleRate = 16000): string {
  const numSamples = samples.length
  const dataBytes = numSamples * 2
  const buf = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buf)

  const writeAscii = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM header size
  view.setUint16(20, 1, true) // Linear PCM
  view.setUint16(22, 1, true) // Mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(36, 'data')
  view.setUint32(40, dataBytes, true)

  let offset = 44
  for (let i = 0; i < numSamples; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }

  const bytes = new Uint8Array(buf)
  let binary = ''
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize))
  }
  if (typeof btoa === 'function') {
    return btoa(binary)
  }
  const nodeBuf = (globalThis as unknown as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } } }).Buffer
  return nodeBuf ? nodeBuf.from(bytes).toString('base64') : ''
}

export function buildProactiveBriefingPrompt(opts: {
  agent?: AgentProfile | null
  hw?: { backend: string; vram_mb: number | null; ram_mb: number | null } | null
  targets?: ChatTargetsReport | null
  memories?: MemoryEntry[]
  webSearchEnabled?: boolean
  contextPct?: number
}): string {
  const hwLine = opts.hw
    ? `Backend=${opts.hw.backend}, VRAM=${opts.hw.vram_mb ?? 0}MB, RAM=${opts.hw.ram_mb ?? 0}MB`
    : 'Hardware telemetry pending'
  const models: string[] = []
  if (opts.targets?.providers.sytra_running) {
    models.push(`Sytra(${opts.targets.providers.sytra_models.join(', ') || 'active'})`)
  }
  if (opts.targets?.providers.ollama_running) {
    models.push(`Ollama(${opts.targets.providers.ollama_models.slice(0, 4).map(m => m.name).join(', ')})`)
  }
  if (opts.targets?.providers.lmstudio_running) {
    models.push(`LMStudio(${opts.targets.providers.lmstudio_models.slice(0, 3).join(', ')})`)
  }
  const memSnippet = (opts.memories ?? [])
    .slice(0, 3)
    .map(m => `- [${m.kind}] ${m.summary.slice(0, 140)}`)
    .join('\n')

  const objective = opts.agent?.routine_prompt?.trim()
    || opts.agent?.role?.trim()
    || 'Run a proactive live status & helpful task briefing using your tools cleanly without waiting to be asked.'

  return [
    `[Proactive Autonomous Update — ${opts.agent?.name || 'Lil Sytra'}]`,
    `Objective: ${objective}`,
    `Live PC State: ${hwLine} | Context Used: ${opts.contextPct ?? 0}%`,
    `Running Models: ${models.length ? models.join('; ') : 'Idle / standby'}`,
    opts.webSearchEnabled ? 'Web Search is ON — verify any fresh topic if relevant.' : '',
    memSnippet ? `Recent Memory:\n${memSnippet}` : '',
    'Deliver concrete useful information concisely (use a markdown table/KPI block if numeric, plus 1 spoken highlight).',
  ]
    .filter(Boolean)
    .join('\n')
}
