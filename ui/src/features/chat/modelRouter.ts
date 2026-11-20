import type { ChatTargetsReport } from '../../types'
import type { ChatEndpoint } from './chatStore.svelte'

export type LanguageScope = 'multilingual' | 'english' | 'unknown'

export interface ModelLanguageInfo {
  scope: LanguageScope
  label: string
}

export interface TaskSignals {
  needsVision: boolean
  needsCoding: boolean
  needsReasoning: boolean
  needsMultilingual: boolean
  isQuick: boolean
}

export interface RoutedDecision {
  endpoint: ChatEndpoint
  reason: string
  language: ModelLanguageInfo
  localFilePath?: string
}

const ENGLISH_ONLY_RE = /(?:^|[-_/])en(?:$|[-_.:])|english|tinyllama|phi-2|phi2|codellama|stablelm-zephyr/i
const MULTILINGUAL_RE = /qwen|llama[-_ ]?3|llama[-_ ]?4|gemma|mistral|mixtral|aya|command-r|deepseek|glm|yi-|internlm|lfm|eurollm|multilingual|bilingual|nemo|kimi/i

const CODE_SIGNAL_RE = /\b(code|bug|fix|refactor|function|class|sql|rust|python|typescript|javascript|svelte|html|css|regex|compile|git|diff|api|debug|traceback|error)\b|```/i
const REASON_SIGNAL_RE = /\b(why|prove|analyze|architecture|tradeoff|math|calculate|equation|plan|compare|evaluate|strategy|step by step|benchmark)\b/i
const VISION_SIGNAL_RE = /\b(image|screenshot|photo|diagram|chart|picture|ocr|visual|gif|draw)\b/i
const NON_EN_HINT_RE = /[^\x00-\x7F]|\b(ol[aá]|obrigad[oa]|voc[eê]|n[aã]o|como|qual|para|por favor|hola|gracias|bonjour|merci|danke|hallo|ciao|buongiorno)\b/i

export function inferLanguageScope(modelName: string, reportedScope?: string): ModelLanguageInfo {
  if (reportedScope === 'multilingual') return { scope: 'multilingual', label: 'Multilingual' }
  if (reportedScope === 'english') return { scope: 'english', label: 'English-focused' }
  if (ENGLISH_ONLY_RE.test(modelName)) return { scope: 'english', label: 'English-focused' }
  if (MULTILINGUAL_RE.test(modelName)) return { scope: 'multilingual', label: 'Multilingual' }
  return { scope: 'unknown', label: 'Language: unverified' }
}

export function classifyTaskSignals(opts: {
  prompt: string
  hasImages?: boolean
  agentRole?: string
  agentPrompt?: string
}): TaskSignals {
  const combined = `${opts.prompt}\n${opts.agentRole ?? ''}\n${opts.agentPrompt ?? ''}`.trim()
  const needsVision = Boolean(opts.hasImages || VISION_SIGNAL_RE.test(combined))
  const needsCoding = CODE_SIGNAL_RE.test(combined)
  const needsReasoning = REASON_SIGNAL_RE.test(combined) || opts.prompt.trim().length > 320
  const needsMultilingual = NON_EN_HINT_RE.test(opts.prompt)
  const isQuick = !needsVision && !needsCoding && !needsReasoning && opts.prompt.trim().length > 0 && opts.prompt.trim().length < 85
  return { needsVision, needsCoding, needsReasoning, needsMultilingual, isQuick }
}

function estimateParamB(name: string): number | null {
  const m = /(\d+(?:\.\d+)?)\s*b\b/i.exec(name)
  if (!m) return null
  const val = Number(m[1])
  return Number.isFinite(val) ? val : null
}

/**
 * Deterministically select the highest-scoring runnable model for the current
 * task and custom agent objective with zero LLM routing overhead.
 */
export function routeBestModel(opts: {
  prompt: string
  hasImages?: boolean
  agentRole?: string
  agentPrompt?: string
  targets: ChatTargetsReport | null
  vramMb?: number | null
  fallbackEndpoint?: ChatEndpoint
  translateReason?: (key: string, params?: Record<string, string>) => string
}): RoutedDecision | null {
  const signals = classifyTaskSignals(opts)
  const candidates: {
    endpoint: ChatEndpoint
    sizeGb: number
    running: boolean
    language: ModelLanguageInfo
    localFilePath?: string
  }[] = []

  const t = opts.targets
  if (t) {
    for (const id of t.providers.sytra_models) {
      candidates.push({
        endpoint: { provider: 'sytra', baseUrl: 'http://127.0.0.1:8080', model: id },
        sizeGb: 0,
        running: true,
        language: inferLanguageScope(id),
      })
    }
    for (const om of t.providers.ollama_models) {
      candidates.push({
        endpoint: { provider: 'ollama', baseUrl: 'http://127.0.0.1:11434', model: om.name },
        sizeGb: om.size_gb || 0,
        running: t.providers.ollama_running,
        language: inferLanguageScope(om.name, om.language_scope),
      })
    }
    for (const lm of t.providers.lmstudio_models) {
      candidates.push({
        endpoint: { provider: 'lmstudio', baseUrl: 'http://127.0.0.1:1234', model: lm },
        sizeGb: 0,
        running: true,
        language: inferLanguageScope(lm),
      })
    }
    // Include local files as runnable fallback candidates when few/no daemon models exist
    for (const loc of t.local_models) {
      if (loc.format.toLowerCase() !== 'gguf') continue
      candidates.push({
        endpoint: { provider: 'sytra', baseUrl: 'http://127.0.0.1:8080', model: loc.name },
        sizeGb: loc.size_gb || 0,
        running: false,
        language: inferLanguageScope(loc.name),
        localFilePath: loc.path,
      })
    }
  }

  if (!candidates.length && opts.fallbackEndpoint && opts.fallbackEndpoint.provider !== 'router' && opts.fallbackEndpoint.model) {
    candidates.push({
      endpoint: opts.fallbackEndpoint,
      sizeGb: 0,
      running: true,
      language: inferLanguageScope(opts.fallbackEndpoint.model),
    })
  }

  if (!candidates.length) return null

  const vramGb = opts.vramMb && opts.vramMb > 0 ? opts.vramMb / 1024 : null

  const scored = candidates.map((cand) => {
    const idLower = cand.endpoint.model.toLowerCase()
    const paramB = estimateParamB(idLower)
    let score = 0
    const tags: string[] = []

    if (cand.running) {
      score += 30
      tags.push('ready')
    } else if (cand.endpoint.provider === 'ollama') {
      score += 12
    }

    if (vramGb && cand.sizeGb > 0 && cand.sizeGb > vramGb * 0.92) {
      score -= 40
    } else if (vramGb && cand.sizeGb > 0 && cand.sizeGb <= vramGb * 0.85) {
      score += 8
    }

    if (signals.needsVision && /(?:vl|vision|llava|pixtral|moondream|minicpm-v|gemma3|qwen3\.5)/i.test(idLower)) {
      score += 55
      tags.push('vision')
    }
    if (signals.needsCoding && /(?:coder|codestral|devstral|deepseek|starcoder|kimi)/i.test(idLower)) {
      score += 45
      tags.push('coding')
    }
    if (signals.needsReasoning) {
      if (/(?:r1|qwq|thinking|reasoning|qwen3|opus|pro)/i.test(idLower)) {
        score += 36
        tags.push('reasoning')
      }
      if (paramB && paramB >= 7) {
        score += Math.min(20, paramB * 1.5)
      }
    }
    if (signals.isQuick) {
      if ((paramB && paramB <= 4) || /(?:mini|flash|lite|nano|0\.5b|1\.5b|1\.7b|3b|4b)/i.test(idLower)) {
        score += 24
        tags.push('fast')
      }
    }

    if (signals.needsMultilingual) {
      if (cand.language.scope === 'multilingual') {
        score += 26
        tags.push('multilingual')
      } else if (cand.language.scope === 'english') {
        score -= 35
      }
    } else if (cand.language.scope === 'multilingual') {
      score += 6
    }

    return { cand, score, tags }
  })

  scored.sort((a, b) =>
    b.score - a.score
    || Number(b.cand.running) - Number(a.cand.running)
    || a.cand.endpoint.model.localeCompare(b.cand.endpoint.model),
  )

  const best = scored[0]
  const reason = (key: string, fallback: string, params?: Record<string, string>) =>
    opts.translateReason?.(key, params) ?? fallback
  const reasons: string[] = []
  if (signals.needsVision && best.tags.includes('vision')) reasons.push(reason('chat.reasonVision', 'vision input'))
  if (signals.needsCoding && best.tags.includes('coding')) reasons.push(reason('chat.reasonCoding', 'coding task'))
  if (signals.needsReasoning && best.tags.includes('reasoning')) reasons.push(reason('chat.reasonReasoning', 'reasoning depth'))
  if (signals.needsMultilingual && best.tags.includes('multilingual')) reasons.push(reason('chat.reasonMultilingual', 'multilingual'))
  if (signals.isQuick && best.tags.includes('fast')) reasons.push(reason('chat.reasonFast', 'low latency'))
  const agent = opts.agentRole?.trim().slice(0, 32)
  if (!reasons.length && agent) reasons.push(reason('chat.reasonAgent', `agent: ${agent}`, { name: agent }))
  if (!reasons.length) {
    reasons.push(best.cand.running
      ? reason('chat.reasonActive', 'active runtime')
      : reason('chat.reasonLocal', 'best local fit'))
  }

  return {
    endpoint: best.cand.endpoint,
    reason: reasons.join(' · '),
    language: best.cand.language,
    localFilePath: best.cand.localFilePath,
  }
}
