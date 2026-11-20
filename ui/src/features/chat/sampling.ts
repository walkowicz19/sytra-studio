export interface ChatServerPreferences {
  temperature?: number
  contextWindow?: number
  model?: string
}

export interface ChatServerPreferencesOptions {
  provider?: string
  knownModel?: string
  bypassCache?: boolean
}

const PREFS_CACHE = new Map<string, { at: number; prefs: ChatServerPreferences }>()
const PREFS_CACHE_TTL_MS = 30_000

function probeSignal(parent: AbortSignal, timeoutMs = 2500): AbortSignal {
  const timeoutFn = (AbortSignal as unknown as { timeout?: (ms: number) => AbortSignal }).timeout
  const anyFn = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any
  if (typeof timeoutFn === 'function' && typeof anyFn === 'function') {
    return anyFn([parent, timeoutFn(timeoutMs)])
  }
  return parent
}

/** Read preferences from a Sytra-managed proxy or external server; caches external probes for 30s to minimize TTFT. */
export async function chatServerPreferences(
  signal: AbortSignal,
  request: typeof fetch = fetch,
  baseUrl = 'http://127.0.0.1:8080',
  opts: ChatServerPreferencesOptions = {},
): Promise<ChatServerPreferences> {
  const cleanBase = baseUrl.replace(/\/+$/, '').replace(/\/v1$/, '')
  const isCustomFetch = request !== fetch
  if (!isCustomFetch && !opts.bypassCache) {
    const cached = PREFS_CACHE.get(cleanBase)
    if (cached && Date.now() - cached.at < PREFS_CACHE_TTL_MS) {
      return { ...cached.prefs, ...(opts.knownModel ? { model: opts.knownModel } : {}) }
    }
  }

  const out: ChatServerPreferences = {}
  const isDefaultProxy = cleanBase === 'http://127.0.0.1:8080' || cleanBase === 'http://localhost:8080'

  if (isDefaultProxy) {
    try {
      const response = await request(`${cleanBase}/sytra/preferences`, { signal: probeSignal(signal) })
      if (response.ok) {
        const data = await response.json()
        const temp: unknown = data?.temperature
        if (typeof temp === 'number' && Number.isFinite(temp) && temp >= 0 && temp <= 2) out.temperature = temp
        const ctx: unknown = data?.context_window
        if (typeof ctx === 'number' && Number.isFinite(ctx) && ctx > 0) out.contextWindow = Math.round(ctx)
        const model: unknown = data?.model
        if (typeof model === 'string' && model.trim()) out.model = model.trim()
      }
    } catch (error) {
      if (signal.aborted) throw error
    }
  }
  if (out.contextWindow === undefined || out.temperature === undefined) {
    try {
      const propsResp = await request(`${cleanBase}/props`, { signal: probeSignal(signal) })
      if (propsResp.ok) {
        const props = await propsResp.json()
        const nCtx: unknown = props?.default_generation_settings?.n_ctx
        if (out.contextWindow === undefined && typeof nCtx === 'number' && Number.isFinite(nCtx) && nCtx > 0) {
          out.contextWindow = Math.round(nCtx)
        }
        const temp: unknown = props?.default_generation_settings?.params?.temp ?? props?.default_generation_settings?.params?.temperature
        if (out.temperature === undefined && typeof temp === 'number' && Number.isFinite(temp) && temp >= 0 && temp <= 2) {
          out.temperature = temp
        }
      }
    } catch (error) {
      if (signal.aborted) throw error
    }
  }
  if (out.model === undefined) {
    if (opts.knownModel?.trim()) {
      out.model = opts.knownModel.trim()
    } else {
      try {
        const modelsResp = await request(`${cleanBase}/v1/models`, { signal: probeSignal(signal) })
        if (modelsResp.ok) {
          const modelsData = await modelsResp.json()
          const firstId: unknown = modelsData?.data?.[0]?.id
          if (typeof firstId === 'string' && firstId.trim()) out.model = firstId.trim()
        }
      } catch (error) {
        if (signal.aborted) throw error
      }
    }
  }
  if (!isCustomFetch) {
    PREFS_CACHE.set(cleanBase, { at: Date.now(), prefs: out })
  }
  return out
}

/** Backwards-compatible wrapper used by chat temperature tests. */
export async function chatTemperature(signal: AbortSignal, request: typeof fetch = fetch): Promise<number | undefined> {
  try {
    const response = await request('http://127.0.0.1:8080/sytra/preferences', { signal })
    if (!response.ok) return undefined
    const value: unknown = (await response.json())?.temperature
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 2 ? value : undefined
  } catch (error) {
    if (signal.aborted) throw error
    return undefined
  }
}
