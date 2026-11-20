export type InlineSpan =
  | { type: 'text'; text: string }
  | { type: 'strong'; text: string }
  | { type: 'em'; text: string }
  | { type: 'code'; text: string }
  | { type: 'link'; text: string; href: string }

export interface ChartSeries {
  name: string
  values: number[]
}

export interface ChartSpec {
  kind: 'bar' | 'line' | 'area' | 'pie'
  title: string
  subtitle?: string
  labels: string[]
  values: number[]
  unit?: string
  series?: ChartSeries[]
}

export interface KpiItem {
  label: string
  value: string
  delta?: string
  tone?: 'positive' | 'negative' | 'neutral'
}

export interface KpiSpec {
  title?: string
  items: KpiItem[]
  split?: {
    leftLabel: string
    leftValue: string
    rightLabel: string
    rightValue: string
    ratio: number
  }
}

export interface CardItem {
  title: string
  subtitle?: string
  body?: string
  tag?: string
  swatch?: string
}

export interface ListItem {
  checked?: boolean
  spans: InlineSpan[]
}

export type RichBlock =
  | { type: 'heading'; level: number; spans: InlineSpan[] }
  | { type: 'paragraph'; spans: InlineSpan[] }
  | { type: 'blockquote'; spans: InlineSpan[] }
  | { type: 'list'; ordered: boolean; items: ListItem[] }
  | { type: 'code'; lang: string; code: string; streaming?: boolean }
  | { type: 'table'; headers: InlineSpan[][]; rows: InlineSpan[][][]; chartable?: ChartSpec }
  | { type: 'chart'; spec: ChartSpec }
  | { type: 'kpi'; spec: KpiSpec }
  | { type: 'cards'; items: CardItem[]; swatches?: { name: string; hex: string }[] }
  | { type: 'hr' }

const HEX_COLOR_RE = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/
const FENCE_RE = /^```([a-zA-Z0-9_-]*)\s*$/
const HEADING_RE = /^(#{1,6})\s+(.+)$/
const HR_RE = /^(?:-{3,}|\*{3,}|_{3,})\s*$/
const TASK_RE = /^[-*+]\s+\[([ xX])\]\s+(.*)$/
const BULLET_RE = /^[-*+]\s+(.*)$/
const ORDERED_RE = /^\d+[.)]\s+(.*)$/

function isSafeHttpUrl(raw: string): boolean {
  try {
    const parsed = new URL(raw.trim())
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

/**
 * Parse inline markdown spans (`code`, `**bold**`, `*italic*`, `[text](url)`).
 * Rejects non-HTTP(S) links to avoid `javascript:` or `data:` XSS vectors.
 */
export function parseInlineSpans(input: string): InlineSpan[] {
  if (!input) return [{ type: 'text', text: '' }]
  const spans: InlineSpan[] = []
  const tokenRe = /`([^`]+)`|\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|\*([^*\n]+)\*/g
  let lastIndex = 0
  for (const match of input.matchAll(tokenRe)) {
    const idx = match.index ?? 0
    if (idx > lastIndex) {
      spans.push({ type: 'text', text: input.slice(lastIndex, idx) })
    }
    if (match[1] !== undefined) {
      spans.push({ type: 'code', text: match[1] })
    } else if (match[2] !== undefined && match[3] !== undefined) {
      if (isSafeHttpUrl(match[3])) {
        spans.push({ type: 'link', text: match[2], href: match[3].trim() })
      } else {
        spans.push({ type: 'text', text: match[0] })
      }
    } else if (match[4] !== undefined || match[5] !== undefined) {
      spans.push({ type: 'strong', text: match[4] ?? match[5] })
    } else if (match[6] !== undefined) {
      spans.push({ type: 'em', text: match[6] })
    }
    lastIndex = idx + match[0].length
  }
  if (lastIndex < input.length) {
    spans.push({ type: 'text', text: input.slice(lastIndex) })
  }
  return spans.length ? spans : [{ type: 'text', text: input }]
}

function splitTableCells(line: string): string[] {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map((cell) => cell.trim())
}

function isTableSeparator(line: string): boolean {
  const trimmed = line.trim()
  if (!trimmed.includes('-')) return false
  const cells = splitTableCells(trimmed)
  return cells.length > 0 && cells.every((c) => /^:?-{2,}:?$/.test(c))
}

function parseNumericCell(raw: string): number | null {
  const cleaned = raw.trim().replace(/^[~$€£¥+]/, '').replace(/[%,\s]+$/g, '').replace(/,/g, '')
  if (!cleaned || !/^-?\d+(?:\.\d+)?$/.test(cleaned)) return null
  const num = Number(cleaned)
  return Number.isFinite(num) ? num : null
}

function extractTableChart(rawHeaders: string[], rawRows: string[][]): ChartSpec | undefined {
  if (rawRows.length < 2 || rawHeaders.length < 2) return undefined
  const labels = rawRows.map((row, idx) => row[0]?.trim() || `Row ${idx + 1}`)
  const numericCols: { name: string; values: number[] }[] = []

  for (let c = 1; c < rawHeaders.length; c++) {
    const values: number[] = []
    let valid = true
    for (const row of rawRows) {
      const parsed = parseNumericCell(row[c] ?? '')
      if (parsed === null) {
        valid = false
        break
      }
      values.push(parsed)
    }
    if (valid && values.length === rawRows.length) {
      numericCols.push({ name: rawHeaders[c] || `Series ${c}`, values })
    }
  }

  if (!numericCols.length) return undefined
  return {
    kind: 'bar',
    title: numericCols.length === 1 ? numericCols[0].name : `${rawHeaders[0] || 'Category'} Comparison`,
    labels,
    values: numericCols[0].values,
    series: numericCols.length > 1 ? numericCols : undefined,
  }
}

function coerceChartNumber(raw: unknown): number {
  if (typeof raw === 'number') return raw
  if (typeof raw === 'string') {
    const parsed = parseNumericCell(raw)
    if (parsed !== null) return parsed
  }
  return Number.NaN
}

function parsePointArray(items: unknown[]): { labels: string[]; values: number[] } | null {
  const labels: string[] = []
  const values: number[] = []
  for (const item of items) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return null
    const row = item as Record<string, unknown>
    const rawLabel = row.label ?? row.name ?? row.date ?? row.month ?? row.quarter ?? row.category ?? row.x
    const rawVal = row.value ?? row.y ?? row.price ?? row.count ?? row.score
    const num = coerceChartNumber(rawVal)
    if (rawLabel === undefined || !Number.isFinite(num)) return null
    labels.push(String(rawLabel))
    values.push(num)
  }
  return labels.length ? { labels, values } : null
}

function parseChartSpec(obj: Record<string, unknown>, strict = false): ChartSpec | null {
  const rawKind = typeof obj.type === 'string' ? obj.type.toLowerCase() : typeof obj.kind === 'string' ? obj.kind.toLowerCase() : ''
  const explicitKind = rawKind === 'bar' || rawKind === 'line' || rawKind === 'area' || rawKind === 'pie' || rawKind === 'doughnut'
  const kind: ChartSpec['kind'] =
    rawKind === 'line' ? 'line' : rawKind === 'area' ? 'area' : (rawKind === 'pie' || rawKind === 'doughnut') ? 'pie' : 'bar'

  // Support Chart.js nested shape `{ type, data: { labels, datasets: [{ label, data }] }, options }`.
  const dataObj = !Array.isArray(obj.data) && obj.data && typeof obj.data === 'object'
    ? (obj.data as Record<string, unknown>)
    : null
  const chartLabelsSource = Array.isArray(obj.labels)
    ? obj.labels
    : dataObj && Array.isArray(dataObj.labels)
      ? dataObj.labels
      : null

  if (chartLabelsSource) {
    const labels = chartLabelsSource.map((l) => String(l))
    const rawValues = Array.isArray(obj.values) ? obj.values.map(coerceChartNumber) : []
    const seriesSource = Array.isArray(obj.series)
      ? obj.series
      : Array.isArray(obj.datasets)
        ? obj.datasets
        : dataObj && Array.isArray(dataObj.datasets)
          ? dataObj.datasets
          : null

    const rawSeries = seriesSource
      ? seriesSource
          .filter((s): s is Record<string, unknown> => Boolean(s && typeof s === 'object'))
          .map((s) => {
            const arr = Array.isArray(s.values) ? s.values : Array.isArray(s.data) ? s.data : []
            return {
              name: typeof s.name === 'string' ? s.name : typeof s.label === 'string' ? s.label : 'Series',
              values: arr.map(coerceChartNumber),
            }
          })
          .filter((s) => s.values.length > 0 && s.values.every(Number.isFinite))
      : undefined

    const values = rawValues.length ? rawValues : rawSeries?.[0]?.values ?? []
    if (!labels.length || !values.length || !values.every(Number.isFinite)) return null
    if (strict && !explicitKind && labels.length < 2) return null
    const len = Math.min(labels.length, values.length)
    const nestedTitle = (dataObj?.title && typeof dataObj.title === 'string') ? dataObj.title : ''
    return {
      kind,
      title: typeof obj.title === 'string' ? obj.title : nestedTitle,
      subtitle: typeof obj.subtitle === 'string' ? obj.subtitle : undefined,
      labels: labels.slice(0, len),
      values: values.slice(0, len),
      unit: typeof obj.unit === 'string' ? obj.unit : undefined,
      series: rawSeries?.map((s) => ({ ...s, values: s.values.slice(0, len) })),
    }
  }

  // Support point-array shape `{ title, data|points: [{ label/x, value/y }] }`.
  const arrSource = Array.isArray(obj.data) ? obj.data : Array.isArray(obj.points) ? obj.points : null
  if (arrSource) {
    const pts = parsePointArray(arrSource)
    if (!pts || (strict && !explicitKind && pts.labels.length < 2)) return null
    return {
      kind,
      title: typeof obj.title === 'string' ? obj.title : '',
      subtitle: typeof obj.subtitle === 'string' ? obj.subtitle : undefined,
      labels: pts.labels,
      values: pts.values,
      unit: typeof obj.unit === 'string' ? obj.unit : undefined,
    }
  }

  return null
}

function inferDeltaTone(delta: unknown): KpiItem['tone'] {
  const s = typeof delta === 'string' ? delta.trim() : ''
  if (!s) return 'neutral'
  if (s.startsWith('+') || /\b(up|gain|growth|higher|improved)\b/i.test(s)) return 'positive'
  if (s.startsWith('-') || /\b(down|loss|drop|lower)\b/i.test(s)) return 'negative'
  return 'neutral'
}

function parseKpiSpec(obj: Record<string, unknown>): KpiSpec | null {
  const rawItems = Array.isArray(obj.items)
    ? obj.items
    : typeof obj.label === 'string' && obj.value !== undefined
      ? [obj]
      : []
  const items: KpiItem[] = rawItems
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
    .map((item) => ({
      label: typeof item.label === 'string' ? item.label : 'Metric',
      value: item.value !== undefined ? String(item.value) : '',
      delta: typeof item.delta === 'string' && item.delta.trim() ? item.delta.trim() : undefined,
      tone:
        item.tone === 'positive' || item.tone === 'negative' || item.tone === 'neutral'
          ? item.tone
          : inferDeltaTone(item.delta),
    }))
    .filter((item) => item.value.length > 0)

  if (!items.length) return null

  let split: KpiSpec['split']
  if (obj.split && typeof obj.split === 'object') {
    const sp = obj.split as Record<string, unknown>
    const ratio = Number(sp.ratio ?? 0.5)
    if (typeof sp.leftLabel === 'string' && typeof sp.rightLabel === 'string') {
      split = {
        leftLabel: sp.leftLabel,
        leftValue: String(sp.leftValue ?? ''),
        rightLabel: sp.rightLabel,
        rightValue: String(sp.rightValue ?? ''),
        ratio: Number.isFinite(ratio) ? Math.min(0.95, Math.max(0.05, ratio)) : 0.5,
      }
    }
  }

  return {
    title: typeof obj.title === 'string' ? obj.title : undefined,
    items,
    split,
  }
}

function parseCardsSpec(obj: Record<string, unknown>): Extract<RichBlock, { type: 'cards' }> | null {
  const rawItems = Array.isArray(obj.items) ? obj.items : []
  const items: CardItem[] = rawItems
    .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === 'object'))
    .map((item) => {
      const swatch = typeof item.swatch === 'string' && HEX_COLOR_RE.test(item.swatch.trim()) ? item.swatch.trim() : undefined
      return {
        title: typeof item.title === 'string' ? item.title : '',
        subtitle: typeof item.subtitle === 'string' ? item.subtitle : undefined,
        body: typeof item.body === 'string' ? item.body : undefined,
        tag: typeof item.tag === 'string' ? item.tag : undefined,
        swatch,
      }
    })
    .filter((item) => item.title.length > 0 || Boolean(item.swatch))

  const swatches = Array.isArray(obj.swatches)
    ? obj.swatches
        .filter((s): s is Record<string, unknown> => Boolean(s && typeof s === 'object'))
        .map((s) => ({
          name: typeof s.name === 'string' ? s.name : '',
          hex: typeof s.hex === 'string' ? s.hex.trim() : '',
        }))
        .filter((s) => HEX_COLOR_RE.test(s.hex))
    : undefined

  if (!items.length && !(swatches && swatches.length)) return null
  return { type: 'cards', items, swatches }
}

function tryParseWidgetBlock(lang: string, body: string): RichBlock | null {
  const normalized = lang.trim().toLowerCase()
  const isExplicitWidget = normalized === 'chart' || normalized === 'kpi' || normalized === 'cards'
  const isChartLikeJson = isExplicitWidget || normalized === '' || normalized === 'json' || normalized === 'chartjs' || normalized === 'graph' || normalized === 'plot'
  if (!isChartLikeJson) return null
  try {
    const cleaned = body.trim().replace(/,\s*([}\]])/g, '$1')
    const parsed: unknown = JSON.parse(cleaned)
    if (Array.isArray(parsed) && normalized === 'chart') {
      const pts = parsePointArray(parsed)
      return pts ? { type: 'chart', spec: { kind: 'bar', title: '', labels: pts.labels, values: pts.values } } : null
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null
    const obj = parsed as Record<string, unknown>
    if (normalized === 'chart') {
      const spec = parseChartSpec(obj)
      return spec ? { type: 'chart', spec } : null
    }
    if (normalized === 'kpi') {
      const spec = parseKpiSpec(obj)
      return spec ? { type: 'kpi', spec } : null
    }
    if (normalized === 'cards') {
      return parseCardsSpec(obj)
    }
    // Auto-promote ```json / ```chartjs / unlabeled fences when payload matches interactive Chart schema.
    const autoChart = parseChartSpec(obj, true)
    if (autoChart) return { type: 'chart', spec: autoChart }
    return null
  } catch {
    return null
  }
}

/**
 * Parse streaming-safe Markdown + lightweight visual widgets (`chart`, `kpi`, `cards`,
 * and numeric-chartable Markdown tables) into a typed AST without raw HTML injection.
 */
export function parseRichBlocks(markdown: string): RichBlock[] {
  if (!markdown || !markdown.trim()) return []
  const lines = markdown.replace(/\r\n/g, '\n').split('\n')
  const blocks: RichBlock[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) {
      i++
      continue
    }

    const fenceMatch = line.match(FENCE_RE)
    if (fenceMatch) {
      const lang = fenceMatch[1] || ''
      const bodyLines: string[] = []
      i++
      let closed = false
      while (i < lines.length) {
        if (/^```\s*$/.test(lines[i])) {
          closed = true
          i++
          break
        }
        bodyLines.push(lines[i])
        i++
      }
      const codeBody = bodyLines.join('\n')
      if (closed) {
        const widget = tryParseWidgetBlock(lang, codeBody)
        if (widget) {
          blocks.push(widget)
          continue
        }
      }
      blocks.push({ type: 'code', lang, code: codeBody, streaming: !closed })
      continue
    }

    if (HR_RE.test(line)) {
      blocks.push({ type: 'hr' })
      i++
      continue
    }

    const headingMatch = line.match(HEADING_RE)
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        spans: parseInlineSpans(headingMatch[2].trim()),
      })
      i++
      continue
    }

    if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const rawHeaders = splitTableCells(line)
      i += 2
      const rawRows: string[][] = []
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
        const cells = splitTableCells(lines[i])
        while (cells.length < rawHeaders.length) cells.push('')
        rawRows.push(cells.slice(0, rawHeaders.length))
        i++
      }
      blocks.push({
        type: 'table',
        headers: rawHeaders.map((h) => parseInlineSpans(h)),
        rows: rawRows.map((row) => row.map((cell) => parseInlineSpans(cell))),
        chartable: extractTableChart(rawHeaders, rawRows),
      })
      continue
    }

    if (line.trimStart().startsWith('>')) {
      const quoteLines: string[] = []
      while (i < lines.length && lines[i].trimStart().startsWith('>')) {
        quoteLines.push(lines[i].trimStart().replace(/^>\s?/, ''))
        i++
      }
      blocks.push({ type: 'blockquote', spans: parseInlineSpans(quoteLines.join(' ')) })
      continue
    }

    if (TASK_RE.test(line.trimStart()) || BULLET_RE.test(line.trimStart())) {
      const items: ListItem[] = []
      while (i < lines.length) {
        const trimmed = lines[i].trimStart()
        const task = trimmed.match(TASK_RE)
        const bullet = trimmed.match(BULLET_RE)
        if (task) {
          items.push({ checked: task[1].toLowerCase() === 'x', spans: parseInlineSpans(task[2]) })
          i++
        } else if (bullet) {
          items.push({ spans: parseInlineSpans(bullet[1]) })
          i++
        } else {
          break
        }
      }
      blocks.push({ type: 'list', ordered: false, items })
      continue
    }

    if (ORDERED_RE.test(line.trimStart())) {
      const items: ListItem[] = []
      while (i < lines.length) {
        const ord = lines[i].trimStart().match(ORDERED_RE)
        if (!ord) break
        items.push({ spans: parseInlineSpans(ord[1]) })
        i++
      }
      blocks.push({ type: 'list', ordered: true, items })
      continue
    }

    const paraLines: string[] = []
    while (
      i < lines.length &&
      lines[i].trim() &&
      !FENCE_RE.test(lines[i]) &&
      !HEADING_RE.test(lines[i]) &&
      !HR_RE.test(lines[i]) &&
      !lines[i].trimStart().startsWith('>') &&
      !BULLET_RE.test(lines[i].trimStart()) &&
      !ORDERED_RE.test(lines[i].trimStart()) &&
      !(lines[i].includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1]))
    ) {
      paraLines.push(lines[i].trim())
      i++
    }
    if (paraLines.length) {
      const joined = paraLines.join('\n').trim()
      if (joined.startsWith('{') && joined.endsWith('}')) {
        const bareWidget = tryParseWidgetBlock('json', joined)
        if (bareWidget) {
          blocks.push(bareWidget)
          continue
        }
      }
      blocks.push({ type: 'paragraph', spans: parseInlineSpans(paraLines.join(' ')) })
    }
  }

  return blocks
}
