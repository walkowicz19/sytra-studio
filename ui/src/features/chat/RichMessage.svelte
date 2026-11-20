<script lang="ts">
  import { t } from '../../i18n.svelte'
  import { parseRichBlocks, type ChartSpec, type InlineSpan } from './markdown'

  interface Props {
    content: string
    fallback?: string
  }

  let { content, fallback = '' }: Props = $props()

  let blocks = $derived(parseRichBlocks(content))
  let tableModes = $state<Record<number, 'table' | 'bar' | 'line'>>({})
  let chartModes = $state<Record<number, ChartSpec['kind'] | 'data'>>({})
  let hoveredPoint = $state<Record<number, number | null>>({})
  let hiddenSeries = $state<Record<number, Record<string, boolean>>>({})
  let copiedIndex = $state<number | null>(null)

  const PALETTE = [
    'var(--color-brand)',
    'var(--color-success)',
    'var(--color-info)',
    '#d97706',
    '#7c3aed',
    '#0891b2',
  ]

  function setTableMode(idx: number, mode: 'table' | 'bar' | 'line') {
    tableModes = { ...tableModes, [idx]: mode }
  }

  function setChartMode(idx: number, mode: ChartSpec['kind'] | 'data') {
    chartModes = { ...chartModes, [idx]: mode }
    hoveredPoint = { ...hoveredPoint, [idx]: null }
  }

  function setHoverPoint(blockIdx: number, pointIdx: number | null) {
    if (blockIdx < 0) return
    hoveredPoint = { ...hoveredPoint, [blockIdx]: pointIdx }
  }

  function handleCartesianPointer(e: PointerEvent, blockIdx: number, count: number, isBar: boolean) {
    if (blockIdx < 0 || count <= 0) return
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    if (rect.width <= 0) return
    const svgX = ((e.clientX - rect.left) / rect.width) * 640
    const plotLeft = 48
    const plotW = 640 - 48 - 18
    const clamped = Math.max(0, Math.min(plotW - 0.01, svgX - plotLeft))
    const idx = count === 1
      ? 0
      : isBar
        ? Math.min(count - 1, Math.floor((clamped / plotW) * count))
        : Math.min(count - 1, Math.round((clamped / plotW) * (count - 1)))
    setHoverPoint(blockIdx, idx)
  }

  function toggleSeriesVisibility(blockIdx: number, seriesName: string) {
    if (blockIdx < 0) return
    const currentMap = hiddenSeries[blockIdx] ?? {}
    hiddenSeries = {
      ...hiddenSeries,
      [blockIdx]: { ...currentMap, [seriesName]: !currentMap[seriesName] },
    }
  }

  function chartStats(values: number[], unit?: string): { min: string; max: string; delta: string | null } {
    if (!values.length) return { min: '—', max: '—', delta: null }
    const min = Math.min(...values)
    const max = Math.max(...values)
    let delta: string | null = null
    if (values.length >= 2 && values[0] !== 0) {
      const pct = ((values[values.length - 1] - values[0]) / Math.abs(values[0])) * 100
      if (Number.isFinite(pct)) {
        delta = `${pct >= 0 ? '+' : ''}${pct.toFixed(1).replace(/\.0$/, '')}%`
      }
    }
    return {
      min: formatNumber(min, unit),
      max: formatNumber(max, unit),
      delta,
    }
  }

  async function copyCode(idx: number, code: string) {
    try {
      await navigator.clipboard?.writeText(code)
      copiedIndex = idx
      setTimeout(() => {
        if (copiedIndex === idx) copiedIndex = null
      }, 1500)
    } catch {
      // Clipboard API unavailable in headless or restricted context.
    }
  }

  function formatNumber(n: number, unit?: string): string {
    const formatted = Number.isInteger(n) ? String(n) : n.toFixed(2).replace(/\.?0+$/, '')
    return unit ? `${formatted}${unit.startsWith('%') ? unit : ` ${unit}`}` : formatted
  }

  function chartSeriesList(spec: ChartSpec): { name: string; values: number[]; color: string }[] {
    if (spec.series && spec.series.length > 0) {
      return spec.series.map((s, i) => ({
        name: s.name,
        values: s.values,
        color: PALETTE[i % PALETTE.length],
      }))
    }
    return [{ name: spec.title || 'Value', values: spec.values, color: PALETTE[0] }]
  }

  function chartBounds(spec: ChartSpec): { min: number; max: number } {
    const allValues = chartSeriesList(spec).flatMap((s) => s.values)
    const rawMin = Math.min(0, ...allValues)
    const rawMax = Math.max(...allValues)
    if (rawMax === rawMin) return { min: Math.min(0, rawMin), max: rawMax + 1 }
    return { min: rawMin, max: rawMax * 1.1 }
  }

  function pieSlices(spec: ChartSpec): { label: string; value: number; pct: number; path: string; color: string }[] {
    const positive = spec.values.map((v) => Math.max(0, v))
    const total = positive.reduce((sum, v) => sum + v, 0) || 1
    let angle = -Math.PI / 2
    const cx = 110
    const cy = 96
    const rOuter = 74
    const rInner = 40
    return spec.labels.map((label, idx) => {
      const value = positive[idx] ?? 0
      const sweep = (value / total) * Math.PI * 2
      const start = angle
      const end = angle + sweep
      angle = end
      const x1 = cx + rOuter * Math.cos(start)
      const y1 = cy + rOuter * Math.sin(start)
      const x2 = cx + rOuter * Math.cos(end)
      const y2 = cy + rOuter * Math.sin(end)
      const x3 = cx + rInner * Math.cos(end)
      const y3 = cy + rInner * Math.sin(end)
      const x4 = cx + rInner * Math.cos(start)
      const y4 = cy + rInner * Math.sin(start)
      const large = sweep > Math.PI ? 1 : 0
      const path = sweep >= Math.PI * 1.999
        ? `M ${cx} ${cy - rOuter} A ${rOuter} ${rOuter} 0 1 1 ${cx - 0.01} ${cy - rOuter} Z M ${cx} ${cy - rInner} A ${rInner} ${rInner} 0 1 0 ${cx - 0.01} ${cy - rInner} Z`
        : `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${rOuter} ${rOuter} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} L ${x3.toFixed(1)} ${y3.toFixed(1)} A ${rInner} ${rInner} 0 ${large} 0 ${x4.toFixed(1)} ${y4.toFixed(1)} Z`
      return {
        label,
        value: spec.values[idx] ?? 0,
        pct: Math.round((value / total) * 100),
        path,
        color: PALETTE[idx % PALETTE.length],
      }
    })
  }

  function truncateLabel(text: string, max = 12): string {
    return text.length > max ? `${text.slice(0, max - 1)}...` : text
  }
</script>

{#snippet renderSpans(spans: InlineSpan[])}
  {#each spans as span}
    {#if span.type === 'strong'}
      <strong>{span.text}</strong>
    {:else if span.type === 'em'}
      <em>{span.text}</em>
    {:else if span.type === 'code'}
      <code class="inline-code">{span.text}</code>
    {:else if span.type === 'link'}
      <a href={span.href} target="_blank" rel="noopener noreferrer">{span.text}</a>
    {:else}
      {span.text}
    {/if}
  {/each}
{/snippet}

{#snippet renderSvgChart(spec: ChartSpec, blockIdx = -1, forceKind?: 'bar' | 'line')}
  {@const activeMode = forceKind ?? (blockIdx >= 0 ? (chartModes[blockIdx] ?? spec.kind) : spec.kind)}
  {@const allSeries = chartSeriesList(spec)}
  {@const hiddenMap = blockIdx >= 0 ? (hiddenSeries[blockIdx] ?? {}) : {}}
  {@const visibleSeries = allSeries.filter((s) => !hiddenMap[s.name]).length ? allSeries.filter((s) => !hiddenMap[s.name]) : allSeries}
  {@const activeHover = blockIdx >= 0 ? (hoveredPoint[blockIdx] ?? null) : null}
  {@const primaryValues = visibleSeries[0]?.values ?? spec.values}
  {@const stats = chartStats(primaryValues, spec.unit)}
  {@const bounds = (() => {
    const vals = visibleSeries.flatMap((s) => s.values)
    const rawMin = Math.min(0, ...vals)
    const rawMax = Math.max(...vals)
    return rawMax === rawMin ? { min: Math.min(0, rawMin), max: rawMax + 1 } : { min: rawMin, max: rawMax * 1.1 }
  })()}
  {@const count = Math.max(1, spec.labels.length)}
  {@const padLeft = 48}
  {@const padRight = 18}
  {@const padTop = 20}
  {@const padBottom = 38}
  {@const plotW = 640 - padLeft - padRight}
  {@const plotH = 236 - padTop - padBottom}
  <div
    class="rich-chart"
    role="region"
    aria-label={spec.title || t('chat.chartInteractive')}
    data-chart-kind={activeMode}
    onpointerleave={() => setHoverPoint(blockIdx, null)}
  >
    <div class="chart-head">
      <div class="chart-title-wrap">
        {#if spec.title}<strong class="chart-title">{spec.title}</strong>{/if}
        {#if spec.subtitle}<span class="chart-sub">{spec.subtitle}</span>{/if}
      </div>
      <div class="chart-controls">
        {#if allSeries.length > 1 && activeMode !== 'pie' && activeMode !== 'data'}
          <div class="chart-legend">
            {#each allSeries as s}
              <button
                type="button"
                class="legend-btn"
                class:muted={Boolean(hiddenMap[s.name])}
                onclick={() => toggleSeriesVisibility(blockIdx, s.name)}
              >
                <span class="legend-dot" style={`background:${s.color}`}></span>
                <span>{s.name}</span>
              </button>
            {/each}
          </div>
        {/if}
        {#if !forceKind && blockIdx >= 0}
          <div class="view-switch" role="group" aria-label={t('chat.chartMode')}>
            {#each ['bar', 'line', 'area', 'pie', 'data'] as const as modeKey}
              <button type="button" class:active={activeMode === modeKey} onclick={() => setChartMode(blockIdx, modeKey)}>
                {t(`chat.${modeKey}`)}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    {#if activeHover !== null && activeHover >= 0 && activeHover < spec.labels.length}
      <div class="chart-hover-hud" role="status">
        <strong>{spec.labels[activeHover]}</strong>
        <div class="hud-values">
          {#each visibleSeries as s}
            <span class="hud-val">
              <span class="legend-dot" style={`background:${s.color}`}></span>
              {visibleSeries.length > 1 ? `${s.name}: ` : ''}{formatNumber(s.values[activeHover] ?? 0, spec.unit)}
            </span>
          {/each}
        </div>
      </div>
    {/if}

    {#if activeMode === 'data'}
      <div class="table-scroll">
        <table class="rich-table">
          <thead>
            <tr>
              <th>{t('chat.chartLabel')}</th>
              {#each visibleSeries as s}
                <th>{s.name}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each spec.labels as label, idx}
              <tr>
                <td>{label}</td>
                {#each visibleSeries as s}
                  <td>{formatNumber(s.values[idx] ?? 0, spec.unit)}</td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else if activeMode === 'pie'}
      {@const slices = pieSlices(spec)}
      <div class="pie-layout">
        <svg viewBox="0 0 220 192" class="pie-svg" role="img" aria-label={spec.title || t('chat.chart')}>
          {#each slices as slice, idx}
            <path
              d={slice.path}
              fill={slice.color}
              opacity={activeHover === null || activeHover === idx ? 0.94 : 0.45}
              style="transition: opacity .15s"
            >
              <title>{slice.label}: {formatNumber(slice.value, spec.unit)} ({slice.pct}%)</title>
            </path>
          {/each}
        </svg>
        <div class="pie-legend">
          {#each slices as slice, idx}
            <button
              type="button"
              class="pie-row"
              class:highlighted={activeHover === idx}
              onpointerenter={() => setHoverPoint(blockIdx, idx)}
              onfocus={() => setHoverPoint(blockIdx, idx)}
              onclick={() => setHoverPoint(blockIdx, activeHover === idx ? null : idx)}
            >
              <span class="legend-dot" style={`background:${slice.color}`}></span>
              <span class="pie-label">{slice.label}</span>
              <span class="pie-val">{formatNumber(slice.value, spec.unit)} ({slice.pct}%)</span>
            </button>
          {/each}
        </div>
      </div>
    {:else}
      <div
        class="cartesian-wrap"
        role="application"
        aria-label={spec.title ? t('chat.chartPlot', { name: spec.title }) : t('chat.interactivePlot')}
        onpointermove={(e) => handleCartesianPointer(e, blockIdx, count, activeMode === 'bar')}
        onpointerleave={() => setHoverPoint(blockIdx, null)}
      >
        <svg viewBox="0 0 640 236" class="cartesian-svg" role="img" aria-label={spec.title || t('chat.chart')}>
          {#each [0, 0.25, 0.5, 0.75, 1] as step}
            {@const y = padTop + plotH * (1 - step)}
            {@const tickVal = bounds.min + (bounds.max - bounds.min) * step}
            <line x1={padLeft} y1={y} x2={640 - padRight} y2={y} stroke="var(--color-chart-grid)" stroke-dasharray="3 3" />
            <text x={padLeft - 8} y={y + 4} text-anchor="end" class="axis-label">
              {Number.isInteger(tickVal) ? tickVal : tickVal.toFixed(1)}
            </text>
          {/each}

          {#if activeMode === 'bar'}
            {@const groupW = plotW / count}
            {@const barGap = 5}
            {@const singleBarW = Math.max(8, Math.min(42, (groupW - 12 - barGap * (visibleSeries.length - 1)) / visibleSeries.length))}
            {#each spec.labels as label, idx}
              {@const groupStart = padLeft + idx * groupW + (groupW - (singleBarW * visibleSeries.length + barGap * (visibleSeries.length - 1))) / 2}
              {#each visibleSeries as s, sIdx}
                {@const val = s.values[idx] ?? 0}
                {@const zeroY = padTop + plotH * (1 - (0 - bounds.min) / (bounds.max - bounds.min))}
                {@const valY = padTop + plotH * (1 - (val - bounds.min) / (bounds.max - bounds.min))}
                {@const rectY = Math.min(zeroY, valY)}
                {@const rectH = Math.max(3, Math.abs(zeroY - valY))}
                <rect
                  x={groupStart + sIdx * (singleBarW + barGap)}
                  y={rectY}
                  width={singleBarW}
                  height={rectH}
                  rx="4"
                  fill={s.color}
                  opacity={activeHover === null || activeHover === idx ? 0.92 : 0.45}
                  style="transition: opacity .15s"
                >
                  <title>{label} · {s.name}: {formatNumber(val, spec.unit)}</title>
                </rect>
              {/each}
              <text x={padLeft + (idx + 0.5) * groupW} y={236 - 12} text-anchor="middle" class="axis-label">
                {truncateLabel(label, count > 7 ? 8 : 14)}
              </text>
            {/each}
          {:else}
            {#if activeHover !== null && activeHover >= 0 && activeHover < count}
              {@const guideX = count === 1 ? padLeft + plotW / 2 : padLeft + (activeHover / (count - 1)) * plotW}
              <line x1={guideX} y1={padTop} x2={guideX} y2={padTop + plotH} stroke="var(--chat-accent, var(--color-brand))" stroke-opacity="0.45" stroke-dasharray="4 3" />
            {/if}
            {#each visibleSeries as s}
              {@const pts = spec.labels.map((_, idx) => {
                const x = count === 1 ? padLeft + plotW / 2 : padLeft + (idx / (count - 1)) * plotW
                const val = s.values[idx] ?? 0
                const y = padTop + plotH * (1 - (val - bounds.min) / (bounds.max - bounds.min))
                return { x, y, val }
              })}
              {@const polyline = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')}
              {#if activeMode === 'area' && pts.length > 1}
                {@const base = padTop + plotH}
                <polygon
                  points={`${pts[0].x.toFixed(1)},${base} ${polyline} ${pts[pts.length - 1].x.toFixed(1)},${base}`}
                  fill={s.color}
                  opacity="0.2"
                />
              {/if}
              {#if pts.length > 1}
                <polyline fill="none" stroke={s.color} stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points={polyline} />
              {/if}
              {#each pts as pt, idx}
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={activeHover === idx ? 6 : 4}
                  fill={s.color}
                  stroke="var(--color-surface)"
                  stroke-width={activeHover === idx ? 2 : 1.5}
                >
                  <title>{spec.labels[idx]} · {s.name}: {formatNumber(pt.val, spec.unit)}</title>
                </circle>
              {/each}
            {/each}
            {#each spec.labels as label, idx}
              {@const x = count === 1 ? padLeft + plotW / 2 : padLeft + (idx / (count - 1)) * plotW}
              <text {x} y={236 - 12} text-anchor="middle" class="axis-label">
                {truncateLabel(label, count > 7 ? 8 : 14)}
              </text>
            {/each}
          {/if}
        </svg>
      </div>
    {/if}

    <div class="chart-stats-bar">
      <span>{t('chat.points')} <strong>{spec.labels.length}</strong></span>
      <span>{t('chat.min')} <strong>{stats.min}</strong></span>
      <span>{t('chat.peak')} <strong>{stats.max}</strong></span>
      {#if stats.delta}
        <span class="chart-delta" class:neg={stats.delta.startsWith('-')}>{t('chat.change')} <strong>{stats.delta}</strong></span>
      {/if}
    </div>
  </div>
{/snippet}

<div class="rich-message">
  {#if !blocks.length}
    {#if fallback}
      <p class="rich-para">{fallback}</p>
    {/if}
  {:else}
    {#each blocks as block, bIdx}
      {#if block.type === 'heading'}
        {#if block.level === 1}
          <h1 class="rich-h1">{@render renderSpans(block.spans)}</h1>
        {:else if block.level === 2}
          <h2 class="rich-h2">{@render renderSpans(block.spans)}</h2>
        {:else}
          <h3 class="rich-h3">{@render renderSpans(block.spans)}</h3>
        {/if}
      {:else if block.type === 'paragraph'}
        <p class="rich-para">{@render renderSpans(block.spans)}</p>
      {:else if block.type === 'blockquote'}
        <blockquote class="rich-quote">{@render renderSpans(block.spans)}</blockquote>
      {:else if block.type === 'hr'}
        <hr class="rich-hr" />
      {:else if block.type === 'list'}
        {#if block.ordered}
          <ol class="rich-list">
            {#each block.items as item}
              <li>{@render renderSpans(item.spans)}</li>
            {/each}
          </ol>
        {:else}
          <ul class="rich-list">
            {#each block.items as item}
              <li class:task-item={item.checked !== undefined}>
                {#if item.checked !== undefined}
                  <i class={`bi ${item.checked ? 'bi-check2-square' : 'bi-square'} task-icon`} aria-hidden="true"></i>
                {/if}
                <span>{@render renderSpans(item.spans)}</span>
              </li>
            {/each}
          </ul>
        {/if}
      {:else if block.type === 'code'}
        <div class="rich-code-wrap">
          <div class="code-bar">
            <span class="code-lang">{block.lang || t('chat.text')}{block.streaming ? ` · ${t('chat.streamingCode')}` : ''}</span>
            <button type="button" class="copy-code" onclick={() => copyCode(bIdx, block.code)}>
              <i class={`bi ${copiedIndex === bIdx ? 'bi-check2' : 'bi-clipboard'}`} aria-hidden="true"></i>
              {t(copiedIndex === bIdx ? 'chat.copied' : 'chat.copy')}
            </button>
          </div>
          <pre class="rich-pre"><code>{block.code}</code></pre>
        </div>
      {:else if block.type === 'table'}
        {@const activeMode = tableModes[bIdx] ?? 'table'}
        <div class="rich-table-card">
          {#if block.chartable}
            <div class="table-toolbar">
              <span class="table-caption">{t('chat.dataView')}</span>
              <div class="view-switch" role="group" aria-label={t('chat.tableMode')}>
                <button type="button" class:active={activeMode === 'table'} onclick={() => setTableMode(bIdx, 'table')}>{t('chat.table')}</button>
                <button type="button" class:active={activeMode === 'bar'} onclick={() => setTableMode(bIdx, 'bar')}>{t('chat.bar')}</button>
                <button type="button" class:active={activeMode === 'line'} onclick={() => setTableMode(bIdx, 'line')}>{t('chat.line')}</button>
              </div>
            </div>
          {/if}
          {#if block.chartable && (activeMode === 'bar' || activeMode === 'line')}
            {@render renderSvgChart(block.chartable, bIdx, activeMode)}
          {:else}
            <div class="table-scroll">
              <table class="rich-table">
                <thead>
                  <tr>
                    {#each block.headers as header}
                      <th>{@render renderSpans(header)}</th>
                    {/each}
                  </tr>
                </thead>
                <tbody>
                  {#each block.rows as row}
                    <tr>
                      {#each row as cell}
                        <td>{@render renderSpans(cell)}</td>
                      {/each}
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        </div>
      {:else if block.type === 'chart'}
        {@render renderSvgChart(block.spec, bIdx)}
      {:else if block.type === 'kpi'}
        <div class="rich-kpi-wrap">
          {#if block.spec.title}
            <div class="kpi-title">{block.spec.title}</div>
          {/if}
          <div class="kpi-grid">
            {#each block.spec.items as item}
              <div class="kpi-card">
                <span class="kpi-label">{item.label}</span>
                <strong class="kpi-value">{item.value}</strong>
                {#if item.delta}
                  <span class={`kpi-delta tone-${item.tone ?? 'neutral'}`}>{item.delta}</span>
                {/if}
              </div>
            {/each}
          </div>
          {#if block.spec.split}
            {@const sp = block.spec.split}
            <div class="kpi-split">
              <div class="split-track">
                <div class="split-fill" style={`width:${Math.round(sp.ratio * 100)}%`}></div>
              </div>
              <div class="split-labels">
                <span>{sp.leftLabel}: <strong>{sp.leftValue}</strong></span>
                <span>{sp.rightLabel}: <strong>{sp.rightValue}</strong></span>
              </div>
            </div>
          {/if}
        </div>
      {:else if block.type === 'cards'}
        <div class="rich-cards-wrap">
          {#if block.items.length}
            <div class="cards-grid">
              {#each block.items as item}
                <div class="info-card">
                  {#if item.swatch}
                    <div class="card-swatch" style={`background:${item.swatch}`}></div>
                  {/if}
                  <div class="card-content">
                    <div class="card-top">
                      <strong>{item.title}</strong>
                      {#if item.tag}
                        <span class="card-tag">{item.tag}</span>
                      {/if}
                    </div>
                    {#if item.subtitle}
                      <div class="card-sub">{item.subtitle}</div>
                    {/if}
                    {#if item.body}
                      <p class="card-body">{item.body}</p>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          {/if}
          {#if block.swatches?.length}
            <div class="swatch-strip">
              {#each block.swatches as sw}
                <div class="swatch-item">
                  <span class="swatch-dot" style={`background:${sw.hex}`}></span>
                  <span class="swatch-name">{sw.name || sw.hex}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    {/each}
  {/if}
</div>

<style>
  .rich-message {
    display: flex;
    flex-direction: column;
    gap: 12px;
    line-height: 1.72;
    font-size: var(--chat-font-size, 15px);
    color: var(--color-ink);
  }
  .rich-h1, .rich-h2, .rich-h3 {
    margin: 8px 0 2px;
    font-family: var(--font-display);
    color: var(--color-ink);
    line-height: 1.3;
    letter-spacing: -0.015em;
  }
  .rich-h1 { font-size: 1.42em; font-weight: 700; }
  .rich-h2 { font-size: 1.22em; font-weight: 650; }
  .rich-h3 { font-size: 1.07em; font-weight: 600; }
  .rich-para { margin: 0; white-space: pre-wrap; line-height: 1.72; }
  .rich-quote {
    margin: 2px 0;
    padding: 10px 14px;
    border-left: 3px solid var(--chat-accent, var(--color-brand));
    background: color-mix(in srgb, var(--color-surface-muted) 65%, transparent);
    color: var(--color-ink-subtle);
    border-radius: 0 10px 10px 0;
  }
  .rich-hr { border: 0; border-top: 1px solid var(--color-border); margin: 8px 0; }
  .rich-list { margin: 0; padding-left: 22px; display: flex; flex-direction: column; gap: 6px; line-height: 1.68; }
  .rich-list li::marker { color: var(--chat-accent, var(--color-brand)); }
  .task-item { list-style: none; margin-left: -16px; display: flex; align-items: baseline; gap: 7px; }
  .task-icon { color: var(--chat-accent, var(--color-brand)); font-size: 0.9em; }
  .inline-code {
    font-family: var(--font-mono);
    font-size: 0.85em;
    padding: 2px 6px;
    border-radius: 5px;
    background: var(--color-surface-muted);
    border: 1px solid color-mix(in srgb, var(--color-border) 85%, transparent);
  }
  a { color: var(--chat-accent, var(--color-brand)); text-decoration: underline; text-underline-offset: 3px; }

  .rich-code-wrap, .rich-table-card, .rich-chart, .rich-kpi-wrap {
    margin: 2px 0;
    border: 1px solid color-mix(in srgb, var(--color-border) 90%, transparent);
    border-radius: var(--radius-card);
    background: color-mix(in srgb, var(--color-surface-muted) 92%, transparent);
    overflow: hidden;
    box-shadow: var(--card-shadow);
  }
  .code-bar, .table-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 14px;
    border-bottom: 1px solid var(--color-border);
    background: var(--color-surface);
    font-size: 0.78em;
    color: var(--color-ink-subtle);
  }
  .code-lang { font-family: var(--font-mono); text-transform: lowercase; font-weight: 600; }
  .copy-code, .view-switch button {
    border: 1px solid var(--color-border);
    background: transparent;
    color: var(--color-ink-subtle);
    font-size: 0.92em;
    padding: 3px 10px;
    border-radius: 6px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .copy-code:hover, .view-switch button:hover {
    color: var(--color-ink);
    border-color: var(--color-border-strong);
  }
  .view-switch { display: inline-flex; gap: 4px; }
  .view-switch button.active {
    background: var(--color-brand-light);
    color: var(--chat-accent, var(--color-brand));
    border-color: var(--chat-accent, var(--color-brand));
    font-weight: 600;
  }
  .rich-pre {
    margin: 0;
    padding: 13px 15px;
    overflow-x: auto;
    font-family: var(--font-mono);
    font-size: 0.86em;
    line-height: 1.56;
  }

  .table-scroll { overflow-x: auto; }
  .rich-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  .rich-table th, .rich-table td {
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--color-border);
    vertical-align: top;
  }
  .rich-table th {
    background: var(--color-surface);
    font-family: var(--font-display);
    font-weight: 600;
    color: var(--color-ink);
  }
  .rich-table tbody tr:last-child td { border-bottom: 0; }
  .rich-table tbody tr:hover td { background: var(--color-surface-hover); }

  .rich-chart { padding: 14px 16px; background: var(--color-surface); }
  .chart-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .chart-title-wrap { display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; }
  .chart-title { font-family: var(--font-display); font-size: 0.98em; font-weight: 650; }
  .chart-sub { font-size: 0.82em; color: var(--color-ink-subtle); }
  .chart-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .chart-legend { display: flex; gap: 6px; flex-wrap: wrap; font-size: 0.8em; color: var(--color-ink-subtle); }
  .legend-btn {
    border: 1px solid var(--color-border);
    background: var(--color-surface-muted);
    color: var(--color-ink-subtle);
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 0.95em;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
  }
  .legend-btn.muted { opacity: 0.45; text-decoration: line-through; }
  .legend-dot { width: 9px; height: 9px; border-radius: 999px; display: inline-block; flex-shrink: 0; }
  .chart-hover-hud {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
    padding: 6px 10px;
    margin-bottom: 8px;
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--chat-accent, var(--color-brand)) 40%, var(--color-border));
    background: color-mix(in srgb, var(--chat-accent, var(--color-brand)) 8%, var(--color-surface-muted));
    font-size: 0.82em;
  }
  .hud-values { display: flex; gap: 10px; flex-wrap: wrap; font-family: var(--font-mono); }
  .hud-val { display: inline-flex; align-items: center; gap: 5px; }
  .cartesian-svg { width: 100%; height: auto; display: block; }
  .axis-label { font-size: 11px; fill: var(--color-ink-subtle); font-family: var(--font-mono); }
  .chart-stats-bar {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    align-items: center;
    padding-top: 8px;
    margin-top: 8px;
    border-top: 1px solid var(--color-border);
    font-size: 0.78em;
    color: var(--color-ink-subtle);
    font-family: var(--font-mono);
  }
  .chart-stats-bar strong { color: var(--color-ink); }
  .chart-delta strong { color: var(--color-success); }
  .chart-delta.neg strong { color: var(--color-error); }

  .cartesian-wrap { width: 100%; cursor: crosshair; }
  .pie-layout { display: grid; grid-template-columns: minmax(170px, 230px) 1fr; gap: 16px; align-items: center; }
  .pie-svg { width: 100%; max-width: 230px; height: auto; }
  .pie-legend { display: flex; flex-direction: column; gap: 6px; }
  .pie-row {
    width: 100%;
    border: 0;
    background: transparent;
    text-align: left;
    font: inherit;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88em;
    padding: 4px 6px;
    border-radius: 6px;
    cursor: pointer;
  }
  .pie-row.highlighted, .pie-row:hover { background: var(--color-surface-hover); }
  .pie-label { flex: 1; color: var(--color-ink); }
  .pie-val { font-family: var(--font-mono); font-size: 0.84em; color: var(--color-ink-subtle); }

  .rich-kpi-wrap { padding: 14px 16px; background: var(--color-surface); display: flex; flex-direction: column; gap: 12px; }
  .kpi-title { font-family: var(--font-display); font-weight: 650; font-size: 0.96em; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .kpi-card {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--color-border);
    background: var(--color-surface-muted);
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .kpi-label { font-size: 0.78em; color: var(--color-ink-subtle); }
  .kpi-value { font-family: var(--font-display); font-size: 1.45em; font-weight: 700; color: var(--color-ink); letter-spacing: -0.02em; }
  .kpi-delta {
    display: inline-block;
    width: fit-content;
    font-size: 0.76em;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 999px;
  }
  .kpi-delta.tone-positive { color: var(--color-success); background: var(--color-success-bg); }
  .kpi-delta.tone-negative { color: var(--color-error); background: color-mix(in srgb, var(--color-error) 14%, transparent); }
  .kpi-delta.tone-neutral { color: var(--color-ink-subtle); background: var(--color-surface); }
  .kpi-split { display: flex; flex-direction: column; gap: 6px; }
  .split-track { height: 8px; border-radius: 999px; background: var(--color-surface-muted); overflow: hidden; border: 1px solid var(--color-border); }
  .split-fill { height: 100%; background: var(--chat-accent, var(--color-brand)); border-radius: 999px; }
  .split-labels { display: flex; justify-content: space-between; font-size: 0.8em; color: var(--color-ink-subtle); }

  .rich-cards-wrap { display: flex; flex-direction: column; gap: 10px; }
  .cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 10px; }
  .info-card {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 12px 14px;
    border: 1px solid var(--color-border);
    border-radius: 12px;
    background: var(--color-surface);
  }
  .card-swatch { width: 14px; align-self: stretch; border-radius: 6px; flex-shrink: 0; }
  .card-content { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .card-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; font-size: 0.92em; }
  .card-tag { font-size: 0.74em; padding: 2px 7px; border-radius: 999px; background: var(--color-brand-subtle); color: var(--color-brand); border: 1px solid var(--color-border); }
  .card-sub { font-size: 0.8em; color: var(--color-ink-subtle); }
  .card-body { margin: 0; font-size: 0.88em; color: var(--color-ink); line-height: 1.5; }
  .swatch-strip { display: flex; gap: 12px; flex-wrap: wrap; padding: 8px 12px; border: 1px solid var(--color-border); border-radius: 12px; background: var(--color-surface); }
  .swatch-item { display: inline-flex; align-items: center; gap: 6px; font-size: 0.82em; color: var(--color-ink-subtle); }
  .swatch-dot { width: 16px; height: 16px; border-radius: 999px; border: 1px solid var(--color-border-strong); display: inline-block; }

  @media (max-width: 600px) {
    .pie-layout { grid-template-columns: 1fr; }
  }
</style>
