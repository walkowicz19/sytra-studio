<script lang="ts">
  /**
   * LiveView — live training telemetry panel with uPlot charts + log console
   */
  import { onMount, onDestroy } from 'svelte'
  import uPlot from 'uplot'
  import 'uplot/dist/uPlot.min.css'
  import { run, resetRun, pushToast, themeStore } from '../store.svelte'
  import { t } from '../i18n.svelte'
  import { api } from '../api'

  let lossEl   = $state<HTMLDivElement | null>(null)
  let lrEl     = $state<HTMLDivElement | null>(null)
  let logEl    = $state<HTMLDivElement | null>(null)
  let lossPlot: uPlot | null = null
  let lrPlot:   uPlot | null = null

  // Chart colour tokens — Sytra red accent on a monochrome base
  const dark   = () => themeStore.dark
  const BRAND  = () => '#e03535'
  const BLUE   = () => dark() ? '#a8a8a0' : '#5a5a55'
  const GRID   = () => dark() ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)'
  const GHOST  = () => dark() ? '#6e6e68' : '#8a8a84'

  function makeOpts(label: string, color: string, container: HTMLDivElement): uPlot.Options {
    const w = container.clientWidth || 300
    const font = '10px "DM Sans", sans-serif'
    return {
      width: w, height: 110,
      class: 'sytra-chart',
      cursor: { show: true, x: true, y: false },
      legend: { show: false },
      axes: [
        {
          stroke: GHOST(), ticks: { stroke: GRID() }, grid: { stroke: GRID() },
          font,
          label: 'Step', labelFont: font, labelSize: 16,
        },
        {
          stroke: GHOST(), ticks: { stroke: GRID() }, grid: { stroke: GRID() },
          font,
          label, labelFont: font, labelSize: 16,
          size: 50,
        },
      ],
      series: [
        {},
        {
          stroke: color, width: 1.5,
          fill: color + '14',
          points: { show: false },
        },
      ],
      scales: {
        x: { time: false },
        y: { auto: true },
      },
    }
  }

  function buildData(points: { step: number; value: number }[]): uPlot.AlignedData {
    if (!points.length) return [[0], [null]] as unknown as uPlot.AlignedData
    return [
      points.map(p => p.step),
      points.map(p => p.value),
    ] as uPlot.AlignedData
  }

  // Mount charts
  onMount(() => {
    if (lossEl) {
      lossPlot = new uPlot(makeOpts('Loss', BRAND(), lossEl), buildData(run.loss), lossEl)
    }
    if (lrEl) {
      lrPlot = new uPlot(makeOpts('LR', BLUE(), lrEl), buildData(run.lr), lrEl)
    }
  })

  // Reactive updates
  $effect(() => {
    if (lossPlot && run.loss.length) lossPlot.setData(buildData(run.loss))
    if (lrPlot && run.lr.length) lrPlot.setData(buildData(run.lr))
    // Auto-scroll log
    if (logEl) logEl.scrollTop = logEl.scrollHeight
  })

  onDestroy(() => { lossPlot?.destroy(); lrPlot?.destroy() })

  // Stop shortcut — status flips immediately so the user can start a new
  // operation without waiting for the backend channel to drain.
  async function stop() {
    if (!run.opId) return
    try {
      await api.stopOp(run.opId)
      run.status = 'stopped'
      pushToast('info', t('live.opStopped'))
    } catch (e) {
      pushToast('error', String(e))
    }
  }

  function formatDuration(ms: number) {
    const s = Math.floor(ms / 1000)
    if (s < 60)  return `${s}s`
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  }

  let elapsed = $state('')
  let timer: ReturnType<typeof setInterval>
  $effect(() => {
    clearInterval(timer)
    if (run.startedAt && run.status === 'running') {
      timer = setInterval(() => {
        elapsed = formatDuration(Date.now() - (run.startedAt ?? 0))
      }, 1000)
    }
  })

  const logLevelClass: Record<string, string> = {
    error: 'log-error', warning: 'log-warn', warn: 'log-warn',
  }
</script>

<div class="live-view">
  <!-- Top bar -->
  <div class="live-bar">
    <div class="live-status">
      {#if run.status === 'running'}
        <span class="pulse-dot"></span>
        <span class="status-label">{t('live.running')}</span>
        {#if run.stage}
          <span class="meta-chip">{run.stage.replace(/_/g, ' ')}</span>
        {/if}
        <span class="text-small" style="margin-left:var(--space-2)">{elapsed}</span>
      {:else if run.status === 'done'}
        <span class="badge badge-success">✓ {t('live.done')}</span>
      {:else if run.status === 'stopped'}
        <span class="badge badge-neutral">⏹ {t('live.stopped')}</span>
      {:else if run.status === 'error'}
        <span class="badge badge-error">✕ {t('live.error')}</span>
      {/if}
    </div>

    <!-- Progress -->
    {#if run.progress > 0}
      <div class="progress-wrap">
        <div class="progress-bar" style="width:180px">
          <div class="progress-fill" style="width:{(run.progress * 100).toFixed(1)}%"></div>
        </div>
        <span class="text-small">{(run.progress * 100).toFixed(0)}%</span>
      </div>
    {/if}

    <!-- Step / Epoch -->
    <div class="live-meta">
      {#if run.step}<span class="meta-chip">Step {run.step}</span>{/if}
      {#if run.epoch}<span class="meta-chip">Epoch {run.epoch}</span>{/if}
      {#if run.loss.length}
        <span class="meta-chip brand">
          loss {run.loss.at(-1)?.value.toFixed(4)}
        </span>
      {/if}
    </div>

    <div style="flex:1"></div>

    {#if run.status === 'running'}
      <button class="btn btn-secondary btn-sm" onclick={stop} id="btn-live-stop">⏹ {t('live.stop')}</button>
    {:else}
      <button class="btn btn-ghost btn-sm" onclick={resetRun} id="btn-live-dismiss">✕ {t('live.dismiss')}</button>
    {/if}
  </div>

  <!-- Charts + log -->
  <div class="live-body">
    <div class="charts">
      <div class="chart-wrap">
        <div class="text-label chart-title">{t('live.loss')}</div>
        <div bind:this={lossEl}></div>
      </div>
      <div class="chart-wrap">
        <div class="text-label chart-title">{t('live.lr')}</div>
        <div bind:this={lrEl}></div>
      </div>
    </div>

    <div class="log-console" bind:this={logEl} id="log-console">
      {#if run.logLines.length === 0}
        <span class="log-placeholder">{t('live.waiting')}</span>
      {:else}
        {#each run.logLines as line, i (i)}
          <div class="log-line {logLevelClass[line.level ?? ''] ?? ''}">
            {#if line.type === 'metric'}
              <span class="log-dim">[step {line.step}]</span>
              {#if line.loss != null}
                <span>loss=<b>{line.loss.toFixed(4)}</b></span>
              {/if}
              {#if line.lr != null}
                <span class="log-dim"> lr={line.lr.toExponential(2)}</span>
              {/if}
              {#if line.grad_norm != null}
                <span class="log-dim"> grad_norm={line.grad_norm.toFixed(3)}</span>
              {/if}
            {:else if line.type === 'event'}
              <span class="log-event">● {line.event}</span>
            {:else}
              <span class="log-dim">{line.ts ?? ''}</span>
              <span>{line.line || line.message || ''}</span>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
  </div>
</div>

<style>
  .live-view {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Top bar */
  .live-bar {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-2) var(--space-5);
    border-bottom: 1px solid var(--color-border);
    background: var(--color-surface);
    flex-shrink: 0;
    min-height: 40px;
  }
  .live-status { display: flex; align-items: center; gap: var(--space-2); }
  .status-label { font-size: 13px; font-weight: 500; color: var(--color-brand); }
  .progress-wrap { display: flex; align-items: center; gap: var(--space-2); }
  .live-meta { display: flex; gap: var(--space-2); }
  .meta-chip {
    font-size: 11px; font-family: var(--font-mono);
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    padding: 2px 6px;
    color: var(--color-ink-subtle);
  }
  .meta-chip.brand { color: var(--color-brand); background: var(--color-brand-light); }

  /* Body */
  .live-body {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .charts {
    display: flex;
    flex-direction: column;
    gap: 0;
    border-right: 1px solid var(--color-border);
    width: 560px;
    flex-shrink: 0;
    overflow: hidden;
  }
  .chart-wrap {
    flex: 1;
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--color-border);
    overflow: hidden;
  }
  .chart-wrap:last-child { border-bottom: none; }
  .chart-title { margin-bottom: var(--space-1); }

  /* Log — always a near-black terminal, regardless of theme */
  .log-console {
    flex: 1;
    overflow-y: auto;
    padding: var(--space-3) var(--space-4);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.8;
    background: #050505;
    color: #a8a8a0;
    border-top: 1px solid var(--color-border);
  }
  .log-line { display: flex; gap: var(--space-2); flex-wrap: wrap; }
  .log-dim { color: #55554f; }
  .log-event { color: #e03535; font-weight: 500; }
  .log-error { color: #d38b82; }
  .log-warn  { color: #cfa96a; }
  .log-placeholder { color: #55554f; font-style: italic; }

  /* uPlot overrides */
  :global(.sytra-chart) {
    background: transparent !important;
  }
  :global(.sytra-chart .u-cursor-x) {
    border-color: var(--color-brand) !important;
    opacity: 0.5;
  }
</style>
