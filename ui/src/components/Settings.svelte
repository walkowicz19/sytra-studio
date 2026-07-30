<script lang="ts">
  import { onMount } from 'svelte'
  import { tabStore, setTab, run, hwStore, themeStore, toggleTheme, uiMode, toggleUiMode, pushToast, customThemeStore, applyCustomTheme } from '../store.svelte'
  import { t, LOCALES, localeStore, setLocale } from '../i18n.svelte'
  import type { Locale } from '../i18n.svelte'
  import { api } from '../api'

  let cacheDir = $state('')
  let detectedRamMb = $state(0)
  let memoryChoice = $state('auto')

  // Visual Studio preview auto-updates live
  $effect(() => {
    // Touch any theme property to trigger re-render
    void customThemeStore.accentColor
    void customThemeStore.effect
    void customThemeStore.bgType
  })

  const ACCENT_PRESETS = [
    { label: 'Sytra Red',      hex: '#e03535' },
    { label: 'Cyber Violet',   hex: '#8b5cf6' },
    { label: 'Neon Cyan',      hex: '#06b6d4' },
    { label: 'Emerald Glow',   hex: '#10b981' },
    { label: 'Amber Flame',    hex: '#f59e0b' },
    { label: 'Pink Neon',      hex: '#ec4899' },
    { label: 'Solar Orange',   hex: '#f97316' },
    { label: 'Ice Blue',       hex: '#38bdf8' },
    { label: 'Lime Acid',      hex: '#a3e635' },
    { label: 'Magenta',        hex: '#e879f9' },
    { label: 'Gold',           hex: '#eab308' },
    { label: 'Coral',          hex: '#fb7185' },
  ]

  const EFFECTS = [
    { id: 'none',          label: 'Standard',       icon: '⬜', desc: 'Clean flat UI' },
    { id: 'glassmorphism', label: 'Glassmorphism',  icon: '🪟', desc: 'Frosted glass cards with blur' },
    { id: 'frosted',       label: 'Frosted',        icon: '❄️', desc: 'Heavy frost, matte surfaces' },
    { id: 'glow',          label: 'Neon Glow',      icon: '✨', desc: 'Luminous brand color glow' },
    { id: 'holographic',   label: 'Holographic',    icon: '🌈', desc: 'Rainbow iridescent shimmer' },
    { id: 'cyberpunk',     label: 'Cyberpunk',      icon: '⚡', desc: 'Hard borders, scanline accents' },
    { id: 'scanlines',     label: 'Scanlines',      icon: '📺', desc: 'CRT scanline overlay' },
    { id: 'matrix',        label: 'Matrix',         icon: '🟩', desc: 'Green code rain aesthetic' },
  ] as const

  const BG_TYPES = [
    { id: 'default',  label: 'Default',          desc: 'System dark/light theme' },
    { id: 'gradient', label: 'Gradient',          desc: 'Linear or radial gradient' },
    { id: 'mesh',     label: 'Mesh Gradient',     desc: 'Multi-point mesh blur gradient' },
    { id: 'aurora',   label: 'Aurora',            desc: 'Northern lights gradient' },
    { id: 'image',    label: 'Custom Image',      desc: 'Any image URL or local path' },
    { id: 'gif',      label: 'Animated GIF',      desc: 'Live animated background' },
  ] as const

  const FONT_PRESETS = [
    { id: 'system',    label: 'System Default',    preview: 'Aa' },
    { id: 'inter',     label: 'Inter',             preview: 'Aa' },
    { id: 'outfit',    label: 'Outfit',            preview: 'Aa' },
    { id: 'geist',     label: 'Geist',             preview: 'Aa' },
    { id: 'jetbrains', label: 'JetBrains Mono',   preview: 'Aa' },
  ] as const

  onMount(async () => {
    try {
      const settings = await api.getSettings()
      cacheDir = settings.hf_cache_dir
      detectedRamMb = settings.detected_ram_mb
      memoryChoice = settings.main_memory_limit_mb?.toString() ?? 'auto'
    } catch {}
  })

  async function pickCacheDir() {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog')
      const picked = await open({ directory: true, title: t('sidebar.storagePick') })
      if (typeof picked === 'string') {
        const res = await api.setCacheDir(picked)
        cacheDir = res.hf_cache_dir
        pushToast('success', t('sidebar.storageSaved'))
      }
    } catch (e) { pushToast('error', String(e)) }
  }

  async function changeMemoryLimit(event: Event) {
    const value = (event.currentTarget as HTMLSelectElement).value
    try {
      const result = await api.setMainMemoryLimit(value === 'auto' ? null : Number(value))
      memoryChoice = result.main_memory_limit_mb?.toString() ?? 'auto'
      pushToast('success', 'Main memory limit saved')
    } catch (e) { pushToast('error', String(e)) }
  }

  function set<K extends keyof typeof customThemeStore>(key: K, val: (typeof customThemeStore)[K]) {
    (customThemeStore as any)[key] = val
    applyCustomTheme()
  }

  function resetTheme() {
    customThemeStore.accentColor = '#e03535'
    customThemeStore.bgType = 'default'
    customThemeStore.bgUrl = ''
    customThemeStore.bgOpacity = 0.15
    customThemeStore.effect = 'none'
    customThemeStore.glassBlur = 16
    customThemeStore.glowIntensity = 60
    customThemeStore.gradientDir = 'radial'
    customThemeStore.gradientSecond = '#1e1b4b'
    customThemeStore.fontFamily = 'system'
    customThemeStore.animSpeed = 'normal'
    customThemeStore.borderRadius = 'default'
    customThemeStore.sidebarBlur = false
    customThemeStore.cardShadow = 'subtle'
    applyCustomTheme()
  }

  function getEffectLabel(id: string): string {
    if (id === 'none') return t('vs.fxStandard')
    if (id === 'glassmorphism') return t('vs.fxGlass')
    if (id === 'frosted') return t('vs.fxFrosted')
    if (id === 'glow') return t('vs.fxGlow')
    if (id === 'holographic') return t('vs.fxHolo')
    if (id === 'cyberpunk') return t('vs.fxCyber')
    if (id === 'scanlines') return t('vs.fxScan')
    if (id === 'matrix') return t('vs.fxMatrix')
    return id
  }

  function getEffectDesc(id: string): string {
    if (id === 'none') return t('vs.fxStandardDesc')
    if (id === 'glassmorphism') return t('vs.fxGlassDesc')
    if (id === 'frosted') return t('vs.fxFrostedDesc')
    if (id === 'glow') return t('vs.fxGlowDesc')
    if (id === 'holographic') return t('vs.fxHoloDesc')
    if (id === 'cyberpunk') return t('vs.fxCyberDesc')
    if (id === 'scanlines') return t('vs.fxScanDesc')
    if (id === 'matrix') return t('vs.fxMatrixDesc')
    return ''
  }

  function getBgLabel(id: string): string {
    if (id === 'default') return t('vs.bgDefault')
    if (id === 'gradient') return t('vs.bgGradient')
    if (id === 'mesh') return t('vs.bgMesh')
    if (id === 'aurora') return t('vs.bgAurora')
    if (id === 'image') return t('vs.bgImage')
    if (id === 'gif') return t('vs.bgGif')
    return id
  }

  function getBgDesc(id: string): string {
    if (id === 'default') return t('vs.bgDefaultDesc')
    if (id === 'gradient') return t('vs.bgGradientDesc')
    if (id === 'mesh') return t('vs.bgMeshDesc')
    if (id === 'aurora') return t('vs.bgAuroraDesc')
    if (id === 'image') return t('vs.bgImageDesc')
    if (id === 'gif') return t('vs.bgGifDesc')
    return ''
  }

  // Active visual studio section tabs
  let vsSection = $state<'colors' | 'effects' | 'background' | 'layout'>('colors')
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">{t('settings.title')}</h1>
      <p class="text-small">{t('settings.subtitle')}</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area">
      <div class="settings-grid animate-in">

        <!-- ── LEFT COLUMN ─────────────────────────────── -->
        <div class="settings-col">

          <!-- Preferences -->
          <section class="card">
            <div class="card-header">
              <span class="text-label" style="display:flex;align-items:center;gap:var(--space-2)">
                <i class="bi bi-sliders"></i> {t('nav.settings')}
              </span>
            </div>
            <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-4)">
              <div class="settings-item">
                <div class="settings-info">
                  <label for="select-language" class="settings-label">{t('sidebar.language')}</label>
                  <p class="settings-hint">Interface language</p>
                </div>
                <div class="settings-control">
                  <select class="select" value={localeStore.current}
                    onchange={(e) => setLocale((e.currentTarget as HTMLSelectElement).value as Locale)}
                    id="select-language">
                    {#each LOCALES as l}
                      <option value={l.id}>{l.label}</option>
                    {/each}
                  </select>
                </div>
              </div>
              <div class="divider"></div>
              <div class="settings-item">
                <div class="settings-info">
                  <span class="settings-label">{t('sidebar.advancedMode')}</span>
                  <p class="settings-hint">{uiMode.advanced ? t('sidebar.allSettings') : t('sidebar.guided')}</p>
                </div>
                <div class="settings-control">
                  <label class="toggle" id="toggle-ui-mode">
                    <input type="checkbox" checked={uiMode.advanced} onchange={toggleUiMode} />
                    <div class="toggle-track"><div class="toggle-thumb"></div></div>
                  </label>
                </div>
              </div>
              <div class="divider"></div>
              <div class="settings-item">
                <div class="settings-info">
                  <span class="settings-label">{t('settings.darkMode')}</span>
                  <p class="settings-hint">{t('settings.darkModeHint')}</p>
                </div>
                <div class="settings-control">
                  <label class="toggle" id="toggle-dark-mode">
                    <input type="checkbox" checked={themeStore.dark} onchange={toggleTheme} />
                    <div class="toggle-track"><div class="toggle-thumb"></div></div>
                  </label>
                </div>
              </div>
            </div>
          </section>

          <!-- Model Storage -->
          <section class="card">
            <div class="card-header">
              <span class="text-label" style="display:flex;align-items:center;gap:var(--space-2)">
                <i class="bi bi-folder2"></i> {t('sidebar.storage')}
              </span>
            </div>
            <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
              <div class="settings-info">
                <span class="settings-label">{t('sidebar.storage')}</span>
                <p class="settings-hint">{t('sidebar.storagePick')}</p>
              </div>
              <div class="picker-row">
                <input type="text" class="input input-mono" readonly value={cacheDir} style="flex:1" />
                <button class="btn btn-secondary" onclick={pickCacheDir} id="btn-pick-cache-dir" style="height:34px">
                  <i class="bi bi-folder2-open" style="font-size:13px;margin-right:var(--space-2)"></i> {t('settings.browse')}
                </button>
              </div>
            </div>
          </section>

          <!-- Memory Limit -->
          <section class="card">
            <div class="card-header">
              <span class="text-label" style="display:flex;align-items:center;gap:var(--space-2)">
                <i class="bi bi-cpu"></i> {t('settings.memoryTitle')}
              </span>
            </div>
            <div class="card-body" style="display:flex;flex-direction:column;gap:var(--space-3)">
              <div class="settings-info">
                <label for="select-memory-limit" class="settings-label">{t('settings.mainMemory')}</label>
                <p class="settings-hint">{t('settings.memoryHint')}</p>
              </div>
              <select id="select-memory-limit" class="select" value={memoryChoice} onchange={changeMemoryLimit}>
                <option value="auto">{t('settings.auto')}</option>
                {#if detectedRamMb > 0}
                  <option value={Math.floor(detectedRamMb * 0.5)}>50% ({(detectedRamMb * 0.5 / 1024).toFixed(0)} GB)</option>
                  <option value={Math.floor(detectedRamMb * 0.75)}>75% ({(detectedRamMb * 0.75 / 1024).toFixed(0)} GB)</option>
                  <option value={Math.floor(detectedRamMb * 0.9)}>90% ({(detectedRamMb * 0.9 / 1024).toFixed(0)} GB)</option>
                {/if}
              </select>
            </div>
          </section>

          <!-- Hardware Info -->
          <section class="card">
            <div class="card-header">
              <span class="text-label" style="display:flex;align-items:center;gap:var(--space-2)">
                <i class="bi bi-info-circle"></i> {t('sidebar.hardware')}
              </span>
            </div>
            <div class="card-body">
              {#if hwStore.info}
                <div class="hw-grid">
                  <div class="hw-row">
                    <span class="hw-label">{t('sidebar.backend')}</span>
                    <span class="badge badge-info">{hwStore.info.backend.toUpperCase()}</span>
                  </div>
                  <div class="hw-row">
                    <span class="hw-label">VRAM</span>
                    <span class="hw-val">{(hwStore.info.vram_mb / 1024).toFixed(1)} GB</span>
                  </div>
                  <div class="hw-row">
                    <span class="hw-label">RAM</span>
                    <span class="hw-val">{(hwStore.info.ram_mb / 1024).toFixed(1)} GB</span>
                  </div>
                </div>
              {:else}
                <span class="text-small">{t('sidebar.detecting')}</span>
              {/if}
            </div>
          </section>
        </div>

        <!-- ── RIGHT COLUMN — Visual Studio ───────────── -->
        <div class="settings-col">
          <section class="card vs-card">
            <div class="card-header vs-header">
              <span class="text-label" style="display:flex;align-items:center;gap:var(--space-2)">
                <i class="bi bi-palette2"></i> {t('vs.title')}
                <span class="vs-badge">{t('vs.fullCustomize')}</span>
              </span>
              <button class="btn-reset" onclick={resetTheme} title="Reset to defaults">
                <i class="bi bi-arrow-counterclockwise"></i> {t('vs.reset')}
              </button>
            </div>

            <!-- Live preview strip -->
            <div class="preview-strip" style="
              background: {customThemeStore.accentColor}18;
              border-bottom: 2px solid {customThemeStore.accentColor}44;
            ">
              <div class="preview-pill" style="background:{customThemeStore.accentColor}">{t('vs.accent')}</div>
              <div class="preview-pill preview-ghost">{t('vs.effectLabel')} <b>{getEffectLabel(customThemeStore.effect)}</b></div>
              <div class="preview-pill preview-ghost">{t('vs.bgLabel')} <b>{getBgLabel(customThemeStore.bgType)}</b></div>
              <div class="preview-mini-card">
                <span style="color:{customThemeStore.accentColor}">✦</span> {t('vs.previewCard')}
              </div>
            </div>

            <!-- Section tabs -->
            <div class="vs-tabs">
              {#each [
                { id: 'colors',     icon: 'bi-palette',    label: t('vs.tabColors') },
                { id: 'effects',    icon: 'bi-stars',      label: t('vs.tabEffects') },
                { id: 'background', icon: 'bi-image',      label: t('vs.tabBackground') },
                { id: 'layout',     icon: 'bi-layout-text-sidebar', label: t('vs.tabLayout') },
              ] as tab}
                <button
                  id="vs-tab-{tab.id}"
                  class="vs-tab"
                  class:vs-tab-active={vsSection === tab.id}
                  onclick={() => vsSection = tab.id as any}
                >
                  <i class="bi {tab.icon}"></i> {tab.label}
                </button>
              {/each}
            </div>

            <div class="vs-body">

              <!-- ═══ COLORS ═══ -->
              {#if vsSection === 'colors'}
                <div class="vs-section">
                  <div class="vs-row-label">{t('vs.accentColor')}</div>
                  <div class="color-swatches">
                    {#each ACCENT_PRESETS as preset}
                      <button
                        class="swatch"
                        class:swatch-active={customThemeStore.accentColor === preset.hex}
                        style="background:{preset.hex}"
                        title={preset.label}
                        onclick={() => set('accentColor', preset.hex)}
                      ></button>
                    {/each}
                    <div class="swatch-custom-wrap" title="Custom HEX">
                      <i class="bi bi-eyedropper swatch-picker-icon"></i>
                      <input
                        type="color"
                        class="color-picker-hidden"
                        value={customThemeStore.accentColor}
                        oninput={(e) => set('accentColor', (e.currentTarget as HTMLInputElement).value)}
                      />
                    </div>
                  </div>

                  <div class="hex-preview-row">
                    <div class="hex-dot" style="background:{customThemeStore.accentColor}"></div>
                    <input class="hex-input" type="text" value={customThemeStore.accentColor}
                      oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; if(/^#[0-9a-f]{6}$/i.test(v)) set('accentColor', v) }}
                    />
                    <span class="hex-label">{t('vs.brandAccent')}</span>
                  </div>

                  <div class="divider"></div>
                  <div class="vs-row-label">{t('vs.gradientSecond')}</div>
                  <p class="settings-hint" style="margin-bottom:8px">{t('vs.gradientSecondHint')}</p>
                  <div class="hex-preview-row">
                    <div class="hex-dot" style="background:{customThemeStore.gradientSecond}"></div>
                    <input class="hex-input" type="text" value={customThemeStore.gradientSecond}
                      oninput={(e) => { const v = (e.currentTarget as HTMLInputElement).value; if(/^#[0-9a-f]{6}$/i.test(v)) set('gradientSecond', v) }}
                    />
                    <input type="color" style="width:28px;height:28px;padding:0;border:none;border-radius:4px;cursor:pointer;background:transparent"
                      value={customThemeStore.gradientSecond}
                      oninput={(e) => set('gradientSecond', (e.currentTarget as HTMLInputElement).value)}
                    />
                  </div>
                </div>

              <!-- ═══ EFFECTS ═══ -->
              {:else if vsSection === 'effects'}
                <div class="vs-section">
                  <div class="vs-row-label">{t('vs.effectPreset')}</div>
                  <div class="effect-grid">
                    {#each EFFECTS as fx}
                      <button
                        id="effect-{fx.id}"
                        class="effect-card"
                        class:effect-card-active={customThemeStore.effect === fx.id}
                        onclick={() => set('effect', fx.id)}
                      >
                        <span class="effect-icon">{fx.icon}</span>
                        <span class="effect-name">{getEffectLabel(fx.id)}</span>
                        <span class="effect-desc">{getEffectDesc(fx.id)}</span>
                      </button>
                    {/each}
                  </div>

                  {#if customThemeStore.effect === 'glassmorphism' || customThemeStore.effect === 'frosted'}
                    <div class="divider" style="margin:12px 0"></div>
                    <div class="vs-row-label">{t('vs.glassBlur')} — {customThemeStore.glassBlur}px</div>
                    <div class="slider-row">
                      <span class="slider-label">0</span>
                      <input type="range" min="0" max="40" step="1" value={customThemeStore.glassBlur}
                        oninput={(e) => set('glassBlur', parseInt((e.currentTarget as HTMLInputElement).value))}
                        class="styled-range" style="--fill:{customThemeStore.accentColor}"
                      />
                      <span class="slider-label">40px</span>
                    </div>
                    <div class="settings-item" style="margin-top:8px">
                      <div class="settings-info">
                        <span class="settings-label">{t('vs.blurSidebar')}</span>
                        <p class="settings-hint">{t('vs.blurSidebarHint')}</p>
                      </div>
                      <label class="toggle">
                        <input type="checkbox" checked={customThemeStore.sidebarBlur}
                          onchange={() => set('sidebarBlur', !customThemeStore.sidebarBlur)} />
                        <div class="toggle-track"><div class="toggle-thumb"></div></div>
                      </label>
                    </div>
                  {/if}

                  {#if customThemeStore.effect === 'glow' || customThemeStore.effect === 'holographic'}
                    <div class="divider" style="margin:12px 0"></div>
                    <div class="vs-row-label">{t('vs.glowIntensity')} — {customThemeStore.glowIntensity}%</div>
                    <div class="slider-row">
                      <span class="slider-label">0</span>
                      <input type="range" min="0" max="100" step="5" value={customThemeStore.glowIntensity}
                        oninput={(e) => set('glowIntensity', parseInt((e.currentTarget as HTMLInputElement).value))}
                        class="styled-range" style="--fill:{customThemeStore.accentColor}"
                      />
                      <span class="slider-label">100%</span>
                    </div>
                  {/if}
                </div>

              <!-- ═══ BACKGROUND ═══ -->
              {:else if vsSection === 'background'}
                <div class="vs-section">
                  <div class="vs-row-label">{t('vs.bgStyle')}</div>
                  <div class="bg-type-grid">
                    {#each BG_TYPES as bg}
                      <button
                        id="bg-{bg.id}"
                        class="bg-card"
                        class:bg-card-active={customThemeStore.bgType === bg.id}
                        onclick={() => set('bgType', bg.id)}
                      >
                        <span class="bg-name">{getBgLabel(bg.id)}</span>
                        <span class="bg-desc">{getBgDesc(bg.id)}</span>
                      </button>
                    {/each}
                  </div>

                  {#if customThemeStore.bgType === 'gradient'}
                    <div class="divider" style="margin:12px 0"></div>
                    <div class="vs-row-label">{t('vs.gradientDir')}</div>
                    <div class="dir-grid">
                      {#each [
                        { id: 'top',          label: '↑ Top' },
                        { id: 'top-right',    label: '↗ Diagonal' },
                        { id: 'right',        label: '→ Right' },
                        { id: 'bottom-right', label: '↘ Diagonal' },
                        { id: 'bottom',       label: '↓ Bottom' },
                        { id: 'radial',       label: '⊙ Radial' },
                      ] as dir}
                        <button
                          class="dir-btn"
                          class:dir-btn-active={customThemeStore.gradientDir === dir.id}
                          onclick={() => set('gradientDir', dir.id as any)}
                        >{dir.label}</button>
                      {/each}
                    </div>
                  {/if}

                  {#if customThemeStore.bgType === 'image' || customThemeStore.bgType === 'gif'}
                    <div class="divider" style="margin:12px 0"></div>
                    <div class="vs-row-label">{t('vs.imageUrl')}</div>
                    <input type="text" class="input input-mono" style="width:100%;margin-bottom:8px"
                      placeholder="https://example.com/wallpaper.gif"
                      value={customThemeStore.bgUrl}
                      oninput={(e) => set('bgUrl', (e.currentTarget as HTMLInputElement).value)}
                    />
                  {/if}

                  {#if customThemeStore.bgType !== 'default'}
                    <div class="divider" style="margin:12px 0"></div>
                    <div class="vs-row-label">{t('vs.bgOpacity')} — {Math.round(customThemeStore.bgOpacity * 100)}%</div>
                    <div class="slider-row">
                      <span class="slider-label">0%</span>
                      <input type="range" min="0" max="0.9" step="0.05" value={customThemeStore.bgOpacity}
                        oninput={(e) => set('bgOpacity', parseFloat((e.currentTarget as HTMLInputElement).value))}
                        class="styled-range" style="--fill:{customThemeStore.accentColor}"
                      />
                      <span class="slider-label">90%</span>
                    </div>
                  {/if}
                </div>

              <!-- ═══ LAYOUT ═══ -->
              {:else if vsSection === 'layout'}
                <div class="vs-section">
                  <div class="vs-row-label">{t('vs.fontFamily')}</div>
                  <div class="font-grid">
                    {#each FONT_PRESETS as fp}
                      <button
                        id="font-{fp.id}"
                        class="font-card"
                        class:font-card-active={customThemeStore.fontFamily === fp.id}
                        onclick={() => set('fontFamily', fp.id)}
                      >
                        <span class="font-preview" style="font-family:{fp.id === 'system' ? 'system-ui' : fp.id === 'jetbrains' ? 'monospace' : fp.label}">{fp.preview}</span>
                        <span class="font-name">{fp.id === 'system' ? t('vs.fontSystem') : fp.label}</span>
                      </button>
                    {/each}
                  </div>

                  <div class="divider" style="margin:12px 0"></div>
                  <div class="vs-row-label">{t('vs.borderRadius')}</div>
                  <div class="radius-grid">
                    {#each [
                      { id: 'sharp',   label: t('vs.radiusSharp'),   style: 'border-radius:2px' },
                      { id: 'default', label: t('vs.radiusDefault'), style: 'border-radius:6px' },
                      { id: 'rounded', label: t('vs.radiusRounded'), style: 'border-radius:12px' },
                    ] as r}
                      <button
                        class="radius-btn"
                        class:radius-btn-active={customThemeStore.borderRadius === r.id}
                        onclick={() => set('borderRadius', r.id as any)}
                      >
                        <div class="radius-preview" style="{r.style};background:{customThemeStore.accentColor}22;border:2px solid {customThemeStore.accentColor}55"></div>
                        <span>{r.label}</span>
                      </button>
                    {/each}
                  </div>

                  <div class="divider" style="margin:12px 0"></div>
                  <div class="vs-row-label">{t('vs.cardShadow')}</div>
                  <div class="shadow-grid">
                    {#each [
                      { id: 'none',     label: t('vs.shadowNone') },
                      { id: 'subtle',   label: t('vs.shadowSubtle') },
                      { id: 'elevated', label: t('vs.shadowElevated') },
                      { id: 'neon',     label: t('vs.shadowNeon') },
                    ] as s}
                      <button
                        class="shadow-btn"
                        class:shadow-btn-active={customThemeStore.cardShadow === s.id}
                        onclick={() => set('cardShadow', s.id as any)}
                      >{s.label}</button>
                    {/each}
                  </div>

                  <div class="divider" style="margin:12px 0"></div>
                  <div class="vs-row-label">Animation Speed</div>
                  <div class="anim-grid">
                    {#each [
                      { id: 'off',    label: 'Off' },
                      { id: 'slow',   label: 'Slow' },
                      { id: 'normal', label: 'Normal' },
                      { id: 'fast',   label: 'Fast' },
                    ] as a}
                      <button
                        class="anim-btn"
                        class:anim-btn-active={customThemeStore.animSpeed === a.id}
                        onclick={() => set('animSpeed', a.id as any)}
                      >{a.label}</button>
                    {/each}
                  </div>
                </div>
              {/if}

            </div>
          </section>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .settings-grid {
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: var(--space-4);
    align-items: start;
  }
  @media (max-width: 900px) {
    .settings-grid { grid-template-columns: 1fr; }
  }
  .settings-col {
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
  }
  .settings-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-4);
  }
  .settings-info { display: flex; flex-direction: column; gap: 2px; flex: 1; }
  .settings-label { font-size: 13px; font-weight: 600; color: var(--color-ink); }
  .settings-hint { font-size: 11px; color: var(--color-ink-ghost); line-height: 1.3; }
  .settings-control { width: 200px; flex-shrink: 0; }
  .divider { height: 1px; background: var(--color-border); }
  .picker-row { display: flex; gap: var(--space-2); align-items: center; }
  .hw-grid { display: flex; flex-direction: column; gap: var(--space-2); }
  .hw-row { display: flex; align-items: center; justify-content: space-between; padding: var(--space-2) 0; border-bottom: 1px dashed var(--color-border); }
  .hw-row:last-child { border-bottom: none; }
  .hw-label { font-size: 12px; color: var(--color-ink-subtle); }
  .hw-val { font-size: 12px; font-weight: 500; font-family: var(--font-mono); color: var(--color-ink); }
  .badge-info { background: var(--color-info-bg); color: var(--color-info); border: 1px solid var(--color-info); }

  /* ── Visual Studio ─────────────────────────────────────────────────── */
  .vs-card { overflow: hidden; }
  .vs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .vs-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 7px;
    border-radius: 20px;
    background: var(--color-brand);
    color: #fff;
    margin-left: 6px;
  }
  .btn-reset {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--color-ink-subtle);
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 3px 10px;
    cursor: pointer;
    transition: color 0.15s, border-color 0.15s;
  }
  .btn-reset:hover { color: var(--color-brand); border-color: var(--color-brand); }

  /* Preview strip */
  .preview-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    flex-wrap: wrap;
  }
  .preview-pill {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    color: #fff;
    white-space: nowrap;
  }
  .preview-ghost {
    background: var(--color-surface-raised);
    color: var(--color-ink-subtle);
    border: 1px solid var(--color-border);
  }
  .preview-mini-card {
    margin-left: auto;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 11px;
    display: flex;
    align-items: center;
    gap: 5px;
  }

  /* Section tabs */
  .vs-tabs {
    display: flex;
    border-bottom: 1px solid var(--color-border);
    padding: 0 16px;
    gap: 2px;
  }
  .vs-tab {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    font-weight: 500;
    padding: 8px 12px;
    border: none;
    background: transparent;
    color: var(--color-ink-subtle);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s;
  }
  .vs-tab:hover { color: var(--color-ink); }
  .vs-tab-active { color: var(--color-brand); border-bottom-color: var(--color-brand); }

  .vs-body { padding: 16px; }
  .vs-section { display: flex; flex-direction: column; gap: 8px; }
  .vs-row-label { font-size: 12px; font-weight: 700; color: var(--color-ink); letter-spacing: 0.03em; }

  /* Color swatches */
  .color-swatches { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
  .swatch {
    width: 26px; height: 26px;
    border-radius: 50%;
    border: 2px solid transparent;
    cursor: pointer;
    transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
    flex-shrink: 0;
  }
  .swatch:hover { transform: scale(1.2); box-shadow: 0 0 0 3px rgba(255,255,255,0.2); }
  .swatch-active { border-color: #fff; transform: scale(1.2); box-shadow: 0 0 0 3px rgba(255,255,255,0.35); }
  .swatch-custom-wrap {
    width: 26px; height: 26px;
    border-radius: 50%;
    border: 2px dashed var(--color-border);
    cursor: pointer;
    position: relative;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .swatch-custom-wrap:hover { border-color: var(--color-brand); }
  .swatch-picker-icon { font-size: 11px; color: var(--color-ink-subtle); pointer-events: none; }
  .color-picker-hidden {
    position: absolute; inset: 0;
    opacity: 0; width: 100%; height: 100%;
    cursor: pointer;
  }
  .hex-preview-row {
    display: flex; align-items: center; gap: 8px; margin-top: 4px;
  }
  .hex-dot { width: 18px; height: 18px; border-radius: 50%; flex-shrink: 0; }
  .hex-input {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--color-surface-raised);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 3px 8px;
    color: var(--color-ink);
    width: 90px;
    outline: none;
    transition: border-color 0.15s;
  }
  .hex-input:focus { border-color: var(--color-brand); }
  .hex-label { font-size: 11px; color: var(--color-ink-ghost); }

  /* Effects */
  .effect-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
    margin-top: 4px;
  }
  .effect-card {
    display: flex; flex-direction: column; align-items: center;
    gap: 4px; padding: 10px 6px;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    background: var(--color-surface-raised);
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s, transform 0.15s;
  }
  .effect-card:hover { border-color: var(--color-brand); transform: translateY(-1px); }
  .effect-card-active { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .effect-icon { font-size: 20px; }
  .effect-name { font-size: 10px; font-weight: 700; color: var(--color-ink); text-align: center; }
  .effect-desc { font-size: 9px; color: var(--color-ink-ghost); text-align: center; line-height: 1.3; }

  /* Background */
  .bg-type-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin-top: 4px;
  }
  .bg-card {
    display: flex; flex-direction: column;
    padding: 10px; gap: 3px;
    border: 1px solid var(--color-border);
    border-radius: 8px;
    background: var(--color-surface-raised);
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .bg-card:hover { border-color: var(--color-brand); }
  .bg-card-active { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .bg-name { font-size: 11px; font-weight: 700; color: var(--color-ink); }
  .bg-desc { font-size: 9px; color: var(--color-ink-ghost); line-height: 1.3; }

  .dir-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 5px; margin-top: 4px;
  }
  .dir-btn {
    padding: 6px; font-size: 11px; font-weight: 500;
    border: 1px solid var(--color-border); border-radius: 6px;
    background: var(--color-surface-raised); color: var(--color-ink);
    cursor: pointer; transition: border-color 0.15s, background 0.15s;
    text-align: center;
  }
  .dir-btn:hover { border-color: var(--color-brand); }
  .dir-btn-active { border-color: var(--color-brand); background: var(--color-brand-subtle); color: var(--color-brand); font-weight: 700; }

  /* Slider */
  .slider-row { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
  .slider-label { font-size: 10px; color: var(--color-ink-ghost); min-width: 28px; }
  .styled-range {
    flex: 1; -webkit-appearance: none; appearance: none;
    height: 4px; border-radius: 2px;
    background: var(--color-border);
    outline: none; cursor: pointer;
  }
  .styled-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--color-brand);
    box-shadow: 0 0 0 2px var(--color-brand-subtle);
    cursor: pointer;
    transition: transform 0.1s;
  }
  .styled-range::-webkit-slider-thumb:hover { transform: scale(1.2); }

  /* Layout tab */
  .font-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 5px; margin-top: 4px;
  }
  .font-card {
    display: flex; flex-direction: column; align-items: center;
    padding: 8px 4px; gap: 4px;
    border: 1px solid var(--color-border); border-radius: 7px;
    background: var(--color-surface-raised);
    cursor: pointer; transition: border-color 0.15s, background 0.15s;
  }
  .font-card:hover { border-color: var(--color-brand); }
  .font-card-active { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .font-preview { font-size: 18px; font-weight: 700; color: var(--color-ink); }
  .font-name { font-size: 8px; color: var(--color-ink-ghost); text-align: center; }

  .radius-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 6px; margin-top: 4px;
  }
  .radius-btn {
    display: flex; flex-direction: column; align-items: center;
    padding: 10px 6px; gap: 6px;
    border: 1px solid var(--color-border); border-radius: 8px;
    background: var(--color-surface-raised);
    cursor: pointer; transition: border-color 0.15s, background 0.15s;
    font-size: 11px; color: var(--color-ink-subtle);
  }
  .radius-btn:hover { border-color: var(--color-brand); }
  .radius-btn-active { border-color: var(--color-brand); background: var(--color-brand-subtle); color: var(--color-brand); font-weight: 600; }
  .radius-preview { width: 32px; height: 18px; }

  .shadow-grid, .anim-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 5px; margin-top: 4px;
  }
  .shadow-btn, .anim-btn {
    padding: 7px; font-size: 11px; font-weight: 500;
    border: 1px solid var(--color-border); border-radius: 6px;
    background: var(--color-surface-raised); color: var(--color-ink);
    cursor: pointer; transition: border-color 0.15s, background 0.15s;
    text-align: center;
  }
  .shadow-btn:hover, .anim-btn:hover { border-color: var(--color-brand); }
  .shadow-btn-active, .anim-btn-active {
    border-color: var(--color-brand);
    background: var(--color-brand-subtle);
    color: var(--color-brand);
    font-weight: 700;
  }
</style>
