<script lang="ts">
  import { onMount } from 'svelte'
  import { tabStore, setTab, run, hwStore, themeStore, toggleTheme, uiMode, toggleUiMode, pushToast } from '../store.svelte'
  import { t, LOCALES, localeStore, setLocale } from '../i18n.svelte'
  import type { Locale } from '../i18n.svelte'
  import { api } from '../api'

  // Storage (HF cache / model downloads) location
  let cacheDir = $state('')
  let detectedRamMb = $state(0)
  let memoryChoice = $state('auto')

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
    } catch (e) {
      pushToast('error', String(e))
    }
  }

  async function changeMemoryLimit(event: Event) {
    const value = (event.currentTarget as HTMLSelectElement).value
    try {
      const result = await api.setMainMemoryLimit(value === 'auto' ? null : Number(value))
      memoryChoice = result.main_memory_limit_mb?.toString() ?? 'auto'
      pushToast('success', 'Main memory limit saved')
    } catch (e) {
      pushToast('error', String(e))
    }
  }
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
      <div class="grid-2 animate-in">
        <!-- Preferences (Language & Advanced Mode) -->
        <section class="card">
          <div class="card-header">
            <span class="text-label" style="display: flex; align-items: center; gap: var(--space-2)">
              <i class="bi bi-sliders"></i> {t('nav.settings')}
            </span>
          </div>
          <div class="card-body" style="display: flex; flex-direction: column; gap: var(--space-4)">
            <!-- Language Switcher -->
            <div class="settings-item">
              <div class="settings-info">
                <label for="select-language" class="settings-label">{t('sidebar.language')}</label>
                <p class="settings-hint">Choose your preferred language for the interface.</p>
              </div>
              <div class="settings-control">
                <select
                  class="select"
                  value={localeStore.current}
                  onchange={(e) => setLocale((e.currentTarget as HTMLSelectElement).value as Locale)}
                  aria-label={t('sidebar.language')}
                  id="select-language"
                >
                  {#each LOCALES as l}
                    <option value={l.id}>{l.label}</option>
                  {/each}
                </select>
              </div>
            </div>

            <div class="divider"></div>

            <!-- Advanced Mode Toggle -->
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
          </div>
        </section>

        <!-- Model Storage -->
        <section class="card">
          <div class="card-header">
            <span class="text-label" style="display: flex; align-items: center; gap: var(--space-2)">
              <i class="bi bi-folder2"></i> {t('sidebar.storage')}
            </span>
          </div>
          <div class="card-body" style="display: flex; flex-direction: column; gap: var(--space-3)">
            <div class="settings-info">
              <span class="settings-label">{t('sidebar.storage')}</span>
              <p class="settings-hint">{t('sidebar.storagePick')}</p>
            </div>
            <div class="picker-row">
              <input type="text" class="input input-mono" readonly value={cacheDir} style="flex: 1" />
              <button
                class="btn btn-secondary"
                onclick={pickCacheDir}
                id="btn-pick-cache-dir"
                aria-label={t('sidebar.storagePick')}
                style="height: 34px"
              >
                <i class="bi bi-folder2-open" style="font-size: 13px; margin-right: var(--space-2)"></i> Browse
              </button>
            </div>
          </div>
        </section>

        <!-- Memory Limit -->
        <section class="card">
          <div class="card-header">
            <span class="text-label" style="display: flex; align-items: center; gap: var(--space-2)">
              <i class="bi bi-cpu"></i> Memory limit
            </span>
          </div>
          <div class="card-body" style="display: flex; flex-direction: column; gap: var(--space-3)">
            <div class="settings-info">
              <label for="select-memory-limit" class="settings-label">{t('settings.mainMemory')}</label>
              <p class="settings-hint">Limit Sytra's allocation of host RAM. Recommended option is Automatic.</p>
            </div>
            <div class="settings-control full-width">
              <select id="select-memory-limit" class="select" value={memoryChoice} onchange={changeMemoryLimit} aria-label="Main memory limit">
                <option value="auto">Automatic</option>
                {#if detectedRamMb > 0}
                  <option value={Math.floor(detectedRamMb * 0.5)}>50% ({(detectedRamMb * 0.5 / 1024).toFixed(0)} GB)</option>
                  <option value={Math.floor(detectedRamMb * 0.75)}>75% ({(detectedRamMb * 0.75 / 1024).toFixed(0)} GB)</option>
                  <option value={Math.floor(detectedRamMb * 0.9)}>90% ({(detectedRamMb * 0.9 / 1024).toFixed(0)} GB)</option>
                {/if}
              </select>
            </div>
          </div>
        </section>

        <!-- Hardware Info -->
        <section class="card">
          <div class="card-header">
            <span class="text-label" style="display: flex; align-items: center; gap: var(--space-2)">
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
    </div>
  </div>
</div>

<style>
  .settings-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: var(--space-4);
  }
  .settings-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }
  .settings-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--color-ink);
  }
  .settings-hint {
    font-size: 11px;
    color: var(--color-ink-ghost);
    line-height: 1.3;
  }
  .settings-control {
    width: 200px;
    flex-shrink: 0;
  }
  .settings-control.full-width {
    width: 100%;
  }
  .divider {
    height: 1px;
    background: var(--color-border);
  }
  .picker-row {
    display: flex;
    gap: var(--space-2);
    align-items: center;
  }
  .hw-grid {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .hw-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-2) 0;
    border-bottom: 1px dashed var(--color-border);
  }
  .hw-row:last-child {
    border-bottom: none;
  }
  .hw-label {
    font-size: 12px;
    color: var(--color-ink-subtle);
  }
  .hw-val {
    font-size: 12px;
    font-weight: 500;
    font-family: var(--font-mono);
    color: var(--color-ink);
  }
  .badge-info {
    background: var(--color-info-bg);
    color: var(--color-info);
    border: 1px solid var(--color-info);
  }
</style>
