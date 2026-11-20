<script lang="ts">
  import { t } from '../../i18n.svelte'
  import { cardRadii, customThemeStore } from '../../store.svelte'
  import {
    applyLayoutPreset,
    chatAppearance,
    isSafeMediaUrl,
    readImageFileAsDataUrl,
    resetChatAppearance,
    saveChatAppearance,
    type BubbleAlign,
    type ChatLayoutPreset,
  } from './chatCustomizer.svelte'

  interface Props {
    open: boolean
    onClose: () => void
  }

  let { open, onClose }: Props = $props()
  let errorMsg = $state('')
  let bgFileInput = $state<HTMLInputElement | undefined>(undefined)
  let userAvatarInput = $state<HTMLInputElement | undefined>(undefined)
  let assistantAvatarInput = $state<HTMLInputElement | undefined>(undefined)
  const effectiveRadius = $derived(chatAppearance.bubbleRadiusPx ?? cardRadii[customThemeStore.borderRadius] ?? cardRadii.default)
  const effectiveAccent = $derived(chatAppearance.accentHex || customThemeStore.accentColor)

  async function pickImage(
    event: Event,
    target: 'wallpaperUrl' | 'userAvatarUrl' | 'assistantAvatarUrl',
  ) {
    errorMsg = ''
    const input = event.currentTarget as HTMLInputElement
    const file = input.files?.[0]
    input.value = ''
    if (!file) return
    try {
      const { dataUrl } = await readImageFileAsDataUrl(file)
      chatAppearance[target] = dataUrl
      saveChatAppearance()
    } catch (err) {
      errorMsg = String(err)
    }
  }

  function applySafeUrl(
    val: string,
    target: 'wallpaperUrl' | 'userAvatarUrl' | 'assistantAvatarUrl',
  ) {
    if (!isSafeMediaUrl(val)) {
      errorMsg = t('chat.mediaUrlError')
      return
    }
    errorMsg = ''
    chatAppearance[target] = val.trim()
    saveChatAppearance()
  }

  const PRESETS: { id: ChatLayoutPreset; label: string; desc: string }[] = [
    { id: 'standard', label: 'chat.presetStandard', desc: 'chat.presetStandardHint' },
    { id: 'messenger', label: 'chat.presetMessenger', desc: 'chat.presetMessengerHint' },
    { id: 'left', label: 'chat.presetLeft', desc: 'chat.presetLeftHint' },
    { id: 'compact', label: 'chat.presetCompact', desc: 'chat.presetCompactHint' },
  ]
  const ALIGNS: BubbleAlign[] = ['left', 'center', 'right']
  const ACCENTS = ['#e03535', '#2563eb', '#059669', '#d97706', '#7c3aed', '#db2777']
  const PETS: { id: 'pochi' | 'kitsune' | 'mochi' | 'owl'; label: string }[] = [
    { id: 'pochi', label: 'Lil Sytra Bot' },
    { id: 'kitsune', label: 'Kitsune' },
    { id: 'mochi', label: 'Mochi' },
    { id: 'owl', label: 'Owl' },
  ]
</script>

{#if open}
  <section class="customizer-drawer" aria-label={t('chat.customizer')}>
    <div class="drawer-head">
      <div>
        <strong><i class="bi bi-palette2" aria-hidden="true"></i> {t('chat.customizerTitle')}</strong>
        <span>{t('chat.customizerIntro')}</span>
      </div>
      <div class="drawer-actions">
        <button type="button" class="btn btn-ghost btn-sm" onclick={resetChatAppearance}>{t('chat.reset')}</button>
        <button type="button" class="btn btn-secondary btn-sm" onclick={onClose}>{t('chat.doneAction')}</button>
      </div>
    </div>

    {#if errorMsg}
      <p class="custom-error" role="status">{errorMsg}</p>
    {/if}

    <div class="custom-grid">
      <div class="custom-card">
        <span class="card-title">{t('chat.preset')}</span>
        <div class="preset-row">
          {#each PRESETS as p}
            <button
              type="button"
              class="preset-btn"
              class:active={chatAppearance.layoutPreset === p.id}
              onclick={() => applyLayoutPreset(p.id)}
            >
              <strong>{t(p.label)}</strong>
              <small>{t(p.desc)}</small>
            </button>
          {/each}
        </div>
      </div>

      <div class="custom-card">
        <span class="card-title">{t('chat.position')}</span>
        <div class="align-controls">
          <div class="align-row">
            <span>{t('chat.yourMessages')}</span>
            <div class="seg">
              {#each ALIGNS as a}
                <button
                  type="button"
                  class:active={chatAppearance.userAlign === a}
                  onclick={() => { chatAppearance.userAlign = a; saveChatAppearance() }}
                >{t(`chat.${a}`)}</button>
              {/each}
            </div>
          </div>
          <div class="align-row">
            <span>{t('chat.assistantReplies')}</span>
            <div class="seg">
              {#each ALIGNS as a}
                <button
                  type="button"
                  class:active={chatAppearance.assistantAlign === a}
                  onclick={() => { chatAppearance.assistantAlign = a; saveChatAppearance() }}
                >{t(`chat.${a}`)}</button>
              {/each}
            </div>
          </div>
          <div class="align-row">
            <span>{t('chat.inputPosition')}</span>
            <div class="seg">
              <button
                type="button"
                class:active={chatAppearance.composerDock === 'bottom'}
                onclick={() => { chatAppearance.composerDock = 'bottom'; saveChatAppearance() }}
              >{t('chat.bottom')}</button>
              <button
                type="button"
                class:active={chatAppearance.composerDock === 'top'}
                onclick={() => { chatAppearance.composerDock = 'top'; saveChatAppearance() }}
              >{t('chat.top')}</button>
            </div>
          </div>
        </div>
      </div>

      <div class="custom-card">
        <span class="card-title">{t('chat.dimensions')}</span>
        <div class="sliders">
          <label>
            {t('chat.width', { size: chatAppearance.maxWidthPx })}
            <input type="range" min="540" max="1120" step="20" bind:value={chatAppearance.maxWidthPx} oninput={saveChatAppearance} />
          </label>
          <label>
            {t('chat.corners', { size: effectiveRadius })}
            <input type="range" min="0" max="22" step="2" value={effectiveRadius}
              oninput={(event) => { chatAppearance.bubbleRadiusPx = event.currentTarget.valueAsNumber; saveChatAppearance() }} />
          </label>
          <label>
            {t('chat.textSize', { size: chatAppearance.fontSizePx })}
            <input type="range" min="12" max="20" step="1" bind:value={chatAppearance.fontSizePx} oninput={saveChatAppearance} />
          </label>
        </div>
        <div class="swatch-row">
          <span>{t('chat.accent')}</span>
          {#each ACCENTS as hex}
            <button
              type="button"
              class="swatch"
              class:active={effectiveAccent === hex}
              style={`background:${hex}`}
              aria-label={t('chat.accentOption', { color: hex })}
              onclick={() => { chatAppearance.accentHex = hex; saveChatAppearance() }}
            ></button>
          {/each}
          <label class="pet-toggle">
            <input type="checkbox" bind:checked={chatAppearance.showFloatingPet} onchange={saveChatAppearance} />
            {t('chat.petCompanion')}
          </label>
        </div>
        {#if chatAppearance.accentHex || chatAppearance.bubbleRadiusPx !== null}
          <button type="button" class="btn btn-ghost btn-sm" onclick={() => {
            chatAppearance.accentHex = ''
            chatAppearance.bubbleRadiusPx = null
            saveChatAppearance()
          }}>{t('chat.inheritTheme')}</button>
        {/if}
        <div class="swatch-row">
          <span>{t('chat.petLook')}</span>
          <div class="seg">
            {#each PETS as pet}
              <button
                type="button"
                class:active={chatAppearance.petSpecies === pet.id}
                onclick={() => { chatAppearance.petSpecies = pet.id; saveChatAppearance() }}
              >{pet.id === 'owl' ? t('chat.owl') : pet.label}</button>
            {/each}
          </div>
        </div>
      </div>

      <div class="custom-card">
        <span class="card-title">{t('chat.wallpaper')}</span>
        <div class="media-inputs">
          <div class="media-row">
            <input
              type="url"
              class="input"
              placeholder={t('chat.wallpaperUrl')}
              value={chatAppearance.wallpaperUrl}
              onchange={(e) => applySafeUrl((e.currentTarget as HTMLInputElement).value, 'wallpaperUrl')}
            />
            <input bind:this={bgFileInput} type="file" accept="image/*,.gif" style="display:none" onchange={(e) => pickImage(e, 'wallpaperUrl')} />
            <button type="button" class="btn btn-secondary btn-sm" onclick={() => bgFileInput?.click()}>{t('chat.upload')}</button>
            {#if chatAppearance.wallpaperUrl}
              <button type="button" class="btn btn-ghost btn-sm" onclick={() => applySafeUrl('', 'wallpaperUrl')}>{t('chat.clear')}</button>
            {/if}
          </div>
          {#if chatAppearance.wallpaperUrl}
            <div class="sliders">
              <label>
                {t('chat.opacity', { size: Math.round(chatAppearance.wallpaperOpacity * 100) })}
                <input type="range" min="0.05" max="0.85" step="0.05" bind:value={chatAppearance.wallpaperOpacity} oninput={saveChatAppearance} />
              </label>
              <label>
                {t('chat.blur', { size: chatAppearance.wallpaperBlur })}
                <input type="range" min="0" max="16" step="1" bind:value={chatAppearance.wallpaperBlur} oninput={saveChatAppearance} />
              </label>
            </div>
          {/if}
          <div class="media-row">
            <input bind:this={userAvatarInput} type="file" accept="image/*,.gif" style="display:none" onchange={(e) => pickImage(e, 'userAvatarUrl')} />
            <button type="button" class="btn btn-secondary btn-sm" onclick={() => userAvatarInput?.click()}>{t('chat.yourAvatar')}</button>
            {#if chatAppearance.userAvatarUrl}
              <button type="button" class="btn btn-ghost btn-sm" onclick={() => applySafeUrl('', 'userAvatarUrl')}>{t('chat.clear')}</button>
            {/if}
            <input bind:this={assistantAvatarInput} type="file" accept="image/*,.gif" style="display:none" onchange={(e) => pickImage(e, 'assistantAvatarUrl')} />
            <button type="button" class="btn btn-secondary btn-sm" onclick={() => assistantAvatarInput?.click()}>{t('chat.botAvatar')}</button>
            {#if chatAppearance.assistantAvatarUrl}
              <button type="button" class="btn btn-ghost btn-sm" onclick={() => applySafeUrl('', 'assistantAvatarUrl')}>{t('chat.clear')}</button>
            {/if}
          </div>
        </div>
      </div>
    </div>
  </section>
{/if}

<style>
  .customizer-drawer {
    border-bottom: 1px solid var(--color-border);
    background: var(--color-surface);
    padding: 12px 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 46vh;
    overflow: auto;
  }
  .drawer-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  .drawer-head strong { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; margin-right: 8px; }
  .drawer-head span { font-size: 11.5px; color: var(--color-ink-subtle); }
  .drawer-actions { display: flex; gap: 6px; }
  .custom-error { margin: 0; color: var(--color-error); font-size: 12px; }
  .custom-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }
  .custom-card {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-card);
    background: var(--color-surface-muted);
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 12px;
  }
  .card-title { font-weight: 600; font-size: 11.5px; color: var(--color-ink-subtle); text-transform: uppercase; letter-spacing: 0.03em; }
  .preset-row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .preset-btn {
    text-align: left;
    padding: 6px 8px;
    border: 1px solid var(--color-border);
    background: var(--color-surface);
    border-radius: var(--radius-sm);
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .preset-btn strong { font-size: 11.5px; }
  .preset-btn small { font-size: 10.5px; color: var(--color-ink-subtle); }
  .preset-btn.active { border-color: var(--color-brand); background: var(--color-brand-subtle); }
  .align-controls { display: flex; flex-direction: column; gap: 6px; }
  .align-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .seg { display: inline-flex; border: 1px solid var(--color-border); border-radius: var(--radius-sm); overflow: hidden; background: var(--color-surface); }
  .seg button {
    border: 0;
    background: transparent;
    padding: 3px 8px;
    font-size: 11px;
    cursor: pointer;
    color: var(--color-ink-subtle);
    text-transform: capitalize;
  }
  .seg button.active { background: var(--color-brand-light); color: var(--color-brand); font-weight: 600; }
  .sliders { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; }
  .sliders label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: var(--color-ink-subtle); }
  .swatch-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .swatch { width: 16px; height: 16px; border-radius: 999px; border: 2px solid transparent; cursor: pointer; }
  .swatch.active { border-color: var(--color-ink); transform: scale(1.1); }
  .pet-toggle { margin-left: auto; display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; cursor: pointer; }
  .media-inputs { display: flex; flex-direction: column; gap: 7px; }
  .media-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .media-row .input { flex: 1; min-width: 140px; font-size: 11.5px; padding: 5px 8px; }
</style>
