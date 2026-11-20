<script lang="ts">
  import { t } from '../../i18n.svelte'
  import type { AgentAvatarKind } from '../../types'

  export type PetMood = 'idle' | 'thinking' | 'working' | 'speaking' | 'compacting'

  interface Props {
    species?: AgentAvatarKind
    customUrl?: string | null
    mood?: PetMood
    name?: string
    accent?: string
    size?: number
    interactive?: boolean
    onclick?: () => void
  }

  let {
    species = 'pochi',
    customUrl = null,
    mood = 'idle',
    name = 'Lil Sytra',
    accent = 'var(--color-brand)',
    size = 42,
    interactive = true,
    onclick,
  }: Props = $props()

  let happyBoost = $state(false)
  const moodLabel = $derived(t(`chat.petMood.${mood}`))

  function handlePoke() {
    happyBoost = true
    setTimeout(() => {
      happyBoost = false
    }, 650)
    onclick?.()
  }
</script>

<button
  type="button"
  class={`pet-stage mood-${mood}`}
  class:happy={happyBoost}
  class:static-pet={!interactive}
  style={`--pet-size:${size}px; --pet-accent:${accent}`}
  title={`${name} · ${moodLabel}`}
  aria-label={t('chat.petA11y', { name, mood: moodLabel })}
  onclick={interactive ? handlePoke : undefined}
>
  {#if mood === 'thinking'}
    <span class="pet-badge badge-think" aria-hidden="true"><i class="bi bi-lightbulb-fill"></i></span>
  {:else if mood === 'working'}
    <span class="pet-badge badge-work" aria-hidden="true"><i class="bi bi-hammer"></i></span>
  {:else if mood === 'compacting'}
    <span class="pet-badge badge-compact" aria-hidden="true"><i class="bi bi-stars"></i></span>
  {/if}

  {#if species === 'custom' && customUrl}
    <img class="pet-custom-img" src={customUrl} alt={name} />
  {:else if species === 'kitsune'}
    <svg viewBox="4 2 56 56" class="pet-svg" aria-hidden="true">
      <path class="tail" d="M47 45 C58 43, 60 28, 52 25 C48 31, 46 38, 43 44 Z" fill={accent} opacity="0.88" />
      <polygon class="ear-l" points="16,24 12,8 26,17" fill="#d97706" />
      <polygon class="ear-r" points="48,24 52,8 38,17" fill="#d97706" />
      <polygon points="17,22 15,12 24,18" fill="#fef3c7" />
      <polygon points="47,22 49,12 40,18" fill="#fef3c7" />
      <ellipse cx="32" cy="36" rx="18" ry="16" fill="#ea580c" />
      <path d="M18 39 Q32 51 46 39 Q39 52 32 52 Q25 52 18 39 Z" fill="#fffbeb" />
      <g class="eyes">
        <ellipse cx="25" cy="34" rx="2.4" ry="3.2" fill="#18181b" />
        <ellipse cx="39" cy="34" rx="2.4" ry="3.2" fill="#18181b" />
        <circle cx="25.8" cy="32.8" r="0.9" fill="#ffffff" />
        <circle cx="39.8" cy="32.8" r="0.9" fill="#ffffff" />
      </g>
      <circle cx="32" cy="39" r="2" fill="#18181b" />
      <ellipse cx="20" cy="38" rx="3" ry="1.6" fill="#fb7185" opacity="0.55" />
      <ellipse cx="44" cy="38" rx="3" ry="1.6" fill="#fb7185" opacity="0.55" />
    </svg>
  {:else if species === 'mochi'}
    <svg viewBox="4 6 56 56" class="pet-svg" aria-hidden="true">
      <ellipse cx="32" cy="54" rx="18" ry="3.5" fill="rgba(0,0,0,0.22)" />
      <path class="jelly" d="M13 42 C13 21, 21 15, 32 15 C43 15, 51 21, 51 42 C51 49, 44 52, 32 52 C20 52, 13 49, 13 42 Z" fill={accent} opacity="0.9" />
      <ellipse cx="23" cy="24" rx="4.5" ry="2.5" transform="rotate(-22 23 24)" fill="#ffffff" opacity="0.45" />
      <g class="eyes">
        <circle cx="25" cy="35" r="2.6" fill="#ffffff" />
        <circle cx="39" cy="35" r="2.6" fill="#ffffff" />
        <circle cx="25.5" cy="35" r="1.3" fill="#09090b" />
        <circle cx="39.5" cy="35" r="1.3" fill="#09090b" />
      </g>
      <path class="mouth" d="M29 40 Q32 43 35 40" stroke="#ffffff" stroke-width="2" fill="none" stroke-linecap="round" />
    </svg>
  {:else if species === 'owl'}
    <svg viewBox="4 4 56 56" class="pet-svg" aria-hidden="true">
      <polygon points="16,20 12,10 23,16" fill="#475569" />
      <polygon points="48,20 52,10 41,16" fill="#475569" />
      <ellipse cx="32" cy="36" rx="19" ry="19" fill="#334155" />
      <ellipse cx="32" cy="43" rx="12" ry="10" fill="#cbd5e1" opacity="0.88" />
      <circle cx="24" cy="30" r="8" fill="#f8fafc" stroke={accent} stroke-width="2" />
      <circle cx="40" cy="30" r="8" fill="#f8fafc" stroke={accent} stroke-width="2" />
      <line x1="30" y1="30" x2="34" y2="30" stroke={accent} stroke-width="2" />
      <g class="eyes">
        <circle cx="24" cy="30" r="3" fill="#0f172a" />
        <circle cx="40" cy="30" r="3" fill="#0f172a" />
        <circle cx="25" cy="29" r="1" fill="#ffffff" />
        <circle cx="41" cy="29" r="1" fill="#ffffff" />
      </g>
      <polygon points="32,33 29,37 35,37" fill="#f59e0b" />
    </svg>
  {:else}
    <!-- Lil Sytra Bot -->
    <svg viewBox="4 -1 56 56" class="pet-svg" aria-hidden="true">
      <g class="antenna">
        <line x1="32" y1="14" x2="32" y2="6" stroke="var(--color-border-strong)" stroke-width="2.5" stroke-linecap="round" />
        <circle cx="32" cy="6" r="3.5" fill={accent} />
      </g>
      <rect x="13" y="16" width="38" height="33" rx="12" fill="#18181b" stroke={accent} stroke-width="2.2" />
      <rect x="18" y="22" width="28" height="19" rx="7" fill="#09090b" />
      <g class="eyes">
        <rect x="22" y="28" width="5" height="6" rx="2.5" fill="#fb7185" />
        <rect x="37" y="28" width="5" height="6" rx="2.5" fill="#fb7185" />
        <circle cx="23.5" cy="29.5" r="1.2" fill="#ffffff" />
        <circle cx="38.5" cy="29.5" r="1.2" fill="#ffffff" />
      </g>
      <path class="mouth" d="M28 36 Q32 39 36 36" stroke="#fda4af" stroke-width="1.8" fill="none" stroke-linecap="round" />
      <circle cx="11" cy="34" r="3.5" fill={accent} class="paw-l" />
      <circle cx="53" cy="34" r="3.5" fill={accent} class="paw-r" />
    </svg>
  {/if}
</button>

<style>
  .pet-stage {
    position: relative;
    width: var(--pet-size);
    height: var(--pet-size);
    min-width: 0;
    min-height: 0;
    padding: 0;
    margin: 0 auto;
    border: 0;
    line-height: 0;
    background: transparent;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: transform var(--anim-duration) var(--ease);
  }
  .pet-stage.static-pet { cursor: default; }
  .pet-stage:hover:not(.static-pet) { transform: scale(1.07); }
  .pet-stage.happy { animation: petJump calc(0.55s * var(--anim-scale)) ease; }

  .pet-svg, .pet-custom-img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 10px;
    animation: petFloat calc(3s * var(--anim-scale)) ease-in-out infinite;
  }
  .pet-custom-img {
    border: 2px solid var(--pet-accent);
  }

  .eyes {
    transform-origin: center;
    animation: petBlink calc(4.2s * var(--anim-scale)) infinite;
  }
  .antenna, .tail { transform-origin: 32px 30px; }

  .mood-thinking .pet-svg { animation: petNod calc(1.4s * var(--anim-scale)) ease-in-out infinite; }
  .mood-thinking .antenna { animation: antennaWiggle calc(0.9s * var(--anim-scale)) ease-in-out infinite; }
  .mood-working .pet-svg { animation: petBounceFast calc(0.7s * var(--anim-scale)) ease-in-out infinite; }
  .mood-working .paw-l, .mood-working .paw-r { animation: pawWork calc(0.5s * var(--anim-scale)) alternate infinite; }
  .mood-speaking .pet-svg { animation: petNod calc(0.9s * var(--anim-scale)) ease-in-out infinite; }
  .mood-speaking .tail { animation: tailWag calc(0.6s * var(--anim-scale)) ease-in-out infinite alternate; }
  .mood-compacting .pet-svg { animation: petJump calc(1.1s * var(--anim-scale)) ease-in-out infinite; }

  .pet-badge {
    position: absolute;
    top: -3px;
    right: -3px;
    width: 15px;
    height: 15px;
    border-radius: 999px;
    background: var(--color-surface);
    border: 1px solid var(--color-border-strong);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 8.5px;
    z-index: 2;
    box-shadow: var(--shadow-sm);
  }
  .badge-think { color: #d97706; }
  .badge-work { color: var(--pet-accent); }
  .badge-compact { color: var(--color-info); }

  @keyframes petFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
  }
  @keyframes petNod {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    40% { transform: translateY(2px) rotate(-4deg); }
    70% { transform: translateY(-2px) rotate(3deg); }
  }
  @keyframes petBounceFast {
    0%, 100% { transform: translateY(0) scale(1, 1); }
    50% { transform: translateY(-4px) scale(0.97, 1.04); }
  }
  @keyframes petJump {
    0%, 100% { transform: translateY(0) scale(1); }
    35% { transform: translateY(-8px) scale(1.06); }
    70% { transform: translateY(1px) scale(0.96); }
  }
  @keyframes petBlink {
    0%, 92%, 100% { transform: scaleY(1); }
    95% { transform: scaleY(0.12); }
  }
  @keyframes antennaWiggle {
    0%, 100% { transform: rotate(0deg); }
    50% { transform: rotate(10deg); }
  }
  @keyframes tailWag {
    from { transform: rotate(-6deg); }
    to { transform: rotate(10deg); }
  }
  @keyframes pawWork {
    from { transform: translateY(0); }
    to { transform: translateY(-4px); }
  }
</style>
