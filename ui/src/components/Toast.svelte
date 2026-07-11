<script lang="ts">
  import type { Toast } from '../store.svelte'

  let { toast }: { toast: Toast } = $props()

  const icons: Record<Toast['kind'], string> = {
    success: '✓', error: '✕', info: 'ℹ',
  }
  const classes: Record<Toast['kind'], string> = {
    success: 'toast-success', error: 'toast-error', info: 'toast-info',
  }
</script>

<div class="toast {classes[toast.kind]} animate-in" role="status" aria-atomic="true">
  <span class="toast-icon">{icons[toast.kind]}</span>
  <span class="toast-msg">{toast.message}</span>
</div>

<style>
  .toast {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    border-radius: var(--radius-md);
    font-size: 13px;
    font-weight: 500;
    box-shadow: var(--shadow-lg);
    pointer-events: all;
    min-width: 220px;
    max-width: 360px;
    border: 1px solid transparent;
  }
  .toast-success {
    background: var(--color-success-bg);
    color: var(--color-success);
    border-color: rgba(26, 127, 74, 0.2);
  }
  .toast-error {
    background: var(--color-error-bg);
    color: var(--color-error);
    border-color: rgba(192, 57, 43, 0.2);
  }
  .toast-info {
    background: var(--color-info-bg);
    color: var(--color-info);
    border-color: rgba(29, 95, 166, 0.2);
  }
  .toast-icon { font-size: 15px; flex-shrink: 0; }
  .toast-msg  { flex: 1; line-height: 1.4; }
</style>
