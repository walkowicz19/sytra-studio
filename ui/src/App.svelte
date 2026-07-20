<script lang="ts">
  import { tabStore, setTab, run, toastStore, hwStore, setHwInfo, uiMode } from './store.svelte'
  import { api } from './api'
  import Sidebar from './components/Sidebar.svelte'
  import TrainForm from './components/TrainForm.svelte'
  import MergeForm from './components/MergeForm.svelte'
  import SimpleTrain from './components/SimpleTrain.svelte'
  import SimpleMerge from './components/SimpleMerge.svelte'
  import LiveView from './components/LiveView.svelte'
  import Runs from './components/Runs.svelte'
  import Guider from './components/Guider.svelte'
  import Help from './components/Help.svelte'
  import Settings from './components/Settings.svelte'
  import Toast from './components/Toast.svelte'
  import { onMount } from 'svelte'

  onMount(async () => {
    try { const info = await api.getHardwareInfo(); setHwInfo(info) } catch {}
  })
</script>

<div class="shell">
  <Sidebar />

  <main class="main">
    {#key `${tabStore.active}-${uiMode.advanced}`}
      <div class="pane animate-in">
        {#if tabStore.active === 'train'}
          {#if uiMode.advanced}<TrainForm />{:else}<SimpleTrain />{/if}
        {:else if tabStore.active === 'merge'}
          {#if uiMode.advanced}<MergeForm />{:else}<SimpleMerge />{/if}
        {:else if tabStore.active === 'runs'}
          <Runs />
        {:else if tabStore.active === 'guider'}
          <Guider />
        {:else if tabStore.active === 'settings'}
          <Settings />
        {:else if tabStore.active === 'help'}
          <Help />
        {/if}
      </div>
    {/key}

    {#if run.status !== 'idle'}
      <div class="live-panel animate-in">
        <LiveView />
      </div>
    {/if}
  </main>
</div>

<div class="toast-layer" aria-live="polite">
  {#each toastStore.items as toast (toast.id)}
    <Toast {toast} />
  {/each}
</div>

<style>
  .shell {
    height: 100vh;
    display: flex;
    overflow: hidden;
    background: var(--color-canvas);
  }
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
  }
  /* The pane fills the entire remaining space */
  .pane {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .live-panel {
    border-top: 1px solid var(--color-border);
    background: var(--color-surface);
    height: 320px;
    flex-shrink: 0;
    overflow: hidden;
  }
  .toast-layer {
    position: fixed;
    bottom: var(--space-5);
    right: var(--space-5);
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    z-index: 1000;
    pointer-events: none;
  }
</style>
