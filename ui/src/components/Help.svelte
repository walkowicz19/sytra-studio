<script lang="ts">
  import { setTab } from '../store.svelte'
  import { t } from '../i18n.svelte'

  let activeSection = $state<'train' | 'merge' | 'export'>('train')
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">{t('help.title')}</h1>
      <p class="text-small">{t('help.subtitle')}</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area" style="display:flex; flex-direction:column; gap:var(--space-4)">
    <!-- Switcher -->
    <div class="segmented-control" style="align-self:flex-start">
      <button class:active={activeSection === 'train'} onclick={() => activeSection = 'train'}>
        <i class="bi bi-fire" style="margin-right:4px"></i> {t('help.tab.train')}
      </button>
      <button class:active={activeSection === 'merge'} onclick={() => activeSection = 'merge'}>
        <i class="bi bi-lightning-charge" style="margin-right:4px"></i> {t('help.tab.merge')}
      </button>
      <button class:active={activeSection === 'export'} onclick={() => activeSection = 'export'}>
        <i class="bi bi-box-arrow-up-right" style="margin-right:4px"></i> {t('help.tab.export')}
      </button>
    </div>

    {#if activeSection === 'train'}
      <div class="grid-2 animate-in">
        <!-- SFT -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.sft.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.sft.desc')}
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.bestFor')}</span>
              <span class="text-small">{t('help.sft.best')}</span>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.commonPitfall')}</span>
              <span class="text-small" style="color:var(--color-warn)">{t('help.sft.pitfall')}</span>
            </div>
          </div>
        </section>

        <!-- DPO -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.dpo.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.dpo.desc')}
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.bestFor')}</span>
              <span class="text-small">{t('help.dpo.best')}</span>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.datasetReq')}</span>
              <span class="text-small">{t('help.dpo.req')}</span>
            </div>
          </div>
        </section>

        <!-- ORPO & CPO -->
        <section class="card col-2">
          <div class="card-header"><span class="text-label">{t('help.orpo.title')}</span></div>
          <div class="card-body grid-2" style="gap:var(--space-4)">
            <div>
              <div class="text-title" style="font-size:13px; font-weight:600">{t('help.orpo.name')}</div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                {t('help.orpo.desc')}
              </p>
            </div>
            <div>
              <div class="text-title" style="font-size:13px; font-weight:600">{t('help.cpo.name')}</div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                {t('help.cpo.desc')}
              </p>
            </div>
          </div>
        </section>
      </div>
    {:else if activeSection === 'merge'}
      <div class="grid-2 animate-in">
        <!-- SLERP -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.slerp.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.slerp.desc')}
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.bestFor')}</span>
              <span class="text-small">{t('help.slerp.best')}</span>
            </div>
          </div>
        </section>

        <!-- TIES & DARE-TIES -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.ties.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.ties.desc')}
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.bestFor')}</span>
              <span class="text-small">{t('help.ties.best')}</span>
            </div>
          </div>
        </section>

        <!-- MoE / FrankenMoE -->
        <section class="card col-2">
          <div class="card-header"><span class="text-label">{t('help.moe.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.moe.desc')}
            </p>
            <div class="tip-card" style="border-left: 3px solid var(--color-warn); background:var(--color-surface-muted); padding:var(--space-3)">
              <div class="text-title" style="font-size:12px; font-weight:600; color:var(--color-warn)">
                <i class="bi bi-exclamation-triangle-fill" style="margin-right:4px"></i> {t('help.moe.warningTitle')}
              </div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                {t('help.moe.warningDesc')}
              </p>
            </div>
          </div>
        </section>
      </div>
    {:else if activeSection === 'export'}
      <div class="grid-2 animate-in">
        <!-- Llama.cpp / GGUF conversion -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.exportGguf.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              {t('help.exportGguf.desc')}
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.exportGguf.step1')}</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt</code></pre>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.exportGguf.step2')}</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>python convert_hf_to_gguf.py /path/to/my-model --outfile my-model.gguf --outtype f16</code></pre>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">{t('help.exportGguf.step3')}</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>./llama-quantize my-model.gguf my-model-Q4_K_M.gguf Q4_K_M</code></pre>
            </div>
          </div>
        </section>

        <!-- Ollama and LM Studio integration -->
        <section class="card">
          <div class="card-header"><span class="text-label">{t('help.runOllama.title')}</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <div class="info-block">
              <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">{t('help.runOllama.ollamaTitle')}</span>
              <span class="text-small" style="margin-top:var(--space-1)">{t('help.runOllama.modelfile')}</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin-top:4px"><code>FROM ./my-model-Q4_K_M.gguf
TEMPLATE "{"{{ .System }}"}\nUSER: {"{{ .Prompt }}"}\nASSISTANT: "
PARAMETER stop "USER:"
PARAMETER stop "ASSISTANT:"</code></pre>
              <span class="text-small" style="margin-top:var(--space-1)">{t('help.runOllama.compile')}</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin-top:4px"><code>ollama create sytra-model -f Modelfile
ollama run sytra-model</code></pre>
            </div>
            <div class="info-block" style="margin-top:var(--space-2)">
              <span class="text-label" style="font-size:11px; font-weight:600; color:#1d5fa6">{t('help.runOllama.lmTitle')}</span>
              <ul class="text-small" style="padding-left:16px; margin:4px 0 0 0; line-height:1.5">
                <li>{t('help.runOllama.lmDesc')}</li>
                <li>{t('help.runOllama.cliDesc')}
                  <pre style="background:var(--color-surface); padding:var(--space-1); border-radius:4px; overflow:auto; margin-top:4px"><code>./llama-cli -m my-model-Q4_K_M.gguf -p "Hello!"</code></pre>
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    {/if}
    </div>
  </div>
</div>

<style>
  .info-block {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: var(--space-2) var(--space-3);
    background: var(--color-surface-muted);
    border-radius: var(--radius-sm);
    border: 1px solid var(--color-border);
  }

  /* Segmented control */
  .segmented-control {
    display: flex;
    background: var(--color-surface-muted);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-sm);
    padding: 2px;
  }
  .segmented-control button {
    background: none;
    border: none;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    color: var(--color-ink-subtle);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
  }
  .segmented-control button.active {
    background: var(--color-surface);
    color: var(--color-ink);
    box-shadow: var(--shadow-sm);
  }
</style>
