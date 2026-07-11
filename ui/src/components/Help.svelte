<script lang="ts">
  import { setTab } from '../store.svelte'

  let activeSection = $state<'train' | 'merge' | 'export'>('train')
</script>

<div class="page-layout">
  <div class="page-header">
    <div class="page-header-left">
      <h1 class="text-display">Algorithm & Merge Guide</h1>
      <p class="text-small">Learn when and how to configure fine-tuning and weight merging</p>
    </div>
  </div>

  <div class="page-content">
    <div class="page-form-area" style="display:flex; flex-direction:column; gap:var(--space-4)">
    <!-- Switcher -->
    <div class="segmented-control" style="align-self:flex-start">
      <button class:active={activeSection === 'train'} onclick={() => activeSection = 'train'}>
        <i class="bi bi-fire" style="margin-right:4px"></i> Fine-Tuning Algorithms
      </button>
      <button class:active={activeSection === 'merge'} onclick={() => activeSection = 'merge'}>
        <i class="bi bi-lightning-charge" style="margin-right:4px"></i> Merge Methods
      </button>
      <button class:active={activeSection === 'export'} onclick={() => activeSection = 'export'}>
        <i class="bi bi-box-arrow-up-right" style="margin-right:4px"></i> Export to Ollama / GGUF
      </button>
    </div>

    {#if activeSection === 'train'}
      <div class="grid-2 animate-in">
        <!-- SFT -->
        <section class="card">
          <div class="card-header"><span class="text-label">SFT (Supervised Fine-Tuning)</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Directly teaches the model to match specific target responses given a prompt. It updates the weights via standard cross-entropy loss.
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Best For</span>
              <span class="text-small">Learning new formatting, style adjustments, or mimicking specific conversation templates.</span>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Common Pitfall</span>
              <span class="text-small" style="color:var(--color-warn)">Overfitting. If trained for too many steps or with high learning rates, the model will lose general knowledge.</span>
            </div>
          </div>
        </section>

        <!-- DPO -->
        <section class="card">
          <div class="card-header"><span class="text-label">DPO (Direct Preference Optimization)</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Bypasses the traditional reinforcement learning reward-model step by training the policy model directly on pairwise choice preferences.
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Best For</span>
              <span class="text-small">Aligning model behavior, matching human tone preferences, or steering formatting rules.</span>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Dataset Requirement</span>
              <span class="text-small">Needs <code>prompt</code>, <code>chosen</code> (preferred response), and <code>rejected</code> (worse response) fields.</span>
            </div>
          </div>
        </section>

        <!-- ORPO & CPO -->
        <section class="card col-2">
          <div class="card-header"><span class="text-label">ORPO & CPO (One-Step Alignment)</span></div>
          <div class="card-body grid-2" style="gap:var(--space-4)">
            <div>
              <div class="text-title" style="font-size:13px; font-weight:600">ORPO</div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                Odds Ratio Preference Optimization integrates preference penalty directly into the SFT objective, eliminating the need for a separate SFT phase.
              </p>
            </div>
            <div>
              <div class="text-title" style="font-size:13px; font-weight:600">CPO</div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                Contrastive Preference Optimization is specifically designed to align translation or reasoning outputs without needing a reference model, lowering VRAM.
              </p>
            </div>
          </div>
        </section>
      </div>
    {:else if activeSection === 'merge'}
      <div class="grid-2 animate-in">
        <!-- SLERP -->
        <section class="card">
          <div class="card-header"><span class="text-label">SLERP (Spherical Linear Interpolation)</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Blends two models spherically on a unit hypersphere, preserving weight norms and tensor angles.
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Best For</span>
              <span class="text-small">Seamlessly combining two models of identical architecture (e.g., base model and its specialist instruct version).</span>
            </div>
          </div>
        </section>

        <!-- TIES & DARE-TIES -->
        <section class="card">
          <div class="card-header"><span class="text-label">TIES & DARE-TIES</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Resolves parameter conflicts across multiple models by dropping non-significant deltas and scaling/aligning task vectors.
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Best For</span>
              <span class="text-small">Merging up to 3 specialist models (e.g., math, code, and chat) back into a single base model.</span>
            </div>
          </div>
        </section>

        <!-- MoE / FrankenMoE -->
        <section class="card col-2">
          <div class="card-header"><span class="text-label">FrankenMoE (Mixture of Experts)</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Combines independent LLMs into a sparse Mixture of Experts (MoE) structure by slicing layers and introducing routing gates.
            </p>
            <div class="tip-card" style="border-left: 3px solid var(--color-warn); background:var(--color-surface-muted); padding:var(--space-3)">
              <div class="text-title" style="font-size:12px; font-weight:600; color:var(--color-warn)">
                <i class="bi bi-exclamation-triangle-fill" style="margin-right:4px"></i> HEALING RUN REQUIRED
              </div>
              <p class="text-small" style="margin-top:4px; color:var(--color-ink-subtle)">
                FrankenMoE models almost always produce gibberish or disjoint outputs initially because routing gates have random weights.
                <strong>You must run a subsequent SFT training (Healing)</strong> immediately on the merged MoE model to align the gating system.
              </p>
            </div>
          </div>
        </section>
      </div>
    {:else if activeSection === 'export'}
      <div class="grid-2 animate-in">
        <!-- Llama.cpp / GGUF conversion -->
        <section class="card">
          <div class="card-header"><span class="text-label">Convert to GGUF (Llama.cpp)</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <p class="text-body" style="font-size:13px">
              Llama.cpp, Ollama, and LM Studio run models in the optimized <strong>GGUF</strong> format. First, convert your local PyTorch or SafeTensors weights.
            </p>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Step 1: Clone llama.cpp</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt</code></pre>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Step 2: Convert weights</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>python convert_hf_to_gguf.py /path/to/my-model --outfile my-model.gguf --outtype f16</code></pre>
            </div>
            <div class="info-block">
              <span class="text-label" style="font-size:10px">Step 3: Quantize (Optional for Q4_K_M)</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin:0"><code>./llama-quantize my-model.gguf my-model-Q4_K_M.gguf Q4_K_M</code></pre>
            </div>
          </div>
        </section>

        <!-- Ollama and LM Studio integration -->
        <section class="card">
          <div class="card-header"><span class="text-label">Run in Ollama & LM Studio</span></div>
          <div class="card-body" style="display:flex; flex-direction:column; gap:var(--space-3)">
            <div class="info-block">
              <span class="text-label" style="font-size:11px; font-weight:600; color:var(--color-brand)">Ollama Integration</span>
              <span class="text-small" style="margin-top:var(--space-1)">Create a file named <code>Modelfile</code> with the following contents:</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin-top:4px"><code>FROM ./my-model-Q4_K_M.gguf
TEMPLATE "{"{{ .System }}"}\nUSER: {"{{ .Prompt }}"}\nASSISTANT: "
PARAMETER stop "USER:"
PARAMETER stop "ASSISTANT:"</code></pre>
              <span class="text-small" style="margin-top:var(--space-1)">Then compile and run it in Ollama:</span>
              <pre class="text-small" style="background:var(--color-surface); padding:var(--space-2); border-radius:4px; overflow:auto; margin-top:4px"><code>ollama create sytra-model -f Modelfile
ollama run sytra-model</code></pre>
            </div>
            <div class="info-block" style="margin-top:var(--space-2)">
              <span class="text-label" style="font-size:11px; font-weight:600; color:#1d5fa6">LM Studio & Llama.cpp</span>
              <ul class="text-small" style="padding-left:16px; margin:4px 0 0 0; line-height:1.5">
                <li><strong>LM Studio</strong>: Open LM Studio, go to the Local Server tab, select "Load Model", and choose your converted <code>my-model-Q4_K_M.gguf</code> file.</li>
                <li><strong>Llama.cpp CLI</strong>: Run inference directly from command line:
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
