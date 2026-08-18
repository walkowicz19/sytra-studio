# Sytra Studio

Sytra Studio is a local, hardware-aware desktop application and MCP server for fine-tuning, model merging, and model management. Both the Svelte/Tauri UI and the `sytra-mcp` server share the same Rust backend host, validation rules, run archives, resource guards, and Python runners.

---

# 📖 User Guide

This section covers Sytra Studio features, requirements, quick installation, and common workflows for end users.

## Features

- **Model Hub & Importer**: Scan local models, download commit-pinned GGUF or SafeTensors checkpoints through Hugging Face Xet with resumption and verification, and convert local SafeTensors/PyTorch weights to GGUF. The catalog is the shared Guider catalog used by both the desktop UI and MCP.
- **Verified Local Serving**: Preflight complete checkpoints before launching llama.cpp, vLLM, or Sytra's native MoE runtime. Unsupported formats are refused rather than guessed.
- **Fine-tuning**: SFT, DPO, ORPO, and CPO using LoRA, QLoRA, or DoRA **require a working CUDA training environment**. Missing CUDA or trainer dependencies is an error, not a simulated run.
- **Model Merging**: SLERP, Linear, TIES, DARE-TIES, Task Arithmetic, Passthrough, and MoE via mergekit. Missing mergekit is an error, not a fake success.
- **Data Sources**: Hugging Face datasets and local JSONL (CSV/Parquet in the desktop build). Synthetic generation requires CUDA + transformers. Klayer materialization calls the klayer MCP `export_dataset` tool (`npx -y klayer-mcp@latest`, override with `KLAYER_MCP_ARGV`).
- **Hardware-Aware Assistance**: Compatibility checks and recommendations use **detected** RAM/VRAM. If hardware cannot be detected, operations refuse to start.
- **Export**: Convert/export scripts run in the provisioned Python env and report real exit status. Hugging Face publish requires `huggingface_hub` and a token.

`main_memory_limit_mb` in `.sytra-settings.json` gates **preflight estimates only**; it does not cap MCP or Python process RSS. Prefer a single shared MCP config across harnesses, and launch `sytra-mcp` directly instead of `npx` when RAM matters.

## System Requirements

- **Operating System**: Windows 10/11 (with WebView2), macOS, or Linux.
- **Python**: Version 3.10 or newer installed and available on your system `PATH`.
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed and on your system `PATH` (used by Sytra to automatically provision isolated Python environments).
- **Disk Storage**: Ample space for model caches and outputs (a 7B model can require tens of gigabytes).
- **Hardware**: A dedicated GPU is highly recommended for fine-tuning. CPU-only merges are supported but can be slow.

> [!IMPORTANT]
> **Reporting Bugs & Issues:**
> Please report any bugs, errors, or unexpected behavior you encounter while using Sytra Studio (desktop UI or MCP server) directly on the repository's GitHub Issues page.
>
> **Hardware Guidance & Model Sizes:**
> It is **highly recommended to always check the size of the models** before initiating any merge or fine-tuning operations. Attempting to load, fine-tune, or merge models that exceed your system's hardware limits (available VRAM on GPU, or system RAM on CPU) can result in system instability, performance throttling, out-of-memory (OOM) crashes, or temporary freezing of your operating system.

## One-Command Launch (NPM / NPX CLI)

The easiest way to launch Sytra Studio without manual compilation is using `npx`:

```bash
# Instant launch Sytra Studio Desktop application
npx -y --prefer-online github:walkowicz19/sytra-studio

# Launch Sytra MCP server (for Claude Code, Cursor, Codex, etc.)
npx -y --prefer-online github:walkowicz19/sytra-studio mcp

# Force update / re-download binaries
npx -y --prefer-online github:walkowicz19/sytra-studio install
```

Alternatively, install globally via `npm`:

```bash
npm install -g sytra-studio

sytra          # Launch Sytra Studio Desktop
sytra mcp      # Launch Sytra MCP Server
```

This CLI installer automatically handles downloading pre-compiled release binaries for your operating system (Windows, macOS, or Linux) and deploys all required Python runner scripts to your user home under `~/.sytra/`.

## Install from GitHub Releases

1. Download the latest release installer for your operating system from the repository's Releases page.
2. Run the installer and launch the application.
3. *Note: On first launch, wait a few minutes for the environment setup to finish preparing the PyTorch and MergeKit dependencies.*

## Storage and Memory Settings

- **Model Storage**: Configures your Hugging Face cache directory. It sets the `HF_HOME` variable for new runner processes.
- **Main Memory Limit**: Restricts Sytra's RAM consumption (Automatic, 50%, 75%, or 90% of detected RAM) to prevent system freezing.
- Settings are stored in `.sytra-settings.json` in your workspace directory and are shared between the desktop UI and MCP server.

## Typical Workflows

### 1. Model Hub & Direct GGUF Download
1. Open the **Models** tab.
2. Download a GGUF or SafeTensors model from Hugging Face. Sytra resolves the requested revision to an immutable commit, preserves every required weight shard, and records `.sytra-model.json`.
3. Click **Scan Custom Folder** to import models stored in external directories.
4. Convert PyTorch/SafeTensors models to GGUF format for export to Ollama or LM Studio.

### Verified inference backends

- GGUF files are served by `llama-server`, with llama.cpp selecting GPU layers automatically and retaining a CPU/GPU hybrid fallback when the model exceeds VRAM.
- Complete SafeTensors model directories use vLLM when they fit the conservative GPU budget, or vLLM's UVA CPU-weight offload when they fit the configured GPU+RAM budget. MoE offload targets expert tensors and multi-GPU MoE plans enable expert parallelism.
- Native out-of-core containers use `sytra-engine`, a Sytra-owned Rust runtime. Its shared core provides immutable multi-span expert indexes, deterministic weighted storage mirrors, byte-exact reads with fallback, RAM and accelerator LRU/heat tiers, lease-safe accelerator eviction, batch-union loading, and bounded asynchronous prefetch.
- Frontier disk-streamed MoE that Colibri actually serves (GLM-5.2, Kimi K3, Inkling, OLMoE safetensors) is handed to `coli serve --auto-tier`. Sytra unpacks a pinned Colibri release into `.tools/colibri` on first `plan_inference` / `serve_model --backend colibri`, and on `auto` for those families. Do not send GGUF to Colibri. Install ahead of time with `python runner/scripts/provision_colibri.py --project-root .`. Set `SYTRA_SKIP_COLIBRI_PROVISION=1` to disable the download, or `SYTRA_COLIBRI_COMMAND` / `SYTRA_COLIBRI_HOME` to use an existing install. On Windows the launcher is a Python script (`python coli serve ...`); the engine binary sits next to it.
- Native memory limits are hard envelopes rather than cache hints: runtime reserve, compact BF16 KV state, dense-tensor working buffers, active expert leases, and hot caches are budgeted together. Evicted-but-live host leases and every CUDA allocation remain counted, routed batches execute in byte-bounded expert waves, oversized matrices stream in row tiles, and context length is reduced instead of overcommitting memory. The planner includes BF16-to-FP32 CPU expansion and simultaneous CUDA input/output buffers in peak staging estimates. Standard MHA/GQA adapters can keep planner-sized BF16 K/V allocations on CUDA and use a bounded two-pass attention kernel; the cache stays in RAM when preserving expert residency is the better fit. MLA adapters retain their separate compact-latent path.
- The performance path supports exact greedy speculative verification. A small draft model can propose multiple positions while the large MoE verifies them as one batch, amortizing dense-weight reads; adaptive lookahead is capped by the active memory envelope.
- `sytra-engine plan` reports conservative cold-decode and perfect-acceptance I/O ceilings from the indexed dense/expert byte ranges, bounded cache allocation, verification batch, and measured drive bandwidth. `--target-tps 5 --storage-bandwidth-mbps 3500` shows the minimum bandwidth needed and explicitly rejects an I/O throughput claim before compute, PCIe, draft-model, and tokenizer costs are considered.
- Model-specific attention, KV representation, quantization, router weighting, and forward kernels remain behind exact compiled adapters. Kimi K2.7 Code and the exact Mixtral, `qwen3_moe`, `qwen2_moe`, OLMoE, and GraniteMoE subsets have complete native forward paths. Standard-model routed experts and matrix projections may be BF16 or compressed-tensors packed symmetric INT4 group-32 with BF16 scales; Granite also has a bounded CPU-streamed F32 reference-checkpoint path. Embeddings, normalization, biases, and Kimi K2.7's family-specific dense contract otherwise remain BF16. Each checkpoint still requires an immutable-checkpoint oracle before serving. The registry also indexes GLM, Kimi K3, Inkling, DeepSeek V2/V3, Qwen3-Next, DBRX, Arctic, and MiniMax MoE, but those profiles remain correctness-gated until their complete family kernels are promoted. `sytra-generic-moe` safely indexes otherwise unknown routed MoEs but is intentionally storage-only.
- Common adapter primitives include softmax and sigmoid top-k routing, normalized/unnormalized weights, group-limited `no_aux_tc`, correction bias, SiLU/GELU/ReLU gated experts, standard MHA/GQA KV decode, compact MLA references, BF16/F16/FP8 decoding, and Qwen/Mixtral/DBRX projection binding. These references do not unlock serving by themselves; every exact family still needs its complete forward kernel and checkpoint oracle.
- Unknown architectures are never guessed into a streaming kernel. Unsupported formats, mixed checkpoints, missing indexed shards, unavailable engines, and checkpoints exceeding every verified memory path are rejected before serving.
- Set `SYTRA_LLAMA_SERVER`, `SYTRA_VLLM_COMMAND`, `SYTRA_ENGINE_COMMAND`, or `SYTRA_COLIBRI_COMMAND` when a binary is not on `PATH`. `SYTRA_VLLM_TENSOR_PARALLEL_SIZE` can override the detected GPU count.

Build and inspect the native runtime:

```bash
cargo build -p sytra-engine --release
python runner/scripts/build_moe_index.py --model /models/checkpoint \
  --adapter auto --expert-format auto
target-build/release/sytra-engine list-adapters
target-build/release/sytra-engine plan --model /models/checkpoint \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 \
  --verification-positions 8 --storage-bandwidth-mbps 3500 --target-tps 5
target-build/release/sytra-engine doctor --model /models/checkpoint
target-build/release/sytra-engine cuda-check --cuda-device 0 # staging + BF16 dense/expert oracles
target-build/release/sytra-engine kimi-k27-check --model /models/Kimi-K2.7-Code
target-build/release/sytra-engine kimi-k27-cuda-check --cuda-device 0
target-build/release/sytra-engine fingerprint --model /models/Kimi-K2.7-Code
target-build/release/sytra-engine benchmark --model /models/checkpoint \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 --cuda-device 0 \
  --prompt "Explain bounded MoE inference" --max-tokens 32 --iterations 3 --target-tps 5
```

On Windows, the accelerator tier loads the CUDA Driver API directly from `nvcuda.dll`; expert staging and compute need an NVIDIA driver but no Python, Torch, or CUDA toolkit. The indexer reads only SafeTensors headers and records every routed and dense tensor's dtype, shape, shard, and byte range in `.sytra-runtime.json`; it does not duplicate or rewrite the checkpoint. It handles separately named experts, axis-0 expert stacks, and merged-row expert tensors. Every indexed MoE layer must contain all configured experts with matching tensor signatures, and the native doctor independently checks the manifest against `config.json`. Dense layer tensors are individually streamable too, which matters when a model's non-routed BF16 backbone exceeds RAM.

The Kimi K2.7 Code adapter validates the complete public text-tower tensor contract (61 layers, 384 experts, top-8 `noaux_tc`, MLA/YaRN, dense prefix, shared experts, and compressed-tensors packed symmetric INT4 group-32 routed experts). Its bounded one-token executor connects embedding lookup, attention, residuals, dense/routed/shared MLPs, final normalization, and streamed vocabulary projection. MLA absorbs `kv_b_proj` algebraically: every projection row is streamed once per decode step while past positions remain compact BF16 latents, so full K/V heads are never materialized. The compact layout is 1,152 bytes/token/layer versus 40,960 bytes for materialized BF16 K/V.

CUDA includes cached (load-once) BF16 forward/transpose matvec kernels and packed INT4 expert kernels. Hot experts execute directly from tensor offsets inside their existing VRAM allocation instead of being copied or uploaded twice. `cuda-check` compares transient and VRAM-resident BF16 batches plus both dense orientations with CPU oracles, while `kimi-k27-cuda-check` checks host-staged and VRAM-resident packed matvec plus a complete routed expert. The Kimi runtime now includes native tokenizer/chat-template handling, bounded deterministic generation, and an OpenAI-compatible HTTP server. Serving still requires a real-checkpoint reference-logit plus teacher-forced token oracle; downloaded metadata cannot bypass that gate, and Kimi K2/K2.7 is never treated as Kimi K3 because their tensor and attention contracts differ.

The Mixtral path shares the bounded generation, oracle, speculative-decode, and OpenAI server stack. It implements residual/RMSNorm layers, half-rotation RoPE, causal standard or sliding-window GQA, planner-selected RAM or CUDA BF16 KV storage, normalized softmax top-k routing, byte-bounded expert waves, and discrete or axis-0 stacked gated experts. It accepts both current `model.layers.N.mlp` and legacy `block_sparse_moe` namespaces only when exactly one is present. BF16 and exact packed INT4/BF16 dense projections stream in bounded row tiles. Planner-sized hot experts remain in VRAM and execute directly from their resident allocation; cold or oversized experts fall back to bounded host/transient projections under the same global CUDA ceiling, without an FP32 weight expansion. F16, AWQ/GPTQ-style INT4, and quantized embeddings or normalization tensors remain locked.

The same bounded standard-MoE executor supports the `qwen3_moe` contract, including per-head Q/K RMSNorm, an explicit or checkpoint-derived head dimension, default RoPE parameters, optional sliding-window attention, normalized or unnormalized softmax top-k weights, stacked-axis-0 or discrete fused experts, and exact sparse/dense MLP layer cadence. Routed experts and matrix projections may use exact BF16 or packed symmetric INT4/BF16 group-32 tensors. `qwen3_next`, non-default RoPE variants, other quantization layouts, and quantized embeddings/norms remain locked.

The `qwen2_moe` subset adds its exact Q/K/V projection biases, discrete gate/up/down expert matrices, sparse/dense layer cadence, and the sigmoid-gated shared expert contribution. Routed experts, the shared expert, and other matrix projections may be BF16 or packed symmetric INT4/BF16 group-32; biases, embeddings, and normalization remain BF16. It uses the same bounded BF16 KV, expert-wave, CUDA projection, oracle, speculative-decode, and serving machinery. Stacked experts, other quantization layouts, and non-default RoPE remain locked.

The OLMoE subset adds full-projection Q/K RMSNorm, optional Q/K/V clipping before RoPE, fused stacked or discrete experts, and unnormalized selected softmax routing weights. Routed experts and matrix projections may be BF16 or packed symmetric INT4/BF16 group-32. Non-default RoPE, scaled routing, F16, other quantization layouts, and quantized embeddings/norms remain locked.

The GraniteMoE subset adds its exact embedding, attention, residual, and logits multipliers; selected-top-k softmax routing; `router.weight` or legacy `router.layer.weight`; and current fused stacked or legacy `input_linear`/`output_linear` experts. Standard BF16 and packed symmetric INT4/BF16 group-32 weights use the accelerator-capable path. Official tiny F32 checkpoints use bounded CPU-streamed dense/expert math while compact KV may remain on CUDA; the planner reports those fallbacks explicitly and never allocates an unsupported F32 expert cache in VRAM.

Generate the oracle on a trusted machine capable of loading the complete reference checkpoint, copy the resulting `.sytra-oracle.json` with that exact immutable checkpoint, then verify and serve it locally:

```bash
# torch/transformers are reference-machine-only dependencies; accelerate is needed for device maps
python runner/scripts/create_sytra_oracle.py --model /models/Kimi-K2.7-Code \
  --engine-command target-build/release/sytra-engine
# For a normal single-device CPU load without accelerate, add: --device-map none
# The same command accepts indexed sytra-mixtral, sytra-qwen3-moe, sytra-qwen2-moe, sytra-olmoe, and sytra-granite-moe checkpoints.
target-build/release/sytra-engine oracle-check --model /models/Kimi-K2.7-Code \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 --cuda-device 0
target-build/release/sytra-engine serve --model /models/Kimi-K2.7-Code \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 --cuda-device 0 \
  --host 127.0.0.1 --port 8080 \
  --draft-url http://127.0.0.1:8081 --draft-model tokenizer-compatible-draft
```

The server exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, including bounded SSE streaming. It limits request concurrency and serializes target-model execution because generations share the same proven staging partition. For greedy decoding, the optional loopback draft endpoint proposes tokens and Sytra verifies them transactionally in a memory-capped target batch; rejected suffix KV entries are truncated, a draft outage falls back to exact target steps, and sampling requests do not use the greedy verifier. Use a small model with the same tokenizer for useful acceptance. The `plan` command's 5 tok/s result is an I/O feasibility ceiling, not a benchmark guarantee; actual throughput also depends on GPU compute, PCIe, storage latency, routing locality, prompt length, and speculative acceptance.

`benchmark` refuses to run until the immutable checkpoint oracle passes. It performs an optional warmup, measures multiple bounded greedy generations, and reports per-run and aggregate tokens/second together with dense I/O, expert-wave, cache, live/peak RAM, and CUDA allocation metrics. Its `target_met` field is evidence for that exact checkpoint, prompt, memory envelope, and machine—not a transferable model-family guarantee.

Sytra does not delete routed MoE experts to reduce download size. Router top-k is dynamic per token, so quality-preserving downloads always keep the complete checkpoint.

### 2. Fine-Tuning a Model
1. Go to the **Train** tab and choose a catalog model or follow hardware recommendations.
2. Select and preview your dataset (Hugging Face or local files).
3. Select your training parameters, adapter types (LoRA/QLoRA), and backend.
4. Click **Start** and monitor the training loss and validation live.

### 3. Merging Models
1. Go to the **Merge** tab and verify the compatibility of your selected models.
2. Choose a merge method (e.g. SLERP for divergent models, TIES or DARE-TIES for related fine-tunes).
3. Start the process. Sytra runs the merge on CPU/GPU and logs progress directly to your runs history.

### 4. Using Sytra via MCP (Claude Code, Cursor, Codex, VSCode)
The `sytra-mcp` server exposes tools for model inspection, catalog recommendations, dataset previews, and execution controls via standard `npx` execution:
- **Codex configuration (`~/.codex/config.toml`)**:
  ```toml
  [mcp_servers.sytra-studio]
  command = "npx"
  args = ["-y", "--prefer-online", "github:walkowicz19/sytra-studio", "mcp"]
  ```
- **Claude Code configuration**:
  ```bash
  claude mcp add sytra-studio -- npx -y --prefer-online github:walkowicz19/sytra-studio mcp
  ```
- **Cursor / VSCode configuration (`mcp.json` or settings)**:
  ```json
  {
    "mcpServers": {
      "sytra-studio": {
        "command": "npx",
        "args": ["-y", "--prefer-online", "github:walkowicz19/sytra-studio", "mcp"]
      }
    }
  }
  ```

---

# 🛠️ Developer Guide (Technical)

This section is intended for developers who want to compile, modify, or test Sytra Studio.

## Project Structure

- `ui/` — Frontend built with Svelte 5, Vite, and CSS.
- `src-tauri/` — Rust Tauri desktop bridge, system configurations, and window wrappers.
- `crates/sytra-contracts/` — Shared data contracts for run configurations, telemetry, and guider lineage.
- `crates/sytra-host/` — Rust orchestration layer, resource guards, validation, and subprocess management.
- `crates/sytra-mcp/` — Stdio MCP server wiring.
- `runner/` — Python execution environments (PyTorch, TRL, MergeKit) and telemetry emitters.
- `npm/` — Global Node.js CLI packaging scripts.
- `binaries/` — Release-ready cross-platform binaries.

## Building from Source

### Prerequisites
- **Rust**: Install the stable Rust toolchain.
- **Node.js**: Version 20 or newer.
- **C++ Build Tools**: (Windows only) Visual Studio 2022 Build Tools with Desktop C++ development workload.

### Step-by-Step Build
1. Clone the repository and navigate to the folder.
2. Build the frontend:
   ```bash
   cd ui
   npm ci
   npm run build
   cd ..
   ```
3. Build the Sytra Studio Desktop application in release mode:
   ```bash
   cargo build -p sytra-studio --release --features custom-protocol
   ```
   *The executable will be located in `target-build/release/sytra-studio.exe`.*
4. Build the Sytra MCP server:
   ```bash
   cargo build -p sytra-mcp --release
   ```
   *The executable will be located in `target-build/release/sytra-mcp.exe`.*

## Running in Development Mode

To run with live-reloading UI and Tauri:

1. Start the Vite dev server:
   ```bash
   cd ui
   npm run dev
   ```
2. In a separate terminal, launch Tauri in dev mode:
   ```bash
   cargo tauri dev --config src-tauri/tauri.conf.json
   ```

## Running Tests

Ensure all components and Python boundary tests pass successfully:

```bash
# Run all Rust unit and integration tests
cargo test --workspace

# Run Python runner tests
cd runner
python -m pytest
cd ..

# Check UI types and formatting
cd ui
npm run check
```

## Credits & Acknowledgements

Sytra Studio's design and user interface layout are inspired by [MLX-LoRA-Studio](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) by Gökdeniz Gülmez.

## License

This project is licensed under the [MIT License](LICENSE).
