# Sytra Studio

Sytra Studio is a local, hardware-aware desktop app and MCP server for downloading models, fine-tuning, merging, and serving them on your machine.

---

# User Guide

## Features

- **Models**: Browse the shared catalog, download GGUF or SafeTensors checkpoints, import folders you already have, and convert weights to GGUF for Ollama or LM Studio.
- **Serving**: Preflight the checkpoint, then run llama.cpp (GGUF), vLLM (SafeTensors that fit), Sytra's native engine (indexed MoE with a complete kernel), or Colibri (GLM-5.2, Kimi K3, Inkling, and OLMoE SafeTensors). Unsupported formats are refused.
- **Fine-tuning**: SFT, DPO, ORPO, and CPO with LoRA, QLoRA, or DoRA. CUDA is required.
- **Merging**: SLERP, Linear, TIES, DARE-TIES, Task Arithmetic, Passthrough, and MoE via mergekit.
- **Data**: Hugging Face datasets and local JSONL (CSV/Parquet in the desktop app). Synthetic generation needs CUDA and transformers.
- **Hardware checks**: Recommendations use detected RAM and VRAM. If hardware cannot be detected, operations do not start.

## System Requirements

- Windows 10/11 (WebView2), macOS, or Linux
- Python 3.10+ on `PATH`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on `PATH` (used to create isolated Python environments)
- Enough disk for models (a 7B checkpoint can be tens of gigabytes)
- A dedicated GPU is strongly recommended for training; CPU merges work but are slow

> [!IMPORTANT]
> Check model size against your VRAM and RAM before download, train, merge, or serve. Oversized jobs can freeze or crash the machine.
>
> Report bugs on this repository's GitHub Issues page.

## Install

```bash
# Desktop
npx -y --prefer-online github:walkowicz19/sytra-studio

# MCP (Claude Code, Cursor, Codex, VS Code)
npx -y --prefer-online github:walkowicz19/sytra-studio mcp

# Re-download binaries and runner scripts
npx -y --prefer-online github:walkowicz19/sytra-studio install
```

Or install globally:

```bash
npm install -g sytra-studio
sytra          # Desktop
sytra mcp      # MCP server
```

The installer places binaries and Python runners under `~/.sytra/`. On first launch, wait for PyTorch and MergeKit setup to finish.

You can also download a desktop build from [GitHub Releases](https://github.com/walkowicz19/sytra-studio/releases).

## Storage and memory

In the app (shared with MCP via `.sytra-settings.json` in the workspace):

- **Model storage**: Hugging Face cache directory (`HF_HOME` for runner processes)
- **Main memory limit**: Automatic, 50%, 75%, or 90% of detected RAM — used for preflight estimates so large jobs are refused before they lock the machine

Prefer one MCP config across tools. Launch `sytra-mcp` directly instead of `npx` when RAM is tight.

## Typical workflows

### 1. Download or import a model

1. Open **Models**.
2. Download from the catalog, or **Scan Custom Folder** for weights you already have.
3. Convert PyTorch/SafeTensors to GGUF if you want to use Ollama or LM Studio.

Downloads keep every required shard. Incomplete checkpoints are not served.

### 2. Serve a model

Sytra picks the backend from the files on disk:

| You have | Backend |
|---|---|
| A GGUF file | llama.cpp |
| Complete SafeTensors that fit GPU or GPU+RAM | vLLM |
| An indexed native MoE with a complete kernel | Sytra engine |
| GLM-5.2, Kimi K3, Inkling, or OLMoE SafeTensors | [Colibri](https://github.com/JustVugg/colibri) |

GGUF never goes to Colibri. Kimi K2.7 Code uses the Sytra engine, not Colibri.

The first time you plan or serve a Colibri model, Sytra installs Colibri automatically from the [Colibri releases](https://github.com/JustVugg/colibri/releases). You can also download those binaries yourself, or run `python runner/scripts/provision_colibri.py`. If Colibri is already installed, set `SYTRA_COLIBRI_COMMAND` or `SYTRA_COLIBRI_HOME`. Frontier MoE on a 12 GB GPU is expected well below 5 tok/s; Sytra reports that instead of promising 5 tok/s.

From the command line:

```bash
python runner/scripts/plan_inference.py --model /path/to/model --backend auto
python runner/scripts/serve_model.py --model /path/to/model --backend auto
```

If llama.cpp, vLLM, the Sytra engine, or Colibri is not on `PATH`, set `SYTRA_LLAMA_SERVER`, `SYTRA_VLLM_COMMAND`, `SYTRA_ENGINE_COMMAND`, or `SYTRA_COLIBRI_COMMAND`.

### 3. Fine-tune

1. Open **Train** and pick a catalog model (or follow the hardware recommendation).
2. Preview a Hugging Face or local dataset.
3. Choose parameters, adapter (LoRA/QLoRA), and backend.
4. **Start** and watch loss and validation.

### 4. Merge

1. Open **Merge** and check that the selected models are compatible.
2. Choose a method (SLERP for divergent models; TIES or DARE-TIES for related fine-tunes).
3. Start the merge and follow progress in run history.

### 5. Use via MCP

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.sytra-studio]
command = "npx"
args = ["-y", "--prefer-online", "github:walkowicz19/sytra-studio", "mcp"]
```

Claude Code:

```bash
claude mcp add sytra-studio -- npx -y --prefer-online github:walkowicz19/sytra-studio mcp
```

Cursor / VS Code (`mcp.json`):

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

# Developer Guide

For compiling, extending, or testing Sytra Studio.

## Project structure

- `ui/` — Svelte 5 frontend
- `src-tauri/` — Tauri desktop shell
- `crates/sytra-contracts/` — Shared run and telemetry contracts
- `crates/sytra-host/` — Orchestration, resource guards, subprocesses
- `crates/sytra-mcp/` — Stdio MCP server
- `runner/` — Python train/merge/serve scripts and Colibri installer
- `npm/` — CLI installer
- `binaries/` — Packaged desktop and MCP builds

## Build from source

Prerequisites: stable Rust, Node.js 20+, and on Windows Visual Studio 2022 Build Tools with the Desktop C++ workload.

```bash
cd ui && npm ci && npm run build && cd ..
cargo build -p sytra-studio --release --features custom-protocol
cargo build -p sytra-mcp --release
```

Binaries land in `target-build/release/` (`sytra-studio.exe` / `sytra-mcp.exe` on Windows).

Dev UI:

```bash
cd ui && npm run dev
# other terminal
cargo tauri dev --config src-tauri/tauri.conf.json
```

## Native MoE runtime

`sytra-engine` serves indexed checkpoints that have a complete forward kernel (Kimi K2.7 Code, Mixtral, Qwen3/Qwen2 MoE, OLMoE, GraniteMoE). GLM-5.2, Kimi K3, and Inkling are cataloged but served through Colibri until those kernels ship. Unknown MoEs are not guessed into a streaming kernel.

```bash
cargo build -p sytra-engine --release
python runner/scripts/build_moe_index.py --model /models/checkpoint --adapter auto --expert-format auto
target-build/release/sytra-engine list-adapters
target-build/release/sytra-engine plan --model /models/checkpoint \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 \
  --verification-positions 8 --storage-bandwidth-mbps 3500 --target-tps 5
target-build/release/sytra-engine doctor --model /models/checkpoint
target-build/release/sytra-engine serve --model /models/checkpoint \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 --cuda-device 0 \
  --host 127.0.0.1 --port 8080
```

Windows CUDA uses `nvcuda.dll` (NVIDIA driver; no Python/Torch toolkit required for the engine). The indexer writes `.sytra-runtime.json` from SafeTensors headers and does not rewrite weights. Quality-preserving downloads keep every routed expert.

Oracles (`.sytra-oracle.json`) are produced on a machine that can load the full reference checkpoint, then copied with that exact checkpoint:

```bash
python runner/scripts/create_sytra_oracle.py --model /models/Kimi-K2.7-Code \
  --engine-command target-build/release/sytra-engine
target-build/release/sytra-engine oracle-check --model /models/Kimi-K2.7-Code \
  --ram-limit-mb 8192 --accelerator-limit-mb 6144 --cuda-device 0
```

The HTTP server exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. `plan`'s tok/s figure is an I/O ceiling, not a measured benchmark.

## Tests

```bash
cargo test --workspace
cd runner && python -m pytest && cd ..
cd ui && npm run check
```

## Credits

- UI layout is inspired by [MLX-LoRA-Studio](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) by Gökdeniz Gülmez.
- Disk-streamed MoE serving for GLM-5.2, Kimi K3, Inkling, and OLMoE uses [Colibri](https://github.com/JustVugg/colibri) by JustVugg. Download engine builds from [Colibri releases](https://github.com/JustVugg/colibri/releases).

## License

[MIT](LICENSE)
