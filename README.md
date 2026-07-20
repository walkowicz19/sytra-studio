# Sytra Studio 1.0.0

Sytra Studio is a local, hardware-aware desktop application and MCP server for fine-tuning and merging language models. The Svelte/Tauri UI and the `sytra-mcp` server use the same Rust host, validation rules, run archive, resource guard, and Python runners.

## Features

- Fine-tuning with SFT, DPO, ORPO, and CPO using LoRA, QLoRA, or DoRA.
- Model merging with SLERP, linear, TIES, DARE-TIES, task arithmetic, passthrough, and MoE workflows.
- Hugging Face, local JSONL/CSV/Parquet, synthetic, multi-dataset, and Klayer data sources.
- Hardware-aware recommendations, compatibility checks, live telemetry, cancellation, and run history.
- User-selectable Hugging Face cache directory and main-memory budget shared by the UI and MCP server.
- Local outputs that can be published to Hugging Face or converted for Ollama, LM Studio, and llama.cpp.

## Requirements

### To run a release build

- Windows 10/11 with WebView2 (included on current Windows releases).
- Python 3.10 or newer on `PATH`.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on `PATH`; Sytra uses it to create isolated training and merge environments on first launch.
- Enough storage for model downloads and outputs. A 7B model commonly needs tens of gigabytes across cache, working files, and output.
- A supported GPU is recommended for fine-tuning. CPU-only model merging is supported but can be slow.

### To build from source

- Rust stable with Cargo.
- Node.js 20 or newer and npm.
- Windows: Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
- The runtime requirements above.

The first launch provisions `.sytra-envs/train-env` with PyTorch, Transformers, TRL, and Datasets, and `.sytra-envs/merge-env` with MergeKit. This requires internet access and can take several minutes.

## Install the desktop UI

### From a GitHub release

Download the v1.0.0 installer for your operating system from the repository's Releases page and run it. On first launch, wait for environment provisioning to finish before starting a job.

### From source

```powershell
git clone https://github.com/walkowicz19/sytra-studio.git
cd sytra-studio\ui
npm ci
npm run build
cd ..
cargo build -p sytra-studio --release --features custom-protocol
```

The Windows executable is written to `target-build/release/sytra-studio.exe`. To run the development UI with Tauri:

```powershell
cd ui
npm run dev
```

In another terminal:

```powershell
cargo tauri dev --config src-tauri/tauri.conf.json
```

## Install the MCP server

Build the server:

```powershell
cargo build -p sytra-mcp --release
```

The executable is `target-build/release/sytra-mcp.exe`. Keep `SYTRA_WORKSPACE` pointed at the cloned project so the server can find the Python runner, shared settings, and run archive.

### Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.sytra-studio]
command = "D:\\Projects\\sytra-studio\\target-build\\release\\sytra-mcp.exe"
env = { SYTRA_WORKSPACE = "D:\\Projects\\sytra-studio" }
```

### Claude Code

```powershell
claude mcp add sytra-studio -e SYTRA_WORKSPACE=D:\Projects\sytra-studio -- D:\Projects\sytra-studio\target-build\release\sytra-mcp.exe
```

### Cursor or another JSON-configured MCP client

```json
{
  "mcpServers": {
    "sytra-studio": {
      "command": "D:\\Projects\\sytra-studio\\target-build\\release\\sytra-mcp.exe",
      "env": {
        "SYTRA_WORKSPACE": "D:\\Projects\\sytra-studio"
      }
    }
  }
}
```

The MCP server exposes hardware/status inspection, catalog recommendations, dataset preview, fine-tune and merge execution, run polling/cancellation, cache and memory settings, and export guidance. See [integrations/README.md](integrations/README.md) for the complete tool list and example prompts.

## Storage and memory settings

In the desktop sidebar:

- **Model storage** selects the Hugging Face cache directory. It sets `HF_HOME` for each newly started runner process; existing cached files are not moved.
- **Main memory** selects Automatic, 50%, 75%, or 90% of detected RAM. The selected ceiling is enforced by merge preflight checks and is shared with MCP.

From MCP, use `get_settings`, `set_cache_dir`, and `set_main_memory_limit`. Settings are persisted in `.sytra-settings.json` in `SYTRA_WORKSPACE` and apply to the next operation.

## Typical workflow

### Fine-tune

1. Open **Train** and choose a catalog model or hardware recommendation.
2. Select and preview a Hugging Face or local dataset.
3. Choose the adapter, backend, and training parameters.
4. Start the run and monitor telemetry. The result is an adapter checkpoint.

### Merge

1. Run the compatibility advisor for two or three related models.
2. Choose a method. Task-vector methods require true fine-tunes of the stated base model; use SLERP for related but divergent lineages.
3. Choose the output path, start the merge, and monitor the shared run history.

## Validation

```powershell
cargo test --workspace
cd runner
python -m pytest
cd ..\ui
npm run check
npm run build
```

## Output and Ollama/LM Studio

Fine-tuning produces a LoRA adapter; merge it into the base model before GGUF conversion. A merge produces a full checkpoint. Use the MCP `export_guide` tool for environment-aware conversion steps, or convert a full checkpoint with llama.cpp:

```powershell
python convert_hf_to_gguf.py D:\models\sytra-output --outfile model.gguf --outtype q8_0
```

Load the resulting GGUF in LM Studio, or reference it from an Ollama `Modelfile` and run `ollama create`.

## Project layout

- `ui/` — Svelte 5 frontend.
- `src-tauri/` — Tauri desktop bridge and packaging.
- `crates/sytra-contracts/` — shared run, merge, telemetry, and guider contracts.
- `crates/sytra-host/` — orchestration, validation, resource guards, settings, and subprocess management.
- `crates/sytra-mcp/` — stdio MCP server.
- `runner/` — Python training, merge, publish, and telemetry runners.

## License

Add a license file before redistributing binaries if the project is not intended to remain all-rights-reserved.
