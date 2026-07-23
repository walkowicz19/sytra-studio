# Sytra Studio 1.0.0

Sytra Studio is a local, hardware-aware desktop application and MCP server for fine-tuning and merging language models. Both the Svelte/Tauri UI and the `sytra-mcp` server share the same Rust backend host, validation rules, run archives, resource guards, and Python runners.

---

# 📖 User Guide

This section covers Sytra Studio features, requirements, quick installation, and common workflows for end users.

## Features

- **Fine-tuning**: Supports SFT, DPO, ORPO, and CPO using LoRA, QLoRA, or DoRA.
- **Model Merging**: Multi-model blending using SLERP, Linear, TIES, DARE-TIES, Task Arithmetic, Passthrough, and Mixture of Experts (MoE) workflows.
- **Flexible Data Sources**: Load datasets from Hugging Face, local JSONL/CSV/Parquet files, synthetic generation, or multi-dataset mixes.
- **Hardware-Aware Assistance**: Compatibility checks, hardware recommendations, live telemetry, and unified cache/memory settings.
- **Downstream Integrations**: Easily export local checkpoints to Ollama, LM Studio, Hugging Face, or GGUF formats.

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

## One-Command Installation (NPM CLI)

The easiest way to install and launch Sytra is globally via NPM:

```bash
# Install Sytra globally
npm install -g sytra-studio

# Launch the Sytra Studio Desktop application
sytra

# Launch the Sytra MCP server
sytra mcp
```

This CLI installer automatically handles downloading the correct pre-compiled binaries for your operating system (Windows, macOS, or Linux) and deploys all required Python runner scripts to your user home under `~/.sytra/`.

## Install from GitHub Releases

1. Download the latest release installer for your operating system from the repository's Releases page.
2. Run the installer and launch the application.
3. *Note: On first launch, wait a few minutes for the environment setup to finish preparing the PyTorch and MergeKit dependencies.*

## Storage and Memory Settings

- **Model Storage**: Configures your Hugging Face cache directory. It sets the `HF_HOME` variable for new runner processes.
- **Main Memory Limit**: Restricts Sytra's RAM consumption (Automatic, 50%, 75%, or 90% of detected RAM) to prevent system freezing.
- Settings are stored in `.sytra-settings.json` in your workspace directory and are shared between the desktop UI and MCP server.

## Typical Workflows

### 1. Fine-Tuning a Model
1. Go to the **Train** tab and choose a catalog model or follow hardware recommendations.
2. Select and preview your dataset (Hugging Face or local files).
3. Select your training parameters, adapter types (LoRA/QLoRA), and backend.
4. Click **Start** and monitor the training loss and validation live.

### 2. Merging Models
1. Go to the **Merge** tab and verify the compatibility of your selected models.
2. Choose a merge method (e.g. SLERP for divergent models, TIES or DARE-TIES for related fine-tunes).
3. Start the process. Sytra runs the merge on CPU/GPU and logs progress directly to your runs history.

### 3. Using Sytra via MCP (Claude Code, Cursor, Codex)
The `sytra-mcp` server exposes tools for model inspection, catalog recommendations, dataset previews, and execution controls.
- **Codex configuration (`~/.codex/config.toml`)**:
  ```toml
  [mcp_servers.sytra-studio]
  command = "D:\\Projects\\sytra-studio\\target-build\\release\\sytra-mcp.exe"
  env = { SYTRA_WORKSPACE = "D:\\Projects\\sytra-studio" }
  ```
- **Claude Code configuration**:
  ```bash
  claude mcp add sytra-studio -e SYTRA_WORKSPACE=D:\Projects\sytra-studio -- D:\Projects\sytra-studio\target-build\release\sytra-mcp.exe
  ```
- **Cursor configuration (`project-level or global JSON`)**:
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

