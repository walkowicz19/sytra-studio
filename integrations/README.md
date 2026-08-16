# Agent Integrations — drive Sytra Studio from your coding agent

Sytra Studio ships an MCP server (`sytra-mcp.exe`) that exposes fine-tuning
and merging as tools. Any MCP-capable agent — **Claude Code, Codex, Cursor,
Antigravity** — can start runs, watch telemetry, and cancel them, using the
same validation, archive, and process machinery as the GUI. Runs started by
an agent show up in the app's Run History and vice versa.

Build it once:

```
cargo build -p sytra-mcp --release
# → target-build/release/sytra-mcp.exe
```

If the exe is moved out of the repo, set `SYTRA_WORKSPACE` to the project
root in the server's `env` so it can find the Python runner and the runs
archive.

## Tools exposed

| Tool | What it does |
|---|---|
| `get_status` | Hardware (backend/VRAM/RAM), whether an op is running |
| `list_catalog` | Pinned Hugging Face `model_id`s Sytra can download (Xet). Each row includes risk `alerts`. `download_model` and `start_train` require an exact match. |
| `guider_recommend` | Recipes (model+adapter+quant) that fit the hardware |
| `preview_dataset` | First rows of an hf/local/synthetic/klayer source |
| `start_train` | Start fine-tune (LoRA/QLoRA; sft/dpo/orpo/cpo), returns `op_id` |
| `merge_check` | Green/amber/red compatibility verdict for a merge |
| `start_merge` | Start a merge (slerp/ties/dare_ties/…), returns `op_id` |
| `get_run` | Status + last N telemetry lines — poll while running |
| `list_runs` | All archived operations |
| `stop_op` | Cancel (kills the whole process tree), idempotent |
| `get_settings` / `set_cache_dir` | Where HF models/datasets download (point it at a big drive) |
| `set_main_memory_limit` | Set or reset the RAM ceiling used by operation preflight checks |
| `export_guide` | Requirement checks + exact commands to run a finished model in Ollama (llama.cpp GGUF conversion, adapter merging, Modelfile) |

## Exporting a model to Ollama

`export_guide` encodes the rules learned the hard way — ask it before exporting:

1. **Never** `ollama create` straight from a safetensors directory. Ollama's
   importer silently produces broken output for some architectures (verified
   on Qwen2.5). Convert with the bundled `.tools/llama.cpp/convert_hf_to_gguf.py`
   (run it with the merge-env python), `--outtype q8_0`.
2. Train runs output a **LoRA adapter**, not a model — merge it into its base
   with peft `merge_and_unload` first (the tool returns a ready snippet).
3. The Modelfile must carry the chat `TEMPLATE` and stop tokens (ChatML for
   Qwen-family), or the model emits endless raw text.
4. To smoke-test from an agent, POST to `http://127.0.0.1:11434/api/generate`
   — `ollama run` hangs without a TTY.

One operation runs at a time. `start_*` returns immediately; agents poll
`get_run` until `status` is `done` / `error` / `stopped`.

## Claude Code

Already configured: the repo's [`.mcp.json`](../.mcp.json) registers the
server at project scope, so opening this folder in Claude Code is enough.
From anywhere else:

```
claude mcp add sytra-studio -- npx -y --prefer-online github:walkowicz19/sytra-studio mcp
```

## Cursor

`.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` globally):

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

## Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.sytra-studio]
command = "npx"
args = ["-y", "--prefer-online", "github:walkowicz19/sytra-studio", "mcp"]
```

## Antigravity

Settings → MCP Servers → Add server (raw config):

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

## Example prompts

Once connected, plain prompts work end to end:

> "Check my hardware, pick a model that fits, and fine-tune it on
> `./data/support-tickets.jsonl` for 500 steps. Watch the loss and stop it
> if it plateaus."

> "Merge org/knowledge-ft and org/toolcalling-ft with dare_ties on base
> mistralai/Mistral-7B-v0.1 — check compatibility first and show me the
> progress until it finishes."
