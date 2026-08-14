use serde_json::{json, Value};
use std::sync::OnceLock;

pub fn tool_definitions() -> &'static Value {
    static DEFS: OnceLock<Value> = OnceLock::new();
    DEFS.get_or_init(|| {
    json!([
        {
            "name": "get_status",
            "description": "Current Sytra Studio state: whether an operation is running, detected backend (cuda/mps/cpu), VRAM/RAM, and the workspace path.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "get_settings",
            "description": "Current app settings: where Hugging Face models/datasets are cached (hf_cache_dir) and whether it is a custom location.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "set_cache_dir",
            "description": "Set where models and datasets are downloaded/cached (HF_HOME) — e.g. point it at a big HDD instead of a small system SSD. Pass path=null to reset to the workspace default. Applies to the next started operation; existing cached files are not moved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": ["string", "null"], "description": "Absolute directory path, or null to reset to default" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "set_main_memory_limit",
            "description": "Choose the maximum system RAM Sytra may budget during preflight checks. Pass limit_mb=null to use all detected RAM. Applies to the next operation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit_mb": { "type": ["integer", "null"], "minimum": 2048, "description": "RAM ceiling in MB, or null for automatic" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "configure_fast_cache",
            "description": "Configure verified Hugging Face/Xet authentication behavior and low-bit defaults.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tokenless": { "type": "boolean", "description": "Use public access when no HF_TOKEN is explicitly available" },
                    "low_bit_mode": { "type": ["integer", "null"], "description": "Quantization bit mode (1, 2, 4 bits)" },
                    "vram_expert_cache_mb": { "type": ["integer", "null"], "description": "Legacy expert-cache budget retained for settings compatibility" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "download_model",
            "description": "Start or poll a resumable, commit-pinned Hugging Face/Xet model download. The model argument MUST be an exact model_id from list_catalog — arbitrary Hugging Face repos are rejected. The response includes architecture/license/memory risk alerts. Complete weight shard sets are always preserved.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Hugging Face model repository ID" },
                    "dest_dir": { "type": "string", "description": "Optional destination directory" },
                    "purpose": { "type": "string", "enum": ["inference", "finetune", "merge"], "description": "Select a complete model format for this workflow" },
                    "quant": { "type": "string", "description": "GGUF quantization such as Q4_K_M, or auto" },
                    "revision": { "type": "string", "description": "Branch, tag, or commit to resolve and pin" }
                },
                "required": ["model"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_catalog",
            "description": "List the pinned Hugging Face catalog Sytra can download (verified Xet transfers). Each entry includes alerts (VRAM/RAM, MoE hybrid, gated license, Qwen3.5≠Qwen2, never import raw SafeTensors into Ollama). download_model and start_train require an exact model_id from this list.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "guider_recommend",
            "description": "Get hardware-aware training recipes (model + adapter + quantization) that fit the given or detected VRAM/RAM.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "accelerator": { "type": "string", "description": "cuda | mps | cpu | rocm (default: cuda)" },
                    "vram_mb": { "type": "integer", "description": "Override detected VRAM in MB" },
                    "ram_mb": { "type": "integer", "description": "Override detected RAM in MB" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "merge_check",
            "description": "Check compatibility of 2-3 models for a merge method before starting. Returns verdict green/amber/red with a reason. IMPORTANT: task-vector methods (ties/dare_ties/task_arithmetic) only work with true FINE-TUNES of the base model (weight delta ~1-2%); continued-pretrained lineages (e.g. a -Coder or -Math variant vs its plain base) are NOT fine-tunes and will produce a broken model — use slerp for those. Pass base_model to enable the lineage check.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "models": { "type": "array", "items": { "type": "string" }, "description": "Model ids to merge (2-3)" },
                    "method": { "type": "string", "description": "linear | slerp | ties | dare_ties | task_arithmetic | passthrough | moe" },
                    "base_model": { "type": "string", "description": "Base model for task-vector methods — enables the lineage-mismatch check" }
                },
                "required": ["models", "method"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_runs",
            "description": "List all archived operations (train and merge) with op_id, kind, status (running/done/error/stopped) and artifact path.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "get_run",
            "description": "Get one operation's status plus the last N telemetry lines (loss/progress metrics, stage events, logs). Poll this to follow a running operation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_id": { "type": "string" },
                    "tail": { "type": "integer", "description": "How many trailing telemetry lines to return (default 20)" }
                },
                "required": ["op_id"],
                "additionalProperties": false
            }
        },
        {
            "name": "start_train",
            "description": "Start a fine-tuning run (LoRA/QLoRA/DoRA; sft/dpo/orpo/cpo). Returns op_id immediately — poll get_run for progress. Only one operation runs at a time. `config` follows the run.yaml contract; unspecified fields get sensible defaults. Minimum: {\"model\": \"<catalog model_id>\", \"data\": {\"source\": \"local\", \"local\": {\"path\": \"data.jsonl\", \"format\": \"jsonl\", \"mapping\": {\"prompt\": \"prompt\", \"completion\": \"completion\"}}}}. Data sources: hf {repo_id, split}, local {path, format, mapping}, synthetic {generator_model, judge_model, mode, count, topic}, klayer {query, min_trust_tier, snapshot}. The output is a LoRA ADAPTER, not a full model — call export_guide for how to merge it and run it in Ollama.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": "run.yaml-shaped config. Required: model, data. Optional: train_mode, adapter{type,rank,alpha,dropout,quant_bits}, optim{learning_rate,schedule,warmup_steps}, train{max_steps,batch_size,max_seq_len,save_every}, output{adapter_path}."
                    }
                },
                "required": ["config"],
                "additionalProperties": false
            }
        },
        {
            "name": "start_merge",
            "description": "Start a model merge (weight arithmetic, CPU-friendly, no dataset). Returns op_id immediately — poll get_run for progress. `config` follows the merge.yaml contract. Minimum: {\"merge_method\": \"dare_ties\", \"base_model\": \"<id>\", \"models\": [\"org/model-a\", \"org/model-b\"]}. models entries may be plain id strings or {model, parameters:{weight,density}}. Method-global parameters go in config.parameters (e.g. slerp needs {\"parameters\": {\"t\": 0.35}}). base_model is required for ties/dare_ties/task_arithmetic — and those methods ONLY work with true fine-tunes of that base: merging a continued-pretrained lineage (-Coder/-Math/-VL variants vs a plain base) produces a broken model; use slerp for related-but-divergent models. The runner verifies this with a weight-delta preflight and aborts lineage mismatches. Compatibility is checked server-side; a red verdict refuses to start. To run the merged model in Ollama afterwards, call export_guide.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "description": "merge.yaml-shaped config. Required: merge_method, models. Optional: base_model, dtype, tokenizer{source}, output{model_path}."
                    }
                },
                "required": ["config"],
                "additionalProperties": false
            }
        },
        {
            "name": "stop_op",
            "description": "Cancel the running operation (kills the whole process tree). Idempotent. Omit op_id to stop the operation started by this session.",
            "inputSchema": {
                "type": "object",
                "properties": { "op_id": { "type": "string" } },
                "additionalProperties": false
            }
        },
        {
            "name": "preview_dataset",
            "description": "Preview the first rows of a dataset source (canonical prompt/completion form) without materializing it for training.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": { "type": "object", "description": "A data spec: {source: hf|local|synthetic|klayer, <source>: {...}}" },
                    "rows": { "type": "integer", "description": "Rows to preview (default 5)" }
                },
                "required": ["source"],
                "additionalProperties": false
            }
        },
        {
            "name": "export_guide",
            "description": "How to export a finished run so it works in Ollama/llama.cpp — returns requirement checks (converter, python envs, ollama on PATH, disk), the exact commands for this workspace, and the known failure modes. Key rules baked in: convert with the bundled llama.cpp converter (never import safetensors straight into Ollama — silently broken for some architectures), merge train-run adapters into their base model first, and always give the Modelfile the chat TEMPLATE + stop tokens.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "artifact_path": { "type": "string", "description": "The run's artifact path (from list_runs/get_run) — commands are rendered with it" },
                    "kind": { "type": "string", "description": "train | merge — train adds the adapter-merge step (default merge)" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "plan_inference",
            "description": "Inspect a local GGUF or checkpoint and return a GPU-first llama.cpp/vLLM/Sytra plan: architecture from metadata (not the filename), estimated VRAM/RAM at 2k/4k/8k, n-gpu-layers, and whether the plan fits the detected hardware envelope. Does not start a server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Absolute GGUF file or complete model directory" },
                    "context": { "type": "integer", "description": "Context tokens (default 4096)" },
                    "export_runtimes": { "type": "boolean", "description": "Also write Ollama Modelfile and LM Studio sidecar next to the GGUF" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        }
    ])
    })
}

