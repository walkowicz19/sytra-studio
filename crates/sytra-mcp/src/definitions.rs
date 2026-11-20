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
            "description": "List the pinned Hugging Face catalog Sytra can download (verified Xet transfers). Each entry includes alerts (VRAM/RAM, MoE hybrid, gated license, Qwen3.5≠Qwen2, never import raw SafeTensors into Ollama). download_model requires an exact model_id; start_train also accepts local LoRA adapters whose base appears here.",
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
                    "method": { "type": "string", "description": "linear | slerp | multi_slerp | nuslerp | ties | dare_ties | dare_linear | task_arithmetic | passthrough | moe" },
                    "base_model": { "type": "string", "description": "Base model for task-vector methods (optional for multi_slerp/nuslerp) — enables the lineage-mismatch check" }
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
            "description": "Start a fine-tuning run (LoRA/QLoRA/DoRA; sft/dpo/orpo/cpo). Returns op_id immediately — poll get_run for progress. Only one operation runs at a time. `config` follows the run.yaml contract; unspecified fields get sensible defaults. Minimum: {\"model\": \"<catalog model_id>\", \"data\": {\"source\": \"local\", \"local\": {\"path\": \"data.jsonl\", \"format\": \"jsonl\", \"mapping\": {\"prompt\": \"prompt\", \"completion\": \"completion\"}}}}. Data sources: hf {repo_id, split}, local {path, format, mapping}, synthetic {generator_model, judge_model, mode, count, topic}, klayer {domain} (MCP export_dataset; omit domain to export all reviewed+user rows). To post-train, set config.init_adapter to a local adapter directory and choose a new output.adapter_path. To resume all Trainer state, set output.resume_from_checkpoint. The output is a LoRA ADAPTER, not a full model — call export_guide for how to merge it and run it in Ollama.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config": serde_json::from_str::<Value>(include_str!("../../sytra-contracts/run.schema.json")).expect("valid training schema")
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
            "name": "sytra_dataset_merge",
            "description": "Combine local JSONL or Parquet conversations into canonical JSONL. Weighted mode requires weights and total_samples. Filtering occurs only if explicitly requested. Paths must be inside the workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "inputs": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 150 },
                    "output_dir": { "type": "string" },
                    "weights": { "type": "array", "items": { "type": "number", "exclusiveMinimum": 0 } },
                    "total_samples": { "type": "integer", "minimum": 1 },
                    "shard_rows": { "type": "integer", "minimum": 0 },
                    "curation": { "type": "object", "properties": {
                        "exact_duplicates": { "type": "boolean" }, "near_duplicates": { "type": "boolean" },
                        "similarity": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 },
                        "repetitive": { "type": "boolean" }, "min_tokens": { "type": "integer", "minimum": 0 },
                        "sanitize_syntax": { "type": "boolean", "description": "Opt-in cleanup of leaked chat template tags, non-printable control characters, and unclosed Markdown code fences. Preserves text untouched when omitted or false." },
                        "drop_invalid": { "type": "boolean", "description": "Explicitly discard and count malformed or incomplete conversations. Invalid JSON still fails the operation." },
                        "semantic": { "type": "object", "description": "Opt-in review with an existing local Transformers model (CUDA/4-bit when available, otherwise CPU). Invalid verdicts or overlong prompts abort the operation.", "properties": {
                            "model_path": { "type": "string" }, "min_density": { "type": "number", "minimum": 0, "maximum": 1 },
                            "max_input_tokens": { "type": "integer", "minimum": 64, "maximum": 32768 },
                            "device": { "type": "string", "enum": ["auto", "cpu", "cuda"] },
                            "load_in_4bit": { "type": "boolean" }
                        }, "required": ["model_path"], "additionalProperties": false },
                        "context": { "type": "object", "description": "Opt-in native tokenizer filter using a local model tokenizer; supports max_seq_length upper bound and/or native min_tokens count without truncation.", "properties": {
                            "tokenizer_path": { "type": "string" }, "max_seq_length": { "type": "integer", "minimum": 2, "maximum": 1048576 },
                            "min_tokens": { "type": "integer", "minimum": 0, "maximum": 1048576 }
                        }, "required": ["tokenizer_path"], "additionalProperties": false },
                        "benchmark_files": { "type": "array", "maxItems": 50, "items": { "type": "string" }, "description": "Local JSONL/Parquet benchmark corpora; only supplied text is indexed." }
                    }, "additionalProperties": false },
                    "seed": { "type": "integer" }
                },
                "required": ["inputs", "output_dir"], "additionalProperties": false
            }
        },
        {
            "name": "sytra_dataset_split",
            "description": "Split a local JSONL or Parquet dataset deterministically into training and validation files, keeping identical conversations together. Paths must be inside the workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string" }, "output_dir": { "type": "string" },
                    "validation_fraction": { "type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1 },
                    "seed": { "type": "integer" }
                },
                "required": ["input", "output_dir"], "additionalProperties": false
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
                    "export_runtimes": { "type": "boolean", "description": "Also write Ollama Modelfile and LM Studio sidecar next to the GGUF" },
                    "allow_unverified": { "type": "boolean", "description": "Allow Sytra-Helix preflight without .sytra-oracle.json" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_local_models",
            "description": "Scan local Hugging Face cache, LM Studio directories, workspace runs, and an optional custom directory for GGUF and SafeTensors models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "custom_dir": { "type": "string", "description": "Optional additional directory to scan" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "get_download_status",
            "description": "Read progress and completion status of the active or most recent verified Hugging Face model download.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dest_dir": { "type": "string", "description": "Destination directory used in download_model (omit for default HF cache)" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "cancel_download",
            "description": "Cancel an active Hugging Face model download and clear its status file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dest_dir": { "type": "string", "description": "Destination directory used in download_model (omit for default HF cache)" }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "build_moe_index",
            "description": "Build the Sytra-Helix zero-copy .sytra-runtime.json byte-range index for a SafeTensors MoE checkpoint directory.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Path to complete SafeTensors model directory" },
                    "adapter": { "type": "string", "description": "Compiled adapter id or 'auto' (default auto)" },
                    "expert_format": { "type": "string", "description": "Expert weight format such as packed_int4_group, bf16, or 'auto' (default auto)" },
                    "expert_regex": { "type": "string", "description": "Optional custom regex to match expert tensor names" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "generate_moe_oracle",
            "description": "Generate a checkpoint-bound .sytra-oracle.json reference verification file for a SafeTensors checkpoint using the provisioned train environment.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Path to complete SafeTensors model directory" },
                    "trust_remote_code": { "type": "boolean", "description": "Pass --trust-remote-code when loading reference weights" },
                    "device_map": { "type": "string", "description": "Optional PyTorch device_map (e.g. 'cpu' or 'auto')" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "set_inference_settings",
            "description": "Update persistent global inference settings shared by the GUI and MCP server.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "default_context_window": { "type": "integer", "minimum": 1, "maximum": 2097152 },
                    "default_temperature": { "type": "number", "minimum": 0, "maximum": 2 },
                    "enable_flash_attention": { "type": "boolean" },
                    "kv_cache_quant": { "type": "string", "enum": ["fp16", "q8_0", "q4_0"] },
                    "vram_limit_mb": { "type": ["integer", "null"], "minimum": 1 },
                    "cpu_kv_cache": { "type": "boolean" },
                    "vram_expert_cache_mb": { "type": ["integer", "null"], "minimum": 1 }
                },
                "additionalProperties": false
            }
        },
        {
            "name": "get_model_settings",
            "description": "Load per-model inference overrides stored for a local GGUF or SafeTensors checkpoint.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Path to GGUF file or SafeTensors directory" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "set_model_settings",
            "description": "Save per-model inference overrides (context, temperature, VRAM limits, KV cache, allow_unverified) for a local model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Path to GGUF file or SafeTensors directory" },
                    "context_length": { "type": ["integer", "null"], "minimum": 1, "maximum": 2097152 },
                    "temperature": { "type": ["number", "null"], "minimum": 0, "maximum": 2 },
                    "vram_limit_mb": { "type": ["integer", "null"], "minimum": 1 },
                    "vram_guardrail_mb": { "type": ["integer", "null"], "minimum": 0 },
                    "kv_cache_quant": { "type": ["string", "null"], "enum": ["fp16", "q8_0", "q4_0", null] },
                    "cpu_kv_cache": { "type": ["boolean", "null"] },
                    "flash_attention": { "type": ["boolean", "null"] },
                    "idle_ttl_minutes": { "type": ["integer", "null"], "minimum": 0, "maximum": 10080 },
                    "draft_model_path": { "type": ["string", "null"] },
                    "allow_unverified": { "type": ["boolean", "null"] }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "start_server",
            "description": "Run preflight validation and start the local OpenAI-compatible inference server (llama.cpp, vLLM, or Sytra-Helix engine) on 127.0.0.1:8080.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_path": { "type": "string", "description": "Path to GGUF file or SafeTensors directory" },
                    "context": { "type": "integer", "minimum": 1 },
                    "vram_limit": { "type": "integer", "minimum": 1 },
                    "cpu_kv_cache": { "type": "boolean" },
                    "allow_unverified": { "type": "boolean", "description": "Allow Sytra-Helix serving without .sytra-oracle.json" }
                },
                "required": ["model_path"],
                "additionalProperties": false
            }
        },
        {
            "name": "stop_server",
            "description": "Stop the MCP-managed local inference server if running.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "server_status",
            "description": "Report whether the local inference server is running and reachable on 127.0.0.1:8080.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "convert_model",
            "description": "Convert a local SafeTensors model directory to GGUF using llama.cpp.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Path to SafeTensors model directory" },
                    "outtype": { "type": "string", "description": "Quantization/output type (default auto)" },
                    "outfile": { "type": "string", "description": "Optional output GGUF path" }
                },
                "required": ["model"],
                "additionalProperties": false
            }
        },
        {
            "name": "export_model",
            "description": "Export a local GGUF or model directory for Ollama / LM Studio runtimes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Path to GGUF file or model directory" },
                    "name": { "type": "string", "description": "Optional exported model name" },
                    "context": { "type": "integer", "minimum": 1 }
                },
                "required": ["model"],
                "additionalProperties": false
            }
        },
        {
            "name": "list_chat_skills",
            "description": "List local SKILL.md chat skills discovered in the workspace and user skill directories.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "list_chat_agents",
            "description": "List configured Sytra chat agent profiles (e.g., Lil Sytra and custom agents).",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        },
        {
            "name": "get_voice_status",
            "description": "Return Kokoro-82M / Whisper voice runtime status, model readiness, and available voices.",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }
    ])
    })
}

