//! Sytra Studio MCP server — lets any MCP client (Claude Code, Codex,
//! Cursor, Antigravity, …) drive fine-tuning and merging headlessly.
//!
//! Transport: newline-delimited JSON-RPC 2.0 over stdio (the MCP stdio
//! transport). Hand-rolled on purpose: a tools-only server needs four
//! methods (`initialize`, `tools/list`, `tools/call`, `ping`), which is
//! not worth an SDK dependency.
//!
//! The server reuses the exact host machinery the GUI uses (JobRunner,
//! RunArchive, Guider, ResourceGuard, commands::start_op), so agent-driven
//! runs land in the same archive and obey the same validation gates.

mod definitions;
mod protocol;

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Value};
use uuid::Uuid;

use sytra_contracts::{
    guider::{Guider, HardwareCapabilities},
    merge_config::{MergeConfig, MergeMethod},
    operation::{MergeSpec, Operation, TrainSpec},
    run_config::RunConfig,
    OpStatus, TelemetryLine,
};
use sytra_host::{
    commands, job_runner::JobRunner, materialize::materialize_dataset_for_config,
    resource_guard::ResourceGuard, run_archive::RunArchive, BackendResolver, DownloadService,
    EnvProvisioner, plan_inference, resolve_workspace,
};

use crate::definitions::tool_definitions;
use crate::protocol::{respond, respond_error, PROTOCOL_VERSION};

struct Server {
    workspace: PathBuf,
    runner: JobRunner,
    archive: RunArchive,
    guider: Guider,
    downloads: DownloadService,
    rt: tokio::runtime::Runtime,
    current_op: Mutex<Option<Uuid>>,
}

impl Server {
    fn operation_guard(&self) -> Result<ResourceGuard, String> {
        let detected_ram = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM; refusing to estimate".to_string()
        })?;
        let ram_limit = sytra_host::settings::AppSettings::load(&self.workspace)
            .effective_main_memory_mb(detected_ram);
        Ok(ResourceGuard::new(
            BackendResolver::detect_system_vram_mb().unwrap_or(0),
            ram_limit,
            500 * 1024,
        ))
    }

    fn new() -> Self {
        let workspace = resolve_workspace();

        let runs_dir = workspace.join("runs");
        std::fs::create_dir_all(&runs_dir).ok();

        Self {
            runner: JobRunner::new(&workspace),
            archive: RunArchive::new(&runs_dir),
            guider: Guider::new(),
            downloads: DownloadService::new(&workspace),
            rt: tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime"),
            current_op: Mutex::new(None),
            workspace,
        }
    }

    fn transcripts_dir(&self) -> PathBuf {
        self.workspace.join("runs").join("transcripts")
    }

    /// Drains the telemetry receiver of a started operation into a
    /// transcript file, and settles the archive status when the runner
    /// exits (unless a user stop already wrote Stopped).
    fn spawn_drain(&self, op_id: Uuid, mut rx: tokio::sync::mpsc::Receiver<TelemetryLine>) {
        let transcripts = self.transcripts_dir();
        let runs_dir = self.workspace.join("runs");
        std::thread::spawn(move || {
            std::fs::create_dir_all(&transcripts).ok();
            let path = transcripts.join(format!("{op_id}.jsonl"));
            let mut file = std::fs::File::create(&path).ok();
            let mut saw_error = false;
            while let Some(line) = rx.blocking_recv() {
                if let TelemetryLine::Event { event, .. } = &line {
                    if event == "error" {
                        saw_error = true;
                    }
                }
                if let (Some(f), Ok(s)) = (file.as_mut(), serde_json::to_string(&line)) {
                    let _ = writeln!(f, "{s}");
                }
            }
            let archive = RunArchive::new(&runs_dir);
            if let Ok(mut record) = archive.load(op_id) {
                if record.status == OpStatus::Running {
                    record.status = if saw_error {
                        OpStatus::Error
                    } else {
                        OpStatus::Done
                    };
                    let _ = archive.store(&record);
                }
            }
        });
    }

    fn write_op_yaml(&self, name_prefix: &str, value: &Value) -> Result<PathBuf, String> {
        let dir = self.workspace.join("runs").join("mcp");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = dir.join(format!("{name_prefix}-{stamp}.yaml"));
        let yaml = serde_yaml::to_string(value).map_err(|e| e.to_string())?;
        std::fs::write(&path, yaml).map_err(|e| e.to_string())?;
        Ok(path)
    }

    // ── Tools ────────────────────────────────────────────────────────────

    fn tool_get_status(&self) -> Result<Value, String> {
        let current = *self.current_op.lock().unwrap();
        let envs = EnvProvisioner::new(&self.workspace).status_report();
        Ok(json!({
            "running": self.runner.is_running(),
            "current_op_id": current.map(|u| u.to_string()),
            "backend": format!("{:?}", BackendResolver::resolve(sytra_contracts::run_config::BackendKind::Auto)).to_lowercase(),
            "vram_mb": BackendResolver::detect_system_vram_mb(),
            "ram_mb": BackendResolver::detect_system_ram_mb(),
            "workspace": self.workspace.display().to_string(),
            "envs": envs,
        }))
    }

    fn tool_list_catalog(&self) -> Result<Value, String> {
        let vram = BackendResolver::detect_system_vram_mb();
        let ram = BackendResolver::detect_system_ram_mb();
        let items: Vec<Value> = self
            .guider
            .catalog()
            .iter()
            .map(|entry| {
                let alerts = sytra_contracts::alerts_for(entry, vram, ram);
                let level = sytra_contracts::peak_alert_level(&alerts);
                let mut value = serde_json::to_value(entry).unwrap_or(Value::Null);
                if let Some(obj) = value.as_object_mut() {
                    obj.insert("alerts".into(), serde_json::to_value(&alerts).unwrap_or(Value::Null));
                    obj.insert("alert_level".into(), json!(level));
                    obj.insert("format".into(), json!(entry.inferred_format()));
                    obj.insert("download_size_gb".into(), json!(entry.download_size_gb()));
                    obj.insert("allows_download".into(), json!(entry.allows_download()));
                }
                value
            })
            .collect();
        Ok(json!({
            "count": items.len(),
            "download_policy": "Downloads are limited to these exact Hugging Face model_id values. Sytra uses the verified Xet/hf-xet downloader. Read alerts before starting a pull.",
            "models": items,
        }))
    }

    fn tool_guider_recommend(&self, args: &Value) -> Result<Value, String> {
        let hw = HardwareCapabilities {
            accelerator: args
                .get("accelerator")
                .and_then(|v| v.as_str())
                .unwrap_or("cuda")
                .to_string(),
            total_vram_mb: args
                .get("vram_mb")
                .and_then(|v| v.as_u64())
                .or_else(BackendResolver::detect_system_vram_mb)
                .ok_or_else(|| "Could not detect VRAM; pass vram_mb".to_string())?,
            total_ram_mb: args
                .get("ram_mb")
                .and_then(|v| v.as_u64())
                .or_else(BackendResolver::detect_system_ram_mb)
                .ok_or_else(|| "Could not detect RAM; pass ram_mb".to_string())?,
        };
        serde_json::to_value(self.guider.recommend(&hw)).map_err(|e| e.to_string())
    }

    fn tool_merge_check(&self, args: &Value) -> Result<Value, String> {
        let models: Vec<String> =
            serde_json::from_value(args.get("models").cloned().ok_or("missing 'models'")?)
                .map_err(|e| e.to_string())?;
        let method: MergeMethod =
            serde_json::from_value(args.get("method").cloned().ok_or("missing 'method'")?)
                .map_err(|e| format!("bad method: {e}"))?;
        let base = args.get("base_model").and_then(|v| v.as_str());
        serde_json::to_value(self.guider.merge_check_with_base(base, &models, method))
            .map_err(|e| e.to_string())
    }

    fn tool_list_runs(&self) -> Result<Value, String> {
        let records = self.archive.list_summaries().map_err(|e| e.to_string())?;
        serde_json::to_value(records).map_err(|e| e.to_string())
    }

    fn tool_get_run(&self, args: &Value) -> Result<Value, String> {
        let op_id = parse_op_id(args)?;
        let tail = args.get("tail").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let record = self.archive.load(op_id).map_err(|e| e.to_string())?;
        let transcript_path = self.transcripts_dir().join(format!("{op_id}.jsonl"));
        let tail_lines = sytra_host::transcript::tail_jsonl(&transcript_path, tail);
        Ok(json!({
            "op_id": record.op_id.to_string(),
            "kind": record.kind,
            "status": record.status,
            "artifact_path": record.artifact_path.display().to_string(),
            "telemetry_tail": tail_lines,
        }))
    }

    fn tool_start_train(&self, args: &Value) -> Result<Value, String> {
        let user_config = args.get("config").cloned().ok_or("missing 'config'")?;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut merged = json!({
            "version": 1,
            "run_id": null,
            "train_mode": "sft",
            "backend": { "kind": "auto", "judge_model": null },
            "adapter": {
                "type": "qlora", "rank": 16, "alpha": 32, "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "quant_bits": 4
            },
            "optim": {
                "learning_rate": 2.0e-4, "schedule": "cosine", "warmup_steps": 20,
                "weight_decay": 0.0, "grad_accumulation_steps": 8
            },
            "train": {
                "max_steps": 200, "epochs": null, "batch_size": 2,
                "max_seq_len": 2048, "save_every": 50, "packing": false
            },
            "algo": {},
            "output": {
                "adapter_path": format!("runs/mcp/adapter-{stamp}"),
                "resume_from": null
            }
        });
        deep_merge(&mut merged, &user_config);

        let mut run_config: RunConfig =
            serde_json::from_value(merged).map_err(|e| format!("invalid train config: {e}"))?;

        // Materialize the dataset exactly like the GUI does.
        let dataset_dir = self
            .workspace
            .join("runs")
            .join("mcp")
            .join(format!("dataset-{stamp}"));
        std::fs::create_dir_all(&dataset_dir).map_err(|e| e.to_string())?;
        self.rt.block_on(materialize_dataset_for_config(
            &mut run_config.data,
            run_config.train_mode,
            &dataset_dir,
        ))?;

        let config_val = serde_json::to_value(&run_config).map_err(|e| e.to_string())?;
        let config_path = self.write_op_yaml("run", &config_val)?;

        let op = Operation::Train(TrainSpec {
            config: run_config,
            config_path,
        });
        let guard = self.operation_guard()?;
        let (op_id, rx) =
            commands::start_op(op, &self.runner, &self.archive, &guard, &self.guider)?;
        self.spawn_drain(op_id, rx);
        *self.current_op.lock().unwrap() = Some(op_id);

        Ok(json!({
            "op_id": op_id.to_string(),
            "message": "Training started. Poll get_run with this op_id to follow progress; status becomes 'done' or 'error' when finished."
        }))
    }

    fn tool_start_merge(&self, args: &Value) -> Result<Value, String> {
        let user_config = args.get("config").cloned().ok_or("missing 'config'")?;

        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut merged = json!({
            "version": 1,
            "op_id": null,
            "merge_method": "dare_ties",
            "base_model": null,
            "dtype": "bfloat16",
            "models": [],
            "tokenizer": { "source": "base" },
            "compat": { "verdict": "green", "fingerprint": null },
            "output": { "model_path": format!("runs/mcp/merged-{stamp}") }
        });
        deep_merge(&mut merged, &user_config);

        // Accept plain model-id strings for convenience.
        if let Some(models) = merged.get_mut("models").and_then(|m| m.as_array_mut()) {
            for entry in models.iter_mut() {
                if let Value::String(s) = entry {
                    *entry = json!({ "model": s });
                }
            }
        }

        let mut merge_config: MergeConfig =
            serde_json::from_value(merged).map_err(|e| format!("invalid merge config: {e}"))?;

        // The host fills compat from Guider.merge_check — never trust the
        // caller's verdict (Contract 2).
        let model_refs: Vec<String> = merge_config
            .models
            .iter()
            .map(|m| m.model.clone())
            .collect();
        let compat = self.guider.merge_check_with_base(
            merge_config.base_model.as_deref(),
            &model_refs,
            merge_config.merge_method,
        );
        merge_config.compat.verdict = compat.verdict;

        let config_val = serde_json::to_value(&merge_config).map_err(|e| e.to_string())?;
        let config_path = self.write_op_yaml("merge", &config_val)?;

        let op = Operation::Merge(MergeSpec {
            config: merge_config,
            config_path,
        });
        let guard = self.operation_guard()?;
        let (op_id, rx) =
            commands::start_op(op, &self.runner, &self.archive, &guard, &self.guider)?;
        self.spawn_drain(op_id, rx);
        *self.current_op.lock().unwrap() = Some(op_id);

        Ok(json!({
            "op_id": op_id.to_string(),
            "compat_verdict": compat.verdict,
            "compat_reason": compat.reason,
            "message": "Merge started. Poll get_run with this op_id to follow progress."
        }))
    }

    fn tool_stop_op(&self, args: &Value) -> Result<Value, String> {
        let op_id = match parse_op_id(args) {
            Ok(id) => id,
            Err(_) => self
                .current_op
                .lock()
                .unwrap()
                .ok_or("no op_id given and no operation was started by this server")?,
        };
        commands::stop_op(op_id, &self.runner, &self.archive)?;
        Ok(json!({ "op_id": op_id.to_string(), "status": "stopped" }))
    }

    fn tool_preview_dataset(&self, args: &Value) -> Result<Value, String> {
        let source = args.get("source").cloned().ok_or("missing 'source'")?;
        let rows = args.get("rows").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let data_spec: sytra_contracts::run_config::DataSpec =
            serde_json::from_value(source).map_err(|e| format!("bad source spec: {e}"))?;

        let (kind, params) = match &data_spec {
            sytra_contracts::run_config::DataSpec::Hf { hf, .. } => (
                sytra_host::SourceKind::Hf,
                serde_json::to_value(hf).unwrap(),
            ),
            sytra_contracts::run_config::DataSpec::Local { local, .. } => (
                sytra_host::SourceKind::Local,
                serde_json::to_value(local).unwrap(),
            ),
            sytra_contracts::run_config::DataSpec::Synthetic { synthetic, .. } => (
                sytra_host::SourceKind::Synthetic,
                serde_json::to_value(synthetic).unwrap(),
            ),
            sytra_contracts::run_config::DataSpec::Klayer { klayer, .. } => (
                sytra_host::SourceKind::Klayer,
                serde_json::to_value(klayer).unwrap(),
            ),
            sytra_contracts::run_config::DataSpec::Multi { .. } => {
                return Err("preview the individual datasets of a multi spec instead".into())
            }
        };

        let spec = sytra_host::DatasetSpec {
            source: kind,
            train_mode: sytra_contracts::run_config::TrainMode::Sft,
            params,
        };
        let provider = sytra_host::get_datasource(kind);
        let preview = self
            .rt
            .block_on(provider.preview(&spec, rows))
            .map_err(|e| format!("preview error: {e}"))?;

        Ok(json!({ "rows": preview.rows, "total_estimate": preview.total_estimate }))
    }

    fn tool_get_settings(&self) -> Result<Value, String> {
        let settings = sytra_host::settings::AppSettings::load(&self.workspace);
        let detected_ram_mb = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM".to_string()
        })?;
        Ok(json!({
            "hf_cache_dir": settings.effective_hf_cache(&self.workspace).display().to_string(),
            "is_custom": settings.hf_cache_dir.is_some(),
            "main_memory_limit_mb": settings.main_memory_limit_mb,
            "effective_main_memory_mb": settings.effective_main_memory_mb(detected_ram_mb),
            "detected_ram_mb": detected_ram_mb,
            "memory_limit_note": "main_memory_limit_mb gates preflight estimates only; it does not cap process RSS. Prefer one shared MCP config and launch sytra-mcp directly instead of npx.",
        }))
    }

    fn tool_set_cache_dir(&self, args: &Value) -> Result<Value, String> {
        let path = args.get("path").and_then(|v| v.as_str());
        if let Some(p) = path {
            let pb = std::path::PathBuf::from(p);
            std::fs::create_dir_all(&pb)
                .map_err(|e| format!("cannot create directory {p}: {e}"))?;
        }
        let mut settings = sytra_host::settings::AppSettings::load(&self.workspace);
        settings.hf_cache_dir = path.map(std::path::PathBuf::from);
        settings.save(&self.workspace)?;
        Ok(json!({
            "hf_cache_dir": settings.effective_hf_cache(&self.workspace).display().to_string(),
            "is_custom": settings.hf_cache_dir.is_some(),
            "note": "Applies to the next started operation. Existing cached files are not moved automatically."
        }))
    }

    fn tool_set_main_memory_limit(&self, args: &Value) -> Result<Value, String> {
        let limit_mb = args.get("limit_mb").and_then(|v| v.as_u64());
        let detected_ram_mb = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM".to_string()
        })?;
        if let Some(limit) = limit_mb {
            if limit < 2048 || limit > detected_ram_mb {
                return Err(format!(
                    "limit_mb must be between 2048 and {detected_ram_mb}"
                ));
            }
        }
        let mut settings = sytra_host::settings::AppSettings::load(&self.workspace);
        settings.main_memory_limit_mb = limit_mb;
        settings.save(&self.workspace)?;
        Ok(json!({
            "main_memory_limit_mb": settings.main_memory_limit_mb,
            "effective_main_memory_mb": settings.effective_main_memory_mb(detected_ram_mb),
            "detected_ram_mb": detected_ram_mb,
            "note": "Applies to preflight checks for operations started after this change."
        }))
    }

    fn tool_export_guide(&self, args: &Value) -> Result<Value, String> {
        let artifact = args
            .get("artifact_path")
            .and_then(|v| v.as_str())
            .unwrap_or("<model directory>");
        let kind = args.get("kind").and_then(|v| v.as_str()).unwrap_or("merge");
        let ws = &self.workspace;

        // Detect effective RAM to decide whether to use disk-backed temp files during
        // GGUF conversion. On machines with <= 20 GB RAM a 7B model at Q8_0 pushes the
        // converter past the physical memory limit and triggers an OOM kill.
        let detected_ram_mb = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM".to_string()
        })?;
        let settings = sytra_host::settings::AppSettings::load(ws);
        let effective_ram_mb = settings.effective_main_memory_mb(detected_ram_mb);
        let low_ram = effective_ram_mb <= 20_480;

        let converter = ws
            .join(".tools")
            .join("llama.cpp")
            .join("convert_hf_to_gguf.py");
        let envs = EnvProvisioner::new(ws);
        let merge_py = envs.merge_python_path();
        let train_py = envs.train_python_path();
        let ollama_ok = std::process::Command::new("ollama")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        let merged_dir = if kind == "train" {
            format!("{artifact}-merged")
        } else {
            artifact.to_string()
        };

        let mut steps = Vec::new();
        if kind == "train" {
            steps.push(json!({
                "step": "merge_adapter",
                "why": "Training outputs a LoRA adapter, not a full model. It must be merged into its base model before GGUF conversion.",
                "command_env": train_py.display().to_string(),
                "python_snippet": format!(
                    "from transformers import AutoModelForCausalLM, AutoTokenizer\nfrom peft import PeftModel\nimport torch\nimport gc\nprint('Loading base model...')\nmodel = AutoModelForCausalLM.from_pretrained('<base model id>', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='cpu')\ngc.collect()\nprint('Loading adapter and merging...')\nmodel = PeftModel.from_pretrained(model, r'{artifact}').merge_and_unload()\ngc.collect()\nprint('Saving merged model...')\nmodel.save_pretrained(r'{merged_dir}', safe_serialization=True, max_shard_size='2GB')\nAutoTokenizer.from_pretrained(r'{artifact}').save_pretrained(r'{merged_dir}')\nprint('Merge complete!')"
                ),
            }));
        }
        steps.push(json!({
            "step": "convert_gguf",
            "why": if low_ram {
                format!("Ollama's own safetensors importer silently produces broken output for some architectures (verified on Qwen2.5) — always convert with llama.cpp. Q8_0 is the best quality/size balance. --use-temp-file is added automatically because your effective RAM ({} MB) is at or below 20 GB — it swaps intermediate tensors to disk to prevent an OOM crash.", effective_ram_mb)
            } else {
                "Ollama's own safetensors importer silently produces broken output for some architectures (verified on Qwen2.5) — always convert with llama.cpp. Q8_0 is the best quality/size balance.".to_string()
            },
            "command": if low_ram {
                format!(
                    "\"{}\" -u \"{}\" \"{merged_dir}\" --outtype q8_0 --outfile model.q8_0.gguf --use-temp-file",
                    merge_py.display(), converter.display()
                )
            } else {
                format!(
                    "\"{}\" -u \"{}\" \"{merged_dir}\" --outtype q8_0 --outfile model.q8_0.gguf",
                    merge_py.display(), converter.display()
                )
            },
            "low_ram_mode": low_ram,
            "effective_ram_mb": effective_ram_mb,
        }));
        steps.push(json!({
            "step": "modelfile",
            "why": "Without TEMPLATE and stop tokens the model emits raw endless text. This template fits ChatML models (Qwen and most coder models).",
            "content": "FROM ./model.q8_0.gguf\n\nTEMPLATE \"\"\"{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n\"\"\"\n\nPARAMETER stop <|im_start|>\nPARAMETER stop <|im_end|>\nPARAMETER num_ctx 8192",
        }));
        steps.push(json!({
            "step": "ollama_create",
            "command": "ollama create <model-name> -f Modelfile",
            "why": "Imports the GGUF into Ollama's model store (set OLLAMA_MODELS to a large drive first if disk space is tight).",
        }));

        Ok(json!({
            "requirements": {
                "llama_cpp_converter": { "path": converter.display().to_string(), "present": converter.exists() },
                "merge_env_python": { "path": merge_py.display().to_string(), "present": merge_py.exists() },
                "train_env_python": { "path": train_py.display().to_string(), "present": train_py.exists(), "needed_for": "adapter merging (train runs only)" },
                "ollama_on_path": ollama_ok,
                "disk_space": "roughly 1.5x the model size free (a 7B at Q8_0 is ~8 GB, plus ~15 GB temporarily for the merged bf16 model on train runs)",
            },
            "steps": steps,
            "gotchas": [
                "NEVER `ollama create` directly from a safetensors directory — use the llama.cpp converter first.",
                "Novel MoE architectures are backend-specific. Use Sytra's verified llama.cpp/vLLM preflight and do not assume an unsupported model can be paged from disk.",
                "Train runs produce an adapter; merge it into the base model before converting.",
                "The Modelfile must carry the model's chat TEMPLATE and stop tokens or output will be unusable.",
                "`ollama run` hangs when spawned without a terminal (TTY detection). To smoke-test programmatically, POST to http://127.0.0.1:11434/api/generate with {\"model\": ..., \"prompt\": ..., \"stream\": false} instead.",
                "First response after import is slow: the model loads from disk (~1-2 min for 8 GB on an HDD).",
            ],
        }))
    }

    fn tool_configure_fast_cache(&self, args: &Value) -> Result<Value, String> {
        let tokenless = args
            .get("tokenless")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let low_bit_mode = args
            .get("low_bit_mode")
            .and_then(|v| v.as_u64())
            .map(|v| v as u8);
        let vram_expert_cache_mb = args.get("vram_expert_cache_mb").and_then(|v| v.as_u64());

        let mut settings = sytra_host::settings::AppSettings::load(&self.workspace);
        settings.tokenless_download = tokenless;
        if low_bit_mode.is_some() {
            settings.low_bit_mode = low_bit_mode;
        }
        if vram_expert_cache_mb.is_some() {
            settings.vram_expert_cache_mb = vram_expert_cache_mb;
        }
        settings.save(&self.workspace).map_err(|e| e.to_string())?;

        Ok(json!({
            "tokenless_download": settings.tokenless_download,
            "low_bit_mode": settings.low_bit_mode,
            "vram_expert_cache_mb": settings.vram_expert_cache_mb,
            "message": "Verified download authentication and low-bit settings updated."
        }))
    }

    fn tool_download_model(&self, args: &Value) -> Result<Value, String> {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or("missing 'model' parameter")?;
        let entry = sytra_host::catalog::require_catalog_download(&self.guider, model)?;
        let vram = BackendResolver::detect_system_vram_mb();
        let ram = BackendResolver::detect_system_ram_mb();
        let alerts = sytra_contracts::alerts_for(entry, vram, ram);
        if let Some(blocked) = alerts.iter().find(|a| a.blocks_download) {
            return Err(format!("{} ({})", blocked.message, blocked.code));
        }
        let dest_dir = args.get("dest_dir").and_then(|v| v.as_str());
        let purpose = args
            .get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("inference");
        let quant = args.get("quant").and_then(|v| v.as_str());
        let revision = args.get("revision").and_then(|v| v.as_str());
        let started = self
            .downloads
            .start(model, purpose, dest_dir, quant, revision)?;
        let mut payload = serde_json::to_value(started).map_err(|e| e.to_string())?;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("alerts".into(), serde_json::to_value(&alerts).unwrap_or(Value::Null));
            obj.insert("alert_level".into(), json!(sytra_contracts::peak_alert_level(&alerts)));
            obj.insert("architecture".into(), json!(entry.architecture));
            obj.insert("license".into(), json!(entry.license));
        }
        Ok(payload)
    }

    fn tool_plan_inference(&self, args: &Value) -> Result<Value, String> {
        let model_path = args
            .get("model_path")
            .and_then(|v| v.as_str())
            .ok_or("missing 'model_path'")?;
        let context = args.get("context").and_then(|v| v.as_u64()).map(|v| v as usize);
        let export = args
            .get("export_runtimes")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let guard = self.operation_guard()?;
        if guard.total_vram_mb == 0 || guard.total_ram_mb == 0 {
            return Err("Hardware memory could not be detected; refusing to plan inference".into());
        }
        plan_inference(
            &self.workspace,
            model_path,
            guard.total_vram_mb,
            Some(guard.total_ram_mb),
            context,
            export,
        )
    }

    fn call_tool(&self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "get_status" => self.tool_get_status(),
            "get_settings" => self.tool_get_settings(),
            "set_cache_dir" => self.tool_set_cache_dir(args),
            "set_main_memory_limit" => self.tool_set_main_memory_limit(args),
            "configure_fast_cache" => self.tool_configure_fast_cache(args),
            "download_model" => self.tool_download_model(args),
            "list_catalog" => self.tool_list_catalog(),
            "guider_recommend" => self.tool_guider_recommend(args),
            "merge_check" => self.tool_merge_check(args),
            "list_runs" => self.tool_list_runs(),
            "get_run" => self.tool_get_run(args),
            "start_train" => self.tool_start_train(args),
            "start_merge" => self.tool_start_merge(args),
            "stop_op" => self.tool_stop_op(args),
            "preview_dataset" => self.tool_preview_dataset(args),
            "export_guide" => self.tool_export_guide(args),
            "plan_inference" => self.tool_plan_inference(args),
            other => Err(format!("unknown tool: {other}")),
        }
    }
}

fn parse_op_id(args: &Value) -> Result<Uuid, String> {
    args.get("op_id")
        .and_then(|v| v.as_str())
        .ok_or("missing 'op_id'".to_string())
        .and_then(|s| Uuid::parse_str(s).map_err(|_| format!("invalid op_id: {s}")))
}

/// Recursively merges `patch` into `base` (objects merge, everything else
/// replaces).
fn deep_merge(base: &mut Value, patch: &Value) {
    match (base, patch) {
        (Value::Object(base_map), Value::Object(patch_map)) => {
            for (k, v) in patch_map {
                match base_map.get_mut(k) {
                    Some(slot) => deep_merge(slot, v),
                    None => {
                        base_map.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot, v) => *slot = v.clone(),
    }
}

fn main() {
    let server = Server::new();
    let stdin = std::io::stdin();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue; // not JSON-RPC; ignore
        };

        let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let id = msg.get("id").cloned();

        // Notifications (no id) never get a response.
        let Some(id) = id else { continue };

        match method {
            "initialize" => {
                let client_version = msg
                    .pointer("/params/protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or(PROTOCOL_VERSION);
                respond(
                    &id,
                    json!({
                        "protocolVersion": client_version,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "sytra-studio", "version": env!("CARGO_PKG_VERSION") },
                        "instructions": "Sytra Studio: local LLM fine-tuning and merging. Typical flow — (1) get_status for hardware, (2) list_catalog or guider_recommend to pick a model, (3) start_train or merge_check + start_merge, (4) poll get_run with the returned op_id until status is done/error, (5) stop_op to cancel. One operation at a time; artifacts land under the workspace runs/ directory."
                    }),
                );
            }
            "ping" => respond(&id, json!({})),
            "tools/list" => respond(&id, json!({ "tools": tool_definitions() })),
            "tools/call" => {
                let name = msg
                    .pointer("/params/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let empty = json!({});
                let args = msg.pointer("/params/arguments").unwrap_or(&empty);
                match server.call_tool(name, args) {
                    Ok(result) => {
                        let text = serde_json::to_string(&result)
                            .unwrap_or_else(|_| result.to_string());
                        respond(
                            &id,
                            json!({
                                "content": [{ "type": "text", "text": text }],
                                "isError": false
                            }),
                        );
                    }
                    Err(err) => {
                        respond(
                            &id,
                            json!({
                                "content": [{ "type": "text", "text": err }],
                                "isError": true
                            }),
                        );
                    }
                }
            }
            _ => respond_error(&id, -32601, &format!("method not found: {method}")),
        }
    }

    // stdin closed — the MCP client is gone. An unattended training or
    // merge subprocess must not outlive its operator: stop it and let
    // stop_op record the Stopped status. (stop is idempotent, so this is
    // a no-op when the operation already finished.)
    let orphan = *server.current_op.lock().unwrap();
    if let Some(op_id) = orphan {
        if let Err(err) = commands::stop_op(op_id, &server.runner, &server.archive) {
            eprintln!("failed to stop orphaned op: {err}");
        }
    }
    server.downloads.shutdown();
}
