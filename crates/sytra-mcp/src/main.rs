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

#![recursion_limit = "256"]

mod definitions;
mod protocol;

use std::io::{BufRead, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::de::DeserializeOwned;
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
    chat_agents, chat_skills, commands, convert,
    ensure_runner_deployed,
    inference::plan_inference_with_options,
    job_runner::JobRunner,
    materialize::prepare_training_datasets,
    mcp_install::{
        install_mcp_clients, mcp_server_entry, status_report, uninstall_mcp_clients, McpClientKind,
    },
    model_settings, resolve_workspace,
    resource_guard::ResourceGuard,
    run_archive::RunArchive,
    scan_local_models,
    settings::{self, AppSettings, InferenceSettingsInput},
    BackendResolver, ChatServer, DownloadService, EnvProvisioner, VoiceService,
};

use crate::definitions::tool_definitions;
use crate::protocol::{respond, respond_error, PROTOCOL_VERSION};

struct Server {
    workspace: PathBuf,
    runner: JobRunner,
    archive: RunArchive,
    guider: Guider,
    downloads: DownloadService,
    chat: ChatServer,
    voice: VoiceService,
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
        if let Err(err) = ensure_runner_deployed(&workspace) {
            eprintln!("failed to deploy embedded runner: {err}");
        }

        let runs_dir = workspace.join("runs");
        std::fs::create_dir_all(&runs_dir).ok();

        Self {
            runner: JobRunner::new(&workspace),
            archive: RunArchive::new(&runs_dir),
            guider: Guider::new(),
            downloads: DownloadService::new(&workspace),
            chat: ChatServer::new(&workspace),
            voice: VoiceService::new(&workspace),
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
        let current = *self.current_op.lock().unwrap_or_else(|p| p.into_inner());
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
        let provenance = std::fs::read_to_string(record.artifact_path.join("sytra-provenance.json"))
            .ok().and_then(|text| serde_json::from_str::<Value>(&text).ok());
        Ok(json!({
            "op_id": record.op_id.to_string(),
            "kind": record.kind,
            "status": record.status,
            "artifact_path": record.artifact_path.display().to_string(),
            "telemetry_tail": tail_lines,
            "provenance": provenance,
        }))
    }

    fn tool_start_train(&self, args: &Value) -> Result<Value, String> {
        let mut user_config = args.get("config").cloned().ok_or("missing 'config'")?;
        sytra_host::training_source::normalize_aliases(&mut user_config)?;

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
                "max_seq_len": 2048, "save_every": 50, "packing": "auto"
            },
            "algo": {},
            "output": {
                "adapter_path": format!("runs/mcp/adapter-{stamp}"),
                "resume_from_checkpoint": null
            }
        });
        deep_merge(&mut merged, &user_config);

        let mut run_config: RunConfig =
            serde_json::from_value(merged).map_err(|e| format!("invalid train config: {e}"))?;

        sytra_host::training_source::resolve(&mut run_config)?;

        // Materialize the dataset exactly like the GUI does.
        let dataset_dir = self
            .workspace
            .join("runs")
            .join("mcp")
            .join(format!("dataset-{stamp}"));
        std::fs::create_dir_all(&dataset_dir).map_err(|e| e.to_string())?;
        self.rt.block_on(prepare_training_datasets(&mut run_config, &dataset_dir))?;

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
        *self.current_op.lock().unwrap_or_else(|p| p.into_inner()) = Some(op_id);

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
        *self.current_op.lock().unwrap_or_else(|p| p.into_inner()) = Some(op_id);

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
                .unwrap_or_else(|p| p.into_inner())
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
        let current = AppSettings::load(&self.workspace);
        let detected_ram_mb = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM".to_string()
        })?;
        Ok(settings::settings_payload(&current, &self.workspace, detected_ram_mb))
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
        let mut ollama_cmd = std::process::Command::new("ollama");
        sytra_host::apply_hidden_window(&mut ollama_cmd);
        let ollama_ok = ollama_cmd
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
        let model_path = req_str(args, "model_path")?;
        let context = args.get("context").and_then(|v| v.as_u64()).map(|v| v as usize);
        let export = args
            .get("export_runtimes")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let allow_unverified = args.get("allow_unverified").and_then(|v| v.as_bool());
        let guard = self.operation_guard()?;
        if guard.total_vram_mb == 0 || guard.total_ram_mb == 0 {
            return Err("Hardware memory could not be detected; refusing to plan inference".into());
        }
        plan_inference_with_options(
            &self.workspace,
            model_path,
            guard.total_vram_mb,
            Some(guard.total_ram_mb),
            context,
            export,
            allow_unverified,
        )
    }

    fn tool_list_local_models(&self, args: &Value) -> Result<Value, String> {
        let custom_dir = args.get("custom_dir").and_then(|v| v.as_str());
        let models = scan_local_models(&self.workspace, custom_dir);
        Ok(json!({ "count": models.len(), "models": models }))
    }

    fn tool_get_download_status(&self, args: &Value) -> Result<Value, String> {
        let dest_dir = args.get("dest_dir").and_then(|v| v.as_str());
        match DownloadService::read_status(dest_dir) {
            Some(status) => serde_json::to_value(status).map_err(|e| e.to_string()),
            None => Ok(json!({ "active": false })),
        }
    }

    fn tool_cancel_download(&self, args: &Value) -> Result<Value, String> {
        let dest_dir = args.get("dest_dir").and_then(|v| v.as_str());
        self.downloads.cancel(dest_dir)?;
        Ok(json!({ "cancelled": true }))
    }

    fn tool_build_moe_index(&self, args: &Value) -> Result<Value, String> {
        let model_path = req_str(args, "model_path")?;
        let adapter = args.get("adapter").and_then(|v| v.as_str()).unwrap_or("auto");
        let expert_format = args
            .get("expert_format")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");
        let expert_regex = args.get("expert_regex").and_then(|v| v.as_str());
        convert::build_moe_index(
            &self.workspace,
            model_path,
            adapter,
            expert_format,
            expert_regex,
        )
    }

    fn tool_generate_moe_oracle(&self, args: &Value) -> Result<Value, String> {
        let model_path = req_str(args, "model_path")?;
        let trust_remote_code = args
            .get("trust_remote_code")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let device_map = args.get("device_map").and_then(|v| v.as_str());
        convert::generate_moe_oracle(&self.workspace, model_path, trust_remote_code, device_map)
    }

    fn tool_set_inference_settings(&self, args: &Value) -> Result<Value, String> {
        let detected_ram_mb = BackendResolver::detect_system_ram_mb()
            .ok_or_else(|| "Could not detect system RAM".to_string())?;
        let input: InferenceSettingsInput =
            serde_json::from_value(args.clone()).map_err(|e| e.to_string())?;
        settings::apply_inference_settings(&self.workspace, detected_ram_mb, input)
    }

    fn tool_get_model_settings(&self, args: &Value) -> Result<Value, String> {
        let model_path = req_str(args, "model_path")?;
        let loaded = model_settings::load(&self.workspace, Path::new(model_path))?;
        serde_json::to_value(loaded).map_err(|e| e.to_string())
    }

    fn tool_set_model_settings(&self, args: &Value) -> Result<Value, String> {
        let model_path = req_str(args, "model_path")?;
        let path = Path::new(model_path);
        let mut current = model_settings::load(&self.workspace, path)?;
        opt_patch(args, "context_length", &mut current.context_length)?;
        opt_patch(args, "temperature", &mut current.temperature)?;
        opt_patch(args, "vram_limit_mb", &mut current.vram_limit_mb)?;
        opt_patch(args, "vram_guardrail_mb", &mut current.vram_guardrail_mb)?;
        opt_patch(args, "kv_cache_quant", &mut current.kv_cache_quant)?;
        opt_patch(args, "cpu_kv_cache", &mut current.cpu_kv_cache)?;
        opt_patch(args, "flash_attention", &mut current.flash_attention)?;
        opt_patch(args, "idle_ttl_minutes", &mut current.idle_ttl_minutes)?;
        opt_patch(args, "draft_model_path", &mut current.draft_model_path)?;
        opt_patch(args, "allow_unverified", &mut current.allow_unverified)?;
        model_settings::save(&self.workspace, path, &current)?;
        serde_json::to_value(current).map_err(|e| e.to_string())
    }

    fn tool_start_server(&self, args: &Value) -> Result<Value, String> {
        let model_path = req_str(args, "model_path")?;
        let context = args.get("context").and_then(|v| v.as_u64()).map(|v| v as usize);
        let vram_limit = args
            .get("vram_limit")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let cpu_kv_cache = args.get("cpu_kv_cache").and_then(|v| v.as_bool());
        let allow_unverified = args.get("allow_unverified").and_then(|v| v.as_bool());
        self.chat
            .start_with_options(model_path, context, vram_limit, cpu_kv_cache, allow_unverified)?;
        Ok(json!({
            "running": true,
            "model_path": model_path,
            "endpoint": "http://127.0.0.1:8080/v1",
            "health_url": "http://127.0.0.1:8080/health",
        }))
    }

    fn tool_stop_server(&self) -> Result<Value, String> {
        self.chat.stop()?;
        Ok(json!({ "running": false }))
    }

    fn tool_server_status(&self) -> Result<Value, String> {
        let addr: SocketAddr = "127.0.0.1:8080".parse().expect("valid loopback addr");
        let port_listening = TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok();
        Ok(json!({
            "running": self.chat.is_running(),
            "port_listening": port_listening,
            "endpoint": "http://127.0.0.1:8080/v1",
            "health_url": "http://127.0.0.1:8080/health",
        }))
    }

    fn tool_convert_model(&self, args: &Value) -> Result<Value, String> {
        let model = req_str(args, "model")?;
        let outtype = args.get("outtype").and_then(|v| v.as_str());
        let outfile = args.get("outfile").and_then(|v| v.as_str());
        let op_id = convert::convert_model(&self.workspace, model, outtype, outfile)?;
        Ok(json!({ "op_id": op_id, "status": "completed" }))
    }

    fn tool_export_model(&self, args: &Value) -> Result<Value, String> {
        let model = req_str(args, "model")?;
        let name = args.get("name").and_then(|v| v.as_str());
        let context = args
            .get("context")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| AppSettings::load(&self.workspace).default_context_window);
        let op_id = convert::export_model(&self.workspace, model, name, context)?;
        Ok(json!({ "op_id": op_id, "status": "completed", "context": context }))
    }

    fn tool_list_chat_skills(&self) -> Result<Value, String> {
        let skills = chat_skills::list(&self.workspace);
        Ok(json!({ "count": skills.len(), "skills": skills }))
    }

    fn tool_list_chat_agents(&self) -> Result<Value, String> {
        let agents = chat_agents::list_agents(&self.workspace)?;
        Ok(json!({ "count": agents.len(), "agents": agents }))
    }

    fn tool_get_voice_status(&self) -> Result<Value, String> {
        serde_json::to_value(self.voice.status()).map_err(|e| e.to_string())
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
            "sytra_dataset_merge" => sytra_host::dataset_fabric::run(&self.workspace, "merge", args),
            "sytra_dataset_split" => sytra_host::dataset_fabric::run(&self.workspace, "split", args),
            "export_guide" => self.tool_export_guide(args),
            "plan_inference" => self.tool_plan_inference(args),
            "list_local_models" => self.tool_list_local_models(args),
            "get_download_status" => self.tool_get_download_status(args),
            "cancel_download" => self.tool_cancel_download(args),
            "build_moe_index" => self.tool_build_moe_index(args),
            "generate_moe_oracle" => self.tool_generate_moe_oracle(args),
            "set_inference_settings" => self.tool_set_inference_settings(args),
            "get_model_settings" => self.tool_get_model_settings(args),
            "set_model_settings" => self.tool_set_model_settings(args),
            "start_server" => self.tool_start_server(args),
            "stop_server" => self.tool_stop_server(),
            "server_status" => self.tool_server_status(),
            "convert_model" => self.tool_convert_model(args),
            "export_model" => self.tool_export_model(args),
            "list_chat_skills" => self.tool_list_chat_skills(),
            "list_chat_agents" => self.tool_list_chat_agents(),
            "get_voice_status" => self.tool_get_voice_status(),
            other => Err(format!("unknown tool: {other}")),
        }
    }
}

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("missing '{key}'"))
}

fn opt_patch<T: DeserializeOwned>(
    args: &Value,
    key: &str,
    slot: &mut Option<T>,
) -> Result<(), String> {
    if let Some(val) = args.get(key) {
        *slot = serde_json::from_value(val.clone()).map_err(|e| format!("invalid '{key}': {e}"))?;
    }
    Ok(())
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

fn print_cli_help() {
    println!(
        "Sytra Studio MCP Server ({version})\n\n\
Usage:\n  \
  sytra-mcp                 Run MCP stdio JSON-RPC server (default)\n  \
  sytra-mcp install         Configure detected MCP clients to launch sytra-mcp directly\n  \
  sytra-mcp uninstall       Remove sytra-studio from MCP client configs\n  \
  sytra-mcp status          Show binary, workspace, runner, and client config status\n  \
  sytra-mcp print-config    Print MCP configuration without NPM/npx\n  \
  sytra-mcp --version       Print version\n  \
  sytra-mcp --help          Show this help\n\n\
Install / Print options:\n  \
  --client <list>           auto (default), all, project, claude-desktop, claude-code, cursor, codex, vscode, antigravity\n  \
  --workspace <dir>         Override SYTRA_WORKSPACE path\n  \
  --binary <path>           Override sytra-mcp executable path recorded in client config\n  \
  --dry-run                 Show actions without modifying config files\n  \
  --json                    Emit machine-readable JSON output",
        version = env!("CARGO_PKG_VERSION")
    );
}

fn parse_flag_value(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    if let Some(rest) = args[*idx].strip_prefix(&format!("{flag}=")) {
        return Ok(rest.to_string());
    }
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| format!("Missing value for {flag}"))
}

fn run_cli(args: &[String]) -> Result<i32, String> {
    let cmd = args[0].as_str();
    match cmd {
        "-h" | "--help" | "help" => {
            print_cli_help();
            return Ok(0);
        }
        "-V" | "--version" | "version" => {
            println!("sytra-mcp {}", env!("CARGO_PKG_VERSION"));
            return Ok(0);
        }
        "status" | "print-config" | "install" | "uninstall" => {}
        other => {
            return Err(format!(
                "Unknown argument '{other}'. Run 'sytra-mcp --help' for usage."
            ));
        }
    }

    let mut clients: Vec<String> = Vec::new();
    let mut workspace_override: Option<PathBuf> = None;
    let mut binary_override: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut json_out = false;

    let mut idx = 1;
    while idx < args.len() {
        let arg = args[idx].as_str();
        if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--json" {
            json_out = true;
        } else if arg == "--client" || arg.starts_with("--client=") {
            clients.push(parse_flag_value(args, &mut idx, "--client")?);
        } else if arg == "--workspace" || arg.starts_with("--workspace=") {
            workspace_override = Some(PathBuf::from(parse_flag_value(
                args,
                &mut idx,
                "--workspace",
            )?));
        } else if arg == "--binary" || arg.starts_with("--binary=") {
            binary_override = Some(PathBuf::from(parse_flag_value(args, &mut idx, "--binary")?));
        } else if arg == "-h" || arg == "--help" {
            print_cli_help();
            return Ok(0);
        } else {
            return Err(format!("Unknown flag '{arg}'"));
        }
        idx += 1;
    }

    let workspace = workspace_override.unwrap_or_else(resolve_workspace);
    if matches!(cmd, "status" | "install") && !dry_run {
        if let Err(err) = ensure_runner_deployed(&workspace) {
            eprintln!("warning: failed to deploy embedded runner: {err}");
        }
    }

    match cmd {
        "status" => {
            let report = status_report(&workspace, binary_override.as_deref());
            if json_out {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
                );
            } else {
                println!(
                    "sytra-mcp binary : {} ({})",
                    report.mcp_binary_path,
                    if report.mcp_binary_exists {
                        "found"
                    } else {
                        "missing"
                    }
                );
                println!("workspace        : {}", report.workspace);
                println!(
                    "python runner    : {}",
                    if report.runner_ready {
                        "ready"
                    } else {
                        "not deployed"
                    }
                );
                println!("\nMCP Clients:");
                for c in &report.clients {
                    let state = if c.configured {
                        "configured"
                    } else if c.detected {
                        "detected"
                    } else {
                        "not detected"
                    };
                    println!("  - {:<15} [{:<12}] {}", c.label, state, c.config_path);
                }
            }
            Ok(0)
        }
        "print-config" => {
            let report = status_report(&workspace, binary_override.as_deref());
            let target = clients.first().map(|s| s.as_str()).unwrap_or("generic");
            match target.to_ascii_lowercase().as_str() {
                "codex" => println!("{}", report.codex_toml),
                "vscode" | "vs-code" | "code" => println!("{}", report.vscode_json),
                "generic" | "auto" | "claude" | "claude-desktop" | "claude-code" | "cursor"
                | "antigravity" => {
                    println!("{}", report.generic_json);
                }
                "project" => {
                    let bin = PathBuf::from(&report.mcp_binary_path);
                    let entry = mcp_server_entry(&bin, &workspace, false);
                    let doc = json!({ "mcpServers": { "sytra-studio": entry } });
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?
                    );
                }
                other => {
                    let kind = McpClientKind::parse(other).map_err(|e| e.to_string())?;
                    if kind == McpClientKind::Codex {
                        println!("{}", report.codex_toml);
                    } else if kind == McpClientKind::VsCode {
                        println!("{}", report.vscode_json);
                    } else {
                        println!("{}", report.generic_json);
                    }
                }
            }
            Ok(0)
        }
        "install" => {
            let sel = if clients.is_empty() {
                None
            } else {
                Some(clients.as_slice())
            };
            let outcomes =
                install_mcp_clients(&workspace, sel, binary_override.as_deref(), dry_run)
                    .map_err(|e| e.to_string())?;
            if json_out {
                let report = status_report(&workspace, binary_override.as_deref());
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({
                        "outcomes": outcomes,
                        "status": report,
                    }))
                    .map_err(|e| e.to_string())?
                );
            } else if outcomes.is_empty() {
                let report = status_report(&workspace, binary_override.as_deref());
                println!(
                    "No existing MCP client directories were auto-detected. Use `--client all` or `--client claude-desktop,cursor,codex,claude-code,vscode,antigravity` to create configs explicitly.\n\nDirect MCP config (no NPM required):\n{}",
                    report.generic_json
                );
            } else {
                for item in &outcomes {
                    println!("[{}] {} -> {}", item.action, item.label, item.config_path);
                }
            }
            Ok(0)
        }
        "uninstall" => {
            let sel = if clients.is_empty() {
                None
            } else {
                Some(clients.as_slice())
            };
            let outcomes =
                uninstall_mcp_clients(&workspace, sel, dry_run).map_err(|e| e.to_string())?;
            if json_out {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&json!({ "outcomes": outcomes }))
                        .map_err(|e| e.to_string())?
                );
            } else if outcomes.is_empty() {
                println!("No matching sytra-studio MCP entries found.");
            } else {
                for item in &outcomes {
                    println!("[{}] {} -> {}", item.action, item.label, item.config_path);
                }
            }
            Ok(0)
        }
        _ => unreachable!(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        match run_cli(&args) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("Error: {err}");
                std::process::exit(2);
            }
        }
    }
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
                        "instructions": "Sytra Studio: local LLM fine-tuning, merging, and Sytra-Helix/GGUF inference. Typical flows — Training/Merging: get_status -> list_catalog/guider_recommend -> start_train or merge_check + start_merge -> poll get_run -> stop_op. Inference: list_local_models -> (build_moe_index + generate_moe_oracle for SafeTensors MoE) -> plan_inference -> start_server -> server_status / stop_server."
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
    let orphan = *server.current_op.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(op_id) = orphan {
        if let Err(err) = commands::stop_op(op_id, &server.runner, &server.archive) {
            eprintln!("failed to stop orphaned op: {err}");
        }
    }
    server.downloads.shutdown();
    server.chat.shutdown();
}
