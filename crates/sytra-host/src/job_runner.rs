use anyhow::{anyhow, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use tokio::sync::mpsc;

use crate::env_provisioner::EnvProvisioner;
use crate::process::kill_process_tree;
use sytra_contracts::{parse_line, Operation, TelemetryLine};

/// The active subprocess, tagged with a generation counter so a stale
/// reader thread from a previous operation can never reap or steal the
/// child of a newer one.
struct ActiveJob {
    generation: u64,
    pid: u32,
    child: Child,
}

pub struct JobRunner {
    workspace_root: PathBuf,
    env_provisioner: EnvProvisioner,
    active: Arc<Mutex<Option<ActiveJob>>>,
    next_generation: AtomicU64,
}

impl JobRunner {
    pub fn new(workspace_root: &Path) -> Self {
        Self {
            workspace_root: workspace_root.to_path_buf(),
            env_provisioner: EnvProvisioner::new(workspace_root),
            active: Arc::new(Mutex::new(None)),
            next_generation: AtomicU64::new(1),
        }
    }

    /// Spawns the operation subprocess and streams TelemetryLines to the returned receiver.
    /// Fails if another operation is still running — stop it first.
    pub fn start(&self, op: &Operation) -> Result<mpsc::Receiver<TelemetryLine>> {
        let runner_cmd = op.runner_cmd();

        // Isolated Sytra venvs only. PATH Python is commonly polluted by
        // other AI tools and is the Windows freeze path during Hub downloads.
        let python_path = match op {
            Operation::Train(_) => self.env_provisioner.ensure_train()?,
            Operation::Merge(_) | Operation::Publish(_) => self.env_provisioner.ensure_merge()?,
        };

        let mut cmd = Command::new(python_path);
        cmd.args(&runner_cmd.args)
            // stdin is closed on purpose: a GUI app has no console, and any
            // interactive prompt (HF gated-model token, mergekit confirm)
            // would otherwise block the runner forever.
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            // Never let HF Hub wait on interactive auth.
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            .env(
                "SYTRA_TOKENLESS",
                if crate::settings::AppSettings::load(&self.workspace_root).tokenless_download {
                    "1"
                } else {
                    "0"
                },
            )
            // Multi-GB model downloads go where the user chose (default:
            // workspace .hf-cache) instead of filling the C: SSD. Loaded
            // fresh per spawn so a settings change applies immediately.
            .env(
                "HF_HOME",
                crate::settings::AppSettings::load(&self.workspace_root)
                    .effective_hf_cache(&self.workspace_root),
            )
            .env("PYTHONPATH", self.workspace_root.join("runner"))
            .env("OMP_NUM_THREADS", "4")
            .env("MKL_NUM_THREADS", "4")
            .env("OPENBLAS_NUM_THREADS", "4")
            .env("VECLIB_MAXIMUM_THREADS", "4")
            .env("NUMEXPR_NUM_THREADS", "4")
            .env("TOKENIZERS_PARALLELISM", "false")
            .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            .env("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")
            ;
        crate::apply_xet_safety(&mut cmd);
        crate::apply_desktop_priority(&mut cmd);

        let generation = self.next_generation.fetch_add(1, Ordering::SeqCst);

        // Hold the lock across the busy-check and the spawn so two
        // concurrent starts can't both pass the check.
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err(anyhow!(
                "An operation is already running. Stop it before starting a new one."
            ));
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("Failed to spawn runner command: {e}"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stdout of runner process"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("Failed to capture stderr of runner process"))?;
        let pid = child.id();

        *active = Some(ActiveJob {
            generation,
            pid,
            child,
        });
        drop(active);

        let (tx, rx) = mpsc::channel::<TelemetryLine>(100);
        let active_ref = self.active.clone();

        // stderr → log lines
        let tx_err = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line_str) = line {
                    let parsed = TelemetryLine::Log {
                        ts: None,
                        stream: Some("stderr".to_string()),
                        line: line_str,
                    };
                    if tx_err.blocking_send(parsed).is_err() {
                        break;
                    }
                }
            }
        });

        // stdout → parsed telemetry; reaps the child when the pipe closes
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                match line {
                    Ok(line_str) => {
                        let parsed = parse_line(&line_str);
                        if tx.blocking_send(parsed).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            // Only reap the child if it is still OUR child (same
            // generation). If stop() already took it, or a newer operation
            // occupies the slot, leave it alone.
            let taken = {
                let mut active = active_ref.lock().unwrap();
                match active.as_ref() {
                    Some(job) if job.generation == generation => active.take(),
                    _ => None,
                }
            };
            if let Some(mut job) = taken {
                if let Ok(status) = job.child.wait() {
                    if !status.success() {
                        let _ = tx.blocking_send(TelemetryLine::Event {
                            ts: 0.0,
                            op_id: None,
                            event: "error".to_string(),
                            payload: serde_json::json!({
                                "message": format!(
                                    "Runner process exited with code {:?} without a terminal event",
                                    status.code()
                                )
                            }),
                        });
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Stops the currently running operation, killing the whole process
    /// tree. Idempotent: stopping when nothing runs is a no-op, so a
    /// double-click on Stop or a stop after natural completion never errors.
    pub fn stop(&self) -> Result<()> {
        let taken = {
            let mut active = self.active.lock().unwrap();
            active.take()
        };
        if let Some(mut job) = taken {
            kill_process_tree(job.pid);
            // Fallback + reap. kill() after taskkill is a no-op error we ignore.
            let _ = job.child.kill();
            let _ = job.child.wait();
        }
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        let active = self.active.lock().unwrap();
        active.is_some()
    }
}
