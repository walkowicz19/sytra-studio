use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::env_provisioner::EnvProvisioner;
use crate::process::kill_process_tree;
use crate::settings::AppSettings;
use crate::workspace::default_model_dir;
use crate::xet_safety::apply_xet_safety;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub repo_id: String,
    pub status: String,
    pub downloaded_gb: f64,
    pub total_gb: f64,
    pub pct: f64,
    pub speed_mbps: f64,
    pub eta_seconds: u64,
    pub eta_formatted: String,
    pub current_file: String,
    pub shard_index: usize,
    pub total_shards: usize,
    pub timestamp: f64,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DownloadStart {
    pub op_id: String,
    pub pid: Option<u32>,
    pub dest_dir: PathBuf,
    pub status: String,
    pub download_status: Option<DownloadStatus>,
    pub message: String,
}

pub struct DownloadService {
    workspace: PathBuf,
    active_pid: Arc<Mutex<Option<u32>>>,
}

impl DownloadService {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            active_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn active_pid(&self) -> Arc<Mutex<Option<u32>>> {
        self.active_pid.clone()
    }

    pub fn status_path(dest_dir: Option<&str>) -> PathBuf {
        dest_dir
            .map(PathBuf::from)
            .unwrap_or_else(default_model_dir)
            .join(".download_status.json")
    }

    pub fn read_status(dest_dir: Option<&str>) -> Option<DownloadStatus> {
        let path = Self::status_path(dest_dir);
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    pub fn start(
        &self,
        repo_id: &str,
        purpose: &str,
        dest_dir: Option<&str>,
        quant: Option<&str>,
        revision: Option<&str>,
    ) -> Result<DownloadStart, String> {
        if !matches!(purpose, "inference" | "finetune" | "merge") {
            return Err(format!("Unsupported model download purpose: {purpose}"));
        }
        if repo_id.trim().starts_with('-') {
            return Err("Invalid repo_id parameter: cannot start with '-'".into());
        }
        if let Some(q) = quant {
            if q.trim().starts_with('-') {
                return Err("Invalid quant parameter: cannot start with '-'".into());
            }
        }
        if let Some(rev) = revision {
            if rev.trim().starts_with('-') {
                return Err("Invalid revision parameter: cannot start with '-'".into());
            }
        }
        if let Some(dest) = dest_dir {
            if dest.trim().starts_with('-') {
                return Err("Invalid dest_dir parameter: cannot start with '-'".into());
            }
        }
        // Hold the reservation through spawn so concurrent callers cannot both
        // pass the check and launch competing downloads.
        let mut active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
        if active.is_some() {
            return Err("Another model download is already running".into());
        }

        let target = dest_dir
            .map(PathBuf::from)
            .unwrap_or_else(default_model_dir);
        // A status file can survive a crash or application restart. The
        // in-process reservation, not that file, determines whether a job runs.

        let python = EnvProvisioner::new(&self.workspace)
            .ensure_download()
            .map_err(|e| e.to_string())?;
        let script = self
            .workspace
            .join("runner")
            .join("scripts")
            .join("download_gguf_model.py");
        let settings = AppSettings::load(&self.workspace);
        let mut cmd = std::process::Command::new(python);
        cmd.arg(&script)
            .arg("--model")
            .arg(repo_id)
            .arg("--quant")
            .arg(quant.unwrap_or("auto"))
            .arg("--purpose")
            .arg(purpose)
            .arg("--revision")
            .arg(revision.unwrap_or("main"));
        if settings.tokenless_download {
            cmd.arg("--tokenless");
        }
        if let Some(dest) = dest_dir {
            if !dest.trim().is_empty() {
                cmd.arg("--dest").arg(dest);
            }
        }
        cmd.env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONPATH", self.workspace.join("runner"))
            .env("HF_HOME", settings.effective_hf_cache(&self.workspace))
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            .current_dir(&self.workspace);
        apply_xet_safety(&mut cmd);
        crate::apply_desktop_priority(&mut cmd);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to launch download script: {e}"))?;
        let pid = child.id();
        *active = Some(pid);
        drop(active);

        let pid_ref = self.active_pid.clone();
        thread::spawn(move || {
            let _ = child.wait();
            let mut g = pid_ref.lock().unwrap_or_else(|p| p.into_inner());
            if *g == Some(pid) {
                *g = None;
            }
        });

        Ok(DownloadStart {
            op_id: Uuid::new_v4().to_string(),
            pid: Some(pid),
            dest_dir: target,
            status: "started".into(),
            download_status: None,
            message: format!("Started commit-pinned download of {repo_id}"),
        })
    }

    pub fn cancel(&self, dest_dir: Option<&str>) -> Result<(), String> {
        if let Some(pid) = self.active_pid.lock().unwrap_or_else(|p| p.into_inner()).take() {
            kill_process_tree(pid);
        }
        let status_file = Self::status_path(dest_dir);
        if status_file.exists() {
            std::fs::remove_file(status_file).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn shutdown(&self) {
        let mut g = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pid) = g.take() {
            kill_process_tree(pid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_purpose() {
        let svc = DownloadService::new(Path::new("."));
        let err = svc
            .start("org/model", "training", None, None, None)
            .unwrap_err();
        assert!(err.contains("Unsupported"));
    }

    #[test]
    fn refuses_second_concurrent_download() {
        let svc = DownloadService::new(Path::new("."));
        *svc.active_pid.lock().unwrap() = Some(1);
        let err = svc
            .start("org/model", "inference", None, None, None)
            .unwrap_err();
        assert!(err.contains("already running"));
    }

    #[test]
    fn status_path_uses_dest_or_default() {
        let custom = DownloadService::status_path(Some("/tmp/models"));
        assert!(custom.ends_with(".download_status.json"));
    }
}
