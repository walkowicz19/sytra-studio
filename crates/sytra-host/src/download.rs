use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::process::kill_process_tree;
use crate::settings::AppSettings;
use crate::workspace::default_model_dir;

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
        {
            let guard = self.active_pid.lock().map_err(|_| "lock")?;
            if guard.is_some() {
                return Err("Another model download is already running".into());
            }
        }

        let target = dest_dir
            .map(PathBuf::from)
            .unwrap_or_else(default_model_dir);
        if let Some(existing) = Self::read_status(dest_dir) {
            if existing.repo_id == repo_id && existing.status != "error" && existing.status != "completed"
            {
                return Ok(DownloadStart {
                    op_id: Uuid::new_v4().to_string(),
                    pid: None,
                    dest_dir: target,
                    status: existing.status.clone(),
                    download_status: Some(existing),
                    message: "Download already in progress".into(),
                });
            }
        }

        let script = self
            .workspace
            .join("runner")
            .join("scripts")
            .join("download_gguf_model.py");
        let settings = AppSettings::load(&self.workspace);
        let uv_exe = if cfg!(target_os = "windows") {
            "uv.exe"
        } else {
            "uv"
        };
        let mut cmd = std::process::Command::new(uv_exe);
        cmd.args([
            "run",
            "--no-project",
            "--with",
            "huggingface-hub>=0.34",
            "--with",
            "hf-xet>=1.1.5",
            "python",
        ])
        .arg(&script)
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
        cmd.env("HF_XET_HIGH_PERFORMANCE", "0")
            .env("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", "1073741824")
            .current_dir(&self.workspace);

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x00004000;
            cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);
        }

        #[cfg(not(target_os = "windows"))]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to launch download script: {e}"))?;
        let pid = child.id();
        *self.active_pid.lock().map_err(|_| "lock")? = Some(pid);

        let pid_ref = self.active_pid.clone();
        thread::spawn(move || {
            let _ = child.wait();
            if let Ok(mut g) = pid_ref.lock() {
                if *g == Some(pid) {
                    *g = None;
                }
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
        if let Some(pid) = self.active_pid.lock().map_err(|_| "lock")?.take() {
            kill_process_tree(pid);
        }
        let status_file = Self::status_path(dest_dir);
        if status_file.exists() {
            std::fs::remove_file(status_file).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn shutdown(&self) {
        if let Ok(mut g) = self.active_pid.lock() {
            if let Some(pid) = g.take() {
                kill_process_tree(pid);
            }
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
