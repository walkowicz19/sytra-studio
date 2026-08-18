use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Bump when the package pins or CUDA index change so stale venvs are rebuilt.
const PROVISION_SCHEMA: u32 = 2;
const TORCH_CUDA_INDEX: &str = "https://download.pytorch.org/whl/cu128";

static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());
static MERGE_LOCK: Mutex<()> = Mutex::new(());
static TRAIN_LOCK: Mutex<()> = Mutex::new(());

fn provision_lock(kind: EnvKind) -> &'static Mutex<()> {
    match kind {
        EnvKind::Download => &DOWNLOAD_LOCK,
        EnvKind::Merge => &MERGE_LOCK,
        EnvKind::Train => &TRAIN_LOCK,
    }
}

pub struct EnvProvisioner {
    base_dir: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvKind {
    Download,
    Merge,
    Train,
}

impl EnvKind {
    fn dir_name(self) -> &'static str {
        match self {
            Self::Download => "download-env",
            Self::Merge => "merge-env",
            Self::Train => "train-env",
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Download => "download",
            Self::Merge => "merge",
            Self::Train => "train",
        }
    }
}

impl EnvProvisioner {
    pub fn new(workspace_root: &Path) -> Self {
        Self {
            base_dir: workspace_root.join(".sytra-envs"),
        }
    }

    pub fn train_env_dir(&self) -> PathBuf {
        self.env_dir(EnvKind::Train)
    }

    pub fn merge_env_dir(&self) -> PathBuf {
        self.env_dir(EnvKind::Merge)
    }

    pub fn download_env_dir(&self) -> PathBuf {
        self.env_dir(EnvKind::Download)
    }

    pub fn env_dir(&self, kind: EnvKind) -> PathBuf {
        self.base_dir.join(kind.dir_name())
    }

    /// Resolves the path to the python executable in the given virtual environment.
    pub fn python_path(&self, env_dir: &Path) -> PathBuf {
        if cfg!(target_os = "windows") {
            env_dir.join("Scripts").join("python.exe")
        } else {
            env_dir.join("bin").join("python")
        }
    }

    pub fn train_python_path(&self) -> PathBuf {
        self.python_path(&self.train_env_dir())
    }

    pub fn merge_python_path(&self) -> PathBuf {
        self.python_path(&self.merge_env_dir())
    }

    pub fn download_python_path(&self) -> PathBuf {
        self.python_path(&self.download_env_dir())
    }

    pub fn is_train_provisioned(&self) -> bool {
        self.is_provisioned(EnvKind::Train)
    }

    pub fn is_merge_provisioned(&self) -> bool {
        self.is_provisioned(EnvKind::Merge)
    }

    pub fn is_download_provisioned(&self) -> bool {
        self.is_provisioned(EnvKind::Download)
    }

    pub fn is_provisioned(&self, kind: EnvKind) -> bool {
        let env_dir = self.env_dir(kind);
        self.python_path(&env_dir).is_file() && self.stamp_matches(&env_dir, kind)
    }

    /// Install or repair the download env, then return its interpreter.
    /// Never falls back to PATH Python — mixed AI-tool packages freeze Windows.
    pub fn ensure_download(&self) -> Result<PathBuf> {
        self.ensure(EnvKind::Download)
    }

    pub fn ensure_merge(&self) -> Result<PathBuf> {
        self.ensure(EnvKind::Merge)
    }

    pub fn ensure_train(&self) -> Result<PathBuf> {
        self.ensure(EnvKind::Train)
    }

    pub fn ensure(&self, kind: EnvKind) -> Result<PathBuf> {
        let _guard = provision_lock(kind)
            .lock()
            .map_err(|_| anyhow!("Sytra env provision lock was poisoned"))?;
        if !self.is_provisioned(kind) {
            self.provision_inner(kind)?;
        }
        if !self.is_provisioned(kind) {
            return Err(anyhow!(
                "Sytra {} environment is not ready. Refusing to use system Python \
                 (other AI tools commonly pollute PATH and freeze Windows during Hub downloads).",
                kind.as_str()
            ));
        }
        Ok(self.python_path(&self.env_dir(kind)))
    }

    pub fn provision_all(&self) -> Result<()> {
        self.provision_download()?;
        self.provision_merge()?;
        self.provision_train()?;
        Ok(())
    }

    pub fn status_report(&self) -> Value {
        json!({
            "schema": PROVISION_SCHEMA,
            "refuses_system_python": true,
            "download": self.env_status(EnvKind::Download),
            "merge": self.env_status(EnvKind::Merge),
            "train": self.env_status(EnvKind::Train),
        })
    }

    fn env_status(&self, kind: EnvKind) -> Value {
        json!({
            "provisioned": self.is_provisioned(kind),
            "python": self.python_path(&self.env_dir(kind)).display().to_string(),
            "stamp_ok": self.stamp_matches(&self.env_dir(kind), kind),
        })
    }

    fn stamp_path(env_dir: &Path) -> PathBuf {
        env_dir.join(".provision-ok")
    }

    fn stamp_matches(&self, env_dir: &Path, kind: EnvKind) -> bool {
        let Ok(raw) = std::fs::read_to_string(Self::stamp_path(env_dir)) else {
            return false;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            // Legacy empty marker from schema 1 — treat as stale so CUDA
            // wheels and integrity checks run on the next provision.
            return false;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            return false;
        };
        value.get("schema").and_then(Value::as_u64) == Some(PROVISION_SCHEMA as u64)
            && value.get("kind").and_then(Value::as_str) == Some(kind.as_str())
    }

    fn write_stamp(&self, env_dir: &Path, kind: EnvKind, extra: Value) -> Result<()> {
        let mut payload = extra;
        if let Some(obj) = payload.as_object_mut() {
            obj.insert("schema".into(), json!(PROVISION_SCHEMA));
            obj.insert("kind".into(), json!(kind.as_str()));
        }
        std::fs::write(
            Self::stamp_path(env_dir),
            serde_json::to_string_pretty(&payload)?,
        )?;
        Ok(())
    }

    fn clear_stamp(env_dir: &Path) {
        let _ = std::fs::remove_file(Self::stamp_path(env_dir));
    }

    pub fn provision_download(&self) -> Result<()> {
        self.locked_provision(EnvKind::Download)
    }

    pub fn provision_train(&self) -> Result<()> {
        self.locked_provision(EnvKind::Train)
    }

    pub fn provision_merge(&self) -> Result<()> {
        self.locked_provision(EnvKind::Merge)
    }

    fn locked_provision(&self, kind: EnvKind) -> Result<()> {
        let _guard = provision_lock(kind)
            .lock()
            .map_err(|_| anyhow!("Sytra env provision lock was poisoned"))?;
        self.provision_inner(kind)
    }

    fn provision_inner(&self, kind: EnvKind) -> Result<()> {
        match kind {
            EnvKind::Download => self.provision_download_inner(),
            EnvKind::Merge => self.provision_merge_inner(),
            EnvKind::Train => self.provision_train_inner(),
        }
    }

    /// Tiny Hub/Xet env used by model downloads. Isolated from train/merge
    /// so a polluted training venv cannot take down downloads.
    fn provision_download_inner(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        let env_dir = self.download_env_dir();
        Self::clear_stamp(&env_dir);
        self.ensure_venv(&env_dir)?;
        let python = self.download_python_path().display().to_string();
        Self::run_uv_cmd(&[
            "pip",
            "install",
            "--python",
            &python,
            "huggingface-hub>=0.34",
            "hf-xet>=1.1.5",
        ])?;
        self.verify_imports(
            &self.download_python_path(),
            "import huggingface_hub, hf_xet",
        )?;
        self.write_stamp(&env_dir, EnvKind::Download, json!({}))?;
        Ok(())
    }

    /// Provision train environment using uv.
    fn provision_train_inner(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        let env_dir = self.train_env_dir();
        Self::clear_stamp(&env_dir);
        self.ensure_venv(&env_dir)?;
        let python = self.train_python_path().display().to_string();

        // PyPI torch wheels are CPU-only. A CPU torch in this venv either
        // refuses CUDA training or, when mixed with another tool's CUDA
        // bits, pages the whole machine while loading weights.
        if !cfg!(target_os = "macos") {
            Self::run_uv_cmd(&[
                "pip",
                "install",
                "--python",
                &python,
                "--index-url",
                TORCH_CUDA_INDEX,
                "torch==2.10.0",
            ])?;
        } else {
            Self::run_uv_cmd(&["pip", "install", "--python", &python, "torch==2.10.0"])?;
        }

        Self::run_uv_cmd(&[
            "pip",
            "install",
            "--python",
            &python,
            // Keep the CUDA training stack reproducible. These versions are
            // validated together on Windows with Qwen3.5 and bitsandbytes
            // 4-bit loading; unpinned upgrades have broken model loading and
            // TRL argument compatibility in the past.
            "transformers==5.2.0",
            "unsloth==2026.7.2",
            "peft==0.19.1",
            "trl==0.24.0",
            "bitsandbytes==0.49.2",
            "datasets==4.3.0",
            "accelerate==1.14.0",
            "pyyaml==6.0.2",
            "huggingface-hub>=0.34",
            "hf-xet>=1.1.5",
        ])?;

        if !cfg!(target_os = "macos") {
            // Unsloth/peft may pull a CPU torch from PyPI. Re-pin last.
            Self::run_uv_cmd(&[
                "pip",
                "install",
                "--python",
                &python,
                "--index-url",
                TORCH_CUDA_INDEX,
                "torch==2.10.0",
            ])?;
        }

        let integrity = if cfg!(target_os = "macos") {
            "import torch, transformers, unsloth, peft, trl, bitsandbytes, datasets, accelerate, huggingface_hub"
        } else {
            "import torch, transformers, unsloth, peft, trl, bitsandbytes, datasets, accelerate, huggingface_hub; \
             assert torch.version.cuda, 'train env has CPU torch; CUDA wheel is required'"
        };
        self.verify_imports(&self.train_python_path(), integrity)?;
        self.write_stamp(
            &env_dir,
            EnvKind::Train,
            json!({ "torch_index": if cfg!(target_os = "macos") { "pypi" } else { "cu128" } }),
        )?;
        Ok(())
    }

    /// Provision merge environment using uv.
    fn provision_merge_inner(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        let env_dir = self.merge_env_dir();
        Self::clear_stamp(&env_dir);
        self.ensure_venv(&env_dir)?;
        let python = self.merge_python_path().display().to_string();
        Self::run_uv_cmd(&[
            "pip",
            "install",
            "--python",
            &python,
            "mergekit",
            // pydantic 2.10 breaks mergekit's torch-typed models
            // (ConfiguredModuleArchitecture "not fully defined").
            "pydantic>=2,<2.10",
            "pyyaml",
            "huggingface-hub>=0.34",
            "hf-xet>=1.1.5",
        ])?;
        self.verify_imports(&self.merge_python_path(), "import mergekit, huggingface_hub")?;
        self.write_stamp(&env_dir, EnvKind::Merge, json!({}))?;
        Ok(())
    }

    fn ensure_venv(&self, env_dir: &Path) -> Result<()> {
        if !self.python_path(env_dir).is_file() {
            Self::run_uv_cmd(&["venv", &env_dir.display().to_string()])?;
        }
        Ok(())
    }

    fn verify_imports(&self, python: &Path, snippet: &str) -> Result<()> {
        let output = Command::new(python)
            .args(["-c", snippet])
            .output()
            .map_err(|e| anyhow!("Failed to verify {}: {e}", python.display()))?;
        if !output.status.success() {
            return Err(anyhow!(
                "Sytra env integrity check failed for {}:\n{}",
                python.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(())
    }

    fn run_uv_cmd(args: &[&str]) -> Result<()> {
        let cmd = if cfg!(target_os = "windows") {
            "uv.exe"
        } else {
            "uv"
        };

        let output = Command::new(cmd)
            .args(args)
            .output()
            .map_err(|e| anyhow!("Failed to execute uv: {e}. Is uv installed and on PATH?"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(anyhow!(
                "uv command failed with status: {}\n{stderr}\n{stdout}",
                output.status
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn empty_legacy_stamp_is_stale() {
        let tmp = std::env::temp_dir().join("sytra-env-stamp-empty");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(".provision-ok"), "").unwrap();
        let provisioner = EnvProvisioner { base_dir: tmp.clone() };
        assert!(!provisioner.stamp_matches(&tmp, EnvKind::Train));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn schema_mismatch_is_stale() {
        let tmp = std::env::temp_dir().join("sytra-env-stamp-old");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(
            tmp.join(".provision-ok"),
            r#"{"schema":1,"kind":"train"}"#,
        )
        .unwrap();
        let provisioner = EnvProvisioner { base_dir: tmp.clone() };
        assert!(!provisioner.stamp_matches(&tmp, EnvKind::Train));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn current_stamp_matches_kind() {
        let tmp = std::env::temp_dir().join("sytra-env-stamp-ok");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let provisioner = EnvProvisioner { base_dir: tmp.clone() };
        provisioner
            .write_stamp(&tmp, EnvKind::Download, json!({}))
            .unwrap();
        assert!(provisioner.stamp_matches(&tmp, EnvKind::Download));
        assert!(!provisioner.stamp_matches(&tmp, EnvKind::Train));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn status_report_never_claims_system_python_is_ok() {
        let tmp = std::env::temp_dir().join("sytra-env-status");
        let _ = fs::remove_dir_all(&tmp);
        let provisioner = EnvProvisioner::new(&tmp);
        let status = provisioner.status_report();
        assert_eq!(status["refuses_system_python"], true);
        assert_eq!(status["download"]["provisioned"], false);
        assert_eq!(status["schema"], PROVISION_SCHEMA);
        let _ = fs::remove_dir_all(&tmp);
    }
}
