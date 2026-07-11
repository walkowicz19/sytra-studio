use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct EnvProvisioner {
    base_dir: PathBuf,
}

impl EnvProvisioner {
    pub fn new(workspace_root: &Path) -> Self {
        Self {
            base_dir: workspace_root.join(".sytra-envs"),
        }
    }

    pub fn train_env_dir(&self) -> PathBuf {
        self.base_dir.join("train-env")
    }

    pub fn merge_env_dir(&self) -> PathBuf {
        self.base_dir.join("merge-env")
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

    pub fn is_train_provisioned(&self) -> bool {
        self.train_python_path().exists() && self.train_env_dir().join(".provision-ok").exists()
    }

    pub fn is_merge_provisioned(&self) -> bool {
        self.merge_python_path().exists() && self.merge_env_dir().join(".provision-ok").exists()
    }

    /// Provision train environment using uv.
    pub fn provision_train(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        let env_dir = self.train_env_dir();

        if !self.train_python_path().exists() {
            Self::run_uv_cmd(&["venv", &env_dir.display().to_string()])?;
        }

        // Install packages
        let python_str = self.train_python_path().display().to_string();
        Self::run_uv_cmd(&[
            "pip",
            "install",
            "--python",
            &python_str,
            "torch",
            "transformers",
            "trl",
            // Old datasets releases break at import time against pyarrow
            // >= 14 (PyExtensionType removed); pin a modern floor.
            "datasets>=3.2",
            "pyyaml",
        ])?;

        // Write marker file to indicate successful provisioning
        std::fs::write(env_dir.join(".provision-ok"), "").ok();

        Ok(())
    }

    /// Provision merge environment using uv.
    pub fn provision_merge(&self) -> Result<()> {
        std::fs::create_dir_all(&self.base_dir)?;
        let env_dir = self.merge_env_dir();

        if !self.merge_python_path().exists() {
            Self::run_uv_cmd(&["venv", &env_dir.display().to_string()])?;
        }

        // Install packages (mergekit runs on CPU)
        let python_str = self.merge_python_path().display().to_string();
        Self::run_uv_cmd(&[
            "pip",
            "install",
            "--python",
            &python_str,
            "mergekit",
            // pydantic 2.10 breaks mergekit's torch-typed models
            // (ConfiguredModuleArchitecture "not fully defined").
            "pydantic>=2,<2.10",
            "pyyaml",
        ])?;

        // Write marker file to indicate successful provisioning
        std::fs::write(env_dir.join(".provision-ok"), "").ok();

        Ok(())
    }

    fn run_uv_cmd(args: &[&str]) -> Result<()> {
        let cmd = if cfg!(target_os = "windows") {
            "uv.exe"
        } else {
            "uv"
        };

        let status = Command::new(cmd)
            .args(args)
            .status()
            .map_err(|e| anyhow!("Failed to execute uv: {e}. Is uv installed and on PATH?"))?;

        if !status.success() {
            return Err(anyhow!("uv command failed with status: {status}"));
        }

        Ok(())
    }
}
