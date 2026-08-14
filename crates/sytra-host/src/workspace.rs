use std::path::{Path, PathBuf};

use crate::env_provisioner::EnvProvisioner;

pub fn find_project_root() -> Option<PathBuf> {
    if let Ok(mut dir) = std::env::current_exe() {
        while dir.pop() {
            if dir.join("runner").join("sytra_runner").exists() {
                return Some(dir);
            }
        }
    }
    None
}

pub fn resolve_workspace() -> PathBuf {
    std::env::var("SYTRA_WORKSPACE")
        .map(PathBuf::from)
        .ok()
        .or_else(find_project_root)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

pub fn user_home_dir() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("USERPROFILE") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    None
}

pub fn default_model_dir() -> PathBuf {
    user_home_dir()
        .map(|home| home.join("lm-studio models"))
        .unwrap_or_else(|| PathBuf::from("./lm-studio models"))
}

/// Python used for runner scripts (convert/export/serve/index).
pub fn python_executable(workspace: &Path) -> PathBuf {
    if let Ok(explicit) = std::env::var("SYTRA_PYTHON") {
        if !explicit.trim().is_empty() {
            return PathBuf::from(explicit);
        }
    }
    let provisioner = EnvProvisioner::new(workspace);
    if provisioner.is_merge_provisioned() {
        return provisioner.merge_python_path();
    }
    if provisioner.is_train_provisioned() {
        return provisioner.train_python_path();
    }
    let managed = if cfg!(target_os = "windows") {
        workspace
            .join("runner")
            .join(".venv")
            .join("Scripts")
            .join("python.exe")
    } else {
        workspace
            .join("runner")
            .join(".venv")
            .join("bin")
            .join("python")
    };
    if managed.is_file() {
        managed
    } else if cfg!(target_os = "windows") {
        PathBuf::from("python.exe")
    } else {
        PathBuf::from("python3")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_model_dir_is_under_home_or_relative() {
        let dir = default_model_dir();
        assert!(dir.ends_with("lm-studio models"));
    }

    #[test]
    fn python_executable_honors_sytra_python() {
        std::env::set_var("SYTRA_PYTHON", "/custom/python");
        let path = python_executable(Path::new("."));
        std::env::remove_var("SYTRA_PYTHON");
        assert_eq!(path, PathBuf::from("/custom/python"));
    }
}
