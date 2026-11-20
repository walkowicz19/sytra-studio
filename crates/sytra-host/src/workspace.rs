use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::env_provisioner::EnvProvisioner;
use crate::mcp_install::safe_relative_components;

include!(concat!(env!("OUT_DIR"), "/embedded_runner.rs"));

const RUNNER_STAMP_FILE: &str = ".sytra-runner-version";
const RUNNER_VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn find_project_root() -> Option<PathBuf> {
    let mut fallback = None;
    if let Ok(mut dir) = std::env::current_exe() {
        while dir.pop() {
            if dir.join("runner").join("sytra_runner").exists() {
                // Tauri resource bundling can copy `runner/` under `target-build/debug/`
                // or `target/debug/`. Prefer the real repository root when running dev/test
                // binaries, falling back to the closest bundled directory in packaged builds.
                if dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir() {
                    return Some(dir);
                }
                if fallback.is_none() {
                    fallback = Some(dir.clone());
                }
            }
        }
    }
    fallback
}

pub fn default_user_workspace() -> Option<PathBuf> {
    user_home_dir().map(|home| home.join(".sytra"))
}

fn resolve_workspace_with(
    env_override: Option<String>,
    project_root: Option<PathBuf>,
    home_dir: Option<PathBuf>,
    cwd: PathBuf,
) -> PathBuf {
    env_override
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .or(project_root)
        .or_else(|| home_dir.map(|home| home.join(".sytra")))
        .unwrap_or(cwd)
}

pub fn resolve_workspace() -> PathBuf {
    resolve_workspace_with(
        std::env::var("SYTRA_WORKSPACE").ok(),
        find_project_root(),
        user_home_dir(),
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    )
}

/// Ensures bundled Python runner files are deployed under `<workspace>/runner/`
/// when launching standalone packaged binaries without NPM or a git checkout.
pub fn ensure_runner_deployed(workspace: &Path) -> std::io::Result<bool> {
    if EMBEDDED_RUNNER_FILES.is_empty() {
        return Ok(false);
    }
    if workspace.join("Cargo.toml").is_file() && workspace.join("crates").is_dir() {
        return Ok(false);
    }

    let runner_dir = workspace.join("runner");
    let stamp_path = runner_dir.join(RUNNER_STAMP_FILE);
    let already_current = runner_dir.join("sytra_runner").join("__init__.py").is_file()
        && runner_dir.join("pyproject.toml").is_file()
        && fs::read_to_string(&stamp_path)
            .map(|s| s.trim() == RUNNER_VERSION)
            .unwrap_or(false);
    if already_current {
        return Ok(false);
    }

    fs::create_dir_all(&runner_dir)?;
    let pid = std::process::id();
    for (rel, bytes) in EMBEDDED_RUNNER_FILES {
        let Some(clean_rel) = safe_relative_components(rel) else {
            continue;
        };
        let dest = runner_dir.join(clean_rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let file_name = dest
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        let tmp = dest.with_file_name(format!(".{file_name}.tmp-{pid}"));
        fs::write(&tmp, bytes)?;
        if let Err(err) = fs::rename(&tmp, &dest) {
            if err.kind() == ErrorKind::AlreadyExists || cfg!(target_os = "windows") {
                let _ = fs::remove_file(&dest);
                fs::rename(&tmp, &dest)?;
            } else {
                let _ = fs::remove_file(&tmp);
                return Err(err);
            }
        }
    }

    fs::write(stamp_path, format!("{RUNNER_VERSION}\n"))?;
    Ok(true)
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
    if provisioner.is_download_provisioned() {
        return provisioner.download_python_path();
    }
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

    #[test]
    fn standalone_workspace_defaults_to_home_sytra() {
        let home = PathBuf::from("/tmp/alice");
        let resolved = resolve_workspace_with(None, None, Some(home), PathBuf::from("/tmp/cwd"));
        assert_eq!(resolved, PathBuf::from("/tmp/alice").join(".sytra"));

        let explicit = resolve_workspace_with(
            Some("/custom/ws".to_string()),
            Some(PathBuf::from("/repo")),
            Some(PathBuf::from("/tmp/alice")),
            PathBuf::from("/tmp/cwd"),
        );
        assert_eq!(explicit, PathBuf::from("/custom/ws"));
    }

    #[test]
    fn ensure_runner_deployed_populates_standalone_workspace_and_is_idempotent() {
        let dir = std::env::temp_dir().join(format!("sytra-runner-deploy-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        assert!(ensure_runner_deployed(&dir).unwrap());
        assert!(dir.join("runner").join("sytra_runner").join("__init__.py").is_file());
        assert!(dir.join("runner").join("pyproject.toml").is_file());
        assert!(!ensure_runner_deployed(&dir).unwrap());

        let _ = fs::remove_dir_all(&dir);
    }
}
