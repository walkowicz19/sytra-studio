//! Persisted app settings shared by every front door (GUI, MCP).
//! One JSON file at the workspace root; loaded fresh on each use so a
//! change made in the GUI applies to the next MCP-started run and
//! vice versa without restarts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppSettings {
    /// Where Hugging Face models/datasets are cached (HF_HOME). None =
    /// `<workspace>/.hf-cache`. Users with a small system SSD point this
    /// at a big HDD.
    pub hf_cache_dir: Option<PathBuf>,
    /// Optional user-selected RAM ceiling for preflight checks. None uses
    /// all detected system memory.
    #[serde(default)]
    pub main_memory_limit_mb: Option<u64>,
}

impl AppSettings {
    pub fn path(workspace: &Path) -> PathBuf {
        workspace.join(".sytra-settings.json")
    }

    pub fn load(workspace: &Path) -> Self {
        std::fs::read_to_string(Self::path(workspace))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, workspace: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(Self::path(workspace), json).map_err(|e| e.to_string())
    }

    /// The HF cache directory runs should use, created if missing.
    pub fn effective_hf_cache(&self, workspace: &Path) -> PathBuf {
        let dir = self
            .hf_cache_dir
            .clone()
            .unwrap_or_else(|| workspace.join(".hf-cache"));
        std::fs::create_dir_all(&dir).ok();
        dir
    }

    pub fn effective_main_memory_mb(&self, detected_mb: u64) -> u64 {
        self.main_memory_limit_mb
            .unwrap_or(detected_mb)
            .clamp(2048, detected_mb.max(2048))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_workspace_hf_cache() {
        let ws = std::env::temp_dir().join("sytra-settings-test-default");
        std::fs::create_dir_all(&ws).unwrap();
        let s = AppSettings::load(&ws); // no file -> defaults
        assert_eq!(s.effective_hf_cache(&ws), ws.join(".hf-cache"));
        std::fs::remove_dir_all(&ws).ok();
    }

    #[test]
    fn round_trips_custom_cache_dir() {
        let ws = std::env::temp_dir().join("sytra-settings-test-roundtrip");
        std::fs::create_dir_all(&ws).unwrap();
        let custom = ws.join("elsewhere");
        let s = AppSettings {
            hf_cache_dir: Some(custom.clone()),
            main_memory_limit_mb: Some(8192),
        };
        s.save(&ws).unwrap();
        let loaded = AppSettings::load(&ws);
        assert_eq!(loaded.hf_cache_dir, Some(custom.clone()));
        assert_eq!(loaded.effective_hf_cache(&ws), custom);
        assert_eq!(loaded.effective_main_memory_mb(16384), 8192);
        std::fs::remove_dir_all(&ws).ok();
    }

    #[test]
    fn memory_limit_is_clamped_to_detected_ram() {
        let s = AppSettings {
            hf_cache_dir: None,
            main_memory_limit_mb: Some(999_999),
        };
        assert_eq!(s.effective_main_memory_mb(32768), 32768);
    }
}
