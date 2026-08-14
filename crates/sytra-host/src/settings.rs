//! Persisted app settings shared by every front door (GUI, MCP).
//! One JSON file at the workspace root; loaded fresh on each use so a
//! change made in the GUI applies to the next MCP-started run and
//! vice versa without restarts.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// Where Hugging Face models/datasets are cached (HF_HOME). None =
    /// `<workspace>/.hf-cache`. Users with a small system SSD point this
    /// at a big HDD.
    pub hf_cache_dir: Option<PathBuf>,
    /// Optional user-selected RAM ceiling for preflight checks. None uses
    /// all detected system memory.
    #[serde(default)]
    pub main_memory_limit_mb: Option<u64>,
    /// Whether to bypass HF token requirement for open models.
    #[serde(default = "default_true")]
    pub tokenless_download: bool,
    /// Quantization bit precision target (1, 2, 4 bits).
    #[serde(default)]
    pub low_bit_mode: Option<u8>,
    /// VRAM allocated for MoE expert caching (in MB).
    #[serde(default)]
    pub vram_expert_cache_mb: Option<u64>,
    /// Default context window for inference & GGUF export (tokens). Default: 4096.
    #[serde(default = "default_context_window")]
    pub default_context_window: usize,
    /// Default sampling temperature for inference. Default: 0.7.
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
    /// Whether to enable Flash Attention in local inference servers. Default: true.
    #[serde(default = "default_true")]
    pub enable_flash_attention: bool,
    /// KV Cache Quantization target ("fp16", "q8_0", "q4_0"). Default: "q8_0".
    #[serde(default = "default_kv_quant")]
    pub kv_cache_quant: String,
    /// Hard VRAM ceiling budget for model weights (in MB). Default: Some(8192) MB (8 GB).
    #[serde(default = "default_vram_limit")]
    pub vram_limit_mb: Option<u64>,
    /// Whether to offload 100% of KV Cache to CPU System RAM to prevent GPU VRAM overflow. Default: false.
    #[serde(default)]
    pub cpu_kv_cache: bool,
}

fn default_true() -> bool {
    true
}

fn default_context_window() -> usize {
    4096
}

fn default_temperature() -> f32 {
    0.7
}

fn default_kv_quant() -> String {
    "q8_0".to_string()
}

fn default_vram_limit() -> Option<u64> {
    Some(8192)
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            hf_cache_dir: None,
            main_memory_limit_mb: None,
            tokenless_download: true,
            low_bit_mode: Some(4),
            vram_expert_cache_mb: Some(4096),
            default_context_window: 4096,
            default_temperature: 0.7,
            enable_flash_attention: true,
            kv_cache_quant: "q8_0".to_string(),
            vram_limit_mb: Some(8192),
            cpu_kv_cache: false,
        }
    }
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
            ..Default::default()
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
            ..Default::default()
        };
        assert_eq!(s.effective_main_memory_mb(32768), 32768);
    }

    #[test]
    fn settings_json_round_trip_preserves_all_fields() {
        let ws = std::env::temp_dir().join("sytra-settings-field-snapshot");
        std::fs::create_dir_all(&ws).unwrap();
        let original = AppSettings {
            hf_cache_dir: Some(ws.join("cache")),
            main_memory_limit_mb: Some(12288),
            tokenless_download: false,
            low_bit_mode: Some(2),
            vram_expert_cache_mb: Some(2048),
            default_context_window: 8192,
            default_temperature: 0.2,
            enable_flash_attention: false,
            kv_cache_quant: "fp16".into(),
            vram_limit_mb: Some(4096),
            cpu_kv_cache: true,
        };
        original.save(&ws).unwrap();
        let loaded = AppSettings::load(&ws);
        let expected = serde_json::to_value(&original).unwrap();
        let actual = serde_json::to_value(&loaded).unwrap();
        assert_eq!(actual, expected);
        let keys: std::collections::BTreeSet<_> = actual.as_object().unwrap().keys().cloned().collect();
        let expected_keys: std::collections::BTreeSet<_> = [
            "hf_cache_dir",
            "main_memory_limit_mb",
            "tokenless_download",
            "low_bit_mode",
            "vram_expert_cache_mb",
            "default_context_window",
            "default_temperature",
            "enable_flash_attention",
            "kv_cache_quant",
            "vram_limit_mb",
            "cpu_kv_cache",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert_eq!(keys, expected_keys);
        std::fs::remove_dir_all(&ws).ok();
    }
}
