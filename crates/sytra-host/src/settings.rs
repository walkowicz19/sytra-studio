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

    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2_097_152).contains(&self.default_context_window) {
            return Err("Default context window must be between 1 and 2097152".into());
        }
        if !self.default_temperature.is_finite() || !(0.0..=2.0).contains(&self.default_temperature) {
            return Err("Default temperature must be between 0 and 2".into());
        }
        if !matches!(self.kv_cache_quant.as_str(), "fp16" | "q8_0" | "q4_0") {
            return Err("KV cache quant must be fp16, q8_0, or q4_0".into());
        }
        if self.vram_limit_mb == Some(0) {
            return Err("VRAM limit must be positive".into());
        }
        if self.vram_expert_cache_mb == Some(0) {
            return Err("VRAM expert cache must be positive".into());
        }
        Ok(())
    }

    pub fn load(workspace: &Path) -> Self {
        std::fs::read_to_string(Self::path(workspace))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, workspace: &Path) -> Result<(), String> {
        self.validate()?;
        let path = Self::path(workspace);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            let temp = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
            let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
            std::fs::write(&temp, json).map_err(|e| e.to_string())?;
            if let Err(err) = std::fs::rename(&temp, &path) {
                let _ = std::fs::remove_file(&temp);
                return Err(err.to_string());
            }
            return Ok(());
        }
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceSettingsInput {
    pub default_context_window: Option<usize>,
    pub default_temperature: Option<f32>,
    pub enable_flash_attention: Option<bool>,
    pub kv_cache_quant: Option<String>,
    pub vram_limit_mb: Option<Option<u64>>,
    pub cpu_kv_cache: Option<bool>,
    pub vram_expert_cache_mb: Option<Option<u64>>,
}

pub fn settings_payload(
    settings: &AppSettings,
    workspace: &Path,
    detected_ram_mb: u64,
) -> serde_json::Value {
    serde_json::json!({
        "hf_cache_dir": settings.effective_hf_cache(workspace).display().to_string(),
        "is_custom": settings.hf_cache_dir.is_some(),
        "main_memory_limit_mb": settings.main_memory_limit_mb,
        "effective_main_memory_mb": settings.effective_main_memory_mb(detected_ram_mb),
        "detected_ram_mb": detected_ram_mb,
        "default_context_window": settings.default_context_window,
        "default_temperature": settings.default_temperature,
        "enable_flash_attention": settings.enable_flash_attention,
        "kv_cache_quant": settings.kv_cache_quant,
        "vram_limit_mb": settings.vram_limit_mb,
        "cpu_kv_cache": settings.cpu_kv_cache,
        "vram_expert_cache_mb": settings.vram_expert_cache_mb,
        "memory_limit_note": "main_memory_limit_mb gates preflight estimates only; it does not cap process RSS.",
    })
}

pub fn apply_inference_settings(
    workspace: &Path,
    detected_ram_mb: u64,
    input: InferenceSettingsInput,
) -> Result<serde_json::Value, String> {
    let mut current = AppSettings::load(workspace);
    if let Some(ctx) = input.default_context_window {
        current.default_context_window = ctx;
    }
    if let Some(temp) = input.default_temperature {
        current.default_temperature = temp;
    }
    if let Some(fa) = input.enable_flash_attention {
        current.enable_flash_attention = fa;
    }
    if let Some(kv) = input.kv_cache_quant {
        current.kv_cache_quant = kv;
    }
    if let Some(vram) = input.vram_limit_mb {
        if let (Some(requested), Some(detected)) =
            (vram, crate::BackendResolver::detect_system_vram_mb())
        {
            if requested > detected {
                return Err(format!(
                    "VRAM limit ({requested} MB) exceeds detected GPU VRAM ({detected} MB)"
                ));
            }
        }
        current.vram_limit_mb = vram;
    }
    if let Some(cpu_kv) = input.cpu_kv_cache {
        current.cpu_kv_cache = cpu_kv;
    }
    if let Some(expert_cache) = input.vram_expert_cache_mb {
        current.vram_expert_cache_mb = expert_cache;
    }
    current.save(workspace)?;
    Ok(settings_payload(&current, workspace, detected_ram_mb))
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

    #[test]
    fn save_validates_and_preserves_prior_settings_on_error() {
        let ws = std::env::temp_dir().join(format!("sytra-settings-validate-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&ws).unwrap();
        let valid = AppSettings {
            default_context_window: 16384,
            default_temperature: 0.3,
            kv_cache_quant: "q4_0".into(),
            ..Default::default()
        };
        valid.save(&ws).unwrap();

        let mut bad = valid.clone();
        bad.default_temperature = 2.5;
        assert!(bad.save(&ws).is_err());

        bad = valid.clone();
        bad.kv_cache_quant = "invalid".into();
        assert!(bad.save(&ws).is_err());

        bad = valid.clone();
        bad.vram_limit_mb = Some(0);
        assert!(bad.save(&ws).is_err());

        let reloaded = AppSettings::load(&ws);
        assert_eq!(reloaded.default_context_window, 16384);
        assert_eq!(reloaded.kv_cache_quant, "q4_0");
        std::fs::remove_dir_all(&ws).ok();
    }
}
