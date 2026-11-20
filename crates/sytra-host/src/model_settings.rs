//! Per-model inference preferences, keyed by a hash of the canonical model path.
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    pub context_length: Option<usize>,
    pub temperature: Option<f32>,
    pub vram_limit_mb: Option<usize>,
    pub vram_guardrail_mb: Option<usize>,
    pub kv_cache_quant: Option<String>,
    pub cpu_kv_cache: Option<bool>,
    pub flash_attention: Option<bool>,
    pub idle_ttl_minutes: Option<usize>,
    pub draft_model_path: Option<String>,
    pub allow_unverified: Option<bool>,
}

impl ModelSettings {
    pub fn validate(&self) -> Result<(), String> {
        if self.context_length.is_some_and(|value| value == 0 || value > 2_097_152) {
            return Err("Context length must be between 1 and 2097152".into());
        }
        if self.temperature.is_some_and(|value| !value.is_finite() || !(0.0..=2.0).contains(&value)) {
            return Err("Temperature must be between 0 and 2".into());
        }
        if self.vram_limit_mb == Some(0) {
            return Err("VRAM limit must be positive".into());
        }
        if self.idle_ttl_minutes.is_some_and(|value| value > 10_080) {
            return Err("Idle timeout must be between 0 and 10080 minutes".into());
        }
        if self.draft_model_path.as_ref().is_some_and(|value| value.trim().is_empty() || value.trim().starts_with('-')) {
            return Err("Draft model path must be non-empty and cannot start with '-'".into());
        }
        if self.kv_cache_quant.as_deref().is_some_and(|value| !matches!(value, "fp16" | "q8_0" | "q4_0")) {
            return Err("KV cache must be fp16, q8_0, or q4_0".into());
        }
        Ok(())
    }
}

fn settings_path(workspace: &Path, model: &Path) -> Result<PathBuf, String> {
    let canonical = model.canonicalize().map_err(|e| format!("Invalid model path: {e}"))?;
    let path = canonical.to_str().ok_or("Model path is not valid UTF-8")?;
    let hash = format!("{:x}", Sha256::digest(path.as_bytes()));
    Ok(workspace.join(".sytra").join("models").join(format!("{hash}.json")))
}

pub fn load(workspace: &Path, model: &Path) -> Result<ModelSettings, String> {
    let path = settings_path(workspace, model)?;
    if path.parent().is_some_and(|parent| parent.exists()) {
        let parent = path.parent().unwrap().canonicalize().map_err(|e| e.to_string())?;
        let root = workspace.canonicalize().map_err(|e| e.to_string())?;
        if !parent.starts_with(root) { return Err("Model settings directory must be inside the workspace".into()); }
    }
    if std::fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err("Model settings file cannot be a symlink".into());
    }
    let contents = match std::fs::read(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(ModelSettings::default()),
        Err(error) => return Err(format!("Cannot read model settings: {error}")),
    };
    let settings: ModelSettings = serde_json::from_slice(&contents).map_err(|e| format!("Invalid model settings: {e}"))?;
    settings.validate()?;
    Ok(settings)
}

pub fn save(workspace: &Path, model: &Path, settings: &ModelSettings) -> Result<(), String> {
    settings.validate()?;
    let path = settings_path(workspace, model)?;
    let parent = path.parent().ok_or("Invalid settings directory")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let resolved_parent = parent.canonicalize().map_err(|e| e.to_string())?;
    let resolved_workspace = workspace.canonicalize().map_err(|e| e.to_string())?;
    if !resolved_parent.starts_with(resolved_workspace) {
        return Err("Model settings directory must be inside the workspace".into());
    }
    let temp = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let bytes = serde_json::to_vec_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(&temp, bytes).map_err(|e| format!("Cannot write model settings: {e}"))?;
    if let Err(error) = std::fs::rename(&temp, &path) {
        let _ = std::fs::remove_file(&temp);
        return Err(format!("Cannot save model settings: {error}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn persists_independently_and_rejects_invalid_settings() {
        let root = std::env::temp_dir().join(format!("sytra-model-settings-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let first = root.join("first.gguf");
        let second = root.join("second.gguf");
        std::fs::write(&first, b"model").unwrap();
        std::fs::write(&second, b"model").unwrap();
        let mut options = ModelSettings { context_length: Some(8192), vram_guardrail_mb: Some(1024), kv_cache_quant: Some("q4_0".into()), ..Default::default() };
        save(&root, &first, &options).unwrap();
        assert_eq!(load(&root, &first).unwrap(), options);
        assert_eq!(load(&root, &second).unwrap(), ModelSettings::default());
        options.context_length = Some(16384);
        options.temperature = Some(0.25);
        save(&root, &first, &options).unwrap();
        assert_eq!(load(&root, &first).unwrap(), options);
        options.temperature = Some(2.1);
        assert!(save(&root, &first, &options).is_err());
        options.temperature = Some(0.25);
        options.kv_cache_quant = Some("unknown".into());
        assert!(save(&root, &first, &options).is_err());
        assert_eq!(load(&root, &first).unwrap().context_length, Some(16384));
        options.kv_cache_quant = None;
        options.idle_ttl_minutes = Some(0);
        save(&root, &first, &options).unwrap();
        assert_eq!(load(&root, &first).unwrap().idle_ttl_minutes, Some(0));
        options.idle_ttl_minutes = Some(10_081);
        assert!(save(&root, &first, &options).is_err());
        options.idle_ttl_minutes = None;
        options.draft_model_path = Some("draft.gguf".into());
        save(&root, &first, &options).unwrap();
        assert_eq!(load(&root, &first).unwrap().draft_model_path.as_deref(), Some("draft.gguf"));
        std::fs::remove_dir_all(root).unwrap();
    }
}
