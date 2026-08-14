use serde::Serialize;
use sytra_contracts::guider::{Guider, ModelCatalogEntry};
use sytra_contracts::model_alerts::{alerts_for, peak_alert_level, ModelAlert};

#[derive(Debug, Clone, Serialize)]
pub struct HubCatalogEntry {
    pub id: String,
    pub name: String,
    pub size_gb: f64,
    pub format: String,
    pub tags: Vec<String>,
    pub recommended: bool,
    pub architecture: String,
    pub license: String,
    pub downloadable: bool,
    pub gated: bool,
    pub alert_level: String,
    pub alerts: Vec<ModelAlert>,
    pub param_count: u64,
    pub moe_active_params: Option<u64>,
}

pub fn require_catalog_download<'a>(
    guider: &'a Guider,
    repo_id: &str,
) -> Result<&'a ModelCatalogEntry, String> {
    let entry = guider.resolve_model(repo_id).ok_or_else(|| {
        format!(
            "Sytra MCP/UI downloads are limited to the pinned catalog. '{repo_id}' is not listed. Call list_catalog (or open Model Hub) and pick an exact model_id. Arbitrary Hugging Face IDs are rejected so Sytra can attach architecture/license/memory risk alerts before the Xet downloader starts."
        )
    })?;
    if !entry.allows_download() {
        return Err(format!(
            "'{repo_id}' is in the catalog but is not downloadable (placeholder, MLX-only, or blocked)."
        ));
    }
    Ok(entry)
}

pub fn hub_entries(
    guider: &Guider,
    vram_mb: Option<u64>,
    ram_mb: Option<u64>,
) -> Vec<HubCatalogEntry> {
    guider
        .catalog()
        .iter()
        .map(|entry| to_hub_entry(entry, vram_mb, ram_mb))
        .collect()
}

fn to_hub_entry(
    entry: &ModelCatalogEntry,
    vram_mb: Option<u64>,
    ram_mb: Option<u64>,
) -> HubCatalogEntry {
    let size_gb = (entry.download_size_gb() * 100.0).round() / 100.0;
    let format = entry.inferred_format().to_string();
    let alerts = alerts_for(entry, vram_mb, ram_mb);
    let alert_level = peak_alert_level(&alerts);
    let recommended = match vram_mb {
        Some(v) if entry.allows_download() && alert_level != "danger" => {
            size_gb <= (v as f64 / 1024.0) * 0.7
        }
        _ => false,
    };
    HubCatalogEntry {
        id: entry.model_id.clone(),
        name: entry.name.clone(),
        size_gb,
        format,
        tags: entry.use_case_tags.clone(),
        recommended,
        architecture: entry.architecture.clone(),
        license: entry.license.clone(),
        downloadable: entry.allows_download(),
        gated: entry.gated,
        alert_level,
        alerts,
        param_count: entry.param_count,
        moe_active_params: entry.moe_active_params,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_comes_from_guider_not_a_hardcoded_list() {
        let guider = Guider::new();
        let entries = hub_entries(&guider, Some(24 * 1024), Some(16 * 1024));
        assert!(!entries.is_empty());
        assert!(entries.iter().any(|e| e.id == "mistralai/Mistral-7B-v0.1"));
        assert!(entries.iter().any(|e| e.alerts.iter().any(|a| a.code == "never_qwen2")));
    }

    #[test]
    fn unknown_vram_marks_nothing_recommended() {
        let guider = Guider::new();
        let entries = hub_entries(&guider, None, None);
        assert!(entries.iter().all(|e| !e.recommended));
    }

    #[test]
    fn download_rejects_unknown_huggingface_ids() {
        let guider = Guider::new();
        let err = require_catalog_download(&guider, "totally-unknown/not-in-catalog").unwrap_err();
        assert!(err.contains("not listed"), "{err}");
        assert!(require_catalog_download(&guider, "org/knowledge-ft").is_err());
        assert!(require_catalog_download(&guider, "Qwen/Qwen2.5-0.5B-Instruct-GGUF").is_ok());
    }
}
