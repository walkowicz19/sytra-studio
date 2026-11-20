//! Shared local model scanning across the desktop GUI and MCP server.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::settings::AppSettings;
use crate::workspace::{default_model_dir, user_home_dir};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LocalModelItem {
    pub id: String,
    pub name: String,
    pub category: String,
    pub path: String,
    pub size_gb: f64,
    pub format: String,
}

fn hf_friendly_name(dir: &Path) -> Option<String> {
    for ancestor in dir.ancestors() {
        if let Some(name) = ancestor.file_name().and_then(|n| n.to_str()) {
            if let Some(rest) = name.strip_prefix("models--") {
                return Some(rest.replace("--", "/"));
            }
        }
    }
    None
}

pub fn scan_local_models(workspace: &Path, custom_dir: Option<&str>) -> Vec<LocalModelItem> {
    let mut items = Vec::new();
    let settings = AppSettings::load(workspace);
    let hf_cache = settings.effective_hf_cache(workspace);
    let mut search_dirs: Vec<(&str, PathBuf)> = vec![
        ("hf-cache", hf_cache.clone()),
        ("hf-cache", hf_cache.join("hub")),
        ("hf-cache", workspace.join("runner").join(".hf-cache")),
        (
            "hf-cache",
            workspace.join("runner").join(".hf-cache").join("hub"),
        ),
        ("lm-studio", default_model_dir()),
        ("finetuned", workspace.join("runs")),
        ("merged", workspace.join("runs").join("merged")),
    ];

    if let Some(hf_home) = std::env::var_os("HF_HOME").filter(|s| !s.is_empty()) {
        let p = PathBuf::from(hf_home);
        search_dirs.push(("hf-cache", p.join("hub")));
        search_dirs.push(("hf-cache", p));
    }
    if let Some(home) = user_home_dir() {
        search_dirs.push(("hf-cache", home.join(".cache").join("huggingface").join("hub")));
        search_dirs.push(("lm-studio", home.join(".lmstudio").join("models")));
        search_dirs.push(("lm-studio", home.join(".cache").join("lm-studio").join("models")));
        search_dirs.push(("lm-studio", home.join("Documents").join("LM Studio").join("models")));
    }

    if let Some(c_dir) = custom_dir {
        if !c_dir.trim().is_empty() {
            search_dirs.insert(0, ("custom", PathBuf::from(c_dir)));
        }
    }

    let mut seen = std::collections::HashSet::new();
    for (cat, base_dir) in search_dirs {
        if !base_dir.exists() {
            continue;
        }
        let mut stack = vec![base_dir.clone()];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                let mut child_dirs = Vec::new();
                let mut gguf_or_bin = Vec::new();
                let mut safetensors_files = Vec::new();
                let mut has_config_json = false;

                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if !fname.starts_with('.')
                            && path.components().count() <= base_dir.components().count() + 6
                        {
                            child_dirs.push(path);
                        }
                    } else if path.is_file() {
                        let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if fname == "config.json" {
                            has_config_json = true;
                        }
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                            let ext_lower = ext.to_lowercase();
                            let ext_owned = ext.to_string();
                            if ext_lower == "safetensors" {
                                safetensors_files.push(path);
                            } else if matches!(ext_lower.as_str(), "gguf" | "bin" | "pth")
                                && !fname.ends_with(".part")
                            {
                                gguf_or_bin.push((path, ext_owned));
                            }
                        }
                    }
                }

                if has_config_json && !safetensors_files.is_empty() {
                    if let Ok(canon) = dir.canonicalize() {
                        let p_str = canon.to_string_lossy().to_string();
                        if seen.insert(p_str.clone()) {
                            let total_bytes: u64 = safetensors_files
                                .iter()
                                .map(|p| p.metadata().map(|m| m.len()).unwrap_or(0))
                                .sum();
                            let size_gb = (total_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
                            let label = hf_friendly_name(&dir).unwrap_or_else(|| {
                                dir.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string()
                            });
                            items.push(LocalModelItem {
                                id: label.clone(),
                                name: label,
                                category: cat.to_string(),
                                path: p_str,
                                size_gb: (size_gb * 100.0).round() / 100.0,
                                format: "safetensors".to_string(),
                            });
                        }
                    }
                } else {
                    for (path, ext) in gguf_or_bin {
                        let canon = path.canonicalize().unwrap_or(path);
                        let p_str = canon.to_string_lossy().to_string();
                        if seen.insert(p_str.clone()) {
                            let size_gb =
                                (canon.metadata().map(|m| m.len()).unwrap_or(0) as f64)
                                    / (1024.0 * 1024.0 * 1024.0);
                            items.push(LocalModelItem {
                                id: canon
                                    .file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                name: canon
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                category: cat.to_string(),
                                path: p_str,
                                size_gb: (size_gb * 100.0).round() / 100.0,
                                format: ext,
                            });
                        }
                    }
                    for path in safetensors_files {
                        let canon = path.canonicalize().unwrap_or(path);
                        let p_str = canon.to_string_lossy().to_string();
                        if seen.insert(p_str.clone()) {
                            let size_gb =
                                (canon.metadata().map(|m| m.len()).unwrap_or(0) as f64)
                                    / (1024.0 * 1024.0 * 1024.0);
                            items.push(LocalModelItem {
                                id: canon
                                    .file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                name: canon
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string(),
                                category: cat.to_string(),
                                path: p_str,
                                size_gb: (size_gb * 100.0).round() / 100.0,
                                format: "safetensors".to_string(),
                            });
                        }
                    }
                }

                stack.extend(child_dirs);
            }
        }
    }
    items.sort_by(|a, b| a.category.cmp(&b.category).then_with(|| a.name.cmp(&b.name)));
    items
}
