use crate::merge_config::{MergeMethod, Verdict};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub model_id: String,
    pub name: String,
    pub param_count: u64,
    pub architecture: String,
    pub dtype: String,
    pub moe_active_params: Option<u64>,
    pub license: String,
    pub default_target_modules: Vec<String>,
    pub tokenizer_id: String,
    pub use_case_tags: Vec<String>,
    pub benchmark_hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    pub accelerator: String, // "cuda" | "mps" | "cpu" | "rocm"
    pub total_vram_mb: u64,
    pub total_ram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRecipe {
    pub model_id: String,
    pub adapter_type: String,
    pub quant_bits: Option<u32>,
    pub quality_tier: String, // "A" | "B" | "C"
    pub estimated_vram_mb: u64,
    pub estimated_step_time_ms: u64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Compatibility {
    pub verdict: Verdict,
    pub reason: String,
}

pub struct Guider {
    catalog: Vec<ModelCatalogEntry>,
}

impl Guider {
    pub fn new() -> Self {
        let catalog_str = include_str!("catalog.json");
        let catalog: Vec<ModelCatalogEntry> = serde_json::from_str(catalog_str).unwrap_or_default();
        Self { catalog }
    }

    pub fn catalog(&self) -> &[ModelCatalogEntry] {
        &self.catalog
    }

    pub fn resolve_model(&self, model_ref: &str) -> Option<&ModelCatalogEntry> {
        self.catalog.iter().find(|m| m.model_id == model_ref)
    }

    /// Specialization markers that indicate a continued-pretrained lineage
    /// (not a fine-tune). Task-vector merges across lineages produce
    /// numerically valid but semantically destroyed models — measured on
    /// Qwen2.5: a true fine-tune differs from its base by ~1-2% weight
    /// norm, a specialized lineage by ~120%.
    const LINEAGE_MARKERS: [&'static str; 8] = [
        "coder", "code", "math", "vl", "vision", "audio", "omni", "moe",
    ];

    fn lineage_markers_of(model_ref: &str) -> Vec<&'static str> {
        let name = model_ref
            .rsplit('/')
            .next()
            .unwrap_or(model_ref)
            .to_lowercase();
        let tokens: Vec<&str> = name
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        Self::LINEAGE_MARKERS
            .iter()
            .copied()
            .filter(|m| tokens.iter().any(|t| t == m))
            .collect()
    }

    /// merge_check plus a lineage heuristic for task-vector methods when
    /// the base model is known: if a model's name carries a specialization
    /// marker (Coder, Math, VL, …) that the base lacks — or vice versa —
    /// the model is very likely a different continued-pretrained lineage,
    /// not a fine-tune of the base, and ties/dare_ties/task_arithmetic
    /// will destroy it. The runner performs the authoritative weight-delta
    /// check; this catches the mistake before anything is downloaded.
    pub fn merge_check_with_base(
        &self,
        base_model: Option<&str>,
        model_refs: &[String],
        method: MergeMethod,
    ) -> Compatibility {
        if method.is_task_vector() {
            if let Some(base) = base_model {
                let base_markers = Self::lineage_markers_of(base);
                for m_ref in model_refs {
                    let model_markers = Self::lineage_markers_of(m_ref);
                    if model_markers != base_markers {
                        return Compatibility {
                            verdict: Verdict::Amber,
                            reason: format!(
                                "'{m_ref}' looks like a different lineage than base '{base}' \
                                 (specialization markers differ). Task-vector methods \
                                 ({method:?}) require true fine-tunes of the base — merging \
                                 across lineages produces a broken model. Use slerp to \
                                 interpolate related models directly instead."
                            ),
                        };
                    }
                }
            }
        }
        self.merge_check(model_refs, method)
    }

    /// Checks the compatibility of multiple models for merging.
    pub fn merge_check(&self, model_refs: &[String], method: MergeMethod) -> Compatibility {
        if model_refs.is_empty() {
            return Compatibility {
                verdict: Verdict::Red,
                reason: "No models selected for merge".to_string(),
            };
        }

        if model_refs.len() > 3 {
            return Compatibility {
                verdict: Verdict::Red,
                reason: format!(
                    "Model count {} exceeds the product limit of 3",
                    model_refs.len()
                ),
            };
        }

        if method == MergeMethod::Slerp && model_refs.len() > 2 {
            return Compatibility {
                verdict: Verdict::Red,
                reason: "SLERP method supports at most 2 models".to_string(),
            };
        }

        // Resolve models
        let mut resolved_models = Vec::new();
        for m_ref in model_refs {
            match self.resolve_model(m_ref) {
                Some(entry) => resolved_models.push(entry),
                None => {
                    // If model is external/local, we can't fully inspect, so we default to amber
                    return Compatibility {
                        verdict: Verdict::Amber,
                        reason: format!("Model '{}' not found in pinned catalog. Merge safety cannot be guaranteed.", m_ref),
                    };
                }
            }
        }

        // Compare architectures
        let architectures: HashSet<&str> = resolved_models
            .iter()
            .map(|m| m.architecture.as_str())
            .collect();
        let tokenizers: HashSet<&str> = resolved_models
            .iter()
            .map(|m| m.tokenizer_id.as_str())
            .collect();

        if architectures.len() > 1 {
            if method.allows_cross_architecture() {
                return Compatibility {
                    verdict: Verdict::Amber,
                    reason: format!(
                        "Cross-architecture merge ({:?}) is experimental and requires healing afterwards.",
                        architectures
                    ),
                };
            } else {
                return Compatibility {
                    verdict: Verdict::Red,
                    reason: format!(
                        "Merge method {:?} does not support merging different architectures: {:?}",
                        method, architectures
                    ),
                };
            }
        }

        if tokenizers.len() > 1 {
            return Compatibility {
                verdict: Verdict::Amber,
                reason: "Tokenizers differ. A union tokenizer or token transplantation (tokensurgeon) will be required.".to_string(),
            };
        }

        Compatibility {
            verdict: Verdict::Green,
            reason: "All models are architectural siblings and share identical tokenizers."
                .to_string(),
        }
    }

    /// Recommends training recipes based on hardware capability.
    pub fn recommend(&self, hw: &HardwareCapabilities) -> Vec<TrainRecipe> {
        let mut recipes = Vec::new();

        for model in &self.catalog {
            // Estimate standard VRAM for full/lora/qlora
            // LLaMA/Mistral estimation formula:
            let model_params = model.param_count as f64;

            // Recipe 1: QLoRA (4-bit quant)
            let qlora_vram = ((model_params * 0.55) / 1e6) as u64 + 1024; // MB
            if qlora_vram < hw.total_vram_mb {
                recipes.push(TrainRecipe {
                    model_id: model.model_id.clone(),
                    adapter_type: "qlora".to_string(),
                    quant_bits: Some(4),
                    quality_tier: "A".to_string(),
                    estimated_vram_mb: qlora_vram,
                    estimated_step_time_ms: 120,
                    reason: "Fits comfortably in VRAM using 4-bit quantization.".to_string(),
                });
            }

            // Recipe 2: LoRA (16-bit)
            let lora_vram = ((model_params * 2.0) / 1e6) as u64 + 2048; // MB
            if lora_vram < hw.total_vram_mb {
                recipes.push(TrainRecipe {
                    model_id: model.model_id.clone(),
                    adapter_type: "lora".to_string(),
                    quant_bits: None,
                    quality_tier: "A".to_string(),
                    estimated_vram_mb: lora_vram,
                    estimated_step_time_ms: 180,
                    reason: "Full 16-bit precision weights with low-rank adapter.".to_string(),
                });
            } else if lora_vram < hw.total_vram_mb + (hw.total_ram_mb / 2) {
                // If it can fit with offloading
                recipes.push(TrainRecipe {
                    model_id: model.model_id.clone(),
                    adapter_type: "lora".to_string(),
                    quant_bits: None,
                    quality_tier: "B".to_string(),
                    estimated_vram_mb: lora_vram,
                    estimated_step_time_ms: 850,
                    reason: "Requires optimizer CPU offloading, which slows step times."
                        .to_string(),
                });
            }
        }

        recipes
    }
}

#[cfg(test)]
mod lineage_tests {
    use super::*;

    #[test]
    fn task_vector_flags_cross_lineage_models() {
        let guider = Guider::new();
        let compat = guider.merge_check_with_base(
            Some("Qwen/Qwen2.5-7B"),
            &[
                "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
                "Qwen/Qwen2.5-7B-Instruct".to_string(),
            ],
            MergeMethod::DareTies,
        );
        assert_eq!(compat.verdict, Verdict::Amber);
        assert!(
            compat.reason.contains("lineage"),
            "reason: {}",
            compat.reason
        );
    }

    #[test]
    fn task_vector_allows_same_lineage_finetunes() {
        let guider = Guider::new();
        let compat = guider.merge_check_with_base(
            Some("Qwen/Qwen2.5-Coder-7B"),
            &[
                "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
                "org/my-coder-ft".to_string(),
            ],
            MergeMethod::DareTies,
        );
        // Same markers on both sides -> falls through to the normal check.
        assert!(
            !compat.reason.contains("lineage"),
            "reason: {}",
            compat.reason
        );
    }

    #[test]
    fn slerp_skips_lineage_heuristic() {
        let guider = Guider::new();
        let compat = guider.merge_check_with_base(
            Some("Qwen/Qwen2.5-Coder-7B-Instruct"),
            &[
                "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
                "Qwen/Qwen2.5-7B-Instruct".to_string(),
            ],
            MergeMethod::Slerp,
        );
        assert!(
            !compat.reason.contains("lineage"),
            "reason: {}",
            compat.reason
        );
    }
}
