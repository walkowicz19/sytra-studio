use crate::merge_config::{MergeMethod, Verdict};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

fn default_downloadable() -> bool {
    true
}

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
    #[serde(default)]
    pub format: String,
    #[serde(default = "default_downloadable")]
    pub downloadable: bool,
    #[serde(default)]
    pub workflows: Vec<String>,
    #[serde(default)]
    pub gated: bool,
    #[serde(default)]
    pub approx_download_gb: Option<f64>,
    #[serde(default)]
    pub explicit_risks: Vec<String>,
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

        let cap = if method == MergeMethod::Moe { 16 } else { 3 };
        if model_refs.len() > cap {
            return Compatibility {
                verdict: Verdict::Red,
                reason: format!(
                    "Model count {} exceeds the product limit of {}",
                    model_refs.len(), cap
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
            if !model.allows_finetune() || !model.allows_download() {
                continue;
            }
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

    #[test]
    fn catalog_contains_exact_qwen_3_5_9b_base_contract() {
        let guider = Guider::new();
        let model = guider
            .resolve_model("Qwen/Qwen3.5-9B-Base")
            .expect("Qwen3.5 9B Base must be present in the embedded catalog");

        assert_eq!(model.architecture, "Qwen3_5ForConditionalGeneration");
        assert_eq!(model.param_count, 10_000_000_000);
        assert_eq!(model.dtype, "bfloat16");
        assert_eq!(model.license, "apache-2.0");
        assert_eq!(model.tokenizer_id, "Qwen/Qwen3.5-9B-Base");
        assert_eq!(
            model.default_target_modules,
            ["q_proj", "k_proj", "v_proj", "o_proj"]
        );
    }

    #[test]
    fn catalog_contains_kairos_family_bases() {
        let guider = Guider::new();
        assert!(
            guider.resolve_model("openbmb/MiniCPM5-2B").is_none(),
            "MiniCPM5-2B was a temporary 2B base and must not stay in the catalog"
        );

        let sophron = guider
            .resolve_model("Qwen/Qwen2.5-Coder-14B-Instruct")
            .expect("Qwen2.5-Coder-14B-Instruct is the Kairos Sophron base");
        assert_eq!(sophron.architecture, "Qwen2ForCausalLM");
        assert!(sophron.allows_download());
        assert!(sophron.allows_finetune());

        let sotos = guider
            .resolve_model("Qwen/Qwen3-14B")
            .expect("Qwen3-14B is the Kairos Sotos base");
        assert_eq!(sotos.architecture, "Qwen3ForCausalLM");
        assert!(sotos.allows_download());
        assert!(sotos.allows_finetune());

        let sophia = guider
            .resolve_model("Qwen/Qwen3.8-27B")
            .expect("Qwen3.8-27B is the Kairos Sophia base");
        assert_eq!(sophia.architecture, "Qwen3_5ForConditionalGeneration");
        assert!(sophia.allows_download());
        assert!(sophia.allows_finetune());
        assert_eq!(
            sophia.default_target_modules,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "in_proj_qkv",
                "in_proj_z",
                "out_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        );
    }

    #[test]
    fn catalog_is_expanded_and_download_gated_placeholders() {
        let guider = Guider::new();
        assert!(
            guider.catalog().len() >= 100,
            "catalog should list a broad Hugging Face download set, got {}",
            guider.catalog().len()
        );
        assert!(
            guider.catalog().iter().all(|m| m.allows_download()),
            "catalog should not include unusable non-downloadable test entries"
        );
        let synthetic_placeholder = ModelCatalogEntry {
            model_id: "org/knowledge-ft".into(),
            name: "Placeholder".into(),
            param_count: 1,
            architecture: "MistralForCausalLM".into(),
            dtype: "bfloat16".into(),
            moe_active_params: None,
            license: "apache-2.0".into(),
            default_target_modules: vec![],
            tokenizer_id: "mistralai/Mistral-7B-v0.1".into(),
            use_case_tags: vec![],
            benchmark_hint: "medium".into(),
            format: "safetensors".into(),
            downloadable: true,
            workflows: vec![],
            gated: false,
            approx_download_gb: None,
            explicit_risks: vec![],
        };
        assert!(!synthetic_placeholder.allows_download());
        let olmoe = guider
            .resolve_model("allenai/OLMoE-1B-7B-0125-Instruct-GGUF")
            .expect("OLMoE GGUF must be downloadable from the catalog");
        assert!(olmoe.allows_download());
        assert_eq!(olmoe.inferred_format(), "gguf");
    }

    #[test]
    fn qwen35_alerts_never_claim_qwen2() {
        use crate::model_alerts::alerts_for;
        let guider = Guider::new();
        let model = guider.resolve_model("Qwen/Qwen3.5-9B-Base").unwrap();
        let alerts = alerts_for(model, Some(12 * 1024), Some(16 * 1024));
        assert!(alerts.iter().any(|a| a.code == "never_qwen2"));
        assert!(alerts.iter().any(|a| a.code == "vision_not_text_lora"));
    }

    #[test]
    fn qwen38_sophia_alerts_hybrid_vision() {
        use crate::model_alerts::alerts_for;
        let guider = Guider::new();
        let model = guider
            .resolve_model("Qwen/Qwen3.8-27B")
            .expect("Qwen3.8-27B must be catalogued before this alert test");
        let alerts = alerts_for(model, Some(12 * 1024), Some(16 * 1024));
        assert!(alerts.iter().any(|a| a.code == "never_qwen2"));
        assert!(alerts.iter().any(|a| a.code == "vision_not_text_lora"));
    }
}
