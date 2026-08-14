//! Hardware-aware risk alerts for catalog models.
use serde::{Deserialize, Serialize};

use crate::guider::ModelCatalogEntry;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelAlert {
    pub level: String,
    pub code: String,
    pub message: String,
    pub blocks_download: bool,
}

pub fn peak_alert_level(alerts: &[ModelAlert]) -> String {
    if alerts.iter().any(|a| a.level == "danger") {
        "danger".into()
    } else if alerts.iter().any(|a| a.level == "warning") {
        "warning".into()
    } else if alerts.is_empty() {
        "none".into()
    } else {
        "info".into()
    }
}

fn fp16_gb(params: u64) -> f64 {
    (params as f64 * 2.0) / 1_000_000_000.0
}

fn q4_gb(params: u64) -> f64 {
    (params as f64 * 0.55) / 1_000_000_000.0
}

impl ModelCatalogEntry {
    pub fn inferred_format(&self) -> &str {
        if !self.format.is_empty() {
            return self.format.as_str();
        }
        if self.model_id.to_lowercase().contains("gguf") {
            "gguf"
        } else if self.model_id.to_lowercase().contains("mlx") {
            "mlx"
        } else {
            "safetensors"
        }
    }

    pub fn download_size_gb(&self) -> f64 {
        if let Some(gb) = self.approx_download_gb {
            return gb;
        }
        let params = self.moe_active_params.unwrap_or(self.param_count);
        if self.inferred_format() == "gguf" {
            q4_gb(self.param_count.max(params))
        } else {
            fp16_gb(self.param_count)
        }
    }

    pub fn allows_download(&self) -> bool {
        self.downloadable && !self.model_id.starts_with("org/")
    }

    pub fn allows_finetune(&self) -> bool {
        if !self.workflows.is_empty() {
            return self.workflows.iter().any(|w| w == "finetune");
        }
        self.inferred_format() == "safetensors"
    }
}

pub fn alerts_for(
    entry: &ModelCatalogEntry,
    vram_mb: Option<u64>,
    ram_mb: Option<u64>,
) -> Vec<ModelAlert> {
    let mut alerts = Vec::new();
    let format = entry.inferred_format();
    let id_l = entry.model_id.to_lowercase();
    let arch_l = entry.architecture.to_lowercase();
    let name_l = entry.name.to_lowercase();

    if entry.model_id.starts_with("org/") {
        alerts.push(ModelAlert {
            level: "danger".into(),
            code: "not_a_huggingface_repo".into(),
            message: format!(
                "{} is a Sytra merge/train placeholder, not a Hugging Face repository. It cannot be downloaded.",
                entry.model_id
            ),
            blocks_download: true,
        });
    }

    if format == "mlx" || id_l.contains("mlx-community") {
        alerts.push(ModelAlert {
            level: "danger".into(),
            code: "mlx_apple_only".into(),
            message: "This checkpoint is an MLX 4-bit package for Apple Silicon. It will not run on Windows CUDA or llama.cpp.".into(),
            blocks_download: cfg!(windows),
        });
    }

    if matches!(
        entry.license.to_lowercase().as_str(),
        "llama3" | "llama3.1" | "llama3.2" | "llama3.3" | "gemma" | "gemma-2" | "llama2"
    ) || entry.gated
    {
        alerts.push(ModelAlert {
            level: "warning".into(),
            code: "gated_license".into(),
            message: format!(
                "License '{}' is gated on Hugging Face. Accept the license and set HF_TOKEN before downloading.",
                entry.license
            ),
            blocks_download: false,
        });
    }

    if format == "safetensors" {
        alerts.push(ModelAlert {
            level: "warning".into(),
            code: "never_ollama_safetensors".into(),
            message: "Never `ollama create` from raw SafeTensors. Convert to GGUF with Sytra first or the tokenizer/architecture can silently break.".into(),
            blocks_download: false,
        });
    }

    if arch_l.contains("qwen3_5")
        || arch_l.contains("qwen3.5")
        || id_l.contains("qwen3.5")
        || name_l.contains("qwen3.5")
    {
        alerts.push(ModelAlert {
            level: "danger".into(),
            code: "never_qwen2".into(),
            message: "This is Qwen3.5, not Qwen2. Serving or training it as Qwen2 will load the wrong kernel and corrupt LoRA matches.".into(),
            blocks_download: false,
        });
        if entry.use_case_tags.iter().any(|t| t == "multimodal" || t == "vision")
            || arch_l.contains("conditionalgeneration")
        {
            alerts.push(ModelAlert {
                level: "warning".into(),
                code: "vision_not_text_lora".into(),
                message: "Qwen3.5 multimodal weights include vision tensors. Sytra text LoRA must target q/k/v/o only; vision modules must stay unmatched.".into(),
                blocks_download: false,
            });
        }
    }

    if entry.moe_active_params.is_some()
        || entry.use_case_tags.iter().any(|t| t == "moe")
        || arch_l.contains("moe")
        || id_l.contains("moe")
        || id_l.contains("mixtral")
        || id_l.contains("olmoe")
    {
        alerts.push(ModelAlert {
            level: "warning".into(),
            code: "moe_hybrid".into(),
            message: format!(
                "MoE: total parameters {:.1}B, active per token {:.1}B. Keep attention/KV/active experts on GPU; mmap cold experts. Do not load every expert into VRAM.",
                entry.param_count as f64 / 1e9,
                entry.moe_active_params.unwrap_or(entry.param_count) as f64 / 1e9
            ),
            blocks_download: false,
        });
    }

    let download_gb = entry.download_size_gb();
    if let Some(vram) = vram_mb {
        let vram_gb = vram as f64 / 1024.0;
        if download_gb > vram_gb * 0.85 && format != "gguf" {
            alerts.push(ModelAlert {
                level: "danger".into(),
                code: "exceeds_vram".into(),
                message: format!(
                    "Estimated {:.1} GB weights exceed ~{:.0} GB VRAM. Fine-tuning this checkpoint on this GPU will OOM without a much smaller quant or a smaller model.",
                    download_gb, vram_gb
                ),
                blocks_download: false,
            });
        } else if format == "gguf" && download_gb > vram_gb + 2.0 {
            if let Some(ram) = ram_mb {
                let ram_gb = ram as f64 / 1024.0;
                if download_gb > vram_gb * 0.8 + ram_gb * 0.7 {
                    alerts.push(ModelAlert {
                        level: "danger".into(),
                        code: "exceeds_hybrid_envelope".into(),
                        message: format!(
                            "Estimated {:.1} GB GGUF does not fit a GPU-first hybrid plan on {:.0} GB VRAM + {:.0} GB Sytra RAM. Pick a smaller quant or a smaller model.",
                            download_gb, vram_gb, ram_gb
                        ),
                        blocks_download: false,
                    });
                } else {
                    alerts.push(ModelAlert {
                        level: "warning".into(),
                        code: "needs_gpu_hybrid".into(),
                        message: format!(
                            "Estimated {:.1} GB GGUF is larger than VRAM. Sytra will plan GPU-first layer offload with mmap; generation will be slower than a fully GPU-resident model.",
                            download_gb
                        ),
                        blocks_download: false,
                    });
                }
            }
        }
    }

    if download_gb >= 40.0 {
        alerts.push(ModelAlert {
            level: "warning".into(),
            code: "huge_download".into(),
            message: format!(
                "About {:.0} GB will be pulled with Sytra's verified Xet downloader. Check free disk before starting.",
                download_gb
            ),
            blocks_download: false,
        });
    }

    if entry.use_case_tags.iter().any(|t| t == "unverified") {
        alerts.push(ModelAlert {
            level: "warning".into(),
            code: "unverified_generation".into(),
            message: "Sytra has not completed a real load+generation test for this architecture on this machine. The catalog lists it for download, not as a proven runtime.".into(),
            blocks_download: false,
        });
    }

    alerts.extend(entry.explicit_risks.iter().cloned().map(|message| ModelAlert {
        level: "warning".into(),
        code: "catalog_note".into(),
        message,
        blocks_download: false,
    }));

    alerts
}
