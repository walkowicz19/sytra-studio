use sytra_contracts::guider::ModelCatalogEntry;
use sytra_contracts::merge_config::MergeMethod;
use sytra_contracts::operation::TrainSpec;
use sytra_contracts::run_config::AdapterType;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResourceError {
    #[error("Estimated training VRAM ({estimated_mb} MB) exceeds available hardware VRAM ({available_mb} MB)")]
    VramOverflow {
        estimated_mb: u64,
        available_mb: u64,
    },
    #[error("Estimated merge RAM ({estimated_mb} MB) exceeds available hardware RAM ({available_mb} MB)")]
    RamOverflow {
        estimated_mb: u64,
        available_mb: u64,
    },
    #[error("Estimated disk space requirement ({estimated_mb} MB) exceeds available disk space ({available_mb} MB)")]
    DiskOverflow {
        estimated_mb: u64,
        available_mb: u64,
    },
    #[error("Hardware memory could not be detected; refusing to estimate or start the operation")]
    UnknownHardware,
}

pub struct ResourceGuard {
    pub total_vram_mb: u64,
    pub total_ram_mb: u64,
    pub available_disk_mb: u64,
}

impl ResourceGuard {
    pub fn new(total_vram_mb: u64, total_ram_mb: u64, available_disk_mb: u64) -> Self {
        Self {
            total_vram_mb,
            total_ram_mb,
            available_disk_mb,
        }
    }

    /// Estimates training VRAM in MB.
    pub fn estimate_train_vram(model: &ModelCatalogEntry, spec: &TrainSpec) -> u64 {
        let params = model.param_count as f64;
        let batch_size = spec.config.train.batch_size as f64;
        let seq_len = spec.config.train.max_seq_len as f64;
        let rank = spec.config.adapter.rank as f64;

        // 1. Model Weights Memory (MB)
        let bytes_per_param = match spec.config.adapter.kind {
            AdapterType::Qlora | AdapterType::Qat => 0.55, // 4-bit quant
            _ => 2.0,                                      // 16-bit
        };
        let model_weights_mb = (params * bytes_per_param) / (1024.0 * 1024.0);

        // 2. Activation Memory (MB) - rough approximation based on batch size and seq length
        let activation_mb = (batch_size * seq_len * rank * 0.15) / 1024.0;

        // 3. Optimizer State Memory (MB) - Adam uses 8 bytes per trainable parameter
        // (roughly 2 * rank * target_modules_count * hidden_size, let's approximate trainable parameters as 1.5% of total params for rank 16)
        let trainable_ratio = (rank / 16.0) * 0.015;
        let optimizer_mb = (params * trainable_ratio * 8.0) / (1024.0 * 1024.0);

        // Sum and add a fixed overhead for PyTorch/CUDA runtime context
        let total_est = model_weights_mb + activation_mb + optimizer_mb + 1024.0;
        total_est as u64
    }

    /// Estimates merge RAM and disk usage in MB.
    pub fn estimate_merge_cost(
        models: &[&ModelCatalogEntry],
        _method: MergeMethod,
        out_of_core: bool,
    ) -> (u64, u64) {
        // Compute sum of parameters of input models
        let total_input_params: u64 = models.iter().map(|m| m.param_count).sum();
        let first_model_params = models.first().map(|m| m.param_count).unwrap_or(0);

        // We assume 16-bit float dtype (2 bytes per param)
        let total_size_mb = (total_input_params * 2) / (1024 * 1024);
        let first_model_size_mb = (first_model_params * 2) / (1024 * 1024);

        // RAM requirement
        let ram_mb = if out_of_core {
            // Out of core loads one shard at a time, so it needs about 1 model size RAM + overhead
            first_model_size_mb + 1500
        } else {
            // In-memory merge loads all models
            total_size_mb + 2048
        };

        // Disk space requirement: we need space for the final merged model + temp workspace
        let final_model_size_mb = first_model_size_mb;
        let disk_mb = if out_of_core {
            // Needs temporary workspace disk space as well
            total_size_mb + final_model_size_mb + 4096
        } else {
            final_model_size_mb + 1024
        };

        (ram_mb, disk_mb)
    }

    /// Checks if a training operation can run without VRAM overflow.
    pub fn check_train(
        &self,
        model: &ModelCatalogEntry,
        spec: &TrainSpec,
    ) -> Result<(), ResourceError> {
        if self.total_vram_mb == 0 {
            return Err(ResourceError::UnknownHardware);
        }
        let est_vram = Self::estimate_train_vram(model, spec);
        if est_vram > self.total_vram_mb {
            return Err(ResourceError::VramOverflow {
                estimated_mb: est_vram,
                available_mb: self.total_vram_mb,
            });
        }
        Ok(())
    }

    /// Checks if a merge operation can run without RAM or Disk overflow.
    pub fn check_merge(
        &self,
        models: &[&ModelCatalogEntry],
        method: MergeMethod,
        out_of_core: bool,
    ) -> Result<(), ResourceError> {
        if self.total_ram_mb == 0 {
            return Err(ResourceError::UnknownHardware);
        }
        let (est_ram, est_disk) = Self::estimate_merge_cost(models, method, out_of_core);

        if est_ram > self.total_ram_mb {
            return Err(ResourceError::RamOverflow {
                estimated_mb: est_ram,
                available_mb: self.total_ram_mb,
            });
        }

        if est_disk > self.available_disk_mb {
            return Err(ResourceError::DiskOverflow {
                estimated_mb: est_disk,
                available_mb: self.available_disk_mb,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sytra_contracts::guider::Guider;

    #[test]
    fn unknown_vram_blocks_train() {
        let guard = ResourceGuard::new(0, 32768, 102400);
        let guider = Guider::new();
        let model = guider.catalog().first().expect("catalog");
        let spec = TrainSpec {
            config: sytra_contracts::run_config::RunConfig {
                version: 1,
                run_id: None,
                train_mode: sytra_contracts::run_config::TrainMode::Sft,
                model: model.model_id.clone(),
                backend: sytra_contracts::run_config::BackendConfig {
                    kind: sytra_contracts::run_config::BackendKind::Cpu,
                    judge_model: None,
                },
                data: sytra_contracts::run_config::DataSpec::Hf {
                    jsonl_path: None,
                    fingerprint: None,
                    hf: sytra_contracts::run_config::HfParams {
                        repo_id: "org/d".into(),
                        split: "train".into(),
                        revision: None,
                        config: None,
                    },
                },
                adapter: sytra_contracts::run_config::AdapterConfig {
                    kind: AdapterType::Lora,
                    rank: 16,
                    alpha: 32,
                    dropout: 0.05,
                    target_modules: vec![],
                    quant_bits: None,
                },
                optim: sytra_contracts::run_config::OptimConfig {
                    learning_rate: 2e-4,
                    schedule: sytra_contracts::run_config::Schedule::Cosine,
                    warmup_steps: 20,
                    weight_decay: 0.0,
                    grad_accumulation_steps: 8,
                },
                train: sytra_contracts::run_config::TrainParams {
                    max_steps: Some(10),
                    epochs: None,
                    batch_size: 2,
                    max_seq_len: 2048,
                    save_every: 2,
                    packing: false,
                },
                algo: Default::default(),
                output: sytra_contracts::run_config::RunOutput {
                    adapter_path: "out/adapter".into(),
                    resume_from: None,
                },
            },
            config_path: "run.yaml".into(),
        };
        assert_eq!(
            guard.check_train(model, &spec),
            Err(ResourceError::UnknownHardware)
        );
    }

    #[test]
    fn unknown_ram_blocks_merge() {
        let guard = ResourceGuard::new(8192, 0, 102400);
        let guider = Guider::new();
        let model = guider.catalog().first().expect("catalog");
        assert_eq!(
            guard.check_merge(&[model], MergeMethod::Linear, false),
            Err(ResourceError::UnknownHardware)
        );
    }
}
