//! Hard-bounded runtime wrapper for exact standard-GQA MoE executors.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    Accelerator, CudaAccelerator, DenseTensorStore, ExpertStore, KimiOracleOutputs,
    KimiRuntimeMetrics, KimiRuntimePlacement, KimiSpeculativeOutput, KimiStepMetrics, KvTier,
    MemoryEnvelope, MixtralConfig, MixtralError, MixtralExecutor, NoAccelerator, ResidencyManager,
    RuntimeError, RuntimeManifest, StandardMoeKvState, StreamingScheduler, WeightedMirror,
};

#[derive(Debug, Clone, Default)]
pub struct MixtralRuntimeOptions {
    pub cuda_device: Option<i32>,
    pub dense_tile_bytes: u64,
    pub ram_dense_cache_bytes: Option<u64>,
    pub ram_expert_cache_bytes: Option<u64>,
    pub accelerator_expert_cache_bytes: Option<u64>,
}

pub struct MixtralRuntime {
    config: MixtralConfig,
    manifest: RuntimeManifest,
    memory: MemoryEnvelope,
    placement: KimiRuntimePlacement,
    dense: DenseTensorStore,
    expert_store: Arc<ExpertStore>,
    residency: Arc<ResidencyManager>,
    scheduler: StreamingScheduler,
    cuda: Option<Arc<CudaAccelerator>>,
    dense_tile_bytes: u64,
    execution: Mutex<()>,
}

impl MixtralRuntime {
    pub fn new(
        root: impl AsRef<Path>,
        manifest: RuntimeManifest,
        memory: MemoryEnvelope,
        mirrors: Vec<WeightedMirror>,
        options: MixtralRuntimeOptions,
    ) -> Result<Self, RuntimeError> {
        if !memory.feasible
            || memory.effective_context_tokens == 0
            || memory.max_verification_positions == 0
        {
            return Err(RuntimeError::Memory(memory.notes.join("; ")));
        }
        let root = root.as_ref();
        let config = MixtralConfig::load(root).map_err(contract)?;
        config.validate_manifest(&manifest).map_err(contract)?;
        let dense_tile_bytes = if options.dense_tile_bytes == 0 {
            memory.host_staging_bytes
        } else {
            options.dense_tile_bytes
        };
        let auto_dense = if manifest.dense_bytes <= memory.ram_cache_bytes {
            manifest.dense_bytes
        } else {
            0
        };
        let requested_dense = options.ram_dense_cache_bytes.unwrap_or(auto_dense);
        let requested_expert = options
            .ram_expert_cache_bytes
            .unwrap_or_else(|| memory.ram_cache_bytes.saturating_sub(requested_dense));
        let (ram_dense_cache, ram_expert_cache) =
            split_budget(memory.ram_cache_bytes, requested_dense, requested_expert);
        let cuda_limit = memory
            .accelerator_limit_bytes
            .saturating_sub(memory.accelerator_runtime_reserve_bytes);
        let cuda = match options.cuda_device {
            Some(device) => Some(Arc::new(
                CudaAccelerator::new_with_budget(device, cuda_limit)
                    .map_err(RuntimeError::Execution)?,
            )),
            None => None,
        };
        if cuda.is_none() && memory.accelerator_limit_bytes > 0 {
            return Err(RuntimeError::Memory(
                "the plan reserves accelerator memory but no CUDA device was selected".into(),
            ));
        }
        let expert_store = Arc::new(ExpertStore::new(
            root,
            mirrors.clone(),
            manifest.storage.experts.iter().cloned(),
        ));
        let max_expert = manifest
            .storage
            .experts
            .iter()
            .map(|expert| expert.byte_len())
            .max()
            .unwrap_or(0);
        if max_expert == 0 || max_expert > memory.host_staging_bytes {
            return Err(RuntimeError::Memory(format!(
                "largest expert payload {max_expert} exceeds host staging {}",
                memory.host_staging_bytes
            )));
        }
        let accelerator_expert_cache = if matches!(
            manifest.architecture.expert_format,
            crate::WeightFormat::Bf16 | crate::WeightFormat::PackedInt4Group32
        ) {
            options
                .accelerator_expert_cache_bytes
                .unwrap_or(memory.accelerator_cache_bytes)
                .min(memory.accelerator_cache_bytes)
        } else {
            0
        };
        let accelerator: Arc<dyn Accelerator> = match &cuda {
            Some(cuda) => cuda.clone(),
            None => Arc::new(NoAccelerator),
        };
        let residency = Arc::new(ResidencyManager::with_budgets(
            expert_store.clone(),
            accelerator,
            ram_expert_cache,
            ram_expert_cache.saturating_add(memory.host_staging_bytes),
            accelerator_expert_cache,
            cuda_limit,
        ));
        let scheduler = StreamingScheduler::with_inflight_budget(residency.clone(), 0, max_expert)
            .map_err(execution)?;
        let dense = DenseTensorStore::with_budgets(
            root,
            mirrors,
            manifest.storage.dense_tensors.iter().cloned(),
            ram_dense_cache,
            ram_dense_cache.saturating_add(memory.host_staging_bytes),
        );
        let runtime = Self {
            config,
            manifest,
            memory,
            placement: KimiRuntimePlacement {
                ram_dense_cache_bytes: ram_dense_cache,
                ram_expert_cache_bytes: ram_expert_cache,
                accelerator_expert_cache_bytes: accelerator_expert_cache,
                max_inflight_expert_bytes: max_expert,
                cuda_allocation_limit_bytes: cuda_limit,
            },
            dense,
            expert_store,
            residency,
            scheduler,
            cuda,
            dense_tile_bytes,
            execution: Mutex::new(()),
        };
        runtime.with_executor(|_| Ok(()))?;
        Ok(runtime)
    }

    pub fn config(&self) -> &MixtralConfig {
        &self.config
    }
    pub fn manifest(&self) -> &RuntimeManifest {
        &self.manifest
    }
    pub fn memory(&self) -> &MemoryEnvelope {
        &self.memory
    }
    pub fn placement(&self) -> &KimiRuntimePlacement {
        &self.placement
    }
    pub fn new_state(&self) -> StandardMoeKvState {
        if self.memory.kv_tier == KvTier::Accelerator {
            if let Some(cuda) = &self.cuda {
                return StandardMoeKvState::new_device(
                    self.config.num_hidden_layers,
                    cuda.clone(),
                    usize::try_from(self.memory.effective_context_tokens).unwrap_or(usize::MAX),
                    self.config.num_key_value_heads,
                    self.config.head_dimension(),
                    self.config.head_dimension(),
                );
            }
        }
        StandardMoeKvState::new(self.config.num_hidden_layers)
    }

    pub fn prefill_next(
        &self,
        tokens: &[u32],
        state: &mut StandardMoeKvState,
    ) -> Result<(u32, KimiStepMetrics), RuntimeError> {
        self.check_capacity(state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, state)?;
            let last = &hidden[hidden.len() - self.config.hidden_size..];
            let (tokens, head) = executor.greedy_tokens(last, 1)?;
            metrics.merge(head);
            Ok((tokens[0], metrics))
        })
    }

    pub fn logits_last(
        &self,
        tokens: &[u32],
        state: &mut StandardMoeKvState,
    ) -> Result<(Vec<f32>, KimiStepMetrics), RuntimeError> {
        self.check_capacity(state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, state)?;
            let last = &hidden[hidden.len() - self.config.hidden_size..];
            let (logits, head) = executor.logits(last, 1)?;
            metrics.merge(head);
            Ok((logits, metrics))
        })
    }

    pub fn verify_greedy_draft(
        &self,
        current: u32,
        draft: &[u32],
        state: &mut StandardMoeKvState,
    ) -> Result<KimiSpeculativeOutput, RuntimeError> {
        self.check_capacity(state, draft.len().saturating_add(1))?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| executor.verify_greedy_draft(current, draft, state))
    }

    pub fn oracle_outputs(&self, tokens: &[u32]) -> Result<KimiOracleOutputs, RuntimeError> {
        let mut state = self.new_state();
        self.check_capacity(&state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, &mut state)?;
            let (predictions, head) = executor.greedy_tokens(&hidden, tokens.len())?;
            metrics.merge(head);
            let last = &hidden[hidden.len() - self.config.hidden_size..];
            let (logits, head) = executor.logits(last, 1)?;
            metrics.merge(head);
            Ok(KimiOracleOutputs {
                teacher_forced_predictions: predictions,
                final_logits: logits,
                metrics,
            })
        })
    }

    pub fn metrics(&self) -> Result<KimiRuntimeMetrics, RuntimeError> {
        Ok(KimiRuntimeMetrics {
            dense: self.dense.metrics().map_err(execution)?,
            experts: self.residency.metrics().map_err(execution)?,
            cuda: self.cuda.as_ref().map(|cuda| cuda.memory_metrics()),
        })
    }

    pub fn storage_metrics(&self) -> crate::StoreMetrics {
        self.expert_store.metrics()
    }

    fn check_capacity(
        &self,
        state: &StandardMoeKvState,
        additional: usize,
    ) -> Result<(), RuntimeError> {
        let maximum = usize::try_from(self.memory.effective_context_tokens)
            .unwrap_or(usize::MAX)
            .min(self.config.max_position_embeddings);
        if additional == 0 || state.position().saturating_add(additional) > maximum {
            return Err(RuntimeError::Memory(format!(
                "request would exceed bounded context {maximum}"
            )));
        }
        Ok(())
    }

    fn with_executor<T>(
        &self,
        execute: impl FnOnce(&MixtralExecutor<'_>) -> Result<T, MixtralError>,
    ) -> Result<T, RuntimeError> {
        let executor = MixtralExecutor::new(
            &self.config,
            &self.dense,
            &self.scheduler,
            self.cuda.as_deref(),
            self.dense_tile_bytes,
        )
        .map_err(execution)?;
        execute(&executor).map_err(execution)
    }
}

fn split_budget(total: u64, dense: u64, expert: u64) -> (u64, u64) {
    let requested = dense.saturating_add(expert);
    if requested <= total {
        return (dense, expert);
    }
    if requested == 0 {
        return (0, 0);
    }
    let dense = ((u128::from(total) * u128::from(dense)) / u128::from(requested)) as u64;
    (dense, total.saturating_sub(dense))
}

fn contract(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Contract(error.to_string())
}

fn execution(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Execution(error.to_string())
}
