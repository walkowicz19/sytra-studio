//! Bounded construction and execution of correctness-gated model runtimes.
//!
//! This module is the bridge between a memory plan and architecture code. It
//! deliberately serializes target-model execution: dense and expert staging
//! share one planned working region, so concurrent generations must not each
//! assume they own that region.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use serde::Serialize;
use thiserror::Error;

use crate::{
    Accelerator, CacheMetrics, CudaAccelerator, CudaMemoryMetrics, DenseStoreMetrics,
    DenseTensorStore, ExpertStore, KimiDecodeState, KimiError, KimiExecutionBackend, KimiK27Config,
    KimiOneTokenExecutor, KimiSpeculativeOutput, KimiStepMetrics, MemoryEnvelope, NoAccelerator,
    ResidencyManager, RuntimeManifest, StreamingScheduler, WeightedMirror,
};

#[derive(Debug, Clone, Default)]
pub struct KimiRuntimeOptions {
    pub cuda_device: Option<i32>,
    pub dense_tile_bytes: u64,
    pub ram_dense_cache_bytes: Option<u64>,
    pub ram_expert_cache_bytes: Option<u64>,
    pub accelerator_expert_cache_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KimiRuntimePlacement {
    pub ram_dense_cache_bytes: u64,
    pub ram_expert_cache_bytes: u64,
    pub accelerator_expert_cache_bytes: u64,
    pub max_inflight_expert_bytes: u64,
    pub cuda_allocation_limit_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KimiRuntimeMetrics {
    pub dense: DenseStoreMetrics,
    pub experts: CacheMetrics,
    pub cuda: Option<CudaMemoryMetrics>,
}

#[derive(Debug, Clone)]
pub struct KimiOracleOutputs {
    pub teacher_forced_predictions: Vec<u32>,
    pub final_logits: Vec<f32>,
    pub metrics: KimiStepMetrics,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime contract is invalid: {0}")]
    Contract(String),
    #[error("runtime memory envelope is infeasible: {0}")]
    Memory(String),
    #[error("runtime execution failed: {0}")]
    Execution(String),
    #[error("runtime execution lock is poisoned")]
    Poisoned,
}

pub struct KimiRuntime {
    config: KimiK27Config,
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

impl KimiRuntime {
    pub fn new(
        model_root: impl AsRef<Path>,
        manifest: RuntimeManifest,
        memory: MemoryEnvelope,
        mirrors: Vec<WeightedMirror>,
        options: KimiRuntimeOptions,
    ) -> Result<Self, RuntimeError> {
        if !memory.feasible {
            return Err(RuntimeError::Memory(memory.notes.join("; ")));
        }
        if memory.effective_context_tokens == 0 || memory.max_verification_positions == 0 {
            return Err(RuntimeError::Memory(
                "no bounded context or verification position remains".into(),
            ));
        }
        let root = model_root.as_ref();
        let config = KimiK27Config::load(root)
            .and_then(|config| {
                config.validate_runtime_manifest(&manifest)?;
                Ok(config)
            })
            .map_err(|error| RuntimeError::Contract(error.to_string()))?;
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
        let accelerator_expert_cache = options
            .accelerator_expert_cache_bytes
            .unwrap_or(memory.accelerator_cache_bytes)
            .min(memory.accelerator_cache_bytes);
        let accelerator_allocation_limit = memory
            .accelerator_limit_bytes
            .saturating_sub(memory.accelerator_runtime_reserve_bytes);

        let cuda = match options.cuda_device {
            Some(device) => Some(Arc::new(
                CudaAccelerator::new_with_budget(device, accelerator_allocation_limit)
                    .map_err(RuntimeError::Execution)?,
            )),
            None => None,
        };
        if cuda.is_none() && memory.accelerator_limit_bytes > 0 {
            return Err(RuntimeError::Memory(
                "the plan reserves accelerator memory but no CUDA device was selected".into(),
            ));
        }
        let accelerator: Arc<dyn Accelerator> = match &cuda {
            Some(cuda) => cuda.clone(),
            None => Arc::new(NoAccelerator),
        };

        let expert_store = Arc::new(ExpertStore::new(
            root,
            mirrors.clone(),
            manifest.storage.experts.iter().cloned(),
        ));
        let max_expert_payload = manifest
            .storage
            .experts
            .iter()
            .map(|expert| expert.byte_len())
            .max()
            .unwrap_or(0);
        if max_expert_payload == 0 || max_expert_payload > memory.host_staging_bytes {
            return Err(RuntimeError::Memory(format!(
                "largest expert payload {max_expert_payload} exceeds host staging {}",
                memory.host_staging_bytes
            )));
        }
        let residency = Arc::new(ResidencyManager::with_budgets(
            expert_store.clone(),
            accelerator,
            ram_expert_cache,
            ram_expert_cache.saturating_add(memory.host_staging_bytes),
            accelerator_expert_cache,
            accelerator_allocation_limit,
        ));
        // One expert per wave is conservative but makes the activation +
        // payload staging proof exact. Cross-layer prefetch stays disabled
        // until it participates in the same global host allocator.
        let scheduler =
            StreamingScheduler::with_inflight_budget(residency.clone(), 0, max_expert_payload)
                .map_err(|error| RuntimeError::Execution(error.to_string()))?;
        let dense = DenseTensorStore::with_budgets(
            root,
            mirrors,
            manifest.storage.dense_tensors.iter().cloned(),
            ram_dense_cache,
            ram_dense_cache.saturating_add(memory.host_staging_bytes),
        );
        let placement = KimiRuntimePlacement {
            ram_dense_cache_bytes: ram_dense_cache,
            ram_expert_cache_bytes: ram_expert_cache,
            accelerator_expert_cache_bytes: accelerator_expert_cache,
            max_inflight_expert_bytes: max_expert_payload,
            cuda_allocation_limit_bytes: accelerator_allocation_limit,
        };
        let runtime = Self {
            config,
            manifest,
            memory,
            placement,
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

    pub fn config(&self) -> &KimiK27Config {
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

    pub fn new_state(&self) -> KimiDecodeState {
        KimiDecodeState::new(&self.config)
    }

    pub fn prefill_next(
        &self,
        tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<(u32, KimiStepMetrics), RuntimeError> {
        self.check_capacity(state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, state)?;
            let last = hidden
                .chunks_exact(self.config.hidden_size)
                .last()
                .ok_or_else(|| KimiError::Shape("prefill returned no hidden state".into()))?;
            let (token, head) = executor.greedy_token(last)?;
            metrics.merge(head);
            Ok((token, metrics))
        })
    }

    pub fn verify_greedy_draft(
        &self,
        current_token: u32,
        draft_tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<KimiSpeculativeOutput, RuntimeError> {
        self.check_capacity(state, draft_tokens.len().saturating_add(1))?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            executor.verify_greedy_draft(current_token, draft_tokens, state)
        })
    }

    pub fn teacher_forced_predictions(
        &self,
        tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<(Vec<u32>, KimiStepMetrics), RuntimeError> {
        self.check_capacity(state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, state)?;
            let (predictions, head) = executor.greedy_tokens(&hidden, tokens.len())?;
            metrics.merge(head);
            Ok((predictions, metrics))
        })
    }

    pub fn logits_last(
        &self,
        tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<(Vec<f32>, KimiStepMetrics), RuntimeError> {
        self.check_capacity(state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, state)?;
            let last = hidden
                .chunks_exact(self.config.hidden_size)
                .last()
                .ok_or_else(|| KimiError::Shape("prefill returned no hidden state".into()))?;
            let (logits, head) = executor.logits(last)?;
            metrics.merge(head);
            Ok((logits, metrics))
        })
    }

    pub fn oracle_outputs(&self, tokens: &[u32]) -> Result<KimiOracleOutputs, RuntimeError> {
        let mut state = self.new_state();
        self.check_capacity(&state, tokens.len())?;
        let _guard = self.execution.lock().map_err(|_| RuntimeError::Poisoned)?;
        self.with_executor(|executor| {
            let (hidden, mut metrics) = executor.forward_tokens(tokens, &mut state)?;
            let (teacher_forced_predictions, head) =
                executor.greedy_tokens(&hidden, tokens.len())?;
            metrics.merge(head);
            let last = hidden
                .chunks_exact(self.config.hidden_size)
                .last()
                .ok_or_else(|| {
                    KimiError::Shape("oracle prefill returned no hidden state".into())
                })?;
            let (final_logits, head) = executor.logits(last)?;
            metrics.merge(head);
            Ok(KimiOracleOutputs {
                teacher_forced_predictions,
                final_logits,
                metrics,
            })
        })
    }

    pub fn metrics(&self) -> Result<KimiRuntimeMetrics, RuntimeError> {
        Ok(KimiRuntimeMetrics {
            dense: self
                .dense
                .metrics()
                .map_err(|error| RuntimeError::Execution(error.to_string()))?,
            experts: self
                .residency
                .metrics()
                .map_err(|error| RuntimeError::Execution(error.to_string()))?,
            cuda: self.cuda.as_ref().map(|cuda| cuda.memory_metrics()),
        })
    }

    pub fn storage_metrics(&self) -> crate::StoreMetrics {
        self.expert_store.metrics()
    }

    fn check_capacity(
        &self,
        state: &KimiDecodeState,
        additional: usize,
    ) -> Result<(), RuntimeError> {
        if additional == 0 {
            return Err(RuntimeError::Execution(
                "token batch cannot be empty".into(),
            ));
        }
        let maximum = usize::try_from(self.memory.effective_context_tokens)
            .unwrap_or(usize::MAX)
            .min(self.config.max_position_embeddings);
        if state.position().saturating_add(additional) > maximum {
            return Err(RuntimeError::Memory(format!(
                "request would exceed bounded context {maximum}"
            )));
        }
        Ok(())
    }

    fn with_executor<T>(
        &self,
        execute: impl FnOnce(&KimiOneTokenExecutor<'_>) -> Result<T, KimiError>,
    ) -> Result<T, RuntimeError> {
        let backend = match &self.cuda {
            Some(cuda) => KimiExecutionBackend::Cuda(cuda),
            None => KimiExecutionBackend::Cpu,
        };
        let executor = KimiOneTokenExecutor::new(
            &self.config,
            &self.dense,
            &self.scheduler,
            backend,
            self.dense_tile_bytes,
        )
        .map_err(|error| RuntimeError::Execution(error.to_string()))?;
        execute(&executor).map_err(|error| RuntimeError::Execution(error.to_string()))
    }
}

fn split_budget(total: u64, requested_dense: u64, requested_expert: u64) -> (u64, u64) {
    let requested = requested_dense.saturating_add(requested_expert);
    if requested <= total {
        return (requested_dense, requested_expert);
    }
    if requested == 0 {
        return (0, 0);
    }
    let dense = ((u128::from(total) * u128::from(requested_dense)) / u128::from(requested)) as u64;
    (dense, total.saturating_sub(dense))
}
