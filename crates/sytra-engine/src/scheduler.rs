use std::{
    collections::BTreeSet,
    sync::{
        mpsc::{self, SyncSender},
        Arc,
    },
    thread::{self, JoinHandle},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    cache::{CacheError, ResidencyManager, ResidentExpert},
    store::ExpertKey,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Route {
    pub expert: u32,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoutingBatch {
    /// One route vector per token/position. Order and weights are preserved.
    pub positions: Vec<Vec<Route>>,
}

impl RoutingBatch {
    pub fn expert_union(&self, layer: u32) -> Vec<ExpertKey> {
        self.positions
            .iter()
            .flat_map(|routes| routes.iter())
            .map(|route| ExpertKey::new(layer, route.expert))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[derive(Debug)]
pub struct PreparedLayer {
    pub layer: u32,
    pub routes: RoutingBatch,
    pub wave: usize,
    pub wave_count: usize,
    /// Unique experts loaded once for all positions in the batch.
    pub experts: Vec<ResidentExpert>,
}

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error(transparent)]
    Cache(#[from] CacheError),
    #[error("routing batch contains no positions")]
    EmptyBatch,
    #[error("expert {key:?} needs {required} in-flight bytes but the budget is {budget}")]
    InflightBudgetTooSmall {
        key: ExpertKey,
        required: u64,
        budget: u64,
    },
    #[error("prepared-wave executor failed: {0}")]
    Executor(String),
}

/// Shared route -> union -> place -> overlap path.
///
/// Architecture code computes routes and executes kernels. This scheduler
/// preserves the routes exactly and only prepares the unique expert payloads.
pub struct StreamingScheduler {
    residency: Arc<ResidencyManager>,
    prefetch_tx: Option<SyncSender<Vec<ExpertKey>>>,
    prefetch_thread: Option<JoinHandle<()>>,
    max_inflight_bytes: u64,
}

impl StreamingScheduler {
    pub fn new(residency: Arc<ResidencyManager>, prefetch_depth: usize) -> Self {
        if prefetch_depth == 0 {
            return Self {
                residency,
                prefetch_tx: None,
                prefetch_thread: None,
                max_inflight_bytes: u64::MAX,
            };
        }
        let (tx, rx) = mpsc::sync_channel::<Vec<ExpertKey>>(prefetch_depth);
        let worker_residency = residency.clone();
        let worker = thread::Builder::new()
            .name("sytra-expert-prefetch".into())
            .spawn(move || {
                while let Ok(keys) = rx.recv() {
                    for key in keys {
                        let _ = worker_residency.prefetch_host(key);
                    }
                }
            })
            .expect("failed to start Sytra expert prefetch worker");
        Self {
            residency,
            prefetch_tx: Some(tx),
            prefetch_thread: Some(worker),
            max_inflight_bytes: u64::MAX,
        }
    }

    pub fn with_inflight_budget(
        residency: Arc<ResidencyManager>,
        prefetch_depth: usize,
        max_inflight_bytes: u64,
    ) -> Result<Self, SchedulerError> {
        if max_inflight_bytes == 0 {
            return Err(SchedulerError::InflightBudgetTooSmall {
                key: ExpertKey::new(0, 0),
                required: 1,
                budget: 0,
            });
        }
        let mut scheduler = Self::new(residency, prefetch_depth);
        scheduler.max_inflight_bytes = max_inflight_bytes;
        Ok(scheduler)
    }

    pub fn prepare_layer(
        &self,
        layer: u32,
        routes: RoutingBatch,
    ) -> Result<PreparedLayer, SchedulerError> {
        if routes.positions.is_empty() {
            return Err(SchedulerError::EmptyBatch);
        }
        let union = routes.expert_union(layer);
        if union.is_empty() {
            return Err(SchedulerError::EmptyBatch);
        }
        let required = union.iter().try_fold(0_u64, |total, key| {
            self.residency
                .expert_byte_len(*key)
                .map(|bytes| total.saturating_add(bytes))
        })?;
        if required > self.max_inflight_bytes {
            return Err(SchedulerError::InflightBudgetTooSmall {
                key: union[0],
                required,
                budget: self.max_inflight_bytes,
            });
        }
        let experts = union
            .into_iter()
            .map(|key| self.residency.get(key))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PreparedLayer {
            layer,
            routes,
            wave: 0,
            wave_count: 1,
            experts,
        })
    }

    /// Execute a routed layer in byte-bounded waves. Expert leases from one
    /// wave are dropped before the next wave is loaded, so active cache leases
    /// cannot turn a large batch union into unbounded RAM/VRAM usage.
    pub fn for_each_prepared_wave<T, F>(
        &self,
        layer: u32,
        routes: RoutingBatch,
        mut execute: F,
    ) -> Result<Vec<T>, SchedulerError>
    where
        F: FnMut(PreparedLayer) -> Result<T, String>,
    {
        let (outputs, _) =
            self.for_each_prepared_wave_fold(layer, routes, Vec::new(), |outputs, prepared| {
                outputs.push(execute(prepared)?);
                Ok(())
            })?;
        Ok(outputs)
    }

    /// Execute bounded waves while folding each result immediately into one
    /// caller-owned accumulator. This prevents output memory from growing
    /// with the number of expert waves.
    pub fn for_each_prepared_wave_fold<S, F>(
        &self,
        layer: u32,
        routes: RoutingBatch,
        mut state: S,
        mut execute: F,
    ) -> Result<(S, usize), SchedulerError>
    where
        F: FnMut(&mut S, PreparedLayer) -> Result<(), String>,
    {
        if routes.positions.is_empty() {
            return Err(SchedulerError::EmptyBatch);
        }
        let mut waves: Vec<Vec<ExpertKey>> = Vec::new();
        let mut current = Vec::new();
        let mut current_bytes = 0_u64;
        let union = routes.expert_union(layer);
        if union.is_empty() {
            return Err(SchedulerError::EmptyBatch);
        }
        for key in union {
            let bytes = self.residency.expert_byte_len(key)?;
            if bytes > self.max_inflight_bytes {
                return Err(SchedulerError::InflightBudgetTooSmall {
                    key,
                    required: bytes,
                    budget: self.max_inflight_bytes,
                });
            }
            if !current.is_empty() && current_bytes.saturating_add(bytes) > self.max_inflight_bytes
            {
                waves.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
            current.push(key);
            current_bytes += bytes;
        }
        if !current.is_empty() {
            waves.push(current);
        }

        let wave_count = waves.len();
        for (wave, keys) in waves.into_iter().enumerate() {
            let selected: BTreeSet<_> = keys.iter().map(|key| key.expert).collect();
            let wave_routes = RoutingBatch {
                positions: routes
                    .positions
                    .iter()
                    .map(|position| {
                        position
                            .iter()
                            .filter(|route| selected.contains(&route.expert))
                            .copied()
                            .collect()
                    })
                    .collect(),
            };
            let experts = keys
                .into_iter()
                .map(|key| self.residency.get(key))
                .collect::<Result<Vec<_>, _>>()?;
            execute(
                &mut state,
                PreparedLayer {
                    layer,
                    routes: wave_routes,
                    wave,
                    wave_count,
                    experts,
                },
            )
            .map_err(SchedulerError::Executor)?;
        }
        Ok((state, wave_count))
    }

    /// Architecture lookahead supplies predicted routes for a future layer.
    /// A full queue is skipped instead of blocking the decode thread.
    pub fn prefetch(&self, layer: u32, predicted: &RoutingBatch) -> bool {
        let Some(sender) = &self.prefetch_tx else {
            return false;
        };
        sender.try_send(predicted.expert_union(layer)).is_ok()
    }
}

impl Drop for StreamingScheduler {
    fn drop(&mut self) {
        self.prefetch_tx.take();
        if let Some(worker) = self.prefetch_thread.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::{
        manifest::{ExpertLocation, TensorSegment},
        ExpertStore, NoAccelerator,
    };

    fn scheduler_fixture() -> (PathBuf, Arc<ResidencyManager>) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-waves-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("experts.bin"), b"AAAABBBBCCCC").unwrap();
        let locations = (0..3).map(|expert| ExpertLocation {
            layer: 0,
            expert,
            segments: vec![TensorSegment {
                tensor: format!("experts.{expert}.gate_proj"),
                dtype: None,
                shape: vec![],
                shard: "experts.bin".into(),
                offset: u64::from(expert) * 4,
                length: 4,
            }],
        });
        let store = Arc::new(ExpertStore::new(&root, vec![], locations));
        let residency = Arc::new(ResidencyManager::new(store, Arc::new(NoAccelerator), 4, 0));
        (root, residency)
    }

    #[test]
    fn batch_union_loads_each_expert_once_without_changing_routes() {
        let routes = RoutingBatch {
            positions: vec![
                vec![
                    Route {
                        expert: 3,
                        weight: 0.7,
                    },
                    Route {
                        expert: 1,
                        weight: 0.3,
                    },
                ],
                vec![
                    Route {
                        expert: 3,
                        weight: 0.6,
                    },
                    Route {
                        expert: 2,
                        weight: 0.4,
                    },
                ],
            ],
        };
        let original = routes.clone();
        assert_eq!(
            routes.expert_union(4),
            vec![
                ExpertKey::new(4, 1),
                ExpertKey::new(4, 2),
                ExpertKey::new(4, 3)
            ]
        );
        assert_eq!(routes, original);
    }

    #[test]
    fn routed_union_executes_in_hard_bounded_waves() {
        let (root, residency) = scheduler_fixture();
        let scheduler = StreamingScheduler::with_inflight_budget(residency, 0, 4).unwrap();
        let routes = RoutingBatch {
            positions: vec![vec![
                Route {
                    expert: 2,
                    weight: 0.2,
                },
                Route {
                    expert: 0,
                    weight: 0.5,
                },
                Route {
                    expert: 1,
                    weight: 0.3,
                },
            ]],
        };
        let outputs = scheduler
            .for_each_prepared_wave(0, routes.clone(), |prepared| {
                assert_eq!(prepared.experts.len(), 1);
                assert_eq!(prepared.routes.positions[0].len(), 1);
                assert_eq!(prepared.wave_count, 3);
                Ok(prepared.experts[0].key.expert)
            })
            .unwrap();
        assert_eq!(outputs, [0, 1, 2]);
        let (sum, wave_count) = scheduler
            .for_each_prepared_wave_fold(0, routes, 0_u32, |sum, prepared| {
                *sum += prepared.experts[0].key.expert;
                Ok(())
            })
            .unwrap();
        assert_eq!(sum, 3);
        assert_eq!(wave_count, 3);
        drop(scheduler);
        fs::remove_dir_all(root).unwrap();
    }
}
