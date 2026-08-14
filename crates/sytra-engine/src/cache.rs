use std::{
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
};

use serde::Serialize;
use thiserror::Error;

use crate::store::{ExpertKey, ExpertPayload, ExpertStore, ResidentTensor, StoreError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorBuffer {
    pub id: u64,
    pub bytes: usize,
}

/// Architecture-independent accelerator memory operations.
///
/// CUDA/ROCm/Metal implementations own allocation and copies. Architecture
/// kernels consume the opaque buffer id; the residency manager never changes
/// or interprets weight bytes.
pub trait Accelerator: Send + Sync {
    fn name(&self) -> &str;
    fn upload(&self, key: ExpertKey, bytes: &[u8]) -> Result<AcceleratorBuffer, String>;
    fn release(&self, buffer: &AcceleratorBuffer);
}

#[derive(Debug, Default)]
pub struct NoAccelerator;

impl Accelerator for NoAccelerator {
    fn name(&self) -> &str {
        "none"
    }

    fn upload(&self, _key: ExpertKey, _bytes: &[u8]) -> Result<AcceleratorBuffer, String> {
        Err("no accelerator backend is active".into())
    }

    fn release(&self, _buffer: &AcceleratorBuffer) {}
}

struct DeviceAllocation {
    buffer: AcceleratorBuffer,
    backend: Arc<dyn Accelerator>,
    live_bytes: Arc<AtomicU64>,
    reserved_bytes: u64,
}

impl fmt::Debug for DeviceAllocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceAllocation")
            .field("buffer", &self.buffer)
            .field("backend", &self.backend.name())
            .finish()
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        self.backend.release(&self.buffer);
        self.live_bytes
            .fetch_sub(self.reserved_bytes, Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidencyTier {
    Accelerator,
    Ram,
    Storage,
}

/// A lease keeps an accelerator allocation alive even if its cache entry is
/// evicted while a kernel is using it.
#[derive(Debug, Clone)]
pub struct ResidentExpert {
    pub key: ExpertKey,
    pub host_bytes: Option<Arc<[u8]>>,
    /// Tensor-name to byte-range mapping within `host_bytes` or the combined
    /// accelerator allocation. This survives every residency tier.
    pub tensors: Arc<[ResidentTensor]>,
    device: Option<Arc<DeviceAllocation>>,
    pub source_tier: ResidencyTier,
}

impl ResidentExpert {
    pub fn accelerator_buffer(&self) -> Option<&AcceleratorBuffer> {
        self.device.as_ref().map(|allocation| &allocation.buffer)
    }

    pub fn tensor_bytes(&self, suffix: &str) -> Option<&[u8]> {
        let bytes = self.host_bytes.as_ref()?;
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name.ends_with(suffix))?;
        bytes.get(tensor.offset..tensor.offset + tensor.length)
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct CacheMetrics {
    pub accelerator_hits: u64,
    pub ram_hits: u64,
    pub storage_loads: u64,
    pub accelerator_evictions: u64,
    pub ram_evictions: u64,
    pub accelerator_bytes: u64,
    /// Includes allocations held by active leases after cache eviction.
    pub accelerator_live_bytes: u64,
    pub accelerator_peak_bytes: u64,
    pub ram_bytes: u64,
    /// Includes host allocations held by active leases after cache eviction.
    pub ram_live_bytes: u64,
    pub ram_peak_bytes: u64,
    pub ram_budget_denials: u64,
    pub accelerator_budget_denials: u64,
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error(transparent)]
    Storage(#[from] StoreError),
    #[error("expert cache state is poisoned")]
    Poisoned,
    #[error("expert {key:?} needs {required} live host bytes but the hard limit is {budget}")]
    HostBudget {
        key: ExpertKey,
        required: u64,
        budget: u64,
    },
}

#[derive(Debug)]
struct HostEntry {
    payload: ExpertPayload,
    last_used: u64,
    heat: u64,
}

#[derive(Debug)]
struct DeviceEntry {
    allocation: Arc<DeviceAllocation>,
    tensors: Arc<[ResidentTensor]>,
    last_used: u64,
    heat: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    clock: u64,
    heat: HashMap<ExpertKey, u64>,
    host: HashMap<ExpertKey, HostEntry>,
    device: HashMap<ExpertKey, DeviceEntry>,
    retired_host: Vec<(Weak<[u8]>, u64)>,
    reserved_host_bytes: u64,
    metrics: CacheMetrics,
}

/// Thread-safe RAM/accelerator expert residency over immutable storage.
pub struct ResidencyManager {
    store: Arc<ExpertStore>,
    accelerator: Arc<dyn Accelerator>,
    ram_cache_budget: u64,
    ram_live_budget: u64,
    accelerator_cache_budget: u64,
    accelerator_live_budget: u64,
    accelerator_live_bytes: Arc<AtomicU64>,
    accelerator_peak_bytes: AtomicU64,
    state: Mutex<CacheState>,
}

impl fmt::Debug for ResidencyManager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidencyManager")
            .field("accelerator", &self.accelerator.name())
            .field("ram_cache_budget", &self.ram_cache_budget)
            .field("ram_live_budget", &self.ram_live_budget)
            .field("accelerator_cache_budget", &self.accelerator_cache_budget)
            .field("accelerator_live_budget", &self.accelerator_live_budget)
            .finish_non_exhaustive()
    }
}

impl ResidencyManager {
    pub fn new(
        store: Arc<ExpertStore>,
        accelerator: Arc<dyn Accelerator>,
        ram_budget: u64,
        accelerator_budget: u64,
    ) -> Self {
        Self::with_budgets(
            store,
            accelerator,
            ram_budget,
            ram_budget,
            accelerator_budget,
            accelerator_budget,
        )
    }

    pub fn with_budgets(
        store: Arc<ExpertStore>,
        accelerator: Arc<dyn Accelerator>,
        ram_cache_budget: u64,
        ram_live_budget: u64,
        accelerator_cache_budget: u64,
        accelerator_live_budget: u64,
    ) -> Self {
        Self {
            store,
            accelerator,
            ram_cache_budget: ram_cache_budget.min(ram_live_budget),
            ram_live_budget,
            accelerator_cache_budget: accelerator_cache_budget.min(accelerator_live_budget),
            accelerator_live_budget,
            accelerator_live_bytes: Arc::new(AtomicU64::new(0)),
            accelerator_peak_bytes: AtomicU64::new(0),
            state: Mutex::new(CacheState::default()),
        }
    }

    pub fn get(&self, key: ExpertKey) -> Result<ResidentExpert, CacheError> {
        let host_hit = {
            let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
            state.clock += 1;
            let now = state.clock;
            let heat = {
                let value = state.heat.entry(key).or_default();
                *value += 1;
                *value
            };
            if let Some(entry) = state.device.get_mut(&key) {
                entry.last_used = now;
                entry.heat = heat;
                let allocation = entry.allocation.clone();
                let tensors = entry.tensors.clone();
                state.metrics.accelerator_hits += 1;
                return Ok(ResidentExpert {
                    key,
                    host_bytes: None,
                    tensors,
                    device: Some(allocation),
                    source_tier: ResidencyTier::Accelerator,
                });
            }
            let host = state.host.get_mut(&key).map(|entry| {
                entry.last_used = now;
                entry.heat = heat;
                entry.payload.clone()
            });
            if host.is_some() {
                state.metrics.ram_hits += 1;
            }
            host
        };

        let (payload, source_tier) = match host_hit {
            Some(payload) => (payload, ResidencyTier::Ram),
            None => {
                let size = self.store.byte_len(key)?;
                self.reserve_host(key, size)?;
                let payload = match self.store.read_payload(key) {
                    Ok(payload) => payload,
                    Err(error) => {
                        self.cancel_host_reservation(size)?;
                        return Err(error.into());
                    }
                };
                let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
                state.metrics.storage_loads += 1;
                self.commit_host(&mut state, key, payload.clone(), size);
                (payload, ResidencyTier::Storage)
            }
        };

        let device = self.try_upload(key, &payload)?;
        Ok(ResidentExpert {
            key,
            host_bytes: Some(payload.bytes),
            tensors: payload.tensors,
            device,
            source_tier,
        })
    }

    /// Prefetch stages bytes into RAM but never steals accelerator bandwidth.
    pub fn prefetch_host(&self, key: ExpertKey) -> Result<(), CacheError> {
        {
            let state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
            if state.host.contains_key(&key) || state.device.contains_key(&key) {
                return Ok(());
            }
        }
        let size = self.store.byte_len(key)?;
        if size > self.ram_cache_budget || self.ram_cache_budget == 0 {
            return Ok(());
        }
        self.reserve_host(key, size)?;
        let payload = match self.store.read_payload(key) {
            Ok(payload) => payload,
            Err(error) => {
                self.cancel_host_reservation(size)?;
                return Err(error.into());
            }
        };
        let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
        if !state.host.contains_key(&key) && !state.device.contains_key(&key) {
            self.commit_host(&mut state, key, payload, size);
        } else {
            state.reserved_host_bytes = state.reserved_host_bytes.saturating_sub(size);
            state
                .retired_host
                .push((Arc::downgrade(&payload.bytes), size));
        }
        Ok(())
    }

    pub fn metrics(&self) -> Result<CacheMetrics, CacheError> {
        let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
        prune_retired_host(&mut state);
        let mut metrics = state.metrics;
        metrics.ram_live_bytes = host_live_bytes(&state);
        metrics.accelerator_live_bytes = self.accelerator_live_bytes.load(Ordering::Acquire);
        metrics.accelerator_peak_bytes = self.accelerator_peak_bytes.load(Ordering::Acquire);
        Ok(metrics)
    }

    pub fn expert_byte_len(&self, key: ExpertKey) -> Result<u64, CacheError> {
        Ok(self.store.byte_len(key)?)
    }

    fn reserve_host(&self, key: ExpertKey, size: u64) -> Result<(), CacheError> {
        let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
        evict_host_to_fit(
            &mut state,
            self.ram_cache_budget.saturating_sub(size),
            Some(key),
        );
        prune_retired_host(&mut state);
        let current = host_live_bytes(&state).saturating_add(state.reserved_host_bytes);
        let required = current.saturating_add(size);
        if required > self.ram_live_budget {
            state.metrics.ram_budget_denials += 1;
            return Err(CacheError::HostBudget {
                key,
                required,
                budget: self.ram_live_budget,
            });
        }
        state.reserved_host_bytes += size;
        state.metrics.ram_peak_bytes = state.metrics.ram_peak_bytes.max(required);
        Ok(())
    }

    fn cancel_host_reservation(&self, size: u64) -> Result<(), CacheError> {
        let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
        state.reserved_host_bytes = state.reserved_host_bytes.saturating_sub(size);
        Ok(())
    }

    fn commit_host(
        &self,
        state: &mut CacheState,
        key: ExpertKey,
        payload: ExpertPayload,
        size: u64,
    ) {
        state.reserved_host_bytes = state.reserved_host_bytes.saturating_sub(size);
        if size > self.ram_cache_budget || self.ram_cache_budget == 0 {
            state
                .retired_host
                .push((Arc::downgrade(&payload.bytes), size));
            return;
        }
        state.clock += 1;
        let last_used = state.clock;
        let heat = *state.heat.get(&key).unwrap_or(&0);
        if let Some(old) = state.host.insert(
            key,
            HostEntry {
                payload,
                last_used,
                heat,
            },
        ) {
            state.retired_host.push((
                Arc::downgrade(&old.payload.bytes),
                old.payload.bytes.len() as u64,
            ));
            state.metrics.ram_bytes = state
                .metrics
                .ram_bytes
                .saturating_sub(old.payload.bytes.len() as u64);
        }
        state.metrics.ram_bytes += size;
    }

    fn try_upload(
        &self,
        key: ExpertKey,
        payload: &ExpertPayload,
    ) -> Result<Option<Arc<DeviceAllocation>>, CacheError> {
        let size = payload.bytes.len() as u64;
        if size > self.accelerator_live_budget || self.accelerator_live_budget == 0 {
            return Ok(None);
        }
        {
            let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
            if let Some(entry) = state.device.get(&key) {
                return Ok(Some(entry.allocation.clone()));
            }
            evict_device_to_fit(
                &mut state,
                self.accelerator_cache_budget.saturating_sub(size),
                Some(key),
            );
        }
        if !reserve_bytes(
            &self.accelerator_live_bytes,
            &self.accelerator_peak_bytes,
            size,
            self.accelerator_live_budget,
        ) {
            let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
            state.metrics.accelerator_budget_denials += 1;
            return Ok(None);
        }
        let buffer = match self.accelerator.upload(key, &payload.bytes) {
            Ok(buffer) => buffer,
            Err(_) => {
                self.accelerator_live_bytes
                    .fetch_sub(size, Ordering::AcqRel);
                return Ok(None);
            }
        };
        if buffer.bytes as u64 != size {
            self.accelerator.release(&buffer);
            self.accelerator_live_bytes
                .fetch_sub(size, Ordering::AcqRel);
            return Ok(None);
        }
        let allocation = Arc::new(DeviceAllocation {
            buffer,
            backend: self.accelerator.clone(),
            live_bytes: self.accelerator_live_bytes.clone(),
            reserved_bytes: size,
        });

        let mut state = self.state.lock().map_err(|_| CacheError::Poisoned)?;
        if let Some(entry) = state.device.get(&key) {
            return Ok(Some(entry.allocation.clone()));
        }
        if size > self.accelerator_cache_budget || self.accelerator_cache_budget == 0 {
            return Ok(Some(allocation));
        }
        state.clock += 1;
        let last_used = state.clock;
        let heat = *state.heat.get(&key).unwrap_or(&0);
        state.device.insert(
            key,
            DeviceEntry {
                allocation: allocation.clone(),
                tensors: payload.tensors.clone(),
                last_used,
                heat,
            },
        );
        state.metrics.accelerator_bytes += size;
        Ok(Some(allocation))
    }
}

fn reserve_bytes(live: &AtomicU64, peak: &AtomicU64, size: u64, budget: u64) -> bool {
    let mut current = live.load(Ordering::Acquire);
    loop {
        let Some(next) = current.checked_add(size) else {
            return false;
        };
        if next > budget {
            return false;
        }
        match live.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                peak.fetch_max(next, Ordering::AcqRel);
                return true;
            }
            Err(actual) => current = actual,
        }
    }
}

fn evict_host_to_fit(state: &mut CacheState, target_bytes: u64, exclude: Option<ExpertKey>) {
    while state.metrics.ram_bytes > target_bytes {
        let candidate = state
            .host
            .iter()
            .filter(|(key, _)| Some(**key) != exclude)
            .min_by_key(|(_, entry)| (entry.heat >= 3, entry.last_used))
            .map(|(key, _)| *key);
        let Some(key) = candidate else {
            break;
        };
        if let Some(entry) = state.host.remove(&key) {
            state.retired_host.push((
                Arc::downgrade(&entry.payload.bytes),
                entry.payload.bytes.len() as u64,
            ));
            state.metrics.ram_bytes = state
                .metrics
                .ram_bytes
                .saturating_sub(entry.payload.bytes.len() as u64);
            state.metrics.ram_evictions += 1;
        }
    }
}

fn prune_retired_host(state: &mut CacheState) {
    state
        .retired_host
        .retain(|(allocation, _)| allocation.strong_count() > 0);
}

fn host_live_bytes(state: &CacheState) -> u64 {
    state
        .retired_host
        .iter()
        .filter(|(allocation, _)| allocation.strong_count() > 0)
        .fold(state.metrics.ram_bytes, |total, (_, bytes)| {
            total.saturating_add(*bytes)
        })
}

fn evict_device_to_fit(state: &mut CacheState, target_bytes: u64, exclude: Option<ExpertKey>) {
    while state.metrics.accelerator_bytes > target_bytes {
        let candidate = state
            .device
            .iter()
            .filter(|(key, _)| Some(**key) != exclude)
            .min_by_key(|(_, entry)| (entry.heat >= 3, entry.last_used))
            .map(|(key, _)| *key);
        let Some(key) = candidate else {
            break;
        };
        if let Some(entry) = state.device.remove(&key) {
            state.metrics.accelerator_bytes = state
                .metrics
                .accelerator_bytes
                .saturating_sub(entry.allocation.buffer.bytes as u64);
            state.metrics.accelerator_evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::manifest::{ExpertLocation, TensorSegment};

    use super::*;

    #[derive(Debug, Default)]
    struct TestAccelerator {
        next: AtomicU64,
        releases: AtomicU64,
    }

    impl Accelerator for TestAccelerator {
        fn name(&self) -> &str {
            "test"
        }

        fn upload(&self, _key: ExpertKey, bytes: &[u8]) -> Result<AcceleratorBuffer, String> {
            Ok(AcceleratorBuffer {
                id: self.next.fetch_add(1, Ordering::Relaxed),
                bytes: bytes.len(),
            })
        }

        fn release(&self, _buffer: &AcceleratorBuffer) {
            self.releases.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn fixture() -> (PathBuf, Arc<ExpertStore>) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-cache-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("experts.bin"), b"AAAABBBB").unwrap();
        let store = Arc::new(ExpertStore::new(
            &root,
            vec![],
            [
                ExpertLocation {
                    layer: 0,
                    expert: 0,
                    segments: vec![TensorSegment {
                        tensor: "gate_proj".into(),
                        dtype: None,
                        shape: vec![],
                        shard: "experts.bin".into(),
                        offset: 0,
                        length: 4,
                    }],
                },
                ExpertLocation {
                    layer: 0,
                    expert: 1,
                    segments: vec![TensorSegment {
                        tensor: "gate_proj".into(),
                        dtype: None,
                        shape: vec![],
                        shard: "experts.bin".into(),
                        offset: 4,
                        length: 4,
                    }],
                },
            ],
        ));
        (root, store)
    }

    #[test]
    fn accelerator_lease_survives_cache_eviction() {
        let (root, store) = fixture();
        let accelerator = Arc::new(TestAccelerator::default());
        let manager = ResidencyManager::new(store, accelerator.clone(), 8, 4);
        let first = manager.get(ExpertKey::new(0, 0)).unwrap();
        let first_id = first.accelerator_buffer().unwrap().id;
        let second = manager.get(ExpertKey::new(0, 1)).unwrap();
        assert_eq!(first.accelerator_buffer().unwrap().id, first_id);
        assert!(second.accelerator_buffer().is_none());
        let metrics = manager.metrics().unwrap();
        assert_eq!(metrics.accelerator_live_bytes, 4);
        assert_eq!(metrics.accelerator_peak_bytes, 4);
        assert_eq!(metrics.accelerator_budget_denials, 1);
        assert_eq!(accelerator.releases.load(Ordering::Relaxed), 0);
        drop(second);
        drop(first);
        assert_eq!(accelerator.releases.load(Ordering::Relaxed), 1);
        let second = manager.get(ExpertKey::new(0, 1)).unwrap();
        assert!(second.accelerator_buffer().is_some());
        assert!(manager.metrics().unwrap().accelerator_peak_bytes <= 4);
        drop(second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_accelerator_keeps_byte_exact_host_payload() {
        let (root, store) = fixture();
        let manager = ResidencyManager::new(store, Arc::new(NoAccelerator), 4, 4);
        let expert = manager.get(ExpertKey::new(0, 0)).unwrap();
        assert_eq!(&**expert.host_bytes.as_ref().unwrap(), b"AAAA");
        assert!(expert.accelerator_buffer().is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn evicted_host_lease_counts_against_the_hard_live_budget() {
        let (root, store) = fixture();
        let manager = ResidencyManager::with_budgets(store, Arc::new(NoAccelerator), 4, 4, 0, 0);
        let first = manager.get(ExpertKey::new(0, 0)).unwrap();
        let denied = manager.get(ExpertKey::new(0, 1)).unwrap_err();
        assert!(matches!(denied, CacheError::HostBudget { .. }));
        let metrics = manager.metrics().unwrap();
        assert_eq!(metrics.ram_live_bytes, 4);
        assert_eq!(metrics.ram_peak_bytes, 4);
        assert_eq!(metrics.ram_budget_denials, 1);
        drop(first);
        let second = manager.get(ExpertKey::new(0, 1)).unwrap();
        assert_eq!(&**second.host_bytes.as_ref().unwrap(), b"BBBB");
        assert!(manager.metrics().unwrap().ram_peak_bytes <= 4);
        drop(second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accelerator_staging_can_be_live_without_becoming_cached() {
        let (root, store) = fixture();
        let accelerator = Arc::new(TestAccelerator::default());
        let manager = ResidencyManager::with_budgets(store, accelerator.clone(), 8, 8, 0, 4);
        let first = manager.get(ExpertKey::new(0, 0)).unwrap();
        assert!(first.accelerator_buffer().is_some());
        assert_eq!(manager.metrics().unwrap().accelerator_bytes, 0);
        let second = manager.get(ExpertKey::new(0, 1)).unwrap();
        assert!(second.accelerator_buffer().is_none());
        drop(first);
        let second = manager.get(ExpertKey::new(0, 1)).unwrap();
        assert!(second.accelerator_buffer().is_some());
        assert_eq!(manager.metrics().unwrap().accelerator_bytes, 0);
        drop(second);
        assert_eq!(accelerator.releases.load(Ordering::Relaxed), 2);
        fs::remove_dir_all(root).unwrap();
    }
}
