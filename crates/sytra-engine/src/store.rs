use std::{
    collections::HashMap,
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::manifest::ExpertLocation;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

#[derive(Debug, Clone)]
pub struct WeightedMirror {
    pub root: PathBuf,
    pub weight: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentTensor {
    pub name: String,
    pub dtype: Option<String>,
    pub shape: Vec<u64>,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone)]
pub struct ExpertPayload {
    pub bytes: Arc<[u8]>,
    pub tensors: Arc<[ResidentTensor]>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct StoreMetrics {
    pub primary_bytes: u64,
    pub mirror_bytes: u64,
    pub mirror_fallbacks: u64,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("expert layer {0} expert {1} is not indexed")]
    UnknownExpert(u32, u32),
    #[error("could not read expert layer {key:?} from {path}: {source}")]
    Read {
        key: ExpertKey,
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("expert layer {key:?} produced {actual} bytes; expected {expected}")]
    ShortRead {
        key: ExpertKey,
        expected: usize,
        actual: usize,
    },
}

#[derive(Debug, Error)]
pub enum TensorStoreError {
    #[error("dense tensor {0} is not indexed")]
    Unknown(String),
    #[error("could not read dense tensor {tensor} from {path}: {source}")]
    Read {
        tensor: String,
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "dense tensor {tensor} window offset={offset} length={length} exceeds tensor length {tensor_length}"
    )]
    Range {
        tensor: String,
        offset: u64,
        length: u64,
        tensor_length: u64,
    },
    #[error(
        "dense window {tensor} needs {required} live bytes but the hard host limit is {budget}"
    )]
    Budget {
        tensor: String,
        required: u64,
        budget: u64,
    },
    #[error("dense tensor cache state is poisoned")]
    Poisoned,
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct DenseStoreMetrics {
    pub cache_hits: u64,
    pub storage_reads: u64,
    pub storage_bytes: u64,
    pub evictions: u64,
    pub cached_bytes: u64,
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    pub budget_denials: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DenseWindowKey {
    tensor: String,
    offset: u64,
    length: u64,
}

#[derive(Debug)]
struct DenseCacheEntry {
    bytes: Arc<[u8]>,
    last_used: u64,
}

#[derive(Debug, Default)]
struct DenseCacheState {
    entries: HashMap<DenseWindowKey, DenseCacheEntry>,
    retired: Vec<(Weak<[u8]>, u64)>,
    clock: u64,
    reserved_bytes: u64,
    metrics: DenseStoreMetrics,
}

#[derive(Debug)]
struct Counters {
    primary_bytes: AtomicU64,
    mirror_bytes: AtomicU64,
    mirror_fallbacks: AtomicU64,
}

/// Byte-exact immutable expert storage with deterministic weighted mirrors.
#[derive(Debug)]
pub struct ExpertStore {
    primary: PathBuf,
    mirrors: Vec<WeightedMirror>,
    index: HashMap<ExpertKey, ExpertLocation>,
    counters: Arc<Counters>,
}

/// Byte-range access to non-routed tensors.
///
/// Keeping these tensors addressable is essential for Kimi-class checkpoints:
/// its BF16 attention/shared backbone is larger than low-end system RAM, so a
/// layer bundle must be streamable instead of being declared permanently
/// resident.
#[derive(Debug)]
pub struct DenseTensorStore {
    primary: PathBuf,
    mirrors: Vec<WeightedMirror>,
    index: HashMap<String, crate::manifest::TensorSegment>,
    cache_budget: u64,
    live_budget: u64,
    cache: Mutex<DenseCacheState>,
}

impl DenseTensorStore {
    pub fn new(
        primary: impl Into<PathBuf>,
        mirrors: Vec<WeightedMirror>,
        tensors: impl IntoIterator<Item = crate::manifest::TensorSegment>,
    ) -> Self {
        Self::with_budgets(primary, mirrors, tensors, 0, u64::MAX)
    }

    pub fn with_budgets(
        primary: impl Into<PathBuf>,
        mirrors: Vec<WeightedMirror>,
        tensors: impl IntoIterator<Item = crate::manifest::TensorSegment>,
        cache_budget: u64,
        live_budget: u64,
    ) -> Self {
        Self {
            primary: primary.into(),
            mirrors: mirrors
                .into_iter()
                .filter(|mirror| mirror.weight > 0)
                .collect(),
            index: tensors
                .into_iter()
                .map(|tensor| (tensor.tensor.clone(), tensor))
                .collect(),
            cache_budget,
            live_budget,
            cache: Mutex::new(DenseCacheState::default()),
        }
    }

    pub fn read(&self, tensor: &str) -> Result<Arc<[u8]>, TensorStoreError> {
        let segment = self
            .index
            .get(tensor)
            .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
        self.read_window(tensor, 0, segment.length)
    }

    /// Read a bounded byte window from a dense tensor. Forward kernels use
    /// this to tile embeddings, attention, and output matrices instead of
    /// allocating a complete multi-gigabyte tensor on low-memory hosts.
    pub fn read_window(
        &self,
        tensor: &str,
        relative_offset: u64,
        length: u64,
    ) -> Result<Arc<[u8]>, TensorStoreError> {
        let segment = self
            .index
            .get(tensor)
            .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
        let end = relative_offset
            .checked_add(length)
            .filter(|end| *end <= segment.length)
            .ok_or_else(|| TensorStoreError::Range {
                tensor: tensor.into(),
                offset: relative_offset,
                length,
                tensor_length: segment.length,
            })?;
        if length == 0 || end <= relative_offset {
            return Err(TensorStoreError::Range {
                tensor: tensor.into(),
                offset: relative_offset,
                length,
                tensor_length: segment.length,
            });
        }
        let key = DenseWindowKey {
            tensor: tensor.into(),
            offset: relative_offset,
            length,
        };
        {
            let mut cache = self.cache.lock().map_err(|_| TensorStoreError::Poisoned)?;
            cache.clock += 1;
            let now = cache.clock;
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.last_used = now;
                let bytes = entry.bytes.clone();
                cache.metrics.cache_hits += 1;
                refresh_dense_live_metrics(&mut cache);
                return Ok(bytes);
            }
            reserve_dense_window(
                &mut cache,
                self.cache_budget,
                self.live_budget,
                length,
                tensor,
            )?;
        }

        let primary_path = self.primary.join(&segment.shard);
        let read_result =
            if let Some(mirror) = self.select_mirror(tensor, &segment.shard, &primary_path) {
                read_named_range(&mirror, tensor, segment.offset + relative_offset, length).or_else(
                    |_| {
                        read_named_range(
                            &primary_path,
                            tensor,
                            segment.offset + relative_offset,
                            length,
                        )
                    },
                )
            } else {
                read_named_range(
                    &primary_path,
                    tensor,
                    segment.offset + relative_offset,
                    length,
                )
            };
        let bytes = match read_result {
            Ok(bytes) => bytes,
            Err(error) => {
                self.cancel_dense_reservation(length);
                return Err(error);
            }
        };

        let mut cache = self.cache.lock().map_err(|_| TensorStoreError::Poisoned)?;
        cache.reserved_bytes = cache.reserved_bytes.saturating_sub(length);
        cache.metrics.storage_reads += 1;
        cache.metrics.storage_bytes += length;
        cache.clock += 1;
        let now = cache.clock;
        if length <= self.cache_budget {
            if let Some(existing) = cache.entries.get_mut(&key) {
                existing.last_used = now;
                let existing = existing.bytes.clone();
                cache.retired.push((Arc::downgrade(&bytes), length));
                refresh_dense_live_metrics(&mut cache);
                return Ok(existing);
            }
            cache.metrics.cached_bytes += length;
            cache.entries.insert(
                key,
                DenseCacheEntry {
                    bytes: bytes.clone(),
                    last_used: now,
                },
            );
        } else {
            cache.retired.push((Arc::downgrade(&bytes), length));
        }
        refresh_dense_live_metrics(&mut cache);
        Ok(bytes)
    }

    pub fn contains(&self, tensor: &str) -> bool {
        self.index.contains_key(tensor)
    }

    pub fn metadata(&self, tensor: &str) -> Option<&crate::manifest::TensorSegment> {
        self.index.get(tensor)
    }

    pub fn layer_tensor_names(&self, layer: usize) -> Vec<&str> {
        let marker = format!(".layers.{layer}.");
        let mut names: Vec<_> = self
            .index
            .keys()
            .filter(|name| name.contains(&marker))
            .map(String::as_str)
            .collect();
        names.sort_unstable();
        names
    }

    pub fn metrics(&self) -> Result<DenseStoreMetrics, TensorStoreError> {
        let mut cache = self.cache.lock().map_err(|_| TensorStoreError::Poisoned)?;
        refresh_dense_live_metrics(&mut cache);
        Ok(cache.metrics)
    }

    fn cancel_dense_reservation(&self, length: u64) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.reserved_bytes = cache.reserved_bytes.saturating_sub(length);
            refresh_dense_live_metrics(&mut cache);
        }
    }

    fn select_mirror(
        &self,
        tensor: &str,
        relative_path: &Path,
        primary_path: &Path,
    ) -> Option<PathBuf> {
        let primary_len = fs::metadata(primary_path).ok()?.len();
        let eligible: Vec<_> = self
            .mirrors
            .iter()
            .filter_map(|mirror| {
                let path = mirror.root.join(relative_path);
                (fs::metadata(&path).ok()?.len() == primary_len).then_some((mirror, path))
            })
            .collect();
        if eligible.is_empty() {
            return None;
        }
        let total = 1_u64 + eligible.iter().map(|(m, _)| m.weight as u64).sum::<u64>();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tensor.hash(&mut hasher);
        let mut bucket = hasher.finish() % total;
        if bucket == 0 {
            return None;
        }
        bucket -= 1;
        for (mirror, path) in eligible {
            if bucket < mirror.weight as u64 {
                return Some(path);
            }
            bucket -= mirror.weight as u64;
        }
        None
    }
}

fn reserve_dense_window(
    state: &mut DenseCacheState,
    cache_budget: u64,
    live_budget: u64,
    length: u64,
    tensor: &str,
) -> Result<(), TensorStoreError> {
    refresh_dense_live_metrics(state);
    while state.metrics.cached_bytes.saturating_add(length) > cache_budget
        || state
            .metrics
            .live_bytes
            .saturating_add(state.reserved_bytes)
            .saturating_add(length)
            > live_budget
    {
        let Some(key) = state
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        let entry = state
            .entries
            .remove(&key)
            .expect("selected dense cache entry exists");
        state.metrics.cached_bytes = state
            .metrics
            .cached_bytes
            .saturating_sub(entry.bytes.len() as u64);
        if Arc::strong_count(&entry.bytes) > 1 {
            state
                .retired
                .push((Arc::downgrade(&entry.bytes), entry.bytes.len() as u64));
        }
        state.metrics.evictions += 1;
        refresh_dense_live_metrics(state);
    }
    let required = state
        .metrics
        .live_bytes
        .saturating_add(state.reserved_bytes)
        .saturating_add(length);
    if required > live_budget {
        state.metrics.budget_denials += 1;
        return Err(TensorStoreError::Budget {
            tensor: tensor.into(),
            required,
            budget: live_budget,
        });
    }
    state.reserved_bytes += length;
    state.metrics.peak_live_bytes = state.metrics.peak_live_bytes.max(required);
    Ok(())
}

fn refresh_dense_live_metrics(state: &mut DenseCacheState) {
    state.retired.retain(|(bytes, _)| bytes.strong_count() > 0);
    let retired = state.retired.iter().map(|(_, length)| *length).sum::<u64>();
    state.metrics.live_bytes = state.metrics.cached_bytes.saturating_add(retired);
    state.metrics.peak_live_bytes = state.metrics.peak_live_bytes.max(
        state
            .metrics
            .live_bytes
            .saturating_add(state.reserved_bytes),
    );
}

impl ExpertStore {
    pub fn new(
        primary: impl Into<PathBuf>,
        mirrors: Vec<WeightedMirror>,
        locations: impl IntoIterator<Item = ExpertLocation>,
    ) -> Self {
        Self {
            primary: primary.into(),
            mirrors: mirrors
                .into_iter()
                .filter(|mirror| mirror.weight > 0)
                .collect(),
            index: locations
                .into_iter()
                .map(|location| (location.key(), location))
                .collect(),
            counters: Arc::new(Counters {
                primary_bytes: AtomicU64::new(0),
                mirror_bytes: AtomicU64::new(0),
                mirror_fallbacks: AtomicU64::new(0),
            }),
        }
    }

    pub fn read(&self, key: ExpertKey) -> Result<Arc<[u8]>, StoreError> {
        Ok(self.read_payload(key)?.bytes)
    }

    pub fn byte_len(&self, key: ExpertKey) -> Result<u64, StoreError> {
        self.index
            .get(&key)
            .map(ExpertLocation::byte_len)
            .ok_or(StoreError::UnknownExpert(key.layer, key.expert))
    }

    pub fn read_payload(&self, key: ExpertKey) -> Result<ExpertPayload, StoreError> {
        let location = self
            .index
            .get(&key)
            .ok_or(StoreError::UnknownExpert(key.layer, key.expert))?;
        let expected = usize::try_from(location.byte_len()).map_err(|_| StoreError::ShortRead {
            key,
            expected: usize::MAX,
            actual: 0,
        })?;
        let mut expert = Vec::with_capacity(expected);
        let mut tensors = Vec::with_capacity(location.segments.len());
        for segment in &location.segments {
            let tensor_offset = expert.len();
            let primary_path = self.primary.join(&segment.shard);
            let selected = self.select_mirror(key, &segment.shard, &primary_path);
            if let Some(mirror_path) = selected {
                match read_range(&mirror_path, key, segment.offset, segment.length) {
                    Ok(bytes) => {
                        self.counters
                            .mirror_bytes
                            .fetch_add(segment.length, Ordering::Relaxed);
                        expert.extend_from_slice(&bytes);
                        tensors.push(ResidentTensor {
                            name: segment.tensor.clone(),
                            dtype: segment.dtype.clone(),
                            shape: segment.shape.clone(),
                            offset: tensor_offset,
                            length: bytes.len(),
                        });
                        continue;
                    }
                    Err(_) => {
                        self.counters
                            .mirror_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            let bytes = read_range(&primary_path, key, segment.offset, segment.length)?;
            self.counters
                .primary_bytes
                .fetch_add(segment.length, Ordering::Relaxed);
            expert.extend_from_slice(&bytes);
            tensors.push(ResidentTensor {
                name: segment.tensor.clone(),
                dtype: segment.dtype.clone(),
                shape: segment.shape.clone(),
                offset: tensor_offset,
                length: bytes.len(),
            });
        }
        if expert.len() != expected {
            return Err(StoreError::ShortRead {
                key,
                expected,
                actual: expert.len(),
            });
        }
        Ok(ExpertPayload {
            bytes: expert.into(),
            tensors: tensors.into(),
        })
    }

    pub fn metrics(&self) -> StoreMetrics {
        StoreMetrics {
            primary_bytes: self.counters.primary_bytes.load(Ordering::Relaxed),
            mirror_bytes: self.counters.mirror_bytes.load(Ordering::Relaxed),
            mirror_fallbacks: self.counters.mirror_fallbacks.load(Ordering::Relaxed),
        }
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.index.contains_key(&key)
    }

    fn select_mirror(
        &self,
        key: ExpertKey,
        relative_path: &Path,
        primary_path: &Path,
    ) -> Option<PathBuf> {
        let primary_len = fs::metadata(primary_path).ok()?.len();
        let eligible: Vec<_> = self
            .mirrors
            .iter()
            .filter_map(|mirror| {
                let path = mirror.root.join(relative_path);
                let length = fs::metadata(&path).ok()?.len();
                (length == primary_len).then_some((mirror, path))
            })
            .collect();
        if eligible.is_empty() {
            return None;
        }

        // Primary participates with weight 1. A key always maps to the same
        // source, so speculative and demand reads do not populate two caches.
        let total = 1u64 + eligible.iter().map(|(m, _)| m.weight as u64).sum::<u64>();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let mut bucket = hasher.finish() % total;
        if bucket == 0 {
            return None;
        }
        bucket -= 1;
        for (mirror, path) in eligible {
            if bucket < mirror.weight as u64 {
                return Some(path);
            }
            bucket -= mirror.weight as u64;
        }
        None
    }
}

fn read_range(
    path: &Path,
    key: ExpertKey,
    offset: u64,
    length: u64,
) -> Result<Arc<[u8]>, StoreError> {
    let expected = usize::try_from(length).map_err(|_| StoreError::ShortRead {
        key,
        expected: usize::MAX,
        actual: 0,
    })?;
    let mut file = File::open(path).map_err(|source| StoreError::Read {
        key,
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| StoreError::Read {
            key,
            path: path.to_owned(),
            source,
        })?;
    let mut bytes = vec![0; expected];
    file.read_exact(&mut bytes)
        .map_err(|source| StoreError::Read {
            key,
            path: path.to_owned(),
            source,
        })?;
    if bytes.len() != expected {
        return Err(StoreError::ShortRead {
            key,
            expected,
            actual: bytes.len(),
        });
    }
    Ok(bytes.into())
}

fn read_named_range(
    path: &Path,
    tensor: &str,
    offset: u64,
    length: u64,
) -> Result<Arc<[u8]>, TensorStoreError> {
    let expected = usize::try_from(length).map_err(|_| TensorStoreError::Read {
        tensor: tensor.into(),
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, "tensor is too large"),
    })?;
    let mut file = File::open(path).map_err(|source| TensorStoreError::Read {
        tensor: tensor.into(),
        path: path.to_owned(),
        source,
    })?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|source| TensorStoreError::Read {
            tensor: tensor.into(),
            path: path.to_owned(),
            source,
        })?;
    let mut bytes = vec![0; expected];
    file.read_exact(&mut bytes)
        .map_err(|source| TensorStoreError::Read {
            tensor: tensor.into(),
            path: path.to_owned(),
            source,
        })?;
    Ok(bytes.into())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::manifest::TensorSegment;

    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sytra-engine-{label}-{nonce}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn reads_exact_expert_range() {
        let root = temp_root("range");
        fs::write(root.join("experts.bin"), b"AAAABBBB").unwrap();
        let store = ExpertStore::new(
            &root,
            vec![],
            [ExpertLocation {
                layer: 0,
                expert: 1,
                segments: vec![TensorSegment {
                    tensor: "down_proj".into(),
                    dtype: None,
                    shape: vec![],
                    shard: "experts.bin".into(),
                    offset: 4,
                    length: 4,
                }],
            }],
        );
        assert_eq!(&*store.read(ExpertKey::new(0, 1)).unwrap(), b"BBBB");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_mirror_is_ignored_without_changing_bytes() {
        let primary = temp_root("primary");
        let mirror = temp_root("mirror");
        fs::write(primary.join("experts.bin"), b"AAAABBBB").unwrap();
        fs::write(mirror.join("experts.bin"), b"bad").unwrap();
        let store = ExpertStore::new(
            &primary,
            vec![WeightedMirror {
                root: mirror.clone(),
                weight: 100,
            }],
            [ExpertLocation {
                layer: 0,
                expert: 0,
                segments: vec![TensorSegment {
                    tensor: "down_proj".into(),
                    dtype: None,
                    shape: vec![],
                    shard: "experts.bin".into(),
                    offset: 0,
                    length: 4,
                }],
            }],
        );
        assert_eq!(&*store.read(ExpertKey::new(0, 0)).unwrap(), b"AAAA");
        assert_eq!(store.metrics().primary_bytes, 4);
        fs::remove_dir_all(primary).unwrap();
        fs::remove_dir_all(mirror).unwrap();
    }

    #[test]
    fn dense_tensor_store_streams_one_indexed_range() {
        let root = temp_root("dense-range");
        fs::write(root.join("model.safetensors"), b"HEADERATTENTIONTAIL").unwrap();
        let store = DenseTensorStore::new(
            &root,
            vec![],
            [crate::manifest::TensorSegment {
                tensor: "language_model.model.layers.1.self_attn.q_a_proj.weight".into(),
                dtype: Some("BF16".into()),
                shape: vec![2, 2],
                shard: "model.safetensors".into(),
                offset: 6,
                length: 9,
            }],
        );
        assert_eq!(
            &*store
                .read("language_model.model.layers.1.self_attn.q_a_proj.weight")
                .unwrap(),
            b"ATTENTION"
        );
        assert_eq!(store.layer_tensor_names(1).len(), 1);
        assert_eq!(
            &*store
                .read_window(
                    "language_model.model.layers.1.self_attn.q_a_proj.weight",
                    2,
                    4,
                )
                .unwrap(),
            b"TENT"
        );
        assert!(matches!(
            store.read_window(
                "language_model.model.layers.1.self_attn.q_a_proj.weight",
                8,
                2,
            ),
            Err(TensorStoreError::Range { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn dense_cache_counts_evicted_live_leases_against_the_hard_budget() {
        let root = temp_root("dense-cache-budget");
        fs::write(root.join("weights.bin"), b"AAAABBBB").unwrap();
        let store = DenseTensorStore::with_budgets(
            &root,
            vec![],
            [TensorSegment {
                tensor: "matrix".into(),
                dtype: Some("BF16".into()),
                shape: vec![2, 2],
                shard: "weights.bin".into(),
                offset: 0,
                length: 8,
            }],
            4,
            4,
        );
        let first = store.read_window("matrix", 0, 4).unwrap();
        let first_hit = store.read_window("matrix", 0, 4).unwrap();
        assert!(matches!(
            store.read_window("matrix", 4, 4),
            Err(TensorStoreError::Budget { .. })
        ));
        drop(first_hit);
        drop(first);
        assert_eq!(&*store.read_window("matrix", 4, 4).unwrap(), b"BBBB");
        let metrics = store.metrics().unwrap();
        assert_eq!(metrics.cache_hits, 1);
        assert_eq!(metrics.budget_denials, 1);
        assert!(metrics.peak_live_bytes <= 4);
        assert!(metrics.live_bytes <= 4);
        fs::remove_dir_all(root).unwrap();
    }
}
