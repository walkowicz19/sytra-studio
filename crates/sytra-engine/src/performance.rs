//! Storage-I/O performance bounds for out-of-core MoE execution.
//!
//! These estimates are deliberately bandwidth-bound and conservative. They
//! prevent a memory-feasible plan from being presented as a throughput
//! promise when the active weights still have to cross NVMe every token.

use std::collections::HashMap;

use serde::Serialize;
use thiserror::Error;

use crate::{ModelFamily, RuntimeManifest};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct IoPerformanceEstimate {
    pub verification_positions: u64,
    pub storage_bandwidth_bytes_per_second: u64,
    pub dense_bytes_per_forward: u64,
    pub cold_expert_bytes_per_token: u64,
    pub expected_expert_union_bytes_per_forward: u64,
    pub dense_storage_bytes_per_forward: u64,
    pub expected_expert_storage_bytes_per_forward: u64,
    pub cold_storage_bytes_per_token: u64,
    pub perfect_acceptance_storage_bytes_per_emitted_token: u64,
    pub cold_io_bound_tokens_per_second: f64,
    pub perfect_acceptance_io_bound_tokens_per_second: f64,
    pub target_tokens_per_second: Option<f64>,
    pub required_bandwidth_bytes_per_second_for_target: Option<u64>,
    pub target_io_feasible: Option<bool>,
    pub notes: Vec<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PerformanceError {
    #[error("verification positions and storage bandwidth must be positive")]
    InvalidInput,
    #[error("performance arithmetic overflow")]
    Overflow,
}

pub fn estimate_io_performance(
    manifest: &RuntimeManifest,
    verification_positions: u64,
    storage_bandwidth_bytes_per_second: u64,
    dense_cache_bytes: u64,
    expert_cache_bytes: u64,
    target_tokens_per_second: Option<f64>,
) -> Result<IoPerformanceEstimate, PerformanceError> {
    if verification_positions == 0 || storage_bandwidth_bytes_per_second == 0 {
        return Err(PerformanceError::InvalidInput);
    }
    if target_tokens_per_second.is_some_and(|target| !target.is_finite() || target <= 0.0) {
        return Err(PerformanceError::InvalidInput);
    }

    let dense_bytes = decode_dense_bytes(manifest)?;
    // A cyclic scan larger than the cache has no guaranteed cross-token LRU
    // reuse. Only claim dense residency when the whole decode set fits.
    let dense_storage = if dense_cache_bytes >= dense_bytes {
        0
    } else {
        dense_bytes
    };

    let mut layers: HashMap<u32, (u64, u64)> = HashMap::new();
    for expert in &manifest.storage.experts {
        let entry = layers.entry(expert.layer).or_default();
        entry.0 = entry.0.saturating_add(expert.byte_len());
        entry.1 += 1;
    }
    let experts_per_layer = u64::from(manifest.architecture.experts_per_layer);
    let top_k = u64::from(manifest.architecture.experts_per_token);
    let mut cold_expert = 0_u64;
    let mut union_expert = 0_u64;
    for (bytes, count) in layers.values() {
        if *count == 0 {
            continue;
        }
        let average = bytes / count;
        cold_expert = cold_expert
            .checked_add(
                average
                    .checked_mul(top_k)
                    .ok_or(PerformanceError::Overflow)?,
            )
            .ok_or(PerformanceError::Overflow)?;
        let miss_probability = if experts_per_layer == 0 {
            0.0
        } else {
            (experts_per_layer.saturating_sub(top_k)) as f64 / experts_per_layer as f64
        };
        let expected_unique =
            experts_per_layer as f64 * (1.0 - miss_probability.powf(verification_positions as f64));
        let expected_bytes = (expected_unique * average as f64).ceil();
        union_expert = union_expert
            .checked_add(expected_bytes.min(u64::MAX as f64) as u64)
            .ok_or(PerformanceError::Overflow)?;
    }
    let total_expert_bytes = manifest.expert_bytes();
    let expert_miss_fraction = if total_expert_bytes == 0 {
        0.0
    } else {
        1.0 - (expert_cache_bytes as f64 / total_expert_bytes as f64).clamp(0.0, 1.0)
    };
    let expert_storage = (union_expert as f64 * expert_miss_fraction).ceil() as u64;
    let cold_per_token = dense_bytes
        .checked_add(cold_expert)
        .ok_or(PerformanceError::Overflow)?;
    let per_forward = dense_storage
        .checked_add(expert_storage)
        .ok_or(PerformanceError::Overflow)?;
    let per_emitted = per_forward.div_ceil(verification_positions);
    let tps = |bytes: u64| {
        if bytes == 0 {
            f64::INFINITY
        } else {
            storage_bandwidth_bytes_per_second as f64 / bytes as f64
        }
    };
    let required = target_tokens_per_second
        .map(|target| (per_emitted as f64 * target).ceil().min(u64::MAX as f64) as u64);
    let target_feasible = required.map(|required| required <= storage_bandwidth_bytes_per_second);
    let mut notes = vec![
        "I/O bounds exclude compute, PCIe copies, synchronization, draft cost, and tokenizer overhead"
            .into(),
        "perfect-acceptance throughput assumes every verification position produces a useful token"
            .into(),
        "expert cache benefit uses a uniform-routing approximation; measured routing heat must replace it"
            .into(),
    ];
    if dense_storage > 0 && dense_cache_bytes > 0 {
        notes.push(
            "dense decode weights exceed the bounded cache, so cyclic LRU cannot guarantee reuse"
                .into(),
        );
    }
    if matches!(target_feasible, Some(false)) {
        notes.push("the requested token rate is impossible at the declared storage bandwidth even before compute".into());
    }
    Ok(IoPerformanceEstimate {
        verification_positions,
        storage_bandwidth_bytes_per_second,
        dense_bytes_per_forward: dense_bytes,
        cold_expert_bytes_per_token: cold_expert,
        expected_expert_union_bytes_per_forward: union_expert,
        dense_storage_bytes_per_forward: dense_storage,
        expected_expert_storage_bytes_per_forward: expert_storage,
        cold_storage_bytes_per_token: cold_per_token,
        perfect_acceptance_storage_bytes_per_emitted_token: per_emitted,
        cold_io_bound_tokens_per_second: tps(cold_per_token),
        perfect_acceptance_io_bound_tokens_per_second: tps(per_emitted),
        target_tokens_per_second,
        required_bandwidth_bytes_per_second_for_target: required,
        target_io_feasible: target_feasible,
        notes,
    })
}

fn decode_dense_bytes(manifest: &RuntimeManifest) -> Result<u64, PerformanceError> {
    if manifest.storage.dense_tensors.is_empty() {
        return Ok(manifest.dense_bytes);
    }
    let mut total = 0_u64;
    for tensor in &manifest.storage.dense_tensors {
        let name = tensor.tensor.as_str();
        let used = match &manifest.architecture.family {
            ModelFamily::KimiK27 => {
                name.starts_with("language_model.model.")
                    || name.starts_with("language_model.lm_head.")
            }
            _ => !name.contains("vision_tower") && !name.contains("mm_projector"),
        };
        if !used {
            continue;
        }
        let bytes = if name.ends_with("embed_tokens.weight") {
            u64::from(manifest.architecture.hidden_size).saturating_mul(2)
        } else {
            tensor.length
        };
        total = total.checked_add(bytes).ok_or(PerformanceError::Overflow)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActivationKind, ArchitectureContract, AttentionContract, AttentionKind, ExpertLocation,
        ExpertStorageManifest, ExpertTensorLayout, QuantizationContract, RouterContract,
        RouterSemantics, TensorSegment, WeightFormat,
    };

    #[test]
    fn batch_amortizes_dense_scan_but_not_routed_union_linearly() {
        let manifest = RuntimeManifest {
            schema_version: 1,
            architecture: ArchitectureContract {
                adapter: "test".into(),
                model_type: "test".into(),
                attention: AttentionKind::Mla,
                router: RouterSemantics::NoAuxTc,
                expert_format: WeightFormat::Int4Group,
                family: ModelFamily::KimiK27,
                expert_layout: ExpertTensorLayout::Discrete,
                activation: ActivationKind::Silu,
                router_config: RouterContract::default(),
                attention_config: AttentionContract::default(),
                quantization: QuantizationContract::default(),
                hidden_size: 4,
                expert_intermediate_size: 4,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 4,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: 1000,
            storage: ExpertStorageManifest {
                contiguous_experts: true,
                experts: (0..4)
                    .map(|expert| ExpertLocation {
                        layer: 0,
                        expert,
                        segments: vec![TensorSegment {
                            tensor: format!("expert.{expert}"),
                            dtype: None,
                            shape: vec![],
                            shard: "experts.bin".into(),
                            offset: u64::from(expert) * 100,
                            length: 100,
                        }],
                    })
                    .collect(),
                dense_tensors: vec![TensorSegment {
                    tensor: "language_model.model.layers.0.weight".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![10, 50],
                    shard: "dense.bin".into(),
                    offset: 0,
                    length: 1000,
                }],
            },
        };
        let single = estimate_io_performance(&manifest, 1, 5_000, 0, 0, Some(5.0)).unwrap();
        let batch = estimate_io_performance(&manifest, 4, 5_000, 0, 0, Some(5.0)).unwrap();
        assert_eq!(single.cold_storage_bytes_per_token, 1100);
        assert!(batch.perfect_acceptance_storage_bytes_per_emitted_token < 1100);
        assert!(batch.expected_expert_union_bytes_per_forward > 100);
        assert_eq!(single.target_io_feasible, Some(false));
    }
}
