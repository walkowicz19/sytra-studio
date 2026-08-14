//! Hard memory-envelope planning for low-memory hosts.
//!
//! Cache limits alone are insufficient: active leases, one dense tile, and KV
//! state coexist. This module partitions the user-provided caps so their sum
//! never exceeds the declared RAM/VRAM envelope.

use serde::Serialize;
use thiserror::Error;

use crate::{AttentionKind, RuntimeManifest};

const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvTier {
    Accelerator,
    Ram,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MemoryEnvelope {
    pub ram_limit_bytes: u64,
    pub accelerator_limit_bytes: u64,
    pub ram_runtime_reserve_bytes: u64,
    pub accelerator_runtime_reserve_bytes: u64,
    pub host_staging_bytes: u64,
    pub accelerator_staging_bytes: u64,
    pub ram_cache_bytes: u64,
    pub accelerator_cache_bytes: u64,
    pub kv_tier: KvTier,
    pub kv_bytes_per_token: Option<u64>,
    pub kv_budget_bytes: u64,
    pub requested_context_tokens: u64,
    pub effective_context_tokens: u64,
    pub context_clamped: bool,
    pub requested_verification_positions: u64,
    pub max_verification_positions: u64,
    pub verification_clamped: bool,
    pub accelerator_dense_execution: bool,
    pub accelerator_expert_execution: bool,
    pub feasible: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ResourceError {
    #[error("RAM, VRAM, context, scalar width, and dense tile must be positive")]
    InvalidInput,
    #[error("memory arithmetic overflow")]
    Overflow,
}

pub fn plan_memory_envelope(
    manifest: &RuntimeManifest,
    requested_context_tokens: u64,
    ram_limit_bytes: u64,
    accelerator_limit_bytes: u64,
    kv_scalar_bytes: u64,
    dense_tile_bytes: u64,
    requested_verification_positions: u64,
) -> Result<MemoryEnvelope, ResourceError> {
    plan_memory_envelope_with_capabilities(
        manifest,
        requested_context_tokens,
        ram_limit_bytes,
        accelerator_limit_bytes,
        kv_scalar_bytes,
        dense_tile_bytes,
        requested_verification_positions,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn plan_memory_envelope_with_capabilities(
    manifest: &RuntimeManifest,
    requested_context_tokens: u64,
    ram_limit_bytes: u64,
    accelerator_limit_bytes: u64,
    kv_scalar_bytes: u64,
    dense_tile_bytes: u64,
    requested_verification_positions: u64,
    accelerator_kv_supported: bool,
) -> Result<MemoryEnvelope, ResourceError> {
    if requested_context_tokens == 0
        || ram_limit_bytes == 0
        || kv_scalar_bytes == 0
        || dense_tile_bytes == 0
        || requested_verification_positions == 0
    {
        return Err(ResourceError::InvalidInput);
    }

    let ram_reserve = reserve(ram_limit_bytes, 256 * MIB, 2 * 1024 * MIB);
    let accelerator_reserve = reserve(accelerator_limit_bytes, 256 * MIB, 1024 * MIB);
    let ram_usable = ram_limit_bytes.saturating_sub(ram_reserve);
    let accelerator_usable = accelerator_limit_bytes.saturating_sub(accelerator_reserve);
    let max_expert = manifest
        .storage
        .experts
        .iter()
        .map(|expert| expert.byte_len())
        .max()
        .unwrap_or_default();
    let max_dense = manifest
        .storage
        .dense_tensors
        .iter()
        .map(|tensor| tensor.length)
        .max()
        .unwrap_or_default();
    let dense_staging = dense_tile_bytes.min(max_dense.max(1));
    let host_position_cap =
        max_fitting_positions(requested_verification_positions, ram_usable, |positions| {
            let (working, _) = dense_working_set(manifest, dense_staging, positions);
            let (expert_working, _) = expert_working_set(manifest, max_expert, positions);
            expert_working.max(working)
        });
    let accelerator_position_cap = max_fitting_positions(
        requested_verification_positions,
        accelerator_usable,
        |positions| {
            let (_, working) = dense_working_set(manifest, dense_staging, positions);
            working
        },
    );
    let accelerator_dense_kernel = manifest
        .storage
        .dense_tensors
        .iter()
        .filter(|tensor| tensor.shape.len() == 2)
        .all(|tensor| {
            tensor.dtype.as_deref() == Some("BF16")
                || (tensor.dtype.as_deref() == Some("I32")
                    && tensor.tensor.ends_with(".weight_packed"))
        });
    let accelerator_dense_execution = accelerator_dense_kernel && accelerator_position_cap > 0;
    let verification_positions = if accelerator_dense_execution {
        host_position_cap.min(accelerator_position_cap)
    } else {
        host_position_cap
    };
    let (host_dense_working, accelerator_dense_working) =
        dense_working_set(manifest, dense_staging, verification_positions.max(1));
    let (host_expert_working, accelerator_expert_working) =
        expert_working_set(manifest, max_expert, verification_positions.max(1));
    let host_staging = host_expert_working.max(host_dense_working);
    let accelerator_expert_kernel = matches!(
        manifest.architecture.expert_format,
        crate::WeightFormat::Bf16 | crate::WeightFormat::PackedInt4Group32
    );
    let accelerator_expert_execution = accelerator_expert_kernel
        && max_expert > 0
        && accelerator_expert_working <= accelerator_usable;
    let accelerator_staging = match (accelerator_expert_execution, accelerator_dense_execution) {
        (true, true) => {
            accelerator_expert_working.max(accelerator_dense_working.min(accelerator_usable))
        }
        (true, false) => accelerator_expert_working,
        (false, true) => accelerator_dense_working.min(accelerator_usable),
        (false, false) => 0,
    };
    let ram_after_staging = ram_usable.saturating_sub(host_staging);
    let accelerator_after_staging = accelerator_usable.saturating_sub(accelerator_staging);
    let kv_per_token = kv_bytes_per_token(manifest, kv_scalar_bytes)?;
    let mut notes = Vec::new();
    if verification_positions < requested_verification_positions {
        notes.push(format!(
            "speculative verification reduced from {requested_verification_positions} to {verification_positions} positions to stay inside working-memory limits"
        ));
    }
    if !accelerator_dense_execution {
        notes.push(if accelerator_dense_kernel {
            "dense verification tiles do not fit the accelerator working set; CPU dense fallback required"
                .into()
        } else {
            "dense tensor dtype has no exact accelerator kernel; bounded CPU dense fallback required"
                .into()
        });
    }

    let (kv_tier, effective_context, kv_budget) = if let Some(per_token) = kv_per_token {
        let requested = kv_size(per_token, requested_context_tokens)?;
        // Compact KV belongs on the accelerator when it consumes at most half
        // of the post-staging capacity; otherwise preserve VRAM for hot experts.
        if accelerator_kv_supported && requested <= accelerator_after_staging / 2 {
            (KvTier::Accelerator, requested_context_tokens, requested)
        } else if requested <= ram_after_staging {
            notes.push(if accelerator_kv_supported {
                "KV cache moved to RAM to preserve accelerator expert residency".into()
            } else {
                "KV cache is bounded in RAM because this executor has no device-resident KV kernel"
                    .into()
            });
            (KvTier::Ram, requested_context_tokens, requested)
        } else {
            let accelerator_context = kv_context_capacity(per_token, accelerator_after_staging);
            let ram_context = kv_context_capacity(per_token, ram_after_staging);
            if ram_context >= accelerator_context && ram_context > 0 {
                notes.push(format!(
                    "context reduced from {requested_context_tokens} to {ram_context} tokens to stay inside RAM"
                ));
                (KvTier::Ram, ram_context, kv_size(per_token, ram_context)?)
            } else if accelerator_kv_supported && accelerator_context > 0 {
                notes.push(format!(
                    "context reduced from {requested_context_tokens} to {accelerator_context} tokens to stay inside VRAM"
                ));
                (
                    KvTier::Accelerator,
                    accelerator_context,
                    kv_size(per_token, accelerator_context)?,
                )
            } else {
                notes.push("no memory remains for even one KV-cache token".into());
                (KvTier::Unavailable, 0, 0)
            }
        }
    } else {
        notes.push("attention contract is incomplete, so KV memory cannot be bounded yet".into());
        (KvTier::Unavailable, 0, 0)
    };

    if !accelerator_expert_execution {
        notes.push(if accelerator_expert_kernel {
            "largest expert does not fit the accelerator staging envelope; CPU fallback required"
                .into()
        } else {
            "expert dtype has no exact accelerator kernel; bounded CPU expert fallback required"
                .into()
        });
    }
    if host_staging > ram_usable {
        notes.push("largest expert/dense tile does not fit the host staging envelope".into());
    }
    let ram_kv = if kv_tier == KvTier::Ram { kv_budget } else { 0 };
    let accelerator_kv = if kv_tier == KvTier::Accelerator {
        kv_budget
    } else {
        0
    };
    let ram_cache = ram_usable
        .saturating_sub(host_staging)
        .saturating_sub(ram_kv);
    let accelerator_cache = accelerator_usable
        .saturating_sub(accelerator_staging)
        .saturating_sub(accelerator_kv);
    let feasible = host_staging <= ram_usable
        && verification_positions > 0
        && effective_context > 0
        && kv_tier != KvTier::Unavailable;

    Ok(MemoryEnvelope {
        ram_limit_bytes,
        accelerator_limit_bytes,
        ram_runtime_reserve_bytes: ram_reserve,
        accelerator_runtime_reserve_bytes: accelerator_reserve,
        host_staging_bytes: host_staging,
        accelerator_staging_bytes: accelerator_staging,
        ram_cache_bytes: ram_cache,
        accelerator_cache_bytes: accelerator_cache,
        kv_tier,
        kv_bytes_per_token: kv_per_token,
        kv_budget_bytes: kv_budget,
        requested_context_tokens,
        effective_context_tokens: effective_context,
        context_clamped: effective_context < requested_context_tokens,
        requested_verification_positions,
        max_verification_positions: verification_positions,
        verification_clamped: verification_positions < requested_verification_positions,
        accelerator_dense_execution,
        accelerator_expert_execution,
        feasible,
        notes,
    })
}

/// Peak simultaneous bytes for one tiled matrix operation. CPU fallback owns
/// the original BF16 tile plus its FP32 expansion, input, and output. CUDA
/// owns the device tile, FP32 input, and FP32 output while the host tile is
/// accounted separately in the RAM envelope.
fn dense_working_set(manifest: &RuntimeManifest, tile_budget: u64, positions: u64) -> (u64, u64) {
    let floating_peak = manifest
        .storage
        .dense_tensors
        .iter()
        .filter_map(|tensor| {
            if tensor.shape.len() != 2 || tensor.shape[0] == 0 || tensor.shape[1] == 0 {
                return None;
            }
            let scalar_bytes = match tensor.dtype.as_deref() {
                Some("BF16" | "F16") => 2_u64,
                Some("F32") => 4_u64,
                _ => return None,
            };
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
            let row_bytes = cols.checked_mul(scalar_bytes)?;
            let tile = tile_budget.min(tensor.length).max(row_bytes);
            let tile_rows = (tile / row_bytes).max(1).min(rows);
            let actual_tile = tile_rows.checked_mul(row_bytes)?;
            let input = cols.checked_mul(4)?.checked_mul(positions)?;
            let output = tile_rows.checked_mul(4)?.checked_mul(positions)?;
            let expanded = if scalar_bytes == 2 {
                actual_tile.checked_mul(2)?
            } else {
                actual_tile
            };
            let host = actual_tile
                .checked_add(expanded)?
                .checked_add(input)?
                .checked_add(output)?;
            let accelerator = actual_tile.checked_add(input)?.checked_add(output)?;
            Some((host, accelerator))
        })
        .fold((tile_budget, tile_budget), |peak, next| {
            (peak.0.max(next.0), peak.1.max(next.1))
        });
    let dense_by_name: std::collections::HashMap<_, _> = manifest
        .storage
        .dense_tensors
        .iter()
        .map(|tensor| (tensor.tensor.as_str(), tensor))
        .collect();
    let packed_peak = manifest
        .storage
        .dense_tensors
        .iter()
        .filter_map(|packed| {
            let prefix = packed.tensor.strip_suffix(".weight_packed")?;
            if packed.dtype.as_deref() != Some("I32")
                || packed.shape.len() != 2
                || packed.shape[0] == 0
                || packed.shape[1] == 0
            {
                return None;
            }
            let scales = dense_by_name.get(format!("{prefix}.weight_scale").as_str())?;
            let rows = packed.shape[0];
            let words_per_row = packed.shape[1];
            let cols = words_per_row.checked_mul(8)?;
            let groups_per_row = cols / 32;
            if !cols.is_multiple_of(32)
                || scales.dtype.as_deref() != Some("BF16")
                || scales.shape != [rows, groups_per_row]
            {
                return None;
            }
            let packed_row = words_per_row.checked_mul(4)?;
            let scale_row = groups_per_row.checked_mul(2)?;
            let row_bytes = packed_row.checked_add(scale_row)?;
            let total = packed.length.checked_add(scales.length)?;
            let tile = tile_budget.min(total).max(row_bytes);
            let tile_rows = (tile / row_bytes).max(1).min(rows);
            let actual_tile = tile_rows.checked_mul(row_bytes)?;
            let input = cols.checked_mul(4)?.checked_mul(positions)?;
            let output = tile_rows.checked_mul(4)?.checked_mul(positions)?;
            let working = actual_tile.checked_add(input)?.checked_add(output)?;
            Some((working, working))
        })
        .fold((tile_budget, tile_budget), |peak, next| {
            (peak.0.max(next.0), peak.1.max(next.1))
        });
    let peak = (
        floating_peak.0.max(packed_peak.0),
        floating_peak.1.max(packed_peak.1),
    );
    // Dense kernels return host FP32 vectors. Q/K/V and gated-MLP
    // activations remain live across subsequent projections, so their retained
    // buffers must be counted in addition to the current matrix tile.
    let attention = &manifest.architecture.attention_config;
    let query_width = u64::from(attention.heads).saturating_mul(u64::from(attention.head_dim));
    let kv_width = u64::from(attention.kv_heads).saturating_mul(u64::from(attention.head_dim));
    let attention_values = query_width
        .saturating_mul(2)
        .saturating_add(kv_width.saturating_mul(2));
    let dense_intermediate = manifest
        .storage
        .dense_tensors
        .iter()
        .filter(|tensor| {
            tensor.shape.len() == 2
                && !tensor.tensor.contains(".experts.")
                && (tensor.tensor.ends_with(".gate_proj.weight")
                    || tensor.tensor.ends_with(".up_proj.weight")
                    || tensor.tensor.ends_with(".gate_proj.weight_packed")
                    || tensor.tensor.ends_with(".up_proj.weight_packed"))
        })
        .map(|tensor| tensor.shape[0])
        .max()
        .unwrap_or(0);
    let hidden = u64::from(manifest.architecture.hidden_size);
    let mlp_values = dense_intermediate
        .saturating_mul(2)
        // Residual, routed/shared result, and output may overlap during a
        // shared-expert contribution.
        .saturating_add(hidden.saturating_mul(3));
    let retained_host = positions
        .saturating_mul(attention_values.max(mlp_values))
        .saturating_mul(4);
    (peak.0.saturating_add(retained_host), peak.1)
}

/// Peak routed-expert working set for a gathered speculative batch. The host
/// owns one immutable expert payload plus gathered input, gate/up activation,
/// wave accumulation, and output vectors. A resident accelerator expert owns
/// the payload plus one projection's FP32 input/output buffers.
fn expert_working_set(
    manifest: &RuntimeManifest,
    expert_payload_bytes: u64,
    positions: u64,
) -> (u64, u64) {
    let hidden = u64::from(manifest.architecture.hidden_size);
    let intermediate = u64::from(manifest.architecture.expert_intermediate_size);
    let host_values = hidden
        .saturating_mul(3)
        .saturating_add(intermediate.saturating_mul(2));
    let host = expert_payload_bytes
        .saturating_add(positions.saturating_mul(host_values).saturating_mul(4));
    let device_values = hidden.saturating_add(intermediate);
    let accelerator = expert_payload_bytes
        .saturating_add(positions.saturating_mul(device_values).saturating_mul(4));
    (host, accelerator)
}

fn max_fitting_positions(requested: u64, budget: u64, working_bytes: impl Fn(u64) -> u64) -> u64 {
    let mut low = 0_u64;
    let mut high = requested;
    while low < high {
        let distance = high - low;
        let middle = low + distance / 2 + distance % 2;
        if working_bytes(middle) <= budget {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn reserve(limit: u64, minimum: u64, maximum: u64) -> u64 {
    (limit / 10).max(minimum).min(maximum).min(limit / 4)
}

fn kv_bytes_per_token(
    manifest: &RuntimeManifest,
    scalar_bytes: u64,
) -> Result<Option<u64>, ResourceError> {
    let architecture = &manifest.architecture;
    let attention = &architecture.attention_config;
    let layers = u64::from(architecture.num_layers);
    let elements = if architecture.attention == AttentionKind::Mla && attention.kv_lora_rank > 0 {
        u64::from(attention.kv_lora_rank)
            .checked_add(u64::from(attention.qk_rope_head_dim))
            .ok_or(ResourceError::Overflow)?
    } else if attention.kv_heads > 0 && attention.head_dim > 0 {
        u64::from(attention.kv_heads)
            .checked_mul(u64::from(attention.head_dim))
            .and_then(|value| value.checked_mul(2))
            .ok_or(ResourceError::Overflow)?
    } else {
        return Ok(None);
    };
    layers
        .checked_mul(elements)
        .and_then(|value| value.checked_mul(scalar_bytes))
        .map(Some)
        .ok_or(ResourceError::Overflow)
}

fn kv_size(per_token: u64, tokens: u64) -> Result<u64, ResourceError> {
    per_token
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(11))
        .map(|value| value.div_ceil(10))
        .ok_or(ResourceError::Overflow)
}

fn kv_context_capacity(per_token: u64, bytes: u64) -> u64 {
    bytes.saturating_mul(10) / per_token.saturating_mul(11).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::CURRENT_SCHEMA;
    use crate::{
        ActivationKind, ArchitectureContract, AttentionContract, ExpertLocation,
        ExpertStorageManifest, ExpertTensorLayout, ModelFamily, QuantizationContract,
        RouterContract, RouterSemantics, TensorSegment, WeightFormat,
    };

    fn fixture() -> RuntimeManifest {
        RuntimeManifest {
            schema_version: CURRENT_SCHEMA,
            architecture: ArchitectureContract {
                adapter: "sytra-test".into(),
                model_type: "test".into(),
                attention: AttentionKind::Mla,
                router: RouterSemantics::NoAuxTc,
                expert_format: WeightFormat::Int4Group,
                family: ModelFamily::Generic,
                expert_layout: ExpertTensorLayout::Discrete,
                activation: ActivationKind::Silu,
                router_config: RouterContract::default(),
                attention_config: AttentionContract {
                    kv_lora_rank: 512,
                    qk_rope_head_dim: 64,
                    ..AttentionContract::default()
                },
                quantization: QuantizationContract::default(),
                hidden_size: 7168,
                expert_intermediate_size: 2048,
                moe_layers: vec![1],
                num_layers: 61,
                experts_per_layer: 384,
                experts_per_token: 8,
                forward_verified: false,
            },
            dense_bytes: 1024 * MIB,
            storage: ExpertStorageManifest {
                contiguous_experts: true,
                experts: vec![ExpertLocation {
                    layer: 1,
                    expert: 0,
                    segments: vec![TensorSegment {
                        tensor: "gate".into(),
                        dtype: Some("I32".into()),
                        shape: vec![],
                        shard: "model.safetensors".into(),
                        offset: 0,
                        length: 24 * MIB,
                    }],
                }],
                dense_tensors: vec![TensorSegment {
                    tensor: "embed".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![],
                    shard: "model.safetensors".into(),
                    offset: 0,
                    length: 2048 * MIB,
                }],
            },
        }
    }

    #[test]
    fn low_end_envelope_keeps_every_partition_inside_the_cap() {
        let plan = plan_memory_envelope(
            &fixture(),
            4096,
            16 * 1024 * MIB,
            12 * 1024 * MIB,
            2,
            64 * MIB,
            8,
        )
        .unwrap();
        assert!(plan.feasible);
        assert_eq!(plan.effective_context_tokens, 4096);
        assert_eq!(plan.kv_tier, KvTier::Ram);
        assert!(
            plan.ram_runtime_reserve_bytes + plan.host_staging_bytes + plan.ram_cache_bytes
                <= plan.ram_limit_bytes
        );
        assert!(
            plan.accelerator_runtime_reserve_bytes
                + plan.accelerator_staging_bytes
                + plan.accelerator_cache_bytes
                <= plan.accelerator_limit_bytes
        );
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("no device-resident KV")));
    }

    #[test]
    fn standard_attention_uses_compact_device_kv_only_when_capability_is_enabled() {
        let mut manifest = fixture();
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.num_layers = 2;
        manifest.architecture.attention_config = AttentionContract {
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            ..AttentionContract::default()
        };
        let plan = plan_memory_envelope_with_capabilities(
            &manifest,
            128,
            2 * 1024 * MIB,
            1024 * MIB,
            2,
            MIB,
            4,
            true,
        )
        .unwrap();
        assert!(plan.feasible);
        assert_eq!(plan.kv_tier, KvTier::Accelerator);
        assert_eq!(plan.kv_bytes_per_token, Some(256));
        assert!(plan.kv_budget_bytes >= 128 * 256);
        assert!(
            plan.accelerator_runtime_reserve_bytes
                + plan.accelerator_staging_bytes
                + plan.accelerator_cache_bytes
                + plan.kv_budget_bytes
                <= plan.accelerator_limit_bytes
        );
    }

    #[test]
    fn f32_reference_weights_report_cpu_compute_while_device_kv_remains_available() {
        let mut manifest = fixture();
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.expert_format = WeightFormat::F32;
        manifest.architecture.num_layers = 2;
        manifest.architecture.attention_config = AttentionContract {
            heads: 4,
            kv_heads: 2,
            head_dim: 16,
            ..AttentionContract::default()
        };
        manifest.storage.dense_tensors[0].dtype = Some("F32".into());
        manifest.storage.dense_tensors[0].shape = vec![32, 32];
        manifest.storage.dense_tensors[0].length = 4096;
        manifest.storage.experts[0].segments[0].dtype = Some("F32".into());
        let plan = plan_memory_envelope_with_capabilities(
            &manifest,
            128,
            2 * 1024 * MIB,
            1024 * MIB,
            2,
            MIB,
            4,
            true,
        )
        .unwrap();
        assert_eq!(plan.kv_tier, KvTier::Accelerator);
        assert!(!plan.accelerator_dense_execution);
        assert!(!plan.accelerator_expert_execution);
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("dense tensor dtype")));
        assert!(plan.notes.iter().any(|note| note.contains("expert dtype")));
    }

    #[test]
    fn context_is_clamped_instead_of_overcommitting_memory() {
        let plan =
            plan_memory_envelope(&fixture(), 1_000_000, 1024 * MIB, 512 * MIB, 2, 64 * MIB, 8)
                .unwrap();
        assert!(plan.context_clamped);
        assert!(plan.effective_context_tokens < 1_000_000);
        assert!(plan.effective_context_tokens > 0);
    }

    #[test]
    fn speculative_batch_is_clamped_to_simultaneous_dense_buffers() {
        let mut manifest = fixture();
        manifest.storage.dense_tensors[0].shape = vec![1024, 1024];
        manifest.storage.dense_tensors[0].length = 2 * MIB;
        manifest.dense_bytes = 2 * MIB;
        let plan = plan_memory_envelope(&manifest, 128, 512 * MIB, 256 * MIB, 2, 2 * MIB, 100_000)
            .unwrap();
        assert!(plan.verification_clamped);
        assert!(plan.max_verification_positions > 0);
        assert!(plan.max_verification_positions < 100_000);
        assert!(
            plan.ram_runtime_reserve_bytes + plan.host_staging_bytes + plan.ram_cache_bytes
                <= plan.ram_limit_bytes
        );
        assert!(
            plan.accelerator_runtime_reserve_bytes
                + plan.accelerator_staging_bytes
                + plan.accelerator_cache_bytes
                + if plan.kv_tier == KvTier::Accelerator {
                    plan.kv_budget_bytes
                } else {
                    0
                }
                <= plan.accelerator_limit_bytes
        );
    }

    #[test]
    fn packed_dense_working_set_counts_scale_tiles_and_retained_mlp_outputs() {
        let mut manifest = fixture();
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.hidden_size = 32;
        manifest.architecture.num_layers = 1;
        manifest.architecture.attention_config = AttentionContract {
            heads: 4,
            kv_heads: 1,
            head_dim: 8,
            ..AttentionContract::default()
        };
        manifest.storage.dense_tensors = vec![
            TensorSegment {
                tensor: "model.layers.0.mlp.gate_proj.weight_packed".into(),
                dtype: Some("I32".into()),
                shape: vec![64, 4],
                shard: "model.safetensors".into(),
                offset: 0,
                length: 1024,
            },
            TensorSegment {
                tensor: "model.layers.0.mlp.gate_proj.weight_scale".into(),
                dtype: Some("BF16".into()),
                shape: vec![64, 1],
                shard: "model.safetensors".into(),
                offset: 1024,
                length: 128,
            },
        ];
        assert_eq!(dense_working_set(&manifest, 36, 2), (2100, 308));
    }

    #[test]
    fn verification_clamp_is_logarithmic_for_untrusted_large_requests() {
        let mut manifest = fixture();
        manifest.storage.dense_tensors[0].shape = vec![1024, 1024];
        manifest.storage.dense_tensors[0].length = 2 * MIB;
        manifest.dense_bytes = 2 * MIB;
        let plan = plan_memory_envelope(&manifest, 128, 512 * MIB, 256 * MIB, 2, 2 * MIB, u64::MAX)
            .unwrap();
        assert!(plan.verification_clamped);
        assert!(plan.max_verification_positions > 0);
        assert!(plan.max_verification_positions < u64::MAX);
    }
}
