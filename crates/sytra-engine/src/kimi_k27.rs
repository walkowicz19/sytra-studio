//! Exact, checkpoint-independent reference primitives for Kimi K2.7 Code.
//!
//! These routines intentionally favor clarity and reproducibility over speed.
//! CUDA kernels are checked against them before an adapter can become serving
//! capable. The streamed routed-expert payload stays in its original
//! compressed-tensors packed INT4 representation.

use std::{collections::HashMap, fs, path::Path};

use serde::Serialize;
use serde_json::Value;
use thiserror::Error;

use crate::manifest::{
    ActivationKind, AttentionKind, ExpertLocation, RouterSemantics, RuntimeManifest, TensorSegment,
    WeightFormat,
};
use crate::moe::{
    apply_activation, decode_float_values, rms_norm, route_topk_logits, softmax_in_place,
    MoeMathError,
};
use crate::{
    bf16_tile_cpu, bf16_tile_matmul_cpu, bf16_transpose_tile_cpu, bf16_transpose_tile_matmul_cpu,
    tiled_bf16_matmul, tiled_bf16_matmul_rows, tiled_bf16_matvec, tiled_bf16_matvec_rows,
    tiled_bf16_transpose_matmul_rows, tiled_bf16_transpose_matvec_rows, CudaAccelerator,
    DenseExecutionError, DenseTensorStore, DenseTileMetrics, ResidentExpert, Route, RoutingBatch,
    SchedulerError, StreamingScheduler, TensorStoreError, TiledMatmulOutput,
};

#[cfg(test)]
use crate::manifest::{RouterContract, RouterScoreKind};
#[cfg(test)]
use crate::moe::route_topk;

pub const ADAPTER_ID: &str = "sytra-kimi-k2.7-code";
pub const OUTER_MODEL_TYPE: &str = "kimi_k25";
pub const TEXT_MODEL_TYPE: &str = "kimi_k2";

#[derive(Debug, Clone, PartialEq)]
pub struct KimiK27Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub vocab_size: usize,
    pub group_size: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub rope_mscale: f32,
    pub rope_mscale_all_dim: f32,
    pub routed_scaling_factor: f32,
    pub rms_norm_eps: f32,
}

impl KimiK27Config {
    pub fn official() -> Self {
        Self {
            hidden_size: 7168,
            intermediate_size: 18432,
            moe_intermediate_size: 2048,
            num_hidden_layers: 61,
            num_attention_heads: 64,
            n_routed_experts: 384,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            first_k_dense_replace: 1,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            n_group: 1,
            topk_group: 1,
            max_position_embeddings: 262_144,
            original_max_position_embeddings: 4096,
            vocab_size: 163_840,
            group_size: 32,
            rope_theta: 50_000.0,
            rope_factor: 64.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            rope_mscale: 1.0,
            rope_mscale_all_dim: 1.0,
            routed_scaling_factor: 2.827,
            rms_norm_eps: 1e-5,
        }
    }

    pub fn load(model_root: impl AsRef<Path>) -> Result<Self, KimiError> {
        let path = model_root.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|source| KimiError::ConfigRead {
            path: path.display().to_string(),
            source,
        })?;
        let root: Value =
            serde_json::from_slice(&bytes).map_err(|source| KimiError::ConfigJson {
                path: path.display().to_string(),
                source,
            })?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &Value) -> Result<Self, KimiError> {
        let text = root
            .get("text_config")
            .and_then(Value::as_object)
            .ok_or_else(|| KimiError::Contract("text_config is missing".into()))?;
        expect_str(root, "model_type", OUTER_MODEL_TYPE)?;
        expect_str_obj(text, "model_type", TEXT_MODEL_TYPE)?;
        expect_str_obj(text, "hidden_act", "silu")?;
        expect_str_obj(text, "scoring_func", "sigmoid")?;
        expect_str_obj(text, "topk_method", "noaux_tc")?;
        expect_bool_obj(text, "norm_topk_prob", true)?;
        expect_bool_obj(text, "attention_bias", false)?;

        let quant = text
            .get("quantization_config")
            .and_then(Value::as_object)
            .ok_or_else(|| KimiError::Contract("quantization_config is missing".into()))?;
        expect_str_obj(quant, "format", "pack-quantized")?;
        expect_str_obj(quant, "quant_method", "compressed-tensors")?;
        let weights = quant
            .get("config_groups")
            .and_then(|value| value.get("group_0"))
            .and_then(|value| value.get("weights"))
            .and_then(Value::as_object)
            .ok_or_else(|| KimiError::Contract("group_0.weights is missing".into()))?;
        expect_u64_obj(weights, "num_bits", 4)?;
        expect_u64_obj(weights, "group_size", 32)?;
        expect_str_obj(weights, "strategy", "group")?;
        expect_str_obj(weights, "type", "int")?;
        expect_bool_obj(weights, "symmetric", true)?;

        let rope = text
            .get("rope_scaling")
            .and_then(Value::as_object)
            .ok_or_else(|| KimiError::Contract("rope_scaling is missing".into()))?;
        expect_str_obj(rope, "type", "yarn")?;

        let parsed = Self {
            hidden_size: usize_field(text, "hidden_size")?,
            intermediate_size: usize_field(text, "intermediate_size")?,
            moe_intermediate_size: usize_field(text, "moe_intermediate_size")?,
            num_hidden_layers: usize_field(text, "num_hidden_layers")?,
            num_attention_heads: usize_field(text, "num_attention_heads")?,
            n_routed_experts: usize_field(text, "n_routed_experts")?,
            n_shared_experts: usize_field(text, "n_shared_experts")?,
            num_experts_per_tok: usize_field(text, "num_experts_per_tok")?,
            first_k_dense_replace: usize_field(text, "first_k_dense_replace")?,
            q_lora_rank: usize_field(text, "q_lora_rank")?,
            kv_lora_rank: usize_field(text, "kv_lora_rank")?,
            qk_nope_head_dim: usize_field(text, "qk_nope_head_dim")?,
            qk_rope_head_dim: usize_field(text, "qk_rope_head_dim")?,
            v_head_dim: usize_field(text, "v_head_dim")?,
            n_group: usize_field(text, "n_group")?,
            topk_group: usize_field(text, "topk_group")?,
            max_position_embeddings: usize_field(text, "max_position_embeddings")?,
            original_max_position_embeddings: usize_field(
                rope,
                "original_max_position_embeddings",
            )?,
            vocab_size: usize_field(text, "vocab_size")?,
            group_size: usize_field(weights, "group_size")?,
            rope_theta: f32_field(text, "rope_theta")?,
            rope_factor: f32_field(rope, "factor")?,
            beta_fast: f32_field(rope, "beta_fast")?,
            beta_slow: f32_field(rope, "beta_slow")?,
            rope_mscale: f32_field(rope, "mscale")?,
            rope_mscale_all_dim: f32_field(rope, "mscale_all_dim")?,
            routed_scaling_factor: f32_field(text, "routed_scaling_factor")?,
            rms_norm_eps: f32_field(text, "rms_norm_eps")?,
        };
        if parsed != Self::official() {
            return Err(KimiError::Contract(format!(
                "checkpoint dimensions differ from the compiled Kimi K2.7 Code contract: {parsed:?}"
            )));
        }
        Ok(parsed)
    }

    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Bytes in the compact per-layer MLA cache for one token.
    ///
    /// The official eager implementation materializes all heads' K/V values.
    /// Sytra keeps the pre-`kv_b_proj` latent plus the shared rotated PE key,
    /// then reconstructs heads inside the attention kernel.
    pub fn compressed_kv_bytes_per_token(&self, scalar_bytes: usize) -> usize {
        (self.kv_lora_rank + self.qk_rope_head_dim) * scalar_bytes
    }

    pub fn materialized_kv_bytes_per_token(&self, scalar_bytes: usize) -> usize {
        self.num_attention_heads * (self.q_head_dim() + self.v_head_dim) * scalar_bytes
    }

    pub fn validate_runtime_manifest(&self, manifest: &RuntimeManifest) -> Result<(), KimiError> {
        let architecture = &manifest.architecture;
        if architecture.adapter != ADAPTER_ID
            || architecture.model_type != OUTER_MODEL_TYPE
            || architecture.attention != AttentionKind::Mla
            || architecture.router != RouterSemantics::GroupLimitedTopK
            || architecture.expert_format != WeightFormat::PackedInt4Group32
            || architecture.num_layers as usize != self.num_hidden_layers
            || architecture.experts_per_layer as usize != self.n_routed_experts
            || architecture.experts_per_token as usize != self.num_experts_per_tok
        {
            return Err(KimiError::Contract(
                "runtime manifest does not match the exact Kimi K2.7 architecture".into(),
            ));
        }
        let expected_count =
            (self.num_hidden_layers - self.first_k_dense_replace) * self.n_routed_experts;
        if manifest.storage.experts.len() != expected_count {
            return Err(KimiError::Contract(format!(
                "expert index contains {} entries; Kimi K2.7 requires {expected_count}",
                manifest.storage.experts.len()
            )));
        }
        let mut entries: Vec<_> = manifest.storage.experts.iter().collect();
        entries.sort_by_key(|entry| (entry.layer, entry.expert));
        for layer in self.first_k_dense_replace..self.num_hidden_layers {
            for expert in 0..self.n_routed_experts {
                let index = (layer - self.first_k_dense_replace) * self.n_routed_experts + expert;
                let entry = entries[index];
                if entry.layer as usize != layer || entry.expert as usize != expert {
                    return Err(KimiError::Contract(format!(
                        "expert index is missing layer {layer} expert {expert}"
                    )));
                }
                validate_expert_segments(entry, self)?;
            }
        }
        self.validate_dense_tensors(manifest)?;
        Ok(())
    }

    /// Validate the complete text-tower tensor binding before any checkpoint
    /// bytes are executed. The multimodal wrapper may contain additional
    /// vision tensors; those are intentionally ignored by text-only serving.
    pub fn validate_dense_tensors(&self, manifest: &RuntimeManifest) -> Result<(), KimiError> {
        let tensors: HashMap<_, _> = manifest
            .storage
            .dense_tensors
            .iter()
            .map(|tensor| (tensor.tensor.as_str(), tensor))
            .collect();
        let bf16 = &["BF16"];
        expect_dense(
            &tensors,
            "language_model.model.embed_tokens.weight",
            bf16,
            &[self.vocab_size as u64, self.hidden_size as u64],
        )?;
        expect_dense(
            &tensors,
            "language_model.model.norm.weight",
            bf16,
            &[self.hidden_size as u64],
        )?;
        expect_dense(
            &tensors,
            "language_model.lm_head.weight",
            bf16,
            &[self.vocab_size as u64, self.hidden_size as u64],
        )?;

        let heads = self.num_attention_heads as u64;
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("language_model.model.layers.{layer}");
            for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                expect_dense(
                    &tensors,
                    &format!("{prefix}.{norm}"),
                    bf16,
                    &[self.hidden_size as u64],
                )?;
            }
            let attention = format!("{prefix}.self_attn");
            for (suffix, shape) in [
                (
                    "q_a_proj.weight",
                    vec![self.q_lora_rank as u64, self.hidden_size as u64],
                ),
                ("q_a_layernorm.weight", vec![self.q_lora_rank as u64]),
                (
                    "q_b_proj.weight",
                    vec![heads * self.q_head_dim() as u64, self.q_lora_rank as u64],
                ),
                (
                    "kv_a_proj_with_mqa.weight",
                    vec![
                        (self.kv_lora_rank + self.qk_rope_head_dim) as u64,
                        self.hidden_size as u64,
                    ],
                ),
                ("kv_a_layernorm.weight", vec![self.kv_lora_rank as u64]),
                (
                    "kv_b_proj.weight",
                    vec![
                        heads * (self.qk_nope_head_dim + self.v_head_dim) as u64,
                        self.kv_lora_rank as u64,
                    ],
                ),
                (
                    "o_proj.weight",
                    vec![self.hidden_size as u64, heads * self.v_head_dim as u64],
                ),
            ] {
                expect_dense(&tensors, &format!("{attention}.{suffix}"), bf16, &shape)?;
            }

            let mlp = format!("{prefix}.mlp");
            if layer < self.first_k_dense_replace {
                for projection in ["gate_proj.weight", "up_proj.weight"] {
                    expect_dense(
                        &tensors,
                        &format!("{mlp}.{projection}"),
                        bf16,
                        &[self.intermediate_size as u64, self.hidden_size as u64],
                    )?;
                }
                expect_dense(
                    &tensors,
                    &format!("{mlp}.down_proj.weight"),
                    bf16,
                    &[self.hidden_size as u64, self.intermediate_size as u64],
                )?;
            } else {
                expect_dense(
                    &tensors,
                    &format!("{mlp}.gate.weight"),
                    bf16,
                    &[self.n_routed_experts as u64, self.hidden_size as u64],
                )?;
                // The official implementation computes this parameter in
                // FP32, while some converted checkpoints serialize BF16.
                expect_dense(
                    &tensors,
                    &format!("{mlp}.gate.e_score_correction_bias"),
                    &["F32", "BF16"],
                    &[self.n_routed_experts as u64],
                )?;
                let shared = self.moe_intermediate_size * self.n_shared_experts;
                for projection in ["gate_proj.weight", "up_proj.weight"] {
                    expect_dense(
                        &tensors,
                        &format!("{mlp}.shared_experts.{projection}"),
                        bf16,
                        &[shared as u64, self.hidden_size as u64],
                    )?;
                }
                expect_dense(
                    &tensors,
                    &format!("{mlp}.shared_experts.down_proj.weight"),
                    bf16,
                    &[self.hidden_size as u64, shared as u64],
                )?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum KimiError {
    #[error("could not read Kimi config {path}: {source}")]
    ConfigRead {
        path: String,
        source: std::io::Error,
    },
    #[error("Kimi config {path} is invalid JSON: {source}")]
    ConfigJson {
        path: String,
        source: serde_json::Error,
    },
    #[error("Kimi K2.7 contract mismatch: {0}")]
    Contract(String),
    #[error("invalid tensor shape: {0}")]
    Shape(String),
    #[error(transparent)]
    Dense(#[from] DenseExecutionError),
    #[error(transparent)]
    Tensor(#[from] TensorStoreError),
    #[error(transparent)]
    Math(#[from] MoeMathError),
    #[error("streaming scheduler failed: {0}")]
    Scheduler(String),
}

/// Decode compressed-tensors' dense INT4 packing.
///
/// Eight offset-binary nibbles are stored low-to-high in each little-endian
/// int32 word. Returned values are in `[-8, 7]`.
pub fn unpack_int4(packed: &[u32], element_count: usize) -> Result<Vec<i8>, KimiError> {
    let needed = element_count.div_ceil(8);
    if packed.len() < needed {
        return Err(KimiError::Shape(format!(
            "packed INT4 needs {needed} words, received {}",
            packed.len()
        )));
    }
    Ok((0..element_count)
        .map(|index| {
            let nibble = (packed[index / 8] >> ((index % 8) * 4)) & 0x0f;
            nibble as i8 - 8
        })
        .collect())
}

/// Reference groupwise INT4 matrix-vector product.
pub fn int4_group_matvec(
    packed: &[u32],
    scales: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
    input: &[f32],
) -> Result<Vec<f32>, KimiError> {
    if rows == 0 || cols == 0 || group_size == 0 || cols % group_size != 0 || input.len() != cols {
        return Err(KimiError::Shape("invalid INT4 matvec dimensions".into()));
    }
    let words_per_row = cols.div_ceil(8);
    let groups_per_row = cols / group_size;
    if packed.len() != rows * words_per_row || scales.len() != rows * groups_per_row {
        return Err(KimiError::Shape(format!(
            "INT4 payload mismatch: packed={} scales={}, expected {} and {}",
            packed.len(),
            scales.len(),
            rows * words_per_row,
            rows * groups_per_row
        )));
    }
    let mut output = vec![0.0; rows];
    for row in 0..rows {
        let packed_row = &packed[row * words_per_row..(row + 1) * words_per_row];
        let scale_row = &scales[row * groups_per_row..(row + 1) * groups_per_row];
        let mut sum = 0.0;
        for column in 0..cols {
            let nibble = (packed_row[column / 8] >> ((column % 8) * 4)) & 0x0f;
            let quantized = nibble as i8 - 8;
            sum += quantized as f32 * scale_row[column / group_size] * input[column];
        }
        output[row] = sum;
    }
    Ok(output)
}

/// Byte-exact CPU reference for compressed-tensors symmetric packed INT4
/// group-32 weights with BF16 scales and a position-major FP32 input batch.
pub fn int4_group32_bf16_matmul_cpu(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    positions: usize,
    input: &[f32],
) -> Result<Vec<f32>, KimiError> {
    let words_per_row = cols.div_ceil(8);
    let groups_per_row = cols / 32;
    let packed_len = rows
        .checked_mul(words_per_row)
        .and_then(|words| words.checked_mul(size_of::<u32>()))
        .ok_or_else(|| KimiError::Shape("packed INT4 batch size overflow".into()))?;
    let scale_len = rows
        .checked_mul(groups_per_row)
        .and_then(|values| values.checked_mul(size_of::<u16>()))
        .ok_or_else(|| KimiError::Shape("packed INT4 scale size overflow".into()))?;
    if rows == 0
        || cols == 0
        || positions == 0
        || !cols.is_multiple_of(32)
        || input.len() != positions * cols
        || packed.len() != packed_len
        || scales.len() != scale_len
    {
        return Err(KimiError::Shape(
            "invalid packed INT4/BF16 batched matrix dimensions".into(),
        ));
    }
    let mut output = vec![0.0_f32; positions * rows];
    for position in 0..positions {
        let input = &input[position * cols..(position + 1) * cols];
        for row in 0..rows {
            let mut sum = 0.0_f32;
            for (column, value) in input.iter().enumerate() {
                let word_offset = (row * words_per_row + column / 8) * size_of::<u32>();
                let word = u32::from_le_bytes(
                    packed[word_offset..word_offset + size_of::<u32>()]
                        .try_into()
                        .expect("validated packed word range"),
                );
                let nibble = (word >> ((column % 8) * 4)) & 0x0f;
                let scale_offset = (row * groups_per_row + column / 32) * size_of::<u16>();
                let scale = bf16_to_f32(u16::from_le_bytes(
                    scales[scale_offset..scale_offset + size_of::<u16>()]
                        .try_into()
                        .expect("validated BF16 scale range"),
                ));
                sum += (nibble as i8 - 8) as f32 * scale * value;
            }
            output[position * rows + row] = sum;
        }
    }
    Ok(output)
}

pub struct PackedInt4Matrix<'a> {
    pub packed: &'a [u32],
    pub scales: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OwnedPackedInt4Matrix {
    pub packed: Vec<u32>,
    /// Scales expanded from the checkpoint's BF16 representation for the CPU
    /// reference path.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

impl OwnedPackedInt4Matrix {
    pub fn as_reference(&self) -> PackedInt4Matrix<'_> {
        PackedInt4Matrix {
            packed: &self.packed,
            scales: &self.scales,
            rows: self.rows,
            cols: self.cols,
            group_size: self.group_size,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KimiExpertWeights {
    pub gate: OwnedPackedInt4Matrix,
    pub up: OwnedPackedInt4Matrix,
    pub down: OwnedPackedInt4Matrix,
}

impl KimiExpertWeights {
    pub fn from_resident(
        expert: &ResidentExpert,
        config: &KimiK27Config,
    ) -> Result<Self, KimiError> {
        let bytes = expert
            .host_bytes
            .as_ref()
            .ok_or_else(|| KimiError::Shape("CPU parsing needs a host expert lease".into()))?;
        let projection = |name: &str, rows: usize, cols: usize| {
            let packed = resident_tensor(expert, bytes, &format!("{name}.weight_packed"))?;
            let scales = resident_tensor(expert, bytes, &format!("{name}.weight_scale"))?;
            let shape = resident_tensor(expert, bytes, &format!("{name}.weight_shape"))?;
            let logical_shape = decode_i32_le(shape)?;
            if logical_shape != [rows as i32, cols as i32] {
                return Err(KimiError::Shape(format!(
                    "{name}.weight_shape is {logical_shape:?}, expected [{rows}, {cols}]"
                )));
            }
            Ok(OwnedPackedInt4Matrix {
                packed: decode_u32_le(packed)?,
                scales: decode_float_values("BF16", scales)
                    .map_err(|error| KimiError::Shape(error.to_string()))?,
                rows,
                cols,
                group_size: config.group_size,
            })
        };
        Ok(Self {
            gate: projection(
                "gate_proj",
                config.moe_intermediate_size,
                config.hidden_size,
            )?,
            up: projection("up_proj", config.moe_intermediate_size, config.hidden_size)?,
            down: projection(
                "down_proj",
                config.hidden_size,
                config.moe_intermediate_size,
            )?,
        })
    }

    pub fn forward_reference(&self, hidden: &[f32]) -> Result<Vec<f32>, KimiError> {
        expert_swiglu(
            hidden,
            self.gate.as_reference(),
            self.up.as_reference(),
            self.down.as_reference(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct BorrowedPackedInt4Matrix<'a> {
    packed: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    cols: usize,
}

impl BorrowedPackedInt4Matrix<'_> {
    fn matmul(&self, positions: usize, input: &[f32]) -> Result<Vec<f32>, KimiError> {
        int4_group32_bf16_matmul_cpu(
            self.packed,
            self.scales,
            self.rows,
            self.cols,
            positions,
            input,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct BorrowedKimiExpertWeights<'a> {
    gate: BorrowedPackedInt4Matrix<'a>,
    up: BorrowedPackedInt4Matrix<'a>,
    down: BorrowedPackedInt4Matrix<'a>,
}

impl<'a> BorrowedKimiExpertWeights<'a> {
    fn from_resident(
        expert: &'a ResidentExpert,
        config: &KimiK27Config,
    ) -> Result<Self, KimiError> {
        let bytes = expert
            .host_bytes
            .as_deref()
            .ok_or_else(|| KimiError::Shape("host expert payload is unavailable".into()))?;
        let projection = |name: &str, rows: usize, cols: usize| {
            let packed = resident_tensor(expert, bytes, &format!("{name}.weight_packed"))?;
            let scales = resident_tensor(expert, bytes, &format!("{name}.weight_scale"))?;
            let shape = decode_i32_le(resident_tensor(
                expert,
                bytes,
                &format!("{name}.weight_shape"),
            )?)?;
            if shape != [rows as i32, cols as i32] {
                return Err(KimiError::Shape(format!(
                    "{name}.weight_shape is {shape:?}, expected [{rows}, {cols}]"
                )));
            }
            Ok(BorrowedPackedInt4Matrix {
                packed,
                scales,
                rows,
                cols,
            })
        };
        Ok(Self {
            gate: projection(
                "gate_proj",
                config.moe_intermediate_size,
                config.hidden_size,
            )?,
            up: projection("up_proj", config.moe_intermediate_size, config.hidden_size)?,
            down: projection(
                "down_proj",
                config.hidden_size,
                config.moe_intermediate_size,
            )?,
        })
    }

    fn forward_batch(&self, positions: usize, hidden: &[f32]) -> Result<Vec<f32>, KimiError> {
        let mut gate = self.gate.matmul(positions, hidden)?;
        let up = self.up.matmul(positions, hidden)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = apply_activation(ActivationKind::Silu, *gate) * up;
        }
        self.down.matmul(positions, &gate)
    }
}

impl PackedInt4Matrix<'_> {
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, KimiError> {
        int4_group_matvec(
            self.packed,
            self.scales,
            self.rows,
            self.cols,
            self.group_size,
            input,
        )
    }
}

pub fn expert_swiglu(
    hidden: &[f32],
    gate: PackedInt4Matrix<'_>,
    up: PackedInt4Matrix<'_>,
    down: PackedInt4Matrix<'_>,
) -> Result<Vec<f32>, KimiError> {
    let mut gated = gate.matvec(hidden)?;
    let up = up.matvec(hidden)?;
    if gated.len() != up.len() || down.cols != gated.len() {
        return Err(KimiError::Shape("incompatible expert projections".into()));
    }
    for (gate, up) in gated.iter_mut().zip(up) {
        *gate = apply_activation(ActivationKind::Silu, *gate) * up;
    }
    down.matvec(&gated)
}

pub fn yarn_inverse_frequencies(config: &KimiK27Config) -> Vec<f32> {
    let dim = config.qk_rope_head_dim;
    let correction = |rotations: f32| {
        dim as f32
            * (config.original_max_position_embeddings as f32
                / (rotations * 2.0 * std::f32::consts::PI))
                .ln()
            / (2.0 * config.rope_theta.ln())
    };
    let low = correction(config.beta_fast).floor().max(0.0) as usize;
    let high = (correction(config.beta_slow).ceil() as usize).min(dim - 1);
    (0..dim / 2)
        .map(|index| {
            let exponent = (2 * index) as f32 / dim as f32;
            let extra = 1.0 / config.rope_theta.powf(exponent);
            let inter = extra / config.rope_factor;
            let ramp = if low == high {
                ((index as f32 - low as f32) / 0.001).clamp(0.0, 1.0)
            } else {
                ((index as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0)
            };
            let inv_freq_mask = 1.0 - ramp;
            inter * (1.0 - inv_freq_mask) + extra * inv_freq_mask
        })
        .collect()
}

/// Apply the pair-to-half permutation and YaRN rotation used in the official
/// Kimi implementation to one RoPE head.
pub fn apply_yarn_rope(
    input: &[f32],
    position: usize,
    config: &KimiK27Config,
) -> Result<Vec<f32>, KimiError> {
    let dim = config.qk_rope_head_dim;
    if input.len() != dim || dim % 2 != 0 {
        return Err(KimiError::Shape("invalid RoPE head".into()));
    }
    let half = dim / 2;
    // view(..., d/2, 2).transpose(-1, -2).reshape(..., d)
    let mut permuted = vec![0.0; dim];
    for index in 0..half {
        permuted[index] = input[index * 2];
        permuted[half + index] = input[index * 2 + 1];
    }
    let frequencies = yarn_inverse_frequencies(config);
    let mscale = yarn_mscale(config.rope_factor, config.rope_mscale)
        / yarn_mscale(config.rope_factor, config.rope_mscale_all_dim);
    let mut output = vec![0.0; dim];
    for index in 0..half {
        let angle = position as f32 * frequencies[index];
        let cos = angle.cos() * mscale;
        let sin = angle.sin() * mscale;
        output[index] = permuted[index] * cos - permuted[half + index] * sin;
        output[half + index] = permuted[half + index] * cos + permuted[index] * sin;
    }
    Ok(output)
}

pub fn attention_softmax_scale(config: &KimiK27Config) -> f32 {
    let base = 1.0 / (config.q_head_dim() as f32).sqrt();
    let mscale = yarn_mscale(config.rope_factor, config.rope_mscale_all_dim);
    base * mscale * mscale
}

pub fn decode_u32_le(bytes: &[u8]) -> Result<Vec<u32>, KimiError> {
    if bytes.len() % 4 != 0 {
        return Err(KimiError::Shape(
            "I32 tensor byte length is not divisible by four".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

pub fn decode_i32_le(bytes: &[u8]) -> Result<Vec<i32>, KimiError> {
    Ok(decode_u32_le(bytes)?
        .into_iter()
        .map(|value| value as i32)
        .collect())
}

/// Compact KV cache entry stored by Sytra for MLA decode.
#[derive(Debug, Clone, PartialEq)]
pub struct MlaCacheEntry {
    /// RMS-normalized `kv_lora_rank` latent used by `kv_b_proj`.
    pub latent: Vec<f32>,
    /// Rotated shared positional key.
    pub rope_key: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MlaKvCache {
    entries: Vec<MlaCacheEntry>,
}

/// Production compact MLA cache. BF16 storage matches the checkpoint compute
/// dtype and the memory-envelope calculation; the FP32 cache above remains an
/// unquantized numerical oracle.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactMlaCacheEntry {
    latent: Vec<u16>,
    rope_key: Vec<u16>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CompactMlaKvCache {
    entries: Vec<CompactMlaCacheEntry>,
}

impl CompactMlaKvCache {
    pub fn push(
        &mut self,
        latent: &[f32],
        rope_key: &[f32],
        config: &KimiK27Config,
    ) -> Result<(), KimiError> {
        if latent.len() != config.kv_lora_rank || rope_key.len() != config.qk_rope_head_dim {
            return Err(KimiError::Shape(
                "invalid compact BF16 MLA cache entry".into(),
            ));
        }
        self.entries.push(CompactMlaCacheEntry {
            latent: latent.iter().map(|value| f32_to_bf16(*value)).collect(),
            rope_key: rope_key.iter().map(|value| f32_to_bf16(*value)).collect(),
        });
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| (entry.latent.len() + entry.rope_key.len()) * size_of::<u16>())
            .sum()
    }

    pub fn truncate(&mut self, positions: usize) {
        self.entries.truncate(positions);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbsorbedMlaOutput {
    pub values: Vec<f32>,
    pub metrics: DenseTileMetrics,
}

/// Decode one token directly against compact MLA latents. The K/V up-
/// projection is algebraically absorbed into the query and output paths, so
/// each row of `kv_b_proj` is streamed exactly once per token regardless of
/// context length. Full per-position K/V heads are never materialized.
pub fn mla_decode_absorbed_bf16<M, T>(
    store: &DenseTensorStore,
    kv_b_tensor: &str,
    q_nope: &[f32],
    q_rope: &[f32],
    cache: &CompactMlaKvCache,
    config: &KimiK27Config,
    tile_bytes: u64,
    mut matvec: M,
    mut transpose_matvec: T,
) -> Result<AbsorbedMlaOutput, KimiError>
where
    M: FnMut(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, String>,
    T: FnMut(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let heads = config.num_attention_heads;
    let nope = config.qk_nope_head_dim;
    let rope = config.qk_rope_head_dim;
    let value_dim = config.v_head_dim;
    let latent_dim = config.kv_lora_rank;
    let per_head = nope + value_dim;
    if cache.is_empty() || q_nope.len() != heads * nope || q_rope.len() != heads * rope {
        return Err(KimiError::Shape(
            "invalid absorbed MLA decode dimensions".into(),
        ));
    }

    let mut output = Vec::with_capacity(heads * value_dim);
    let mut metrics = DenseTileMetrics::default();
    let scale = attention_softmax_scale(config);
    for head in 0..heads {
        let row_base = head * per_head;
        let absorbed_query = tiled_bf16_transpose_matvec_rows(
            store,
            kv_b_tensor,
            row_base,
            nope,
            tile_bytes,
            &q_nope[head * nope..(head + 1) * nope],
            &mut transpose_matvec,
        )?;
        merge_dense_metrics(&mut metrics, absorbed_query.metrics);

        let mut probabilities = Vec::with_capacity(cache.entries.len());
        for entry in &cache.entries {
            let latent_score = absorbed_query
                .values
                .iter()
                .zip(&entry.latent)
                .map(|(query, latent)| query * bf16_to_f32(*latent))
                .sum::<f32>();
            let rope_score = q_rope[head * rope..(head + 1) * rope]
                .iter()
                .zip(&entry.rope_key)
                .map(|(query, key)| query * bf16_to_f32(*key))
                .sum::<f32>();
            probabilities.push((latent_score + rope_score) * scale);
        }
        softmax_in_place(&mut probabilities);
        let mut weighted_latent = vec![0.0_f32; latent_dim];
        for (probability, entry) in probabilities.into_iter().zip(&cache.entries) {
            for (sum, latent) in weighted_latent.iter_mut().zip(&entry.latent) {
                *sum += probability * bf16_to_f32(*latent);
            }
        }
        let values = tiled_bf16_matvec_rows(
            store,
            kv_b_tensor,
            row_base + nope,
            value_dim,
            tile_bytes,
            &weighted_latent,
            &mut matvec,
        )?;
        merge_dense_metrics(&mut metrics, values.metrics);
        output.extend(values.values);
    }
    Ok(AbsorbedMlaOutput {
        values: output,
        metrics,
    })
}

/// Causal multi-position counterpart to `mla_decode_absorbed_bf16`. All new
/// positions share one scan of `kv_b_proj`; query position `i` can see the
/// existing prefix and new positions through `i`, never later draft tokens.
pub fn mla_decode_absorbed_batch_bf16<M, T>(
    store: &DenseTensorStore,
    kv_b_tensor: &str,
    q_nope: &[f32],
    q_rope: &[f32],
    new_latents: &[f32],
    new_rope_keys: &[f32],
    cache: &mut CompactMlaKvCache,
    config: &KimiK27Config,
    positions: usize,
    tile_bytes: u64,
    mut matmul: M,
    mut transpose_matmul: T,
) -> Result<AbsorbedMlaOutput, KimiError>
where
    M: FnMut(&[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
    T: FnMut(&[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let heads = config.num_attention_heads;
    let nope = config.qk_nope_head_dim;
    let rope = config.qk_rope_head_dim;
    let latent_dim = config.kv_lora_rank;
    let value_dim = config.v_head_dim;
    let per_head = nope + value_dim;
    if positions == 0
        || q_nope.len() != positions * heads * nope
        || q_rope.len() != positions * heads * rope
        || new_latents.len() != positions * latent_dim
        || new_rope_keys.len() != positions * rope
    {
        return Err(KimiError::Shape(
            "invalid absorbed MLA batch dimensions".into(),
        ));
    }
    let prefix_len = cache.len();
    let mut staged = cache.clone();
    for position in 0..positions {
        staged.push(
            &new_latents[position * latent_dim..(position + 1) * latent_dim],
            &new_rope_keys[position * rope..(position + 1) * rope],
            config,
        )?;
    }

    let mut output = vec![0.0_f32; positions * heads * value_dim];
    let mut metrics = DenseTileMetrics::default();
    let scale = attention_softmax_scale(config);
    for head in 0..heads {
        let mut head_queries = Vec::with_capacity(positions * nope);
        for position in 0..positions {
            let start = position * heads * nope + head * nope;
            head_queries.extend_from_slice(&q_nope[start..start + nope]);
        }
        let row_base = head * per_head;
        let absorbed = tiled_bf16_transpose_matmul_rows(
            store,
            kv_b_tensor,
            row_base,
            nope,
            tile_bytes,
            positions,
            &head_queries,
            &mut transpose_matmul,
        )?;
        merge_dense_metrics(&mut metrics, absorbed.metrics);

        let mut weighted = vec![0.0_f32; positions * latent_dim];
        for position in 0..positions {
            let visible = prefix_len + position + 1;
            let query = &absorbed.values[position * latent_dim..(position + 1) * latent_dim];
            let rope_query_start = position * heads * rope + head * rope;
            let rope_query = &q_rope[rope_query_start..rope_query_start + rope];
            let mut probabilities = Vec::with_capacity(visible);
            for entry in &staged.entries[..visible] {
                let latent_score = query
                    .iter()
                    .zip(&entry.latent)
                    .map(|(query, latent)| query * bf16_to_f32(*latent))
                    .sum::<f32>();
                let rope_score = rope_query
                    .iter()
                    .zip(&entry.rope_key)
                    .map(|(query, key)| query * bf16_to_f32(*key))
                    .sum::<f32>();
                probabilities.push((latent_score + rope_score) * scale);
            }
            softmax_in_place(&mut probabilities);
            let target = &mut weighted[position * latent_dim..(position + 1) * latent_dim];
            for (probability, entry) in probabilities.into_iter().zip(&staged.entries[..visible]) {
                for (sum, latent) in target.iter_mut().zip(&entry.latent) {
                    *sum += probability * bf16_to_f32(*latent);
                }
            }
        }
        let values = tiled_bf16_matmul_rows(
            store,
            kv_b_tensor,
            row_base + nope,
            value_dim,
            tile_bytes,
            positions,
            &weighted,
            &mut matmul,
        )?;
        merge_dense_metrics(&mut metrics, values.metrics);
        for position in 0..positions {
            let source = &values.values[position * value_dim..(position + 1) * value_dim];
            let start = position * heads * value_dim + head * value_dim;
            output[start..start + value_dim].copy_from_slice(source);
        }
    }
    *cache = staged;
    Ok(AbsorbedMlaOutput {
        values: output,
        metrics,
    })
}

#[derive(Debug, Clone, Copy)]
pub enum KimiExecutionBackend<'a> {
    Cpu,
    Cuda(&'a CudaAccelerator),
}

impl KimiExecutionBackend<'_> {
    fn matvec(
        self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Cpu => bf16_tile_cpu(weights, rows, cols, input),
            Self::Cuda(cuda) => cuda.bf16_matvec_bytes(weights, rows, cols, input),
        }
    }

    fn transpose_matvec(
        self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Cpu => bf16_transpose_tile_cpu(weights, rows, cols, input),
            Self::Cuda(cuda) => cuda.bf16_transpose_matvec_bytes(weights, rows, cols, input),
        }
    }

    fn matmul(
        self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        positions: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Cpu => bf16_tile_matmul_cpu(weights, rows, cols, positions, input),
            Self::Cuda(cuda) => cuda.bf16_matmul_bytes(weights, rows, cols, positions, input),
        }
    }

    fn transpose_matmul(
        self,
        weights: &[u8],
        rows: usize,
        cols: usize,
        positions: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        match self {
            Self::Cpu => bf16_transpose_tile_matmul_cpu(weights, rows, cols, positions, input),
            Self::Cuda(cuda) => {
                cuda.bf16_transpose_matmul_bytes(weights, rows, cols, positions, input)
            }
        }
    }

    fn expert(
        self,
        resident: &ResidentExpert,
        config: &KimiK27Config,
        hidden: &[f32],
    ) -> Result<Vec<f32>, KimiError> {
        self.expert_batch(resident, config, 1, hidden)
    }

    fn expert_batch(
        self,
        resident: &ResidentExpert,
        config: &KimiK27Config,
        positions: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>, KimiError> {
        if positions == 0 || hidden.len() != positions * config.hidden_size {
            return Err(KimiError::Shape(
                "invalid routed expert batch dimensions".into(),
            ));
        }
        match self {
            Self::Cpu => BorrowedKimiExpertWeights::from_resident(resident, config)?
                .forward_batch(positions, hidden),
            Self::Cuda(cuda) if resident.accelerator_buffer().is_some() => {
                let buffer = resident
                    .accelerator_buffer()
                    .expect("matched resident buffer");
                let projection = |name: &str,
                                  rows: usize,
                                  cols: usize,
                                  input: &[f32]|
                 -> Result<Vec<f32>, KimiError> {
                    let packed = resident
                        .tensors
                        .iter()
                        .find(|tensor| tensor.name.ends_with(&format!("{name}.weight_packed")))
                        .ok_or_else(|| {
                            KimiError::Shape(format!(
                                "resident expert is missing {name}.weight_packed"
                            ))
                        })?;
                    let scales = resident
                        .tensors
                        .iter()
                        .find(|tensor| tensor.name.ends_with(&format!("{name}.weight_scale")))
                        .ok_or_else(|| {
                            KimiError::Shape(format!(
                                "resident expert is missing {name}.weight_scale"
                            ))
                        })?;
                    cuda.resident_int4_group32_bf16_matmul(
                        buffer,
                        packed.offset,
                        scales.offset,
                        rows,
                        cols,
                        positions,
                        input,
                    )
                    .map_err(KimiError::Shape)
                };
                let mut gate = projection(
                    "gate_proj",
                    config.moe_intermediate_size,
                    config.hidden_size,
                    hidden,
                )?;
                let up = projection(
                    "up_proj",
                    config.moe_intermediate_size,
                    config.hidden_size,
                    hidden,
                )?;
                for (gate, up) in gate.iter_mut().zip(up) {
                    *gate = apply_activation(ActivationKind::Silu, *gate) * up;
                }
                projection(
                    "down_proj",
                    config.hidden_size,
                    config.moe_intermediate_size,
                    &gate,
                )
            }
            Self::Cuda(cuda) => {
                let weights = BorrowedKimiExpertWeights::from_resident(resident, config)?;
                let mut gate = cuda
                    .int4_group32_bf16_bytes_matmul(
                        weights.gate.packed,
                        weights.gate.scales,
                        weights.gate.rows,
                        weights.gate.cols,
                        positions,
                        hidden,
                    )
                    .map_err(KimiError::Shape)?;
                let up = cuda
                    .int4_group32_bf16_bytes_matmul(
                        weights.up.packed,
                        weights.up.scales,
                        weights.up.rows,
                        weights.up.cols,
                        positions,
                        hidden,
                    )
                    .map_err(KimiError::Shape)?;
                for (gate, up) in gate.iter_mut().zip(up) {
                    *gate = apply_activation(ActivationKind::Silu, *gate) * up;
                }
                cuda.int4_group32_bf16_bytes_matmul(
                    weights.down.packed,
                    weights.down.scales,
                    weights.down.rows,
                    weights.down.cols,
                    positions,
                    &gate,
                )
                .map_err(KimiError::Shape)
            }
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct KimiDecodeState {
    pub layers: Vec<CompactMlaKvCache>,
}

impl KimiDecodeState {
    pub fn new(config: &KimiK27Config) -> Self {
        Self {
            layers: vec![CompactMlaKvCache::default(); config.num_hidden_layers],
        }
    }

    pub fn position(&self) -> usize {
        self.layers.first().map(CompactMlaKvCache::len).unwrap_or(0)
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(CompactMlaKvCache::bytes).sum()
    }

    pub fn truncate(&mut self, positions: usize) {
        for layer in &mut self.layers {
            layer.truncate(positions);
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct KimiStepMetrics {
    pub dense_tiles: u64,
    pub dense_storage_bytes: u64,
    pub peak_dense_tile_bytes: u64,
    pub expert_waves: u64,
}

impl KimiStepMetrics {
    pub fn merge(&mut self, other: Self) {
        self.dense_tiles = self.dense_tiles.saturating_add(other.dense_tiles);
        self.dense_storage_bytes = self
            .dense_storage_bytes
            .saturating_add(other.dense_storage_bytes);
        self.peak_dense_tile_bytes = self.peak_dense_tile_bytes.max(other.peak_dense_tile_bytes);
        self.expert_waves = self.expert_waves.saturating_add(other.expert_waves);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KimiSpeculativeOutput {
    pub verification: crate::GreedyVerification,
    pub target_predictions: Vec<u32>,
    pub metrics: KimiStepMetrics,
}

/// Correctness-first, storage-backed one-token executor for the official Kimi
/// text tower. Every dense matrix is tiled, compact MLA is retained in BF16,
/// and routed experts are consumed through byte-bounded scheduler waves.
/// This path is deliberately not advertised by the adapter registry until it
/// passes real checkpoint reference-logit and teacher-forced token tests.
pub struct KimiOneTokenExecutor<'a> {
    config: &'a KimiK27Config,
    dense: &'a DenseTensorStore,
    scheduler: &'a StreamingScheduler,
    backend: KimiExecutionBackend<'a>,
    dense_tile_bytes: u64,
}

impl<'a> KimiOneTokenExecutor<'a> {
    pub fn new(
        config: &'a KimiK27Config,
        dense: &'a DenseTensorStore,
        scheduler: &'a StreamingScheduler,
        backend: KimiExecutionBackend<'a>,
        dense_tile_bytes: u64,
    ) -> Result<Self, KimiError> {
        let minimum_row = (config
            .hidden_size
            .max(config.q_lora_rank)
            .max(config.kv_lora_rank)
            .max(config.intermediate_size)
            .max(config.moe_intermediate_size) as u64)
            .checked_mul(2)
            .ok_or_else(|| KimiError::Shape("dense row byte size overflow".into()))?;
        if dense_tile_bytes < minimum_row {
            return Err(KimiError::Shape(format!(
                "dense tile budget {dense_tile_bytes} is smaller than the largest {minimum_row}-byte row"
            )));
        }
        Ok(Self {
            config,
            dense,
            scheduler,
            backend,
            dense_tile_bytes,
        })
    }

    pub fn token_embedding(&self, token: u32) -> Result<Vec<f32>, KimiError> {
        if token as usize >= self.config.vocab_size {
            return Err(KimiError::Shape(format!(
                "token {token} is outside the vocabulary"
            )));
        }
        let name = "language_model.model.embed_tokens.weight";
        let row_bytes = (self.config.hidden_size as u64) * 2;
        let bytes = self
            .dense
            .read_window(name, u64::from(token) * row_bytes, row_bytes)?;
        decode_float_values("BF16", &bytes).map_err(KimiError::Math)
    }

    pub fn forward_token(
        &self,
        token: u32,
        state: &mut KimiDecodeState,
    ) -> Result<(Vec<f32>, KimiStepMetrics), KimiError> {
        if state.layers.len() != self.config.num_hidden_layers {
            return Err(KimiError::Shape(
                "decode state layer count does not match the model".into(),
            ));
        }
        let position = state.position();
        if state.layers.iter().any(|cache| cache.len() != position) {
            return Err(KimiError::Shape(
                "decode state layers have inconsistent positions".into(),
            ));
        }
        let mut hidden = self.token_embedding(token)?;
        let mut metrics = KimiStepMetrics::default();
        for layer in 0..self.config.num_hidden_layers {
            hidden = self.forward_layer(
                layer,
                position,
                &hidden,
                &mut state.layers[layer],
                &mut metrics,
            )?;
        }
        Ok((hidden, metrics))
    }

    /// Run a causal target batch for prefill or speculative verification.
    /// Dense matrices are scanned once per layer for the entire batch.
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<(Vec<f32>, KimiStepMetrics), KimiError> {
        if tokens.is_empty() {
            return Err(KimiError::Shape("token batch cannot be empty".into()));
        }
        if state.layers.len() != self.config.num_hidden_layers {
            return Err(KimiError::Shape(
                "decode state layer count does not match the model".into(),
            ));
        }
        let base_position = state.position();
        if state
            .layers
            .iter()
            .any(|cache| cache.len() != base_position)
            || base_position.saturating_add(tokens.len()) > self.config.max_position_embeddings
        {
            return Err(KimiError::Shape(
                "decode state positions are inconsistent or exceed the model context".into(),
            ));
        }
        let mut hidden = Vec::with_capacity(tokens.len() * self.config.hidden_size);
        for token in tokens {
            hidden.extend(self.token_embedding(*token)?);
        }
        let mut metrics = KimiStepMetrics::default();
        for layer in 0..self.config.num_hidden_layers {
            hidden = self.forward_layer_batch(
                layer,
                base_position,
                tokens.len(),
                &hidden,
                &mut state.layers[layer],
                &mut metrics,
            )?;
        }
        Ok((hidden, metrics))
    }

    pub fn greedy_token(&self, hidden: &[f32]) -> Result<(u32, KimiStepMetrics), KimiError> {
        let (logits, metrics) = self.logits(hidden)?;
        let token = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| KimiError::Shape("language-model head returned no logits".into()))?;
        Ok((token, metrics))
    }

    pub fn logits(&self, hidden: &[f32]) -> Result<(Vec<f32>, KimiStepMetrics), KimiError> {
        if hidden.len() != self.config.hidden_size {
            return Err(KimiError::Shape(
                "language-model head input has the wrong hidden size".into(),
            ));
        }
        let norm = self.read_vector("language_model.model.norm.weight", self.config.hidden_size)?;
        let hidden = rms_norm(hidden, &norm, self.config.rms_norm_eps)?;
        let output = self.dense_matvec("language_model.lm_head.weight", &hidden)?;
        Ok((output.values, step_metrics(output.metrics)))
    }

    pub fn greedy_tokens(
        &self,
        hidden: &[f32],
        positions: usize,
    ) -> Result<(Vec<u32>, KimiStepMetrics), KimiError> {
        if positions == 0 || hidden.len() != positions * self.config.hidden_size {
            return Err(KimiError::Shape(
                "batched hidden states have inconsistent dimensions".into(),
            ));
        }
        let norm = self.read_vector("language_model.model.norm.weight", self.config.hidden_size)?;
        let hidden = rms_norm_batch(
            hidden,
            positions,
            self.config.hidden_size,
            &norm,
            self.config.rms_norm_eps,
        )?;
        let output = self.dense_matmul("language_model.lm_head.weight", positions, &hidden)?;
        let mut tokens = Vec::with_capacity(positions);
        for logits in output.values.chunks_exact(self.config.vocab_size) {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index as u32)
                .ok_or_else(|| KimiError::Shape("language-model head returned no logits".into()))?;
            tokens.push(token);
        }
        Ok((tokens, step_metrics(output.metrics)))
    }

    /// Verify a greedy draft transactionally. `current_token` is the token
    /// whose KV entry has not yet been appended. Rejected suffix KV entries
    /// are truncated without replay because causal prefix states are already
    /// exact and independent of later draft positions.
    pub fn verify_greedy_draft(
        &self,
        current_token: u32,
        draft_tokens: &[u32],
        state: &mut KimiDecodeState,
    ) -> Result<KimiSpeculativeOutput, KimiError> {
        if draft_tokens.is_empty() {
            return Err(KimiError::Shape("draft batch cannot be empty".into()));
        }
        let base_position = state.position();
        let mut inputs = Vec::with_capacity(draft_tokens.len() + 1);
        inputs.push(current_token);
        inputs.extend_from_slice(draft_tokens);
        let (hidden, mut metrics) = match self.forward_tokens(&inputs, state) {
            Ok(output) => output,
            Err(error) => {
                state.truncate(base_position);
                return Err(error);
            }
        };
        let (target_predictions, head_metrics) = match self.greedy_tokens(&hidden, inputs.len()) {
            Ok(output) => output,
            Err(error) => {
                state.truncate(base_position);
                return Err(error);
            }
        };
        merge_step_metrics(&mut metrics, head_metrics);
        let verification = crate::verify_greedy(draft_tokens, &target_predictions)
            .map_err(|error| KimiError::Shape(error.to_string()))?;
        let committed_inputs = 1 + verification.accepted_draft_tokens;
        state.truncate(base_position + committed_inputs);
        Ok(KimiSpeculativeOutput {
            verification,
            target_predictions,
            metrics,
        })
    }

    fn forward_layer_batch(
        &self,
        layer: usize,
        base_position: usize,
        positions: usize,
        hidden: &[f32],
        cache: &mut CompactMlaKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let prefix = format!("language_model.model.layers.{layer}");
        let input_norm = self.read_vector(
            &format!("{prefix}.input_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = rms_norm_batch(
            hidden,
            positions,
            self.config.hidden_size,
            &input_norm,
            self.config.rms_norm_eps,
        )?;
        let attention = self.forward_attention_batch(
            &prefix,
            base_position,
            positions,
            &normalized,
            cache,
            metrics,
        )?;
        let residual = add_vectors(hidden, &attention)?;
        let post_norm = self.read_vector(
            &format!("{prefix}.post_attention_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = rms_norm_batch(
            &residual,
            positions,
            self.config.hidden_size,
            &post_norm,
            self.config.rms_norm_eps,
        )?;
        let mlp = if layer < self.config.first_k_dense_replace {
            self.forward_dense_mlp_batch(&format!("{prefix}.mlp"), positions, &normalized, metrics)?
        } else {
            self.forward_moe_batch(
                layer,
                &format!("{prefix}.mlp"),
                positions,
                &normalized,
                metrics,
            )?
        };
        add_vectors(&residual, &mlp)
    }

    fn forward_attention_batch(
        &self,
        prefix: &str,
        base_position: usize,
        positions: usize,
        hidden: &[f32],
        cache: &mut CompactMlaKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let attention = format!("{prefix}.self_attn");
        let q_a = self.dense_matmul(&format!("{attention}.q_a_proj.weight"), positions, hidden)?;
        merge_step_dense(metrics, q_a.metrics);
        let q_norm_weight = self.read_vector(
            &format!("{attention}.q_a_layernorm.weight"),
            self.config.q_lora_rank,
        )?;
        let q_a = rms_norm_batch(
            &q_a.values,
            positions,
            self.config.q_lora_rank,
            &q_norm_weight,
            self.config.rms_norm_eps,
        )?;
        let q = self.dense_matmul(&format!("{attention}.q_b_proj.weight"), positions, &q_a)?;
        merge_step_dense(metrics, q.metrics);
        let mut q_nope = Vec::with_capacity(
            positions * self.config.num_attention_heads * self.config.qk_nope_head_dim,
        );
        let mut q_rope = Vec::with_capacity(
            positions * self.config.num_attention_heads * self.config.qk_rope_head_dim,
        );
        for position in 0..positions {
            let position_q =
                &q.values[position * self.config.num_attention_heads * self.config.q_head_dim()
                    ..(position + 1) * self.config.num_attention_heads * self.config.q_head_dim()];
            for head in position_q.chunks_exact(self.config.q_head_dim()) {
                q_nope.extend_from_slice(&head[..self.config.qk_nope_head_dim]);
                q_rope.extend(apply_yarn_rope(
                    &head[self.config.qk_nope_head_dim..],
                    base_position + position,
                    self.config,
                )?);
            }
        }

        let compressed = self.dense_matmul(
            &format!("{attention}.kv_a_proj_with_mqa.weight"),
            positions,
            hidden,
        )?;
        merge_step_dense(metrics, compressed.metrics);
        let kv_norm_weight = self.read_vector(
            &format!("{attention}.kv_a_layernorm.weight"),
            self.config.kv_lora_rank,
        )?;
        let compressed_width = self.config.kv_lora_rank + self.config.qk_rope_head_dim;
        let mut latents = Vec::with_capacity(positions * self.config.kv_lora_rank);
        let mut rope_keys = Vec::with_capacity(positions * self.config.qk_rope_head_dim);
        for position in 0..positions {
            let values =
                &compressed.values[position * compressed_width..(position + 1) * compressed_width];
            latents.extend(rms_norm(
                &values[..self.config.kv_lora_rank],
                &kv_norm_weight,
                self.config.rms_norm_eps,
            )?);
            rope_keys.extend(apply_yarn_rope(
                &values[self.config.kv_lora_rank..],
                base_position + position,
                self.config,
            )?);
        }
        let backend = self.backend;
        let attended = mla_decode_absorbed_batch_bf16(
            self.dense,
            &format!("{attention}.kv_b_proj.weight"),
            &q_nope,
            &q_rope,
            &latents,
            &rope_keys,
            cache,
            self.config,
            positions,
            self.dense_tile_bytes,
            move |weights, rows, cols, positions, input| {
                backend.matmul(weights, rows, cols, positions, input)
            },
            move |weights, rows, cols, positions, input| {
                backend.transpose_matmul(weights, rows, cols, positions, input)
            },
        )?;
        merge_step_dense(metrics, attended.metrics);
        let output = self.dense_matmul(
            &format!("{attention}.o_proj.weight"),
            positions,
            &attended.values,
        )?;
        merge_step_dense(metrics, output.metrics);
        Ok(output.values)
    }

    fn forward_dense_mlp_batch(
        &self,
        prefix: &str,
        positions: usize,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let gate = self.dense_matmul(&format!("{prefix}.gate_proj.weight"), positions, hidden)?;
        merge_step_dense(metrics, gate.metrics);
        let up = self.dense_matmul(&format!("{prefix}.up_proj.weight"), positions, hidden)?;
        merge_step_dense(metrics, up.metrics);
        let activated: Vec<_> = gate
            .values
            .into_iter()
            .zip(up.values)
            .map(|(gate, up)| apply_activation(ActivationKind::Silu, gate) * up)
            .collect();
        let down =
            self.dense_matmul(&format!("{prefix}.down_proj.weight"), positions, &activated)?;
        merge_step_dense(metrics, down.metrics);
        Ok(down.values)
    }

    fn forward_moe_batch(
        &self,
        layer: usize,
        prefix: &str,
        positions: usize,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let gate = self.dense_matmul(&format!("{prefix}.gate.weight"), positions, hidden)?;
        merge_step_dense(metrics, gate.metrics);
        let bias = self.read_vector_any(
            &format!("{prefix}.gate.e_score_correction_bias"),
            self.config.n_routed_experts,
        )?;
        let contract = crate::RouterContract {
            score: crate::RouterScoreKind::Sigmoid,
            normalize_selected: true,
            scaling_factor: self.config.routed_scaling_factor,
            correction_bias: true,
            groups: self.config.n_group as u32,
            selected_groups: self.config.topk_group as u32,
        };
        let routing = RoutingBatch {
            positions: (0..positions)
                .map(|position| {
                    let gate = &gate.values[position * self.config.n_routed_experts
                        ..(position + 1) * self.config.n_routed_experts];
                    route_topk_logits(
                        gate,
                        Some(&bias),
                        self.config.num_experts_per_tok,
                        RouterSemantics::NoAuxTc,
                        &contract,
                    )
                    .map(|routes| {
                        routes
                            .into_iter()
                            .map(|route| Route {
                                expert: route.expert as u32,
                                weight: route.weight,
                            })
                            .collect()
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let backend = self.backend;
        let config = self.config;
        let (routed, wave_count) = self
            .scheduler
            .for_each_prepared_wave_fold(
                layer as u32,
                routing,
                vec![0.0_f32; positions * self.config.hidden_size],
                |routed, prepared| {
                    for resident in &prepared.experts {
                        let active: Vec<_> = (0..positions)
                            .filter_map(|position| {
                                prepared.routes.positions[position]
                                    .iter()
                                    .find(|route| route.expert == resident.key.expert)
                                    .map(|route| (position, route.weight))
                            })
                            .collect();
                        if active.is_empty() {
                            continue;
                        }
                        let mut expert_input =
                            Vec::with_capacity(active.len() * config.hidden_size);
                        for (position, _) in &active {
                            expert_input.extend_from_slice(
                                &hidden[*position * config.hidden_size
                                    ..(*position + 1) * config.hidden_size],
                            );
                        }
                        let output = backend
                            .expert_batch(resident, config, active.len(), &expert_input)
                            .map_err(|error| error.to_string())?;
                        for (active_position, (position, weight)) in active.into_iter().enumerate()
                        {
                            let target = &mut routed[position * config.hidden_size
                                ..(position + 1) * config.hidden_size];
                            let source = &output[active_position * config.hidden_size
                                ..(active_position + 1) * config.hidden_size];
                            for (sum, value) in target.iter_mut().zip(source) {
                                *sum += weight * value;
                            }
                        }
                    }
                    Ok(())
                },
            )
            .map_err(scheduler_error)?;
        metrics.expert_waves += wave_count as u64;
        let shared = self.forward_dense_mlp_batch(
            &format!("{prefix}.shared_experts"),
            positions,
            hidden,
            metrics,
        )?;
        add_vectors(&routed, &shared)
    }

    fn forward_layer(
        &self,
        layer: usize,
        position: usize,
        hidden: &[f32],
        cache: &mut CompactMlaKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let prefix = format!("language_model.model.layers.{layer}");
        let input_norm = self.read_vector(
            &format!("{prefix}.input_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = rms_norm(hidden, &input_norm, self.config.rms_norm_eps)?;
        let attention = self.forward_attention(&prefix, position, &normalized, cache, metrics)?;
        let residual = add_vectors(hidden, &attention)?;
        let post_norm = self.read_vector(
            &format!("{prefix}.post_attention_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = rms_norm(&residual, &post_norm, self.config.rms_norm_eps)?;
        let mlp = if layer < self.config.first_k_dense_replace {
            self.forward_dense_mlp(&format!("{prefix}.mlp"), &normalized, metrics)?
        } else {
            self.forward_moe(layer, &format!("{prefix}.mlp"), &normalized, metrics)?
        };
        add_vectors(&residual, &mlp)
    }

    fn forward_attention(
        &self,
        prefix: &str,
        position: usize,
        hidden: &[f32],
        cache: &mut CompactMlaKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let attention = format!("{prefix}.self_attn");
        let q_a = self.dense_matvec(&format!("{attention}.q_a_proj.weight"), hidden)?;
        merge_step_dense(metrics, q_a.metrics);
        let q_norm_weight = self.read_vector(
            &format!("{attention}.q_a_layernorm.weight"),
            self.config.q_lora_rank,
        )?;
        let q_a = rms_norm(&q_a.values, &q_norm_weight, self.config.rms_norm_eps)?;
        let q = self.dense_matvec(&format!("{attention}.q_b_proj.weight"), &q_a)?;
        merge_step_dense(metrics, q.metrics);
        let mut q_nope =
            Vec::with_capacity(self.config.num_attention_heads * self.config.qk_nope_head_dim);
        let mut q_rope =
            Vec::with_capacity(self.config.num_attention_heads * self.config.qk_rope_head_dim);
        for head in q.values.chunks_exact(self.config.q_head_dim()) {
            q_nope.extend_from_slice(&head[..self.config.qk_nope_head_dim]);
            q_rope.extend(apply_yarn_rope(
                &head[self.config.qk_nope_head_dim..],
                position,
                self.config,
            )?);
        }

        let compressed =
            self.dense_matvec(&format!("{attention}.kv_a_proj_with_mqa.weight"), hidden)?;
        merge_step_dense(metrics, compressed.metrics);
        let kv_norm_weight = self.read_vector(
            &format!("{attention}.kv_a_layernorm.weight"),
            self.config.kv_lora_rank,
        )?;
        let latent = rms_norm(
            &compressed.values[..self.config.kv_lora_rank],
            &kv_norm_weight,
            self.config.rms_norm_eps,
        )?;
        let rope_key = apply_yarn_rope(
            &compressed.values[self.config.kv_lora_rank..],
            position,
            self.config,
        )?;
        cache.push(&latent, &rope_key, self.config)?;
        let backend = self.backend;
        let attended = mla_decode_absorbed_bf16(
            self.dense,
            &format!("{attention}.kv_b_proj.weight"),
            &q_nope,
            &q_rope,
            cache,
            self.config,
            self.dense_tile_bytes,
            move |weights, rows, cols, input| backend.matvec(weights, rows, cols, input),
            move |weights, rows, cols, input| backend.transpose_matvec(weights, rows, cols, input),
        )?;
        merge_step_dense(metrics, attended.metrics);
        let output = self.dense_matvec(&format!("{attention}.o_proj.weight"), &attended.values)?;
        merge_step_dense(metrics, output.metrics);
        Ok(output.values)
    }

    fn forward_dense_mlp(
        &self,
        prefix: &str,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let gate = self.dense_matvec(&format!("{prefix}.gate_proj.weight"), hidden)?;
        merge_step_dense(metrics, gate.metrics);
        let up = self.dense_matvec(&format!("{prefix}.up_proj.weight"), hidden)?;
        merge_step_dense(metrics, up.metrics);
        let activated: Vec<_> = gate
            .values
            .into_iter()
            .zip(up.values)
            .map(|(gate, up)| apply_activation(ActivationKind::Silu, gate) * up)
            .collect();
        let down = self.dense_matvec(&format!("{prefix}.down_proj.weight"), &activated)?;
        merge_step_dense(metrics, down.metrics);
        Ok(down.values)
    }

    fn forward_moe(
        &self,
        layer: usize,
        prefix: &str,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, KimiError> {
        let gate = self.dense_matvec(&format!("{prefix}.gate.weight"), hidden)?;
        merge_step_dense(metrics, gate.metrics);
        let bias = self.read_vector_any(
            &format!("{prefix}.gate.e_score_correction_bias"),
            self.config.n_routed_experts,
        )?;
        let routes = route_topk_logits(
            &gate.values,
            Some(&bias),
            self.config.num_experts_per_tok,
            RouterSemantics::NoAuxTc,
            &crate::RouterContract {
                score: crate::RouterScoreKind::Sigmoid,
                normalize_selected: true,
                scaling_factor: self.config.routed_scaling_factor,
                correction_bias: true,
                groups: self.config.n_group as u32,
                selected_groups: self.config.topk_group as u32,
            },
        )?;
        let routing = RoutingBatch {
            positions: vec![routes
                .iter()
                .map(|route| Route {
                    expert: route.expert as u32,
                    weight: route.weight,
                })
                .collect()],
        };
        let backend = self.backend;
        let config = self.config;
        let (routed, wave_count) = self
            .scheduler
            .for_each_prepared_wave_fold(
                layer as u32,
                routing,
                vec![0.0_f32; self.config.hidden_size],
                |routed, prepared| {
                    for resident in &prepared.experts {
                        let weight = prepared.routes.positions[0]
                            .iter()
                            .find(|route| route.expert == resident.key.expert)
                            .map(|route| route.weight)
                            .ok_or_else(|| "prepared expert has no route weight".to_string())?;
                        let output = backend
                            .expert(resident, config, hidden)
                            .map_err(|error| error.to_string())?;
                        for (sum, value) in routed.iter_mut().zip(output) {
                            *sum += weight * value;
                        }
                    }
                    Ok(())
                },
            )
            .map_err(scheduler_error)?;
        metrics.expert_waves += wave_count as u64;
        let shared =
            self.forward_dense_mlp(&format!("{prefix}.shared_experts"), hidden, metrics)?;
        add_vectors(&routed, &shared)
    }

    fn dense_matvec(
        &self,
        tensor: &str,
        input: &[f32],
    ) -> Result<crate::TiledMatvecOutput, KimiError> {
        let backend = self.backend;
        Ok(tiled_bf16_matvec(
            self.dense,
            tensor,
            self.dense_tile_bytes,
            input,
            move |weights, rows, cols, input| backend.matvec(weights, rows, cols, input),
        )?)
    }

    fn dense_matmul(
        &self,
        tensor: &str,
        positions: usize,
        input: &[f32],
    ) -> Result<TiledMatmulOutput, KimiError> {
        let backend = self.backend;
        Ok(tiled_bf16_matmul(
            self.dense,
            tensor,
            self.dense_tile_bytes,
            positions,
            input,
            move |weights, rows, cols, positions, input| {
                backend.matmul(weights, rows, cols, positions, input)
            },
        )?)
    }

    fn read_vector(&self, tensor: &str, length: usize) -> Result<Vec<f32>, KimiError> {
        let metadata = self
            .dense
            .metadata(tensor)
            .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
        if metadata.dtype.as_deref() != Some("BF16") || metadata.shape != [length as u64] {
            return Err(KimiError::Shape(format!(
                "{tensor} is not the expected BF16 vector of length {length}"
            )));
        }
        let bytes = self.dense.read(tensor)?;
        decode_float_values("BF16", &bytes).map_err(KimiError::Math)
    }

    fn read_vector_any(&self, tensor: &str, length: usize) -> Result<Vec<f32>, KimiError> {
        let metadata = self
            .dense
            .metadata(tensor)
            .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
        let dtype = metadata
            .dtype
            .as_deref()
            .ok_or_else(|| KimiError::Shape(format!("{tensor} has no dtype")))?;
        if !matches!(dtype, "BF16" | "F32") || metadata.shape != [length as u64] {
            return Err(KimiError::Shape(format!(
                "{tensor} is not a BF16/F32 vector of length {length}"
            )));
        }
        let bytes = self.dense.read(tensor)?;
        decode_float_values(dtype, &bytes).map_err(KimiError::Math)
    }
}

fn scheduler_error(error: SchedulerError) -> KimiError {
    KimiError::Scheduler(error.to_string())
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>, KimiError> {
    if left.len() != right.len() {
        return Err(KimiError::Shape(
            "residual vectors have different lengths".into(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

fn rms_norm_batch(
    input: &[f32],
    positions: usize,
    width: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, KimiError> {
    if positions == 0 || input.len() != positions * width || weight.len() != width {
        return Err(KimiError::Shape(
            "batched RMSNorm dimensions are inconsistent".into(),
        ));
    }
    let mut output = Vec::with_capacity(input.len());
    for position in input.chunks_exact(width) {
        output.extend(rms_norm(position, weight, epsilon)?);
    }
    Ok(output)
}

fn step_metrics(dense: DenseTileMetrics) -> KimiStepMetrics {
    let mut result = KimiStepMetrics::default();
    merge_step_dense(&mut result, dense);
    result
}

fn merge_step_dense(total: &mut KimiStepMetrics, next: DenseTileMetrics) {
    total.dense_tiles += next.tiles;
    total.dense_storage_bytes += next.storage_bytes;
    total.peak_dense_tile_bytes = total.peak_dense_tile_bytes.max(next.peak_tile_bytes);
}

fn merge_step_metrics(total: &mut KimiStepMetrics, next: KimiStepMetrics) {
    total.dense_tiles += next.dense_tiles;
    total.dense_storage_bytes += next.dense_storage_bytes;
    total.peak_dense_tile_bytes = total.peak_dense_tile_bytes.max(next.peak_dense_tile_bytes);
    total.expert_waves += next.expert_waves;
}

fn merge_dense_metrics(total: &mut DenseTileMetrics, next: DenseTileMetrics) {
    total.tiles += next.tiles;
    total.storage_bytes += next.storage_bytes;
    total.peak_tile_bytes = total.peak_tile_bytes.max(next.peak_tile_bytes);
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

impl MlaKvCache {
    pub fn push(&mut self, entry: MlaCacheEntry, config: &KimiK27Config) -> Result<(), KimiError> {
        if entry.latent.len() != config.kv_lora_rank
            || entry.rope_key.len() != config.qk_rope_head_dim
        {
            return Err(KimiError::Shape(
                "invalid compressed MLA cache entry".into(),
            ));
        }
        self.entries.push(entry);
        Ok(())
    }

    pub fn entries(&self) -> &[MlaCacheEntry] {
        &self.entries
    }

    pub fn bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|entry| (entry.latent.len() + entry.rope_key.len()) * size_of::<f32>())
            .sum()
    }
}

/// Reconstruct K/V heads from compact MLA cache and execute one-token causal
/// attention. `kv_b` uses PyTorch Linear row-major layout:
/// `[heads * (qk_nope + v), kv_lora_rank]`.
pub fn mla_decode_reference(
    q_nope: &[f32],
    q_rope: &[f32],
    cache: &MlaKvCache,
    kv_b: &[f32],
    config: &KimiK27Config,
) -> Result<Vec<f32>, KimiError> {
    let heads = config.num_attention_heads;
    let nope = config.qk_nope_head_dim;
    let rope = config.qk_rope_head_dim;
    let value_dim = config.v_head_dim;
    let per_head_projection = nope + value_dim;
    if q_nope.len() != heads * nope
        || q_rope.len() != heads * rope
        || kv_b.len() != heads * per_head_projection * config.kv_lora_rank
        || cache.entries.is_empty()
    {
        return Err(KimiError::Shape("invalid MLA decode dimensions".into()));
    }
    let mut output = vec![0.0; heads * value_dim];
    let softmax_scale = attention_softmax_scale(config);
    for head in 0..heads {
        let mut logits = Vec::with_capacity(cache.entries.len());
        let mut values = Vec::with_capacity(cache.entries.len());
        for entry in &cache.entries {
            let projection_base = head * per_head_projection * config.kv_lora_rank;
            let mut reconstructed = vec![0.0; per_head_projection];
            for row in 0..per_head_projection {
                reconstructed[row] = kv_b[projection_base + row * config.kv_lora_rank
                    ..projection_base + (row + 1) * config.kv_lora_rank]
                    .iter()
                    .zip(&entry.latent)
                    .map(|(weight, latent)| weight * latent)
                    .sum();
            }
            let qn = &q_nope[head * nope..(head + 1) * nope];
            let qr = &q_rope[head * rope..(head + 1) * rope];
            let logit = qn
                .iter()
                .zip(&reconstructed[..nope])
                .map(|(q, k)| q * k)
                .sum::<f32>()
                + qr.iter()
                    .zip(&entry.rope_key)
                    .map(|(q, k)| q * k)
                    .sum::<f32>();
            logits.push(logit * softmax_scale);
            values.push(reconstructed[nope..].to_vec());
        }
        softmax_in_place(&mut logits);
        for (probability, value) in logits.into_iter().zip(values) {
            for index in 0..value_dim {
                output[head * value_dim + index] += probability * value[index];
            }
        }
    }
    Ok(output)
}

fn yarn_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn expect_str(root: &Value, key: &str, expected: &str) -> Result<(), KimiError> {
    let object = root
        .as_object()
        .ok_or_else(|| KimiError::Contract("config root is not an object".into()))?;
    expect_str_obj(object, key, expected)
}

fn expect_str_obj(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<(), KimiError> {
    match object.get(key).and_then(Value::as_str) {
        Some(actual) if actual == expected => Ok(()),
        actual => Err(KimiError::Contract(format!(
            "{key} must be {expected:?}, got {actual:?}"
        ))),
    }
}

fn expect_bool_obj(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: bool,
) -> Result<(), KimiError> {
    match object.get(key).and_then(Value::as_bool) {
        Some(actual) if actual == expected => Ok(()),
        actual => Err(KimiError::Contract(format!(
            "{key} must be {expected}, got {actual:?}"
        ))),
    }
}

fn expect_u64_obj(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: u64,
) -> Result<(), KimiError> {
    match object.get(key).and_then(Value::as_u64) {
        Some(actual) if actual == expected => Ok(()),
        actual => Err(KimiError::Contract(format!(
            "{key} must be {expected}, got {actual:?}"
        ))),
    }
}

fn usize_field(object: &serde_json::Map<String, Value>, key: &str) -> Result<usize, KimiError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| KimiError::Contract(format!("{key} must be a positive integer")))
}

fn f32_field(object: &serde_json::Map<String, Value>, key: &str) -> Result<f32, KimiError> {
    object
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .ok_or_else(|| KimiError::Contract(format!("{key} must be numeric")))
}

fn validate_expert_segments(
    entry: &ExpertLocation,
    config: &KimiK27Config,
) -> Result<(), KimiError> {
    if entry.segments.len() != 9 {
        return Err(KimiError::Contract(format!(
            "layer {} expert {} has {} tensors; expected the nine packed/scale/shape tensors",
            entry.layer,
            entry.expert,
            entry.segments.len()
        )));
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let (rows, cols) = if projection == "down_proj" {
            (config.hidden_size, config.moe_intermediate_size)
        } else {
            (config.moe_intermediate_size, config.hidden_size)
        };
        expect_tensor(
            entry,
            &format!("{projection}.weight_packed"),
            "I32",
            &[rows as u64, cols.div_ceil(8) as u64],
        )?;
        expect_tensor(
            entry,
            &format!("{projection}.weight_scale"),
            "BF16",
            &[rows as u64, cols.div_ceil(config.group_size) as u64],
        )?;
        expect_tensor(entry, &format!("{projection}.weight_shape"), "I32", &[2])?;
    }
    Ok(())
}

fn expect_tensor(
    entry: &ExpertLocation,
    suffix: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<(), KimiError> {
    let segment = entry
        .segments
        .iter()
        .find(|segment| segment.tensor.ends_with(suffix))
        .ok_or_else(|| {
            KimiError::Contract(format!(
                "layer {} expert {} is missing {suffix}",
                entry.layer, entry.expert
            ))
        })?;
    validate_tensor_metadata(segment, dtype, shape).map_err(|reason| {
        KimiError::Contract(format!(
            "layer {} expert {} {suffix}: {reason}",
            entry.layer, entry.expert
        ))
    })
}

fn validate_tensor_metadata(
    segment: &TensorSegment,
    dtype: &str,
    shape: &[u64],
) -> Result<(), String> {
    if segment.dtype.as_deref() != Some(dtype) {
        return Err(format!("dtype {:?}, expected {dtype}", segment.dtype));
    }
    if segment.shape != shape {
        return Err(format!("shape {:?}, expected {shape:?}", segment.shape));
    }
    let element_bytes = match dtype {
        "I32" => 4,
        "BF16" => 2,
        _ => return Err(format!("unsupported dtype {dtype}")),
    };
    let expected_bytes = shape
        .iter()
        .try_fold(element_bytes as u64, |size, dimension| {
            size.checked_mul(*dimension)
        })
        .ok_or_else(|| "tensor byte size overflow".to_string())?;
    if segment.length != expected_bytes {
        return Err(format!(
            "{} bytes, expected {expected_bytes}",
            segment.length
        ));
    }
    Ok(())
}

fn expect_dense(
    tensors: &HashMap<&str, &TensorSegment>,
    name: &str,
    dtypes: &[&str],
    shape: &[u64],
) -> Result<(), KimiError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| KimiError::Contract(format!("dense tensor {name} is missing")))?;
    let dtype = tensor
        .dtype
        .as_deref()
        .ok_or_else(|| KimiError::Contract(format!("dense tensor {name} has no dtype")))?;
    if !dtypes.contains(&dtype) || tensor.shape != shape {
        return Err(KimiError::Contract(format!(
            "dense tensor {name} has dtype {dtype} and shape {:?}; expected {:?} and {shape:?}",
            tensor.shape, dtypes
        )));
    }
    let element_bytes = match dtype {
        "BF16" => 2_u64,
        "F32" => 4_u64,
        _ => unreachable!("dtype was checked above"),
    };
    let expected = shape
        .iter()
        .try_fold(element_bytes, |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| KimiError::Contract(format!("dense tensor {name} byte size overflow")))?;
    if tensor.length != expected {
        return Err(KimiError::Contract(format!(
            "dense tensor {name} has {} bytes; expected {expected}",
            tensor.length
        )));
    }
    Ok(())
}

fn resident_tensor<'a>(
    expert: &ResidentExpert,
    bytes: &'a [u8],
    suffix: &str,
) -> Result<&'a [u8], KimiError> {
    let tensor = expert
        .tensors
        .iter()
        .find(|tensor| tensor.name.ends_with(suffix))
        .ok_or_else(|| KimiError::Shape(format!("resident expert is missing {suffix}")))?;
    bytes
        .get(tensor.offset..tensor.offset + tensor.length)
        .ok_or_else(|| KimiError::Shape(format!("{suffix} is outside the resident payload")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "index {index}: {left} != {right}"
            );
        }
    }

    #[test]
    fn compact_mla_cache_is_24_times_smaller_than_materialized_kv() {
        let config = KimiK27Config::official();
        assert_eq!(config.compressed_kv_bytes_per_token(2), 1152);
        assert_eq!(config.materialized_kv_bytes_per_token(2), 40_960);
        assert!(
            config.materialized_kv_bytes_per_token(2) / config.compressed_kv_bytes_per_token(2)
                > 24
        );
    }

    #[test]
    fn unpack_matches_compressed_tensors_offset_binary_order() {
        // [-8,-7,-1,0,1,6,7,-3] -> [0,1,7,8,9,14,15,5]
        let packed = 0x5fe9_8710;
        assert_eq!(
            unpack_int4(&[packed], 8).unwrap(),
            vec![-8, -7, -1, 0, 1, 6, 7, -3]
        );
    }

    #[test]
    fn checkpoint_bf16_scales_decode_without_native_endian_assumptions() {
        let bytes = [
            0x80, 0x3f, // 1.0
            0x00, 0xc0, // -2.0
            0x00, 0x3e, // 0.125
        ];
        assert_eq!(
            decode_float_values("BF16", &bytes).unwrap(),
            vec![1.0, -2.0, 0.125]
        );
    }

    #[test]
    fn groupwise_matvec_applies_one_scale_per_32_columns() {
        let values: Vec<i8> = (-8..8).cycle().take(64).collect();
        let mut packed = vec![0_u32; 8];
        for (index, value) in values.iter().enumerate() {
            packed[index / 8] |= ((*value + 8) as u32) << ((index % 8) * 4);
        }
        let input = vec![1.0; 64];
        let result = int4_group_matvec(&packed, &[0.5, 2.0], 1, 64, 32, &input).unwrap();
        let expected = values[..32]
            .iter()
            .map(|value| *value as f32 * 0.5)
            .sum::<f32>()
            + values[32..]
                .iter()
                .map(|value| *value as f32 * 2.0)
                .sum::<f32>();
        assert_close(&result, &[expected], 1e-6);
    }

    #[test]
    fn router_selects_with_bias_but_weights_with_uncorrected_scores() {
        let hidden = [1.0, 1.0];
        let gate = [
            3.0, 0.0, // high raw score
            1.0, 0.0, // selected through correction
            2.0, 0.0, 0.0, 0.0,
        ];
        let routes = route_topk(
            &hidden,
            &gate,
            Some(&[0.0, 5.0, 0.0, 0.0]),
            4,
            2,
            RouterSemantics::NoAuxTc,
            &RouterContract {
                score: RouterScoreKind::Sigmoid,
                normalize_selected: true,
                scaling_factor: 2.827,
                correction_bias: true,
                groups: 1,
                selected_groups: 1,
            },
        )
        .unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            vec![1, 0]
        );
        let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());
        let raw = [sigmoid(1.0), sigmoid(3.0)];
        assert_close(
            &routes.iter().map(|route| route.weight).collect::<Vec<_>>(),
            &[
                raw[0] / (raw[0] + raw[1]) * 2.827,
                raw[1] / (raw[0] + raw[1]) * 2.827,
            ],
            1e-6,
        );
    }

    #[test]
    fn rope_position_zero_performs_only_the_official_permutation() {
        let config = KimiK27Config::official();
        let input: Vec<_> = (0..64).map(|value| value as f32).collect();
        let output = apply_yarn_rope(&input, 0, &config).unwrap();
        let expected: Vec<_> = (0..32)
            .map(|index| (index * 2) as f32)
            .chain((0..32).map(|index| (index * 2 + 1) as f32))
            .collect();
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn mla_decode_uses_compact_latents_and_stable_softmax() {
        let mut config = KimiK27Config::official();
        config.num_attention_heads = 1;
        config.kv_lora_rank = 2;
        config.qk_nope_head_dim = 2;
        config.qk_rope_head_dim = 2;
        config.v_head_dim = 1;
        config.rope_factor = 1.0;
        let mut cache = MlaKvCache::default();
        cache
            .push(
                MlaCacheEntry {
                    latent: vec![1.0, 0.0],
                    rope_key: vec![1.0, 0.0],
                },
                &config,
            )
            .unwrap();
        cache
            .push(
                MlaCacheEntry {
                    latent: vec![0.0, 1.0],
                    rope_key: vec![0.0, 1.0],
                },
                &config,
            )
            .unwrap();
        // rows: k0, k1, v
        let kv_b = [1.0, 0.0, 0.0, 1.0, 2.0, 4.0];
        let output =
            mla_decode_reference(&[1.0, 0.0], &[1.0, 0.0], &cache, &kv_b, &config).unwrap();
        let scale = 0.5; // 1/sqrt(q_head_dim=4)
        let p0 = (2.0_f32 * scale).exp();
        let p1 = 0.0_f32.exp();
        let expected = (p0 * 2.0 + p1 * 4.0) / (p0 + p1);
        assert_close(&output, &[expected], 1e-6);
    }

    #[test]
    fn absorbed_mla_streams_each_kv_projection_row_once() {
        let mut config = KimiK27Config::official();
        config.num_attention_heads = 1;
        config.kv_lora_rank = 2;
        config.qk_nope_head_dim = 2;
        config.qk_rope_head_dim = 2;
        config.v_head_dim = 1;
        config.rope_factor = 1.0;
        let mut cache = CompactMlaKvCache::default();
        cache.push(&[1.0, 0.0], &[1.0, 0.0], &config).unwrap();
        cache.push(&[0.0, 1.0], &[0.0, 1.0], &config).unwrap();
        assert_eq!(cache.bytes(), 16);

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-absorbed-mla-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let kv_b = [1.0_f32, 0.0, 0.0, 1.0, 2.0, 4.0];
        let bytes: Vec<u8> = kv_b
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        std::fs::write(root.join("weights.bin"), &bytes).unwrap();
        let store = DenseTensorStore::new(
            &root,
            vec![],
            [TensorSegment {
                tensor: "kv_b".into(),
                dtype: Some("BF16".into()),
                shape: vec![3, 2],
                shard: "weights.bin".into(),
                offset: 0,
                length: bytes.len() as u64,
            }],
        );
        let output = mla_decode_absorbed_bf16(
            &store,
            "kv_b",
            &[1.0, 0.0],
            &[1.0, 0.0],
            &cache,
            &config,
            4,
            crate::bf16_tile_cpu,
            crate::bf16_transpose_tile_cpu,
        )
        .unwrap();
        let mut oracle_cache = MlaKvCache::default();
        oracle_cache
            .push(
                MlaCacheEntry {
                    latent: vec![1.0, 0.0],
                    rope_key: vec![1.0, 0.0],
                },
                &config,
            )
            .unwrap();
        oracle_cache
            .push(
                MlaCacheEntry {
                    latent: vec![0.0, 1.0],
                    rope_key: vec![0.0, 1.0],
                },
                &config,
            )
            .unwrap();
        let expected =
            mla_decode_reference(&[1.0, 0.0], &[1.0, 0.0], &oracle_cache, &kv_b, &config).unwrap();
        assert_close(&output.values, &expected, 1e-6);
        assert_eq!(output.metrics.storage_bytes, bytes.len() as u64);
        assert_eq!(output.metrics.peak_tile_bytes, 4);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_executor_completes_a_dense_decoder_token() {
        let mut config = KimiK27Config::official();
        config.hidden_size = 2;
        config.intermediate_size = 2;
        config.moe_intermediate_size = 2;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 1;
        config.n_routed_experts = 1;
        config.n_shared_experts = 1;
        config.num_experts_per_tok = 1;
        config.first_k_dense_replace = 1;
        config.q_lora_rank = 2;
        config.kv_lora_rank = 2;
        config.qk_nope_head_dim = 2;
        config.qk_rope_head_dim = 2;
        config.v_head_dim = 2;
        config.vocab_size = 3;
        config.group_size = 2;
        config.rope_factor = 1.0;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-kimi-step-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let mut payload = Vec::new();
        let mut tensors = Vec::new();
        let mut append = |name: &str, shape: &[u64], values: &[f32]| {
            let offset = payload.len() as u64;
            payload.extend(
                values
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes()),
            );
            tensors.push(TensorSegment {
                tensor: name.into(),
                dtype: Some("BF16".into()),
                shape: shape.to_vec(),
                shard: "weights.bin".into(),
                offset,
                length: payload.len() as u64 - offset,
            });
        };
        let identity = [1.0, 0.0, 0.0, 1.0];
        append(
            "language_model.model.embed_tokens.weight",
            &[3, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
        );
        append("language_model.model.norm.weight", &[2], &[1.0, 1.0]);
        append(
            "language_model.lm_head.weight",
            &[3, 2],
            &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
        );
        let prefix = "language_model.model.layers.0";
        append(
            &format!("{prefix}.input_layernorm.weight"),
            &[2],
            &[1.0, 1.0],
        );
        append(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[2],
            &[1.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.q_a_proj.weight"),
            &[2, 2],
            &identity,
        );
        append(
            &format!("{prefix}.self_attn.q_a_layernorm.weight"),
            &[2],
            &[1.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.q_b_proj.weight"),
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
            &[2],
            &[1.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.kv_b_proj.weight"),
            &[4, 2],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        );
        append(
            &format!("{prefix}.self_attn.o_proj.weight"),
            &[2, 2],
            &identity,
        );
        append(
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[2, 2],
            &identity,
        );
        append(&format!("{prefix}.mlp.up_proj.weight"), &[2, 2], &identity);
        append(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[2, 2],
            &identity,
        );
        std::fs::write(root.join("weights.bin"), &payload).unwrap();
        std::fs::write(root.join("experts.bin"), b"x").unwrap();
        let dense = DenseTensorStore::new(&root, vec![], tensors);
        let experts = Arc::new(crate::ExpertStore::new(
            &root,
            vec![],
            [ExpertLocation {
                layer: 0,
                expert: 0,
                segments: vec![TensorSegment {
                    tensor: "unused".into(),
                    dtype: Some("I32".into()),
                    shape: vec![1],
                    shard: "experts.bin".into(),
                    offset: 0,
                    length: 1,
                }],
            }],
        ));
        let residency = Arc::new(crate::ResidencyManager::new(
            experts,
            Arc::new(crate::NoAccelerator),
            1,
            0,
        ));
        let scheduler = StreamingScheduler::with_inflight_budget(residency, 0, 1).unwrap();
        let executor =
            KimiOneTokenExecutor::new(&config, &dense, &scheduler, KimiExecutionBackend::Cpu, 8)
                .unwrap();
        let mut state = KimiDecodeState::new(&config);
        let (hidden, metrics) = executor.forward_token(0, &mut state).unwrap();
        assert_eq!(hidden.len(), 2);
        assert!(hidden.iter().all(|value| value.is_finite()));
        assert_eq!(state.position(), 1);
        assert_eq!(state.bytes(), 8);
        assert!(metrics.dense_storage_bytes > 0);
        assert!(metrics.peak_dense_tile_bytes <= 8);
        let (token, head_metrics) = executor.greedy_token(&hidden).unwrap();
        assert!(token < 3);
        assert!(head_metrics.peak_dense_tile_bytes <= 8);

        let mut sequential_state = KimiDecodeState::new(&config);
        let (first, first_metrics) = executor.forward_token(0, &mut sequential_state).unwrap();
        let (second, second_metrics) = executor.forward_token(1, &mut sequential_state).unwrap();
        let mut expected = first;
        expected.extend(second);
        let mut batch_state = KimiDecodeState::new(&config);
        let (batch, batch_metrics) = executor.forward_tokens(&[0, 1], &mut batch_state).unwrap();
        assert_close(&batch, &expected, 1e-5);
        assert_eq!(batch_state, sequential_state);
        assert_eq!(batch_state.position(), 2);
        assert!(
            batch_metrics.dense_storage_bytes
                < first_metrics.dense_storage_bytes + second_metrics.dense_storage_bytes
        );
        let (batch_tokens, _) = executor.greedy_tokens(&batch, 2).unwrap();
        assert_eq!(batch_tokens.len(), 2);
        assert!(batch_tokens.iter().all(|token| *token < 3));
        let accepted_draft = [batch_tokens[0]];
        let mut accepted_state = KimiDecodeState::new(&config);
        let accepted = executor
            .verify_greedy_draft(0, &accepted_draft, &mut accepted_state)
            .unwrap();
        assert_eq!(accepted.verification.accepted_draft_tokens, 1);
        assert!(accepted.verification.used_bonus_token);
        assert_eq!(accepted_state.position(), 2);
        let rejected_draft = [(batch_tokens[0] + 1) % 3];
        let mut rejected_state = KimiDecodeState::new(&config);
        let rejected = executor
            .verify_greedy_draft(0, &rejected_draft, &mut rejected_state)
            .unwrap();
        assert_eq!(rejected.verification.accepted_draft_tokens, 0);
        assert!(!rejected.verification.used_bonus_token);
        assert_eq!(rejected_state.position(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn batched_moe_routes_union_once_and_matches_sequential_positions() {
        let mut config = KimiK27Config::official();
        config.hidden_size = 32;
        config.intermediate_size = 32;
        config.moe_intermediate_size = 32;
        config.q_lora_rank = 2;
        config.kv_lora_rank = 2;
        config.n_routed_experts = 2;
        config.num_experts_per_tok = 1;
        config.n_shared_experts = 1;
        config.group_size = 32;
        config.n_group = 1;
        config.topk_group = 1;

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-kimi-moe-batch-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let mut dense_payload = Vec::new();
        let mut dense_tensors = Vec::new();
        let mut append_dense = |name: &str, shape: &[u64], dtype: &str, bytes: Vec<u8>| {
            let offset = dense_payload.len() as u64;
            dense_payload.extend(bytes);
            dense_tensors.push(TensorSegment {
                tensor: name.into(),
                dtype: Some(dtype.into()),
                shape: shape.to_vec(),
                shard: "dense.bin".into(),
                offset,
                length: dense_payload.len() as u64 - offset,
            });
        };
        let bf16_bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let mut router = vec![1.0_f32; 32];
        router.extend(vec![-1.0_f32; 32]);
        append_dense("moe.gate.weight", &[2, 32], "BF16", bf16_bytes(&router));
        append_dense(
            "moe.gate.e_score_correction_bias",
            &[2],
            "F32",
            [0.0_f32, 0.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
        let zeros = vec![0.0_f32; 32 * 32];
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            append_dense(
                &format!("moe.shared_experts.{projection}.weight"),
                &[32, 32],
                "BF16",
                bf16_bytes(&zeros),
            );
        }
        std::fs::write(root.join("dense.bin"), &dense_payload).unwrap();

        let mut expert_payload = Vec::new();
        let mut locations = Vec::new();
        for expert in 0..2_u32 {
            let mut segments = Vec::new();
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let quantized = expert as u32 + 1;
                let nibble = quantized + 8;
                let word = (0..8).fold(0_u32, |word, index| word | (nibble << (index * 4)));
                let packed = vec![word; 32 * 4];
                let scales = vec![(1.0_f32.to_bits() >> 16) as u16; 32];
                for (suffix, dtype, shape, bytes) in [
                    (
                        "weight_packed",
                        "I32",
                        vec![32, 4],
                        packed
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "weight_scale",
                        "BF16",
                        vec![32, 1],
                        scales
                            .iter()
                            .flat_map(|value| value.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                    (
                        "weight_shape",
                        "I32",
                        vec![2],
                        [32_i32, 32_i32]
                            .into_iter()
                            .flat_map(i32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    ),
                ] {
                    let offset = expert_payload.len() as u64;
                    expert_payload.extend(bytes);
                    segments.push(TensorSegment {
                        tensor: format!(
                            "language_model.model.layers.0.mlp.experts.{expert}.{projection}.{suffix}"
                        ),
                        dtype: Some(dtype.into()),
                        shape,
                        shard: "experts.bin".into(),
                        offset,
                        length: expert_payload.len() as u64 - offset,
                    });
                }
            }
            locations.push(ExpertLocation {
                layer: 0,
                expert,
                segments,
            });
        }
        std::fs::write(root.join("experts.bin"), &expert_payload).unwrap();
        let dense = DenseTensorStore::new(&root, vec![], dense_tensors);
        let expert_store = Arc::new(crate::ExpertStore::new(&root, vec![], locations));
        let one_expert_bytes = (expert_payload.len() / 2) as u64;
        let residency = Arc::new(crate::ResidencyManager::new(
            expert_store,
            Arc::new(crate::NoAccelerator),
            one_expert_bytes,
            0,
        ));
        let scheduler =
            StreamingScheduler::with_inflight_budget(residency, 0, one_expert_bytes).unwrap();
        let executor =
            KimiOneTokenExecutor::new(&config, &dense, &scheduler, KimiExecutionBackend::Cpu, 128)
                .unwrap();
        let mut hidden = vec![1.0_f32; 32];
        hidden.extend(vec![-1.0_f32; 32]);
        let mut batch_metrics = KimiStepMetrics::default();
        let batch = executor
            .forward_moe_batch(0, "moe", 2, &hidden, &mut batch_metrics)
            .unwrap();
        let mut expected = Vec::new();
        let mut sequential_storage = 0_u64;
        for position in hidden.chunks_exact(32) {
            let mut metrics = KimiStepMetrics::default();
            expected.extend(
                executor
                    .forward_moe(0, "moe", position, &mut metrics)
                    .unwrap(),
            );
            sequential_storage += metrics.dense_storage_bytes;
        }
        assert_close(&batch, &expected, 1e-5);
        assert_eq!(batch_metrics.expert_waves, 2);
        assert!(batch_metrics.dense_storage_bytes < sequential_storage);
        std::fs::remove_dir_all(root).unwrap();
    }
}
