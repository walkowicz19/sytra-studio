//! Architecture-neutral MoE reference math.
//!
//! Family adapters bind checkpoint tensors and select the exact semantics;
//! this module implements the common, oracle-friendly operations once.

use std::{cmp::Ordering, mem::size_of, sync::Arc};

use thiserror::Error;

use crate::manifest::{
    ActivationKind, ExpertLocation, RouterContract, RouterScoreKind, RouterSemantics, TensorSegment,
};
use crate::{Accelerator, AcceleratorBuffer, CudaAccelerator};

#[derive(Debug, Error, PartialEq)]
pub enum MoeMathError {
    #[error("invalid MoE shape: {0}")]
    Shape(String),
    #[error("unsupported router contract: {0}")]
    Router(String),
    #[error("unsupported weight representation: {0}")]
    Weight(String),
    #[error("MoE execution failed: {0}")]
    Execution(String),
}

#[derive(Debug)]
pub struct GatedExpertBinding<'a> {
    pub gate: Option<&'a TensorSegment>,
    pub up: Option<&'a TensorSegment>,
    pub fused_gate_up: Option<&'a TensorSegment>,
    pub down: &'a TensorSegment,
}

/// Resolve the projection matrices used by the dominant gated-MLP layouts:
/// gate/up/down, Mixtral w1/w3/w2, and DBRX w1/v1/w2. Quantization metadata
/// remains addressable as auxiliary segments and is never mistaken for a matrix.
pub fn bind_gated_expert(
    location: &ExpertLocation,
) -> Result<GatedExpertBinding<'_>, MoeMathError> {
    let mut gate = None;
    let mut up = None;
    let mut fused = None;
    let mut down = None;
    for segment in &location.segments {
        let name = segment.tensor.to_ascii_lowercase();
        if name.contains("scale") || name.contains("zero_point") || name.contains("weight_shape") {
            continue;
        }
        let tail = name.rsplit('.').next().unwrap_or(&name);
        if name.contains("gate_up_proj") || name.contains("gate_up.weight") {
            assign_once(&mut fused, segment, "fused gate/up")?;
        } else if name.contains("gate_proj") || tail == "w1" || name.ends_with(".w1.weight") {
            assign_once(&mut gate, segment, "gate/w1")?;
        } else if name.contains("up_proj")
            || tail == "w3"
            || tail == "v1"
            || name.ends_with(".w3.weight")
            || name.ends_with(".v1.weight")
        {
            assign_once(&mut up, segment, "up/w3/v1")?;
        } else if name.contains("down_proj") || tail == "w2" || name.ends_with(".w2.weight") {
            assign_once(&mut down, segment, "down/w2")?;
        }
    }
    let down = down.ok_or_else(|| MoeMathError::Weight("expert is missing down/w2".into()))?;
    if fused.is_some() == (gate.is_some() || up.is_some()) {
        return Err(MoeMathError::Weight(
            "expert must contain either fused gate/up or separate gate and up projections".into(),
        ));
    }
    if fused.is_none() && (gate.is_none() || up.is_none()) {
        return Err(MoeMathError::Weight(
            "expert is missing gate or up projection".into(),
        ));
    }
    Ok(GatedExpertBinding {
        gate,
        up,
        fused_gate_up: fused,
        down,
    })
}

fn assign_once<'a>(
    slot: &mut Option<&'a TensorSegment>,
    value: &'a TensorSegment,
    role: &str,
) -> Result<(), MoeMathError> {
    if slot.replace(value).is_some() {
        return Err(MoeMathError::Weight(format!(
            "expert has duplicate {role} matrices"
        )));
    }
    Ok(())
}

/// Decode portable floating checkpoint values for CPU reference oracles.
pub fn decode_float_values(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, MoeMathError> {
    let upper = dtype.to_ascii_uppercase();
    match upper.as_str() {
        "F32" => decode_chunks(bytes, 4, |chunk| {
            f32::from_le_bytes(chunk.try_into().unwrap())
        }),
        "BF16" => decode_chunks(bytes, 2, |chunk| {
            f32::from_bits(u32::from(u16::from_le_bytes(chunk.try_into().unwrap())) << 16)
        }),
        "F16" => decode_chunks(bytes, 2, |chunk| {
            f16_to_f32(u16::from_le_bytes(chunk.try_into().unwrap()))
        }),
        "F8_E4M3" | "F8_E4M3FN" | "FP8_E4M3" => Ok(bytes
            .iter()
            .map(|value| fp8_e4m3fn_to_f32(*value))
            .collect()),
        _ => Err(MoeMathError::Weight(format!(
            "dtype {dtype:?} is not a floating checkpoint type"
        ))),
    }
}

fn decode_chunks<F>(bytes: &[u8], width: usize, decode: F) -> Result<Vec<f32>, MoeMathError>
where
    F: Fn(&[u8]) -> f32,
{
    if bytes.len() % width != 0 {
        return Err(MoeMathError::Weight(format!(
            "{} bytes are not aligned to {width}",
            bytes.len()
        )));
    }
    Ok(bytes.chunks_exact(width).map(decode).collect())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let result = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let shift = mantissa.leading_zeros() - 6;
            let normalized = u32::from(mantissa) << (shift + 1);
            sign | ((127 - 15 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (u32::from(mantissa) << 13)
    } else {
        sign | (u32::from(exponent + (127 - 15)) << 23) | (u32::from(mantissa) << 13)
    };
    f32::from_bits(result)
}

fn fp8_e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        sign * f32::from(mantissa) * 2.0_f32.powi(-9)
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoERoute {
    pub expert: usize,
    pub weight: f32,
}

/// General top-k router covering the common Mixtral/Qwen/OLMoE softmax path
/// and DeepSeek/Kimi sigmoid + correction-bias path.
pub fn route_topk(
    hidden: &[f32],
    gate: &[f32],
    correction_bias: Option<&[f32]>,
    experts: usize,
    top_k: usize,
    semantics: RouterSemantics,
    contract: &RouterContract,
) -> Result<Vec<MoERoute>, MoeMathError> {
    if hidden.is_empty() || experts == 0 || gate.len() != experts * hidden.len() {
        return Err(MoeMathError::Shape(
            "router dimensions are inconsistent".into(),
        ));
    }
    let logits: Vec<f32> = gate
        .chunks_exact(hidden.len())
        .map(|row| row.iter().zip(hidden).map(|(w, x)| w * x).sum())
        .collect();
    route_topk_logits(&logits, correction_bias, top_k, semantics, contract)
}

/// Select routes from router logits that were already computed by a native
/// dense kernel. This prevents adapters from accidentally applying the gate
/// matrix twice and lets a batched router reuse one streamed matrix scan.
pub fn route_topk_logits(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    top_k: usize,
    semantics: RouterSemantics,
    contract: &RouterContract,
) -> Result<Vec<MoERoute>, MoeMathError> {
    let experts = logits.len();
    if experts == 0
        || top_k == 0
        || top_k > experts
        || contract.groups == 0
        || contract.selected_groups == 0
        || contract.selected_groups > contract.groups
        || experts % contract.groups as usize != 0
    {
        return Err(MoeMathError::Shape(
            "router logits and group dimensions are inconsistent".into(),
        ));
    }
    if contract.correction_bias && correction_bias.map(|bias| bias.len()) != Some(experts) {
        return Err(MoeMathError::Shape(
            "correction-bias router requires one bias per expert".into(),
        ));
    }
    let scores = match contract.score {
        RouterScoreKind::Softmax => softmax(&logits),
        RouterScoreKind::Sigmoid => logits.iter().map(|value| sigmoid(*value)).collect(),
    };
    let choice: Vec<f32> = match correction_bias {
        Some(bias) if contract.correction_bias => scores
            .iter()
            .zip(bias)
            .map(|(score, bias)| score + bias)
            .collect(),
        _ => scores.clone(),
    };

    let groups = contract.groups as usize;
    let selected_groups = contract.selected_groups as usize;
    let experts_per_group = experts / groups;
    let allowed = if groups > 1 || selected_groups < groups {
        let mut group_scores = Vec::with_capacity(groups);
        for group in 0..groups {
            let start = group * experts_per_group;
            let mut group_choice = choice[start..start + experts_per_group].to_vec();
            group_choice.sort_by(descending_f32);
            // DeepSeek noaux_tc defines group strength as the sum of its top 2.
            group_scores.push(group_choice[0] + group_choice.get(1).copied().unwrap_or_default());
        }
        let chosen = top_indices(&group_scores, selected_groups);
        let mut mask = vec![false; experts];
        for group in chosen {
            mask[group * experts_per_group..(group + 1) * experts_per_group].fill(true);
        }
        mask
    } else {
        vec![true; experts]
    };
    let masked: Vec<_> = choice
        .iter()
        .zip(allowed)
        .map(|(score, allowed)| if allowed { *score } else { f32::NEG_INFINITY })
        .collect();
    let selected = top_indices(&masked, top_k);
    let must_normalize =
        contract.normalize_selected || matches!(semantics, RouterSemantics::TopKNormalized);
    let denominator = if must_normalize {
        selected.iter().map(|index| scores[*index]).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    Ok(selected
        .into_iter()
        .map(|expert| MoERoute {
            expert,
            weight: scores[expert] / denominator * contract.scaling_factor,
        })
        .collect())
}

pub fn apply_activation(kind: ActivationKind, value: f32) -> f32 {
    match kind {
        ActivationKind::Silu => value * sigmoid(value),
        ActivationKind::Gelu => value * 0.5 * (1.0 + erf_approx(value / std::f32::consts::SQRT_2)),
        ActivationKind::GeluTanh => {
            0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value * value * value)).tanh())
        }
        ActivationKind::Relu => value.max(0.0),
        ActivationKind::Relu2 => value.max(0.0).powi(2),
    }
}

/// Architecture-neutral RMS normalization used by Kimi, DeepSeek, Qwen,
/// Mixtral, and other decoder families.
pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>, MoeMathError> {
    if input.is_empty() || input.len() != weight.len() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(MoeMathError::Shape(
            "invalid RMSNorm dimensions or epsilon".into(),
        ));
    }
    let variance = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let inverse = 1.0 / (variance + epsilon).sqrt();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * inverse * weight)
        .collect())
}

#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix<'a> {
    pub values: &'a [f32],
    pub rows: usize,
    pub cols: usize,
}

impl DenseMatrix<'_> {
    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>, MoeMathError> {
        if input.len() != self.cols || self.values.len() != self.rows * self.cols {
            return Err(MoeMathError::Shape(
                "dense matvec dimensions are invalid".into(),
            ));
        }
        Ok(self
            .values
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
            .collect())
    }
}

/// SwiGLU/GEGLU-style expert: down(activation(gate(x)) * up(x)).
pub fn gated_expert_reference(
    hidden: &[f32],
    gate: DenseMatrix<'_>,
    up: DenseMatrix<'_>,
    down: DenseMatrix<'_>,
    activation: ActivationKind,
) -> Result<Vec<f32>, MoeMathError> {
    let mut gate_output = gate.matvec(hidden)?;
    let up_output = up.matvec(hidden)?;
    if gate_output.len() != up_output.len() || down.cols != gate_output.len() {
        return Err(MoeMathError::Shape(
            "expert projection dimensions are incompatible".into(),
        ));
    }
    for (gate, up) in gate_output.iter_mut().zip(up_output) {
        *gate = apply_activation(activation.clone(), *gate) * up;
    }
    down.matvec(&gate_output)
}

#[derive(Debug, Clone, PartialEq)]
pub struct StandardKvEntry {
    /// `[kv_heads, head_dim]`
    pub key: Vec<f32>,
    /// `[kv_heads, value_dim]`
    pub value: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct StandardKvCache {
    entries: Vec<CompactStandardKvEntry>,
    device: Option<DeviceStandardKvCache>,
}

#[derive(Debug, Clone, PartialEq)]
struct CompactStandardKvEntry {
    key: Vec<u16>,
    value: Vec<u16>,
}

#[derive(Debug)]
struct DeviceStandardKvCache {
    cuda: Arc<CudaAccelerator>,
    keys: Option<AcceleratorBuffer>,
    values: Option<AcceleratorBuffer>,
    len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
}

impl Drop for DeviceStandardKvCache {
    fn drop(&mut self) {
        if let Some(values) = self.values.take() {
            self.cuda.release(&values);
        }
        if let Some(keys) = self.keys.take() {
            self.cuda.release(&keys);
        }
    }
}

impl PartialEq for StandardKvCache {
    fn eq(&self, other: &Self) -> bool {
        if self.entries != other.entries {
            return false;
        }
        match (&self.device, &other.device) {
            (None, None) => true,
            (Some(left), Some(right)) => {
                left.len == right.len
                    && left.capacity == right.capacity
                    && left.kv_heads == right.kv_heads
                    && left.head_dim == right.head_dim
                    && left.value_dim == right.value_dim
                    && left.keys == right.keys
                    && left.values == right.values
            }
            _ => false,
        }
    }
}

impl StandardKvCache {
    pub fn device(
        cuda: Arc<CudaAccelerator>,
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Self {
        Self {
            entries: Vec::new(),
            device: Some(DeviceStandardKvCache {
                cuda,
                keys: None,
                values: None,
                len: 0,
                capacity,
                kv_heads,
                head_dim,
                value_dim,
            }),
        }
    }

    pub fn push(
        &mut self,
        entry: StandardKvEntry,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Result<(), MoeMathError> {
        if entry.key.len() != kv_heads * head_dim || entry.value.len() != kv_heads * value_dim {
            return Err(MoeMathError::Shape("invalid standard KV entry".into()));
        }
        if let Some(device) = &mut self.device {
            if device.kv_heads != kv_heads
                || device.head_dim != head_dim
                || device.value_dim != value_dim
                || device.len >= device.capacity
            {
                return Err(MoeMathError::Shape(
                    "device standard KV cache dimensions or capacity are invalid".into(),
                ));
            }
            let keys: Vec<u16> = entry.key.into_iter().map(f32_to_bf16).collect();
            let values: Vec<u16> = entry.value.into_iter().map(f32_to_bf16).collect();
            device.ensure_allocated().map_err(MoeMathError::Execution)?;
            let key_buffer = device.keys.as_ref().ok_or_else(|| {
                MoeMathError::Execution("device KV key allocation is missing".into())
            })?;
            let value_buffer = device.values.as_ref().ok_or_else(|| {
                MoeMathError::Execution("device KV value allocation is missing".into())
            })?;
            let key_offset = device.len * kv_heads * head_dim * size_of::<u16>();
            let value_offset = device.len * kv_heads * value_dim * size_of::<u16>();
            let key_bytes: Vec<u8> = keys.iter().flat_map(|value| value.to_le_bytes()).collect();
            let value_bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            device
                .cuda
                .write_buffer(key_buffer, key_offset, &key_bytes)
                .map_err(MoeMathError::Execution)?;
            device
                .cuda
                .write_buffer(value_buffer, value_offset, &value_bytes)
                .map_err(MoeMathError::Execution)?;
            device.len += 1;
            return Ok(());
        }
        self.entries.push(CompactStandardKvEntry {
            key: entry.key.into_iter().map(f32_to_bf16).collect(),
            value: entry.value.into_iter().map(f32_to_bf16).collect(),
        });
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.device
            .as_ref()
            .map(|device| device.len)
            .unwrap_or(self.entries.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        if let Some(device) = &self.device {
            return device
                .keys
                .as_ref()
                .map(|buffer| buffer.bytes)
                .unwrap_or(0)
                .saturating_add(
                    device
                        .values
                        .as_ref()
                        .map(|buffer| buffer.bytes)
                        .unwrap_or(0),
                );
        }
        self.entries
            .iter()
            .map(|entry| (entry.key.len() + entry.value.len()) * size_of::<u16>())
            .sum()
    }

    pub fn truncate(&mut self, positions: usize) {
        if let Some(device) = &mut self.device {
            device.len = device.len.min(positions);
        } else {
            self.entries.truncate(positions);
        }
    }
}

impl DeviceStandardKvCache {
    fn ensure_allocated(&mut self) -> Result<(), String> {
        if self.keys.is_some() && self.values.is_some() {
            return Ok(());
        }
        let key_bytes = self
            .capacity
            .checked_mul(self.kv_heads)
            .and_then(|count| count.checked_mul(self.head_dim))
            .and_then(|count| count.checked_mul(size_of::<u16>()))
            .ok_or_else(|| "device KV key allocation overflow".to_string())?;
        let value_bytes = self
            .capacity
            .checked_mul(self.kv_heads)
            .and_then(|count| count.checked_mul(self.value_dim))
            .and_then(|count| count.checked_mul(size_of::<u16>()))
            .ok_or_else(|| "device KV value allocation overflow".to_string())?;
        let keys = self.cuda.allocate(key_bytes)?;
        let values = match self.cuda.allocate(value_bytes) {
            Ok(values) => values,
            Err(error) => {
                self.cuda.release(&keys);
                return Err(error);
            }
        };
        self.keys = Some(keys);
        self.values = Some(values);
        Ok(())
    }
}

fn f32_to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

/// One-token MHA/GQA decode reference. Query layout is `[heads, head_dim]`.
pub fn standard_attention_decode(
    query: &[f32],
    cache: &StandardKvCache,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Result<Vec<f32>, MoeMathError> {
    standard_attention_decode_window(
        query,
        cache,
        heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn standard_attention_decode_window(
    query: &[f32],
    cache: &StandardKvCache,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    window: Option<usize>,
) -> Result<Vec<f32>, MoeMathError> {
    if heads == 0
        || kv_heads == 0
        || heads % kv_heads != 0
        || query.len() != heads * head_dim
        || cache.is_empty()
    {
        return Err(MoeMathError::Shape(
            "standard attention dimensions are invalid".into(),
        ));
    }
    if let Some(device) = &cache.device {
        let keys = device
            .keys
            .as_ref()
            .ok_or_else(|| MoeMathError::Execution("device KV key allocation is missing".into()))?;
        let values = device.values.as_ref().ok_or_else(|| {
            MoeMathError::Execution("device KV value allocation is missing".into())
        })?;
        return device
            .cuda
            .standard_attention_bf16(
                keys,
                values,
                device.capacity,
                device.len,
                query,
                heads,
                kv_heads,
                head_dim,
                value_dim,
                softmax_scale,
                window,
            )
            .map_err(MoeMathError::Execution);
    }
    let heads_per_kv = heads / kv_heads;
    let start = window
        .filter(|window| *window > 0)
        .map(|window| cache.entries.len().saturating_sub(window))
        .unwrap_or(0);
    let visible = &cache.entries[start..];
    let mut output = vec![0.0; heads * value_dim];
    for head in 0..heads {
        let kv_head = head / heads_per_kv;
        let q = &query[head * head_dim..(head + 1) * head_dim];
        let mut logits: Vec<f32> = visible
            .iter()
            .map(|entry| {
                q.iter()
                    .zip(&entry.key[kv_head * head_dim..(kv_head + 1) * head_dim])
                    .map(|(q, k)| q * bf16_to_f32(*k))
                    .sum::<f32>()
                    * softmax_scale
            })
            .collect();
        softmax_in_place(&mut logits);
        for (probability, entry) in logits.into_iter().zip(visible) {
            let value = &entry.value[kv_head * value_dim..(kv_head + 1) * value_dim];
            for index in 0..value_dim {
                output[head * value_dim + index] += probability * bf16_to_f32(value[index]);
            }
        }
    }
    Ok(output)
}

/// Hugging Face/Llama-style half-rotation RoPE used by Mixtral and Qwen.
pub fn apply_standard_rope(
    values: &[f32],
    position: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
) -> Result<Vec<f32>, MoeMathError> {
    if heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || values.len() != heads * head_dim
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(MoeMathError::Shape(
            "invalid rotary dimensions or theta".into(),
        ));
    }
    let half = head_dim / 2;
    let mut output = vec![0.0; values.len()];
    for head in 0..heads {
        let base = head * head_dim;
        for index in 0..half {
            let frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let left = values[base + index];
            let right = values[base + half + index];
            output[base + index] = left * cos - right * sin;
            output[base + half + index] = right * cos + left * sin;
        }
    }
    Ok(output)
}

pub fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let mut output = values.to_vec();
    softmax_in_place(&mut output);
    output
}

pub(crate) fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        denominator += *value;
    }
    for value in values {
        *value /= denominator;
    }
}

// Abramowitz-Stegun approximation; sufficient for a portable f32 oracle.
fn erf_approx(value: f32) -> f32 {
    let sign = value.signum();
    let x = value.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let polynomial = (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
        * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - polynomial * (-x * x).exp())
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|left, right| {
        descending_f32(&values[*left], &values[*right]).then_with(|| left.cmp(right))
    });
    indices.truncate(count);
    indices
}

fn descending_f32(left: &f32, right: &f32) -> Ordering {
    right.total_cmp(left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert!((left - right).abs() <= tolerance, "{left} != {right}");
        }
    }

    #[test]
    fn softmax_router_covers_mixtral_and_qwen_normalization() {
        let contract = RouterContract {
            score: RouterScoreKind::Softmax,
            normalize_selected: true,
            ..RouterContract::default()
        };
        let routes = route_topk(
            &[1.0],
            &[3.0, 2.0, 1.0, 0.0],
            None,
            4,
            2,
            RouterSemantics::TopKSoftmax,
            &contract,
        )
        .unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_close(
            &routes.iter().map(|route| route.weight).collect::<Vec<_>>(),
            &[0.731_058_6, 0.268_941_43],
            1e-6,
        );
        assert_eq!(
            routes,
            route_topk_logits(
                &[3.0, 2.0, 1.0, 0.0],
                None,
                2,
                RouterSemantics::TopKSoftmax,
                &contract,
            )
            .unwrap()
        );
    }

    #[test]
    fn sigmoid_group_router_uses_bias_only_for_selection() {
        let contract = RouterContract {
            score: RouterScoreKind::Sigmoid,
            normalize_selected: true,
            scaling_factor: 2.0,
            correction_bias: true,
            groups: 2,
            selected_groups: 1,
        };
        let routes = route_topk(
            &[1.0],
            &[4.0, 3.0, 2.0, 1.0],
            Some(&[-10.0, -10.0, 10.0, 10.0]),
            4,
            2,
            RouterSemantics::NoAuxTc,
            &contract,
        )
        .unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            [2, 3]
        );
        assert!((routes.iter().map(|route| route.weight).sum::<f32>() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn standard_gqa_attention_reuses_each_kv_head_for_its_query_group() {
        let mut cache = StandardKvCache::default();
        cache
            .push(
                StandardKvEntry {
                    key: vec![1.0, 0.0, 0.0, 1.0],
                    value: vec![2.0, 4.0],
                },
                2,
                2,
                1,
            )
            .unwrap();
        let output = standard_attention_decode(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            &cache,
            4,
            2,
            2,
            1,
            1.0 / 2.0_f32.sqrt(),
        )
        .unwrap();
        assert_eq!(output, vec![2.0, 2.0, 4.0, 4.0]);
        assert_eq!(cache.bytes(), 12);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn standard_rope_uses_half_rotation_and_windowed_kv_is_bounded() {
        let rotated = apply_standard_rope(&[1.0, 2.0, 3.0, 4.0], 0, 1, 4, 10_000.0).unwrap();
        assert_eq!(rotated, [1.0, 2.0, 3.0, 4.0]);

        let mut cache = StandardKvCache::default();
        for value in [1.0, 2.0, 4.0] {
            cache
                .push(
                    StandardKvEntry {
                        key: vec![1.0, 0.0],
                        value: vec![value],
                    },
                    1,
                    2,
                    1,
                )
                .unwrap();
        }
        let output =
            standard_attention_decode_window(&[1.0, 0.0], &cache, 1, 1, 2, 1, 1.0, Some(1))
                .unwrap();
        assert_eq!(output, [4.0]);
        cache.truncate(1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn decodes_bf16_f16_and_fp8_reference_values() {
        assert_eq!(decode_float_values("BF16", &[0x80, 0x3f]).unwrap(), [1.0]);
        assert_eq!(decode_float_values("F16", &[0x00, 0x3c]).unwrap(), [1.0]);
        assert_eq!(
            decode_float_values("F8_E4M3", &[0x38, 0xb8]).unwrap(),
            [1.0, -1.0]
        );
        assert!(decode_float_values("F16", &[0]).is_err());
    }

    #[test]
    fn shared_rms_norm_matches_the_reference_formula() {
        let output = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 1e-5).unwrap();
        let inverse = 1.0 / (12.5_f32 + 1e-5).sqrt();
        assert_close(&output, &[3.0 * inverse, 8.0 * inverse], 1e-6);
    }

    #[test]
    fn binds_qwen_mixtral_and_dbrx_projection_names() {
        let segment = |tensor: &str| TensorSegment {
            tensor: tensor.into(),
            dtype: Some("BF16".into()),
            shape: vec![2, 2],
            shard: PathBuf::from("model.safetensors"),
            offset: 1,
            length: 8,
        };
        for names in [
            ["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
            ["w1", "w3", "w2"],
            ["w1", "v1", "w2"],
        ] {
            let location = ExpertLocation {
                layer: 0,
                expert: 0,
                segments: names.into_iter().map(segment).collect(),
            };
            let binding = bind_gated_expert(&location).unwrap();
            assert!(binding.gate.is_some());
            assert!(binding.up.is_some());
            assert!(binding.fused_gate_up.is_none());
        }
    }
}
