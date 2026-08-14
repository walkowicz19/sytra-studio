//! Shared bounded execution primitives for standard rotary-GQA MoE decoders.

use std::sync::Arc;

use thiserror::Error;

use crate::{
    apply_activation, int4_group32_bf16_matmul_cpu, AcceleratorBuffer, ActivationKind,
    CudaAccelerator, ResidentExpert, ResidentTensor, StandardKvCache,
};

#[derive(Debug, Error, PartialEq)]
pub enum StandardMoeError {
    #[error("standard MoE tensor contract is invalid: {0}")]
    Contract(String),
    #[error("standard MoE execution failed: {0}")]
    Execution(String),
}

#[derive(Debug, Default, PartialEq)]
pub struct StandardMoeKvState {
    pub layers: Vec<StandardKvCache>,
}

impl StandardMoeKvState {
    pub fn new(layers: usize) -> Self {
        Self {
            layers: (0..layers).map(|_| StandardKvCache::default()).collect(),
        }
    }

    pub fn new_device(
        layers: usize,
        cuda: Arc<CudaAccelerator>,
        capacity: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Self {
        Self {
            layers: (0..layers)
                .map(|_| {
                    StandardKvCache::device(cuda.clone(), capacity, kv_heads, head_dim, value_dim)
                })
                .collect(),
        }
    }

    pub fn position(&self) -> usize {
        self.layers.first().map(StandardKvCache::len).unwrap_or(0)
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(StandardKvCache::bytes).sum()
    }

    pub fn truncate(&mut self, positions: usize) {
        for layer in &mut self.layers {
            layer.truncate(positions);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MatrixRef<'a> {
    bytes: Option<&'a [u8]>,
    dtype: &'a str,
    rows: usize,
    cols: usize,
    resident_offset: usize,
}

#[derive(Debug, Clone, Copy)]
struct PackedInt4MatrixRef<'a> {
    packed: Option<&'a [u8]>,
    scales: Option<&'a [u8]>,
    packed_offset: usize,
    scale_offset: usize,
    rows: usize,
    cols: usize,
}

/// Execute one floating-point gated expert without expanding its immutable
/// weight payload to FP32. The only allocations are the planner-accounted
/// gathered input, two intermediate activations, and hidden-size output.
pub fn floating_gated_expert_batch(
    bytes: &[u8],
    tensors: &[ResidentTensor],
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
) -> Result<Vec<f32>, StandardMoeError> {
    floating_gated_expert_batch_with_cuda(
        bytes,
        tensors,
        hidden_size,
        intermediate_size,
        positions,
        hidden,
        activation,
        None,
    )
}

/// Execute a floating-point gated expert with optional bounded CUDA BF16
/// projections. Unsupported floating dtypes remain on the scalar CPU path.
#[allow(clippy::too_many_arguments)]
pub fn floating_gated_expert_batch_with_cuda(
    bytes: &[u8],
    tensors: &[ResidentTensor],
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
    cuda: Option<&CudaAccelerator>,
) -> Result<Vec<f32>, StandardMoeError> {
    floating_gated_expert_batch_impl(
        Some(bytes),
        tensors,
        hidden_size,
        intermediate_size,
        positions,
        hidden,
        activation,
        cuda,
        None,
    )
}

/// Execute a gated BF16 expert directly from its persistent CUDA allocation
/// when available, falling back to its bounded host lease otherwise.
#[allow(clippy::too_many_arguments)]
pub fn floating_gated_expert_batch_resident(
    resident: &ResidentExpert,
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
    cuda: Option<&CudaAccelerator>,
) -> Result<Vec<f32>, StandardMoeError> {
    floating_gated_expert_batch_impl(
        resident.host_bytes.as_deref(),
        &resident.tensors,
        hidden_size,
        intermediate_size,
        positions,
        hidden,
        activation,
        cuda,
        resident.accelerator_buffer(),
    )
}

/// Execute either an exact floating expert or compressed-tensors packed
/// symmetric INT4 group-32 expert from a bounded resident lease.
#[allow(clippy::too_many_arguments)]
pub fn standard_gated_expert_batch_resident(
    resident: &ResidentExpert,
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
    cuda: Option<&CudaAccelerator>,
) -> Result<Vec<f32>, StandardMoeError> {
    if resident
        .tensors
        .iter()
        .any(|tensor| tensor.name.to_ascii_lowercase().ends_with("weight_packed"))
    {
        packed_int4_gated_expert_batch_impl(
            resident.host_bytes.as_deref(),
            &resident.tensors,
            hidden_size,
            intermediate_size,
            positions,
            hidden,
            activation,
            cuda,
            resident.accelerator_buffer(),
        )
    } else {
        floating_gated_expert_batch_resident(
            resident,
            hidden_size,
            intermediate_size,
            positions,
            hidden,
            activation,
            cuda,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn floating_gated_expert_batch_impl(
    bytes: Option<&[u8]>,
    tensors: &[ResidentTensor],
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
    cuda: Option<&CudaAccelerator>,
    resident: Option<&AcceleratorBuffer>,
) -> Result<Vec<f32>, StandardMoeError> {
    if positions == 0 || hidden.len() != positions * hidden_size {
        return Err(StandardMoeError::Contract(
            "expert input positions do not match hidden size".into(),
        ));
    }
    let fused = find_tensor(tensors, |name| {
        name.contains("gate_up_proj")
            || name.contains("gate_up.weight")
            || name.contains("input_linear")
    });
    let gate = find_tensor(tensors, |name| {
        !name.contains("gate_up_proj")
            && (name.contains("gate_proj") || name.ends_with(".w1") || name.ends_with(".w1.weight"))
    });
    let up = find_tensor(tensors, |name| {
        !name.contains("gate_up_proj")
            && (name.contains("up_proj")
                || name.ends_with(".w3")
                || name.ends_with(".w3.weight")
                || name.ends_with(".v1")
                || name.ends_with(".v1.weight"))
    });
    let down = find_tensor(tensors, |name| {
        name.contains("down_proj")
            || name.contains("output_linear")
            || name.ends_with(".w2")
            || name.ends_with(".w2.weight")
    })
    .ok_or_else(|| StandardMoeError::Contract("expert down projection is missing".into()))?;

    let (mut gate_output, up_output) = if let Some(fused) = fused {
        if gate.is_some() || up.is_some() {
            return Err(StandardMoeError::Contract(
                "expert mixes fused and separate gate/up projections".into(),
            ));
        }
        let matrix = matrix(bytes, fused, 2 * intermediate_size, hidden_size)?;
        (
            matmul_rows(
                matrix,
                0,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
            matmul_rows(
                matrix,
                intermediate_size,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
        )
    } else {
        let gate = gate.ok_or_else(|| {
            StandardMoeError::Contract("expert gate projection is missing".into())
        })?;
        let up =
            up.ok_or_else(|| StandardMoeError::Contract("expert up projection is missing".into()))?;
        (
            matmul(
                matrix(bytes, gate, intermediate_size, hidden_size)?,
                positions,
                hidden,
                cuda,
                resident,
            )?,
            matmul(
                matrix(bytes, up, intermediate_size, hidden_size)?,
                positions,
                hidden,
                cuda,
                resident,
            )?,
        )
    };
    for (gate, up) in gate_output.iter_mut().zip(up_output) {
        *gate = apply_activation(activation.clone(), *gate) * up;
    }
    matmul(
        matrix(bytes, down, hidden_size, intermediate_size)?,
        positions,
        &gate_output,
        cuda,
        resident,
    )
}

#[allow(clippy::too_many_arguments)]
fn packed_int4_gated_expert_batch_impl(
    bytes: Option<&[u8]>,
    tensors: &[ResidentTensor],
    hidden_size: usize,
    intermediate_size: usize,
    positions: usize,
    hidden: &[f32],
    activation: ActivationKind,
    cuda: Option<&CudaAccelerator>,
    resident: Option<&AcceleratorBuffer>,
) -> Result<Vec<f32>, StandardMoeError> {
    if positions == 0 || hidden.len() != positions * hidden_size {
        return Err(StandardMoeError::Contract(
            "expert input positions do not match hidden size".into(),
        ));
    }
    let fused_predicate =
        |name: &str| name.contains("gate_up_proj") || name.contains("gate_up.weight");
    let gate_predicate = |name: &str| {
        !fused_predicate(name)
            && (name.contains("gate_proj")
                || name.ends_with(".w1.weight_packed")
                || name.ends_with(".w1.weight_scale")
                || name.ends_with(".w1.weight_shape"))
    };
    let up_predicate = |name: &str| {
        !fused_predicate(name)
            && (name.contains("up_proj")
                || name.ends_with(".w3.weight_packed")
                || name.ends_with(".w3.weight_scale")
                || name.ends_with(".w3.weight_shape")
                || name.ends_with(".v1.weight_packed")
                || name.ends_with(".v1.weight_scale")
                || name.ends_with(".v1.weight_shape"))
    };
    let down_predicate = |name: &str| {
        name.contains("down_proj")
            || name.ends_with(".w2.weight_packed")
            || name.ends_with(".w2.weight_scale")
            || name.ends_with(".w2.weight_shape")
    };

    let has_fused = has_packed_projection(tensors, fused_predicate);
    let has_gate = has_packed_projection(tensors, gate_predicate);
    let has_up = has_packed_projection(tensors, up_predicate);
    if has_fused == (has_gate || has_up) || (!has_fused && !(has_gate && has_up)) {
        return Err(StandardMoeError::Contract(
            "packed expert must have either fused or separate gate/up projections".into(),
        ));
    }
    let down = packed_matrix(
        bytes,
        tensors,
        down_predicate,
        hidden_size,
        intermediate_size,
    )?;
    let (mut gate_output, up_output) = if has_fused {
        let matrix = packed_matrix(
            bytes,
            tensors,
            fused_predicate,
            2 * intermediate_size,
            hidden_size,
        )?;
        (
            packed_matmul_rows(
                matrix,
                0,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
            packed_matmul_rows(
                matrix,
                intermediate_size,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
        )
    } else {
        (
            packed_matmul_rows(
                packed_matrix(
                    bytes,
                    tensors,
                    gate_predicate,
                    intermediate_size,
                    hidden_size,
                )?,
                0,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
            packed_matmul_rows(
                packed_matrix(bytes, tensors, up_predicate, intermediate_size, hidden_size)?,
                0,
                intermediate_size,
                positions,
                hidden,
                cuda,
                resident,
            )?,
        )
    };
    for (gate, up) in gate_output.iter_mut().zip(up_output) {
        *gate = apply_activation(activation.clone(), *gate) * up;
    }
    packed_matmul_rows(
        down,
        0,
        hidden_size,
        positions,
        &gate_output,
        cuda,
        resident,
    )
}

fn has_packed_projection(tensors: &[ResidentTensor], predicate: impl Fn(&str) -> bool) -> bool {
    tensors.iter().any(|tensor| {
        let name = tensor.name.to_ascii_lowercase();
        name.ends_with("weight_packed") && predicate(&name)
    })
}

fn packed_matrix<'a>(
    payload: Option<&'a [u8]>,
    tensors: &'a [ResidentTensor],
    predicate: impl Fn(&str) -> bool + Copy,
    rows: usize,
    cols: usize,
) -> Result<PackedInt4MatrixRef<'a>, StandardMoeError> {
    if !cols.is_multiple_of(32) {
        return Err(StandardMoeError::Contract(
            "packed INT4 group-32 projection width is not divisible by 32".into(),
        ));
    }
    let find = |suffix: &str| {
        tensors.iter().find(|tensor| {
            let name = tensor.name.to_ascii_lowercase();
            name.ends_with(suffix) && predicate(&name)
        })
    };
    let packed = find("weight_packed")
        .ok_or_else(|| StandardMoeError::Contract("packed expert weight is missing".into()))?;
    let scales = find("weight_scale")
        .ok_or_else(|| StandardMoeError::Contract("packed expert scale is missing".into()))?;
    let shape = find("weight_shape").ok_or_else(|| {
        StandardMoeError::Contract("packed expert logical shape is missing".into())
    })?;
    let words_per_row = cols.div_ceil(8);
    let groups_per_row = cols / 32;
    validate_packed_tensor(
        packed,
        "I32",
        &[rows as u64, words_per_row as u64],
        rows * words_per_row * 4,
    )?;
    validate_packed_tensor(
        scales,
        "BF16",
        &[rows as u64, groups_per_row as u64],
        rows * groups_per_row * 2,
    )?;
    validate_packed_tensor(shape, "I32", &[2], 8)?;

    let slice = |tensor: &ResidentTensor| {
        payload
            .map(|payload| {
                payload
                    .get(tensor.offset..tensor.offset.saturating_add(tensor.length))
                    .ok_or_else(|| {
                        StandardMoeError::Contract(format!("{} exceeds its payload", tensor.name))
                    })
            })
            .transpose()
    };
    let packed_bytes = slice(packed)?;
    let scale_bytes = slice(scales)?;
    if let Some(logical) = slice(shape)? {
        let logical_rows = i32::from_le_bytes(logical[0..4].try_into().expect("validated shape"));
        let logical_cols = i32::from_le_bytes(logical[4..8].try_into().expect("validated shape"));
        if logical_rows != rows as i32 || logical_cols != cols as i32 {
            return Err(StandardMoeError::Contract(format!(
                "packed expert logical shape [{logical_rows}, {logical_cols}] does not match [{rows}, {cols}]"
            )));
        }
    }
    Ok(PackedInt4MatrixRef {
        packed: packed_bytes,
        scales: scale_bytes,
        packed_offset: packed.offset,
        scale_offset: scales.offset,
        rows,
        cols,
    })
}

fn validate_packed_tensor(
    tensor: &ResidentTensor,
    dtype: &str,
    shape: &[u64],
    length: usize,
) -> Result<(), StandardMoeError> {
    if tensor.dtype.as_deref() != Some(dtype) || tensor.shape != shape || tensor.length != length {
        return Err(StandardMoeError::Contract(format!(
            "{} must be {dtype} {shape:?} with {length} bytes",
            tensor.name
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn packed_matmul_rows(
    matrix: PackedInt4MatrixRef<'_>,
    first_row: usize,
    rows: usize,
    positions: usize,
    input: &[f32],
    cuda: Option<&CudaAccelerator>,
    resident: Option<&AcceleratorBuffer>,
) -> Result<Vec<f32>, StandardMoeError> {
    if first_row.saturating_add(rows) > matrix.rows || input.len() != positions * matrix.cols {
        return Err(StandardMoeError::Contract(
            "packed matrix batch dimensions are inconsistent".into(),
        ));
    }
    let words_per_row = matrix.cols.div_ceil(8);
    let groups_per_row = matrix.cols / 32;
    let packed_start = first_row * words_per_row * 4;
    let packed_end = (first_row + rows) * words_per_row * 4;
    let scale_start = first_row * groups_per_row * 2;
    let scale_end = (first_row + rows) * groups_per_row * 2;
    if let Some(cuda) = cuda {
        if let Some(resident) = resident {
            return cuda
                .resident_int4_group32_bf16_matmul(
                    resident,
                    matrix.packed_offset + packed_start,
                    matrix.scale_offset + scale_start,
                    rows,
                    matrix.cols,
                    positions,
                    input,
                )
                .map_err(StandardMoeError::Execution);
        }
        let packed = matrix.packed.ok_or_else(|| {
            StandardMoeError::Execution("packed expert has no host or CUDA lease".into())
        })?;
        let scales = matrix.scales.ok_or_else(|| {
            StandardMoeError::Execution("packed expert has no host scales".into())
        })?;
        return cuda
            .int4_group32_bf16_bytes_matmul(
                &packed[packed_start..packed_end],
                &scales[scale_start..scale_end],
                rows,
                matrix.cols,
                positions,
                input,
            )
            .map_err(StandardMoeError::Execution);
    }
    let packed = matrix.packed.ok_or_else(|| {
        StandardMoeError::Execution("packed expert has no executable weight lease".into())
    })?;
    let scales = matrix.scales.ok_or_else(|| {
        StandardMoeError::Execution("packed expert has no executable scale lease".into())
    })?;
    let packed = &packed[packed_start..packed_end];
    let scales = &scales[scale_start..scale_end];
    int4_group32_bf16_matmul_cpu(packed, scales, rows, matrix.cols, positions, input)
        .map_err(|error| StandardMoeError::Execution(error.to_string()))
}

fn find_tensor(
    tensors: &[ResidentTensor],
    predicate: impl Fn(&str) -> bool,
) -> Option<&ResidentTensor> {
    tensors.iter().find(|tensor| {
        let name = tensor.name.to_ascii_lowercase();
        !name.contains("scale") && !name.contains("zero_point") && predicate(&name)
    })
}

fn matrix<'a>(
    payload: Option<&'a [u8]>,
    tensor: &'a ResidentTensor,
    rows: usize,
    cols: usize,
) -> Result<MatrixRef<'a>, StandardMoeError> {
    if tensor.shape != [rows as u64, cols as u64] {
        return Err(StandardMoeError::Contract(format!(
            "tensor {} has shape {:?}, expected [{rows}, {cols}]",
            tensor.name, tensor.shape
        )));
    }
    let dtype = tensor
        .dtype
        .as_deref()
        .ok_or_else(|| StandardMoeError::Contract(format!("{} has no dtype", tensor.name)))?;
    let bytes = payload
        .map(|payload| {
            payload
                .get(tensor.offset..tensor.offset.saturating_add(tensor.length))
                .ok_or_else(|| {
                    StandardMoeError::Contract(format!("{} exceeds its payload", tensor.name))
                })
        })
        .transpose()?;
    let scalar_bytes = match dtype.to_ascii_uppercase().as_str() {
        "BF16" | "F16" => 2,
        "F32" => 4,
        other => {
            return Err(StandardMoeError::Contract(format!(
                "floating expert dtype {other} is unsupported"
            )))
        }
    };
    if tensor.length != rows * cols * scalar_bytes {
        return Err(StandardMoeError::Contract(format!(
            "{} byte length does not match its matrix shape",
            tensor.name
        )));
    }
    Ok(MatrixRef {
        bytes,
        dtype,
        rows,
        cols,
        resident_offset: tensor.offset,
    })
}

fn matmul(
    matrix: MatrixRef<'_>,
    positions: usize,
    input: &[f32],
    cuda: Option<&CudaAccelerator>,
    resident: Option<&AcceleratorBuffer>,
) -> Result<Vec<f32>, StandardMoeError> {
    matmul_rows(matrix, 0, matrix.rows, positions, input, cuda, resident)
}

fn matmul_rows(
    matrix: MatrixRef<'_>,
    first_row: usize,
    rows: usize,
    positions: usize,
    input: &[f32],
    cuda: Option<&CudaAccelerator>,
    resident: Option<&AcceleratorBuffer>,
) -> Result<Vec<f32>, StandardMoeError> {
    if first_row.saturating_add(rows) > matrix.rows || input.len() != positions * matrix.cols {
        return Err(StandardMoeError::Contract(
            "matrix batch dimensions are inconsistent".into(),
        ));
    }
    let scalar_bytes = match matrix.dtype.to_ascii_uppercase().as_str() {
        "BF16" | "F16" => 2,
        "F32" => 4,
        _ => unreachable!("matrix dtype was validated"),
    };
    let start = first_row * matrix.cols * scalar_bytes;
    let end = (first_row + rows) * matrix.cols * scalar_bytes;
    if matrix.dtype.eq_ignore_ascii_case("BF16") {
        if let Some(cuda) = cuda {
            if let Some(resident) = resident {
                return cuda
                    .resident_bf16_matmul(
                        resident,
                        matrix.resident_offset + start,
                        rows,
                        matrix.cols,
                        positions,
                        input,
                    )
                    .map_err(StandardMoeError::Execution);
            }
            let weights = matrix.bytes.ok_or_else(|| {
                StandardMoeError::Execution("BF16 expert has no host or CUDA lease".into())
            })?;
            return cuda
                .bf16_matmul_bytes(&weights[start..end], rows, matrix.cols, positions, input)
                .map_err(StandardMoeError::Execution);
        }
    }
    let weights = matrix.bytes.ok_or_else(|| {
        StandardMoeError::Execution("floating expert has no executable weight lease".into())
    })?;
    let weights = &weights[start..end];
    let mut output = vec![0.0; positions * rows];
    for position in 0..positions {
        let input = &input[position * matrix.cols..(position + 1) * matrix.cols];
        for row in 0..rows {
            let mut sum = 0.0;
            for (column, value) in input.iter().enumerate() {
                sum += decode_scalar(matrix.dtype, weights, row * matrix.cols + column)? * value;
            }
            output[position * rows + row] = sum;
        }
    }
    Ok(output)
}

fn decode_scalar(dtype: &str, bytes: &[u8], index: usize) -> Result<f32, StandardMoeError> {
    match dtype.to_ascii_uppercase().as_str() {
        "BF16" => {
            let offset = index * 2;
            let raw = bytes
                .get(offset..offset + 2)
                .ok_or_else(|| StandardMoeError::Execution("BF16 matrix is truncated".into()))?;
            Ok(f32::from_bits(
                u32::from(u16::from_le_bytes([raw[0], raw[1]])) << 16,
            ))
        }
        "F16" => {
            let offset = index * 2;
            let raw = bytes
                .get(offset..offset + 2)
                .ok_or_else(|| StandardMoeError::Execution("F16 matrix is truncated".into()))?;
            Ok(f16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]])))
        }
        "F32" => {
            let offset = index * 4;
            let raw = bytes
                .get(offset..offset + 4)
                .ok_or_else(|| StandardMoeError::Execution("F32 matrix is truncated".into()))?;
            Ok(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        }
        other => Err(StandardMoeError::Contract(format!(
            "floating expert dtype {other} is unsupported"
        ))),
    }
}

fn f16_bits_to_f32(bits: u16) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn append_packed_projection(
        payload: &mut Vec<u8>,
        tensors: &mut Vec<ResidentTensor>,
        name: &str,
        rows: usize,
        cols: usize,
        phase: usize,
    ) -> Vec<f32> {
        let quantized: Vec<i8> = (0..rows * cols)
            .map(|index| ((index + phase) % 15) as i8 - 7)
            .collect();
        let values: Vec<f32> = quantized
            .iter()
            .map(|value| *value as f32 * 0.0625)
            .collect();
        let packed_offset = payload.len();
        let words_per_row = cols.div_ceil(8);
        for row in 0..rows {
            for word in 0..words_per_row {
                let mut packed = 0_u32;
                for nibble in 0..8 {
                    let column = word * 8 + nibble;
                    if column < cols {
                        let value = quantized[row * cols + column];
                        packed |= u32::from((value + 8) as u8) << (nibble * 4);
                    }
                }
                payload.extend_from_slice(&packed.to_le_bytes());
            }
        }
        tensors.push(ResidentTensor {
            name: format!("{name}.weight_packed"),
            dtype: Some("I32".into()),
            shape: vec![rows as u64, words_per_row as u64],
            offset: packed_offset,
            length: rows * words_per_row * 4,
        });
        let scale_offset = payload.len();
        let scale = (0.0625_f32.to_bits() >> 16) as u16;
        for _ in 0..rows * (cols / 32) {
            payload.extend_from_slice(&scale.to_le_bytes());
        }
        tensors.push(ResidentTensor {
            name: format!("{name}.weight_scale"),
            dtype: Some("BF16".into()),
            shape: vec![rows as u64, (cols / 32) as u64],
            offset: scale_offset,
            length: rows * (cols / 32) * 2,
        });
        let shape_offset = payload.len();
        payload.extend_from_slice(&(rows as i32).to_le_bytes());
        payload.extend_from_slice(&(cols as i32).to_le_bytes());
        tensors.push(ResidentTensor {
            name: format!("{name}.weight_shape"),
            dtype: Some("I32".into()),
            shape: vec![2],
            offset: shape_offset,
            length: 8,
        });
        values
    }

    #[test]
    fn fused_floating_expert_executes_a_position_batch_without_weight_expansion_contracts() {
        // gate rows are identity, up rows are constant one, down is identity.
        let mut bytes = bf16(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let down_offset = bytes.len();
        bytes.extend(bf16(&[1.0, 0.0, 0.0, 1.0]));
        let tensors = vec![
            ResidentTensor {
                name: "model.layers.0.block_sparse_moe.experts.gate_up_proj".into(),
                dtype: Some("BF16".into()),
                shape: vec![4, 2],
                offset: 0,
                length: down_offset,
            },
            ResidentTensor {
                name: "model.layers.0.block_sparse_moe.experts.down_proj".into(),
                dtype: Some("BF16".into()),
                shape: vec![2, 2],
                offset: down_offset,
                length: bytes.len() - down_offset,
            },
        ];
        let output = floating_gated_expert_batch(
            &bytes,
            &tensors,
            2,
            2,
            2,
            &[1.0, 2.0, 3.0, 4.0],
            ActivationKind::Silu,
        )
        .unwrap();
        assert!((output[0] - apply_activation(ActivationKind::Silu, 1.0)).abs() < 1e-6);
        assert!((output[3] - apply_activation(ActivationKind::Silu, 4.0) * 4.0).abs() < 1e-6);
    }

    #[test]
    fn fused_packed_int4_expert_matches_its_exact_dequantized_reference() {
        let hidden_size = 32;
        let intermediate_size = 32;
        let positions = 2;
        let mut payload = Vec::new();
        let mut tensors = Vec::new();
        let gate_up = append_packed_projection(
            &mut payload,
            &mut tensors,
            "expert.gate_up_proj",
            2 * intermediate_size,
            hidden_size,
            2,
        );
        let down = append_packed_projection(
            &mut payload,
            &mut tensors,
            "expert.down_proj",
            hidden_size,
            intermediate_size,
            7,
        );
        let input: Vec<f32> = (0..positions * hidden_size)
            .map(|index| (index as f32 - 19.0) / 23.0)
            .collect();
        let actual = packed_int4_gated_expert_batch_impl(
            Some(&payload),
            &tensors,
            hidden_size,
            intermediate_size,
            positions,
            &input,
            ActivationKind::Silu,
            None,
            None,
        )
        .unwrap();

        let mut floating = bf16(&gate_up);
        let down_offset = floating.len();
        floating.extend(bf16(&down));
        let reference = floating_gated_expert_batch(
            &floating,
            &[
                ResidentTensor {
                    name: "expert.gate_up_proj.weight".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![(2 * intermediate_size) as u64, hidden_size as u64],
                    offset: 0,
                    length: down_offset,
                },
                ResidentTensor {
                    name: "expert.down_proj.weight".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![hidden_size as u64, intermediate_size as u64],
                    offset: down_offset,
                    length: floating.len() - down_offset,
                },
            ],
            hidden_size,
            intermediate_size,
            positions,
            &input,
            ActivationKind::Silu,
        )
        .unwrap();
        let error = reference
            .iter()
            .zip(actual)
            .map(|(reference, actual)| (reference - actual).abs())
            .fold(0.0_f32, f32::max);
        assert!(error < 1e-5, "max absolute error {error}");
    }

    #[test]
    fn standard_kv_state_reports_exact_bf16_storage_and_truncates_all_layers() {
        let mut state = StandardMoeKvState::new(2);
        for layer in &mut state.layers {
            layer
                .push(
                    crate::StandardKvEntry {
                        key: vec![1.0, 2.0],
                        value: vec![3.0, 4.0],
                    },
                    1,
                    2,
                    2,
                )
                .unwrap();
        }
        assert_eq!(state.position(), 1);
        assert_eq!(state.bytes(), 16);
        state.truncate(0);
        assert_eq!(state.position(), 0);
    }
}
