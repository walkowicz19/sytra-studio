//! Bounded dense-tensor execution primitives.

use serde::Serialize;
use thiserror::Error;

use crate::{decode_float_values, DenseMatrix, DenseTensorStore, TensorStoreError};

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct DenseTileMetrics {
    pub tiles: u64,
    pub storage_bytes: u64,
    pub peak_tile_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TiledMatvecOutput {
    pub values: Vec<f32>,
    pub metrics: DenseTileMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TiledMatmulOutput {
    /// Position-major `[positions, rows]` output.
    pub values: Vec<f32>,
    pub positions: usize,
    pub rows: usize,
    pub metrics: DenseTileMetrics,
}

#[derive(Debug, Error)]
pub enum DenseExecutionError {
    #[error(transparent)]
    Storage(#[from] TensorStoreError),
    #[error("invalid dense tensor contract: {0}")]
    Contract(String),
    #[error("dense execution backend failed: {0}")]
    Backend(String),
}

/// Execute a row-major BF16 matrix-vector product in storage-backed row
/// tiles. At most `tile_bytes` of weight data is read and passed to the
/// backend at once.
pub fn tiled_bf16_matvec<F>(
    store: &DenseTensorStore,
    tensor: &str,
    tile_bytes: u64,
    input: &[f32],
    execute_tile: F,
) -> Result<TiledMatvecOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    tiled_bf16_matvec_rows(store, tensor, 0, rows, tile_bytes, input, execute_tile)
}

/// Multiply a storage-backed row-major BF16 matrix by a position-major batch
/// of FP32 activations. A weight tile is read once and reused for every
/// position, which is the required primitive for speculative target
/// verification to amortize dense-weight I/O.
pub fn tiled_bf16_matmul<F>(
    store: &DenseTensorStore,
    tensor: &str,
    tile_bytes: u64,
    positions: usize,
    input: &[f32],
    execute_tile: F,
) -> Result<TiledMatmulOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    tiled_bf16_matmul_rows(
        store,
        tensor,
        0,
        rows,
        tile_bytes,
        positions,
        input,
        execute_tile,
    )
}

/// Multiply an exact row-major F32 matrix in bounded storage tiles. This is
/// primarily the immutable-reference path for official tiny checkpoints;
/// no full matrix copy is created and CUDA residency is deliberately not
/// claimed without an F32 device kernel.
pub fn tiled_f32_matmul(
    store: &DenseTensorStore,
    tensor: &str,
    tile_bytes: u64,
    positions: usize,
    input: &[f32],
) -> Result<TiledMatmulOutput, DenseExecutionError> {
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("F32") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 F32 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    let cols = usize::try_from(metadata.shape[1])
        .map_err(|_| DenseExecutionError::Contract("column count exceeds usize".into()))?;
    if rows == 0 || cols == 0 || positions == 0 || input.len() != positions.saturating_mul(cols) {
        return Err(DenseExecutionError::Contract(
            "F32 matrix and input dimensions are incompatible".into(),
        ));
    }
    let row_bytes = cols
        .checked_mul(4)
        .ok_or_else(|| DenseExecutionError::Contract("F32 row size overflow".into()))?;
    if tile_bytes < row_bytes as u64 {
        return Err(DenseExecutionError::Contract(format!(
            "tile budget {tile_bytes} is smaller than one {row_bytes}-byte F32 row"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / row_bytes as u64)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut output = vec![0.0; positions * rows];
    let mut metrics = DenseTileMetrics::default();
    for first_row in (0..rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(rows - first_row);
        let length = tile_rows * row_bytes;
        let bytes = store.read_window(tensor, (first_row * row_bytes) as u64, length as u64)?;
        let weights = decode_float_values("F32", &bytes)
            .map_err(|error| DenseExecutionError::Backend(error.to_string()))?;
        for position in 0..positions {
            let activation = &input[position * cols..(position + 1) * cols];
            for row in 0..tile_rows {
                output[position * rows + first_row + row] = weights[row * cols..(row + 1) * cols]
                    .iter()
                    .zip(activation)
                    .map(|(weight, value)| weight * value)
                    .sum();
            }
        }
        metrics.tiles += 1;
        metrics.storage_bytes += length as u64;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(length as u64);
    }
    Ok(TiledMatmulOutput {
        values: output,
        positions,
        rows,
        metrics,
    })
}

/// Multiply an exact compressed-tensors packed symmetric INT4 group-32
/// matrix by a position-major FP32 batch. Packed rows and BF16 scale rows are
/// streamed together under one host tile budget; the full matrix is never
/// materialized or dequantized.
pub fn tiled_packed_int4_group32_bf16_matmul<F>(
    store: &DenseTensorStore,
    tensor: &str,
    tile_bytes: u64,
    positions: usize,
    input: &[f32],
    mut execute_tile: F,
) -> Result<TiledMatmulOutput, DenseExecutionError>
where
    F: FnMut(&[u8], &[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let prefix = tensor.strip_suffix(".weight").ok_or_else(|| {
        DenseExecutionError::Contract(format!(
            "packed dense tensor {tensor} must use a .weight logical name"
        ))
    })?;
    let packed_name = format!("{prefix}.weight_packed");
    let scale_name = format!("{prefix}.weight_scale");
    let shape_name = format!("{prefix}.weight_shape");
    let packed = store
        .metadata(&packed_name)
        .ok_or_else(|| TensorStoreError::Unknown(packed_name.clone()))?;
    let scales = store
        .metadata(&scale_name)
        .ok_or_else(|| TensorStoreError::Unknown(scale_name.clone()))?;
    let shape = store
        .metadata(&shape_name)
        .ok_or_else(|| TensorStoreError::Unknown(shape_name.clone()))?;
    if shape.dtype.as_deref() != Some("I32") || shape.shape != [2] || shape.length != 8 {
        return Err(DenseExecutionError::Contract(format!(
            "{shape_name} must be an I32 logical shape pair"
        )));
    }
    let logical = store.read(&shape_name)?;
    let rows_i32 = i32::from_le_bytes(logical[0..4].try_into().expect("validated shape bytes"));
    let cols_i32 = i32::from_le_bytes(logical[4..8].try_into().expect("validated shape bytes"));
    drop(logical);
    let rows = usize::try_from(rows_i32).map_err(|_| {
        DenseExecutionError::Contract(format!("{shape_name} has a non-positive row count"))
    })?;
    let cols = usize::try_from(cols_i32).map_err(|_| {
        DenseExecutionError::Contract(format!("{shape_name} has a non-positive column count"))
    })?;
    if rows == 0
        || cols == 0
        || positions == 0
        || !cols.is_multiple_of(32)
        || input.len() != positions.saturating_mul(cols)
    {
        return Err(DenseExecutionError::Contract(
            "packed dense matrix and input dimensions are incompatible".into(),
        ));
    }
    let words_per_row = cols.div_ceil(8);
    let groups_per_row = cols / 32;
    let packed_row_bytes = words_per_row
        .checked_mul(4)
        .ok_or_else(|| DenseExecutionError::Contract("packed row size overflow".into()))?;
    let scale_row_bytes = groups_per_row
        .checked_mul(2)
        .ok_or_else(|| DenseExecutionError::Contract("scale row size overflow".into()))?;
    if packed.dtype.as_deref() != Some("I32")
        || packed.shape != [rows as u64, words_per_row as u64]
        || packed.length != (rows * packed_row_bytes) as u64
        || scales.dtype.as_deref() != Some("BF16")
        || scales.shape != [rows as u64, groups_per_row as u64]
        || scales.length != (rows * scale_row_bytes) as u64
    {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} does not have the exact packed INT4/BF16 group-32 triplet"
        )));
    }
    let combined_row_bytes = packed_row_bytes
        .checked_add(scale_row_bytes)
        .ok_or_else(|| DenseExecutionError::Contract("combined row size overflow".into()))?;
    if tile_bytes < combined_row_bytes as u64 {
        return Err(DenseExecutionError::Contract(format!(
            "tile budget {tile_bytes} is smaller than one {combined_row_bytes}-byte packed row"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / combined_row_bytes as u64)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut output = vec![0.0_f32; positions * rows];
    let mut metrics = DenseTileMetrics {
        storage_bytes: 8,
        ..DenseTileMetrics::default()
    };
    for first_row in (0..rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(rows - first_row);
        let packed_length = tile_rows * packed_row_bytes;
        let scale_length = tile_rows * scale_row_bytes;
        let packed_bytes = store.read_window(
            &packed_name,
            (first_row * packed_row_bytes) as u64,
            packed_length as u64,
        )?;
        let scale_bytes = store.read_window(
            &scale_name,
            (first_row * scale_row_bytes) as u64,
            scale_length as u64,
        )?;
        let tile = execute_tile(
            &packed_bytes,
            &scale_bytes,
            tile_rows,
            cols,
            positions,
            input,
        )
        .map_err(DenseExecutionError::Backend)?;
        if tile.len() != positions * tile_rows {
            return Err(DenseExecutionError::Backend(format!(
                "backend returned {} values for a {positions}x{tile_rows} packed output tile",
                tile.len()
            )));
        }
        for position in 0..positions {
            let source = &tile[position * tile_rows..(position + 1) * tile_rows];
            let start = position * rows + first_row;
            output[start..start + tile_rows].copy_from_slice(source);
        }
        let tile_bytes = (packed_length + scale_length) as u64;
        metrics.tiles += 1;
        metrics.storage_bytes += tile_bytes;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(tile_bytes);
    }
    Ok(TiledMatmulOutput {
        values: output,
        positions,
        rows,
        metrics,
    })
}

pub fn tiled_bf16_matmul_rows<F>(
    store: &DenseTensorStore,
    tensor: &str,
    first_row: usize,
    selected_rows: usize,
    tile_bytes: u64,
    positions: usize,
    input: &[f32],
    mut execute_tile: F,
) -> Result<TiledMatmulOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    let cols = usize::try_from(metadata.shape[1])
        .map_err(|_| DenseExecutionError::Contract("column count exceeds usize".into()))?;
    let row_end = first_row.checked_add(selected_rows).ok_or_else(|| {
        DenseExecutionError::Contract("selected matrix row range overflow".into())
    })?;
    let input_elements = positions
        .checked_mul(cols)
        .ok_or_else(|| DenseExecutionError::Contract("batched input size overflow".into()))?;
    if rows == 0
        || cols == 0
        || selected_rows == 0
        || row_end > rows
        || positions == 0
        || input.len() != input_elements
    {
        return Err(DenseExecutionError::Contract(
            "matrix and batched input dimensions are incompatible".into(),
        ));
    }
    let row_bytes = (cols as u64)
        .checked_mul(2)
        .ok_or_else(|| DenseExecutionError::Contract("row byte size overflow".into()))?;
    let expected = row_bytes
        .checked_mul(rows as u64)
        .ok_or_else(|| DenseExecutionError::Contract("matrix byte size overflow".into()))?;
    if metadata.length != expected || tile_bytes < row_bytes {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} cannot execute under the {tile_bytes}-byte tile contract"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / row_bytes)
        .unwrap_or(usize::MAX)
        .max(1);
    let output_elements = positions
        .checked_mul(selected_rows)
        .ok_or_else(|| DenseExecutionError::Contract("batched output size overflow".into()))?;
    let mut output = vec![0.0_f32; output_elements];
    let mut metrics = DenseTileMetrics::default();
    for relative_row in (0..selected_rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(selected_rows - relative_row);
        let absolute_row = first_row + relative_row;
        let length = (tile_rows as u64) * row_bytes;
        let bytes = store.read_window(tensor, (absolute_row as u64) * row_bytes, length)?;
        let tile = execute_tile(&bytes, tile_rows, cols, positions, input)
            .map_err(DenseExecutionError::Backend)?;
        if tile.len() != positions * tile_rows {
            return Err(DenseExecutionError::Backend(format!(
                "backend returned {} values for a {positions}x{tile_rows} output tile",
                tile.len()
            )));
        }
        for position in 0..positions {
            let source = &tile[position * tile_rows..(position + 1) * tile_rows];
            let start = position * selected_rows + relative_row;
            output[start..start + tile_rows].copy_from_slice(source);
        }
        metrics.tiles += 1;
        metrics.storage_bytes += length;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(length);
    }
    Ok(TiledMatmulOutput {
        values: output,
        positions,
        rows: selected_rows,
        metrics,
    })
}

/// Batched transpose row-range multiplication. Input and output are both
/// position-major. Each weight row tile is read once for every position.
pub fn tiled_bf16_transpose_matmul_rows<F>(
    store: &DenseTensorStore,
    tensor: &str,
    first_row: usize,
    selected_rows: usize,
    tile_bytes: u64,
    positions: usize,
    input: &[f32],
    mut execute_tile: F,
) -> Result<TiledMatmulOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    let cols = usize::try_from(metadata.shape[1])
        .map_err(|_| DenseExecutionError::Contract("column count exceeds usize".into()))?;
    let row_end = first_row.checked_add(selected_rows).ok_or_else(|| {
        DenseExecutionError::Contract("selected transpose row range overflow".into())
    })?;
    let input_elements = positions
        .checked_mul(selected_rows)
        .ok_or_else(|| DenseExecutionError::Contract("transpose batch input overflow".into()))?;
    if cols == 0
        || selected_rows == 0
        || row_end > rows
        || positions == 0
        || input.len() != input_elements
    {
        return Err(DenseExecutionError::Contract(
            "transpose matrix row range and batched input are incompatible".into(),
        ));
    }
    let row_bytes = (cols as u64)
        .checked_mul(2)
        .ok_or_else(|| DenseExecutionError::Contract("row byte size overflow".into()))?;
    if tile_bytes < row_bytes {
        return Err(DenseExecutionError::Contract(format!(
            "tile budget {tile_bytes} is smaller than one {row_bytes}-byte matrix row"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / row_bytes)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut output = vec![0.0_f32; positions * cols];
    let mut metrics = DenseTileMetrics::default();
    for relative_row in (0..selected_rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(selected_rows - relative_row);
        let absolute_row = first_row + relative_row;
        let length = (tile_rows as u64) * row_bytes;
        let bytes = store.read_window(tensor, (absolute_row as u64) * row_bytes, length)?;
        let mut tile_input = Vec::with_capacity(positions * tile_rows);
        for position in 0..positions {
            let start = position * selected_rows + relative_row;
            tile_input.extend_from_slice(&input[start..start + tile_rows]);
        }
        let partial = execute_tile(&bytes, tile_rows, cols, positions, &tile_input)
            .map_err(DenseExecutionError::Backend)?;
        if partial.len() != positions * cols {
            return Err(DenseExecutionError::Backend(format!(
                "transpose backend returned {} values for a {positions}x{cols} output",
                partial.len()
            )));
        }
        for (sum, value) in output.iter_mut().zip(partial) {
            *sum += value;
        }
        metrics.tiles += 1;
        metrics.storage_bytes += length;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(length);
    }
    Ok(TiledMatmulOutput {
        values: output,
        positions,
        rows: cols,
        metrics,
    })
}

/// Execute a contiguous row range of a row-major BF16 matrix. This permits
/// MLA kernels to address the interleaved per-head K and V blocks without
/// reading unrelated rows.
pub fn tiled_bf16_matvec_rows<F>(
    store: &DenseTensorStore,
    tensor: &str,
    first_row: usize,
    selected_rows: usize,
    tile_bytes: u64,
    input: &[f32],
    mut execute_tile: F,
) -> Result<TiledMatvecOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    let cols = usize::try_from(metadata.shape[1])
        .map_err(|_| DenseExecutionError::Contract("column count exceeds usize".into()))?;
    let row_end = first_row.checked_add(selected_rows).ok_or_else(|| {
        DenseExecutionError::Contract("selected matrix row range overflow".into())
    })?;
    if rows == 0 || cols == 0 || selected_rows == 0 || row_end > rows || input.len() != cols {
        return Err(DenseExecutionError::Contract(
            "matrix row range and input dimensions are incompatible".into(),
        ));
    }
    let row_bytes = u64::try_from(cols)
        .ok()
        .and_then(|cols| cols.checked_mul(2))
        .ok_or_else(|| DenseExecutionError::Contract("row byte size overflow".into()))?;
    let expected = row_bytes
        .checked_mul(rows as u64)
        .ok_or_else(|| DenseExecutionError::Contract("matrix byte size overflow".into()))?;
    if metadata.length != expected {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} stores {} bytes, expected {expected}",
            metadata.length
        )));
    }
    if tile_bytes < row_bytes {
        return Err(DenseExecutionError::Contract(format!(
            "tile budget {tile_bytes} is smaller than one {row_bytes}-byte matrix row"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / row_bytes)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut output = Vec::with_capacity(selected_rows);
    let mut metrics = DenseTileMetrics::default();
    for relative_row in (0..selected_rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(selected_rows - relative_row);
        let absolute_row = first_row + relative_row;
        let offset = (absolute_row as u64) * row_bytes;
        let length = (tile_rows as u64) * row_bytes;
        let bytes = store.read_window(tensor, offset, length)?;
        let values =
            execute_tile(&bytes, tile_rows, cols, input).map_err(DenseExecutionError::Backend)?;
        if values.len() != tile_rows {
            return Err(DenseExecutionError::Backend(format!(
                "backend returned {} rows for a {tile_rows}-row tile",
                values.len()
            )));
        }
        output.extend(values);
        metrics.tiles += 1;
        metrics.storage_bytes += length;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(length);
    }
    Ok(TiledMatvecOutput {
        values: output,
        metrics,
    })
}

/// Execute `matrix_rows.transpose() * input` while retaining only one BF16
/// row tile at a time. Partial tile results are accumulated into the fixed
/// column-sized output.
pub fn tiled_bf16_transpose_matvec_rows<F>(
    store: &DenseTensorStore,
    tensor: &str,
    first_row: usize,
    selected_rows: usize,
    tile_bytes: u64,
    input: &[f32],
    mut execute_tile: F,
) -> Result<TiledMatvecOutput, DenseExecutionError>
where
    F: FnMut(&[u8], usize, usize, &[f32]) -> Result<Vec<f32>, String>,
{
    let metadata = store
        .metadata(tensor)
        .ok_or_else(|| TensorStoreError::Unknown(tensor.into()))?;
    if metadata.dtype.as_deref() != Some("BF16") || metadata.shape.len() != 2 {
        return Err(DenseExecutionError::Contract(format!(
            "{tensor} must be a rank-2 BF16 matrix"
        )));
    }
    let rows = usize::try_from(metadata.shape[0])
        .map_err(|_| DenseExecutionError::Contract("row count exceeds usize".into()))?;
    let cols = usize::try_from(metadata.shape[1])
        .map_err(|_| DenseExecutionError::Contract("column count exceeds usize".into()))?;
    let row_end = first_row.checked_add(selected_rows).ok_or_else(|| {
        DenseExecutionError::Contract("selected matrix row range overflow".into())
    })?;
    if cols == 0 || selected_rows == 0 || row_end > rows || input.len() != selected_rows {
        return Err(DenseExecutionError::Contract(
            "transpose matrix row range and input dimensions are incompatible".into(),
        ));
    }
    let row_bytes = (cols as u64)
        .checked_mul(2)
        .ok_or_else(|| DenseExecutionError::Contract("row byte size overflow".into()))?;
    if tile_bytes < row_bytes {
        return Err(DenseExecutionError::Contract(format!(
            "tile budget {tile_bytes} is smaller than one {row_bytes}-byte matrix row"
        )));
    }
    let rows_per_tile = usize::try_from(tile_bytes / row_bytes)
        .unwrap_or(usize::MAX)
        .max(1);
    let mut output = vec![0.0_f32; cols];
    let mut metrics = DenseTileMetrics::default();
    for relative_row in (0..selected_rows).step_by(rows_per_tile) {
        let tile_rows = rows_per_tile.min(selected_rows - relative_row);
        let absolute_row = first_row + relative_row;
        let length = (tile_rows as u64) * row_bytes;
        let bytes = store.read_window(tensor, (absolute_row as u64) * row_bytes, length)?;
        let partial = execute_tile(
            &bytes,
            tile_rows,
            cols,
            &input[relative_row..relative_row + tile_rows],
        )
        .map_err(DenseExecutionError::Backend)?;
        if partial.len() != cols {
            return Err(DenseExecutionError::Backend(format!(
                "transpose backend returned {} columns for a {cols}-column tile",
                partial.len()
            )));
        }
        for (sum, value) in output.iter_mut().zip(partial) {
            *sum += value;
        }
        metrics.tiles += 1;
        metrics.storage_bytes += length;
        metrics.peak_tile_bytes = metrics.peak_tile_bytes.max(length);
    }
    Ok(TiledMatvecOutput {
        values: output,
        metrics,
    })
}

/// CPU oracle for a BF16 tile; useful for adapter verification and systems
/// without a usable accelerator.
pub fn bf16_tile_cpu(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let values = decode_float_values("BF16", weights).map_err(|error| error.to_string())?;
    DenseMatrix {
        values: &values,
        rows,
        cols,
    }
    .matvec(input)
    .map_err(|error| error.to_string())
}

pub fn bf16_transpose_tile_cpu(
    weights: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if input.len() != rows {
        return Err("transpose BF16 tile input has the wrong row count".into());
    }
    let values = decode_float_values("BF16", weights).map_err(|error| error.to_string())?;
    if values.len() != rows * cols {
        return Err("transpose BF16 tile byte length is inconsistent".into());
    }
    let mut output = vec![0.0_f32; cols];
    for row in 0..rows {
        for column in 0..cols {
            output[column] += values[row * cols + column] * input[row];
        }
    }
    Ok(output)
}

pub fn bf16_tile_matmul_cpu(
    weights: &[u8],
    rows: usize,
    cols: usize,
    positions: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if positions == 0 || input.len() != positions * cols {
        return Err("BF16 tile batch input has inconsistent dimensions".into());
    }
    let values = decode_float_values("BF16", weights).map_err(|error| error.to_string())?;
    if values.len() != rows * cols {
        return Err("BF16 tile byte length is inconsistent".into());
    }
    let mut output = vec![0.0_f32; positions * rows];
    for position in 0..positions {
        let activation = &input[position * cols..(position + 1) * cols];
        for row in 0..rows {
            output[position * rows + row] = values[row * cols..(row + 1) * cols]
                .iter()
                .zip(activation)
                .map(|(weight, value)| weight * value)
                .sum();
        }
    }
    Ok(output)
}

pub fn bf16_transpose_tile_matmul_cpu(
    weights: &[u8],
    rows: usize,
    cols: usize,
    positions: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if positions == 0 || input.len() != positions * rows {
        return Err("transpose BF16 tile batch input has inconsistent dimensions".into());
    }
    let values = decode_float_values("BF16", weights).map_err(|error| error.to_string())?;
    if values.len() != rows * cols {
        return Err("transpose BF16 tile byte length is inconsistent".into());
    }
    let mut output = vec![0.0_f32; positions * cols];
    for position in 0..positions {
        let activation = &input[position * rows..(position + 1) * rows];
        for row in 0..rows {
            for column in 0..cols {
                output[position * cols + column] += values[row * cols + column] * activation[row];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorSegment;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn bf16_matrix_streams_by_rows_under_the_tile_cap() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-dense-tiles-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        fs::write(root.join("model.safetensors"), &bytes).unwrap();
        let store = DenseTensorStore::new(
            &root,
            vec![],
            [TensorSegment {
                tensor: "linear.weight".into(),
                dtype: Some("BF16".into()),
                shape: vec![3, 2],
                shard: "model.safetensors".into(),
                offset: 0,
                length: 12,
            }],
        );
        let output =
            tiled_bf16_matvec(&store, "linear.weight", 4, &[2.0, 1.0], bf16_tile_cpu).unwrap();
        assert_eq!(output.values, [4.0, 10.0, 16.0]);
        assert_eq!(output.metrics.tiles, 3);
        assert_eq!(output.metrics.peak_tile_bytes, 4);
        let transpose = tiled_bf16_transpose_matvec_rows(
            &store,
            "linear.weight",
            1,
            2,
            4,
            &[2.0, 3.0],
            bf16_transpose_tile_cpu,
        )
        .unwrap();
        assert_eq!(transpose.values, [21.0, 26.0]);
        assert_eq!(transpose.metrics.peak_tile_bytes, 4);
        let batch = tiled_bf16_matmul(
            &store,
            "linear.weight",
            4,
            2,
            &[2.0, 1.0, 1.0, 3.0],
            bf16_tile_matmul_cpu,
        )
        .unwrap();
        assert_eq!(batch.values, [4.0, 10.0, 16.0, 7.0, 15.0, 23.0]);
        assert_eq!(batch.metrics.storage_bytes, 12);
        assert_eq!(batch.metrics.peak_tile_bytes, 4);
        let transpose_batch = tiled_bf16_transpose_matmul_rows(
            &store,
            "linear.weight",
            1,
            2,
            4,
            2,
            &[2.0, 3.0, 1.0, 4.0],
            bf16_transpose_tile_matmul_cpu,
        )
        .unwrap();
        assert_eq!(transpose_batch.values, [21.0, 26.0, 23.0, 28.0]);
        assert_eq!(transpose_batch.metrics.storage_bytes, 8);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn f32_reference_matrix_streams_without_a_full_weight_copy() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-f32-dense-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        fs::write(root.join("weights.bin"), bytes).unwrap();
        let store = DenseTensorStore::new(
            &root,
            vec![],
            [TensorSegment {
                tensor: "linear.weight".into(),
                dtype: Some("F32".into()),
                shape: vec![3, 2],
                shard: "weights.bin".into(),
                offset: 0,
                length: 24,
            }],
        );
        let output =
            tiled_f32_matmul(&store, "linear.weight", 8, 2, &[2.0, 1.0, 1.0, 3.0]).unwrap();
        assert_eq!(output.values, [4.0, 10.0, 16.0, 7.0, 15.0, 23.0]);
        assert_eq!(output.metrics.tiles, 3);
        assert_eq!(output.metrics.storage_bytes, 24);
        assert_eq!(output.metrics.peak_tile_bytes, 8);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packed_int4_matrix_streams_weights_and_scales_under_one_tile_cap() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-packed-dense-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let rows = 3_usize;
        let cols = 32_usize;
        let quantized: Vec<i8> = (0..rows * cols)
            .map(|index| (index % 15) as i8 - 7)
            .collect();
        let mut packed = vec![0_u8; rows * (cols / 8) * 4];
        for row in 0..rows {
            for column in 0..cols {
                let word_offset = (row * (cols / 8) + column / 8) * 4;
                let mut word =
                    u32::from_le_bytes(packed[word_offset..word_offset + 4].try_into().unwrap());
                word |= u32::from((quantized[row * cols + column] + 8) as u8) << ((column % 8) * 4);
                packed[word_offset..word_offset + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        let scale = (0.0625_f32.to_bits() >> 16) as u16;
        let scales: Vec<u8> = (0..rows).flat_map(|_| scale.to_le_bytes()).collect();
        let mut payload = packed.clone();
        let scale_offset = payload.len() as u64;
        payload.extend_from_slice(&scales);
        let shape_offset = payload.len() as u64;
        payload.extend_from_slice(&(rows as i32).to_le_bytes());
        payload.extend_from_slice(&(cols as i32).to_le_bytes());
        fs::write(root.join("weights.bin"), payload).unwrap();
        let store = DenseTensorStore::new(
            &root,
            vec![],
            [
                TensorSegment {
                    tensor: "linear.weight_packed".into(),
                    dtype: Some("I32".into()),
                    shape: vec![rows as u64, (cols / 8) as u64],
                    shard: "weights.bin".into(),
                    offset: 0,
                    length: packed.len() as u64,
                },
                TensorSegment {
                    tensor: "linear.weight_scale".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![rows as u64, 1],
                    shard: "weights.bin".into(),
                    offset: scale_offset,
                    length: scales.len() as u64,
                },
                TensorSegment {
                    tensor: "linear.weight_shape".into(),
                    dtype: Some("I32".into()),
                    shape: vec![2],
                    shard: "weights.bin".into(),
                    offset: shape_offset,
                    length: 8,
                },
            ],
        );
        let positions = 2;
        let input: Vec<f32> = (0..positions * cols)
            .map(|index| (index as f32 - 11.0) / 13.0)
            .collect();
        let reference =
            crate::int4_group32_bf16_matmul_cpu(&packed, &scales, rows, cols, positions, &input)
                .unwrap();
        let actual = tiled_packed_int4_group32_bf16_matmul(
            &store,
            "linear.weight",
            36,
            positions,
            &input,
            |packed, scales, rows, cols, positions, input| {
                crate::int4_group32_bf16_matmul_cpu(packed, scales, rows, cols, positions, input)
                    .map_err(|error| error.to_string())
            },
        )
        .unwrap();
        assert_eq!(actual.values, reference);
        assert_eq!(actual.metrics.tiles, 2);
        assert_eq!(actual.metrics.storage_bytes, 62);
        assert_eq!(actual.metrics.peak_tile_bytes, 36);
        fs::remove_dir_all(root).unwrap();
    }
}
