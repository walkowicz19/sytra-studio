"""Boundary-aware rotary positions and sparse attention masks for Helix."""
from __future__ import annotations

import bisect

import torch


def packed_attention_layout(cu_seqlens: torch.Tensor, device: str | torch.device, *, sliding_window: int | None = None):
    """Build reset positions and a causal FlexAttention mask for one packed row.

    The mask isolates documents and optionally applies Mixtral's sliding window.
    Block metadata is built at 128-token granularity without allocating an
    attention-sized [sequence, sequence] tensor.
    """
    if (not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dtype != torch.long or
            cu_seqlens.device.type != 'cpu' or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or
            cu_seqlens[0].item() != 0 or torch.any(cu_seqlens[1:] - cu_seqlens[:-1] < 2)):
        raise ValueError('Invalid packed conversation boundaries')
    if sliding_window is not None and (type(sliding_window) is not int or sliding_window <= 0):
        raise ValueError('Sliding window must be a positive integer')
    from torch.nn.attention.flex_attention import BlockMask
    boundaries = cu_seqlens.tolist()
    length = boundaries[-1]
    # FlexAttention may query indices rounded up to the sparse block size.
    padded = ((length + 127) // 128) * 128
    sizes = cu_seqlens[1:] - cu_seqlens[:-1]
    documents = torch.full((padded,), -1, dtype=torch.long)
    documents[:length] = torch.repeat_interleave(torch.arange(len(sizes)), sizes)
    positions = torch.arange(length) - torch.repeat_interleave(cu_seqlens[:-1], sizes)
    documents = documents.to(device)
    positions = positions.to(device)

    def allowed(_batch, _head, query, key):
        result = ((query < length) & (key < length) & (query >= key) &
                  (documents[query] == documents[key]))
        if sliding_window is not None:
            result = result & (query - key < sliding_window)
        return result

    blocks = padded // 128
    # Build rectangular BlockMask metadata sized to the maximum active spans
    # rather than a dense [blocks, blocks] matrix, avoiding O(blocks^2) memory
    # when documents or sliding windows are bounded.
    kv_spans: list[tuple[int, int]] = []
    q_spans: list[tuple[int, int]] = []
    max_kv = 1
    max_q = 1
    win_blocks = (sliding_window + 126) // 128 if sliding_window is not None else blocks
    for q_block in range(blocks):
        q_start = q_block * 128
        if q_start >= length:
            kv_spans.append((0, 0))
            continue
        doc_start = boundaries[bisect.bisect_right(boundaries, q_start) - 1]
        earliest = max(doc_start, q_start - sliding_window + 1) if sliding_window is not None else doc_start
        first_block = earliest // 128
        count = q_block - first_block + 1
        kv_spans.append((first_block, count))
        if count > max_kv:
            max_kv = count

    for k_block in range(blocks):
        k_start = k_block * 128
        if k_start >= length:
            q_spans.append((0, 0))
            continue
        k_end = min(length - 1, k_start + 127)
        last_doc_end = boundaries[bisect.bisect_right(boundaries, k_end)] - 1
        last_block = min(blocks - 1, last_doc_end // 128)
        if sliding_window is not None:
            last_block = min(last_block, k_block + win_blocks)
        q_count = max(0, last_block - k_block + 1)
        q_spans.append((k_block, q_count))
        if q_count > max_q:
            max_q = q_count

    kv_counts = torch.zeros((1, 1, blocks), dtype=torch.int32)
    kv_indices = torch.zeros((1, 1, blocks, max_kv), dtype=torch.int32)
    for q_block, (first_block, count) in enumerate(kv_spans):
        kv_counts[0, 0, q_block] = count
        if count:
            kv_indices[0, 0, q_block, :count] = torch.arange(first_block, first_block + count, dtype=torch.int32)

    q_counts = torch.zeros((1, 1, blocks), dtype=torch.int32)
    q_indices = torch.zeros((1, 1, blocks, max_q), dtype=torch.int32)
    for k_block, (first_q, q_count) in enumerate(q_spans):
        q_counts[0, 0, k_block] = q_count
        if q_count:
            q_indices[0, 0, k_block, :q_count] = torch.arange(first_q, first_q + q_count, dtype=torch.int32)

    block_mask = BlockMask(
        seq_lengths=(length, length),
        kv_num_blocks=kv_counts.to(device),
        kv_indices=kv_indices.to(device),
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_counts.to(device),
        q_indices=q_indices.to(device),
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(128, 128),
        mask_mod=allowed,
    )
    return positions.unsqueeze(0), block_mask


def packed_causal_4d_mask(
    cu_seqlens: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str | torch.device = 'cpu',
    sliding_window: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build packed position IDs and an explicit 4D causal block-diagonal attention mask."""
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.long:
        raise ValueError('Packed sequences require a 1D int64 cu_seqlens tensor')
    if sliding_window is not None and (type(sliding_window) is not int or sliding_window <= 0):
        raise ValueError('sliding_window must be a positive integer')
    boundaries = cu_seqlens.tolist()
    if boundaries[0] != 0 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('cu_seqlens must be strictly increasing from zero')
    length = boundaries[-1]
    target_device = torch.device(device)
    positions = torch.empty(length, dtype=torch.long)
    documents = torch.empty(length, dtype=torch.long)
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        positions[start:end] = torch.arange(end - start, dtype=torch.long)
        documents[start:end] = index
    idx = torch.arange(length)
    allowed = (documents[:, None] == documents[None, :]) & (idx[:, None] >= idx[None, :])
    if sliding_window is not None:
        allowed = allowed & ((idx[:, None] - idx[None, :]) < sliding_window)
    mask = torch.full((1, 1, length, length), torch.finfo(dtype).min, dtype=dtype, device=target_device)
    mask[0, 0].masked_fill_(allowed.to(target_device), 0.0)
    return positions.unsqueeze(0).to(target_device), mask
