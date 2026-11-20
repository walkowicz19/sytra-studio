"""Fused Triton kernels for Helix 4-bit LoRA execution and chunked cross-entropy."""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl


def triton_kernels_available(device: torch.device | str | None = None) -> bool:
    """Return True when CUDA and Triton are usable and not explicitly disabled."""
    if os.environ.get("SYTRA_HELIX_TRITON_NF4") == "0":
        return False
    if not torch.cuda.is_available():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return True


@triton.jit
def _nf4_decode_tile(
    packed_ptr,
    scales_ptr,
    codebook_ptr,
    linear_idx,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    LOW_FIRST: tl.constexpr,
):
    in_bounds = linear_idx < total_elements
    safe_idx = tl.where(in_bounds, linear_idx, 0)
    bytes_val = tl.load(packed_ptr + (safe_idx // 2), mask=in_bounds, other=0).to(tl.int32)
    if LOW_FIRST:
        shift = (safe_idx & 1) * 4
    else:
        shift = (1 - (safe_idx & 1)) * 4
    nibbles = (bytes_val >> shift) & 15
    q_vals = tl.load(codebook_ptr + nibbles, mask=in_bounds, other=0.0).to(tl.float32)
    block_scales = tl.load(scales_ptr + (safe_idx // BLOCK_SIZE), mask=in_bounds, other=0.0).to(tl.float32)
    return tl.where(in_bounds, q_vals * block_scales, 0.0)


@triton.jit
def _nf4_lora_fwd_kernel(
    x_ptr,
    packed_ptr,
    scales_ptr,
    codebook_ptr,
    lr_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    R,
    stride_xm,
    stride_xk,
    stride_lrm,
    stride_lrr,
    stride_bn,
    stride_br,
    stride_om,
    stride_on,
    lora_scale,
    BLOCK_SIZE: tl.constexpr,
    LOW_FIRST: tl.constexpr,
    HAS_LORA: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    total_elements = N * K

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k_start in range(0, K, BK):
        offs_k = k_start + tl.arange(0, BK)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # W has shape [N, K]; we form W^T tile [BK, BN] directly.
        w_lin = offs_n[None, :] * K + offs_k[:, None]
        w_valid = mask_k[:, None] & mask_n[None, :]
        wt_tile = _nf4_decode_tile(
            packed_ptr, scales_ptr, codebook_ptr, w_lin, total_elements, BLOCK_SIZE, LOW_FIRST
        )
        wt_tile = tl.where(w_valid, wt_tile, 0.0)
        acc = tl.dot(x_tile, wt_tile, acc, input_precision="ieee")

    if HAS_LORA:
        lora_acc = tl.zeros((BM, BN), dtype=tl.float32)
        for r_start in range(0, R, BR):
            offs_r = r_start + tl.arange(0, BR)
            mask_r = offs_r < R
            lr_ptrs = lr_ptr + offs_m[:, None] * stride_lrm + offs_r[None, :] * stride_lrr
            lr_tile = tl.load(lr_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
            # Load B^T tile [BR, BN] where B is [N, R]
            b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_br
            bt_tile = tl.load(b_ptrs, mask=mask_r[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            lora_acc = tl.dot(lr_tile, bt_tile, lora_acc, input_precision="ieee")
        acc = acc + lora_acc * lora_scale

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _nf4_lora_bwd_dx_kernel(
    g_ptr,
    packed_ptr,
    scales_ptr,
    codebook_ptr,
    gl_ptr,
    a_ptr,
    mask_ptr,
    dx_ptr,
    M,
    N,
    K,
    R,
    stride_gm,
    stride_gn,
    stride_glm,
    stride_glr,
    stride_ar,
    stride_ak,
    stride_mm,
    stride_mk,
    stride_dxm,
    stride_dxk,
    drop_scale,
    BLOCK_SIZE: tl.constexpr,
    LOW_FIRST: tl.constexpr,
    HAS_LORA: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    mask_m = offs_m < M
    mask_k = offs_k < K
    total_elements = N * K

    acc = tl.zeros((BM, BK), dtype=tl.float32)
    for n_start in range(0, N, BN):
        offs_n = n_start + tl.arange(0, BN)
        mask_n = offs_n < N
        g_ptrs = g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
        g_tile = tl.load(g_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # W tile [BN, BK]
        w_lin = offs_n[:, None] * K + offs_k[None, :]
        w_valid = mask_n[:, None] & mask_k[None, :]
        w_tile = _nf4_decode_tile(
            packed_ptr, scales_ptr, codebook_ptr, w_lin, total_elements, BLOCK_SIZE, LOW_FIRST
        )
        w_tile = tl.where(w_valid, w_tile, 0.0)
        acc = tl.dot(g_tile, w_tile, acc, input_precision="ieee")

    if HAS_LORA:
        lora_dx = tl.zeros((BM, BK), dtype=tl.float32)
        for r_start in range(0, R, BR):
            offs_r = r_start + tl.arange(0, BR)
            mask_r = offs_r < R
            gl_ptrs = gl_ptr + offs_m[:, None] * stride_glm + offs_r[None, :] * stride_glr
            gl_tile = tl.load(gl_ptrs, mask=mask_m[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
            a_ptrs = a_ptr + offs_r[:, None] * stride_ar + offs_k[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            lora_dx = tl.dot(gl_tile, a_tile, lora_dx, input_precision="ieee")
        if HAS_DROPOUT:
            m_ptrs = mask_ptr + offs_m[:, None] * stride_mm + offs_k[None, :] * stride_mk
            keep = tl.load(m_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0).to(tl.float32)
            lora_dx = lora_dx * keep * drop_scale
        acc = acc + lora_dx

    dx_ptrs = dx_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
    tl.store(dx_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


class _FusedNF4LoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        codebook: torch.Tensor,
        out_features: int,
        in_features: int,
        block_size: int,
        low_first: bool,
        lora_scale: float,
        dropout_p: float,
        training: bool,
    ) -> torch.Tensor:
        orig_shape = inputs.shape
        x_2d = inputs.reshape(-1, in_features).contiguous()
        M, K = x_2d.shape
        N = out_features
        R = lora_a.shape[0]
        if training and dropout_p > 0.0:
            drop_mask = (torch.rand(x_2d.shape, device=x_2d.device, dtype=torch.float32) >= dropout_p).to(torch.uint8)
            drop_scale = 1.0 / (1.0 - dropout_p)
            drop_x = torch.where(drop_mask.bool(), x_2d.to(torch.float32) * drop_scale, 0.0).to(x_2d.dtype)
        else:
            drop_mask = None
            drop_scale = 1.0
            drop_x = x_2d

        has_lora = R > 0 and lora_scale != 0.0
        if has_lora:
            low_rank = torch.matmul(drop_x.to(lora_a.dtype), lora_a.t()).contiguous()
            b_mat = lora_b.contiguous()
        else:
            low_rank = torch.empty((M, 0), device=x_2d.device, dtype=x_2d.dtype)
            b_mat = torch.empty((N, 0), device=x_2d.device, dtype=x_2d.dtype)

        output = torch.empty((M, N), device=x_2d.device, dtype=x_2d.dtype)
        if M > 0 and N > 0 and K > 0:
            BM, BN, BK, BR = 32, 32, 32, 16
            grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
            _nf4_lora_fwd_kernel[grid](
                x_2d,
                packed,
                scales,
                codebook,
                low_rank,
                b_mat,
                output,
                M,
                N,
                K,
                R,
                x_2d.stride(0),
                x_2d.stride(1),
                low_rank.stride(0),
                low_rank.stride(1),
                b_mat.stride(0),
                b_mat.stride(1),
                output.stride(0),
                output.stride(1),
                float(lora_scale),
                BLOCK_SIZE=block_size,
                LOW_FIRST=low_first,
                HAS_LORA=has_lora,
                BM=BM,
                BN=BN,
                BK=BK,
                BR=BR,
            )
        else:
            output.zero_()

        ctx.orig_shape = orig_shape
        ctx.input_dtype = inputs.dtype
        ctx.out_features = out_features
        ctx.in_features = in_features
        ctx.block_size = block_size
        ctx.low_first = low_first
        ctx.lora_scale = float(lora_scale)
        ctx.drop_scale = float(drop_scale)
        ctx.has_lora = has_lora
        ctx.has_dropout = drop_mask is not None
        to_save = [packed, scales, codebook, lora_a, lora_b, drop_x, low_rank]
        if drop_mask is not None:
            to_save.append(drop_mask)
        ctx.save_for_backward(*to_save)
        return output.view(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        packed, scales, codebook, lora_a, lora_b, drop_x, low_rank = saved[:7]
        drop_mask = saved[7] if ctx.has_dropout else None

        in_features = ctx.in_features
        out_features = ctx.out_features
        g_2d = grad_output.reshape(-1, out_features).contiguous()
        M, N = g_2d.shape
        K = in_features
        R = lora_a.shape[0]

        grad_a = grad_b = None
        if ctx.has_lora:
            g_b = g_2d.to(lora_b.dtype)
            if ctx.needs_input_grad[2]:
                grad_b = torch.matmul(g_b.t(), low_rank.to(lora_b.dtype)) * ctx.lora_scale
            grad_low = (torch.matmul(g_b, lora_b) * ctx.lora_scale).contiguous()
            if ctx.needs_input_grad[1]:
                grad_a = torch.matmul(grad_low.to(lora_a.dtype).t(), drop_x.to(lora_a.dtype))
            a_mat = lora_a.contiguous()
        else:
            grad_low = torch.empty((M, 0), device=g_2d.device, dtype=g_2d.dtype)
            a_mat = torch.empty((0, K), device=g_2d.device, dtype=g_2d.dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty((M, K), device=g_2d.device, dtype=ctx.input_dtype)
            if M > 0 and N > 0 and K > 0:
                BM, BK, BN, BR = 32, 32, 32, 16
                grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
                mask_arg = drop_mask if drop_mask is not None else grad_x
                _nf4_lora_bwd_dx_kernel[grid](
                    g_2d,
                    packed,
                    scales,
                    codebook,
                    grad_low,
                    a_mat,
                    mask_arg,
                    grad_x,
                    M,
                    N,
                    K,
                    R,
                    g_2d.stride(0),
                    g_2d.stride(1),
                    grad_low.stride(0),
                    grad_low.stride(1),
                    a_mat.stride(0),
                    a_mat.stride(1),
                    mask_arg.stride(0),
                    mask_arg.stride(1),
                    grad_x.stride(0),
                    grad_x.stride(1),
                    float(ctx.drop_scale),
                    BLOCK_SIZE=ctx.block_size,
                    LOW_FIRST=ctx.low_first,
                    HAS_LORA=ctx.has_lora,
                    HAS_DROPOUT=ctx.has_dropout,
                    BM=BM,
                    BK=BK,
                    BN=BN,
                    BR=BR,
                )
            else:
                grad_x.zero_()
            grad_x = grad_x.view(ctx.orig_shape)

        return grad_x, grad_a, grad_b, None, None, None, None, None, None, None, None, None, None


def fused_nf4_lora_linear(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    codebook: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int,
    *,
    low_first: bool = True,
    lora_scale: float = 1.0,
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Compute F.linear(inputs, W_nf4) + scaling * F.linear(dropout(inputs) @ A^T, B) fused."""
    return _FusedNF4LoRAFunction.apply(
        inputs,
        lora_a,
        lora_b,
        packed.contiguous(),
        scales.contiguous(),
        codebook.to(device=inputs.device, dtype=torch.float32).contiguous(),
        out_features,
        in_features,
        block_size,
        bool(low_first),
        float(lora_scale),
        float(dropout_p),
        bool(training),
    )


@triton.jit
def _chunked_ce_fwd_kernel(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    lse_ptr,
    stride_lm,
    V,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(targets_ptr + row)
    if target == ignore_index:
        tl.store(losses_ptr + row, 0.0)
        tl.store(lse_ptr + row, 0.0)
        return

    row_ptr = logits_ptr + row * stride_lm
    m_i = -float("inf")
    d_i = 0.0
    for v_start in range(0, V, BLOCK_V):
        offs_v = v_start + tl.arange(0, BLOCK_V)
        mask_v = offs_v < V
        vals = tl.load(row_ptr + offs_v, mask=mask_v, other=-float("inf")).to(tl.float32)
        m_tile = tl.max(vals, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        d_i = d_i * tl.exp(m_i - m_new) + tl.sum(tl.where(mask_v, tl.exp(vals - m_new), 0.0), axis=0)
        m_i = m_new

    lse = m_i + tl.log(d_i)
    target_val = tl.load(row_ptr + target).to(tl.float32)
    tl.store(losses_ptr + row, lse - target_val)
    tl.store(lse_ptr + row, lse)


@triton.jit
def _chunked_ce_bwd_kernel(
    logits_ptr,
    targets_ptr,
    lse_ptr,
    grad_losses_ptr,
    grad_logits_ptr,
    stride_lm,
    stride_gm,
    V,
    ignore_index,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    v_blk = tl.program_id(1)
    offs_v = v_blk * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V

    target = tl.load(targets_ptr + row)
    grad_scale = tl.load(grad_losses_ptr + row).to(tl.float32)
    g_ptr = grad_logits_ptr + row * stride_gm + offs_v

    if target == ignore_index:
        tl.store(g_ptr, tl.zeros((BLOCK_V,), dtype=tl.float32), mask=mask_v)
        return

    lse = tl.load(lse_ptr + row).to(tl.float32)
    row_ptr = logits_ptr + row * stride_lm + offs_v
    vals = tl.load(row_ptr, mask=mask_v, other=0.0).to(tl.float32)
    probs = tl.exp(vals - lse)
    is_target = tl.where(offs_v == target, 1.0, 0.0)
    grad = grad_scale * (probs - is_target)
    tl.store(g_ptr, grad, mask=mask_v)


class _TritonChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int) -> torch.Tensor:
        if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
            raise ValueError("Chunked cross-entropy expects [N, V] logits and [N] targets")
        logits = logits.contiguous()
        targets = targets.to(device=logits.device, dtype=torch.int64).contiguous()
        N, V = logits.shape
        losses = torch.empty((N,), device=logits.device, dtype=torch.float32)
        lse = torch.empty((N,), device=logits.device, dtype=torch.float32)
        if N > 0 and V > 0:
            BLOCK_V = 4096
            _chunked_ce_fwd_kernel[(N,)](
                logits,
                targets,
                losses,
                lse,
                logits.stride(0),
                V,
                int(ignore_index),
                BLOCK_V=BLOCK_V,
            )
        ctx.save_for_backward(logits, targets, lse)
        ctx.ignore_index = int(ignore_index)
        return losses.sum()

    @staticmethod
    def backward(ctx, grad_total: torch.Tensor):
        logits, targets, lse = ctx.saved_tensors
        N, V = logits.shape
        grad_logits = torch.empty_like(logits)
        if N > 0 and V > 0:
            grad_per_row = grad_total.to(device=logits.device, dtype=torch.float32).reshape(())
            grad_losses = grad_per_row.expand((N,)).contiguous()
            BLOCK_V = 4096
            grid = (N, triton.cdiv(V, BLOCK_V))
            _chunked_ce_bwd_kernel[grid](
                logits,
                targets,
                lse,
                grad_losses,
                grad_logits,
                logits.stride(0),
                grad_logits.stride(0),
                V,
                ctx.ignore_index,
                BLOCK_V=BLOCK_V,
            )
        return grad_logits, None, None


def chunked_cross_entropy_sum(
    logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int = -100
) -> torch.Tensor:
    """Compute exact summed cross-entropy loss using tiled Triton online log-sum-exp."""
    if os.environ.get("SYTRA_HELIX_TRITON_CE") == "0" or logits.device.type != "cuda":
        return torch.nn.functional.cross_entropy(
            logits.float(), targets, ignore_index=ignore_index, reduction="sum"
        )
    return _TritonChunkedCrossEntropy.apply(logits, targets, int(ignore_index))
