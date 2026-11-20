"""Blockwise NF4 frozen linear primitives for the experimental Helix path.

The format here is Sytra's explicit packed-code + float32-scale format; it is
not a parser for existing bitsandbytes, GPTQ or AWQ checkpoint metadata.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F


NF4_VALUES = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
              0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
              0.7229568362236023, 1.0)


def quantize_nf4(weight: torch.Tensor, *, block_size: int = 64, blocks_per_chunk: int = 4096):
    """Quantize one contiguous CPU matrix without a full-size float32 copy."""
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or
            weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not weight.is_contiguous() or weight.numel() == 0):
        raise ValueError('NF4 requires a non-empty contiguous CPU floating-point matrix')
    if type(block_size) is not int or block_size < 2 or block_size % 2 or type(blocks_per_chunk) is not int or blocks_per_chunk <= 0:
        raise ValueError('NF4 block size must be positive and even')
    values = weight.detach().view(-1)
    total = values.numel()
    blocks = (total + block_size - 1) // block_size
    packed = torch.empty((total + 1) // 2, dtype=torch.uint8)
    scales = torch.empty(blocks, dtype=torch.float32)
    codebook = torch.tensor(NF4_VALUES, dtype=torch.float32)
    boundaries = (codebook[1:] + codebook[:-1]) * .5
    for block_start in range(0, blocks, blocks_per_chunk):
        block_end = min(block_start + blocks_per_chunk, blocks)
        start, end = block_start * block_size, min(block_end * block_size, total)
        data = values[start:end].to(torch.float32)
        if not torch.isfinite(data).all():
            raise ValueError('NF4 cannot quantize NaN or infinite weights')
        missing = (block_end - block_start) * block_size - data.numel()
        if missing:
            data = F.pad(data, (0, missing))
        data = data.view(-1, block_size)
        maxima = data.abs().amax(dim=1)
        scaled = data / torch.where(maxima > 0, maxima, torch.ones_like(maxima))[:, None]
        indices = torch.bucketize(scaled.contiguous(), boundaries).to(torch.uint8).view(-1)
        valid = end - start
        indices = indices[:valid]
        if valid % 2:
            indices = F.pad(indices, (0, 1))
        packed[start // 2:(end + 1) // 2] = indices[0::2] | (indices[1::2] << 4)
        scales[block_start:block_end] = maxima
    return packed, scales


def _decode_range(packed, scales, start, end, block_size, codebook, dtype, *, low_first: bool = True):
    # Opt in until this kernel has been validated on a wider set of CUDA and
    # Triton versions. The format and the CPU implementation stay unchanged.
    if low_first and packed.device.type == 'cuda' and os.environ.get('SYTRA_HELIX_TRITON_NF4') == '1':
        try:
            from .helix_nf4_triton import decode_nf4
        except ImportError as exc:
            raise RuntimeError('Triton NF4 decoding was enabled but Triton is unavailable') from exc
        return decode_nf4(packed, scales, start, end, block_size, codebook, dtype)
    offsets = torch.arange(start, end, dtype=torch.long, device=packed.device)
    encoded = packed[offsets // 2]
    is_even = offsets.remainder(2) == 0
    if low_first:
        indices = torch.where(is_even, encoded & 15, encoded >> 4).long()
    else:
        indices = torch.where(is_even, encoded >> 4, encoded & 15).long()
    return (codebook[indices] * scales[offsets // block_size]).to(dtype)


class _FrozenNF4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, packed, scales, codebook, out_features, in_features, block_size, chunk_rows, low_first):
        if inputs.shape[-1] != in_features or inputs.device != packed.device or packed.device != scales.device:
            raise ValueError('NF4 input shape and packed-weight device must match')
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('NF4 compute requires float16, bfloat16 or float32 inputs')
        ctx.save_for_backward(packed, scales, codebook)
        ctx.shape, ctx.dtype = inputs.shape, inputs.dtype
        ctx.out_features, ctx.in_features, ctx.block_size, ctx.chunk_rows = out_features, in_features, block_size, chunk_rows
        ctx.low_first = low_first
        ctx.fused_matmul = low_first and inputs.device.type == 'cuda' and os.environ.get('SYTRA_HELIX_TRITON_NF4_MATMUL') == '1'
        flat = inputs.reshape(-1, in_features)
        if ctx.fused_matmul:
            from .helix_nf4_triton import matmul_nf4
            output = matmul_nf4(flat, packed, scales, in_features, out_features, block_size, codebook)
            return output.view(*inputs.shape[:-1], out_features)
        output = inputs.new_empty((flat.shape[0], out_features))
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            weight = _decode_range(packed, scales, start * in_features, end * in_features,
                                   block_size, codebook, inputs.dtype, low_first=low_first).view(end - start, in_features)
            output[:, start:end] = F.linear(flat, weight)
        return output.view(*inputs.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 9
        packed, scales, codebook = ctx.saved_tensors
        grad = grad_output.reshape(-1, ctx.out_features)
        if ctx.fused_matmul:
            from .helix_nf4_triton import matmul_nf4
            grad_input = matmul_nf4(grad.to(ctx.dtype), packed, scales, ctx.in_features,
                                    ctx.out_features, ctx.block_size, codebook, backward=True)
            return grad_input.view(ctx.shape), None, None, None, None, None, None, None, None
        grad_input = grad_output.new_zeros((grad.shape[0], ctx.in_features), dtype=ctx.dtype)
        for start in range(0, ctx.out_features, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, ctx.out_features)
            weight = _decode_range(packed, scales, start * ctx.in_features, end * ctx.in_features,
                                   ctx.block_size, codebook, ctx.dtype, low_first=ctx.low_first).view(end - start, ctx.in_features)
            grad_input.add_(grad[:, start:end].to(ctx.dtype) @ weight)
        return grad_input.view(ctx.shape), None, None, None, None, None, None, None, None


class FrozenNF4Linear(torch.nn.Module):
    """Frozen 4-bit NF4/FP4 matrix supporting Sytra cache and bitsandbytes layouts."""

    def __init__(self, packed, scales, in_features: int, out_features: int, *, block_size: int = 64, chunk_rows: int = 32,
                 codebook=None, low_first: bool = True, nested_absmax=None, nested_quant_map=None,
                 nested_offset: float = 0.0, nested_blocksize: int = 256, bias: torch.Tensor | None = None):
        super().__init__()
        if (type(in_features) is not int or in_features <= 0 or type(out_features) is not int or out_features <= 0 or
                type(block_size) is not int or block_size < 2 or block_size % 2 or type(chunk_rows) is not int or chunk_rows <= 0 or
                type(low_first) is not bool):
            raise ValueError('Invalid NF4 matrix dimensions or block size')
        total = in_features * out_features
        expected_blocks = (total + block_size - 1) // block_size
        is_double = nested_absmax is not None or nested_quant_map is not None
        if is_double:
            if (nested_absmax is None or nested_quant_map is None or type(nested_blocksize) is not int or nested_blocksize <= 0 or
                    not isinstance(nested_absmax, torch.Tensor) or not isinstance(nested_quant_map, torch.Tensor) or
                    nested_absmax.dtype != torch.float32 or nested_quant_map.dtype != torch.float32 or
                    nested_absmax.ndim != 1 or nested_quant_map.shape != (256,) or
                    scales.dtype != torch.uint8 or scales.numel() != expected_blocks):
                raise ValueError('Invalid double-quant NF4 scale buffers')
            if nested_absmax.numel() != (expected_blocks + nested_blocksize - 1) // nested_blocksize:
                raise ValueError('Mismatched double-quant nested_absmax length')
        elif (not isinstance(scales, torch.Tensor) or scales.dtype != torch.float32 or scales.ndim != 1 or
              scales.numel() != expected_blocks or not scales.is_contiguous()):
            raise ValueError('Invalid NF4 packed weights or scales')
        if (not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8 or packed.ndim != 1 or
                packed.numel() != (total + 1) // 2 or packed.device != scales.device or not packed.is_contiguous()):
            raise ValueError('Invalid NF4 packed weights or scales')
        if not is_double and (not torch.isfinite(scales).all() or torch.any(scales < 0)):
            raise ValueError('NF4 scales must be finite and non-negative')
        if bias is not None:
            if (not isinstance(bias, torch.Tensor) or not bias.is_floating_point() or
                    bias.ndim != 1 or bias.shape[0] != out_features or bias.device != packed.device):
                raise ValueError('Invalid NF4 bias tensor')
            self.register_parameter('bias', torch.nn.Parameter(bias.detach().contiguous(), requires_grad=False))
        else:
            self.register_parameter('bias', None)
        cb_tensor = (torch.tensor(NF4_VALUES, dtype=torch.float32, device=packed.device) if codebook is None
                     else torch.as_tensor(codebook, dtype=torch.float32, device=packed.device).contiguous().view(-1))
        if cb_tensor.shape != (16,) or not torch.isfinite(cb_tensor).all():
            raise ValueError('4-bit codebook must contain 16 finite float32 entries')
        self.register_buffer('packed_weight', packed.detach())
        self.register_buffer('weight_scales', scales.detach())
        if codebook is not None:
            self.register_buffer('weight_codebook', cb_tensor.detach())
        else:
            self.register_buffer('weight_codebook', None)
        if is_double:
            self.register_buffer('nested_absmax', nested_absmax.detach().contiguous())
            self.register_buffer('nested_quant_map', nested_quant_map.detach().contiguous())
        else:
            self.register_buffer('nested_absmax', None)
            self.register_buffer('nested_quant_map', None)
        self.in_features, self.out_features = in_features, out_features
        self.block_size, self.chunk_rows = block_size, chunk_rows
        self.low_first = low_first
        self.nested_offset = float(nested_offset)
        self.nested_blocksize = int(nested_blocksize)

    @classmethod
    def from_float(cls, weight, *, block_size=64, chunk_rows=32, bias=None):
        packed, scales = quantize_nf4(weight, block_size=block_size)
        return cls(packed, scales, weight.shape[1], weight.shape[0], block_size=block_size, chunk_rows=chunk_rows, bias=bias)

    @property
    def qweight(self):
        """Expose the packed buffer for PEFT's device discovery."""
        return self.packed_weight

    def resolve_scales(self) -> torch.Tensor:
        """Return contiguous float32 block scales, expanding double-quant metadata when present."""
        if self.nested_absmax is None:
            if self.weight_scales.dtype != torch.float32:
                raise ValueError('NF4 scales must remain float32; move the layer by device without changing its dtype')
            return self.weight_scales
        if self.nested_absmax.dtype != torch.float32 or self.nested_quant_map.dtype != torch.float32:
            raise ValueError('NF4 nested scales must remain float32; move the layer by device without changing its dtype')
        idx = torch.arange(self.weight_scales.numel(), dtype=torch.long, device=self.weight_scales.device)
        return self.nested_quant_map[self.weight_scales.long()] * self.nested_absmax[idx // self.nested_blocksize] + self.nested_offset

    def resolve_codebook(self, device: torch.device) -> torch.Tensor:
        """Return the 16-entry float32 codebook on ``device``."""
        if self.weight_codebook is not None:
            return self.weight_codebook if self.weight_codebook.device == device else self.weight_codebook.to(device)
        return torch.tensor(NF4_VALUES, device=device, dtype=torch.float32)

    def forward(self, inputs):
        scales = self.resolve_scales()
        codebook = self.resolve_codebook(inputs.device)
        out = _FrozenNF4Function.apply(inputs, self.packed_weight, scales, codebook,
                                       self.out_features, self.in_features, self.block_size,
                                       self.chunk_rows, self.low_first)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


class NF4LoRALinear(torch.nn.Module):
    """LoRA residual around a frozen NF4 base, with optional rank stabilization."""

    def __init__(self, base: FrozenNF4Linear, rank: int, alpha: float, *, dropout: float = 0., use_rslora: bool = False):
        super().__init__()
        if (not isinstance(base, FrozenNF4Linear) or type(rank) is not int or rank <= 0 or
                isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0 or
                isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1 or
                type(use_rslora) is not bool):
            raise ValueError('Invalid NF4 LoRA configuration')
        self.base_layer = base
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False, device=base.packed_weight.device)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False, device=base.packed_weight.device)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = torch.nn.Dropout(dropout)
        self.scaling = alpha / (math.sqrt(rank) if use_rslora else rank)

    def forward(self, inputs):
        from .helix_triton_ops import fused_nf4_lora_linear, triton_kernels_available
        if inputs.device.type == 'cuda' and triton_kernels_available(inputs.device):
            out = fused_nf4_lora_linear(
                inputs,
                self.lora_A.weight,
                self.lora_B.weight,
                self.base_layer.packed_weight,
                self.base_layer.resolve_scales(),
                self.base_layer.resolve_codebook(inputs.device),
                self.base_layer.out_features,
                self.base_layer.in_features,
                self.base_layer.block_size,
                low_first=self.base_layer.low_first,
                lora_scale=self.scaling,
                dropout_p=float(self.lora_dropout.p),
                training=self.training,
            )
            if self.base_layer.bias is not None:
                out = out + self.base_layer.bias.to(dtype=out.dtype)
            return out
        result = self.base_layer(inputs)
        adapted = self.lora_B(self.lora_A(self.lora_dropout(inputs.to(self.lora_A.weight.dtype))))
        return result + adapted.to(result.dtype) * self.scaling
