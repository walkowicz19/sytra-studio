"""Quantized 4-bit and 8-bit frozen linear primitives and bitsandbytes checkpoint parsers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from peft.tuners.lora.layer import Linear as PeftLoraLinear
from safetensors import safe_open
import torch
import torch.nn.functional as F

from .helix_nf4 import FrozenNF4Linear, NF4_VALUES

FP4_VALUES = (
    0.0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0,
    -0.0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0,
)
_SUPPORTED_BNB_METHODS = {'bitsandbytes'}


def detect_checkpoint_quantization(config_data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return normalized quantization metadata from ``config.json``, rejecting unsupported schemes."""
    qcfg = config_data.get('quantization_config')
    if qcfg is None:
        return None
    if not isinstance(qcfg, Mapping):
        raise ValueError('Invalid quantization_config in checkpoint config.json')
    method = str(qcfg.get('quant_method', 'bitsandbytes')).lower()
    if method not in _SUPPORTED_BNB_METHODS:
        raise ValueError(f'Unsupported checkpoint quantization format: {method}')
    if bool(qcfg.get('load_in_4bit')):
        bits = 4
    elif bool(qcfg.get('load_in_8bit')):
        bits = 8
    else:
        raise ValueError('bitsandbytes quantization_config must enable load_in_4bit or load_in_8bit')
    return {
        'method': method,
        'bits': bits,
        'quant_type': str(qcfg.get('bnb_4bit_quant_type', 'nf4')).lower(),
        'double_quant': bool(qcfg.get('bnb_4bit_use_double_quant', False)),
        'skip_modules': set(qcfg.get('llm_int8_skip_modules') or ()),
    }


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize a 2D CPU weight matrix row-wise to int8 + float32 scales."""
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or
            weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            not weight.is_contiguous() or weight.numel() == 0):
        raise ValueError('Int8 requires a non-empty contiguous CPU floating-point 2D weight matrix')
    values = weight.detach().to(torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError('Int8 cannot quantize NaN or infinite weights')
    row_max = values.abs().amax(dim=1)
    scales = row_max / 127.0
    inv = torch.where(scales > 0, 1.0 / scales, torch.zeros_like(scales))[:, None]
    quant = torch.round(values * inv).clamp(-127.0, 127.0).to(torch.int8).contiguous()
    return quant, scales.contiguous()


class _FrozenInt8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight_i8, weight_scales, out_features, in_features, chunk_rows):
        if inputs.shape[-1] != in_features or inputs.device != weight_i8.device or weight_i8.device != weight_scales.device:
            raise ValueError('Int8 input shape and weight device must match')
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Int8 compute requires float16, bfloat16 or float32 inputs')
        ctx.save_for_backward(weight_i8, weight_scales)
        ctx.shape, ctx.dtype = inputs.shape, inputs.dtype
        ctx.out_features, ctx.in_features, ctx.chunk_rows = out_features, in_features, chunk_rows
        flat = inputs.reshape(-1, in_features)
        output = inputs.new_empty((flat.shape[0], out_features))
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            w = weight_i8[start:end].to(inputs.dtype) * weight_scales[start:end, None].to(inputs.dtype)
            output[:, start:end] = F.linear(flat, w)
        return output.view(*inputs.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return (None,) * 6
        weight_i8, weight_scales = ctx.saved_tensors
        grad = grad_output.reshape(-1, ctx.out_features)
        grad_input = grad_output.new_zeros((grad.shape[0], ctx.in_features), dtype=ctx.dtype)
        for start in range(0, ctx.out_features, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, ctx.out_features)
            w = weight_i8[start:end].to(ctx.dtype) * weight_scales[start:end, None].to(ctx.dtype)
            grad_input.add_(grad[:, start:end].to(ctx.dtype) @ w)
        return grad_input.view(ctx.shape), None, None, None, None, None


class FrozenInt8Linear(torch.nn.Module):
    """Frozen row-wise int8 matrix compatible with bitsandbytes ``SCB`` checkpoints."""

    def __init__(self, weight_i8: torch.Tensor, scales: torch.Tensor, *,
                 bias: torch.Tensor | None = None, chunk_rows: int = 64):
        super().__init__()
        if (not isinstance(weight_i8, torch.Tensor) or not isinstance(scales, torch.Tensor) or
                weight_i8.dtype != torch.int8 or scales.dtype != torch.float32 or
                weight_i8.ndim != 2 or scales.ndim != 1 or weight_i8.shape[0] != scales.shape[0] or
                weight_i8.shape[0] <= 0 or weight_i8.shape[1] <= 0 or
                weight_i8.device != scales.device or not weight_i8.is_contiguous() or not scales.is_contiguous() or
                type(chunk_rows) is not int or chunk_rows <= 0):
            raise ValueError('Invalid Int8 weight or scale buffers')
        if not torch.isfinite(scales).all() or torch.any(scales < 0):
            raise ValueError('Int8 scales must be finite and non-negative')
        if bias is not None:
            if (not isinstance(bias, torch.Tensor) or not bias.is_floating_point() or
                    bias.ndim != 1 or bias.shape[0] != weight_i8.shape[0] or bias.device != weight_i8.device):
                raise ValueError('Invalid Int8 bias tensor')
            self.register_parameter('bias', torch.nn.Parameter(bias.detach().contiguous(), requires_grad=False))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('weight_i8', weight_i8.detach())
        self.register_buffer('weight_scales', scales.detach())
        self.out_features, self.in_features = int(weight_i8.shape[0]), int(weight_i8.shape[1])
        self.chunk_rows = chunk_rows

    @classmethod
    def from_float(cls, weight: torch.Tensor, *, bias: torch.Tensor | None = None,
                   chunk_rows: int = 64) -> 'FrozenInt8Linear':
        qweight, scales = quantize_int8(weight)
        return cls(qweight, scales, bias=bias, chunk_rows=chunk_rows)

    @classmethod
    def from_bnb(cls, weight_i8: torch.Tensor, scb: torch.Tensor, *,
                 bias: torch.Tensor | None = None, chunk_rows: int = 64) -> 'FrozenInt8Linear':
        """Construct from a serialized bitsandbytes 8-bit ``.weight`` (int8) + ``.SCB`` pair."""
        if not isinstance(scb, torch.Tensor) or scb.ndim != 1 or scb.shape[0] != weight_i8.shape[0]:
            raise ValueError('Invalid bitsandbytes SCB scale vector')
        scales = (scb.detach().to(torch.float32) / 127.0).contiguous()
        return cls(weight_i8.contiguous(), scales, bias=bias, chunk_rows=chunk_rows)

    @property
    def qweight(self) -> torch.Tensor:
        """Expose the int8 weight buffer for PEFT device discovery."""
        return self.weight_i8

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.weight_scales.dtype != torch.float32:
            raise ValueError('Int8 scales must remain float32; move the layer by device without changing its dtype')
        out = _FrozenInt8Function.apply(inputs, self.weight_i8, self.weight_scales,
                                         self.out_features, self.in_features, self.chunk_rows)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


class Int8LoRALinear(torch.nn.Module):
    """Standalone LoRA wrapper around a frozen int8 base."""

    def __init__(self, base: FrozenInt8Linear, rank: int, alpha: float, *, dropout: float = 0., use_rslora: bool = False):
        super().__init__()
        if not isinstance(base, FrozenInt8Linear) or type(rank) is not int or rank <= 0 or alpha <= 0:
            raise ValueError('Invalid Int8 LoRA configuration')
        self.base_layer = base
        self.lora_A = torch.nn.Linear(base.in_features, rank, bias=False, device=base.weight_i8.device)
        self.lora_B = torch.nn.Linear(rank, base.out_features, bias=False, device=base.weight_i8.device)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = torch.nn.Dropout(dropout)
        self.scaling = float(alpha) / (math.sqrt(rank) if use_rslora else rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(inputs)
        adapted = self.lora_B(self.lora_A(self.lora_dropout(inputs.to(self.lora_A.weight.dtype))))
        return result + adapted.to(result.dtype) * self.scaling


class Int8PeftLinear(PeftLoraLinear):
    """PEFT custom LoRA module for frozen 8-bit int8 matrices."""

    def _get_in_out_features(self, module):
        if not isinstance(module, FrozenInt8Linear):
            raise TypeError('Int8 PEFT adapter requires a frozen Int8 linear base')
        return module.in_features, module.out_features

    def merge(self, *args, **kwargs):
        raise RuntimeError('Merging an adapter into Int8 weights is not supported')

    def unmerge(self):
        raise RuntimeError('Merging an adapter into Int8 weights is not supported')


def _read_tensor(sources: Mapping[str, tuple[Path, str]], key: str) -> torch.Tensor:
    path, tensor_key = sources[key]
    with safe_open(str(path), framework='pt', device='cpu') as file:
        return file.get_tensor(tensor_key)


def load_quantized_linear_from_sources(
    sources: Mapping[str, tuple[Path, str]],
    key_prefix: str,
    *,
    bias: torch.Tensor | None = None,
    chunk_rows: int = 32,
) -> torch.nn.Module | None:
    """Load a serialized bitsandbytes 4-bit or 8-bit linear weight as a frozen Helix layer."""
    weight_key = f'{key_prefix}.weight'
    if weight_key not in sources:
        return None
    def _resolve_bias() -> torch.Tensor | None:
        if bias is not None:
            return bias
        bias_key = f'{key_prefix}.bias'
        return _read_tensor(sources, bias_key) if bias_key in sources else None
    qs_prefix = f'{weight_key}.quant_state.bitsandbytes__'
    qs_keys = [k for k in sources if k.startswith(qs_prefix)]
    if qs_keys or f'{weight_key}.absmax' in sources:
        meta: dict[str, Any] = {}
        if qs_keys:
            raw_bytes = _read_tensor(sources, qs_keys[0]).reshape(-1).tolist()
            meta = json.loads(bytes(raw_bytes).decode('utf-8'))
        shape = meta.get('shape')
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError(f'Missing or invalid bitsandbytes 4-bit shape metadata for {weight_key}')
        out_features, in_features = int(shape[0]), int(shape[1])
        block_size = int(meta.get('blocksize', 64))
        quant_type = str(meta.get('quant_type', 'nf4')).lower()
        packed = _read_tensor(sources, weight_key).reshape(-1).contiguous()
        scales = _read_tensor(sources, f'{weight_key}.absmax').reshape(-1).contiguous()
        if f'{weight_key}.quant_map' in sources:
            codebook = _read_tensor(sources, f'{weight_key}.quant_map').reshape(-1).to(torch.float32).contiguous()
        elif quant_type == 'fp4':
            codebook = torch.tensor(FP4_VALUES, dtype=torch.float32)
        elif quant_type == 'nf4':
            codebook = torch.tensor(NF4_VALUES, dtype=torch.float32)
        else:
            raise ValueError(f'Unsupported bitsandbytes 4-bit quant_type {quant_type!r} for {weight_key}')
        nested_absmax = None
        nested_quant_map = None
        if f'{weight_key}.nested_absmax' in sources:
            nested_absmax = _read_tensor(sources, f'{weight_key}.nested_absmax').reshape(-1).to(torch.float32).contiguous()
            nested_quant_map = _read_tensor(sources, f'{weight_key}.nested_quant_map').reshape(-1).to(torch.float32).contiguous()
        return FrozenNF4Linear(
            packed,
            scales,
            in_features,
            out_features,
            block_size=block_size,
            chunk_rows=chunk_rows,
            codebook=codebook,
            low_first=False,
            nested_absmax=nested_absmax,
            nested_quant_map=nested_quant_map,
            nested_offset=float(meta.get('nested_offset', 0.0)),
            nested_blocksize=int(meta.get('nested_blocksize', 256)),
            bias=_resolve_bias(),
        )
    scb_key = f'{key_prefix}.SCB'
    if scb_key in sources:
        qweight = _read_tensor(sources, weight_key)
        scb = _read_tensor(sources, scb_key)
        return FrozenInt8Linear.from_bnb(qweight, scb, bias=_resolve_bias(), chunk_rows=max(chunk_rows, 64))
    return None
