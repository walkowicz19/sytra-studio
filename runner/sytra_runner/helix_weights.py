"""Map immutable, unquantized frozen parameters to safetensors storage.

This is an opt-in building block for windowed execution, not a checkpoint
loader or a quantization implementation. The caller must supply the exact
parameter-to-checkpoint mapping and keep the checkpoint files unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class TensorSlice:
    """Reference one contiguous slice inside a multi-expert checkpoint tensor."""

    path: str | Path
    key: str
    dim0_index: int
    row_start: int | None = None
    row_end: int | None = None


def _load_source_tensor(source: tuple[str | Path, str] | TensorSlice) -> torch.Tensor:
    if isinstance(source, TensorSlice):
        with safe_open(str(source.path), framework='pt', device='cpu') as file:
            tensor = file.get_slice(source.key)[source.dim0_index]
            if source.row_start is not None or source.row_end is not None:
                tensor = tensor[source.row_start:source.row_end]
            return tensor.contiguous()
    path, key = source
    with safe_open(str(path), framework='pt', device='cpu') as file:
        return file.get_tensor(key)


class MappedFrozenWeights:
    """Bind selected frozen CPU parameters to memory-mapped checkpoint tensors."""

    def __init__(self, model: torch.nn.Module, sources: Mapping[str, tuple[str | Path, str] | TensorSlice],
                 *, allow_empty: bool = False):
        if not sources and not allow_empty:
            raise ValueError("At least one frozen parameter mapping is required")
        named = list(model.named_parameters(remove_duplicate=False))
        parameters = dict(named)
        counts = Counter(id(parameter) for _, parameter in named)
        staged: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        for name, source in sources.items():
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"Unknown parameter: {name}")
            if parameter.requires_grad or parameter.device.type != 'cpu' or counts[id(parameter)] != 1:
                raise ValueError(f"Mapped parameter must be frozen, unshared and on CPU: {name}")
            if parameter.is_quantized or not parameter.is_contiguous():
                raise ValueError(f"Unsupported parameter storage: {name}")
            tensor = _load_source_tensor(source)
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"Checkpoint tensor shape or dtype mismatch: {name}")
            staged.append((parameter, tensor))
        # Nothing is rebound until all mappings have passed validation.
        self._mapped = {id(parameter): (parameter, tensor) for parameter, tensor in staged}
        self._mapped_buffers: dict[int, tuple[torch.nn.Module, dict[str, torch.Tensor]]] = {}
        for parameter, tensor in staged:
            parameter.data = tensor

    def register_quant_buffers(self, model: torch.nn.Module) -> None:
        """Keep references to mapped 4-bit and 8-bit buffers so device copies can be discarded."""
        from .helix_nf4 import FrozenNF4Linear
        from .helix_quant import FrozenInt8Linear
        for module in model.modules():
            if not isinstance(module, (FrozenNF4Linear, FrozenInt8Linear)):
                continue
            buffers = {name: buf for name, buf in module._buffers.items() if buf is not None}
            if not buffers or any(value.device.type != 'cpu' for value in buffers.values()):
                raise ValueError('Mapped quantized buffers must start on CPU')
            self._mapped_buffers[id(module)] = (module, buffers)

    def register_nf4_buffers(self, model: torch.nn.Module) -> None:
        """Compatibility alias for ``register_quant_buffers``."""
        self.register_quant_buffers(model)

    def restore(self, module: torch.nn.Module) -> None:
        """Drop device copies of mapped parameters before moving adapters to CPU."""
        for child in module.modules():
            release = getattr(child, '_helix_release_experts', None)
            if callable(release):
                release()
        for parameter in module.parameters():
            entry = self._mapped.get(id(parameter))
            if entry is not None:
                parameter.data = entry[1]
        for child in module.modules():
            entry = self._mapped_buffers.get(id(child))
            if entry is not None:
                for name, tensor in entry[1].items():
                    child._buffers[name] = tensor

    def prepare(self, module: torch.nn.Module, device: torch.device) -> None:
        """Align the active CPU window; CUDA transfers allocate aligned storage."""
        if device.type != 'cpu':
            return
        deferred = {id(parameter) for child in module.modules()
                    if getattr(child, '_helix_deferred_experts', False)
                    for parameter in child.parameters()}
        staged = [(parameter, entry[1].clone()) for parameter in module.parameters()
                  if (entry := self._mapped.get(id(parameter))) is not None
                  and id(parameter) not in deferred and entry[1].data_ptr() % 16]
        for parameter, tensor in staged:
            parameter.data = tensor

    @property
    def mapped_parameters(self) -> int:
        return len(self._mapped)
