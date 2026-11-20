"""Layer-wise recomputation primitives for low-memory adapter training.

Architecture-specific loaders and model adapters must supply the ordered
layers and their forward callback. This module does not infer a model layout
or claim that a given checkpoint fits a particular device.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .helix_triton_ops import chunked_cross_entropy_sum


@dataclass(frozen=True)
class WindowedStepResult:
    loss: float
    supervised_tokens: int
    input_gradient: torch.Tensor
    activation_peak_resident_bytes: int = 0
    activation_spill_count: int = 0
    activation_spill_bytes: int = 0
    prefetch_peak_pinned_bytes: int = 0
    prefetch_wait_ms: float = 0.0


@dataclass(frozen=True)
class WindowedEvaluationResult:
    loss: float
    supervised_tokens: int


def windowed_lora_evaluate(layers, lm_head, inputs, labels, *, device='cpu', loss_chunk_tokens=256,
                           forward_layer=None, weight_pager=None):
    """Evaluate a frozen-base model one layer at a time without saving activations."""
    device = torch.device(device)
    if device.type not in {'cpu', 'cuda'} or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Evaluation requires an available CPU or CUDA device')
    if type(loss_chunk_tokens) is not int or loss_chunk_tokens <= 0 or not layers:
        raise ValueError('Evaluation requires layers and a positive loss chunk size')
    if inputs.ndim != 3 or not inputs.is_floating_point() or labels.shape != inputs.shape[:2] or labels.dtype != torch.long or inputs.shape[1] < 2:
        raise ValueError('Expected floating [batch, sequence, hidden] inputs and int64 labels')
    _validate_adapter_only([*layers, lm_head])
    targets = labels[:, 1:].contiguous().view(-1).cpu()
    supervised = int((targets != -100).sum().item())
    if not supervised:
        raise ValueError('No supervised tokens after causal shifting')
    hidden = inputs.detach().to(device)
    with torch.no_grad():
        for index, layer in enumerate(layers):
            try:
                if weight_pager is not None:
                    weight_pager.prepare(layer, device)
                _move_module(layer, device)
                hidden = _forward(layer, hidden, index, forward_layer)
            finally:
                try:
                    if weight_pager is not None:
                        weight_pager.restore(layer)
                finally:
                    _move_module(layer, 'cpu')
        if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
            raise ValueError('Final hidden state must retain batch and sequence dimensions')
        hidden = hidden[:, :-1, :].contiguous().view(-1, hidden.shape[-1])
        total_loss = 0.0
        try:
            if weight_pager is not None:
                weight_pager.prepare(lm_head, device)
            _move_module(lm_head, device)
            for start in range(0, targets.numel(), loss_chunk_tokens):
                end = min(start + loss_chunk_tokens, targets.numel())
                chunk_labels = targets[start:end].to(device)
                if not bool((chunk_labels != -100).any()):
                    continue
                logits = lm_head(hidden[start:end])
                total_loss += float(chunked_cross_entropy_sum(logits, chunk_labels, ignore_index=-100).cpu())
        finally:
            try:
                if weight_pager is not None:
                    weight_pager.restore(lm_head)
            finally:
                _move_module(lm_head, 'cpu')
    return WindowedEvaluationResult(total_loss / supervised, supervised)


def _validate_adapter_only(modules: Sequence[torch.nn.Module]) -> None:
    trainable = 0
    seen: set[int] = set()
    for module in modules:
        for name, parameter in module.named_parameters():
            if parameter.device.type != "cpu":
                raise ValueError("All windowed modules must start on CPU")
            if parameter.requires_grad:
                if "lora_" not in name.lower():
                    raise ValueError(f"Trainable parameter is not a LoRA adapter: {name}")
                if id(parameter) in seen:
                    raise ValueError("Shared trainable parameters across windows are not supported")
                seen.add(id(parameter))
                trainable += 1
    if not trainable:
        raise ValueError("No trainable LoRA parameters were found")


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, cuda


def _move_module(module: torch.nn.Module, device: str | torch.device, *, non_blocking: bool = False) -> None:
    identities = {name: id(parameter) for name, parameter in module.named_parameters() if parameter.requires_grad}
    module.to(device, non_blocking=non_blocking)
    if identities != {name: id(parameter) for name, parameter in module.named_parameters() if parameter.requires_grad}:
        module.to("cpu")
        raise RuntimeError("Device transfer replaced adapter parameters; optimizer references would be invalid")


def _forward(layer: torch.nn.Module, hidden: torch.Tensor, index: int,
             callback: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor] | None) -> torch.Tensor:
    result = callback(layer, hidden, index) if callback else layer(hidden)
    if not isinstance(result, torch.Tensor) or not result.is_floating_point():
        raise ValueError("A window must return one floating-point tensor")
    return result


def windowed_lora_backward(
    layers: Sequence[torch.nn.Module], lm_head: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor, *,
    device: str | torch.device = 'cpu', loss_chunk_tokens: int = 256,
    forward_layer: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor] | None = None,
    weight_pager: object | None = None, gradient_scale: float = 1.0,
    auxiliary_loss: Callable[[int], torch.Tensor | None] | None = None,
    activation_memory_bytes: int = 512 * 1024 * 1024, activation_spill_dir: str | None = None,
    prefetch_layers: bool = False, prefetch_pinned_memory_bytes: int = 256 * 1024 * 1024,
) -> WindowedStepResult:
    """Replay layer-wise LoRA with bounded CPU activation storage."""
    from .helix_activations import ActivationStash
    from .helix_prefetch import CudaLayerPrefetcher
    if type(prefetch_layers) is not bool:
        raise ValueError('prefetch_layers must be a boolean')
    if type(prefetch_pinned_memory_bytes) is not int or prefetch_pinned_memory_bytes < 0:
        raise ValueError('prefetch_pinned_memory_bytes must be a non-negative integer')
    prefetch_context = (CudaLayerPrefetcher(device, weight_pager, _move_module, pinned_bytes=prefetch_pinned_memory_bytes)
                        if prefetch_layers else nullcontext(None))
    with ActivationStash(activation_memory_bytes, device=device, directory=activation_spill_dir) as stash, prefetch_context as prefetcher:
        return _windowed_lora_backward(layers, lm_head, inputs, labels, device=device,
            loss_chunk_tokens=loss_chunk_tokens, forward_layer=forward_layer, weight_pager=weight_pager,
            gradient_scale=gradient_scale, auxiliary_loss=auxiliary_loss, activation_stash=stash, prefetcher=prefetcher)


def _windowed_lora_backward(
    layers: Sequence[torch.nn.Module],
    lm_head: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    loss_chunk_tokens: int = 256,
    forward_layer: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor] | None = None,
    weight_pager: object | None = None,
    gradient_scale: float = 1.0,
    auxiliary_loss: Callable[[int], torch.Tensor | None] | None = None,
    activation_stash: object,
    prefetcher: object | None,
) -> WindowedStepResult:
    """Accumulate causal cross-entropy gradients by replaying one layer at a time.

    Inputs are floating activations of shape [batch, sequence, hidden]. Labels
    have shape [batch, sequence] and use -100 for ignored positions. Optimizer
    stepping and gradient clearing remain the caller's responsibility; all
    modules and their accumulated gradients are returned to CPU. Discard
    accumulated gradients if this function raises. Layer
    forwards must not mutate buffers or external state; stochastic operations
    using PyTorch's CPU/CUDA generators are replayed with their saved RNG state.
    When supplied, weight_pager prepares each window and restores frozen CPU
    mappings when it is unloaded; it must provide prepare(module, device)
    and restore(module).
    gradient_scale scales accumulated gradients without changing the reported
    mean loss, allowing token-weighted gradient accumulation across batches.
    auxiliary_loss returns an already-weighted scalar from the latest layer
    forward, or None. It is recomputed with the layer during backward.
    """
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if not layers or loss_chunk_tokens <= 0:
        raise ValueError("At least one layer and a positive loss chunk size are required")
    if isinstance(gradient_scale, bool) or not isinstance(gradient_scale, (int, float)) or not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be a finite positive number")
    if inputs.ndim != 3 or not inputs.is_floating_point() or labels.shape != inputs.shape[:2] or labels.dtype != torch.long:
        raise ValueError("Expected floating [batch, sequence, hidden] inputs and int64 [batch, sequence] labels")
    if inputs.shape[1] < 2:
        raise ValueError("Causal training requires at least two positions")
    _validate_adapter_only([*layers, lm_head])
    targets = labels[:, 1:].contiguous().view(-1).cpu()
    supervised = int((targets != -100).sum().item())
    if supervised == 0:
        raise ValueError("No supervised tokens after causal shifting")

    activations = activation_stash
    activations.push(inputs.detach())
    states: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    auxiliary_present: list[bool] = []
    total_auxiliary = 0.0
    for index, layer in enumerate(layers):
        try:
            if prefetcher is not None:
                prefetcher.take(layer)
            else:
                if weight_pager is not None:
                    weight_pager.prepare(layer, device)
                _move_module(layer, device)
            states.append(_rng_state(device))
            with torch.no_grad():
                output = _forward(layer, activations.last().to(device, non_blocking=device.type == 'cuda'), index, forward_layer)
                auxiliary = auxiliary_loss(index) if auxiliary_loss is not None else None
                auxiliary_present.append(auxiliary is not None)
                if auxiliary is not None:
                    if not isinstance(auxiliary, torch.Tensor) or auxiliary.ndim != 0 or not bool(torch.isfinite(auxiliary)):
                        raise ValueError("Auxiliary loss must be a finite scalar tensor")
                    total_auxiliary += float(auxiliary.cpu())
            activations.push(output.detach())
            if prefetcher is not None and index + 1 < len(layers):
                prefetcher.schedule(layers[index + 1])
            del output
        finally:
            try:
                if weight_pager is not None:
                    weight_pager.restore(layer)
            finally:
                _move_module(layer, "cpu")

    final = activations.pop()
    if final.ndim != 3 or final.shape[:2] != labels.shape:
        raise ValueError("Final hidden state must retain batch and sequence dimensions")
    batch, sequence, width = final.shape
    # The head is evaluated per token block. Each graph is immediately freed
    # after backward, so the full [batch * sequence, vocabulary] tensor is not kept.
    hidden = final[:, :-1, :].contiguous().view(-1, width)
    hidden_grad = torch.zeros_like(hidden)
    total_loss = 0.0
    try:
        if weight_pager is not None:
            weight_pager.prepare(lm_head, device)
        _move_module(lm_head, device)
        for start in range(0, targets.numel(), loss_chunk_tokens):
            end = min(start + loss_chunk_tokens, targets.numel())
            chunk_labels = targets[start:end].to(device)
            if not bool((chunk_labels != -100).any()):
                continue
            chunk = hidden[start:end].to(device).detach().requires_grad_(True)
            logits = lm_head(chunk)
            if logits.ndim != 2 or logits.shape[0] != end - start:
                raise ValueError("The language-model head must produce [tokens, vocabulary] logits")
            loss_sum = chunked_cross_entropy_sum(logits, chunk_labels, ignore_index=-100)
            total_loss += float(loss_sum.detach().cpu())
            (loss_sum * (gradient_scale / supervised)).backward()
            hidden_grad[start:end].copy_(chunk.grad.detach().to("cpu"))
            del logits, loss_sum, chunk, chunk_labels
    finally:
        try:
            if weight_pager is not None:
                weight_pager.restore(lm_head)
        finally:
            _move_module(lm_head, "cpu")

    gradient = torch.zeros_like(final)
    gradient[:, :-1, :] = hidden_grad.view(batch, sequence - 1, width)
    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        try:
            if prefetcher is not None:
                prefetcher.take(layer)
            else:
                if weight_pager is not None:
                    weight_pager.prepare(layer, device)
                _move_module(layer, device)
            activation = activations.pop().to(device, non_blocking=device.type == 'cuda').detach().requires_grad_(True)
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                cpu_state, cuda_state = states[index]
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state, device)
                output = _forward(layer, activation, index, forward_layer)
                if output.shape != gradient.shape:
                    raise ValueError(f"Layer {index} replay shape differs from the stored gradient")
                auxiliary = auxiliary_loss(index) if auxiliary_loss is not None else None
                if (auxiliary is not None) != auxiliary_present[index]:
                    raise RuntimeError('Auxiliary loss changed between forward and replay')
                if auxiliary is None:
                    torch.autograd.backward(output, gradient.to(device))
                else:
                    if not isinstance(auxiliary, torch.Tensor) or auxiliary.ndim != 0 or not bool(torch.isfinite(auxiliary)) or not auxiliary.requires_grad:
                        raise ValueError("Replayed auxiliary loss must be a finite scalar with gradients")
                    torch.autograd.backward((output, auxiliary), (gradient.to(device), torch.ones_like(auxiliary) * gradient_scale))
            if activation.grad is None:
                raise RuntimeError(f"Layer {index} did not propagate input gradients")
            if prefetcher is not None and index > 0:
                prefetcher.schedule(layers[index - 1])
            gradient = activation.grad.detach().to("cpu").clone()
            del output, activation
        finally:
            try:
                if weight_pager is not None:
                    weight_pager.restore(layer)
            finally:
                _move_module(layer, "cpu")
    return WindowedStepResult(
        total_loss / supervised + total_auxiliary,
        supervised,
        gradient,
        activation_peak_resident_bytes=getattr(activation_stash, 'peak_resident_bytes', 0),
        activation_spill_count=getattr(activation_stash, 'spill_count', 0),
        activation_spill_bytes=getattr(activation_stash, 'spill_bytes', 0),
        prefetch_peak_pinned_bytes=getattr(prefetcher, 'peak_pinned_bytes', 0) if prefetcher is not None else 0,
        prefetch_wait_ms=getattr(prefetcher, 'wait_ms', 0.0) if prefetcher is not None else 0.0,
    )
