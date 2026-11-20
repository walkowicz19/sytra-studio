"""Optional one-window-ahead CUDA weight transfers for Helix."""
from __future__ import annotations

import time
import torch


class CudaLayerPrefetcher:
    """Move one future layer on a transfer stream and synchronize before use.

    Deferred expert modules remain on CPU until routing selects them. Transfers
    outside the pinned staging budget can still block; actual overlap depends
    on the host and CUDA runtime. The caller must budget two layer windows.
    """

    def __init__(self, device, pager, move_module, *, pinned_bytes: int = 256 * 1024 * 1024):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('Layer prefetch requires an available CUDA device')
        if type(pinned_bytes) is not int or pinned_bytes < 0:
            raise ValueError('Pinned prefetch budget must be a non-negative integer')
        self.pager = pager
        self.move_module = move_module
        self.pinned_bytes = pinned_bytes
        self.peak_pinned_bytes = 0
        self.total_staged_bytes = 0
        self.fallback_sync_count = 0
        self.wait_ms = 0.0
        self.stream = torch.cuda.Stream(device=self.device)
        self.pending = None
        self._inflight = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def take(self, layer):
        started = time.perf_counter()
        if self.pending is None:
            self.fallback_sync_count += 1
            if self.pager is not None:
                self.pager.prepare(layer, self.device)
            self.move_module(layer, self.device)
            self.wait_ms += (time.perf_counter() - started) * 1000.0
            return
        expected, event, retained, staged_bytes = self.pending
        if expected is not layer:
            raise RuntimeError('Prefetched layer does not match execution order')
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(event)
        # Allocations created on the transfer stream remain in use by compute
        # after the host has rebound the corresponding mapped CPU weights.
        for tensor in list(layer.parameters()) + list(layer.buffers()):
            if tensor.device.type == 'cuda':
                tensor.record_stream(compute)
            if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor.grad.device.type == 'cuda':
                tensor.grad.record_stream(compute)
        self.pending = None
        self.total_staged_bytes += staged_bytes
        if retained:
            self._inflight.append((event, retained, staged_bytes))
        self.wait_ms += (time.perf_counter() - started) * 1000.0

    def _collect(self):
        self._inflight = [(event, tensors, size) for event, tensors, size in self._inflight if not event.query()]

    def _stage(self, layer):
        """Pin eligible CPU tensors without visiting deferred expert modules."""
        self._collect()
        used = sum(size for _, _, size in self._inflight)
        remaining = max(0, self.pinned_bytes - used)
        retained = []
        staged_bytes = 0
        visited = set()

        def visit(module):
            nonlocal remaining, staged_bytes
            if id(module) in visited or getattr(module, '_helix_deferred_experts', False):
                return
            visited.add(id(module))
            for name, tensor in list(module._parameters.items()) + list(module._buffers.items()):
                if tensor is None or tensor.device.type != 'cpu' or tensor.layout != torch.strided or not tensor.is_contiguous():
                    continue
                if tensor.is_pinned():
                    retained.append(tensor)
                    continue
                size = tensor.numel() * tensor.element_size()
                if not size or size > remaining:
                    continue
                try:
                    staged = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu', pin_memory=True)
                except RuntimeError:
                    continue
                staged.copy_(tensor.detach())
                if name in module._parameters:
                    tensor.data = staged
                else:
                    module._buffers[name] = staged
                retained.append(staged)
                remaining -= size
                staged_bytes += size
            for child in module.children():
                visit(child)

        visit(layer)
        self.peak_pinned_bytes = max(self.peak_pinned_bytes, used + staged_bytes)
        return retained, staged_bytes

    def schedule(self, layer):
        if self.pending is not None:
            raise RuntimeError('Only one layer may be prefetched at a time')
        try:
            if self.pager is not None:
                self.pager.prepare(layer, self.device)
            retained, staged_bytes = self._stage(layer)
            with torch.cuda.stream(self.stream):
                self.move_module(layer, self.device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.stream)
            self.pending = (layer, event, retained, staged_bytes)
        except BaseException:
            try:
                self.stream.synchronize()
            finally:
                try:
                    if self.pager is not None:
                        self.pager.restore(layer)
                finally:
                    self.move_module(layer, 'cpu')
            raise

    def close(self):
        pending = self.pending
        try:
            self.stream.synchronize()
            if pending is not None and self.pager is not None:
                self.pager.restore(pending[0])
        finally:
            try:
                if pending is not None:
                    self.move_module(pending[0], 'cpu')
            finally:
                self.pending = None
                self._inflight.clear()
