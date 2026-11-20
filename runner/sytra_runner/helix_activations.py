"""Bounded activation storage for Helix replay, with optional CUDA D2H overlap."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch


@dataclass
class _Entry:
    size: int
    tensor: torch.Tensor | None = None
    path: Path | None = None
    event: torch.cuda.Event | None = None


class ActivationStash:
    """Store detached activations under a RAM budget and spill the rest to disk.

    The budget covers stored tensors, excluding one transient loaded tensor and
    the active GPU window. CUDA copies to pinned memory use a dedicated stream;
    consumers wait for the corresponding event before reading. Spill files are
    temporary and removed on both successful and failed runs.
    """

    def __init__(self, max_bytes: int, *, device: str | torch.device = 'cpu', directory: str | Path | None = None):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Activation memory budget must be a non-negative integer')
        self.max_bytes = max_bytes
        self.device = torch.device(device)
        if self.device.type not in {'cpu', 'cuda'} or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Activation stash requires CPU or available CUDA')
        self.directory = None if directory is None else str(directory)
        self._temp = None
        self._entries: list[_Entry] = []
        self._used = 0
        self._peak_used = 0
        self._spill_count = 0
        self._spill_bytes = 0
        self._sequence = 0
        self._stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    @property
    def resident_bytes(self) -> int:
        return self._used

    @property
    def peak_resident_bytes(self) -> int:
        return self._peak_used

    @property
    def spill_count(self) -> int:
        return self._spill_count

    @property
    def spill_bytes(self) -> int:
        return self._spill_bytes

    def push(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor) or value.device.type not in {'cpu', 'cuda'}:
            raise ValueError('Activation must be a CPU or CUDA tensor')
        if value.device.type == 'cuda' and (self.device.type != 'cuda' or value.device != torch.device('cuda', self.device.index if self.device.index is not None else torch.cuda.current_device())):
            raise ValueError('Activation device does not match the stash')
        size = value.numel() * value.element_size()
        tensor = None
        event = None
        if size <= self.max_bytes - self._used:
            try:
                tensor = torch.empty(value.shape, dtype=value.dtype, device='cpu', pin_memory=self._stream is not None)
            except RuntimeError:
                # Pinned allocations can fail under OS pressure; spill instead.
                tensor = None
            if tensor is not None:
                if value.device.type == 'cuda':
                    self._stream.wait_stream(torch.cuda.current_stream(value.device))
                    with torch.cuda.stream(self._stream):
                        tensor.copy_(value.detach(), non_blocking=True)
                        event = torch.cuda.Event()
                        event.record(self._stream)
                        value.record_stream(self._stream)
                else:
                    tensor.copy_(value.detach())
                self._used += size
                if self._used > self._peak_used:
                    self._peak_used = self._used
                self._entries.append(_Entry(size=size, tensor=tensor, event=event))
                return
        if self._temp is None:
            self._temp = tempfile.TemporaryDirectory(prefix='.helix-activations-', dir=self.directory)
        path = Path(self._temp.name) / f'{self._sequence:08d}.pt'
        self._sequence += 1
        torch.save(value.detach().to('cpu'), path)
        self._spill_count += 1
        self._spill_bytes += size
        self._entries.append(_Entry(size=size, path=path))

    def last(self) -> torch.Tensor:
        if not self._entries:
            raise IndexError('Activation stash is empty')
        entry = self._entries[-1]
        if entry.event is not None:
            entry.event.synchronize()
            entry.event = None
        if entry.tensor is not None:
            return entry.tensor
        value = torch.load(entry.path, map_location='cpu', weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError('Invalid spilled activation')
        return value

    def pop(self) -> torch.Tensor:
        value = self.last()
        entry = self._entries.pop()
        if entry.tensor is not None:
            self._used -= entry.size
        else:
            entry.path.unlink()
        return value

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.synchronize()
        finally:
            self._entries.clear()
            self._used = 0
            if self._temp is not None:
                self._temp.cleanup()
                self._temp = None
