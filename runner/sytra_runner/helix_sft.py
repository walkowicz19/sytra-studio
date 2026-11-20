"""Token-weighted optimization loop for Helix windowed SFT."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .helix_mixtral import moe_lora_backward


@dataclass(frozen=True)
class HelixBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def supervised_tokens(self) -> int:
        if (self.input_ids.ndim != 2 or self.input_ids.shape[1] < 2 or
                self.input_ids.dtype != torch.long or self.labels.dtype != torch.long or
                self.attention_mask.shape != self.input_ids.shape or self.labels.shape != self.input_ids.shape or
                any(t.device.type != 'cpu' for t in (self.input_ids, self.attention_mask, self.labels))):
            raise ValueError('Helix batches require matching CPU [batch, sequence] tensors')
        if not torch.all((self.attention_mask == 0) | (self.attention_mask == 1)):
            raise ValueError('Helix attention masks must be binary')
        if torch.any((self.attention_mask == 0) & (self.labels != -100)):
            raise ValueError('Padding must be excluded from supervised labels')
        count = int((self.labels[:, 1:] != -100).sum().item())
        if not count:
            raise ValueError('Every Helix batch must contain supervised tokens')
        return count


@dataclass(frozen=True)
class HelixPackedBatch(HelixBatch):
    """One unpadded token sequence plus its original conversation boundaries."""
    cu_seqlens: torch.Tensor

    def supervised_tokens(self) -> int:
        count = super().supervised_tokens()
        boundaries = self.cu_seqlens
        if (self.input_ids.shape[0] != 1 or not torch.all(self.attention_mask == 1) or
                boundaries.device.type != 'cpu' or boundaries.dtype != torch.long or
                boundaries.ndim != 1 or boundaries.numel() < 2 or boundaries[0].item() != 0 or
                boundaries[-1].item() != self.input_ids.shape[1] or
                torch.any(boundaries[1:] - boundaries[:-1] < 2)):
            raise ValueError('Invalid packed conversation boundaries')
        if torch.any(self.labels[0, boundaries[:-1]] != -100):
            raise ValueError('First token of each packed conversation must be excluded from loss')
        return count


def _coerce_token_tensor(raw: Any, *, dtype: torch.dtype, name: str) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 1 or raw.numel() < 2:
            raise ValueError(f'{name} tensor must be 1D with at least 2 elements')
        return raw.to(dtype=dtype)
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.size < 2:
            raise ValueError(f'{name} array must be 1D with at least 2 elements')
        return torch.from_numpy(np.array(raw, copy=True)).to(dtype=dtype)
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError('Dataset rows require aligned token IDs and binary assistant masks')
    return torch.tensor(raw, dtype=dtype)


class HelixDatasetBatches(Sequence[HelixBatch]):
    """Lazily collate ordered, pretokenized assistant-masked dataset rows."""

    def __init__(self, dataset, batch_size: int, pad_token_id: int):
        if not len(dataset) or type(batch_size) is not int or batch_size <= 0:
            raise ValueError('A non-empty dataset and positive batch size are required')
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError('A non-negative tokenizer pad token ID is required')
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.batch_size
        rows = [self.dataset[i] for i in range(start, min(start + self.batch_size, len(self.dataset)))]
        parsed: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row in rows:
            ids_raw, mask_raw = row.get('input_ids'), row.get('assistant_masks')
            if isinstance(ids_raw, list) and (
                    any(type(value) is not int or value < 0 for value in ids_raw)):
                raise ValueError('Dataset rows require aligned token IDs and binary assistant masks')
            if isinstance(mask_raw, list) and (
                    any(type(value) not in (int, bool) or value not in (0, 1) for value in mask_raw)):
                raise ValueError('Dataset rows require aligned token IDs and binary assistant masks')
            ids = _coerce_token_tensor(ids_raw, dtype=torch.long, name='input_ids')
            mask = _coerce_token_tensor(mask_raw, dtype=torch.long, name='assistant_masks')
            if ids.numel() != mask.numel() or bool((ids < 0).any()) or not bool(torch.all((mask == 0) | (mask == 1))):
                raise ValueError('Dataset rows require aligned token IDs and binary assistant masks')
            parsed.append((ids, mask.bool()))
        width = max(ids.numel() for ids, _ in parsed)
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for i, (ids, supervised) in enumerate(parsed):
            input_ids[i, :ids.numel()] = ids
            attention[i, :ids.numel()] = 1
            labels[i, :ids.numel()] = ids.masked_fill(~supervised, -100)
        result = HelixBatch(input_ids, attention, labels)
        result.supervised_tokens()
        return result


class HelixPackedDatasetBatches(Sequence[HelixPackedBatch]):
    """Pack consecutive complete rows up to a token limit without padding or splitting."""

    def __init__(self, dataset, max_seq_len: int):
        if not len(dataset) or type(max_seq_len) is not int or max_seq_len < 2:
            raise ValueError('A non-empty dataset and sequence limit of at least two are required')
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.ranges = []
        first = width = 0
        seq_len_fn = getattr(dataset, 'sequence_length', None)
        for index in range(len(dataset)):
            if callable(seq_len_fn):
                length = seq_len_fn(index)
            else:
                ids = dataset[index].get('input_ids')
                if not isinstance(ids, (list, torch.Tensor, np.ndarray)):
                    raise ValueError('Every packed conversation must fit the sequence limit and contain at least two tokens')
                length = len(ids)
            if not 2 <= length <= max_seq_len:
                raise ValueError('Every packed conversation must fit the sequence limit and contain at least two tokens')
            if width + length > max_seq_len:
                self.ranges.append((first, index))
                first, width = index, 0
            width += length
        self.ranges.append((first, len(dataset)))

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        start, end = self.ranges[index]
        rows = [self.dataset[i] for i in range(start, end)]
        # Reuse the existing row validation, including binary assistant masks.
        parts = [HelixDatasetBatches([row], 1, 0)[0] for row in rows]
        sizes = [part.input_ids.shape[1] for part in parts]
        if sum(sizes) > self.max_seq_len:
            raise ValueError('Packed dataset changed after the index was built')
        ids = torch.cat([part.input_ids for part in parts], dim=1)
        labels = torch.cat([part.labels for part in parts], dim=1)
        boundaries = [0]
        for size in sizes:
            boundaries.append(boundaries[-1] + size)
        labels[0, boundaries[:-1]] = -100
        batch = HelixPackedBatch(ids, torch.ones_like(ids), labels, torch.tensor(boundaries, dtype=torch.long))
        batch.supervised_tokens()
        return batch


def train_helix_batches(model, pager, batches: Sequence[HelixBatch], optimizer, *, max_steps: int,
                        accumulation_steps: int = 1, start_step: int = 0, device: str = 'cpu', scheduler=None,
                        on_step: Callable[[int, float, int], None] | None = None,
                        on_stats: Callable[[dict[str, Any]], None] | None = None,
                        backward: Callable = moe_lora_backward, router_aux_coef: float = 0.0,
                        activation_memory_bytes: int | None = None, activation_spill_dir: str | None = None,
                        prefetch_layers: bool = False, prefetch_pinned_memory_bytes: int = 256 * 1024 * 1024,
                        neftune_noise_alpha: float = 0.0, prefetch_batches: bool = False) -> None:
    """Train over complete batches, cycling the dataset without dropping rows.

    Each optimizer step uses a single token-normalized loss across its
    accumulation window. The scheduler, if provided, advances once per step.
    start_step restores the deterministic position in this ordered batch list.
    This primitive does not implement shuffling or validation. When batch
    prefetch is enabled, the sequence must be deterministic and safe for
    read-only access from one background thread.
    """
    if not batches or type(max_steps) is not int or max_steps <= 0 or type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError('Non-empty batches and positive step counts are required')
    if type(prefetch_layers) is not bool:
        raise ValueError('prefetch_layers must be a boolean')
    if type(prefetch_batches) is not bool:
        raise ValueError('prefetch_batches must be a boolean')
    if type(prefetch_pinned_memory_bytes) is not int or prefetch_pinned_memory_bytes < 0:
        raise ValueError('prefetch_pinned_memory_bytes must be a non-negative integer')
    if type(start_step) is not int or not 0 <= start_step < max_steps:
        raise ValueError('start_step must be non-negative and below max_steps')
    cursor = (start_step * accumulation_steps) % len(batches)
    optimizer.zero_grad(set_to_none=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='helix-batch') if prefetch_batches else None
    future = None
    try:
        for step in range(start_step + 1, max_steps + 1):
            step_started = time.perf_counter()
            indexes = [(cursor + offset) % len(batches) for offset in range(accumulation_steps)]
            cursor = (cursor + accumulation_steps) % len(batches)
            wait_started = time.perf_counter()
            first_batch = future.result() if future is not None else batches[indexes[0]]
            data_wait_ms = (time.perf_counter() - wait_started) * 1000.0
            future = None
            counts = [first_batch.supervised_tokens()]
            counts.extend(batches[index].supervised_tokens() for index in indexes[1:])
            total = sum(counts)
            loss_sum = 0.0
            spill_count = 0
            spill_bytes = 0
            peak_resident = 0
            peak_pinned = 0
            prefetch_wait_ms = 0.0
            for offset, (index, count) in enumerate(zip(indexes, counts)):
                if offset == 0:
                    batch = first_batch
                    first_batch = None
                else:
                    w0 = time.perf_counter()
                    batch = future.result() if future is not None else batches[index]
                    data_wait_ms += (time.perf_counter() - w0) * 1000.0
                    future = None
                if executor is not None:
                    next_index = (indexes[offset + 1] if offset + 1 < len(indexes) else
                                  cursor if step < max_steps else None)
                    if next_index is not None:
                        future = executor.submit(batches.__getitem__, next_index)
                storage_options = {} if activation_memory_bytes is None else {
                    'activation_memory_bytes': activation_memory_bytes, 'activation_spill_dir': activation_spill_dir}
                result = backward(model, batch.input_ids, batch.attention_mask, batch.labels,
                                  device=device, weight_pager=pager,
                                  gradient_scale=count / total,
                                  **storage_options,
                                  **({'prefetch_layers': True, 'prefetch_pinned_memory_bytes': prefetch_pinned_memory_bytes} if prefetch_layers else {}),
                                  **({'neftune_noise_alpha': neftune_noise_alpha} if neftune_noise_alpha else {}),
                                  **({'cu_seqlens': batch.cu_seqlens} if isinstance(batch, HelixPackedBatch) else {}),
                                  **({'router_aux_coef': router_aux_coef} if router_aux_coef else {}))
                if result.supervised_tokens != count:
                    raise RuntimeError('Helix backward returned an inconsistent supervised token count')
                loss_sum += result.loss * count
                spill_count += getattr(result, 'activation_spill_count', 0)
                spill_bytes += getattr(result, 'activation_spill_bytes', 0)
                peak_resident = max(peak_resident, getattr(result, 'activation_peak_resident_bytes', 0))
                peak_pinned = max(peak_pinned, getattr(result, 'prefetch_peak_pinned_bytes', 0))
                prefetch_wait_ms += getattr(result, 'prefetch_wait_ms', 0.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_ms = (time.perf_counter() - step_started) * 1000.0
            if on_stats is not None:
                on_stats({
                    'step': step,
                    'step_ms': step_ms,
                    'data_wait_ms': data_wait_ms,
                    'prefetch_wait_ms': prefetch_wait_ms,
                    'gpu_wait_ms': data_wait_ms + prefetch_wait_ms,
                    'activation_spill_count': spill_count,
                    'activation_spill_bytes': spill_bytes,
                    'activation_peak_resident_bytes': peak_resident,
                    'prefetch_peak_pinned_bytes': peak_pinned,
                })
            if on_step is not None:
                on_step(step, loss_sum / total, total)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if executor is not None:
            if future is not None:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
