"""Memory-mapped pre-tokenized binary dataset cache for Helix training."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import torch

_FORMAT_VERSION = 1


class MmapTokenDataset(Sequence):
    """Read-only sequence backed by flat ``int32`` token and ``uint8`` mask mmaps."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve(strict=True)
        meta = json.loads((self.directory / 'meta.json').read_text(encoding='utf-8'))
        if meta.get('version') != _FORMAT_VERSION:
            raise ValueError('Unsupported Helix memmap dataset cache version')
        self.meta = meta
        self._offsets = np.load(self.directory / 'offsets.npy', mmap_mode='r')
        if self._offsets.ndim != 1 or len(self._offsets) < 2 or int(self._offsets[0]) != 0:
            raise ValueError('Invalid Helix memmap offsets')
        total = int(self._offsets[-1])
        tokens_path = self.directory / 'tokens.bin'
        masks_path = self.directory / 'masks.bin'
        if tokens_path.stat().st_size != total * 4 or masks_path.stat().st_size != total:
            raise ValueError('Helix memmap storage size does not match offsets')
        self._tokens = np.memmap(tokens_path, dtype=np.int32, mode='r', shape=(total,)) if total else np.empty((0,), dtype=np.int32)
        self._masks = np.memmap(masks_path, dtype=np.uint8, mode='r', shape=(total,)) if total else np.empty((0,), dtype=np.uint8)
        self.count = len(self._offsets) - 1

    def __len__(self) -> int:
        return self.count

    def sequence_length(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError(index)
        return int(self._offsets[index + 1] - self._offsets[index])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if type(index) is not int or not 0 <= index < self.count:
            raise IndexError(index)
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        tokens = torch.from_numpy(np.array(self._tokens[start:end], dtype=np.int64, copy=True))
        masks = torch.from_numpy(np.array(self._masks[start:end], dtype=np.uint8, copy=True))
        return {'input_ids': tokens, 'assistant_masks': masks}

    def close(self) -> None:
        for attr in ('_tokens', '_masks', '_offsets'):
            arr = getattr(self, attr, None)
            if isinstance(arr, np.memmap) and arr._mmap is not None:
                arr._mmap.close()
            setattr(self, attr, None)


def build_memmap_dataset(
    rows: Iterable[Mapping[str, Any]],
    cache_dir: str | Path,
    *,
    fingerprint: str | None = None,
    force: bool = False,
) -> MmapTokenDataset:
    """Atomically write ``tokens.bin`` + ``masks.bin`` + ``offsets.npy`` and open via mmap."""
    target = Path(cache_dir).resolve()
    meta_path = target / 'meta.json'
    if not force and meta_path.is_file():
        try:
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
            if (existing.get('version') == _FORMAT_VERSION and
                    (fingerprint is None or existing.get('fingerprint') == fingerprint)):
                return MmapTokenDataset(target)
        except Exception:
            pass

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix='.helix-mmap-', dir=target.parent))
    try:
        offsets = [0]
        digest = hashlib.sha256()
        with open(staged / 'tokens.bin', 'wb') as tok_f, open(staged / 'masks.bin', 'wb') as mask_f:
            for row in rows:
                ids_raw = row['input_ids']
                mask_raw = row['assistant_masks']
                ids = np.asarray(ids_raw, dtype=np.int64)
                masks = np.asarray(mask_raw, dtype=np.int64)
                if ids.ndim != 1 or masks.ndim != 1 or len(ids) != len(masks) or len(ids) < 2:
                    raise ValueError('Memmap rows require aligned 1D token IDs and masks of length >= 2')
                if np.any(ids < 0) or np.any(ids > np.iinfo(np.int32).max):
                    raise ValueError('Token ID exceeds int32 range')
                if not np.all((masks == 0) | (masks == 1)):
                    raise ValueError('Assistant masks must be binary 0/1')
                tok_bytes = ids.astype(np.int32).tobytes()
                mask_bytes = masks.astype(np.uint8).tobytes()
                tok_f.write(tok_bytes)
                mask_f.write(mask_bytes)
                digest.update(tok_bytes)
                digest.update(mask_bytes)
                offsets.append(offsets[-1] + len(ids))
        if len(offsets) < 2:
            raise ValueError('Cannot build memmap cache for an empty dataset')
        offsets_arr = np.asarray(offsets, dtype=np.int64)
        np.save(staged / 'offsets.npy', offsets_arr)
        meta = {
            'version': _FORMAT_VERSION,
            'count': len(offsets) - 1,
            'total_tokens': int(offsets[-1]),
            'digest': digest.hexdigest(),
            **({'fingerprint': fingerprint} if fingerprint is not None else {}),
        }
        (staged / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if target.exists():
            shutil.rmtree(target)
        os.replace(staged, target)
    finally:
        if staged.exists():
            shutil.rmtree(staged, ignore_errors=True)
    return MmapTokenDataset(target)
