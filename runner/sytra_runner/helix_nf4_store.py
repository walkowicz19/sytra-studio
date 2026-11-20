"""Stream an individual frozen matrix to Sytra's memory-mappable NF4 format."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from safetensors import safe_open

from .helix_nf4 import FrozenNF4Linear, quantize_nf4


def write_nf4_matrix(weight: torch.Tensor, path: str | Path, *, block_size: int = 64,
                     blocks_per_chunk: int = 4096) -> Path:
    """Atomically create a new safetensors matrix, quantizing one block chunk at a time.

    Existing files are never replaced. The input must remain unchanged during
    conversion; callers may pass a memory-mapped safetensors tensor.
    """
    if sys.byteorder != 'little':
        raise ValueError('NF4 storage currently requires a little-endian host')
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or
            weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            not weight.is_contiguous() or weight.numel() == 0):
        raise ValueError('NF4 storage requires a non-empty contiguous CPU floating-point matrix')
    if type(block_size) is not int or block_size < 2 or block_size % 2 or type(blocks_per_chunk) is not int or blocks_per_chunk <= 0:
        raise ValueError('NF4 block size must be positive and even')
    target = Path(path)
    if target.suffix != '.safetensors' or not target.parent.is_dir():
        raise ValueError('NF4 output must be a safetensors file in an existing directory')
    if target.exists():
        raise FileExistsError(f'NF4 output already exists: {target}')
    rows, cols = weight.shape
    total = weight.numel()
    blocks = (total + block_size - 1) // block_size
    packed_bytes = (total + 1) // 2
    scale_bytes = blocks * 4
    header = {
        '__metadata__': {'format': 'sytra_nf4_v1', 'rows': str(rows), 'cols': str(cols),
                         'block_size': str(block_size), 'codebook': 'nf4', 'nibble_order': 'low_first'},
        'packed_weight': {'dtype': 'U8', 'shape': [packed_bytes], 'data_offsets': [0, packed_bytes]},
        'weight_scales': {'dtype': 'F32', 'shape': [blocks], 'data_offsets': [packed_bytes, packed_bytes + scale_bytes]},
    }
    encoded = json.dumps(header, separators=(',', ':')).encode('utf-8')
    encoded += b' ' * (-len(encoded) % 8)
    if shutil.disk_usage(target.parent).free < 8 + len(encoded) + packed_bytes + scale_bytes:
        raise OSError(f'Insufficient free disk space for NF4 cache entry in {target.parent}')
    descriptor, temporary = tempfile.mkstemp(prefix='.sytra-nf4-', suffix='.tmp', dir=target.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w+b') as output:
            output.write(len(encoded).to_bytes(8, 'little'))
            output.write(encoded)
            data_start = output.tell()
            output.truncate(data_start + packed_bytes + scale_bytes)
            flat = weight.detach().view(-1)
            for first_block in range(0, blocks, blocks_per_chunk):
                last_block = min(first_block + blocks_per_chunk, blocks)
                first = first_block * block_size
                last = min(last_block * block_size, total)
                packed, scales = quantize_nf4(flat[first:last].view(1, -1), block_size=block_size,
                                               blocks_per_chunk=blocks_per_chunk)
                output.seek(data_start + first // 2)
                output.write(packed.numpy().tobytes())
                output.seek(data_start + packed_bytes + first_block * 4)
                output.write(scales.numpy().tobytes())
            output.flush()
            os.fsync(output.fileno())
        # The hard link is an atomic create-if-absent on the same filesystem.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def open_nf4_matrix(path: str | Path, *, chunk_rows: int = 32) -> FrozenNF4Linear:
    """Open a Sytra NF4 matrix as CPU-mapped tensors without copying its data."""
    with safe_open(str(path), framework='pt', device='cpu') as file:
        meta = file.metadata() or {}
        if (meta.get('format') != 'sytra_nf4_v1' or meta.get('codebook') != 'nf4' or
                meta.get('nibble_order') != 'low_first' or set(file.keys()) != {'packed_weight', 'weight_scales'}):
            raise ValueError('Unsupported NF4 storage format')
        try:
            rows, cols, block_size = (int(meta[key]) for key in ('rows', 'cols', 'block_size'))
        except (ValueError, KeyError) as exc:
            raise ValueError('Invalid NF4 matrix metadata') from exc
        packed, scales = file.get_tensor('packed_weight'), file.get_tensor('weight_scales')
    return FrozenNF4Linear(packed, scales, cols, rows, block_size=block_size, chunk_rows=chunk_rows)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(weight: torch.Tensor, block_size: int) -> str:
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.ndim != 2 or
            not weight.is_contiguous() or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not weight.numel()):
        raise ValueError('NF4 cache requires a non-empty contiguous CPU floating-point matrix')
    digest = hashlib.sha256()
    digest.update(json.dumps({'format': 'sytra_nf4_v1', 'shape': list(weight.shape),
                              'dtype': str(weight.dtype), 'block_size': block_size}, sort_keys=True).encode('utf-8'))
    raw = weight.detach().view(torch.uint8).view(-1)
    for start in range(0, raw.numel(), 1024 * 1024):
        digest.update(memoryview(raw[start:start + 1024 * 1024].numpy()))
    return digest.hexdigest()


def cached_nf4_matrix(weight: torch.Tensor, cache_dir: str | Path, *, block_size: int = 64,
                      chunk_rows: int = 32) -> FrozenNF4Linear:
    """Create or reuse a checksummed NF4 cache entry keyed by source contents."""
    if type(block_size) is not int or block_size < 2 or block_size % 2:
        raise ValueError('NF4 block size must be positive and even')
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    source_hash = _source_digest(weight, block_size)
    path = cache / f'{source_hash}.safetensors'
    checksum = cache / f'{source_hash}.sha256'
    if path.exists() != checksum.exists():
        raise ValueError(f'Incomplete NF4 cache entry: {path.name}')
    if not path.exists():
        write_nf4_matrix(weight, path, block_size=block_size)
        try:
            if _source_digest(weight, block_size) != source_hash:
                raise RuntimeError('Source weights changed while the NF4 cache was being created')
            digest = _file_digest(path)
            descriptor, temporary = tempfile.mkstemp(prefix='.sytra-nf4-checksum-', dir=cache)
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, 'w', encoding='ascii') as handle:
                    handle.write(digest + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, checksum)
            finally:
                temporary.unlink(missing_ok=True)
        except BaseException:
            if not checksum.exists():
                path.unlink(missing_ok=True)
            raise
    else:
        expected = checksum.read_text(encoding='ascii').strip()
        if len(expected) != 64 or expected != _file_digest(path):
            raise ValueError(f'NF4 cache checksum mismatch: {path.name}')
        if _source_digest(weight, block_size) != source_hash:
            raise RuntimeError('Source weights changed while the NF4 cache was being verified')
    result = open_nf4_matrix(path, chunk_rows=chunk_rows)
    if result.out_features != weight.shape[0] or result.in_features != weight.shape[1] or result.block_size != block_size:
        raise ValueError('NF4 cache dimensions do not match source weights')
    result.sytra_source_digest = source_hash
    return result
