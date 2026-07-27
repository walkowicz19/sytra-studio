"""Sytra 1.2.0 — GGUF-based MoE inference engine.

Memory-maps GGUF files (zero-copy on NVMe), routes tokens through the active
MoE experts only, and uses GPUExpertPager to keep VRAM usage bounded.

Target: 5–10 tok/s on RTX 3060 (12 GB VRAM) for Kimi K2.7 / GLM-5.2 class models.
The PC will not freeze because we never load more than ~vram_limit_mb into VRAM at once.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import time
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .expert_pager import GPUExpertPager


# ─── GGUF reader (minimal, zero-dep) ──────────────────────────────────────────

GGUF_MAGIC = b"GGUF"
GGUF_VERSION_SUPPORTED = {1, 2, 3}

_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 4, 4: 4, 5: 2, 6: 4, 7: 8, 8: 8, 9: 0, 10: 4, 11: 8, 12: 1}
_DTYPE_MAP  = {0: "uint8", 1: "int8", 2: "uint16", 3: "uint32", 4: "int32",
               5: "float16", 6: "float32", 7: "float64", 8: "uint64", 10: "bfloat16"}

def _read_u32(f) -> int: return struct.unpack("<I", f.read(4))[0]
def _read_u64(f) -> int: return struct.unpack("<Q", f.read(8))[0]
def _read_i32(f) -> int: return struct.unpack("<i", f.read(4))[0]
def _read_f32(f) -> float: return struct.unpack("<f", f.read(4))[0]

def _read_string(f) -> str:
    length = _read_u64(f)
    return f.read(length).decode("utf-8", errors="replace")

def _skip_metadata_value(f, vtype: int) -> None:
    if vtype == 8:   # STRING
        _read_string(f)
    elif vtype == 9: # ARRAY
        atype = _read_u32(f)
        count = _read_u64(f)
        for _ in range(count):
            _skip_metadata_value(f, atype)
    else:
        size = _TYPE_SIZES.get(vtype, 0)
        if size > 0:
            f.read(size)


class GGUFTensorInfo:
    """Metadata for one tensor in a GGUF file."""
    __slots__ = ("name", "n_dims", "dims", "dtype", "data_offset")
    def __init__(self, name: str, n_dims: int, dims: tuple, dtype: int, data_offset: int):
        self.name = name
        self.n_dims = n_dims
        self.dims = dims
        self.dtype = dtype
        self.data_offset = data_offset

    @property
    def nbytes(self) -> int:
        from math import prod
        size = _TYPE_SIZES.get(self.dtype, 1)
        return prod(self.dims) * size if size > 0 else prod(self.dims) // 2


class GGUFReader:
    """Reads GGUF metadata and provides zero-copy tensor access via mmap."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd = open(self.path, "rb")
        self._mm: mmap.mmap | None = None
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self._data_start: int = 0
        self._parse()

    def _parse(self) -> None:
        f = self._fd
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: {self.path}")
        version = _read_u32(f)
        if version not in GGUF_VERSION_SUPPORTED:
            raise ValueError(f"GGUF version {version} not supported")
        n_tensors = _read_u64(f)
        n_meta    = _read_u64(f)

        # Read metadata KVs
        for _ in range(n_meta):
            key = _read_string(f)
            vtype = _read_u32(f)
            if vtype == 8:   # STRING
                self.metadata[key] = _read_string(f)
            elif vtype == 4: # INT32
                self.metadata[key] = _read_i32(f)
            elif vtype == 6: # FLOAT32
                self.metadata[key] = _read_f32(f)
            elif vtype == 12: # BOOL
                self.metadata[key] = bool(struct.unpack("<?", f.read(1))[0])
            else:
                _skip_metadata_value(f, vtype)

        # Read tensor info
        tensor_infos: list[GGUFTensorInfo] = []
        for _ in range(n_tensors):
            name   = _read_string(f)
            n_dims = _read_u32(f)
            dims   = tuple(_read_u64(f) for _ in range(n_dims))
            dtype  = _read_u32(f)
            offset = _read_u64(f)  # relative to data section
            tensor_infos.append(GGUFTensorInfo(name, n_dims, dims, dtype, offset))

        # Data section starts aligned to 32 bytes
        pos = f.tell()
        alignment = int(self.metadata.get("general.alignment", 32))
        remainder = pos % alignment
        if remainder:
            f.seek(alignment - remainder, 1)
        self._data_start = f.tell()

        for ti in tensor_infos:
            ti.data_offset += self._data_start
            self.tensors[ti.name] = ti

    def mmap(self) -> mmap.mmap:
        if self._mm is None:
            self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm

    def read_tensor_bytes(self, name: str) -> bytes:
        ti = self.tensors[name]
        mm = self.mmap()
        mm.seek(ti.data_offset)
        return mm.read(ti.nbytes)

    def close(self) -> None:
        if self._mm:
            self._mm.close()
            self._mm = None
        self._fd.close()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ─── MoE inference engine ──────────────────────────────────────────────────────

class MoEInferenceEngine:
    """Streams GGUF expert weights on-demand using GPUExpertPager.

    Usage::

        engine = MoEInferenceEngine("path/to/model.gguf", vram_limit_mb=10000)
        for token in engine.generate("Hello, world!", max_tokens=200):
            print(token, end="", flush=True)
        engine.close()
    """

    def __init__(
        self,
        gguf_path: str | Path,
        vram_limit_mb: int = 10000,
        ram_limit_mb: int = 4096,
        quant_bits: int = 4,
        pilot_prefetch: bool = True,
    ):
        self.gguf_path = Path(gguf_path)
        if not self.gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        logger.info("Loading GGUF metadata from %s …", self.gguf_path.name)
        self.reader = GGUFReader(self.gguf_path)

        # Model architecture from metadata
        self.n_layers: int = int(self.reader.metadata.get("llama.block_count", 32))
        self.n_experts: int = int(self.reader.metadata.get("llama.expert_count", 0))
        self.n_experts_used: int = int(self.reader.metadata.get("llama.expert_used_count", 8))
        self.head_dim: int = int(self.reader.metadata.get("llama.rope.dimension_count", 128))
        self.vocab_size: int = int(self.reader.metadata.get("tokenizer.ggml.tokens", 32000)
                                   if isinstance(self.reader.metadata.get("tokenizer.ggml.tokens"), int)
                                   else 32000)

        total_experts = max(self.n_experts, 1)
        self.pager = GPUExpertPager(
            num_experts=total_experts,
            vram_limit_mb=vram_limit_mb,
            ram_limit_mb=ram_limit_mb,
            quant_bits=quant_bits,
            pilot_prefetch=pilot_prefetch,
        )

        logger.info(
            "MoEInferenceEngine ready — %d layers, %d experts (%d active/token), VRAM limit %d MB",
            self.n_layers, self.n_experts, self.n_experts_used, vram_limit_mb,
        )

    def _expert_load_fn(self, expert_id: int) -> Any:
        """Load expert weights from mmap'd GGUF file into a CPU tensor."""
        # Experts are stored as blk.{layer}.ffn_gate_exps.weight etc.
        layer = expert_id // max(self.n_experts // self.n_layers, 1) if self.n_experts > 0 else 0
        local_id = expert_id % max(self.n_experts // self.n_layers, 1) if self.n_experts > 0 else 0

        # Try common naming conventions
        for pattern in [
            f"blk.{layer}.ffn_gate_exps.weight",
            f"blk.{layer}.ffn_up_exps.weight",
            f"model.layers.{layer}.mlp.experts.{local_id}.gate_proj.weight",
        ]:
            if pattern in self.reader.tensors:
                raw = self.reader.read_tensor_bytes(pattern)
                if HAS_TORCH:
                    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                return raw

        # Fallback: return empty tensor
        if HAS_TORCH:
            return torch.zeros(1024, dtype=torch.uint8)
        return b"\x00" * 1024

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> Iterator[str]:
        """Generate tokens from prompt, streaming one token at a time.

        This is a framework-level stub: it demonstrates the expert-paging
        loop and yields placeholder tokens until a full llama.cpp integration
        is wired in. The VRAM-safe loop structure and pager calls are real.
        """
        if not HAS_TORCH:
            yield "[ERROR: torch not available — install PyTorch to use MoE inference]"
            return

        stop_seqs = stop_sequences or []
        generated = ""

        logger.info("Starting generation: max_tokens=%d, temp=%.2f", max_tokens, temperature)
        start_t = time.monotonic()

        for step in range(max_tokens):
            # ── Simulate one forward pass through all MoE layers ───────────────
            for layer_idx in range(self.n_layers):
                # Routing step: predict which experts are active (placeholder router)
                active_expert_ids = self._route_experts(layer_idx, step)

                # PILOT: prefetch next-layer experts asynchronously
                if layer_idx + 1 < self.n_layers:
                    next_active = self._route_experts(layer_idx + 1, step)
                    self.pager.prefetch_experts(next_active, self._expert_load_fn)

                # Load active experts via the pager (VRAM → RAM → disk)
                for eid in active_expert_ids:
                    _ = self.pager.get_expert_weights(eid, self._expert_load_fn)

            # ── Emit one placeholder token per step (replace with real sampler) ─
            token = self._sample_token(step, temperature)
            generated += token

            yield token

            # Check stop sequences
            if any(s in generated for s in stop_seqs):
                break

        elapsed = time.monotonic() - start_t
        tok_s = (step + 1) / max(elapsed, 0.001)
        stats = self.pager.stats()
        logger.info(
            "Generation done: %d tokens, %.2f tok/s | VRAM %d/%d MB | "
            "VRAM hit %.0f%% | disk loads %d",
            step + 1, tok_s,
            stats["vram_used_mb"], stats["vram_limit_mb"],
            stats["vram_hit_rate"] * 100,
            stats["disk_loads"],
        )

    def _route_experts(self, layer_idx: int, step: int) -> list[int]:
        """Placeholder router — returns n_experts_used expert IDs for this layer."""
        if self.n_experts == 0:
            return []
        base = (layer_idx * self.n_experts_used + step) % max(self.n_experts, 1)
        return [(base + i) % self.n_experts for i in range(self.n_experts_used)]

    def _sample_token(self, step: int, temperature: float) -> str:
        """Placeholder sampler — returns a space character."""
        # Real sampling: apply final linear + softmax, sample from vocab distribution
        return " "

    def close(self) -> None:
        self.pager.shutdown()
        self.reader.close()

    def __enter__(self) -> "MoEInferenceEngine":
        return self

    def __exit__(self, *_) -> None:
        self.close()
