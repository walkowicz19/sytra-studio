"""Conservative memory planning for model weights and KV cache."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass(frozen=True)
class KVCachePlan:
    tier: str
    dtype: str
    context_tokens: int
    estimated_bytes: int | None
    bytes_per_token: int | None
    formulation: str
    persistent: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WeightPlacementPlan:
    strategy: str
    weight_bytes: int
    vram_budget_bytes: int
    ram_budget_bytes: int
    estimated_vram_weight_bytes: int
    estimated_ram_weight_bytes: int
    estimated_nvme_weight_bytes: int
    cpu_offload_gb_per_gpu: float

    def to_dict(self) -> dict:
        return asdict(self)


def _positive_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def estimate_kv_cache(
    config: dict[str, Any],
    *,
    context_tokens: int,
    dtype: str,
    cpu_cache: bool,
    persistent: bool = False,
    tier: str | None = None,
) -> KVCachePlan:
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    normalized = dtype.lower()
    bytes_per_element = {
        "q4_0": 0.5,
        "q4_1": 0.5,
        "q5_0": 0.625,
        "q5_1": 0.625,
        "q8_0": 1.0,
        "fp8": 1.0,
        "float8": 1.0,
        "fp16": 2.0,
        "float16": 2.0,
        "bf16": 2.0,
        "bfloat16": 2.0,
        "fp32": 4.0,
        "float32": 4.0,
        "auto": 2.0,
    }.get(normalized)

    layers = _positive_int(config, "num_hidden_layers", "n_layer")
    kv_lora_rank = _positive_int(config, "kv_lora_rank")
    rope_dim = _positive_int(config, "qk_rope_head_dim") or 0
    formulation = "unknown"
    elements_per_token: int | None = None
    if layers and kv_lora_rank:
        # MLA stores the compressed latent plus the decoupled RoPE component
        # for every layer rather than full K and V heads.
        elements_per_token = layers * (kv_lora_rank + rope_dim)
        formulation = "MLA compressed latent"
    elif layers:
        kv_heads = _positive_int(
            config,
            "num_key_value_heads",
            "n_head_kv",
            "num_attention_heads",
            "n_head",
        )
        head_dim = _positive_int(config, "head_dim")
        hidden = _positive_int(config, "hidden_size", "n_embd")
        attention_heads = _positive_int(config, "num_attention_heads", "n_head")
        if head_dim is None and hidden and attention_heads:
            head_dim = hidden // attention_heads
        if kv_heads and head_dim:
            elements_per_token = layers * 2 * kv_heads * head_dim
            formulation = "standard K+V heads"

    if elements_per_token is None or bytes_per_element is None:
        total = None
        per_token = None
    else:
        per_token = math.ceil(elements_per_token * bytes_per_element)
        total = (per_token * context_tokens * 11 + 9) // 10

    return KVCachePlan(
        tier=tier or ("cpu" if cpu_cache else "gpu"),
        dtype=dtype,
        context_tokens=context_tokens,
        estimated_bytes=total,
        bytes_per_token=per_token,
        formulation=formulation,
        persistent=persistent,
    )


def plan_weight_placement(
    *,
    weight_bytes: int,
    vram_budget_mb: int,
    ram_budget_mb: int,
    tensor_parallel_size: int = 1,
    allow_cpu_offload: bool,
    allow_nvme_streaming: bool,
) -> WeightPlacementPlan:
    if weight_bytes <= 0:
        raise ValueError("weight_bytes must be positive")
    if vram_budget_mb <= 0 or ram_budget_mb < 0 or tensor_parallel_size <= 0:
        raise ValueError("invalid memory budget")

    # Keep 20% of configured VRAM for activations, CUDA graphs, and KV state.
    vram_weights = min(weight_bytes, int(vram_budget_mb * MIB * 0.80))
    remaining = max(weight_bytes - vram_weights, 0)
    ram_weights = 0
    nvme_weights = 0
    strategy = "gpu-resident"
    cpu_offload_gb = 0.0

    if remaining and allow_cpu_offload:
        ram_weights = min(remaining, int(ram_budget_mb * MIB * 0.85))
        remaining -= ram_weights
        if ram_weights:
            strategy = "vllm-uva"
            cpu_offload_gb = math.ceil(
                (ram_weights / GIB / tensor_parallel_size) * 10
            ) / 10
    if remaining and allow_nvme_streaming:
        nvme_weights = remaining
        remaining = 0
        strategy = "router-aware-vram-ram-nvme"
    if remaining:
        strategy = "insufficient-memory"

    return WeightPlacementPlan(
        strategy=strategy,
        weight_bytes=weight_bytes,
        vram_budget_bytes=vram_budget_mb * MIB,
        ram_budget_bytes=ram_budget_mb * MIB,
        estimated_vram_weight_bytes=vram_weights,
        estimated_ram_weight_bytes=ram_weights,
        estimated_nvme_weight_bytes=nvme_weights,
        cpu_offload_gb_per_gpu=cpu_offload_gb,
    )


@dataclass(frozen=True)
class LlamaCppOffloadPlan:
    gpu_layers: int
    cpu_layers: int
    n_layer: int | None
    peak_vram_mb: int
    peak_ram_mb: int
    mmap: bool
    mlock: bool
    threads: int
    batch: int
    ubatch: int
    strategy: str
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def plan_llama_cpp_offload(
    *,
    weight_bytes: int,
    vram_budget_mb: int,
    ram_budget_mb: int,
    n_layer: int | None,
    kv_bytes: int | None,
    cpu_count: int,
    windows: bool,
) -> LlamaCppOffloadPlan:
    """GPU-first hybrid: fill VRAM with layers, keep the rest mmap'd in RAM.

    20% of the VRAM budget is reserved for KV, CUDA context, and activations.
    15% of the RAM budget is reserved for the OS and llama.cpp working set.
    mlock is never enabled on Windows — it causes page-file thrashing on 16 GB PCs.
    """
    if weight_bytes <= 0 or vram_budget_mb <= 0 or ram_budget_mb < 0:
        raise ValueError("invalid llama.cpp offload budget")

    vram_weight_budget = int(vram_budget_mb * MIB * 0.80)
    ram_weight_budget = int(ram_budget_mb * MIB * 0.85)
    kv = kv_bytes or 0
    notes: list[str] = []
    threads = max(2, min(8, (cpu_count or 4) // 2 or 2))
    batch, ubatch = 512, 128
    mmap = True
    mlock = False
    if windows:
        notes.append("Windows: mmap on, mlock off to avoid page-file thrashing under a 12 GB RAM cap.")

    if n_layer and n_layer > 0:
        embedding_share = int(weight_bytes * 0.08)
        layer_bytes = max(1, (weight_bytes - embedding_share) // n_layer)
        gpu_layers = min(n_layer, vram_weight_budget // layer_bytes)
        if gpu_layers <= 0 and vram_weight_budget > embedding_share:
            gpu_layers = 1
        cpu_layers = max(n_layer - gpu_layers, 0)
        ngl = gpu_layers
    else:
        n_layer = None
        if weight_bytes <= vram_weight_budget:
            gpu_layers, cpu_layers, ngl = 99, 0, 99
        else:
            ratio = vram_weight_budget / weight_bytes
            ngl = max(1, int(99 * ratio))
            gpu_layers, cpu_layers = ngl, max(99 - ngl, 0)
        notes.append("GGUF header has no block_count; GPU layer count is size-proportional.")

    gpu_weight = min(weight_bytes, vram_weight_budget)
    ram_weight = max(weight_bytes - gpu_weight, 0)
    peak_vram = (gpu_weight + kv + int(0.20 * vram_budget_mb * MIB) + MIB - 1) // MIB
    peak_ram = (ram_weight + int(0.15 * ram_budget_mb * MIB) + MIB - 1) // MIB
    if ram_weight > ram_weight_budget:
        strategy = "insufficient-memory"
        notes.append(
            "Weights plus KV exceed the conservative GPU+RAM envelope; "
            "choose a smaller GGUF quant or a shorter context."
        )
    elif cpu_layers:
        strategy = "gpu-first-hybrid"
        notes.append(
            f"Keep {gpu_layers} transformer blocks on GPU; mmap the remaining {cpu_layers} in RAM."
        )
    else:
        strategy = "gpu-resident"
        notes.append("All transformer blocks fit the VRAM weight budget.")

    return LlamaCppOffloadPlan(
        gpu_layers=ngl,
        cpu_layers=cpu_layers,
        n_layer=n_layer,
        peak_vram_mb=peak_vram,
        peak_ram_mb=peak_ram,
        mmap=mmap,
        mlock=mlock,
        threads=threads,
        batch=batch,
        ubatch=ubatch,
        strategy=strategy,
        notes=tuple(notes),
    )


def estimate_streamed_moe_tps(
    *,
    nvme_weight_bytes: int,
    n_expert: int | None,
    n_expert_used: int | None,
    storage_bandwidth_mbps: int,
) -> float | None:
    """Upper-bound decode tok/s if cold experts must cross NVMe every token.

    Returns None when nothing is placed on disk, so RAM/VRAM-resident plans are
    not falsely treated as I/O-bound.
    """
    if nvme_weight_bytes <= 0 or storage_bandwidth_mbps <= 0:
        return None
    if n_expert and n_expert_used and n_expert > 0:
        per_token = nvme_weight_bytes * n_expert_used / n_expert
    else:
        per_token = nvme_weight_bytes * 0.02
    if per_token <= 0:
        return None
    return (storage_bandwidth_mbps * 1_000_000.0) / per_token
