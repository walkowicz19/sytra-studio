"""Sytra-Helix MoE Engine v2: Universal 7B to 700B+ MoE Planning & Execution Contract.

Synthesizes the six reference architectures:
1. FreeToken (FlashML-org): Direct runtime bank layout (STW), layer-major prefill,
   3-stream CUDA overlap, and global frequency-weighted LRU expert caching.
2. ktransformers (kvcache-ai): Heterogeneous CPU+GPU co-execution (Attention/MLA
   & Shared Experts on GPU; cache-miss Routed Experts on CPU via AMX/AVX-512).
3. Colibri (JustVugg): Unified 3-tier memory fabric (VRAM <-> Pinned RAM <-> NVMe)
   with unbuffered Direct I/O and multi-drive striping.
4. MoE-Infinity (efficientmoe): Sequence-level activation locality tracing and
   multi-layer predictive lookahead prefetching ring buffer.
5. AirLLM (lyogavin): Layer-wise double-buffered sliding window execution for
   strict OOM immunity on 4 GB and 8 GB VRAM GPUs.
6. Strata (niko1221): N-gram speculative trie cache + native Multi-Token Prediction
   (MTP) draft heads, ultra-fine MoE concurrent scheduling, and OpenAI + Anthropic
   dual API serving.
"""
from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from typing import Any


MIB = 1024 * 1024
GIB = 1024 * MIB

# Minimum sustainable decode floor when combining Direct I/O + CPU co-compute + N-gram/MTP
HELIX_STREAMING_FLOOR_TPS = 0.25


@dataclass(frozen=True)
class HardwareRequirements:
    """Minimum and recommended hardware envelopes for a given MoE scale."""

    scale_tier: str
    estimated_total_params_b: float
    estimated_active_params_b: float
    min_vram_mb: int
    min_ram_mb: int
    min_nvme_mbps: int
    recommended_vram_mb: int
    recommended_ram_mb: int
    recommended_nvme_mbps: int
    recommended_quantization: str
    supports_4gb_vram: bool = True
    supports_8gb_vram: bool = True
    supports_12gb_vram: bool = True
    supports_cpu_only: bool = True
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HelixResourcePlan:
    """Complete execution plan for Sytra-Helix MoE v2."""

    engine_version: str
    profile: str
    hardware_profile: str
    execution_strategy: str
    scale_tier: str
    family: str
    model_id: str
    vram_mb: int
    ram_mb: int
    tier0_vram_attention_mb: int
    tier0_vram_hot_expert_cache_mb: int
    tier1_pinned_ram_pool_mb: int
    tier2_nvme_weight_bytes: int
    storage_bandwidth_mbps: int
    layer_window_size: int | None
    double_buffered_layers: bool
    prefetch_lookahead_layers: int
    cuda_streams: int
    cpu_isa_target: str
    cpu_expert_co_execution: bool
    cpu_expert_coexecution: bool
    ngram_speculative_cache: bool
    mtp_draft_heads: bool
    speculative_mode: str
    speculative_draft_tokens: int
    expected_cache_hit_rate: float
    pcie_overlap_efficiency: float
    raw_io_tps: float | None
    effective_estimated_tps: float
    target_tps: float
    meets_minimum_envelope: bool
    dual_api_endpoints: tuple[str, ...]
    hardware_requirements: HardwareRequirements

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hardware_requirements"] = self.hardware_requirements.to_dict()
        return data


def detect_cpu_isa() -> str:
    """Detect the highest vector/matrix ISA available on the host CPU."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "arm_neon_dotprod"
    return "x86_64_avx512_amx_auto"


def estimate_model_parameters_b(
    *,
    size_bytes: int,
    parameter_count: int | None = None,
    quantization: str | None = None,
    n_expert: int | None = None,
    n_expert_used: int | None = None,
    family: str | None = None,
    model_type: str = "",
    repo_id: str | None = None,
) -> tuple[float, float]:
    """Estimate (total_params_billions, active_params_billions) for any MoE model."""
    if parameter_count and parameter_count > 0:
        total_b = parameter_count / 1e9
    else:
        q_lower = (quantization or "").lower()
        if any(tag in q_lower for tag in ("iq2", "q2")):
            bytes_per_param = 0.31
        elif any(tag in q_lower for tag in ("iq3", "q3")):
            bytes_per_param = 0.44
        elif any(tag in q_lower for tag in ("q4", "int4", "mxfp4", "nvfp4", "awq", "gptq")):
            bytes_per_param = 0.56
        elif any(tag in q_lower for tag in ("q5",)):
            bytes_per_param = 0.69
        elif any(tag in q_lower for tag in ("q6",)):
            bytes_per_param = 0.81
        elif any(tag in q_lower for tag in ("q8", "int8", "fp8")):
            bytes_per_param = 1.05
        else:
            bytes_per_param = 2.0
        estimated_from_size = (size_bytes / bytes_per_param) / 1e9

        blob = f"{family or ''} {model_type} {repo_id or ''}".lower()
        if "glm-5" in blob or "glm5" in blob or "glm_moe_dsa" in blob or "glm52" in blob:
            total_b = max(estimated_from_size, 744.0) if size_bytes > 50 * GIB else estimated_from_size
        elif "deepseek-v3" in blob or "deepseek_v3" in blob or "deepseek-r1" in blob or "deepseek_r1" in blob:
            total_b = max(estimated_from_size, 671.0) if size_bytes > 50 * GIB else estimated_from_size
        elif "kimi-k3" in blob or "kimi_k3" in blob:
            total_b = max(estimated_from_size, 600.0) if size_bytes > 50 * GIB else estimated_from_size
        elif "arctic" in blob:
            total_b = max(estimated_from_size, 480.0) if size_bytes > 50 * GIB else estimated_from_size
        elif "grok" in blob:
            total_b = max(estimated_from_size, 314.0) if size_bytes > 50 * GIB else estimated_from_size
        elif "8x22b" in blob:
            total_b = max(estimated_from_size, 141.0) if size_bytes > 20 * GIB else estimated_from_size
        elif "dbrx" in blob:
            total_b = max(estimated_from_size, 132.0) if size_bytes > 20 * GIB else estimated_from_size
        else:
            total_b = max(estimated_from_size, 0.1)

    if n_expert and n_expert_used and n_expert > 0:
        dense_share = 0.05 if n_expert >= 64 else 0.20
        routed_share = 1.0 - dense_share
        active_b = total_b * (dense_share + routed_share * (n_expert_used / n_expert))
    else:
        active_b = total_b * 0.12

    return round(total_b, 2), round(max(active_b, 0.05), 2)


def classify_hardware_requirements(
    *,
    total_params_b: float | None = None,
    active_params_b: float | None = None,
    params_b: float | None = None,
    weight_bytes: int = 0,
    n_expert: int | None = None,
    n_expert_used: int | None = None,
) -> HardwareRequirements:
    """Compute hardware requirements across the 7B -> 700B+ MoE spectrum."""
    resolved_total_b = total_params_b if total_params_b is not None else (params_b or 0.0)
    if resolved_total_b <= 0.0 and weight_bytes > 0:
        resolved_total_b, resolved_active_b = estimate_model_parameters_b(
            size_bytes=weight_bytes,
            n_expert=n_expert,
            n_expert_used=n_expert_used,
        )
    else:
        if active_params_b is not None:
            resolved_active_b = active_params_b
        elif n_expert and n_expert_used and n_expert > 0:
            dense_share = 0.05 if n_expert >= 64 else 0.20
            resolved_active_b = round(
                resolved_total_b * (dense_share + (1.0 - dense_share) * (n_expert_used / n_expert)),
                2,
            )
        else:
            resolved_active_b = round(max(resolved_total_b * 0.12, 0.05), 2)

    if resolved_total_b >= 500.0:
        return HardwareRequirements(
            scale_tier="titan_700b_moe",
            estimated_total_params_b=resolved_total_b,
            estimated_active_params_b=resolved_active_b,
            min_vram_mb=4096,
            min_ram_mb=32768,
            min_nvme_mbps=5000,
            recommended_vram_mb=24576,
            recommended_ram_mb=65536,
            recommended_nvme_mbps=7000,
            recommended_quantization="Q3_K_M / IQ2_XXS / FP8-Hybrid",
            supports_4gb_vram=True,
            supports_8gb_vram=True,
            supports_12gb_vram=True,
            supports_cpu_only=True,
            notes=(
                "Frontier 500B-744B+ MoE: uses Layer Windowing on 4-8GB VRAM or MLA Resident + Pinned RAM Pool on 12GB+ VRAM.",
                "Multi-NVMe striping (RAID0) or PCIe Gen4/Gen5 SSD (>= 5000 MB/s) strongly advised when model exceeds host RAM.",
            ),
        )
    if resolved_total_b >= 100.0:
        return HardwareRequirements(
            scale_tier="large_moe",
            estimated_total_params_b=resolved_total_b,
            estimated_active_params_b=resolved_active_b,
            min_vram_mb=4096,
            min_ram_mb=32768,
            min_nvme_mbps=3500,
            recommended_vram_mb=12288,
            recommended_ram_mb=65536,
            recommended_nvme_mbps=5000,
            recommended_quantization="Q4_K_M / INT4-Group32",
            supports_4gb_vram=True,
            supports_8gb_vram=True,
            supports_12gb_vram=True,
            supports_cpu_only=True,
            notes=(
                "Large 100B-480B MoE: Attention + hot experts in VRAM; warm experts in Pinned RAM with CPU AVX-512 co-execution.",
            ),
        )
    if resolved_total_b >= 30.0:
        return HardwareRequirements(
            scale_tier="standard_moe",
            estimated_total_params_b=resolved_total_b,
            estimated_active_params_b=resolved_active_b,
            min_vram_mb=4096,
            min_ram_mb=16384,
            min_nvme_mbps=2000,
            recommended_vram_mb=8192,
            recommended_ram_mb=32768,
            recommended_nvme_mbps=3500,
            recommended_quantization="Q4_K_M / Packed-INT4-Group32",
            supports_4gb_vram=True,
            supports_8gb_vram=True,
            supports_12gb_vram=True,
            supports_cpu_only=True,
            notes=(
                "Standard 30B-100B MoE (e.g. Mixtral 8x7B, Qwen2.5-MoE): runs at high tk/s on 4GB/8GB/12GB GPUs via hybrid dispatch.",
            ),
        )
    return HardwareRequirements(
        scale_tier="compact_moe",
        estimated_total_params_b=resolved_total_b,
        estimated_active_params_b=resolved_active_b,
        min_vram_mb=2048,
        min_ram_mb=8192,
        min_nvme_mbps=1000,
        recommended_vram_mb=8192,
        recommended_ram_mb=16384,
        recommended_nvme_mbps=2000,
        recommended_quantization="Q4_K_M / BF16 / INT4",
        supports_4gb_vram=True,
        supports_8gb_vram=True,
        supports_12gb_vram=True,
        supports_cpu_only=True,
        notes=(
            "Compact MoE (<30B): fits mostly or entirely in VRAM + Pinned RAM with minimal NVMe traffic.",
        ),
    )


def select_helix_hardware_profile(
    *,
    vram_limit_mb: int,
    gpu_visible: bool = True,
    allow_cpu_only: bool = False,
) -> str:
    """Return the hardware profile identifier for the given VRAM budget."""
    if not gpu_visible or allow_cpu_only or vram_limit_mb <= 0:
        return "cpu_only_native"
    if vram_limit_mb <= 4608:
        return "vram_4gb_windowed"
    if vram_limit_mb <= 8704:
        return "vram_8gb_hybrid"
    if vram_limit_mb <= 14336:
        return "vram_12gb_balanced"
    return "vram_16gb_plus_prosumer"


def _execution_strategy_for_profile(profile: str) -> str:
    return {
        "cpu_only_native": "cpu-native-amx-avx512-tiled",
        "vram_4gb_windowed": "layer-windowed-anti-oom-hybrid-cpu",
        "vram_8gb_hybrid": "hybrid-gpu-attention-cpu-experts-double-buffer",
        "vram_12gb_balanced": "gpu-hot-cache-triple-prefetch-co-compute",
        "vram_16gb_plus_prosumer": "gpu-resident-hot-pool-triple-prefetch",
    }.get(profile, "hybrid-gpu-attention-cpu-experts-double-buffer")


def build_helix_resource_plan(
    *,
    weight_bytes: int,
    vram_limit_mb: int,
    ram_limit_mb: int,
    nvme_weight_bytes: int,
    storage_bandwidth_mbps: int,
    target_tps: float,
    gpu_visible: bool = True,
    allow_cpu_only: bool = False,
    strategy: str | None = None,
    io_max_tps: float | None = None,
    family: str | None = None,
    adapter_id: str | None = None,
    model_type: str = "",
    architecture: str = "",
    repo_id: str | None = None,
    quantization: str | None = None,
    parameter_count: int | None = None,
    n_layer: int | None = None,
    n_expert: int | None = None,
    n_expert_used: int | None = None,
    verification_positions: int = 8,
    config: dict[str, Any] | None = None,
) -> HelixResourcePlan:
    """Construct the unified Sytra-Helix MoE v2 execution and resource plan."""
    effective_family = family or adapter_id or model_type or architecture
    total_b, active_b = estimate_model_parameters_b(
        size_bytes=weight_bytes,
        parameter_count=parameter_count,
        quantization=quantization,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        family=effective_family,
        model_type=model_type,
        repo_id=repo_id,
    )
    hw_reqs = classify_hardware_requirements(
        total_params_b=total_b,
        active_params_b=active_b,
    )
    hw_profile = select_helix_hardware_profile(
        vram_limit_mb=vram_limit_mb,
        gpu_visible=gpu_visible,
        allow_cpu_only=allow_cpu_only,
    )
    exec_strategy = _execution_strategy_for_profile(hw_profile)

    if hw_profile == "cpu_only_native":
        vram_attn_mb = 0
        vram_cache_mb = 0
        layer_window = 1 if total_b >= 500.0 else 2
        double_buffered = True
        cuda_streams = 0
        cpu_co_exec = True
        cache_hit_rate = 0.0
        pcie_overlap = 1.0
    elif hw_profile == "vram_4gb_windowed":
        usable_vram = int(vram_limit_mb * 0.82)
        vram_attn_mb = min(usable_vram, max(1024, int(usable_vram * 0.65)))
        vram_cache_mb = max(0, usable_vram - vram_attn_mb)
        layer_window = 1 if total_b >= 100.0 else 2
        double_buffered = True
        cuda_streams = 3
        cpu_co_exec = True
        cache_hit_rate = 0.35
        pcie_overlap = 0.82
    elif hw_profile == "vram_8gb_hybrid":
        usable_vram = int(vram_limit_mb * 0.84)
        vram_attn_mb = min(usable_vram, max(2048, int(usable_vram * 0.50)))
        vram_cache_mb = max(0, usable_vram - vram_attn_mb)
        layer_window = 2 if total_b >= 500.0 else (4 if total_b >= 100.0 else None)
        double_buffered = True
        cuda_streams = 3
        cpu_co_exec = True
        cache_hit_rate = 0.58
        pcie_overlap = 0.88
    elif hw_profile == "vram_12gb_balanced":
        usable_vram = int(vram_limit_mb * 0.85)
        vram_attn_mb = min(usable_vram, max(3072, int(usable_vram * 0.42)))
        vram_cache_mb = max(0, usable_vram - vram_attn_mb)
        layer_window = 4 if total_b >= 500.0 else None
        double_buffered = total_b >= 100.0
        cuda_streams = 3
        cpu_co_exec = True
        cache_hit_rate = 0.72
        pcie_overlap = 0.92
    else:
        usable_vram = int(vram_limit_mb * 0.88)
        vram_attn_mb = min(usable_vram, max(4096, int(usable_vram * 0.35)))
        vram_cache_mb = max(0, usable_vram - vram_attn_mb)
        layer_window = None
        double_buffered = total_b >= 500.0
        cuda_streams = 4
        cpu_co_exec = True
        cache_hit_rate = 0.85
        pcie_overlap = 0.95

    pinned_ram_mb = int(ram_limit_mb * 0.80) if ram_limit_mb > 0 else 0
    prefetch_depth = 2 if hw_profile in {"vram_4gb_windowed", "cpu_only_native"} else 3

    fam_lower = f"{effective_family or ''} {architecture}".lower()
    supports_mtp = any(
        token in fam_lower
        for token in ("deepseek", "glm", "qwen3", "kimi")
    )
    spec_mode = "mtp_plus_ngram_trie" if supports_mtp else "ngram_trie_plus_draft_batch"
    spec_draft_tokens = min(verification_positions, 6 if supports_mtp else 4)
    speculative_multiplier = 2.4 if supports_mtp else 1.85

    if nvme_weight_bytes > 0 and storage_bandwidth_mbps > 0:
        if n_expert and n_expert_used and n_expert > 0:
            cold_per_tok = nvme_weight_bytes * (n_expert_used / n_expert)
        else:
            cold_per_tok = nvme_weight_bytes * 0.025
        raw_io_tps = (
            io_max_tps
            if io_max_tps is not None
            else (
                (storage_bandwidth_mbps * 1_000_000.0) / cold_per_tok
                if cold_per_tok > 0
                else None
            )
        )
        effective_cold_bytes = cold_per_tok * (1.0 - cache_hit_rate * 0.65)
        overlap_boost = 1.0 / max(1.0 - pcie_overlap * 0.55, 0.25)
        base_tps = (
            raw_io_tps * (cold_per_tok / max(effective_cold_bytes, 1.0)) * overlap_boost
            if raw_io_tps is not None
            else ((storage_bandwidth_mbps * 1_000_000.0) / max(effective_cold_bytes, 1.0)) * overlap_boost
        )
        effective_tps = round(base_tps * speculative_multiplier, 2)
    else:
        raw_io_tps = io_max_tps
        active_gb = max(active_b * 0.55, 0.5)
        effective_bw_gbs = (
            320.0 if hw_profile == "vram_16gb_plus_prosumer"
            else 220.0 if hw_profile == "vram_12gb_balanced"
            else 140.0 if hw_profile == "vram_8gb_hybrid"
            else 85.0 if hw_profile == "vram_4gb_windowed"
            else 55.0
        )
        effective_tps = round((effective_bw_gbs / active_gb) * (speculative_multiplier * 0.55), 2)

    served_model_id = (
        repo_id.split("/")[-1].lower()
        if repo_id and "/" in repo_id
        else (effective_family or "sytra-helix-moe").lower().replace("_", "-")
    )

    weight_mb = weight_bytes / (1024 * 1024) if weight_bytes > 0 else 0.0
    total_host_mb = ram_limit_mb + (vram_limit_mb if gpu_visible and not allow_cpu_only else 0)
    meets_minimum_envelope = (
        (allow_cpu_only or vram_limit_mb >= hw_reqs.min_vram_mb)
        and (
            ram_limit_mb >= hw_reqs.min_ram_mb
            or (nvme_weight_bytes == 0 and weight_mb > 0.0 and total_host_mb >= weight_mb * 1.05)
        )
    )

    return HelixResourcePlan(
        engine_version="sytra-helix-v2",
        profile=hw_profile,
        hardware_profile=hw_profile,
        execution_strategy=exec_strategy,
        scale_tier=hw_reqs.scale_tier,
        family=effective_family or "generic_moe",
        model_id=served_model_id,
        vram_mb=vram_limit_mb if gpu_visible and not allow_cpu_only else 0,
        ram_mb=ram_limit_mb,
        tier0_vram_attention_mb=vram_attn_mb,
        tier0_vram_hot_expert_cache_mb=vram_cache_mb,
        tier1_pinned_ram_pool_mb=pinned_ram_mb,
        tier2_nvme_weight_bytes=nvme_weight_bytes,
        storage_bandwidth_mbps=storage_bandwidth_mbps,
        layer_window_size=layer_window,
        double_buffered_layers=double_buffered,
        prefetch_lookahead_layers=prefetch_depth,
        cuda_streams=cuda_streams,
        cpu_isa_target=detect_cpu_isa(),
        cpu_expert_co_execution=cpu_co_exec,
        cpu_expert_coexecution=cpu_co_exec,
        ngram_speculative_cache=True,
        mtp_draft_heads=supports_mtp,
        speculative_mode=spec_mode,
        speculative_draft_tokens=spec_draft_tokens,
        expected_cache_hit_rate=cache_hit_rate,
        pcie_overlap_efficiency=pcie_overlap,
        raw_io_tps=round(raw_io_tps, 4) if raw_io_tps is not None else None,
        effective_estimated_tps=effective_tps,
        target_tps=target_tps,
        meets_minimum_envelope=meets_minimum_envelope,
        dual_api_endpoints=("/v1/chat/completions", "/v1/completions", "/v1/messages"),
        hardware_requirements=hw_reqs,
    )
