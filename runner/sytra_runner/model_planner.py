"""Capability-gated local inference backend planning.

Sytra verifies the local artifact and launches an OpenAI- and Anthropic-compatible
inference server:

* llama.cpp for GGUF and CPU/GPU hybrid inference.
* vLLM for complete Hugging Face/Safetensors checkpoints that fit GPU memory
  or a conservative GPU+RAM UVA-offload budget.
* Sytra-Helix MoE Engine v2 (`sytra_moe` / `sytra-engine`) for explicitly indexed
  or architecture-verified MoE checkpoints across 4GB, 8GB, 12GB, 16GB+ VRAM and
  CPU-only hosts, combining FreeToken (3-stream STW + frequency LRU),
  ktransformers (GPU Attention + CPU AVX-512/AMX expert co-execution),
  Colibri-tier Direct NVMe streaming, MoE-Infinity (predictive lookahead),
  AirLLM (layer windowing anti-OOM), and Strata (N-gram trie + MTP speculation)
  for models from 7B to 744B+ parameters.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .architecture_adapters import (
    AdapterCompatibilityError,
    ArchitectureAdapter,
    adapter_by_id,
    resolve_architecture_adapter,
    validate_adapter_payload,
)
from .gguf_meta import read_gguf_layer_sizes, try_read_gguf_metadata
from .helix_engine import (
    HELIX_STREAMING_FLOOR_TPS,
    build_helix_resource_plan,
)
from .subprocess_utils import silent_subprocess_kwargs
from .memory_hierarchy import (
    KVCachePlan,
    LlamaCppOffloadPlan,
    WeightPlacementPlan,
    estimate_kv_cache,
    estimate_streamed_moe_tps,
    plan_llama_cpp_offload,
    plan_weight_placement,
)
from .runtime_detect import (
    find_llama_server,
    find_sytra_engine,
    llama_server_provision_hint,
    prepend_runtime_path,
)

BackendName = Literal["llama_cpp", "vllm", "sytra_moe"]

_SUPPORTED_KV_QUANTS = {
    "fp16", "f16", "float16", "bf16", "bfloat16",
    "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl",
    "f32", "fp32", "float32", "auto", "fp8", "float8",
}
_QUANTIZED_KV_QUANTS = {"q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
_LLAMA_KV_TYPE_MAP = {
    "fp16": "f16",
    "float16": "f16",
    "fp32": "f32",
    "float32": "f32",
    "bfloat16": "bf16",
}


@dataclass(frozen=True)
class ModelArtifact:
    requested_path: str
    model_path: str
    format: Literal["gguf", "safetensors", "sytra_moe"]
    architecture: str
    model_type: str
    size_bytes: int
    is_moe: bool
    resolved_revision: str | None
    mmproj_path: str | None
    repo_id: str | None
    adapter_id: str | None
    quantization: str | None = None
    n_layer: int | None = None
    n_expert: int | None = None
    n_expert_used: int | None = None
    parameter_count: int | None = None


@dataclass(frozen=True)
class BackendPlan:
    compatible: bool
    backend: BackendName | None
    artifact: ModelArtifact
    command: list[str]
    reasons: list[str]
    warnings: list[str]
    required_vram_mb: int | None
    available_vram_mb: int
    available_ram_mb: int
    kv_cache: KVCachePlan
    weight_placement: WeightPlacementPlan
    preflight_command: list[str]
    llama_offload: LlamaCppOffloadPlan | None = None
    runtime_version: str | None = None
    estimates: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelCompatibilityError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"Could not read valid JSON from {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _manifest_metadata(directory: Path) -> dict:
    path = directory / ".sytra-model.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _validate_safetensors_file(path: Path) -> None:
    size = path.stat().st_size
    if size < 12:
        raise ModelCompatibilityError(f"Safetensors file is too small to be valid: {path.name}")
    try:
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), byteorder="little", signed=False)
            if header_size <= 0 or header_size > min(size - 8, 100 * 1024 * 1024):
                raise ModelCompatibilityError(f"Safetensors file has an invalid header length: {path.name}")
            header = json.loads(handle.read(header_size).decode("utf-8").rstrip())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelCompatibilityError(f"Safetensors header is invalid in {path.name}: {exc}") from exc
    tensor_entries = [value for key, value in header.items() if key != "__metadata__"]
    if not tensor_entries or not all(
        isinstance(value, dict)
        and "dtype" in value
        and "shape" in value
        and "data_offsets" in value
        for value in tensor_entries
    ):
        raise ModelCompatibilityError(f"Safetensors file contains no valid tensor metadata: {path.name}")
    data_size = size - 8 - header_size
    for value in tensor_entries:
        offsets = value["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            raise ModelCompatibilityError(f"Safetensors tensor offsets are invalid in {path.name}")


def _safetensors_quantization(config: dict) -> str | None:
    sources = [config]
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(config.get(key), dict):
            sources.append(config[key])
    for source in sources:
        quant = source.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        groups = quant.get("config_groups")
        if isinstance(groups, dict) and groups:
            group = groups.get("group_0") or groups.get("group0") or next(iter(groups.values()), None)
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict):
                bits = weights.get("num_bits")
                gs = weights.get("group_size")
                wtype = str(weights.get("type") or "int").upper()
                if bits and gs:
                    return f"{wtype}{bits}-G{gs}"
                if bits:
                    return f"{wtype}{bits}"
        method = str(quant.get("quant_method") or quant.get("format") or "").strip()
        if method:
            return method.upper()
    dtype = str(config.get("dtype") or config.get("torch_dtype") or "").strip()
    return dtype.upper() if dtype else None


def _estimate_safetensors_param_count(config: dict) -> int | None:
    src = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    hidden = src.get("hidden_size")
    layers = src.get("num_hidden_layers") or src.get("n_layers")
    vocab = src.get("vocab_size")
    if not (isinstance(hidden, int) and hidden > 0 and isinstance(layers, int) and layers > 0 and isinstance(vocab, int) and vocab > 0):
        return None
    heads = src.get("num_attention_heads") or 1
    kv_heads = src.get("num_key_value_heads") or heads
    head_dim = src.get("head_dim") or (hidden // heads)
    attn_per_layer = (
        hidden * (heads * head_dim)
        + hidden * (kv_heads * head_dim) * 2
        + (heads * head_dim) * hidden
    )
    n_expert = (
        src.get("num_local_experts")
        or src.get("num_experts")
        or src.get("n_routed_experts")
        or 0
    )
    moe_inter = src.get("moe_intermediate_size") or src.get("intermediate_size") or 0
    dense_inter = src.get("intermediate_size") or moe_inter
    mlp_only = set(src.get("mlp_only_layers") or ())
    sparse_step = src.get("decoder_sparse_step") or 1
    shared_experts = src.get("num_shared_experts") or src.get("n_shared_experts") or 0
    shared_inter = src.get("shared_expert_intermediate_size") or moe_inter
    total_mlp = 0
    for layer in range(layers):
        if n_expert and layer not in mlp_only and (layer + 1) % sparse_step == 0:
            total_mlp += n_expert * 3 * hidden * moe_inter
            if shared_experts:
                total_mlp += shared_experts * 3 * hidden * shared_inter
        elif dense_inter:
            total_mlp += 3 * hidden * dense_inter
    embed = vocab * hidden * (1 if src.get("tie_word_embeddings") else 2)
    return int(embed + layers * attn_per_layer + total_mlp)


def inspect_model(model_path: str | Path) -> ModelArtifact:
    requested = Path(model_path).expanduser()
    if not requested.exists():
        raise ModelCompatibilityError(f"Model path does not exist: {requested}")
    requested = requested.resolve()

    if requested.is_file():
        if requested.suffix.lower() != ".gguf":
            raise ModelCompatibilityError(
                "A standalone model file must be GGUF. For Safetensors, select the complete model directory."
            )
        if requested.stat().st_size <= 0:
            raise ModelCompatibilityError(f"GGUF file is empty: {requested}")
        with requested.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                raise ModelCompatibilityError(f"File does not contain a GGUF header: {requested}")
        projectors = sorted(
            path
            for path in requested.parent.glob("*.gguf")
            if path != requested
            and ("mmproj" in path.name.lower() or "projector" in path.name.lower())
        )
        mmproj = next(
            (
                path
                for marker in ("F16", "BF16", "Q8_0")
                for path in projectors
                if marker in path.name.upper()
            ),
            projectors[0] if projectors else None,
        )
        if mmproj is not None:
            with mmproj.open("rb") as handle:
                if handle.read(4) != b"GGUF":
                    raise ModelCompatibilityError(f"Multimodal projector has no GGUF header: {mmproj}")
        manifest = _manifest_metadata(requested.parent)
        try:
            adapter = resolve_architecture_adapter(
                requested,
                config={},
                manifest=manifest,
                file_name=requested.name,
            )
        except AdapterCompatibilityError as exc:
            raise ModelCompatibilityError(str(exc)) from exc
        meta = try_read_gguf_metadata(requested)
        architecture = meta.architecture if meta else "gguf"
        name_lower = requested.name.lower()
        arch_lower = architecture.lower().replace("-", "_")
        if arch_lower in {"qwen2", "qwen2_moe"} and (
            "qwen3" in name_lower or "qwen3.5" in name_lower or "qwen3_5" in name_lower
        ):
            raise ModelCompatibilityError(
                "GGUF metadata names this file as qwen2 but the filename indicates Qwen3/Qwen3.5; "
                "refusing to treat Qwen3.5 as Qwen2. Re-convert with a llama.cpp that writes the "
                "correct architecture keys."
            )
        return ModelArtifact(
            requested_path=str(requested),
            model_path=str(requested),
            format="gguf",
            architecture=architecture,
            model_type=architecture,
            size_bytes=requested.stat().st_size,
            is_moe=bool(meta and meta.is_moe),
            resolved_revision=manifest.get("resolved_revision"),
            mmproj_path=str(mmproj.resolve()) if mmproj else None,
            repo_id=manifest.get("repo_id"),
            adapter_id=adapter.id if adapter else None,
            quantization=meta.quantization if meta else None,
            n_layer=meta.n_layer if meta else None,
            n_expert=meta.n_expert if meta else None,
            n_expert_used=meta.n_expert_used if meta else None,
            parameter_count=meta.parameter_count if meta else None,
        )

    config_path = requested / "config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    manifest = _manifest_metadata(requested)
    try:
        adapter = resolve_architecture_adapter(
            requested,
            config=config,
            manifest=manifest,
        )
    except AdapterCompatibilityError as exc:
        raise ModelCompatibilityError(str(exc)) from exc

    all_ggufs = [
        path
        for path in requested.rglob("*.gguf")
        if ".cache" not in path.parts and not path.name.startswith(".")
    ]
    ggufs = [
        path
        for path in all_ggufs
        if "mmproj" not in path.name.lower() and "projector" not in path.name.lower()
    ]
    tensor_files = [
        path
        for path in requested.rglob("*.safetensors")
        if ".cache" not in path.parts and not path.name.startswith(".")
    ]

    if ggufs and tensor_files:
        raise ModelCompatibilityError(
            "Directory contains both GGUF and Safetensors weights. Select the exact GGUF file "
            "or a directory containing only one checkpoint format."
        )
    if ggufs:
        if len(ggufs) != 1:
            raise ModelCompatibilityError(
                f"Directory contains {len(ggufs)} GGUF files. Select the exact model file to avoid loading the wrong quantization."
            )
        return inspect_model(ggufs[0])
    if not tensor_files:
        if adapter is None:
            raise ModelCompatibilityError(
                f"No GGUF, Safetensors, or recognized streaming-adapter weights found under {requested}"
            )
        try:
            size_bytes, _ = validate_adapter_payload(requested, adapter)
        except AdapterCompatibilityError as exc:
            raise ModelCompatibilityError(str(exc)) from exc
        architectures = config.get("architectures") or []
        architecture = (
            architectures[0]
            if isinstance(architectures, list) and architectures
            else str(config.get("model_type") or adapter.engine_arch)
        )
        n_layer = int(config.get("num_hidden_layers") or config.get("n_layers") or 0) or None
        n_expert = int(
            config.get("num_local_experts")
            or config.get("num_experts")
            or config.get("n_routed_experts")
            or config.get("moe_num_experts")
            or 0
        ) or None
        n_expert_used = int(
            config.get("num_experts_per_tok")
            or config.get("num_experts_per_token")
            or config.get("num_selected_experts")
            or config.get("top_k")
            or config.get("moe_top_k")
            or 0
        ) or None
        safetensors_quant = _safetensors_quantization(config)
        safetensors_params = _estimate_safetensors_param_count(config)
        return ModelArtifact(
            requested_path=str(requested),
            model_path=str(requested),
            format="sytra_moe",
            architecture=str(architecture),
            model_type=str(config.get("model_type") or adapter.engine_arch),
            size_bytes=size_bytes,
            is_moe=True,
            resolved_revision=manifest.get("resolved_revision"),
            mmproj_path=None,
            repo_id=manifest.get("repo_id"),
            adapter_id=adapter.id,
            quantization=safetensors_quant,
            n_layer=n_layer,
            n_expert=n_expert,
            n_expert_used=n_expert_used,
            parameter_count=safetensors_params,
        )

    if not config_path.exists():
        raise ModelCompatibilityError(
            f"Safetensors checkpoint is missing config.json at its model root: {requested}"
        )
    config = _load_json(config_path)
    architectures = config.get("architectures") or []
    architecture = architectures[0] if architectures else str(config.get("model_type") or "unknown")
    model_type = str(config.get("model_type") or "unknown")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    n_expert = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or config.get("moe_num_experts")
        or text_config.get("num_local_experts")
        or text_config.get("num_experts")
        or text_config.get("n_routed_experts")
        or text_config.get("moe_num_experts")
        or 0
    ) or None
    n_expert_used = int(
        config.get("num_experts_per_tok")
        or config.get("num_experts_per_token")
        or config.get("num_selected_experts")
        or config.get("top_k")
        or config.get("moe_top_k")
        or text_config.get("num_experts_per_tok")
        or text_config.get("num_experts_per_token")
        or text_config.get("num_selected_experts")
        or text_config.get("top_k")
        or text_config.get("moe_top_k")
        or 0
    ) or None
    n_layer = int(
        config.get("num_hidden_layers")
        or config.get("n_layers")
        or text_config.get("num_hidden_layers")
        or 0
    ) or None
    is_moe = n_expert is not None and n_expert > 0
    size_bytes = sum(path.stat().st_size for path in tensor_files)
    if size_bytes <= 0:
        raise ModelCompatibilityError(f"Safetensors checkpoint contains no weight data: {requested}")
    for tensor_file in tensor_files:
        _validate_safetensors_file(tensor_file)

    index_path = requested / "model.safetensors.index.json"
    if len(tensor_files) > 1 and not index_path.exists():
        raise ModelCompatibilityError(
            "Sharded Safetensors checkpoint is missing model.safetensors.index.json; completeness cannot be verified."
        )
    if index_path.exists():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelCompatibilityError("Safetensors index has no valid weight_map")
        expected = {str(name) for name in weight_map.values()}
        missing = sorted(name for name in expected if not (requested / name).is_file())
        if missing:
            preview = ", ".join(missing[:3])
            raise ModelCompatibilityError(
                f"Safetensors checkpoint is incomplete; {len(missing)} indexed shards are missing: {preview}"
            )

    safetensors_quant = _safetensors_quantization(config)
    safetensors_params = _estimate_safetensors_param_count(config)
    return ModelArtifact(
        requested_path=str(requested),
        model_path=str(requested),
        format="sytra_moe" if adapter is not None else "safetensors",
        architecture=str(architecture),
        model_type=model_type,
        size_bytes=size_bytes,
        is_moe=is_moe,
        resolved_revision=manifest.get("resolved_revision"),
        mmproj_path=None,
        repo_id=manifest.get("repo_id"),
        adapter_id=adapter.id if adapter else None,
        quantization=safetensors_quant,
        n_layer=n_layer,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        parameter_count=safetensors_params,
    )


def _command_override(env_name: str) -> list[str] | None:
    value = os.environ.get(env_name)
    return shlex.split(value, posix=os.name != "nt") if value else None


def _find_llama_server(project_root: Path | None = None) -> list[str] | None:
    return find_llama_server(project_root)


def _llama_server_help(launcher: list[str]) -> str:
    try:
        result = subprocess.run(
            launcher + ["-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            **silent_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return f"{result.stdout}\n{result.stderr}"


def _llama_server_version(launcher: list[str]) -> str | None:
    try:
        result = subprocess.run(
            launcher + ["--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            **silent_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return text[0][:200] if text else None


def _help_has(help_text: str, *needles: str) -> bool:
    lowered = help_text.lower()
    return any(re.search(r"(?<![\w-])" + re.escape(needle.lower()) + r"(?![\w-])", lowered) for needle in needles)


def _find_vllm() -> list[str] | None:
    override = _command_override("SYTRA_VLLM_COMMAND")
    if override:
        return override
    executable = shutil.which("vllm") or shutil.which("vllm.exe")
    if executable:
        return [executable]
    if importlib.util.find_spec("vllm") is not None:
        return [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    return None


def _find_sytra_engine(project_root: Path | None = None) -> list[str] | None:
    return find_sytra_engine(project_root)


def _system_memory_mb() -> int:
    override = os.environ.get("SYTRA_SYSTEM_MEMORY_MB")
    if override:
        try:
            return max(int(override), 0)
        except ValueError:
            return 0
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical // (1024 * 1024))
        except (AttributeError, OSError, ValueError):
            return 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size // (1024 * 1024))
    except (AttributeError, OSError, ValueError):
        return 0


def _visible_gpu_memory_mb() -> list[int]:
    override = os.environ.get("SYTRA_GPU_MEMORY_MB")
    if override:
        try:
            return [int(value.strip()) for value in override.split(",") if value.strip()]
        except ValueError:
            return []
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not executable:
        return []
    try:
        output = subprocess.run(
            [executable, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            **silent_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if output.returncode:
        return []
    return [int(line.strip()) for line in output.stdout.splitlines() if line.strip().isdigit()]


def _first_visible_gpu_free_mb() -> int | None:
    """Read current free VRAM for the first CUDA-visible NVIDIA GPU."""
    executable = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not executable:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        device = visible.split(",")[0].strip()
        if not device or device == "-1":
            return None
    else:
        device = "0"
    try:
        result = subprocess.run([executable, "--id=" + device, "--query-gpu=memory.free",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=False,
                                **silent_subprocess_kwargs())
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return int(value) if result.returncode == 0 and value.isdigit() else None


def build_backend_plan(
    model_path: str | Path,
    *,
    backend: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8080,
    context: int = 4096,
    context_length: int | None = None,
    verification_positions: int = 8,
    storage_bandwidth_mbps: int = 3500,
    target_tps: float = 5.0,
    draft_url: str | None = None,
    draft_model: str | None = None,
    draft_model_path: str | None = None,
    vram_limit_mb: int | None = 8192,
    ram_limit_mb: int | None = None,
    cpu_kv_cache: bool = False,
    kv_cache_quant: str = "q8_0",
    flash_attention: bool = True,
    project_root: str | Path | None = None,
    force_gpu_layers: int | None = None,
    allow_cpu_only: bool = False,
    prefer_cpu_offload: bool = False,
    prefer_gpu: bool = True,
    moe_expert_offload: str = "auto",
    vram_guardrail_mb: int | None = None,
    allow_unverified: bool = False,
) -> BackendPlan:
    if context_length is not None:
        context = context_length
    if not prefer_gpu:
        allow_cpu_only = True
    if prefer_cpu_offload and moe_expert_offload == "auto":
        moe_expert_offload = "cpu"

    if (
        context <= 0
        or verification_positions <= 0
        or storage_bandwidth_mbps <= 0
        or target_tps <= 0
        or (vram_limit_mb is not None and vram_limit_mb <= 0)
    ):
        raise ModelCompatibilityError(
            "Context, speculative verification positions, storage bandwidth, target TPS, "
            "and VRAM limit must be positive"
        )
    kv_norm = kv_cache_quant.strip().lower()
    if kv_norm not in _SUPPORTED_KV_QUANTS:
        raise ModelCompatibilityError(f"Unsupported KV cache quantization: {kv_cache_quant!r}")
    if moe_expert_offload not in {"auto", "cpu", "off"}:
        raise ModelCompatibilityError("moe_expert_offload must be auto, cpu, or off")
    if force_gpu_layers is not None and force_gpu_layers < 0:
        raise ModelCompatibilityError("GPU layers cannot be negative")
    artifact = inspect_model(model_path)
    reasons: list[str] = []
    warnings: list[str] = []
    requested_backend = backend.strip().lower()
    if requested_backend not in {"auto", "llama_cpp", "vllm", "sytra_moe"}:
        raise ModelCompatibilityError(f"Unknown inference backend: {backend!r}")

    root = Path(project_root).resolve() if project_root else None
    ram_budget_mb = ram_limit_mb if ram_limit_mb is not None else _system_memory_mb()
    if ram_limit_mb is None and ram_budget_mb <= 0:
        raise ModelCompatibilityError("Could not detect system RAM; pass an explicit --ram-limit-mb.")
    if ram_budget_mb < 0:
        raise ModelCompatibilityError("RAM limit cannot be negative")
    config_root = (
        Path(artifact.model_path)
        if Path(artifact.model_path).is_dir()
        else Path(artifact.model_path).parent
    )
    config_path = config_root / "config.json"
    config = _load_json(config_path) if config_path.is_file() else {}
    gguf_meta = (
        try_read_gguf_metadata(artifact.model_path) if artifact.format == "gguf" else None
    )
    if gguf_meta and not config:
        config = gguf_meta.config_for_kv()
    adapter: ArchitectureAdapter | None = (
        adapter_by_id(artifact.adapter_id) if artifact.adapter_id else None
    )

    if artifact.format == "gguf":
        automatic: BackendName = "llama_cpp"
    elif artifact.format == "sytra_moe":
        automatic = "sytra_moe"
    else:
        automatic = "vllm"
    selected: BackendName = automatic if requested_backend == "auto" else requested_backend  # type: ignore[assignment]

    gpu_memory = _visible_gpu_memory_mb()
    gpu_visible = bool(gpu_memory) and prefer_gpu
    if vram_limit_mb is None:
        if gpu_visible:
            vram_limit_mb = gpu_memory[0] if selected == "llama_cpp" else sum(gpu_memory)
        elif not prefer_gpu or allow_cpu_only:
            vram_limit_mb = 1
            if force_gpu_layers is None:
                force_gpu_layers = 0
            warnings.append("No GPU VRAM detected or GPU disabled; planning CPU-only inference.")
        elif selected == "sytra_moe":
            vram_limit_mb = 1
            warnings.append("No GPU VRAM detected; Sytra-Helix MoE plan uses CPU/RAM/NVMe execution.")
        else:
            raise ModelCompatibilityError(
                "Could not detect GPU VRAM; pass an explicit --vram-limit-mb or --no-gpu for CPU-only inference."
            )
    elif gpu_visible:
        visible_limit = gpu_memory[0] if selected == "llama_cpp" else sum(gpu_memory)
        if vram_limit_mb > visible_limit:
            warnings.append(
                f"Requested VRAM limit ({vram_limit_mb} MiB) exceeded detected GPU VRAM ({visible_limit} MiB); capped to {visible_limit} MiB."
            )
            vram_limit_mb = visible_limit

    draft_reserve_mb = 0
    resolved_draft = None
    if draft_model_path:
        if selected != 'llama_cpp':
            raise ModelCompatibilityError('A local draft model currently requires llama.cpp and GGUF')
        draft = Path(draft_model_path).expanduser().resolve()
        if not draft.is_file() or draft.suffix.lower() != '.gguf' or draft == Path(artifact.model_path).resolve():
            raise ModelCompatibilityError('Choose a distinct local GGUF draft model')
        draft_meta = try_read_gguf_metadata(draft)
        if gguf_meta is None or draft_meta is None:
            raise ModelCompatibilityError('Cannot verify target and draft GGUF tokenizers')
        keys = ('tokenizer.ggml.model', 'tokenizer.ggml.tokens', 'tokenizer.ggml.scores',
                'tokenizer.ggml.pre', 'tokenizer.ggml.token_type',
                'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
                'tokenizer.ggml.unknown_token_id', 'tokenizer.ggml.padding_token_id',
                'tokenizer.ggml.add_bos_token',
                'tokenizer.ggml.add_eos_token', 'tokenizer.ggml.merges')
        if not gguf_meta.raw.get(keys[0]) or not gguf_meta.raw.get(keys[1]) or any(
                gguf_meta.raw.get(key) != draft_meta.raw.get(key) for key in keys):
            raise ModelCompatibilityError('Target and draft tokenizer metadata must match')
        target_arch = gguf_meta.raw.get('general.architecture')
        draft_arch = draft_meta.raw.get('general.architecture')
        if target_arch and draft_arch and target_arch != draft_arch:
            raise ModelCompatibilityError(
                f'Target ({target_arch}) and draft ({draft_arch}) model architectures must match'
            )
        draft_kv = estimate_kv_cache(draft_meta.config_for_kv(), context_tokens=context,
                                     dtype='fp16', cpu_cache=True).estimated_bytes
        draft_reserve_mb = (draft.stat().st_size + (draft_kv or 512 * 1024**2) + 1024**2 - 1) // 1024**2 + 512
        if draft_reserve_mb >= ram_budget_mb:
            raise ModelCompatibilityError(f'Draft model requires an estimated {draft_reserve_mb} MiB of RAM; increase the RAM limit')
        resolved_draft = str(draft)
        warnings.append(f'Draft model reserves an estimated {draft_reserve_mb} MiB of RAM and runs on CPU; speedup depends on the models and hardware.')

    if vram_guardrail_mb is not None:
        if type(vram_guardrail_mb) is not int or vram_guardrail_mb < 0:
            raise ModelCompatibilityError("VRAM guardrail must be a non-negative integer")
        if selected != "llama_cpp" or not prefer_gpu:
            raise ModelCompatibilityError("Live VRAM guardrail currently requires a GPU llama.cpp backend")
        free_mb = _first_visible_gpu_free_mb()
        if free_mb is None:
            raise ModelCompatibilityError("Cannot read free NVIDIA VRAM for the configured guardrail")
        if free_mb <= vram_guardrail_mb:
            raise ModelCompatibilityError("Free VRAM does not exceed the configured guardrail")
        vram_limit_mb = min(vram_limit_mb, free_mb - vram_guardrail_mb)
        warnings.append(f"VRAM budget limited to {vram_limit_mb} MiB from {free_mb} MiB free with a {vram_guardrail_mb} MiB guardrail on the first visible GPU.")

    if artifact.format == "gguf" and selected != "llama_cpp":
        raise ModelCompatibilityError("Standalone GGUF checkpoints use llama_cpp")
    if artifact.format == "sytra_moe" and selected != "sytra_moe":
        raise ModelCompatibilityError(
            "A native out-of-core MoE container must use Sytra-Helix's native MoE runtime (`sytra_moe`)."
        )
    if selected == "sytra_moe" and (
        adapter is None or adapter.engine != "sytra_moe"
    ):
        raise ModelCompatibilityError(
            "No native Sytra adapter matches this checkpoint. Create and validate an exact "
            ".sytra-runtime.json contract; Sytra will not guess an architecture kernel."
        )

    effective_kv_quant = kv_norm
    kv_plan = estimate_kv_cache(
        config,
        context_tokens=context,
        dtype=effective_kv_quant,
        cpu_cache=cpu_kv_cache,
        persistent=selected == "sytra_moe",
        tier="engine-managed" if selected == "sytra_moe" else None,
    )
    tensor_parallel = int(
        os.environ.get("SYTRA_VLLM_TENSOR_PARALLEL_SIZE", str(max(len(gpu_memory), 1)))
    )
    if tensor_parallel <= 0:
        raise ModelCompatibilityError("SYTRA_VLLM_TENSOR_PARALLEL_SIZE must be positive")
    placement = plan_weight_placement(
        weight_bytes=artifact.size_bytes,
        vram_budget_mb=vram_limit_mb,
        ram_budget_mb=ram_budget_mb,
        tensor_parallel_size=tensor_parallel,
        allow_cpu_offload=selected in {"llama_cpp", "vllm", "sytra_moe"},
        allow_nvme_streaming=selected == "sytra_moe",
    )

    weight_mb = (artifact.size_bytes + 1024**2 - 1) // 1024**2
    required_vram_mb: int | None = (
        max(int(weight_mb * 1.10), int(weight_mb + 1024))
        if selected == "vllm"
        else None
    )
    command: list[str] = []
    preflight_command: list[str] = []
    compatible = True
    llama_offload: LlamaCppOffloadPlan | None = None
    runtime_version: str | None = None

    if selected == "llama_cpp":
        launcher = _find_llama_server(root)
        help_text = _llama_server_help(launcher) if launcher else ""
        if resolved_draft:
            model_flag = next((flag for flag in ('--spec-draft-model', '--model-draft', '-md') if _help_has(help_text, flag)), None)
            layers_flag = next((flag for flag in ('--spec-draft-ngl', '--n-gpu-layers-draft', '-ngld') if _help_has(help_text, flag)), None)
            if not model_flag or not layers_flag:
                raise ModelCompatibilityError('Installed llama.cpp does not advertise draft model and draft GPU-layer options')
        runtime_version = _llama_server_version(launcher) if launcher else None
        layout = read_gguf_layer_sizes(artifact.model_path, artifact.n_layer) if artifact.n_layer else None
        if cpu_kv_cache and not _help_has(help_text, "--no-kv-offload", "-nkvo"):
            raise ModelCompatibilityError("Installed llama.cpp does not advertise CPU KV offload")
        llama_kv_type = _LLAMA_KV_TYPE_MAP.get(kv_norm, kv_norm)
        has_ctk = _help_has(help_text, "--cache-type-k", "-ctk")
        # If cache quant flags cannot be verified, budget the runtime's F16 default.
        if not has_ctk:
            if kv_norm not in {"auto", "fp16", "f16", "float16"}:
                warnings.append(
                    f"Installed llama.cpp does not advertise -ctk/-ctv; planning FP16 KV cache instead of {kv_cache_quant}."
                )
            effective_kv_quant = "fp16"
            kv_plan = estimate_kv_cache(config, context_tokens=context, dtype="fp16", cpu_cache=cpu_kv_cache)
        elif kv_norm in _QUANTIZED_KV_QUANTS and not flash_attention:
            compatible = False
            reasons.append(
                f"Quantized KV cache ({kv_cache_quant}) requires Flash Attention enabled in llama.cpp; enable Flash Attention or select fp16."
            )
        elif kv_norm in {"q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}:
            warnings.append(
                f"{kv_cache_quant.upper()} KV cache reduces KV memory versus FP16/Q8_0 and may affect perplexity on sensitive tasks; evaluate on your prompts."
            )
        offload = plan_llama_cpp_offload(
            weight_bytes=artifact.size_bytes,
            vram_budget_mb=vram_limit_mb,
            ram_budget_mb=ram_budget_mb - draft_reserve_mb,
            n_layer=artifact.n_layer,
            kv_bytes=kv_plan.estimated_bytes,
            cpu_count=os.cpu_count() or 4,
            windows=os.name == "nt",
            cpu_kv_cache=cpu_kv_cache,
            force_gpu_layers=force_gpu_layers,
            layer_sizes=layout[0] if layout else None,
        )
        if artifact.is_moe and moe_expert_offload != "off" and layout and sum(layout[1]) and _help_has(help_text, "--cpu-moe"):
            candidate = plan_llama_cpp_offload(
                weight_bytes=artifact.size_bytes,
                vram_budget_mb=vram_limit_mb,
                ram_budget_mb=ram_budget_mb - draft_reserve_mb,
                n_layer=artifact.n_layer,
                kv_bytes=kv_plan.estimated_bytes,
                cpu_count=os.cpu_count() or 4,
                windows=os.name == "nt",
                cpu_kv_cache=cpu_kv_cache,
                force_gpu_layers=force_gpu_layers,
                layer_sizes=layout[0],
                expert_sizes=layout[1],
                cpu_moe=True,
            )
            if moe_expert_offload == "cpu" or (candidate.strategy != "insufficient-memory" and candidate.gpu_layers > offload.gpu_layers):
                offload = candidate
        elif moe_expert_offload == "cpu":
            raise ModelCompatibilityError("CPU expert offload requires an MoE GGUF with readable tensor sizes and llama.cpp --cpu-moe support")
        llama_offload = offload
        if gpu_visible and offload.gpu_layers == 0 and not allow_cpu_only:
            compatible = False
            reasons.append(
                "An NVIDIA GPU is visible but this plan would offload 0 layers (CPU-only). "
                "Sytra will not silently degrade. Raise the VRAM budget, pick a smaller GGUF, "
                "or pass --allow-cpu-only for an explicit CPU baseline."
            )
        if launcher is None:
            compatible = False
            reasons.append(llama_server_provision_hint(root))
        elif offload.strategy == "insufficient-memory":
            compatible = False
            command = []
            reasons.extend(offload.notes)
        elif not compatible:
            command = []
        else:
            command = launcher + [
                "-m",
                artifact.model_path,
                "--host",
                host,
                "--port",
                str(port),
                "-c",
                str(context),
                "-ngl",
                str(offload.gpu_layers),
                "-t",
                str(offload.threads),
                "-b",
                str(offload.batch),
                "-ub",
                str(offload.ubatch),
            ]
            if offload.cpu_moe:
                command.append("--cpu-moe")
            if help_text:
                if _help_has(help_text, "-fa", "--flash-attn"):
                    command.extend(["-fa", "on" if flash_attention else "off"])
                if _help_has(help_text, "-ctk") and llama_kv_type != "auto":
                    command.extend(["-ctk", llama_kv_type, "-ctv", llama_kv_type])
                if cpu_kv_cache and _help_has(help_text, "-nkvo"):
                    command.append("-nkvo")
                if _help_has(help_text, "--no-mmap") and not offload.mmap:
                    command.append("--no-mmap")
                if offload.mlock and _help_has(help_text, "--mlock"):
                    command.append("--mlock")
                if _help_has(help_text, "--split-mode"):
                    command.extend(["--split-mode", "layer"])
            if artifact.mmproj_path:
                command.extend(["--mmproj", artifact.mmproj_path])
            if resolved_draft:
                command.extend([model_flag, resolved_draft, layers_flag, '0'])
                if _help_has(help_text, '--spec-type') and 'draft-simple' in help_text:
                    command.extend(['--spec-type', 'draft-simple'])
            reasons.append(
                "GGUF uses llama.cpp GPU-first hybrid: transformer blocks fill VRAM; "
                "cold layers stay mmap'd in RAM. mlock stays off on Windows."
            )
            reasons.extend(offload.notes)
            if artifact.is_moe:
                warnings.append(
                    "MoE GGUF: expert placement follows the resolved layer/CPU-expert plan. "
                    "Sytra does not claim native expert-pager speed unless .sytra-runtime.json exists."
                )
            if artifact.n_expert and artifact.n_expert_used:
                warnings.append(
                    f"Total routed experts {artifact.n_expert}; active per token {artifact.n_expert_used}."
                )
    elif selected == "vllm":
        launcher = _find_vllm()
        if gpu_memory and tensor_parallel > len(gpu_memory):
            compatible = False
            reasons.append(
                f"vLLM tensor parallel size is {tensor_parallel}, but only {len(gpu_memory)} NVIDIA GPUs are visible."
            )
        if launcher is None:
            compatible = False
            reasons.append(
                "vLLM was not found. Install it in the serving environment or set SYTRA_VLLM_COMMAND."
            )
        if placement.strategy == "insufficient-memory":
            compatible = False
            reasons.append(
                f"Checkpoint needs approximately {required_vram_mb} MiB before runtime overhead, "
                f"but the conservative GPU+RAM budget is {vram_limit_mb + ram_budget_mb} MiB. "
                "No verified NVMe adapter matches this architecture. "
                "Build sytra-engine (`cargo build -p sytra-engine --release`) and create a "
                ".sytra-runtime.json index to serve via Sytra-Helix MoE."
            )
        if launcher is not None and compatible:
            command = launcher + [
                "serve",
                artifact.model_path,
                "--host",
                host,
                "--port",
                str(port),
                "--max-model-len",
                str(context),
                "--gpu-memory-utilization",
                "0.90",
            ]
            if placement.cpu_offload_gb_per_gpu > 0:
                command.extend(
                    [
                        "--offload-backend",
                        "uva",
                        "--cpu-offload-gb",
                        str(placement.cpu_offload_gb_per_gpu),
                    ]
                )
                if artifact.is_moe:
                    command.extend(["--cpu-offload-params", "experts"])
            if kv_norm in {"fp8", "float8"}:
                command.extend(["--kv-cache-dtype", "fp8"])
            elif kv_norm in _QUANTIZED_KV_QUANTS:
                effective_kv_quant = "fp16"
                kv_plan = estimate_kv_cache(config, context_tokens=context, dtype="fp16", cpu_cache=cpu_kv_cache)
                warnings.append(
                    f"vLLM does not support ggml {kv_cache_quant} KV cache quantization; planning FP16 KV cache instead."
                )
            if tensor_parallel > 1:
                command.extend(["--tensor-parallel-size", str(tensor_parallel)])
                if artifact.is_moe:
                    command.append("--enable-expert-parallel")
            if placement.cpu_offload_gb_per_gpu > 0:
                reasons.append(
                    "Complete Safetensors checkpoint fits the conservative vLLM GPU+RAM UVA-offload budget."
                )
            else:
                reasons.append(
                    "Complete Safetensors checkpoint fits the declared vLLM accelerator-memory budget."
                )
        if cpu_kv_cache:
            warnings.append(
                "CPU KV-cache placement is not forced on vLLM; vLLM owns its paged KV-cache policy."
            )
    elif selected == "sytra_moe":
        assert adapter is not None
        launcher = _find_sytra_engine(root)
        try:
            validate_adapter_payload(artifact.model_path, adapter)
        except AdapterCompatibilityError as exc:
            compatible = False
            reasons.append(str(exc))
        if launcher is None:
            compatible = False
            reasons.append(
                "Sytra-Helix native MoE engine (`sytra-engine`) was not found. "
                "Build `cargo build -p sytra-engine --release`, install sytra-engine on PATH, "
                "or set SYTRA_ENGINE_COMMAND."
            )
        if launcher is not None and compatible:
            command = launcher + [
                "serve",
                "--model",
                artifact.model_path,
                "--host",
                host,
                "--port",
                str(port),
                "--context",
                str(context),
                "--verification-positions",
                str(verification_positions),
                "--ram-limit-mb",
                str(ram_budget_mb),
                "--accelerator-limit-mb",
                str(vram_limit_mb),
                "--dense-tile-mb",
                "64",
                "--kv-scalar-bytes",
                "2",
                "--storage-bandwidth-mbps",
                str(storage_bandwidth_mbps),
                "--target-tps",
                str(target_tps),
            ]
            if draft_url:
                command.extend(["--draft-url", draft_url])
                if draft_model:
                    command.extend(["--draft-model", draft_model])
            oracle_path = Path(artifact.model_path) / ".sytra-oracle.json"
            if not oracle_path.is_file():
                if allow_unverified:
                    command.append("--allow-unverified-checkpoint")
                    warnings.append(
                        "Checkpoint has no .sytra-oracle.json; starting Helix with explicit unverified checkpoint override."
                    )
                else:
                    warnings.append(
                        "Checkpoint has no .sytra-oracle.json; generate an oracle first or enable unverified checkpoint startup."
                    )
            preflight_command = launcher + [
                "doctor",
                "--model",
                artifact.model_path,
                "--context",
                str(context),
                "--verification-positions",
                str(verification_positions),
                "--ram-limit-mb",
                str(ram_budget_mb),
                "--accelerator-limit-mb",
                str(vram_limit_mb),
                "--dense-tile-mb",
                "64",
                "--kv-scalar-bytes",
                "2",
                "--storage-bandwidth-mbps",
                str(storage_bandwidth_mbps),
                "--target-tps",
                str(target_tps),
            ]
            if allow_unverified and not oracle_path.is_file():
                preflight_command.append("--allow-unverified-checkpoint")
            reasons.append(
                f"{adapter.display_name} uses Sytra-Helix MoE v2 native byte-exact expert store, "
                "batch-union scheduler, RAM/accelerator cache, heterogeneous CPU+GPU co-execution, "
                "layer-windowed anti-OOM execution, and NVMe prefetch path."
            )
            if kv_cache_quant.lower() not in {"auto", "fp16", "f16", "float16", "bf16", "bfloat16"}:
                effective_kv_quant = "fp16"
                kv_plan = estimate_kv_cache(
                    config,
                    context_tokens=context,
                    dtype="fp16",
                    cpu_cache=cpu_kv_cache,
                    persistent=True,
                    tier="engine-managed",
                )
                warnings.append(
                    "The native architecture adapter owns compressed/persistent KV state; "
                    f"the generic {kv_cache_quant} KV setting is not forced."
                )
            if cpu_kv_cache:
                warnings.append(
                    "The native architecture adapter controls KV placement; the generic CPU-KV switch is ignored."
                )

    io_max_tps = estimate_streamed_moe_tps(
        nvme_weight_bytes=placement.estimated_nvme_weight_bytes,
        n_expert=artifact.n_expert,
        n_expert_used=artifact.n_expert_used,
        storage_bandwidth_mbps=storage_bandwidth_mbps,
    )
    helix_plan = (
        build_helix_resource_plan(
            weight_bytes=artifact.size_bytes,
            vram_limit_mb=vram_limit_mb,
            ram_limit_mb=ram_budget_mb,
            nvme_weight_bytes=placement.estimated_nvme_weight_bytes,
            storage_bandwidth_mbps=storage_bandwidth_mbps,
            target_tps=target_tps,
            strategy=placement.strategy,
            io_max_tps=io_max_tps,
            gpu_visible=gpu_visible or vram_limit_mb > 0,
            allow_cpu_only=allow_cpu_only or not prefer_gpu,
            parameter_count=artifact.parameter_count,
            quantization=artifact.quantization,
            n_layer=artifact.n_layer,
            n_expert=artifact.n_expert,
            n_expert_used=artifact.n_expert_used,
            adapter_id=artifact.adapter_id,
            architecture=artifact.architecture,
            config=config,
        )
        if selected == "sytra_moe" or artifact.is_moe
        else None
    )

    if io_max_tps is not None and selected == "sytra_moe" and compatible:
        if io_max_tps < HELIX_STREAMING_FLOOR_TPS:
            compatible = False
            command = []
            reasons.append(
                f"Storage bandwidth {storage_bandwidth_mbps} MB/s bounds streamed MoE decode at "
                f"about {io_max_tps:.3f} tok/s, below Sytra-Helix's minimum streaming floor "
                f"({HELIX_STREAMING_FLOOR_TPS:g} tok/s). This host cannot stream this checkpoint."
            )
        elif io_max_tps < target_tps:
            compatible = False
            command = []
            reasons.append(
                f"Storage bandwidth {storage_bandwidth_mbps} MB/s bounds streamed MoE decode at "
                f"about {io_max_tps:.3f} tok/s, below --target-tps {target_tps:g}. "
                "Lower --target-tps, raise measured --storage-bandwidth-mbps, or use a smaller model. "
                f"Sytra will not claim {target_tps:g} tok/s on this envelope."
            )

    kv_snapshots = {}
    for ctx in (2048, 4096, 8192):
        snap = estimate_kv_cache(
            config,
            context_tokens=ctx,
            dtype=effective_kv_quant,
            cpu_cache=cpu_kv_cache,
            persistent=selected == "sytra_moe",
        )
        kv_snapshots[f"kv_{ctx // 1024}k_mb"] = (
            None if snap.estimated_bytes is None else (snap.estimated_bytes + 1024**2 - 1) // 1024**2
        )
    active_params = None
    if artifact.parameter_count and artifact.n_expert and artifact.n_expert_used:
        dense_share = 0.05 if artifact.n_expert >= 64 else 0.20
        dense = int(artifact.parameter_count * dense_share)
        routed = artifact.parameter_count - dense
        active_params = dense + int(routed * artifact.n_expert_used / artifact.n_expert)
    estimates = {
        "weight_mb": (artifact.size_bytes + 1024**2 - 1) // 1024**2,
        **kv_snapshots,
        "gpu_layers": llama_offload.gpu_layers if llama_offload else None,
        "cpu_layers": llama_offload.cpu_layers if llama_offload else None,
        "peak_vram_mb": llama_offload.peak_vram_mb if llama_offload else required_vram_mb,
        "peak_ram_mb": llama_offload.peak_ram_mb + draft_reserve_mb if llama_offload else None,
        "draft_ram_reserve_mb": draft_reserve_mb or None,
        "total_params": artifact.parameter_count,
        "active_params_per_token": active_params,
        "quantization": artifact.quantization,
        "architecture": artifact.architecture,
        "is_moe": artifact.is_moe,
        "runtime_version": runtime_version,
        "io_max_tps": io_max_tps,
        "target_tps": target_tps,
        "storage_bandwidth_mbps": storage_bandwidth_mbps,
        "helix_resource_plan": helix_plan.to_dict() if helix_plan else None,
    }

    return BackendPlan(
        compatible=compatible,
        backend=selected,
        artifact=artifact,
        command=command,
        reasons=reasons,
        warnings=warnings,
        required_vram_mb=required_vram_mb,
        available_vram_mb=vram_limit_mb,
        available_ram_mb=ram_budget_mb,
        kv_cache=kv_plan,
        weight_placement=placement,
        preflight_command=preflight_command,
        llama_offload=llama_offload,
        runtime_version=runtime_version,
        estimates=estimates,
    )


def launch_backend_plan(plan: BackendPlan) -> subprocess.Popen:
    if not plan.compatible or not plan.command:
        reasons = "; ".join(plan.reasons) if plan.reasons else "No valid launch command"
        raise ModelCompatibilityError(f"Cannot launch incompatible backend plan: {reasons}")
    env = prepend_runtime_path(os.environ.copy(), plan.command)
    return subprocess.Popen(plan.command, env=env, **silent_subprocess_kwargs())
