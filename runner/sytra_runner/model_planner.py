"""Capability-gated local inference backend planning.

Sytra does not implement model kernels here. It verifies the local artifact and
launches a mature OpenAI-compatible server:

* llama.cpp for GGUF and CPU/GPU hybrid inference.
* vLLM for complete Hugging Face/Safetensors checkpoints that fit GPU memory
  or a conservative GPU+RAM UVA-offload budget.
* Sytra's native runtime for explicitly indexed, architecture-validated
  VRAM/RAM/NVMe MoE containers.
* Colibri (`coli serve --auto-tier`) for frontier disk-streamed MoE (GLM-5.2,
  Kimi K3, Inkling, OLMoE safetensors). Sytra unpacks a pinned `coli` release
  into `.tools/colibri` on that path when the launcher is missing.
"""
from __future__ import annotations

import importlib.util
import json
import os
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
from .gguf_meta import try_read_gguf_metadata
from .colibri_bridge import (
    COLIBRI_PREFERRED_FAMILIES,
    COLIBRI_STREAMING_FLOOR_TPS,
    build_colibri_preflight_command,
    build_colibri_serve_command,
    colibri_install_hint,
    colibri_model_id,
    colibri_resource_plan,
    detect_colibri_family,
)
from .colibri_provision import maybe_provision_colibri
from .runtime_detect import (
    find_colibri,
    find_llama_server,
    find_sytra_engine,
    llama_server_provision_hint,
)
from .memory_hierarchy import (
    KVCachePlan,
    LlamaCppOffloadPlan,
    WeightPlacementPlan,
    estimate_kv_cache,
    estimate_streamed_moe_tps,
    plan_llama_cpp_offload,
    plan_weight_placement,
)

BackendName = Literal["llama_cpp", "vllm", "sytra_moe", "colibri"]


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
    is_moe = any(
        int(value or 0) > 0
        for value in (
            config.get("num_local_experts"),
            config.get("n_routed_experts"),
            text_config.get("num_local_experts"),
            text_config.get("n_routed_experts"),
        )
    )
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
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr or "").strip().splitlines()
    return text[0][:200] if text else None


def _help_has(help_text: str, *needles: str) -> bool:
    lowered = help_text.lower()
    return any(needle.lower() in lowered for needle in needles)


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
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if output.returncode:
        return []
    return [int(line.strip()) for line in output.stdout.splitlines() if line.strip().isdigit()]


def build_backend_plan(
    model_path: str | Path,
    *,
    backend: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8080,
    context: int = 4096,
    verification_positions: int = 8,
    storage_bandwidth_mbps: int = 3500,
    target_tps: float = 5.0,
    draft_url: str | None = None,
    draft_model: str | None = None,
    vram_limit_mb: int = 8192,
    ram_limit_mb: int | None = None,
    cpu_kv_cache: bool = False,
    kv_cache_quant: str = "q8_0",
    flash_attention: bool = True,
    project_root: str | Path | None = None,
    force_gpu_layers: int | None = None,
    allow_cpu_only: bool = False,
) -> BackendPlan:
    if (
        context <= 0
        or verification_positions <= 0
        or storage_bandwidth_mbps <= 0
        or target_tps <= 0
        or vram_limit_mb <= 0
    ):
        raise ModelCompatibilityError(
            "Context, speculative verification positions, storage bandwidth, target TPS, "
            "and VRAM limit must be positive"
        )
    artifact = inspect_model(model_path)
    reasons: list[str] = []
    warnings: list[str] = []
    requested_backend = backend.strip().lower()
    if requested_backend not in {"auto", "llama_cpp", "vllm", "sytra_moe", "colibri"}:
        raise ModelCompatibilityError(f"Unknown inference backend: {backend!r}")

    root = Path(project_root).resolve() if project_root else None
    ram_budget_mb = ram_limit_mb if ram_limit_mb is not None else _system_memory_mb()
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

    coli_launcher = find_colibri(root)
    colibri_family = detect_colibri_family(
        adapter_family=adapter.family if adapter else None,
        adapter_id=artifact.adapter_id,
        model_type=artifact.model_type,
        architecture=artifact.architecture,
        repo_id=artifact.repo_id,
    )
    if coli_launcher is None:
        coli_launcher = maybe_provision_colibri(
            root,
            requested_backend=requested_backend,
            colibri_family=colibri_family,
        )
    native_engine = _find_sytra_engine(root)

    if artifact.format == "gguf":
        automatic: BackendName = "llama_cpp"
    elif (
        artifact.format == "sytra_moe"
        and native_engine is not None
        and (adapter is None or adapter.family not in COLIBRI_PREFERRED_FAMILIES)
    ):
        automatic = "sytra_moe"
    elif coli_launcher and (
        colibri_family
        or artifact.format == "sytra_moe"
        or (artifact.is_moe and native_engine is None)
    ):
        automatic = "colibri"
    elif artifact.format == "sytra_moe":
        automatic = "sytra_moe"
    else:
        automatic = "vllm"
    selected: BackendName = automatic if requested_backend == "auto" else requested_backend  # type: ignore[assignment]

    if artifact.format == "gguf" and selected != "llama_cpp":
        raise ModelCompatibilityError("Standalone GGUF checkpoints use llama_cpp")
    if artifact.format == "sytra_moe" and selected not in {"sytra_moe", "colibri"}:
        raise ModelCompatibilityError(
            "A native out-of-core container must use Sytra's native MoE runtime or the Colibri bridge"
        )
    if selected == "sytra_moe" and (
        adapter is None or adapter.engine != "sytra_moe"
    ):
        raise ModelCompatibilityError(
            "No native Sytra adapter matches this checkpoint. Create and validate an exact "
            ".sytra-runtime.json contract; Sytra will not guess an architecture kernel."
        )

    kv_plan = estimate_kv_cache(
        config,
        context_tokens=context,
        dtype=kv_cache_quant,
        cpu_cache=cpu_kv_cache,
        persistent=selected in {"sytra_moe", "colibri"},
        tier="engine-managed" if selected == "sytra_moe" else None,
    )
    gpu_memory = _visible_gpu_memory_mb()
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
        allow_cpu_offload=selected in {"llama_cpp", "vllm", "sytra_moe", "colibri"},
        allow_nvme_streaming=selected in {"sytra_moe", "colibri"},
    )

    weight_mb = (artifact.size_bytes + 1024**2 - 1) // 1024**2
    required_vram_mb: int | None = (
        max(int(weight_mb * 1.10), int(weight_mb + 1024))
        if selected == "vllm"
        else None
    )
    if (
        selected == "vllm"
        and requested_backend == "auto"
        and placement.strategy == "insufficient-memory"
        and coli_launcher
        and (colibri_family or artifact.is_moe)
    ):
        selected = "colibri"
        warnings.append(
            "Checkpoint exceeds the vLLM GPU+RAM envelope; Sytra hands token generation to Colibri."
        )
        placement = plan_weight_placement(
            weight_bytes=artifact.size_bytes,
            vram_budget_mb=vram_limit_mb,
            ram_budget_mb=ram_budget_mb,
            tensor_parallel_size=tensor_parallel,
            allow_cpu_offload=True,
            allow_nvme_streaming=True,
        )
        required_vram_mb = None
    command: list[str] = []
    preflight_command: list[str] = []
    compatible = True
    llama_offload: LlamaCppOffloadPlan | None = None
    runtime_version: str | None = None

    if selected == "llama_cpp":
        launcher = _find_llama_server(root)
        help_text = _llama_server_help(launcher) if launcher else ""
        runtime_version = _llama_server_version(launcher) if launcher else None
        offload = plan_llama_cpp_offload(
            weight_bytes=artifact.size_bytes,
            vram_budget_mb=vram_limit_mb,
            ram_budget_mb=ram_budget_mb,
            n_layer=artifact.n_layer,
            kv_bytes=kv_plan.estimated_bytes,
            cpu_count=os.cpu_count() or 4,
            windows=os.name == "nt",
        )
        llama_offload = offload
        if force_gpu_layers is not None:
            n_layer = artifact.n_layer or offload.n_layer or force_gpu_layers
            gpu_layers = max(0, force_gpu_layers)
            cpu_layers = max((n_layer or gpu_layers) - gpu_layers, 0)
            llama_offload = LlamaCppOffloadPlan(
                gpu_layers=gpu_layers,
                cpu_layers=cpu_layers,
                n_layer=offload.n_layer,
                peak_vram_mb=offload.peak_vram_mb,
                peak_ram_mb=offload.peak_ram_mb,
                mmap=offload.mmap,
                mlock=offload.mlock,
                threads=offload.threads,
                batch=offload.batch,
                ubatch=offload.ubatch,
                strategy="cpu-only" if gpu_layers == 0 else offload.strategy,
                notes=offload.notes + (f"Forced n-gpu-layers={gpu_layers}.",),
            )
            offload = llama_offload
        gpu_visible = bool(gpu_memory)
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
            if help_text:
                if _help_has(help_text, "-fa", "--flash-attn"):
                    command.extend(["-fa", "on" if flash_attention else "off"])
                if _help_has(help_text, "-ctk"):
                    command.extend(["-ctk", kv_cache_quant, "-ctv", kv_cache_quant])
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
            reasons.append(
                "GGUF uses llama.cpp GPU-first hybrid: transformer blocks fill VRAM; "
                "cold layers stay mmap'd in RAM. mlock stays off on Windows."
            )
            reasons.extend(offload.notes)
            if artifact.is_moe:
                warnings.append(
                    "MoE GGUF: llama.cpp keeps inactive experts in mmap'd host memory. "
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
                "If Colibri (`coli`) is installed, retry with --backend colibri; otherwise "
                "build sytra-engine and create a .sytra-runtime.json index."
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
        if launcher is None and requested_backend == "auto" and compatible and find_colibri(root):
            selected = "colibri"
            warnings.append(
                "Native sytra-engine was not found; planning a Colibri (`coli serve`) bridge instead."
            )
        elif launcher is None:
            compatible = False
            reasons.append(
                "Sytra's native engine was not found. Build `cargo build -p sytra-engine "
                "--release`, install sytra-engine on PATH, set SYTRA_ENGINE_COMMAND, "
                "or install Colibri (`coli`) and pass --backend colibri."
            )
        if selected == "sytra_moe" and launcher is not None and compatible:
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
            reasons.append(
                f"{adapter.display_name} uses Sytra's native byte-exact expert store, "
                "batch-union scheduler, RAM/accelerator cache, and NVMe prefetch path."
            )
            if kv_cache_quant.lower() not in {"auto", "fp16"}:
                warnings.append(
                    "The native architecture adapter owns compressed/persistent KV state; "
                    f"the generic {kv_cache_quant} KV setting is not forced."
                )
            if cpu_kv_cache:
                warnings.append(
                    "The native architecture adapter controls KV placement; the generic CPU-KV switch is ignored."
                )

    if selected == "colibri":
        launcher = coli_launcher or find_colibri(root)
        coli_model = os.environ.get("COLI_MODEL") or artifact.model_path
        served_id = colibri_model_id(
            repo_id=artifact.repo_id,
            family=colibri_family,
            architecture=artifact.architecture,
        )
        if launcher is None:
            compatible = False
            command = []
            reasons.append(colibri_install_hint(root))
        elif compatible:
            command = build_colibri_serve_command(
                launcher,
                model_path=coli_model,
                host=host,
                port=port,
                context=context,
                vram_limit_mb=vram_limit_mb,
                ram_limit_mb=ram_budget_mb,
                gpu_visible=bool(gpu_memory),
                model_id=served_id,
            )
            preflight_command = build_colibri_preflight_command(
                launcher, model_path=coli_model
            )
            reasons.append(
                "Sytra handles catalog, download, and the VRAM/RAM/NVMe envelope; "
                "Colibri (`coli serve --auto-tier`) streams experts and generates tokens. "
                "On a 12 GB GPU expect well below 5 tok/s for frontier MoE — see estimates.io_max_tps."
            )

    io_max_tps = estimate_streamed_moe_tps(
        nvme_weight_bytes=placement.estimated_nvme_weight_bytes,
        n_expert=artifact.n_expert,
        n_expert_used=artifact.n_expert_used,
        storage_bandwidth_mbps=storage_bandwidth_mbps,
    )
    if io_max_tps is not None and selected in {"sytra_moe", "colibri"} and compatible:
        if io_max_tps < COLIBRI_STREAMING_FLOOR_TPS:
            compatible = False
            command = []
            reasons.append(
                f"Storage bandwidth {storage_bandwidth_mbps} MB/s bounds streamed MoE decode at "
                f"about {io_max_tps:.3f} tok/s, below Colibri's proven floor "
                f"({COLIBRI_STREAMING_FLOOR_TPS:g} tok/s). This host cannot stream this checkpoint."
            )
        elif io_max_tps < target_tps:
            if selected == "sytra_moe":
                compatible = False
                command = []
                reasons.append(
                    f"Storage bandwidth {storage_bandwidth_mbps} MB/s bounds streamed MoE decode at "
                    f"about {io_max_tps:.3f} tok/s, below --target-tps {target_tps:g}. "
                    "Lower --target-tps, raise measured --storage-bandwidth-mbps, or use a smaller model. "
                    f"Sytra will not claim {target_tps:g} tok/s on this envelope."
                )
            else:
                warnings.append(
                    f"Colibri I/O math caps decode around {io_max_tps:.3f} tok/s, below "
                    f"--target-tps {target_tps:g}. Serving anyway; this is not a 5 tok/s claim."
                )

    kv_snapshots = {}
    for ctx in (2048, 4096, 8192):
        snap = estimate_kv_cache(
            config,
            context_tokens=ctx,
            dtype=kv_cache_quant,
            cpu_cache=cpu_kv_cache,
            persistent=selected in {"sytra_moe", "colibri"},
        )
        kv_snapshots[f"kv_{ctx // 1024}k_mb"] = (
            None if snap.estimated_bytes is None else (snap.estimated_bytes + 1024**2 - 1) // 1024**2
        )
    active_params = None
    if artifact.parameter_count and artifact.n_expert and artifact.n_expert_used:
        # Rough: routed experts scale by k/n; dense remainder assumed 30%.
        dense = int(artifact.parameter_count * 0.30)
        routed = artifact.parameter_count - dense
        active_params = dense + int(routed * artifact.n_expert_used / artifact.n_expert)
    estimates = {
        "weight_mb": (artifact.size_bytes + 1024**2 - 1) // 1024**2,
        **kv_snapshots,
        "gpu_layers": llama_offload.gpu_layers if llama_offload else None,
        "cpu_layers": llama_offload.cpu_layers if llama_offload else None,
        "peak_vram_mb": llama_offload.peak_vram_mb if llama_offload else required_vram_mb,
        "peak_ram_mb": llama_offload.peak_ram_mb if llama_offload else None,
        "total_params": artifact.parameter_count,
        "active_params_per_token": active_params,
        "quantization": artifact.quantization,
        "architecture": artifact.architecture,
        "is_moe": artifact.is_moe,
        "runtime_version": runtime_version,
        "io_max_tps": io_max_tps,
        "target_tps": target_tps,
        "storage_bandwidth_mbps": storage_bandwidth_mbps,
        "colibri_resource_plan": colibri_resource_plan(
            vram_limit_mb=vram_limit_mb,
            ram_limit_mb=ram_budget_mb,
            nvme_weight_bytes=placement.estimated_nvme_weight_bytes,
            storage_bandwidth_mbps=storage_bandwidth_mbps,
            target_tps=target_tps,
            strategy=placement.strategy,
            io_max_tps=io_max_tps,
            family=colibri_family,
            model_id=colibri_model_id(
                repo_id=artifact.repo_id,
                family=colibri_family,
                architecture=artifact.architecture,
            ),
        )
        if selected == "colibri"
        else None,
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
