"""Headless merge entrypoint.

Runs a real mergekit merge. Missing mergekit or any merge failure is an
error — never a simulated success.
"""
from __future__ import annotations

import os
import sys
import subprocess
import traceback
from typing import Any

from . import telemetry
from .config import MergeConfig
from .performance.hardware import detect_hardware
from .performance.planner import merge_plan, apply_cpu_threads
from .performance.checkpoints import checkpoint_memory_gb, checkpoint_fingerprint, materialization_ready
from .adapters import adapter_metadata
from .subprocess_utils import silent_subprocess_kwargs
from .xet_safety import apply_runtime_safety, child_process_env

apply_runtime_safety()

# Check if mergekit is installed
try:
    import mergekit
    HAS_MERGEKIT = True
except ImportError:
    HAS_MERGEKIT = False


SPHERICAL_METHODS = ("multi_slerp", "nuslerp")
TASK_VECTOR_METHODS = ("ties", "dare_ties", "dare_linear", "task_arithmetic")

# A true fine-tune differs from its base by ~1-2% relative weight norm; a
# continued-pretrained lineage (e.g. Qwen2.5-Coder vs Qwen2.5) by ~120%.
# Anything above this is not a fine-tune of the declared base.
KINSHIP_MAX_REL_DELTA = 0.25
KINSHIP_PROBE_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.norm.weight",
)


def _local_snapshot(model_ref: str) -> str | None:
    """Local directory of a model — a path as-is, or the HF cache snapshot
    if already downloaded. None if not available locally (we never download
    just for the preflight)."""
    if os.path.isdir(model_ref):
        return model_ref
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    repo_dir = os.path.join(hf_home, "hub", "models--" + model_ref.replace("/", "--"), "snapshots")
    if not os.path.isdir(repo_dir):
        return None
    ref = os.path.join(os.path.dirname(repo_dir), "refs", "main")
    if os.path.isfile(ref):
        from pathlib import Path
        revision = Path(ref).read_text(encoding="utf-8").strip()
        snapshot = os.path.join(repo_dir, revision)
        return snapshot if os.path.isdir(snapshot) else None
    snaps = [os.path.join(repo_dir, d) for d in os.listdir(repo_dir)]
    snaps = [s for s in snaps if os.path.isdir(s)]
    return snaps[0] if len(snaps) == 1 else None


def _probe_tensor(snapshot: str, name: str):
    import json as _json

    from safetensors import safe_open

    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = _json.load(open(index_path))["weight_map"]
        shard = weight_map.get(name)
        if shard is None:
            return None
        shard_path = os.path.join(snapshot, shard)
    else:
        shard_path = os.path.join(snapshot, "model.safetensors")
        if not os.path.exists(shard_path):
            return None
    with safe_open(shard_path, framework="pt") as f:
        if name not in f.keys():
            return None
        return f.get_tensor(name).float()


def check_task_vector_kinship(config: MergeConfig) -> str | None:
    """For task-vector methods (and multi-model spherical methods with a base_model):
    verify each model really is a fine-tune of the base by measuring relative weight deltas
    on sampled tensors. Returns an error message when a model is a different lineage,
    None when the merge may proceed (or the check cannot run locally)."""
    is_task_vec = config.merge_method in TASK_VECTOR_METHODS or (
        config.merge_method in SPHERICAL_METHODS and bool(config.base_model)
    )
    if not is_task_vec or not config.base_model:
        return None

    base_snap = _local_snapshot(config.base_model)
    if base_snap is None:
        telemetry.emit_log(
            "kinship preflight skipped: base model not in local cache yet", stream="stderr"
        )
        return None

    try:
        for entry in config.models:
            model_ref = entry["model"] if isinstance(entry, dict) else entry
            snap = _local_snapshot(model_ref)
            if snap is None:
                continue
            deltas = []
            for tensor_name in KINSHIP_PROBE_TENSORS:
                base_t = _probe_tensor(base_snap, tensor_name)
                model_t = _probe_tensor(snap, tensor_name)
                if base_t is None or model_t is None or base_t.shape != model_t.shape:
                    continue
                rel = ((model_t - base_t).norm() / (base_t.norm() + 1e-8)).item()
                deltas.append(rel)
            if deltas and min(deltas) > KINSHIP_MAX_REL_DELTA:
                return (
                    f"'{model_ref}' is not a fine-tune of base '{config.base_model}' "
                    f"(relative weight delta {min(deltas):.2f}, fine-tunes are < {KINSHIP_MAX_REL_DELTA}). "
                    f"A {config.merge_method} merge across lineages produces a broken model. "
                    f"Use slerp to interpolate related models directly, or pick a base the "
                    f"models were actually fine-tuned from."
                )
    except Exception as exc:
        telemetry.emit_log(f"kinship preflight inconclusive: {exc}", stream="stderr")
    return None


def get_system_resources() -> tuple[int, float]:
    import multiprocessing
    import sys
    
    cores = multiprocessing.cpu_count()
    ram_gb = 16.0  # default fallback
    
    if sys.platform == "win32":
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                ram_gb = stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    else:
        try:
            ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
        except Exception:
            pass
            
    return cores, ram_gb


def estimate_model_params_from_path(model_path: str) -> float:
    if not os.path.exists(model_path):
        return 0.0
    total_bytes = 0
    if os.path.isdir(model_path):
        for root, _, files in os.walk(model_path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".pt")):
                    total_bytes += os.path.getsize(os.path.join(root, f))
    elif os.path.isfile(model_path):
        total_bytes = os.path.getsize(model_path)
    
    return total_bytes / (2 * 1024 * 1024 * 1024)


def estimate_model_params(model_ref: str) -> float:
    if os.path.exists(model_ref):
        return estimate_model_params_from_path(model_ref)
    
    import re
    m = re.search(r'[-_]([0-9]+[.]?[0-9]*)[bb]', model_ref.lower())
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 7.0


def configure_safety_limits(model_refs: list[str]) -> tuple[int, dict[str, str]]:
    cores, ram_gb = get_system_resources()

    # Always stay below the desktop compositor. High-RAM machines still
    # page if Xet or mergekit over-commits; NORMAL priority made that freeze
    # the UI on Linux and macOS as well as Windows.
    apply_runtime_safety()
    if ram_gb <= 16.5:
        threads = min(2, cores)
        category = "Severe Control (8-16GB RAM)"
    elif ram_gb <= 32.5:
        threads = min(4, max(2, cores // 2))
        category = "Strict Control (32GB RAM)"
    elif ram_gb <= 64.5:
        threads = min(8, max(4, cores - 2))
        category = "Balanced Control (64GB RAM)"
    else:
        threads = min(8, max(4, cores - 1))
        category = "Capped threads (>64GB RAM)"

    # Create thread environment dictionary
    env_vars = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    
    # Apply to current process environment
    for k, v in env_vars.items():
        os.environ[k] = v
        
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except ImportError:
        pass

    # 2. Check model parameters and print warnings
    warnings = []
    for ref in model_refs:
        params = estimate_model_params(ref)
        if ram_gb <= 16.5 and params >= 6.5:
            warnings.append(
                f"WARNING: Model '{ref}' (~{params:.1f}B params) exceeds recommended memory size for {category}. "
                f"Merging or converting this model may cause high paging and system slowdown."
            )
        elif ram_gb <= 32.5 and params >= 12.5:
            warnings.append(
                f"WARNING: Model '{ref}' (~{params:.1f}B params) exceeds recommended memory size for {category}."
            )
        elif ram_gb <= 64.5 and params >= 29.5:
            warnings.append(
                f"WARNING: Model '{ref}' (~{params:.1f}B params) exceeds recommended memory size for {category}."
            )

    for w in warnings:
        telemetry.emit_log(w, stream="stderr")
        print(w, file=sys.stderr)

    return threads, env_vars


def _merge_lora_adapter(adapter_path: str, base_model: str, output_path: str, *, plan=None, dtype="bfloat16"):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    apply_runtime_safety()
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported materialization dtype: {dtype}")
    if plan is None:
        hw = detect_hardware(torch)
        snapshot = _local_snapshot(base_model)
        size = checkpoint_memory_gb(snapshot, dtype) if snapshot else None
        plan = merge_plan(hw, {}, adapter_memory_gb=size * 1.5 if size is not None else None)
    apply_cpu_threads(plan.cpu_threads, torch)
    
    telemetry.emit_log(f"Merging LoRA adapter {adapter_path} into base model {base_model}...", stream="stdout")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        device_map={"": torch.cuda.current_device()} if plan.adapter_materialization_device == "cuda" else "cpu"
    )
    gc.collect()
    
    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload(safe_merge=True)
    gc.collect()
    
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="1GB")
    
    tokenizer_source = adapter_path if os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(output_path)
    telemetry.emit_log(f"LoRA merge completed: saved to {output_path}", stream="stdout")


def _resolve_or_materialize_model(
    model_ref: str,
    *,
    config_path: str,
    dtype: str,
    hardware=None,
    performance: dict[str, Any] | None = None,
) -> str:
    """Materialize a standalone LoRA adapter into a cached merged checkpoint when needed."""
    if not (os.path.isdir(model_ref) and os.path.exists(os.path.join(model_ref, "adapter_config.json"))):
        return model_ref

    import hashlib
    import json
    import tempfile
    from pathlib import Path
    from .backends.unsloth_sft import _dependency_versions

    info = adapter_metadata(model_ref)
    base_model = info["base_model_name_or_path"]
    snapshot = _local_snapshot(base_model)
    if info.get("revision"):
        if os.path.isdir(base_model):
            raise ValueError("A local base with an adapter revision requires an explicitly pinned snapshot")
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(base_model, revision=info["revision"])

    base_hash = checkpoint_fingerprint(snapshot) if snapshot else None
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "base": base_hash,
                "adapter": info["sha256"],
                "dtype": dtype,
                "runtime": _dependency_versions(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache_root = Path(config_path).resolve().parent / "materialized_adapters"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / cache_key
    if base_hash and materialization_ready(cached, cache_key):
        telemetry.emit_event("adapter_cache", {"hit": True, "key": cache_key})
        return str(cached)

    temp_dir = tempfile.mkdtemp(prefix="materializing-", dir=cache_root)
    hw = hardware or detect_hardware()
    size = checkpoint_memory_gb(snapshot, dtype) if snapshot else None
    adapter_plan = merge_plan(
        hw,
        performance or {},
        adapter_memory_gb=size * 1.5 if size is not None else None,
    )
    telemetry.emit_event("adapter_materialization_plan", adapter_plan.to_dict())
    _merge_lora_adapter(model_ref, snapshot or base_model, temp_dir, plan=adapter_plan, dtype=dtype)
    (Path(temp_dir) / ".sytra-complete.json").write_text(
        json.dumps({"key": cache_key}), encoding="utf-8"
    )
    if base_hash and not cached.exists():
        Path(temp_dir).rename(cached)
        result = str(cached)
    else:
        result = temp_dir
    telemetry.emit_event("adapter_cache", {"hit": False, "key": cache_key, "reusable": bool(base_hash)})
    return result


def _derive_slerp_config(mk: dict[str, Any]) -> None:
    """Populate MergeKit SLERP defaults (parameters.t and base_model) from 2 weighted models."""
    models = mk.get("models") or []
    params = dict(mk.get("parameters") or {})
    if len(models) == 2:
        m0_ref = models[0].get("model") if isinstance(models[0], dict) else models[0]
        m1_ref = models[1].get("model") if isinstance(models[1], dict) else models[1]
        if not mk.get("base_model") and isinstance(m0_ref, str):
            mk["base_model"] = m0_ref
        if "t" not in params or params["t"] is None:
            w0 = (models[0].get("parameters") or {}).get("weight") if isinstance(models[0], dict) else None
            w1 = (models[1].get("parameters") or {}).get("weight") if isinstance(models[1], dict) else None
            if w0 is not None and w1 is not None and (float(w0) + float(w1)) > 0:
                if mk.get("base_model") == m1_ref and mk.get("base_model") != m0_ref:
                    t = float(w0) / (float(w0) + float(w1))
                else:
                    t = float(w1) / (float(w0) + float(w1))
                t = max(0.0, min(1.0, t))
            else:
                t = 0.5
            params["t"] = round(t, 6)
            mk["parameters"] = params


def write_mergekit_config(config_path: str, *, hardware=None) -> str:
    """Translate a Sytra merge.yaml into a mergekit-native config.

    Sytra-only keys (version, op_id, compat, output) are dropped. The
    `tokenizer:` section is dropped when source == base: the models share
    the base tokenizer, mergekit copies it by default, and mergekit's
    tokenizer-merge path (PermutedEmbeddings) crashes on `tokens: None`
    when the section is present without token overrides.
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mk = {
        k: raw[k]
        for k in ("merge_method", "base_model", "dtype", "models", "parameters", "slices")
        if raw.get(k) is not None
    }
    tok_source = (raw.get("tokenizer") or {}).get("source")
    if tok_source and tok_source != "base":
        mk["tokenizer_source"] = tok_source

    dtype = raw.get("dtype", "bfloat16")
    performance = raw.get("performance") or {}

    # Materialize LoRA adapters in models (and base_model if provided)
    base_ref = mk.get("base_model")
    if isinstance(base_ref, str) and base_ref:
        mk["base_model"] = _resolve_or_materialize_model(
            base_ref, config_path=config_path, dtype=dtype, hardware=hardware, performance=performance
        )

    models = mk.get("models") or []
    for idx, model_entry in enumerate(models):
        model_ref = model_entry.get("model") if isinstance(model_entry, dict) else model_entry
        if not model_ref or not isinstance(model_ref, str):
            continue
        resolved = _resolve_or_materialize_model(
            model_ref, config_path=config_path, dtype=dtype, hardware=hardware, performance=performance
        )
        if resolved != model_ref:
            if isinstance(model_entry, dict):
                model_entry["model"] = resolved
            else:
                models[idx] = resolved

    if mk.get("merge_method") == "slerp":
        _derive_slerp_config(mk)

    out_path = config_path + ".mergekit.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mk, f, sort_keys=False)
    return out_path


def mergekit_cmd() -> list[str]:
    """Invoke mergekit as a module through THIS interpreter.

    Never use the mergekit-yaml.exe console-script shim: on Windows the
    shim re-executes via the shebang embedded at install time, which can
    point at a different interpreter (e.g. uv's build python after a
    git+https install) that lacks the merge stack — producing a nested,
    wedged process chain instead of a merge.
    """
    return [sys.executable, "-m", "mergekit.scripts.run_yaml"]


def write_moe_config(config: MergeConfig, config_path: str) -> str:
    """Write the separate MergeKit MoE schema with hidden-state routing."""
    import yaml
    if config.merge_method != "moe" or not config.moe:
        raise ValueError("MoE options are required")
    for entry in config.moe.get("shared_experts", []):
        source = entry["source_model"]
        if os.path.isfile(os.path.join(source, "adapter_config.json")):
            raise ValueError("Shared experts must be materialized checkpoints, not standalone LoRA adapters")
    experts = [{"source_model": entry["model"],
                "positive_prompts": entry["positive_prompts"],
                "negative_prompts": entry.get("negative_prompts", [])} for entry in config.models]
    payload = {"base_model": config.base_model, "experts": experts, "gate_mode": "hidden",
               "dtype": config.dtype, "architecture": config.moe["architecture"],
               "experts_per_token": config.moe["experts_per_token"]}
    if config.moe.get("shared_experts"):
        payload["shared_experts"] = config.moe["shared_experts"]
    out = config_path + ".mergekit-moe.yaml"
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return out


def validate_moe_architecture_available(config: MergeConfig) -> None:
    """Fail before materialization when the installed MergeKit lacks the architecture."""
    from mergekit.moe import ALL_OUTPUT_ARCHITECTURES
    wanted = config.moe["architecture"].lower()
    if not any(arch.name().lower().startswith(wanted) for arch in ALL_OUTPUT_ARCHITECTURES):
        available = ", ".join(arch.name() for arch in ALL_OUTPUT_ARCHITECTURES)
        raise ValueError(f"Installed MergeKit does not provide MoE architecture {wanted!r}; available: {available}")


def validate_moe_runtime() -> None:
    """Reject the known-incompatible Transformers 5 / older Accelerate stack."""
    import transformers
    from transformers.utils import is_accelerate_available
    major = int(transformers.__version__.split('.')[0])
    if major != 4 or not is_accelerate_available():
        raise ValueError('MoE router generation requires the provisioned MergeKit environment with compatible Transformers 4 and Accelerate; repair the merge environment')


def mergekit_memory_flags() -> list[str]:
    """Keep mergekit from mapping entire checkpoints into RAM on any OS."""
    return [
        "--lazy-unpickle",
        "--low-cpu-memory",
        "--out-shard-size",
        "1B",
    ]


def _resolve_checkpoint_dir(model_ref: str):
    """Return a local directory for `model_ref`, downloading HF snapshot only when absent."""
    from pathlib import Path

    snap = _local_snapshot(model_ref)
    if snap and os.path.isdir(snap):
        return Path(snap)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_ref))


def _validate_merged_artifact(stage_dir) -> None:
    """Verify staged merge outputs exist and are structurally non-empty before publish."""
    import json
    from pathlib import Path

    stage = Path(stage_dir)
    cfg_path = stage / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"Merged output is missing config.json in {stage}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Merged config.json must be a JSON object")

    index_path = stage / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") or {}
        if not weight_map:
            raise RuntimeError("Merged model.safetensors.index.json has an empty weight_map")
        for shard_name in sorted(set(weight_map.values())):
            shard = stage / shard_name
            if not shard.is_file() or shard.stat().st_size <= 0:
                raise RuntimeError(f"Merged shard missing or empty: {shard_name}")
        return

    weight_files = [
        p for p in stage.iterdir()
        if p.is_file() and (p.name.endswith(".safetensors") or p.name.endswith(".bin"))
    ]
    if not weight_files or all(p.stat().st_size <= 0 for p in weight_files):
        raise RuntimeError(f"Merged output contains no non-empty weight files in {stage}")


def _atomic_publish_dir(stage_dir, target_dir) -> None:
    """Atomically promote a verified staging directory to target_dir."""
    import shutil
    import uuid
    from pathlib import Path

    stage = Path(stage_dir)
    target = Path(target_dir)
    backup = None
    if target.exists():
        backup = target.parent / f".sytra-merge-backup-{uuid.uuid4().hex[:8]}"
        target.rename(backup)
    try:
        stage.rename(target)
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _write_provenance(stage_dir, hardware, plan, method: str) -> None:
    import importlib.metadata
    import json
    from pathlib import Path

    versions = {"python": sys.version.split()[0]}
    try:
        from .backends.unsloth_sft import _dependency_versions

        versions.update(_dependency_versions())
    except Exception:
        pass
    if method not in SPHERICAL_METHODS:
        try:
            versions["mergekit"] = importlib.metadata.version("mergekit")
        except Exception:
            pass
    versions["merge_backend"] = "sytra_spherical" if method in SPHERICAL_METHODS else "mergekit"
    (Path(stage_dir) / "sytra-provenance.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "hardware": hardware.to_dict(),
                "merge_plan": plan.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_real_merge(config: MergeConfig, config_path: str) -> int:
    """Execute real merge (MergeKit or Sytra spherical merger) using atomic staging."""
    import dataclasses
    import re
    import shutil
    import tempfile
    from pathlib import Path
    import yaml

    op_id = config.op_id or "00000000-0000-0000-0000-000000000000"
    model_path = config.output.get("model_path", "./merged")
    target_dir = Path(model_path).resolve()

    telemetry.emit_starting(op_id, {
        "protocol_version": 1,
        "op": "merge",
        "method": config.merge_method,
        "models": len(config.models),
    })
    if config.merge_method == "moe":
        validate_moe_runtime()
        validate_moe_architecture_available(config)

    # Refuse lineage-mismatched task-vector merges before spending time on weights.
    kinship_error = check_task_vector_kinship(config)
    if kinship_error:
        telemetry.emit_error(kinship_error)
        return 1

    state = {"progress": 0.02, "stage": "downloading_models"}
    telemetry.emit_event("stage", {"stage": state["stage"]})
    telemetry.emit_metric(progress=state["progress"], stage=state["stage"])

    apply_runtime_safety()
    env = child_process_env()
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = env.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

    model_refs = [entry["model"] if isinstance(entry, dict) else entry for entry in config.models]
    if config.base_model:
        model_refs.append(config.base_model)
    if config.merge_method == "moe":
        model_refs.extend(entry["source_model"] for entry in config.moe.get("shared_experts", []))
    hardware = detect_hardware()
    sizes = []
    for ref in model_refs:
        snapshot = _local_snapshot(ref)
        sizes.append(checkpoint_memory_gb(snapshot, config.dtype) if snapshot else None)
    expected_ram = sum(sizes) * 1.5 if sizes and all(size is not None for size in sizes) else None
    plan = merge_plan(hardware, config.performance, estimated_ram_gb=expected_ram)
    env.update(apply_cpu_threads(plan.cpu_threads))
    telemetry.emit_event("hardware_profile", hardware.to_dict())
    telemetry.emit_event("merge_plan", plan.to_dict())

    mergekit_config = write_mergekit_config(config_path, hardware=hardware)
    with open(mergekit_config, encoding="utf-8") as handle:
        loaded_mk = yaml.safe_load(handle) or {}
    resolved_models = loaded_mk.get("models", config.models)
    resolved_base = loaded_mk.get("base_model", config.base_model)
    resolved_config = dataclasses.replace(config, models=resolved_models, base_model=resolved_base)

    # Repeat the lineage gate after materializing LoRA adapters, before execution.
    kinship_error = check_task_vector_kinship(resolved_config)
    if kinship_error:
        raise ValueError(kinship_error)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".sytra-merge-stage-", dir=target_dir.parent))
    try:
        if config.merge_method in SPHERICAL_METHODS:
            from .spherical_merge import parse_shard_bytes, run_spherical_merge

            state.update({"progress": 0.45, "stage": "loading_models"})
            telemetry.emit_metric(progress=state["progress"], stage=state["stage"])
            normalized_entries = [
                entry if isinstance(entry, dict) else {"model": entry}
                for entry in resolved_models
            ]
            model_dirs = [
                _resolve_checkpoint_dir(entry["model"])
                for entry in normalized_entries
            ]
            base_dir = _resolve_checkpoint_dir(resolved_base) if resolved_base else None
            tok_source = (config.tokenizer or {}).get("source", "base")
            if tok_source and tok_source != "base":
                tokenizer_dir = _resolve_checkpoint_dir(tok_source)
            else:
                tokenizer_dir = base_dir or model_dirs[0]

            state.update({"progress": 0.60, "stage": "merging_tensors"})
            telemetry.emit_event("stage", {"stage": "merging_tensors"})
            telemetry.emit_metric(progress=state["progress"], stage=state["stage"])
            max_shard_bytes = parse_shard_bytes(getattr(plan, "out_shard_size", "1B"), config.dtype)
            summary = run_spherical_merge(
                method=config.merge_method,
                model_dirs=model_dirs,
                model_entries=normalized_entries,
                base_dir=base_dir,
                tokenizer_dir=tokenizer_dir,
                dtype=config.dtype,
                stage_dir=stage_dir,
                parameters=config.parameters,
                max_shard_bytes=max_shard_bytes,
            )
            telemetry.emit_log(
                f"Spherical merge ({config.merge_method}) wrote {summary.get('tensors', 0)} tensors "
                f"across {summary.get('shards', 0)} shard(s).",
                stream="stdout",
            )
            state.update({"progress": 0.95, "stage": "writing_shards"})
            telemetry.emit_metric(progress=state["progress"], stage=state["stage"])
        else:
            if config.merge_method == "moe":
                mergekit_config = write_moe_config(resolved_config, config_path)
            plan_flags = list(plan.flags()) if callable(getattr(plan, "flags", None)) else []
            if config.merge_method == "moe":
                runner_flags = list(mergekit_memory_flags())
                if "--out-shard-size" in plan_flags and "--out-shard-size" in runner_flags:
                    pidx = plan_flags.index("--out-shard-size")
                    ridx = runner_flags.index("--out-shard-size")
                    if pidx + 1 < len(plan_flags) and ridx + 1 < len(runner_flags):
                        runner_flags[ridx + 1] = plan_flags[pidx + 1]
                if getattr(plan, "cuda", False) and "--cuda" not in runner_flags:
                    runner_flags.append("--cuda")
            else:
                if not plan_flags:
                    plan_flags = [*mergekit_memory_flags(), *(["--cuda"] if getattr(plan, "cuda", False) else [])]
                runner_flags = ["--allow-crimes", *plan_flags]

            if config.merge_method == "moe":
                runner_root = str(Path(__file__).resolve().parent.parent)
                existing_py_path = env.get("PYTHONPATH", "")
                if runner_root not in existing_py_path.split(os.pathsep):
                    env["PYTHONPATH"] = (
                        f"{runner_root}{os.pathsep}{existing_py_path}" if existing_py_path else runner_root
                    )

            cmd = [
                *([sys.executable, "-m", "sytra_runner.moe_exec"] if config.merge_method == "moe" else mergekit_cmd()),
                mergekit_config,
                str(stage_dir),
                *runner_flags,
            ]

            # stdin MUST be detached: the GUI host has no console, so any prompt
            # (gated model, license ack) would block the merge forever otherwise.
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # Windows defaults text pipes to cp1252, which cannot decode
                # tqdm's UTF-8 block glyphs — the reader would fail mid-merge.
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                **silent_subprocess_kwargs(),
            )

            def snapshot():
                return {"progress": round(state["progress"], 3), "stage": state["stage"]}

            with telemetry.Heartbeat(snapshot):
                if process.stdout:
                    for line in process.stdout:
                        line_str = line.strip()
                        if not line_str:
                            continue

                        lower = line_str.lower()
                        m = re.search(r"(\d+)%", line_str)
                        if m:
                            percent = int(m.group(1)) / 100.0
                            if any(x in lower for x in ("fetch", "download")):
                                stage, progress = "downloading_models", 0.02 + percent * 0.43
                            elif "load" in lower or "warmup" in lower:
                                stage, progress = "loading_models", 0.45 + percent * 0.10
                            elif any(x in lower for x in ("writ", "sav", "shard", "copy")):
                                stage, progress = "writing_shards", 0.85 + percent * 0.14
                            else:
                                stage, progress = "merging_tensors", 0.55 + percent * 0.30
                            if progress > state["progress"]:
                                state["progress"] = progress
                                state["stage"] = stage
                                telemetry.emit_metric(progress=round(progress, 3), stage=stage)
                        elif "merging" in lower or "executing" in lower:
                            if state["stage"] != "merging_tensors":
                                state["stage"] = "merging_tensors"
                                state["progress"] = max(state["progress"], 0.55)
                                telemetry.emit_event("stage", {"stage": "merging_tensors"})

                        telemetry.emit_log(line_str, stream="stdout")

            ret_code = process.wait()
            if ret_code != 0:
                raise RuntimeError(f"mergekit exited with code {ret_code}")

        _validate_merged_artifact(stage_dir)
        _write_provenance(stage_dir, hardware, plan, config.merge_method)
        _atomic_publish_dir(stage_dir, target_dir)
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)

    telemetry.emit_metric(progress=1.0, stage="finished")
    telemetry.emit_done({
        "model_path": str(model_path),
        "method": config.merge_method,
        "models": len(config.models),
    })
    return 0


def run(config_path: str) -> int:
    """Merge dispatcher."""
    try:
        config = MergeConfig.load(config_path)
    except Exception as exc:
        telemetry.emit_error(f"Failed to load merge config: {exc}", traceback.format_exc())
        return 1

    if not HAS_MERGEKIT and config.merge_method not in SPHERICAL_METHODS:
        telemetry.emit_error(
            "mergekit is not installed in this Python environment. "
            "Provision the merge env (.sytra-envs/merge-env) before starting a merge."
        )
        return 1

    try:
        return run_real_merge(config, config_path)
    except Exception as exc:
        telemetry.emit_error(f"Merge failed: {exc}", traceback.format_exc())
        return 1


def main() -> int:
    if len(sys.argv) < 2:
        telemetry.emit_error("usage: python -m sytra_runner.merge <merge.yaml>")
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
