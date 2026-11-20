from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from statistics import median

from .hardware import HardwareProfile


def validate_performance(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("performance must be an object")
    allowed = {"mode", "packing", "dataloader_workers", "pin_memory", "prefetch_factor",
               "gradient_checkpointing", "merge_cuda", "merge_low_cpu_memory", "merge_shard_size",
               "adapter_materialization_device", "cpu_threads", "training_engine", "activation_memory_mb", "helix_prefetch", "prefetch_pinned_memory_mb"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unsupported performance settings: {sorted(unknown)}")
    if raw.get("mode", "auto") not in {"auto", "safe", "fast", "custom"}:
        raise ValueError("performance.mode must be auto, safe, fast, or custom")
    if raw.get("training_engine", "auto") not in {"auto", "helix"}:
        raise ValueError("performance.training_engine must be auto or helix")
    if 'activation_memory_mb' in raw and (type(raw['activation_memory_mb']) is not int or not 0 <= raw['activation_memory_mb'] <= 65536):
        raise ValueError('performance.activation_memory_mb must be an integer between 0 and 65536')
    if 'activation_memory_mb' in raw and raw.get('training_engine', 'auto') != 'helix':
        raise ValueError('performance.activation_memory_mb requires the Helix training engine')
    if 'helix_prefetch' in raw and (type(raw['helix_prefetch']) is not bool or raw.get('training_engine', 'auto') != 'helix'):
        raise ValueError('performance.helix_prefetch must be a boolean and requires the Helix training engine')
    if 'prefetch_pinned_memory_mb' in raw and (type(raw['prefetch_pinned_memory_mb']) is not int or
            not 0 <= raw['prefetch_pinned_memory_mb'] <= 4096 or raw.get('helix_prefetch') is not True or
            raw.get('training_engine') != 'helix'):
        raise ValueError('performance.prefetch_pinned_memory_mb requires Helix prefetch and an integer between 0 and 4096')
    for key in ("packing", "pin_memory", "merge_cuda", "merge_low_cpu_memory"):
        value = raw.get(key, "auto")
        if value != "auto" and not isinstance(value, bool):
            raise ValueError(f"performance.{key} must be auto, true, or false")
    for key in ("dataloader_workers", "prefetch_factor", "cpu_threads"):
        value = raw.get(key, "auto")
        if value != "auto" and (type(value) is not int or value < (0 if key == "dataloader_workers" else 1)):
            raise ValueError(f"Invalid performance.{key}: {value!r}")
    for key, choices in {
        "gradient_checkpointing": {"auto", "unsloth", "standard", "off"},
        "adapter_materialization_device": {"auto", "cpu", "cuda"},
        "merge_shard_size": {"auto", "1B", "2B", "4B", "5B"},
    }.items():
        if raw.get(key, "auto") not in choices:
            raise ValueError(f"Invalid performance.{key}")
    return dict(raw)


def cpu_threads(hw: HardwareProfile, options: dict) -> int:
    cores = hw.cpu_physical_cores or max(1, hw.cpu_logical_cores // 2)
    mode = options.get("mode", "auto")
    default = min(4, cores) if mode == "safe" else max(1, cores - 1) if mode == "fast" else max(1, min(cores - 2, cores * 3 // 4))
    requested = options.get("cpu_threads", "auto")
    return min(hw.cpu_logical_cores, default if requested == "auto" else requested)


def apply_cpu_threads(threads: int, torch_module=None) -> dict[str, str]:
    env = {key: str(threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
    os.environ.update(env)
    if torch_module is not None:
        torch_module.set_num_threads(threads)
        try:
            torch_module.set_num_interop_threads(min(2, threads))
        except RuntimeError:
            pass  # Torch permits setting interop threads only before parallel work starts.
    return env


@dataclass(frozen=True)
class TrainingPlan:
    mode: str
    cpu_threads: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: str
    dataloader_workers: int
    pin_memory: bool
    prefetch_factor: int | None
    persistent_workers: bool
    packing: bool
    packing_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def training_plan(hw: HardwareProfile, options: dict, train: dict, optim: dict, *, lengths=(), packing_supported=False, train_mode="sft") -> TrainingPlan:
    options = validate_performance(options)
    mode = options.get("mode", "auto")
    batch, accum = train.get("batch_size", 2), optim.get("grad_accumulation_steps", 8)
    micro = 1 if mode == "safe" or 0 < hw.vram_total_gb <= 16 else batch
    accum = batch * accum // micro  # Preserve the entire effective batch, including accumulation.
    low_ram = hw.ram_total_gb <= 16.5 or hw.ram_available_gb < 4
    workers = 0 if mode == "safe" or low_ram else min(2 if hw.ram_total_gb <= 32.5 else 4, max(1, (hw.cpu_physical_cores or hw.cpu_logical_cores) // 4))
    if options.get("dataloader_workers", "auto") != "auto":
        workers = options["dataloader_workers"]
    pin = hw.cuda_available and not low_ram and mode != "safe"
    if options.get("pin_memory", "auto") != "auto":
        pin = options["pin_memory"] and hw.cuda_available
    prefetch = options.get("prefetch_factor", "auto")
    checkpointing = options.get("gradient_checkpointing", "auto")
    if checkpointing == "auto":
        checkpointing = "unsloth"
    requested = options.get("packing", train.get("packing", "auto"))
    packing, reason = False, "disabled by request"
    if requested is True and (train_mode != "sft" or not packing_supported):
        raise ValueError("Packing requires SFT with completion-mask and boundary-aware packing support in the installed TRL runtime")
    if requested is True:
        packing, reason = True, "explicit request"
    elif requested == "auto":
        packing = bool(train_mode == "sft" and packing_supported and mode != "safe" and lengths and median(lengths) < 0.6 * train.get("max_seq_len", 2048))
        reason = "short sampled sequences; boundary-aware packing supported" if packing else "safe profile, unavailable boundary support, or no measured benefit"
    return TrainingPlan(mode, cpu_threads(hw, options), micro, accum, checkpointing, workers, pin,
                        (2 if prefetch == "auto" else prefetch) if workers else None, workers > 0, packing, reason)


@dataclass(frozen=True)
class MergePlan:
    mode: str
    cuda: bool
    low_cpu_memory: bool
    cpu_threads: int
    out_shard_size: str
    adapter_materialization_device: str

    def to_dict(self) -> dict:
        return asdict(self)

    def flags(self) -> list[str]:
        return ["--lazy-unpickle", *(["--cuda"] if self.cuda else []),
                *(["--low-cpu-memory"] if self.low_cpu_memory else []), "--out-shard-size", self.out_shard_size]


def merge_plan(hw: HardwareProfile, options: dict, *, estimated_ram_gb: float | None = None, adapter_memory_gb: float | None = None) -> MergePlan:
    options = validate_performance(options)
    mode = options.get("mode", "auto")
    cuda = hw.cuda_available and hw.vram_free_gb >= 8 and mode != "safe"
    if options.get("merge_cuda", "auto") != "auto":
        cuda = options["merge_cuda"]
    if cuda and not hw.cuda_available:
        raise ValueError("CUDA merge requested but CUDA is unavailable")
    low = mode == "safe" or estimated_ram_gb is None or estimated_ram_gb > hw.ram_available_gb * 0.80
    if options.get("merge_low_cpu_memory", "auto") != "auto":
        low = options["merge_low_cpu_memory"]
    shard = options.get("merge_shard_size", "auto")
    if shard == "auto":
        shard = "1B" if mode == "safe" or hw.ram_available_gb < 16 else "4B" if hw.storage_kind == "nvme" else "2B" if hw.storage_kind == "ssd" else "1B"
    device = options.get("adapter_materialization_device", "auto")
    fits = hw.cuda_available and adapter_memory_gb is not None and adapter_memory_gb <= max(0, hw.vram_free_gb * .85 - 1)
    if device == "auto":
        device = "cuda" if fits and mode != "safe" else "cpu"
    if device == "cuda" and not fits:
        raise ValueError("Adapter materialization does not fit verified free VRAM; use cpu or auto")
    return MergePlan(mode, cuda, low, cpu_threads(hw, options), shard, device)
