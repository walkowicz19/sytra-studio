"""Loaders for run.yaml / merge.yaml (Contracts 1 and 2).

These dataclasses mirror the Rust structs in
crates/sytra-contracts/src/{run_config,merge_config}.rs field-for-field so
the two implementations can be diffed by eye and tested against the same
golden fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_VERSION = 1

TRAIN_MODES = {
    "sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo",
    "rlhf_reinforce", "ppo",
}
DATA_SOURCES = {"hf", "local", "synthetic", "klayer", "multi"}
ADAPTER_TYPES = {"lora", "dora", "qlora", "full", "qat"}
SCHEDULES = {"cosine", "linear", "constant"}
BACKEND_KINDS = {"auto", "cuda", "rocm", "mps", "cpu"}

MERGE_METHODS = {
    "linear", "slerp", "multi_slerp", "nuslerp",
    "ties", "dare_ties", "dare_linear", "task_arithmetic",
    "passthrough", "moe",
}
TASK_VECTOR_METHODS = {"ties", "dare_ties", "dare_linear", "task_arithmetic"}
VERDICTS = {"green", "amber", "red"}


class ConfigError(ValueError):
    """Raised when a run.yaml / merge.yaml fails to validate."""


def _require_version(raw: dict[str, Any]) -> int:
    version = raw.get("version")
    if version != SUPPORTED_VERSION:
        raise ConfigError(
            f"unsupported config version {version!r}; runner only accepts {SUPPORTED_VERSION}"
        )
    return version


@dataclass
class DataSpec:
    source: str
    jsonl_path: str | None
    fingerprint: str | None
    params: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataSpec":
        source = raw.get("source")
        if source not in DATA_SOURCES:
            raise ConfigError(f"unknown data.source: {source!r}")
        params = raw.get(source) or {}
        return cls(
            source=source,
            jsonl_path=raw.get("jsonl_path"),
            fingerprint=raw.get("fingerprint"),
            params=params,
        )


@dataclass
class RunConfig:
    version: int
    run_id: str | None
    train_mode: str
    model: str
    backend_kind: str
    judge_model: str | None
    data: DataSpec
    adapter: dict[str, Any]
    optim: dict[str, Any]
    train: dict[str, Any]
    algo: dict[str, Any]
    output: dict[str, Any]
    downloader: dict[str, Any] = field(default_factory=dict)
    init_adapter: str | None = None
    performance: dict[str, Any] = field(default_factory=dict)
    train_dataset: str | None = None
    val_dataset: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        _require_version(raw)
        train_mode = raw.get("train_mode")
        if train_mode not in TRAIN_MODES:
            raise ConfigError(f"unknown train_mode: {train_mode!r}")

        backend = raw.get("backend") or {}
        if backend.get("kind") not in BACKEND_KINDS:
            raise ConfigError(f"unknown backend.kind: {backend.get('kind')!r}")

        adapter = dict(raw.get("adapter") or {})
        if adapter.get("type") not in ADAPTER_TYPES:
            raise ConfigError(f"unknown adapter.type: {adapter.get('type')!r}")
        if type(adapter.get("use_rslora", False)) is not bool:
            raise ConfigError("adapter.use_rslora must be a boolean")
        if adapter.get("use_rslora") and adapter.get("type") not in {"lora", "qlora", "dora"}:
            raise ConfigError("rsLoRA requires a LoRA adapter")
        if adapter.get("moe_topology", "attention") not in {"attention", "moe_hybrid", "moe_full"}:
            raise ConfigError("Unknown MoE LoRA topology")

        optim = dict(raw.get("optim") or {})
        ratio = optim.get("router_lr_ratio")
        if ratio is not None and (type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1):
            raise ConfigError("optim.router_lr_ratio must be greater than 0 and at most 1")
        aux = optim.get("router_aux_coef")
        if aux is not None and (type(aux) not in (int, float) or not math.isfinite(aux) or not 0 <= aux <= 1):
            raise ConfigError("optim.router_aux_coef must be between 0 and 1")
        if aux and train_mode != "sft":
            raise ConfigError("Router auxiliary loss is only supported for SFT")
        train = dict(raw.get("train") or {})
        noise = train.get("neftune_noise_alpha")
        if noise is not None and (type(noise) not in (int, float) or not math.isfinite(noise) or not 0 <= noise <= 100):
            raise ConfigError("train.neftune_noise_alpha must be between 0 and 100")
        if noise and train_mode != "sft":
            raise ConfigError("NEFTune is only supported for SFT")
        accumulation = [block[key] for block, key in (
            (optim, "grad_accumulation_steps"), (optim, "gradient_accumulation_steps"),
            (train, "gradient_accumulation_steps"), (train, "grad_accumulation_steps"),
        ) if key in block]
        if accumulation and any(value != accumulation[0] for value in accumulation):
            raise ConfigError("Conflicting gradient accumulation aliases")
        if accumulation:
            optim["grad_accumulation_steps"] = accumulation[0]
        optim.pop("gradient_accumulation_steps", None)
        train.pop("gradient_accumulation_steps", None)
        train.pop("grad_accumulation_steps", None)
        for name, value in (("batch_size", train.get("batch_size", 2)),
                            ("max_seq_len", train.get("max_seq_len", 2048)),
                            ("grad_accumulation_steps", optim.get("grad_accumulation_steps", 8))):
            if type(value) is not int or value <= 0:
                raise ConfigError(f"{name} must be a positive integer")
        if train.get("packing", "auto") != "auto" and not isinstance(train.get("packing"), bool):
            raise ConfigError("train.packing must be auto, true, or false")
        output = dict(raw.get("output") or {})
        legacy = output.pop("resume_from", None)
        if legacy and output.get("resume_from_checkpoint") and legacy != output["resume_from_checkpoint"]:
            raise ConfigError("Conflicting resume_from and resume_from_checkpoint")
        if legacy:
            output["resume_from_checkpoint"] = legacy
        for name, value in (("init_adapter", raw.get("init_adapter")),
                            ("resume_from_checkpoint", output.get("resume_from_checkpoint")),
                            ("resume_from", legacy)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"{name} must be a non-empty path or null")
        for name in ("max_steps", "epochs", "save_every"):
            if train.get(name) is not None and (type(train[name]) is not int or train[name] <= 0):
                raise ConfigError(f"train.{name} must be positive or null")
        if raw.get("init_adapter") and output.get("resume_from_checkpoint"):
            raise ConfigError("init_adapter and resume_from_checkpoint are mutually exclusive")
        from .performance.planner import validate_performance
        try:
            performance = validate_performance(raw.get("performance") or {})
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if optim.get("schedule") not in SCHEDULES:
            raise ConfigError(f"unknown optim.schedule: {optim.get('schedule')!r}")

        train_dataset = raw.get("train_dataset")
        val_dataset = raw.get("val_dataset")
        for name, value in (("train_dataset", train_dataset), ("val_dataset", val_dataset)):
            extensions = {".jsonl", ".parquet", ".json"} if name == "train_dataset" else {".jsonl", ".parquet"}
            if value is not None and (not isinstance(value, str) or not value.strip() or Path(value).suffix.lower() not in extensions):
                raise ConfigError(f"{name} has an unsupported dataset format")
        data = raw.get("data")
        if data is None and train_dataset:
            data = {"source": "local", "jsonl_path": train_dataset, "local": {"path": train_dataset}}
        if not isinstance(data, dict):
            raise ConfigError("A training dataset is required")
        if data.get("source") == "multi":
            raise ConfigError("Training accepts one dataset; combine sources first")
        effective_train = train_dataset or data.get("jsonl_path")
        if effective_train and val_dataset and Path(effective_train).resolve() == Path(val_dataset).resolve():
            raise ConfigError("Training and validation must use different files")

        return cls(
            version=raw["version"],
            run_id=raw.get("run_id"),
            train_mode=train_mode,
            model=raw["model"],
            backend_kind=backend["kind"],
            judge_model=backend.get("judge_model"),
            data=DataSpec.from_dict(data),
            adapter=adapter,
            optim=optim,
            train=train,
            algo=raw.get("algo") or {},
            output=output,
            downloader=raw.get("downloader") or {},
            init_adapter=raw.get("init_adapter"),
            performance=performance,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)


@dataclass
class MergeConfig:
    version: int
    op_id: str | None
    merge_method: str
    base_model: str | None
    dtype: str
    models: list[dict[str, Any]]
    tokenizer: dict[str, Any]
    compat_verdict: str
    compat_fingerprint: str | None
    output: dict[str, Any]
    performance: dict[str, Any] = field(default_factory=dict)
    moe: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MergeConfig":
        _require_version(raw)
        merge_method = raw.get("merge_method")
        if merge_method not in MERGE_METHODS:
            raise ConfigError(f"unknown merge_method: {merge_method!r}")

        compat = raw.get("compat") or {}
        verdict = compat.get("verdict")
        if verdict not in VERDICTS:
            raise ConfigError(f"unknown compat.verdict: {verdict!r}")
        if verdict == "red":
            raise ConfigError("compat.verdict is red; refusing to run merge")

        if merge_method in TASK_VECTOR_METHODS and not raw.get("base_model"):
            raise ConfigError(f"base_model is required for merge_method={merge_method!r}")

        models = raw.get("models") or []
        cap = 16 if merge_method == "moe" else 3
        if len(models) > cap:
            raise ConfigError(f"model count {len(models)} exceeds the product cap of {cap}")
        if merge_method == "slerp" and len(models) > 2:
            raise ConfigError(f"slerp accepts at most 2 models, got {len(models)}")
        if merge_method in {"slerp", "multi_slerp", "nuslerp"} and len(models) < 2:
            raise ConfigError(f"{merge_method} requires at least 2 models, got {len(models)}")
        moe = raw.get("moe")
        if merge_method == "moe":
            if not isinstance(moe, dict) or not isinstance(raw.get("base_model"), str) or not raw["base_model"].strip() or len(models) < 2:
                raise ConfigError("MoE requires a base model, options and at least two experts")
            if moe.get("architecture") not in {"mixtral", "deepseek", "qwen", "qwen3"}:
                raise ConfigError("Unsupported MoE output architecture")
            top_k = moe.get("experts_per_token")
            if type(top_k) is not int or not 1 <= top_k <= len(models):
                raise ConfigError("experts_per_token must be between 1 and the number of experts")
            for expert in models:
                if not isinstance(expert, dict) or not isinstance(expert.get("model"), str) or not expert["model"].strip():
                    raise ConfigError("Each expert needs a model")
                for field in ("positive_prompts", "negative_prompts"):
                    prompts = expert.get(field, [])
                    if not isinstance(prompts, list) or any(not isinstance(p, str) or not p.strip() for p in prompts) or (field == "positive_prompts" and not prompts):
                        raise ConfigError(f"Each expert needs valid {field}")
            prompt_sets = {(tuple(e["positive_prompts"]), tuple(e.get("negative_prompts", []))) for e in models}
            if len(prompt_sets) < 2 or len({e["model"] for e in models}) < 2:
                raise ConfigError("Experts must have distinct models and routing prompts")
            shared = moe.get("shared_experts", [])
            if not isinstance(shared, list) or len(shared) > 16 or any(not isinstance(e, dict) or not isinstance(e.get("source_model"), str) or not e["source_model"].strip() for e in shared):
                raise ConfigError("Invalid shared experts")
            if any(not isinstance(e.get(field, []), list) or any(not isinstance(p, str) or not p.strip() for p in e.get(field, [])) for e in shared for field in ("positive_prompts", "negative_prompts")):
                raise ConfigError("Shared expert prompts must be non-empty strings")
            if shared and any(bool(e.get("positive_prompts")) != bool(shared[0].get("positive_prompts")) for e in shared):
                raise ConfigError("Shared experts must either all have positive prompts or all omit them")
            architecture = moe['architecture']
            if architecture == 'mixtral' and shared:
                raise ConfigError('Mixtral output does not support shared experts')
            if architecture == 'deepseek' and (len(shared) > 1 or any(e.get('positive_prompts') or e.get('negative_prompts') for e in shared)):
                raise ConfigError('DeepSeek output supports at most one shared expert without routing prompts')
            if architecture == 'qwen' and (len(shared) != 1 or not shared[0].get('positive_prompts')):
                raise ConfigError('Qwen MoE output requires one shared expert with positive routing prompts')

        from .performance.planner import validate_performance
        return cls(
            version=raw["version"],
            op_id=raw.get("op_id"),
            merge_method=merge_method,
            base_model=raw.get("base_model"),
            dtype=raw["dtype"],
            models=models,
            tokenizer=raw.get("tokenizer") or {},
            compat_verdict=verdict,
            compat_fingerprint=compat.get("fingerprint"),
            output=raw.get("output") or {},
            performance=validate_performance(raw.get("performance") or {}),
            moe=moe,
            parameters=raw.get("parameters"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MergeConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
