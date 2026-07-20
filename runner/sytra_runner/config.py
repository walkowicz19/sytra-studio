"""Loaders for run.yaml / merge.yaml (Contracts 1 and 2).

These dataclasses mirror the Rust structs in
crates/sytra-contracts/src/{run_config,merge_config}.rs field-for-field so
the two implementations can be diffed by eye and tested against the same
golden fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    "linear", "slerp", "ties", "dare_ties", "task_arithmetic",
    "passthrough", "moe",
}
TASK_VECTOR_METHODS = {"ties", "dare_ties", "task_arithmetic"}
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        _require_version(raw)
        train_mode = raw.get("train_mode")
        if train_mode not in TRAIN_MODES:
            raise ConfigError(f"unknown train_mode: {train_mode!r}")

        backend = raw.get("backend") or {}
        if backend.get("kind") not in BACKEND_KINDS:
            raise ConfigError(f"unknown backend.kind: {backend.get('kind')!r}")

        adapter = raw.get("adapter") or {}
        if adapter.get("type") not in ADAPTER_TYPES:
            raise ConfigError(f"unknown adapter.type: {adapter.get('type')!r}")

        optim = raw.get("optim") or {}
        if optim.get("schedule") not in SCHEDULES:
            raise ConfigError(f"unknown optim.schedule: {optim.get('schedule')!r}")

        return cls(
            version=raw["version"],
            run_id=raw.get("run_id"),
            train_mode=train_mode,
            model=raw["model"],
            backend_kind=backend["kind"],
            judge_model=backend.get("judge_model"),
            data=DataSpec.from_dict(raw["data"]),
            adapter=adapter,
            optim=optim,
            train=raw.get("train") or {},
            algo=raw.get("algo") or {},
            output=raw.get("output") or {},
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
        if len(models) > 3:
            raise ConfigError(f"model count {len(models)} exceeds the product cap of 3")
        if merge_method == "slerp" and len(models) > 2:
            raise ConfigError(f"slerp accepts at most 2 models, got {len(models)}")

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
        )

    @classmethod
    def load(cls, path: str | Path) -> "MergeConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
