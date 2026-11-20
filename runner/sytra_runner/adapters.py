"""Validate adapter initialization and full Trainer resumption before model loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import ConfigError


def adapter_metadata(path: str | Path) -> dict:
    directory = Path(path).expanduser().resolve()
    try:
        raw = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read adapter_config.json in {directory}: {exc}") from exc
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ConfigError(f"Adapter requires non-empty adapter_model.safetensors: {directory}")
    if raw.get("peft_type") != "LORA" or raw.get("task_type") not in (None, "CAUSAL_LM"):
        raise ConfigError("Continuation requires a causal-LM LoRA/DoRA adapter")
    if not raw.get("base_model_name_or_path") or not raw.get("r") or not raw.get("target_modules"):
        raise ConfigError("Adapter metadata must include base_model_name_or_path, r, and target_modules")
    digest = hashlib.sha256()
    for file in sorted([weights, *directory.glob("*.json"), *directory.glob("*.model")]):
        digest.update(file.name.encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {**raw, "path": str(directory), "sha256": digest.hexdigest()}


def resolve_training_source(config) -> dict | None:
    init = config.init_adapter
    resume = config.output.get("resume_from_checkpoint")
    model_path = Path(config.model).expanduser()
    local_adapter = model_path.is_dir() and (model_path / "adapter_config.json").exists()
    if init and resume:
        raise ConfigError("Use init_adapter for new training OR output.resume_from_checkpoint for full resumption")
    if init and local_adapter and Path(init).expanduser().resolve() != model_path.resolve():
        raise ConfigError("model and init_adapter refer to different adapters")
    source = resume or init or (str(model_path) if local_adapter else None)
    if not source:
        return None
    info = adapter_metadata(source)
    base = info["base_model_name_or_path"]
    if not local_adapter and config.model != base:
        if not (Path(config.model).is_dir() and Path(base).is_dir() and Path(config.model).resolve() == Path(base).resolve()):
            raise ConfigError(f"Adapter base {base!r} does not match model {config.model!r}")
    if config.adapter.get("type") in {"full", "qat"}:
        raise ConfigError("Existing LoRA adapters cannot initialize full/QAT training")
    if resume:
        directory = Path(source).expanduser().resolve()
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise ConfigError(f"Full resume is missing {missing}; use init_adapter for weights-only continuation")
        try:
            state = json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"Invalid trainer_state.json: {exc}") from exc
        if type(state.get("global_step")) is not int or state["global_step"] < 0:
            raise ConfigError("Checkpoint has no valid global_step")
        if config.train.get("max_steps") and config.train["max_steps"] <= state["global_step"]:
            raise ConfigError("For resume, max_steps is the total target and must exceed checkpoint global_step")
        provenance_path = directory.parent / "sytra-provenance.json"
        saved_ratio = None
        saved_aux = None
        if provenance_path.is_file():
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                saved_ratio = provenance.get("optim", {}).get("router_lr_ratio")
                saved_aux = provenance.get("optim", {}).get("router_aux_coef")
            except (OSError, ValueError, AttributeError) as exc:
                raise ConfigError(f"Invalid source provenance: {exc}") from exc
        if saved_ratio is not None and (type(saved_ratio) not in (int, float) or not 0 < saved_ratio <= 1):
            raise ConfigError("Invalid saved router LR ratio")
        requested_ratio = config.optim.get("router_lr_ratio")
        if requested_ratio is not None and requested_ratio != saved_ratio:
            raise ConfigError("Resume must keep the saved router LR ratio and optimizer groups")
        config.optim["router_lr_ratio"] = saved_ratio
        if saved_aux is not None and (type(saved_aux) not in (int, float) or not 0 <= saved_aux <= 1):
            raise ConfigError("Invalid saved router auxiliary coefficient")
        requested_aux = config.optim.get("router_aux_coef")
        if requested_aux is not None and requested_aux != saved_aux:
            raise ConfigError("Resume must keep the saved router auxiliary coefficient")
        config.optim["router_aux_coef"] = saved_aux
        config.output["resume_from_checkpoint"] = str(directory)
    else:
        output = Path(config.output.get("adapter_path", "./output")).expanduser().resolve()
        source_path = Path(info["path"])
        if output == source_path or output in source_path.parents or source_path in output.parents:
            raise ConfigError("Adapter continuation requires a separate output directory outside the source run")
    # The saved adapter defines its structure; never reinitialize it with form defaults.
    if type(info.get("use_rslora", False)) is not bool:
        raise ConfigError("Adapter use_rslora metadata must be boolean")
    config.adapter.update(rank=info["r"], alpha=info.get("lora_alpha", info["r"]), use_rslora=info.get("use_rslora", False),
                          dropout=info.get("lora_dropout", 0), target_modules=info["target_modules"])
    info["workflow"] = "resume" if resume else "continue_adapter"
    return info
