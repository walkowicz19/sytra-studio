"""Unsloth QLoRA training backend (Phase 7).

Supports SFT, DPO, ORPO, and CPO training modes.
Lazy-loads heavy ML libraries to avoid startup delays and timeouts.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import importlib.metadata
from typing import Any

from .. import telemetry
from ..xet_safety import apply_runtime_safety
from ..config import RunConfig
from ..adapters import resolve_training_source
from ..performance.hardware import detect_hardware
from ..performance.planner import training_plan, apply_cpu_threads
from ..performance.moe import detect_moe
from ..performance.metrics import TokenMeter, instrument_trainer

apply_runtime_safety()


_ML_PACKAGES = (
    "torch",
    "transformers",
    "unsloth",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
    "accelerate",
)


def _dependency_versions() -> dict[str, str]:
    """Return the exact training stack for telemetry and run provenance."""
    versions: dict[str, str] = {"python": sys.version.split()[0], "sytra": "1.3.1"}
    for package in _ML_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _matched_lora_modules(model: Any, targets: list[str]) -> list[str]:
    """List exact module names selected by the requested LoRA suffixes."""
    target_set = set(targets)
    return sorted(
        name
        for name, _module in model.named_modules()
        if name.rsplit(".", 1)[-1] in target_set
    )


def _verify_rslora(model: Any, requested: bool) -> None:
    """Do not silently train with a different adapter scaling rule."""
    configs = getattr(model, "peft_config", None)
    if requested and (not isinstance(configs, dict) or not configs or
                      any(getattr(config, "use_rslora", None) is not True for config in configs.values())):
        raise RuntimeError("The loaded adapter did not enable the requested rsLoRA scaling")


def _to_prompt_completion(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert one SFT row to conversational prompt-completion format.

    TRL can then mask the prompt and compute loss only on the assistant
    completion. Flattening the conversation to a single ``text`` field would
    train the model to reproduce system and user turns as well.
    """
    raw_messages = row.get("messages")
    if raw_messages:
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in raw_messages
        ]
    elif isinstance(row.get("prompt"), list) and isinstance(row.get("completion"), list):
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in row["prompt"] + row["completion"]
        ]
    else:
        messages = [
            {"role": "user", "content": str(row.get("prompt", ""))},
            {"role": "assistant", "content": str(row.get("completion", ""))},
        ]

    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("SFT rows must end with a non-empty assistant message")
    if not messages[-1]["content"].strip():
        raise ValueError("SFT assistant completion must not be empty")

    return {"prompt": messages[:-1], "completion": [messages[-1]]}


def _assistant_masked_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    """Tokenize once with the native template and mask every assistant turn.

    Prefix boundaries are accepted only when they exactly match the full
    tokenization; templates with context-dependent boundaries fail closed.
    """
    converted = _to_prompt_completion(row)
    messages = converted["prompt"] + converted["completion"]
    if any(message["role"] not in {"system", "user", "assistant"} for message in messages):
        raise ValueError("Unsupported conversation role for assistant-only SFT")
    if any(message["role"] == "assistant" and not message["content"].strip() for message in messages):
        raise ValueError("Assistant responses must not be empty")

    def render(conversation, *, generation=False):
        result = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=generation, return_dict=False)
        if hasattr(result, "tolist"):
            result = result.tolist()
        if not isinstance(result, list) or any(type(token) is not int for token in result):
            raise ValueError("Chat template must return a flat list of token IDs")
        return result

    full = render(messages)
    if len(full) > max_length:
        raise ValueError(f"Conversation has {len(full)} tokens and exceeds max_seq_len={max_length}; refusing truncation")
    mask = [0] * len(full)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index == 0:
            raise ValueError("An assistant response requires a preceding prompt")
        previous = render(messages[:index])
        prompt = render(messages[:index], generation=True)
        through = render(messages[:index + 1])
        if (len(prompt) <= len(previous) or prompt[:len(previous)] != previous or
                full[:len(prompt)] != prompt or full[:len(through)] != through or
                through[:len(prompt)] != prompt or len(prompt) >= len(through)):
            raise ValueError("Native chat template does not expose stable assistant boundaries; refusing unsafe masking")
        mask[len(prompt):len(through)] = [1] * (len(through) - len(prompt))
    if not any(mask[1:]):
        raise ValueError("No supervised assistant tokens remain after causal shifting")
    return {"input_ids": full, "assistant_masks": mask}


def _verify_assistant_mask_collator(trainer: Any) -> None:
    """Fail before training if the installed TRL stack drops assistant masks."""
    columns = getattr(trainer.train_dataset, "column_names", [])
    if "assistant_masks" not in columns:
        raise RuntimeError("Installed TRL dropped assistant masks during dataset preparation")
    evaluation = getattr(trainer, "eval_dataset", None)
    evaluations = evaluation.values() if isinstance(evaluation, dict) else ([evaluation] if evaluation is not None else [])
    if any("assistant_masks" not in getattr(dataset, "column_names", []) for dataset in evaluations):
        raise RuntimeError("Installed TRL dropped assistant masks from evaluation data")
    if getattr(trainer.args, "remove_unused_columns", True):
        setup = getattr(trainer, "_set_signature_columns_if_needed", None)
        if not callable(setup):
            raise RuntimeError("Cannot verify whether the trainer retains assistant masks")
        setup()
        if "assistant_masks" not in (getattr(trainer, "_signature_columns", None) or []):
            raise RuntimeError("Trainer would remove assistant masks before collation")
    try:
        batch = trainer.data_collator([{"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0], "seq_lengths": [3]}])
        labels = batch["labels"].reshape(-1).tolist()
    except Exception as exc:
        raise RuntimeError(f"Cannot verify assistant-only loss masking: {exc}") from exc
    if labels != [-100, 2, -100]:
        raise RuntimeError("Installed TRL collator does not preserve assistant-only loss masking")


def run_real_training(config: RunConfig) -> int:
    """Execute real Unsloth training on CUDA."""
    source = resolve_training_source(config)
    # Heavy imports deferred to execution time. unsloth MUST come first —
    # it patches transformers/trl at import time.
    try:
        import gc
        import unsloth  # noqa: F401  (must precede transformers)
        from unsloth import FastLanguageModel

        import torch
        from datasets import load_dataset
        from transformers import TrainerCallback, TrainingArguments

    except ImportError as exc:
        telemetry.emit_error(f"Failed to import ML dependencies: {exc}", traceback.format_exc())
        return 1

    op_id = config.run_id or "00000000-0000-0000-0000-000000000000"
    max_steps = config.train.get("max_steps") or (-1 if config.train.get("epochs") else 100)
    save_every = config.train.get("save_every") or 200
    train_mode = getattr(config, "train_mode", "sft").lower()

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but CUDA backend was requested.")

    telemetry.emit_starting(op_id, {
        "protocol_version": 1,
        "op": "train",
        "backend": "cuda",
        "model": config.model,
        "total_steps": max_steps,
        "train_mode": train_mode
    })
    telemetry.emit_event("dependency_versions", _dependency_versions())

    hardware = detect_hardware(torch)
    telemetry.emit_event("hardware_profile", {**hardware.to_dict(), "cuda_version": torch.version.cuda})
    plan = training_plan(hardware, config.performance, config.train, config.optim,
                         train_mode=train_mode, packing_supported=train_mode == "sft")
    apply_cpu_threads(plan.cpu_threads, torch)
    meter = TokenMeter()
    torch.cuda.reset_peak_memory_stats()

    class TelemetryCallback(TrainerCallback):
        def __init__(self, op_id_val, max_steps_val):
            self.op_id = op_id_val

        def on_train_begin(self, args, state, control, **kwargs):
            meter.reset()
            meter.step = state.global_step

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "eval_loss" in logs:
                telemetry.emit_event("evaluation", {"step": state.global_step, "loss": logs["eval_loss"]})
            if not logs or "loss" not in logs:
                return
            telemetry.emit_metric(
                step=state.global_step,
                progress=min(state.global_step / state.max_steps, 1.0) if state.max_steps > 0 else None,
                loss=logs.get("loss"), lr=logs.get("learning_rate"), grad_norm=logs.get("grad_norm"),
                mem_used_mb=torch.cuda.memory_allocated() // 1024**2,
                gpu_memory_allocated_mb=torch.cuda.memory_allocated() // 1024**2,
                gpu_memory_reserved_mb=torch.cuda.memory_reserved() // 1024**2,
                peak_vram_mb=torch.cuda.max_memory_allocated() // 1024**2,
                reward_margin=logs.get("rewards/margins", logs.get("reward_margin")),
                kl_divergence=logs.get("kl_divergence", logs.get("rewards/kl_div")),
                **meter.snapshot(state.global_step),
            )

        def on_epoch_end(self, args, state, control, **kwargs):
            telemetry.emit_event("epoch", {"epoch": state.epoch})

    # Quantization and context are user choices, not automatic quality reductions.
    quant_bits = config.adapter.get("quant_bits")
    if quant_bits not in (None, 4, 16):
        raise ValueError("This Unsloth training path supports 4-bit or 16-bit loading")
    load_in_4bit = quant_bits == 4
    max_seq_len = config.train.get("max_seq_len", 2048)

    def _empty_device_caches() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Clear memory caches before heavy load
    _empty_device_caches()

    # 1. Load model and tokenizer via Unsloth. Downloading + loading can be
    # minutes of dead air, so wrap it in stage events and a heartbeat: the
    # UI keeps receiving lines and never mistakes the download for a hang.
    telemetry.emit_event("stage", {"stage": "loading_model"})
    with telemetry.Heartbeat(lambda: {"stage": "loading_model", "step": 0}):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=source["path"] if source else config.model,
            max_seq_length=max_seq_len,
            dtype=None,  # auto detect
            load_in_4bit=load_in_4bit,
        )

    _empty_device_caches()

    if load_in_4bit:
        is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        if not is_4bit:
            raise RuntimeError(
                "4-bit loading was requested, but the loaded model did not report "
                "is_loaded_in_4bit=True; refusing a full-precision fallback"
            )

    moe = detect_moe(model.config.to_dict())
    telemetry.emit_event("moe_profile", moe)
    checkpointing = {"unsloth": "unsloth", "standard": True, "off": False}[plan.gradient_checkpointing]
    if source:
        if not getattr(model, "peft_config", None):
            raise RuntimeError("Adapter loading returned a base model; refusing fresh LoRA training")
        FastLanguageModel.for_training(model, use_gradient_checkpointing=checkpointing)
        telemetry.emit_event("adapter_fingerprint", {
            key: source.get(key) for key in ("path", "sha256", "base_model_name_or_path", "r", "target_modules", "workflow")
        })
    else:
        if config.adapter.get("type") not in {"lora", "qlora", "dora"}:
            raise ValueError("This backend supports LoRA/QLoRA/DoRA; full and QAT training are not implemented")
        topology = config.adapter.get("moe_topology", "attention")
        if topology == "attention":
            target_modules = config.adapter.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            matched_modules = _matched_lora_modules(model, target_modules)
        else:
            from ..performance.moe_lora import resolve_moe_targets
            target_modules = resolve_moe_targets(model, moe, topology)
            matched_modules = target_modules
        if not matched_modules:
            raise RuntimeError(f"No modules matched the requested LoRA targets: {target_modules}")
        if any("vision" in name.lower() or "visual" in name.lower() for name in matched_modules):
            raise RuntimeError("LoRA targets unexpectedly matched vision modules")
        telemetry.emit_event("lora_module_matches", {"topology": topology, "targets": target_modules, "modules": matched_modules})
        model = FastLanguageModel.get_peft_model(
            model, r=config.adapter.get("rank", 16), target_modules=target_modules,
            lora_alpha=config.adapter.get("alpha", 32), lora_dropout=config.adapter.get("dropout", 0.05),
            bias="none", use_gradient_checkpointing=checkpointing, random_state=3407,
            max_seq_length=max_seq_len, use_dora=config.adapter.get("type") == "dora",
            **({"use_rslora": True} if config.adapter.get("use_rslora", False) else {}),
        )
        if topology != "attention":
            from ..performance.moe_lora import verify_moe_targets
            verify_moe_targets(model, matched_modules)
    _verify_rslora(model, config.adapter.get("use_rslora", False))
    _empty_device_caches()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("PEFT configured zero trainable parameters")
    telemetry.emit_event("trainable_parameters", {
        "trainable": trainable_params,
        "total": total_params,
    })
    telemetry.emit_event("stage", {"stage": "preparing_dataset"})

    # 3. Load canonical data. For SFT, keep prompt and completion separate so
    # TRL can apply the model's chat template and mask prompt tokens in loss.
    jsonl_path = config.train_dataset or config.data.jsonl_path
    if not jsonl_path or not os.path.exists(jsonl_path):
        raise ValueError(f"Dataset path does not exist: {jsonl_path}")

    def load_training_data(path):
        if not path or not os.path.isfile(path):
            raise ValueError(f"Dataset path does not exist: {path}")
        extension = os.path.splitext(path)[1].lower()
        if extension not in {".jsonl", ".parquet", ".json"}:
            raise ValueError(f"Unsupported dataset format: {path}")
        files = path
        if extension == ".json":
            from ..dataset_fabric import resolve_manifest
            files = resolve_manifest(path)
        result = load_dataset("parquet" if extension == ".parquet" else "json", data_files=files, split="train")
        if not len(result):
            raise ValueError(f"Dataset is empty: {path}")
        if train_mode == "sft":
            if config.algo.get("completion_only_loss") is False:
                raise ValueError("SFT requires assistant-only loss masking")
            result = result.map(lambda row: _assistant_masked_row(row, tokenizer, max_seq_len), remove_columns=result.column_names)
        return result, (files if isinstance(files, list) else [files])

    dataset, train_files = load_training_data(jsonl_path)
    if config.val_dataset:
        eval_dataset, val_files = load_training_data(config.val_dataset)
        if {os.path.realpath(file) for file in train_files} & {os.path.realpath(file) for file in val_files}:
            raise ValueError("Training and validation must use separate dataset files")
    else:
        eval_dataset = None

    # 4/5. Trainer — built version-robustly: TRL has renamed several
    # SFTConfig fields and the tokenizer kwarg across releases, so pass
    # only what the installed version actually declares.
    if train_mode == "sft":
        import dataclasses
        import inspect

        from trl import SFTConfig, SFTTrainer

        cfg_fields = {f.name for f in dataclasses.fields(SFTConfig)}
        if "completion_only_loss" not in cfg_fields:
            raise RuntimeError("Installed TRL cannot preserve completion-only loss; install a compatible runtime")
        noise = config.train.get("neftune_noise_alpha") or None
        if noise is not None and "neftune_noise_alpha" not in cfg_fields:
            raise RuntimeError("Installed TRL cannot enable NEFTune; install a compatible runtime")
        # BFD resets positions, but needs a FlashAttention implementation that isolates examples.
        attention = getattr(model.config, "_attn_implementation", "")
        packing_supported = ({"packing_strategy", "completion_only_loss", "padding_free"} <= cfg_fields
                             and attention in {"flash_attention_2", "flash_attention_3"}
                             and not config.optim.get("router_aux_coef"))
        if config.optim.get("router_aux_coef") and config.performance.get("packing", config.train.get("packing", "auto")) is True:
            raise ValueError("Router balance requires packing to be disabled so padding masks remain available")
        lengths = []
        for row in dataset.select(range(min(len(dataset), 256))):
            lengths.append(len(row["input_ids"]))
        plan = training_plan(hardware, config.performance, config.train, config.optim,
                             lengths=lengths, packing_supported=packing_supported, train_mode=train_mode)
        wanted = {
            "per_device_train_batch_size": plan.micro_batch_size,
            "gradient_accumulation_steps": plan.gradient_accumulation_steps,
            "warmup_steps": config.optim.get("warmup_steps", 20),
            "max_steps": max_steps,
            "num_train_epochs": config.train.get("epochs") or 1,
            "learning_rate": config.optim.get("learning_rate", 2e-4),
            "fp16": not torch.cuda.is_bf16_supported(),
            "bf16": torch.cuda.is_bf16_supported(),
            "logging_steps": 1,
            "output_dir": str(config.output.get("adapter_path", "./output")),
            "save_steps": save_every,
            "weight_decay": config.optim.get("weight_decay", 0.0),
            "lr_scheduler_type": config.optim.get("schedule", "cosine"),
            "seed": 3407,
            "report_to": "none",
            # Windows freeze guard: extra DataLoader processes duplicate
            # CUDA/Xet working sets and page the desktop.
            "dataloader_num_workers": plan.dataloader_workers,
            "dataloader_persistent_workers": plan.persistent_workers,
            "dataloader_prefetch_factor": plan.prefetch_factor,
            "dataloader_pin_memory": plan.pin_memory,
            # Explicitly restrict loss to the assistant completion. This is
            # supported because the dataset above is prompt-completion shaped.
            "completion_only_loss": False,
            "remove_unused_columns": False,
            "packing": plan.packing,
            "packing_strategy": "bfd",
            "gradient_checkpointing": checkpointing is not False,
            "neftune_noise_alpha": noise,
            # old and new names for the sequence-length cap
            "max_seq_length": max_seq_len,
            "max_length": max_seq_len,
        }
        if config.optim.get("router_aux_coef"):
            wanted["padding_free"] = False
        if eval_dataset is not None:
            strategy_field = next((name for name in ("eval_strategy", "evaluation_strategy") if name in cfg_fields), None)
            if not strategy_field:
                raise RuntimeError("Installed TRL cannot configure evaluation")
            wanted[strategy_field] = "steps"
            wanted["eval_steps"] = min(save_every, max_steps) if max_steps > 0 else save_every
        cfg_fields = {f.name for f in dataclasses.fields(SFTConfig)}
        training_args = SFTConfig(**{k: v for k, v in wanted.items() if k in cfg_fields})

        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "args": training_args,
            "callbacks": [TelemetryCallback(op_id, max_steps)],
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
        _verify_assistant_mask_collator(trainer)
    elif train_mode in ("dpo", "orpo", "cpo"):
        import dataclasses
        import inspect
        import trl
        if train_mode == "dpo":
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
        prefix = train_mode.upper()
        config_type = getattr(trl, prefix + "Config")
        trainer_type = getattr(trl, prefix + "Trainer")
        fields = {field.name for field in dataclasses.fields(config_type)}
        wanted = dict(
            per_device_train_batch_size=plan.micro_batch_size,
            gradient_accumulation_steps=plan.gradient_accumulation_steps,
            warmup_steps=config.optim.get("warmup_steps", 20),
            max_steps=max_steps,
            learning_rate=config.optim.get("learning_rate", 2e-4),
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=str(config.output.get("adapter_path", "./output")),
            save_steps=save_every,
            weight_decay=config.optim.get("weight_decay", 0.0),
            lr_scheduler_type=config.optim.get("schedule", "cosine"),
            seed=3407,
            report_to="none",
            dataloader_num_workers=plan.dataloader_workers,
            dataloader_pin_memory=plan.pin_memory,
            gradient_checkpointing=checkpointing is not False,
        )
        if eval_dataset is not None:
            strategy_field = next((name for name in ("eval_strategy", "evaluation_strategy") if name in fields), None)
            if not strategy_field:
                raise RuntimeError("Installed TRL cannot configure evaluation")
            wanted[strategy_field] = "steps"
            wanted["eval_steps"] = min(save_every, max_steps) if max_steps > 0 else save_every
        wanted.update(max_length=max_seq_len, max_prompt_length=max_seq_len // 2,
                      num_train_epochs=config.train.get("epochs") or 1)
        if config.algo.get("beta") is not None:
            wanted["beta"] = config.algo["beta"]
        if config.algo.get("label_smoothing") is not None:
            wanted["label_smoothing"] = config.algo["label_smoothing"]
        if train_mode == "dpo" and source:
            if not {"model_adapter_name", "ref_adapter_name"} <= fields:
                raise RuntimeError("Installed DPO runtime cannot preserve the source adapter as reference")
            model.load_adapter(source["path"], adapter_name="sytra_reference", is_trainable=False)
            wanted.update(model_adapter_name="default", ref_adapter_name="sytra_reference")
            model.set_adapter("default")
        training_args = config_type(**{key: value for key, value in wanted.items() if key in fields})
        trainer_kwargs = dict(model=model, args=training_args, train_dataset=dataset,
                              callbacks=[TelemetryCallback(op_id, max_steps)])
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset
        params = inspect.signature(trainer_type.__init__).parameters
        trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
        if train_mode == "dpo":
            trainer_kwargs["ref_model"] = None
        trainer = trainer_type(**trainer_kwargs)
    else:
        raise ValueError(f"Unsupported training mode: {train_mode}")

    router_ratio = config.optim.get("router_lr_ratio")
    if router_ratio is None and source and not config.output.get("resume_from_checkpoint") and moe.get("is_moe"):
        from ..performance.moe_lora import router_adapter_parameters
        if router_adapter_parameters(model):
            router_ratio = 0.1
            config.optim["router_lr_ratio"] = router_ratio
    if router_ratio is not None:
        if not moe.get("is_moe"):
            raise ValueError("Router learning-rate ratio requires a recognized MoE model")
        from ..performance.moe_lora import apply_router_lr_ratio
        count = apply_router_lr_ratio(trainer, model, router_ratio)
        telemetry.emit_event("router_learning_rate", {"ratio": router_ratio, "adapter_parameters": count})

    # 6. Run Training
    telemetry.emit_event("stage", {"stage": "training"})
    telemetry.emit_event("training_plan", plan.to_dict())
    from pathlib import Path
    import json
    output_dir = Path(config.output.get("adapter_path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"versions": _dependency_versions(), "hardware": hardware.to_dict(),
                  "training_plan": plan.to_dict(), "moe": moe, "adapter_source": source,
                  "trainable_parameters": trainable_params, "model": config.model,
                  "train": config.train, "optim": config.optim, "adapter": config.adapter}
    (output_dir / "sytra-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    instrument_trainer(trainer, meter)
    router_balance = None
    try:
        if config.optim.get("router_aux_coef"):
            from ..performance.moe_balance import RouterBalance
            router_balance = RouterBalance(model, moe, config.optim["router_aux_coef"])
            router_balance.attach(trainer)
            telemetry.emit_event("router_balance", {"coefficient": config.optim["router_aux_coef"], "top_k": moe.get("experts_per_token")})
        trainer.train(resume_from_checkpoint=config.output.get("resume_from_checkpoint"))
    finally:
        if router_balance is not None:
            router_balance.close()

    # 7. Save final adapter
    telemetry.emit_event("stage", {"stage": "saving_adapter"})
    adapter_path = config.output.get("adapter_path", "./output")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    telemetry.emit_done({
        "adapter_path": str(adapter_path),
        "final_loss": next((entry["loss"] for entry in reversed(trainer.state.log_history) if "loss" in entry), None),
        "steps": trainer.state.global_step,
    })
    return 0


def train(config_path: str) -> int:
    """Train dispatcher."""
    try:
        config = RunConfig.load(config_path)
    except Exception as exc:
        telemetry.emit_error(f"Failed to load run config: {exc}", traceback.format_exc())
        return 1

    if config.performance.get("training_engine") == "helix":
        try:
            from .helix_sft import run_helix_training
            return run_helix_training(config)
        except Exception as exc:
            telemetry.emit_error(f"Helix training failed: {exc}", traceback.format_exc())
            return 1

    if config.backend_kind not in ("auto", "cuda"):
        telemetry.emit_error(
            "Training requires a working CUDA stack. CPU-only simulation is not supported.",
            repr(_dependency_versions()),
        )
        return 1

    import contextlib

    has_ml_deps = False
    try:
        # Third-party training packages (notably unsloth) emit startup banners directly to
        # stdout on import; redirect them to stderr so stdout stays pure JSON telemetry lines.
        with contextlib.redirect_stdout(sys.stderr):
            import unsloth  # noqa: F401  (must precede transformers to patch it)
            import torch
            import transformers
            import trl
            import datasets
        has_ml_deps = True
    except Exception as exc:
        telemetry.emit_error(
            "CUDA training dependencies are unavailable or failed to import; "
            "refusing to simulate a training run",
            f"{exc!r} {_dependency_versions()}",
        )
        return 1

    if has_ml_deps and torch.cuda.is_available() and config.backend_kind in ("auto", "cuda"):
        try:
            return run_real_training(config)
        except Exception as exc:
            telemetry.emit_error(f"Unsloth training failed: {exc}", traceback.format_exc())
            return 1

    telemetry.emit_error(
        "Training requires a working CUDA stack. CPU-only simulation is not supported.",
        repr(_dependency_versions()),
    )
    return 1
