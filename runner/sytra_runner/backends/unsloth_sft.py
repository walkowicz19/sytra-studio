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
from ..config import RunConfig


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
    versions: dict[str, str] = {"python": sys.version.split()[0]}
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


def run_real_training(config: RunConfig) -> int:
    """Execute real Unsloth training on CUDA."""
    # Heavy imports deferred to execution time. unsloth MUST come first —
    # it patches transformers/trl at import time.
    try:
        import unsloth  # noqa: F401  (must precede transformers)
        from unsloth import FastLanguageModel

        import torch
        from datasets import load_dataset
        from transformers import TrainerCallback, TrainingArguments
    except ImportError as exc:
        telemetry.emit_error(f"Failed to import ML dependencies: {exc}", traceback.format_exc())
        return 1

    op_id = config.run_id or "00000000-0000-0000-0000-000000000000"
    max_steps = config.train.get("max_steps") or 100
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

    class TelemetryCallback(TrainerCallback):
        """Custom transformers callback to pipe training progress to stdout telemetry."""
        def __init__(self, op_id_val: str, max_steps_val: int):
            self.op_id = op_id_val
            self.max_steps = max_steps_val
            self.last_time = time.time()
            self.last_steps = 0

        def on_log(self, args_val: Any, state_val: Any, control_val: Any, logs_val: dict[str, Any] = None, **kwargs_val: Any) -> None:
            if not logs_val:
                return
            
            current_time = time.time()
            step = state_val.global_step
            
            step_diff = step - self.last_steps
            time_diff = current_time - self.last_time
            tokens_s = 0.0
            if time_diff > 0 and step_diff > 0:
                tokens_s = (step_diff * args_val.per_device_train_batch_size * 1024) / time_diff
                
            self.last_time = current_time
            self.last_steps = step

            mem_used_mb = 0
            if torch.cuda.is_available():
                mem_used_mb = torch.cuda.memory_allocated() // (1024 * 1024)

            # Capture preference alignment metrics if present (DPO / ORPO / CPO)
            reward_margin = logs_val.get("rewards/margins")
            if reward_margin is None:
                reward_margin = logs_val.get("reward_margin")
            
            kl_div = logs_val.get("kl_divergence")
            if kl_div is None:
                kl_div = logs_val.get("rewards/kl_div")

            telemetry.emit_metric(
                step=step,
                progress=round(min(step / self.max_steps, 1.0), 4) if self.max_steps else None,
                loss=round(logs_val.get("loss", 0.0), 4),
                lr=logs_val.get("learning_rate", 0.0),
                grad_norm=round(logs_val.get("grad_norm", 0.0), 4) if "grad_norm" in logs_val else None,
                tokens_s=int(tokens_s),
                mem_used_mb=mem_used_mb,
                reward_margin=round(reward_margin, 4) if reward_margin is not None else None,
                kl_divergence=round(kl_div, 4) if kl_div is not None else None,
            )

        def on_epoch_end(self, args_val: Any, state_val: Any, control_val: Any, **kwargs_val: Any) -> None:
            telemetry.emit_event("epoch", {"epoch": int(state_val.epoch or 1)})

    # ── VRAM-aware configuration ───────────────────────────────────────────
    # Detect GPU memory and automatically pick the safest settings that still
    # maximise speed. On cards with ≤16 GB we enforce 4-bit + Unsloth's
    # gradient checkpointing which cuts activation memory by ~70% with only
    # ~5–10% throughput cost (much better than OOM).
    vram_total_mb = 0
    vram_free_mb = 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_total_mb = props.total_memory // (1024 * 1024)
        vram_free_mb  = (props.total_memory - torch.cuda.memory_allocated()) // (1024 * 1024)
        telemetry.emit_event("vram_info", {"total_mb": vram_total_mb, "free_mb": vram_free_mb})

    low_vram = vram_total_mb > 0 and vram_total_mb <= 16384  # ≤16 GB

    # Force 4-bit on low VRAM; respect user override otherwise
    quant_bits = config.adapter.get("quant_bits")
    if low_vram and quant_bits not in (4,):
        quant_bits = 4
        telemetry.emit_event("vram_override", {"reason": "low_vram", "quant_bits": 4})
    load_in_4bit = quant_bits == 4

    # Cap max_seq_len to 2048 on ≤12 GB to preserve training VRAM headroom.
    # Users with more VRAM can override via config.
    max_seq_len = config.train.get("max_seq_len", 2048)
    if low_vram and vram_total_mb <= 12288 and max_seq_len > 2048:
        max_seq_len = 2048
        telemetry.emit_event("vram_override", {"reason": "low_vram", "max_seq_len": 2048})

    # 1. Load model and tokenizer via Unsloth. Downloading + loading can be
    # minutes of dead air, so wrap it in stage events and a heartbeat: the
    # UI keeps receiving lines and never mistakes the download for a hang.
    telemetry.emit_event("stage", {"stage": "loading_model"})
    with telemetry.Heartbeat(lambda: {"stage": "loading_model", "step": 0}):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model,
            max_seq_length=max_seq_len,
            dtype=None,  # auto detect
            load_in_4bit=load_in_4bit,
        )

    if load_in_4bit:
        is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        if not is_4bit:
            raise RuntimeError(
                "4-bit loading was requested, but the loaded model did not report "
                "is_loaded_in_4bit=True; refusing a full-precision fallback"
            )

    target_modules = config.adapter.get(
        "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    matched_modules = _matched_lora_modules(model, target_modules)
    if not matched_modules:
        raise RuntimeError(
            f"No modules matched the requested LoRA targets: {target_modules}"
        )
    unexpected_vision = [
        name for name in matched_modules
        if any(part in name.lower() for part in ("vision", "visual"))
    ]
    if unexpected_vision:
        raise RuntimeError(
            "LoRA targets unexpectedly matched vision modules: "
            + ", ".join(unexpected_vision)
        )
    telemetry.emit_event("lora_module_matches", {
        "targets": target_modules,
        "count": len(matched_modules),
        "modules": matched_modules,
    })

    # 2. Configure PEFT adapter — always use Unsloth's gradient checkpointing
    # ("unsloth" mode stores only 1/sqrt(n) activations, saving ~70% VRAM
    # compared to standard gradient checkpointing at the cost of one extra
    # forward pass per backward, ~5–10% slower but never OOM).
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.adapter.get("rank", 16),
        target_modules=target_modules,
        lora_alpha=config.adapter.get("alpha", 32),
        lora_dropout=config.adapter.get("dropout", 0.05),
        bias="none",
        use_gradient_checkpointing="unsloth",  # best VRAM/speed tradeoff
        random_state=3407,
        max_seq_length=max_seq_len,
    )
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
    jsonl_path = config.data.jsonl_path
    if not jsonl_path or not os.path.exists(jsonl_path):
        raise ValueError(f"Dataset path does not exist: {jsonl_path}")

    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    if train_mode == "sft":
        dataset = dataset.map(_to_prompt_completion)

    # 4/5. Trainer — built version-robustly: TRL has renamed several
    # SFTConfig fields and the tokenizer kwarg across releases, so pass
    # only what the installed version actually declares.
    if train_mode == "sft":
        import dataclasses
        import inspect

        from trl import SFTConfig, SFTTrainer

        # Auto batch size: start from user config, but cap at 1 on low VRAM.
        # We compensate with gradient_accumulation_steps so effective batch
        # size stays the same (e.g. batch=1, accum=8 ≡ batch=8).
        requested_batch = config.train.get("batch_size", 2)
        auto_batch = 1 if (low_vram and requested_batch > 1) else requested_batch
        # Adjust accumulation to preserve effective batch size
        base_accum = config.optim.get("grad_accumulation_steps", 8)
        if auto_batch < requested_batch and requested_batch > 0:
            base_accum = max(base_accum, requested_batch // auto_batch)

        wanted = {
            "per_device_train_batch_size": auto_batch,
            "gradient_accumulation_steps": base_accum,
            "warmup_steps": config.optim.get("warmup_steps", 20),
            "max_steps": max_steps,
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
            # Explicitly restrict loss to the assistant completion. This is
            # supported because the dataset above is prompt-completion shaped.
            "completion_only_loss": True,
            "packing": config.train.get("packing", False),
            # old and new names for the sequence-length cap
            "max_seq_length": config.train.get("max_seq_len", 2048),
            "max_length": config.train.get("max_seq_len", 2048),
        }
        cfg_fields = {f.name for f in dataclasses.fields(SFTConfig)}
        training_args = SFTConfig(**{k: v for k, v in wanted.items() if k in cfg_fields})

        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "args": training_args,
            "callbacks": [TelemetryCallback(op_id, max_steps)],
        }
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)
    elif train_mode in ("dpo", "orpo", "cpo"):
        # Preference-mode branches keep the plain TrainingArguments they
        # were written against; only touched when those modes get the same
        # version-robust treatment as sft.
        training_args = TrainingArguments(
            per_device_train_batch_size=config.train.get("batch_size", 2),
            gradient_accumulation_steps=config.optim.get("grad_accumulation_steps", 8),
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
        )
        if train_mode == "dpo":
            from trl import DPOTrainer
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()  # patch DPO for fast training

            trainer = DPOTrainer(
                model=model,
                ref_model=None,  # reference model uses base model weights internally to save VRAM
                beta=0.1,
                train_dataset=dataset,
                max_prompt_length=config.train.get("max_seq_len", 2048) // 2,
                max_length=config.train.get("max_seq_len", 2048),
                tokenizer=tokenizer,
                args=training_args,
                callbacks=[TelemetryCallback(op_id, max_steps)],
            )
        elif train_mode == "orpo":
            from trl import ORPOTrainer

            trainer = ORPOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                callbacks=[TelemetryCallback(op_id, max_steps)],
            )
        else:
            from trl import CPOTrainer

            trainer = CPOTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                callbacks=[TelemetryCallback(op_id, max_steps)],
            )
    else:
        raise ValueError(f"Unsupported training mode: {train_mode}")

    # 6. Run Training
    telemetry.emit_event("stage", {"stage": "training"})
    trainer.train()

    # 7. Save final adapter
    telemetry.emit_event("stage", {"stage": "saving_adapter"})
    adapter_path = config.output.get("adapter_path", "./output")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    telemetry.emit_done({
        "adapter_path": str(adapter_path),
        "final_loss": round(trainer.state.log_history[-1].get("loss", 0.0), 4),
        "steps": max_steps,
    })
    return 0


def train(config_path: str) -> int:
    """Train dispatcher."""
    try:
        config = RunConfig.load(config_path)
    except Exception as exc:
        telemetry.emit_error(f"Failed to load run config: {exc}", traceback.format_exc())
        return 1

    has_ml_deps = False
    try:
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
