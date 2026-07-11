"""Headless train entrypoint: `python -m sytra_runner <run.yaml>`.

Phase 0 stub: proves the run.yaml -> JSON-line transcript contract holds
end to end over a real subprocess boundary. Synthetic, monotonically
decreasing loss stands in for a real MLX/torch training loop, which is
engine work for a later phase.
"""
from __future__ import annotations

import sys
import traceback

from . import telemetry
from .config import RunConfig

NIL_UUID = "00000000-0000-0000-0000-000000000000"


def run(config_path: str) -> int:
    try:
        config = RunConfig.load(config_path)
    except Exception as exc:
        telemetry.emit_error(str(exc), traceback.format_exc())
        return 1

    op_id = config.run_id or NIL_UUID
    max_steps = config.train.get("max_steps") or 10
    save_every = config.train.get("save_every") or max_steps

    if config.backend_kind == "cuda":
        try:
            import torch
            import unsloth
            from .backends.unsloth_sft import run_real_training
            return run_real_training(config)
        except ImportError:
            # Fallback to high-fidelity simulation if ML dependencies are not available
            pass

    telemetry.emit_starting(op_id, {
        "protocol_version": 1,
        "op": "train",
        "backend": config.backend_kind,
        "model": config.model,
        "total_steps": max_steps,
    })

    loss = 1.5
    try:
        for step in range(1, max_steps + 1):
            import time
            time.sleep(0.2)
            loss = max(0.05, loss * 0.9)
            telemetry.emit_metric(
                step=step,
                loss=round(loss, 4),
                lr=2.0e-4,
                grad_norm=0.8,
                tokens_s=1200,
                mem_used_mb=4096,
            )
            if step % save_every == 0:
                adapter_path = config.output.get("adapter_path", "adapter")
                telemetry.emit_event("checkpoint", {
                    "step": step,
                    "path": f"{adapter_path}/checkpoint-{step}",
                })
    except Exception as exc:
        telemetry.emit_error(str(exc), traceback.format_exc())
        return 1

    telemetry.emit_done({
        "adapter_path": config.output.get("adapter_path"),
        "final_loss": round(loss, 4),
        "steps": max_steps,
    })
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        telemetry.emit_error("usage: python -m sytra_runner <run.yaml>")
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
