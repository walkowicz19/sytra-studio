"""Launch a verified OpenAI-compatible local inference backend."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.xet_safety import apply_runtime_safety, child_process_env
from sytra_runner.model_planner import ModelCompatibilityError, build_backend_plan
from sytra_runner.runtime_detect import prepend_runtime_path
from sytra_runner.serve_ports import require_free_port

apply_runtime_safety()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sytra.serve_model")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capability-gated launcher for llama.cpp, vLLM, or Sytra's native MoE runtime"
        )
    )
    parser.add_argument("--model", required=True, help="Exact GGUF file or complete Safetensors model directory")
    parser.add_argument(
        "--backend",
        choices=("auto", "llama_cpp", "vllm", "sytra_moe"),
        default="auto",
    )
    parser.add_argument("--vram-limit", type=int, default=8192, help="Aggregate serving VRAM budget in MiB")
    parser.add_argument(
        "--ram-limit",
        type=int,
        default=None,
        help="Serving RAM budget in MiB (default: detected physical RAM)",
    )
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument(
        "--verification-positions",
        type=int,
        default=8,
        help="Maximum target-model positions verified per speculative batch",
    )
    parser.add_argument(
        "--storage-bandwidth-mbps",
        type=int,
        default=3500,
        help="Measured sequential model-drive bandwidth in decimal MB/s",
    )
    parser.add_argument(
        "--target-tps",
        type=float,
        default=5.0,
        help="Desired decode rate used by the native I/O feasibility check",
    )
    parser.add_argument(
        "--draft-url",
        help="Loopback OpenAI-compatible endpoint for a small tokenizer-compatible draft model",
    )
    parser.add_argument("--draft-model", help="Model name sent to the draft endpoint")
    parser.add_argument("--cpu-kv-cache", action="store_true")
    parser.add_argument("--kv-cache-quant", default="q8_0")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--ngl", type=int, default=None, help="Force llama.cpp -ngl (GPU layers)")
    parser.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="Allow n-gpu-layers=0 when a GPU is visible (explicit CPU baseline only)",
    )
    parser.add_argument(
        "--replace-port",
        action="store_true",
        help="Do not refuse to start when the listen port is already occupied",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan as JSON without starting a server")
    parser.add_argument(
        "--verify-engine",
        action="store_true",
        help="Run an architecture engine's doctor during a dry-run preflight",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_backend_plan(
            args.model,
            backend=args.backend,
            host=args.host,
            port=args.port,
            context=args.context,
            verification_positions=args.verification_positions,
            storage_bandwidth_mbps=args.storage_bandwidth_mbps,
            target_tps=args.target_tps,
            draft_url=args.draft_url,
            draft_model=args.draft_model,
            vram_limit_mb=args.vram_limit,
            ram_limit_mb=args.ram_limit,
            cpu_kv_cache=args.cpu_kv_cache,
            kv_cache_quant=args.kv_cache_quant,
            flash_attention=not args.no_flash_attention,
            project_root=args.project_root,
            force_gpu_layers=args.ngl,
            allow_cpu_only=args.allow_cpu_only,
        )
    except ModelCompatibilityError as exc:
        logger.error("Model preflight failed: %s", exc)
        return 2

    print(json.dumps(plan.to_dict(), indent=2), flush=True)
    if not plan.compatible:
        for reason in plan.reasons:
            logger.error(reason)
        return 2
    if plan.preflight_command and (not args.dry_run or args.verify_engine):
        logger.info("Running %s architecture preflight", plan.backend)
        try:
            checked = subprocess.run(
                plan.preflight_command, env=child_process_env(), check=False
            )
        except OSError as exc:
            logger.error("Could not run %s preflight: %s", plan.backend, exc)
            return 3
        if checked.returncode:
            logger.error("%s architecture preflight exited with status %d", plan.backend, checked.returncode)
            return checked.returncode
    if args.dry_run:
        return 0

    if not args.replace_port:
        try:
            require_free_port(args.host, args.port)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 4

    logger.info(
        "Starting %s for %s on http://%s:%d",
        plan.backend,
        plan.artifact.model_path,
        args.host,
        args.port,
    )
    for warning in plan.warnings:
        logger.warning(warning)

    env = prepend_runtime_path(child_process_env(), plan.command)
    try:
        completed = subprocess.run(plan.command, env=env, check=False)
    except OSError as exc:
        logger.error("Could not launch %s: %s", plan.backend, exc)
        return 3
    if completed.returncode:
        logger.error("%s exited with status %d", plan.backend, completed.returncode)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
