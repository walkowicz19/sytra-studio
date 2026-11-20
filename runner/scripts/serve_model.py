from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sytra_runner.model_planner import ModelCompatibilityError, build_backend_plan, launch_backend_plan
from sytra_runner.runtime_detect import prepend_runtime_path
from sytra_runner.subprocess_utils import silent_subprocess_kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select, validate, and optionally launch llama.cpp, vLLM, or Sytra-Helix MoE runtime."
    )
    parser.add_argument("model_path", nargs="?", help="Local path to a GGUF file, Safetensors checkpoint, or Sytra MoE folder.")
    parser.add_argument("--model", dest="model_option")
    parser.add_argument("--backend", default="auto", choices=("auto", "llama_cpp", "vllm", "sytra_moe"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--context-length", "--context", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None, help="Default temperature for the built-in chat.")
    parser.add_argument("--vram-limit-mb", "--vram-limit", type=int, default=None)
    parser.add_argument("--ram-limit-mb", "--ram-limit", type=int, default=None)
    parser.add_argument("--kv-cache-quant", choices=("fp16", "q8_0", "q4_0"), default="q8_0")
    parser.add_argument("--cpu-kv-cache", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--vram-guardrail-mb", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-engine", action="store_true")
    parser.add_argument("--prefer-cpu-offload", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the computed backend plan as JSON.")
    parser.add_argument("--launch", action="store_true", help="Launch the planned runtime command.")
    parser.add_argument("--idle-ttl-seconds", type=int, default=0, help="Stop the model after this many idle seconds; 0 disables the timer.")
    parser.add_argument("--managed-proxy", action="store_true", help="Serve through the local Sytra proxy.")
    parser.add_argument("--draft-model-path", default=None)
    parser.add_argument("--target-tps", type=float, default=0.25)
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    if not (args.model_option or args.model_path) or (args.model_option and args.model_path):
        parser.error("provide one model path, either positionally or with --model")
    if args.idle_ttl_seconds < 0:
        parser.error("--idle-ttl-seconds must be non-negative")
    if args.temperature is not None and (not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2):
        parser.error("--temperature must be between 0 and 2")

    def build_plan(port):
        return build_backend_plan(
            args.model_option or args.model_path,
            backend=args.backend,
            host=args.host,
            port=port,
            context_length=args.context_length,
            vram_limit_mb=args.vram_limit_mb,
            ram_limit_mb=args.ram_limit_mb,
            prefer_cpu_offload=args.prefer_cpu_offload,
            prefer_gpu=not args.no_gpu,
            kv_cache_quant=args.kv_cache_quant,
            cpu_kv_cache=args.cpu_kv_cache,
            flash_attention=not args.no_flash_attention,
            project_root=args.project_root,
            vram_guardrail_mb=args.vram_guardrail_mb,
            draft_model_path=args.draft_model_path,
            target_tps=args.target_tps,
            allow_unverified=args.allow_unverified,
        )

    plan = build_plan(args.port)

    if args.verify_engine and plan.compatible and plan.preflight_command:
        try:
            result = subprocess.run(plan.preflight_command, capture_output=True, text=True, timeout=120,
                                    env=prepend_runtime_path(os.environ.copy(), plan.preflight_command),
                                    **silent_subprocess_kwargs())
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModelCompatibilityError(f"Engine preflight failed: {exc}") from exc
        if result.returncode:
            raise ModelCompatibilityError(f"Engine preflight failed: {(result.stderr or result.stdout)[-4000:]}")

    if args.json or args.dry_run or not args.launch:
        print(json.dumps(asdict(plan), indent=2))
        return 0 if plan.compatible else 2

    if args.managed_proxy or args.idle_ttl_seconds:
        from sytra_runner.idle_proxy import serve_with_idle_timeout
        model_label = Path(args.model_option or args.model_path).name
        return serve_with_idle_timeout(args.host, args.port, args.idle_ttl_seconds,
                                       lambda port: launch_backend_plan(build_plan(port)),
                                       temperature=args.temperature,
                                       context_window=args.context_length,
                                       model_name=model_label)
    process = launch_backend_plan(plan)
    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
