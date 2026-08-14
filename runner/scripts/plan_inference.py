"""Print a JSON inference plan for a local GGUF or checkpoint directory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.model_planner import ModelCompatibilityError, build_backend_plan
from sytra_runner.runtime_export import export_runtime_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a local model and emit a GPU-first serve plan")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vram-limit", type=int, default=8192)
    parser.add_argument("--ram-limit", type=int, default=None)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--kv-cache-quant", default="q8_0")
    parser.add_argument("--cpu-kv-cache", action="store_true")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--export-runtimes", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_backend_plan(
            args.model,
            backend=args.backend,
            context=args.context,
            vram_limit_mb=args.vram_limit,
            ram_limit_mb=args.ram_limit,
            cpu_kv_cache=args.cpu_kv_cache,
            kv_cache_quant=args.kv_cache_quant,
            flash_attention=not args.no_flash_attention,
            project_root=args.project_root,
        )
        payload = plan.to_dict()
        if args.export_runtimes:
            payload["runtime_export"] = export_runtime_configs(
                args.model, context=args.context
            )
    except ModelCompatibilityError as exc:
        print(json.dumps({"compatible": False, "error": str(exc)}), flush=True)
        return 2
    print(json.dumps(payload), flush=True)
    return 0 if plan.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
