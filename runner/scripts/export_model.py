"""Sytra GGUF exporter for Ollama and LM Studio.

Refuses raw SafeTensors. Prints JSON so MCP/Tauri can parse stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.model_planner import ModelCompatibilityError
from sytra_runner.runtime_export import export_runtime_configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export GGUF configs for Ollama and LM Studio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default=None, help="Unused; kept for CLI compatibility")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--dest", default=None)
    args = parser.parse_args(argv)
    try:
        result = export_runtime_configs(args.model, context=args.context, dest_dir=args.dest)
    except ModelCompatibilityError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
