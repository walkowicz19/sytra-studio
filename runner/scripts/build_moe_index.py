"""Create a native Sytra expert byte-range index from SafeTensors headers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.moe_index import (
    MoEIndexError,
    WEIGHT_FORMATS,
    build_runtime_manifest,
    write_runtime_manifest,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Index routed expert tensors for Sytra's native VRAM/RAM/NVMe runtime "
            "without copying or rewriting checkpoint weights"
        )
    )
    command.add_argument("--model", required=True, help="Complete SafeTensors model directory")
    command.add_argument(
        "--adapter",
        default="auto",
        help="Compiled Sytra adapter id, or auto (default) for trusted config detection",
    )
    command.add_argument(
        "--expert-format",
        default="auto",
        choices=["auto", *sorted(WEIGHT_FORMATS)],
    )
    command.add_argument(
        "--expert-regex",
        help="Optional regex with exactly two capture groups: (layer, expert)",
    )
    command.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the index without writing .sytra-runtime.json",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = build_runtime_manifest(
            args.model,
            adapter=args.adapter,
            expert_format=args.expert_format,
            expert_regex=args.expert_regex,
        )
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        else:
            output = write_runtime_manifest(args.model, manifest)
            print(
                json.dumps(
                    {
                        "runtime_manifest": str(output),
                        "experts_indexed": len(manifest["storage"]["experts"]),
                        "dense_bytes": manifest["dense_bytes"],
                        "forward_verified": False,
                    },
                    indent=2,
                )
            )
    except MoEIndexError as exc:
        print(f"Sytra MoE index failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
