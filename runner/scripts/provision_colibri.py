"""Download a pinned Colibri (`coli`) release into .tools (gitignored)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.colibri_provision import ColibriProvisionError, provision_colibri


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unpack the pinned Colibri Windows/Linux/macOS release into .tools/colibri"
    )
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else None
    try:
        launcher = provision_colibri(root)
    except ColibriProvisionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(" ".join(launcher))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
