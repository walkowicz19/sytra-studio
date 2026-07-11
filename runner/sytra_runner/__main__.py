"""Main entrypoint dispatcher (Phase 1).

Detects the config kind (run.yaml vs merge.yaml) and routes to either
the Unsloth trainer or the mergekit merger.
"""
from __future__ import annotations

import sys
import traceback
import yaml

from . import telemetry


def main() -> int:
    if len(sys.argv) < 2:
        telemetry.emit_error("Usage: python -m sytra_runner <config.yaml>")
        return 2

    config_path = sys.argv[1]
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            
        if not isinstance(raw, dict):
            raise ValueError("Configuration file must be a YAML dictionary")

        if "train_mode" in raw:
            from .backends import unsloth_sft
            return unsloth_sft.train(config_path)
        elif "merge_method" in raw:
            from . import merge
            return merge.run(config_path)
        else:
            raise ValueError("Could not determine operation kind. Neither 'train_mode' nor 'merge_method' found.")
            
    except Exception as exc:
        # Guarantee that any uncaught exception emits a terminal telemetry error
        telemetry.emit_error(str(exc), traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
