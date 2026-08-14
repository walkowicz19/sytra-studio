"""Generate a checkpoint-bound oracle on a trusted reference machine."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.oracle_generator import main


if __name__ == "__main__":
    raise SystemExit(main())
