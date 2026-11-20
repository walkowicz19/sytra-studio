"""Platform subprocess helpers for suppressing transient console windows on Windows."""
from __future__ import annotations

import os
import subprocess
from typing import Any


def silent_subprocess_kwargs() -> dict[str, Any]:
    """Return Popen/run kwargs that prevent console windows from flashing in GUI apps."""
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            return {"creationflags": flags}
    return {}
