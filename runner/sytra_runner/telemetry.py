"""Emitter for the shared train/merge telemetry protocol (Contract 3).

One JSON object per line, flushed immediately. The host's parser
(crates/sytra-contracts/src/telemetry.rs) treats any line that fails to
parse as JSON as a raw log line, so this module's only real obligation is
to keep emitting valid, flushed JSON lines.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any


def _emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_starting(op_id: str, payload: dict[str, Any]) -> None:
    _emit({
        "type": "event",
        "event": "starting",
        "ts": time.time(),
        "op_id": op_id,
        "payload": payload,
    })


def emit_event(event: str, payload: dict[str, Any]) -> None:
    _emit({"type": "event", "event": event, "ts": time.time(), "payload": payload})


def emit_metric(**fields: Any) -> None:
    _emit({"type": "metric", "ts": time.time(), **fields})


def emit_done(payload: dict[str, Any]) -> None:
    emit_event("done", payload)


def emit_error(message: str, traceback: str = "") -> None:
    emit_event("error", {"message": message, "traceback": traceback})


def emit_log(line: str, stream: str = "stdout") -> None:
    _emit({"type": "log", "ts": time.time(), "stream": stream, "line": line})


class Heartbeat:
    """Emits a metric line every `interval` seconds while a long blocking
    call (model download, model load, merge) produces no output of its own,
    so the host never sees a silent stream and the UI never looks frozen.

    Use as a context manager; update `fields` from the outside to move the
    reported progress forward:

        with Heartbeat(lambda: {"progress": state["progress"], "stage": state["stage"]}):
            long_blocking_call()
    """

    def __init__(self, fields_fn, interval: float = 5.0):
        import threading
        self._fields_fn = fields_fn
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                fields = dict(self._fields_fn())
                fields["heartbeat"] = True
                emit_metric(**fields)
            except Exception:
                pass  # a heartbeat must never take the runner down

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
