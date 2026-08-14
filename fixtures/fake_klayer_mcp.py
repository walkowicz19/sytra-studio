#!/usr/bin/env python3
"""Stdio MCP stub that implements export_dataset the same way klayer-mcp does."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def read_message() -> dict | None:
    header = sys.stdin.readline()
    if not header:
        return None
    if header.lower().startswith("content-length:"):
        length = int(header.split(":", 1)[1].strip())
        while True:
            line = sys.stdin.readline()
            if line in ("", "\n", "\r\n"):
                break
        body = sys.stdin.read(length)
        return json.loads(body)
    return json.loads(header)


def write_message(payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":"))
    sys.stdout.write(f"Content-Length: {len(body.encode('utf-8'))}\r\n\r\n{body}")
    sys.stdout.flush()


def export_dataset(arguments: dict) -> dict:
    out_dir = Path(arguments["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = arguments.get("domain") or "demo"
    path = out_dir / f"{domain}.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return {"content": [{"type": "text", "text": json.dumps({"wrote": str(path), "rows": 1})}]}


def main() -> int:
    while True:
        msg = read_message()
        if msg is None:
            return 0
        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fake-klayer", "version": "0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if name != "export_dataset":
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"unknown tool {name}"},
                    }
                )
                continue
            write_message({"jsonrpc": "2.0", "id": msg_id, "result": export_dataset(arguments)})
        elif msg_id is not None:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"unknown method {method}"},
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
