"""Detect occupied inference ports so Sytra does not attach to a stale server."""
from __future__ import annotations

import socket


def port_in_use(host: str, port: int, timeout: float = 0.4) -> bool:
    del timeout
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def require_free_port(host: str, port: int) -> None:
    if port_in_use(host, port):
        raise RuntimeError(
            f"Port {host}:{port} is already in use. Stop the stale llama.cpp / Ollama / "
            "LM Studio server, or pick another port. Sytra will not silently reuse it."
        )
