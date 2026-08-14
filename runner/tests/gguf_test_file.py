"""Write a GGUF header with metadata only (zero tensors) for tests."""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from sytra_runner.gguf_meta import (
    GGUF_ARRAY,
    GGUF_INT32,
    GGUF_MAGIC,
    GGUF_STRING,
    GGUF_UINT32,
    GGUF_UINT64,
)


def _write_string(parts: list[bytes], value: str) -> None:
    encoded = value.encode("utf-8")
    parts.append(struct.pack("<Q", len(encoded)))
    parts.append(encoded)


def _write_value(parts: list[bytes], value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("bool not used in test writer")
    if isinstance(value, int):
        if value < 0:
            parts.append(struct.pack("<i", value))
            return GGUF_INT32
        if value > 0xFFFFFFFF:
            parts.append(struct.pack("<Q", value))
            return GGUF_UINT64
        parts.append(struct.pack("<I", value))
        return GGUF_UINT32
    if isinstance(value, str):
        _write_string(parts, value)
        return GGUF_STRING
    if isinstance(value, list):
        if not value or not all(isinstance(item, str) for item in value):
            raise TypeError("test writer only supports string arrays")
        body: list[bytes] = [struct.pack("<I", GGUF_STRING), struct.pack("<Q", len(value))]
        for item in value:
            _write_string(body, item)
        parts.extend(body)
        return GGUF_ARRAY
    raise TypeError(f"unsupported test GGUF value {type(value)}")


def write_metadata_gguf(path: Path, kv: dict[str, Any], *, payload_bytes: int = 256) -> Path:
    entries: list[bytes] = []
    for key, value in kv.items():
        item: list[bytes] = []
        _write_string(item, key)
        typed: list[bytes] = []
        value_type = _write_value(typed, value)
        item.append(struct.pack("<I", value_type))
        item.extend(typed)
        entries.append(b"".join(item))
    header = [
        GGUF_MAGIC,
        struct.pack("<I", 3),
        struct.pack("<Q", 0),
        struct.pack("<Q", len(kv)),
        *entries,
        b"\0" * payload_bytes,
    ]
    path.write_bytes(b"".join(header))
    return path
