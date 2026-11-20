"""Read checkpoint sizes and content identities without loading tensors."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def checkpoint_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        files = [directory / name for name in sorted(names)]
        if any(not file.resolve().is_relative_to(directory.resolve()) for file in files):
            raise ValueError("Checkpoint index points outside its directory")
        if not all(file.is_file() for file in files):
            raise ValueError("Checkpoint is missing indexed weight shards")
        return files
    return [directory / "model.safetensors"] if (directory / "model.safetensors").is_file() else []


def checkpoint_memory_gb(directory: str | Path, dtype: str) -> float | None:
    config_path = Path(directory) / "config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")).get("quantization_config"):
        return None
    files = checkpoint_files(directory)
    if not files:
        return None
    elements = 0
    for file in files:
        with file.open("rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            if size <= 0 or size > min(100 * 1024**2, file.stat().st_size - 8):
                return None
            header = json.loads(handle.read(size))
            # Packed checkpoints cannot be sized as full-precision parameters from shape alone.
            if any(key.endswith((".qweight", ".qzeros", ".scales")) for key in header):
                return None
            elements += sum(math.prod(value["shape"]) for key, value in header.items() if key != "__metadata__")
    return elements * (4 if dtype == "float32" else 2) / 1024**3


def checkpoint_fingerprint(directory: str | Path) -> str | None:
    directory = Path(directory)
    files = checkpoint_files(directory)
    if not files:
        return None
    digest = hashlib.sha256()
    for file in sorted([*files, *directory.glob("*.json"), *directory.glob("*.model")]):
        digest.update(file.name.encode("utf-8"))
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024**2), b""):
                digest.update(block)
    return digest.hexdigest()


def materialization_ready(directory: str | Path, key: str) -> bool:
    directory = Path(directory)
    try:
        marker = json.loads((directory / ".sytra-complete.json").read_text(encoding="utf-8"))
        files = checkpoint_files(directory)
        return marker.get("key") == key and (directory / "config.json").is_file() and bool(files) and all(file.stat().st_size > 8 for file in files)
    except (OSError, ValueError, KeyError, TypeError):
        return False
