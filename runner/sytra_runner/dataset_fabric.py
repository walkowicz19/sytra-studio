"""Streaming preparation of conversational training datasets.

Filtering is opt-in and never rewrites message text. Outputs are staged in a
new directory so failed jobs cannot leave a seemingly complete dataset behind.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize common conversation formats without changing message text."""
    if not isinstance(row, dict):
        raise ValueError("Each dataset row must be an object")
    raw = row.get("messages") or row.get("conversations")
    if raw is None:
        prompt = row.get("prompt", row.get("instruction", row.get("user")))
        answer = row.get("completion", row.get("output", row.get("response", row.get("assistant"))))
        if isinstance(prompt, list) and isinstance(answer, list):
            raw = prompt + answer
        elif isinstance(prompt, str) and isinstance(answer, str):
            # Alpaca's input is part of the user message, not a system prompt.
            extra = row.get("input")
            if extra is not None and not isinstance(extra, str):
                raise ValueError("Alpaca input must be text")
            user = prompt + ("\n\n" + extra if extra else "")
            system = row.get("system")
            if system is not None and not isinstance(system, str):
                raise ValueError("system prompt must be text")
            raw = ([{"role": "system", "content": system}] if isinstance(system, str) and system else []) + [
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
        else:
            raise ValueError("Expected messages, conversations, or a prompt/completion pair")
    if not isinstance(raw, list) or not raw:
        raise ValueError("A conversation must be a non-empty array")
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
    messages = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each message must be an object")
        role = item.get("role", item.get("from"))
        content = item.get("content", item.get("value"))
        if role not in roles or not isinstance(content, str):
            raise ValueError("Messages require a supported role and text content")
        messages.append({"role": roles[role], "content": content})
    if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
        raise ValueError("A conversation must end with a non-empty assistant response")
    return {"messages": messages}


def _rows(path: Path, *, on_invalid=None) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}:{number}: {exc}") from exc
                    try:
                        row = normalize_row(parsed)
                    except (ValueError, TypeError) as exc:
                        if on_invalid is None:
                            raise ValueError(f"{path}:{number}: {exc}") from exc
                        on_invalid()
                        continue
                    yield row
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet input requires pyarrow") from exc
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                try:
                    normalized = normalize_row(row)
                except (ValueError, TypeError):
                    if on_invalid is None:
                        raise
                    on_invalid()
                    continue
                yield normalized
    else:
        raise ValueError(f"Unsupported dataset format: {path}")


def _encode(row: dict[str, Any]) -> bytes:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def resolve_manifest(path: str | Path) -> list[str]:
    """Resolve one Fabric index without permitting traversal or duplicate shards."""
    manifest_path = Path(path).resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Invalid or oversized dataset manifest")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    shards = manifest.get("shards") if isinstance(manifest, dict) else None
    if not isinstance(manifest, dict) or type(manifest.get("version")) is not int or manifest["version"] != 1 or not isinstance(shards, list) or not 1 <= len(shards) <= 10000:
        raise ValueError("Invalid dataset manifest version or shard list")
    result: list[Path] = []
    total = 0
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValueError("Invalid dataset shard")
        name = shard["path"]
        if Path(name).name != name or Path(name).suffix.lower() != ".jsonl":
            raise ValueError("Shard paths must be JSONL filenames in the manifest directory")
        resolved = (manifest_path.parent / name).resolve(strict=True)
        rows = shard.get("rows")
        if resolved.parent != manifest_path.parent or not resolved.is_file() or resolved in result or type(rows) is not int or rows <= 0:
            raise ValueError(f"Invalid or duplicate shard: {name}")
        actual_rows = 0
        last_byte = b""
        with resolved.open("rb") as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b""):
                actual_rows += chunk.count(b"\n")
                last_byte = chunk[-1:]
        if last_byte and last_byte != b"\n":
            actual_rows += 1
        if actual_rows != rows:
            raise ValueError(f"Shard row count mismatch for {name}")
        total += rows
        result.append(resolved)
    if type(manifest.get("rows")) is not int or manifest["rows"] != total:
        raise ValueError("Manifest total does not match shard counts")
    return [str(path) for path in result]


def _quotas(weights: list[float], total: int) -> list[int]:
    if total <= 0 or not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("total_samples and all weights must be positive")
    scale = max(weights)
    normalized = [weight / scale for weight in weights]
    exact = [total * weight / sum(normalized) for weight in normalized]
    quotas = [math.floor(value) for value in exact]
    for index in sorted(range(len(weights)), key=lambda i: (-(exact[i] - quotas[i]), i))[:total - sum(quotas)]:
        quotas[index] += 1
    return quotas


def merge_datasets(inputs: list[str | Path], output_dir: str | Path, *,
                   weights: list[float] | None = None, total_samples: int | None = None,
                   shard_rows: int = 0, seed: int = 3407,
                   curation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge JSONL/Parquet inputs into canonical JSONL, optionally in shards.

    Weighted mode takes the requested quota from each input without replacement;
    it fails if any input has too few rows instead of silently changing weights.
    """
    paths = [Path(p).resolve(strict=True) for p in inputs]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("At least one input file is required")
    if shard_rows < 0:
        raise ValueError("shard_rows must be non-negative")
    if (weights is None) != (total_samples is None):
        raise ValueError("weights and total_samples must be supplied together")
    if weights is not None and len(weights) != len(paths):
        raise ValueError("Supply one weight per input")
    quotas = _quotas(weights, total_samples) if weights is not None else None
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    counts = [0] * len(paths)
    shards: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=".sytra-dataset-", dir=output.parent) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        curator = None
        if curation is not None:
            from .dataset_curation import DatasetCurator
            curator = DatasetCurator(Path(tmp) / "curation.sqlite", curation)
        on_invalid = curator.reject_invalid if curator is not None and curator.drop_invalid else None
        readers = [iter(_rows(path, on_invalid=on_invalid)) for path in paths]
        active = list(range(len(paths)))
        writer = None
        try:
            while active:
                if quotas is None:
                    index = active[0]
                    active = active[1:] + active[:1]
                else:
                    active = [i for i in active if counts[i] < quotas[i]]
                    if not active:
                        break
                    index = rng.choices(active, weights=[quotas[i] - counts[i] for i in active], k=1)[0]
                try:
                    row = next(readers[index])
                except StopIteration:
                    if quotas is not None and counts[index] < quotas[index]:
                        raise ValueError(f"Insufficient samples in {paths[index]}: needed {quotas[index]}, found {counts[index]}")
                    active.remove(index)
                    continue
                if curator is not None:
                    row = curator.prepare_row(row)
                encoded = _encode(row)
                if curator is not None and not curator.accept(row, encoded):
                    continue
                if writer is None or (shard_rows and shards[-1]["rows"] >= shard_rows):
                    if writer is not None:
                        writer.close()
                    name = f"dataset-{len(shards) + 1:05d}.jsonl" if shard_rows else "dataset.jsonl"
                    writer = (staging / name).open("wb")
                    shards.append({"path": name, "rows": 0})
                writer.write(encoded)
                shards[-1]["rows"] += 1
                counts[index] += 1
        finally:
            if writer is not None:
                writer.close()
            for reader in readers:
                reader.close()
            if curator is not None:
                curator.close()
        if not shards:
            raise ValueError("No dataset samples found")
        manifest = {"version": 1, "inputs": [str(p) for p in paths], "counts": counts,
                    "rows": sum(counts), "shards": shards, "seed": seed, "weights": weights}
        if curator is not None:
            manifest["curation"] = {
                "options": curation,
                "removed": curator.stats,
                "benchmarks": curator.benchmark_info,
                "token_counting": curator.token_counting,
            }
            if curator.sanitization_info is not None:
                manifest["curation"]["sanitization"] = curator.sanitization_info
            if curator.reviewer is not None:
                manifest['curation']['semantic'] = curator.reviewer.info
            if curator.context_filter is not None:
                manifest['curation']['context'] = curator.context_filter.info
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(staging, output)
    return manifest


def split_dataset(input_path: str | Path, output_dir: str | Path, *, validation_fraction: float = 0.1,
                  seed: int = 3407) -> dict[str, Any]:
    """Assign identical normalized rows to the same split using a stable hash."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    source = Path(input_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0}
    threshold = int(validation_fraction * (1 << 64))
    with tempfile.TemporaryDirectory(prefix=".sytra-split-", dir=output.parent) as tmp:
        staging = Path(tmp) / "result"
        staging.mkdir()
        with ExitStack() as stack:
            writers = {name: stack.enter_context((staging / f"{name}.jsonl").open("wb")) for name in counts}
            for row in _rows(source):
                encoded = _encode(row)
                digest = hashlib.sha256(str(seed).encode("ascii") + b"\0" + encoded).digest()
                name = "validation" if int.from_bytes(digest[:8], "big") < threshold else "train"
                writers[name].write(encoded)
                counts[name] += 1
        if not all(counts.values()):
            raise ValueError("Split produced an empty partition; use more rows or adjust fraction/seed")
        manifest = {"version": 1, "input": str(source), "counts": counts,
                    "validation_fraction": validation_fraction, "seed": seed}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(staging, output)
    return manifest
