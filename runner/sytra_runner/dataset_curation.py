"""Opt-in, disk-backed deterministic filters for conversational datasets."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

_TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORDS = re.compile(r"\w+", re.UNICODE)


def _text_fields(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _text_fields(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _text_fields(item)


def _benchmark_texts(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield from _text_fields(json.loads(line))
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"Invalid benchmark row {path}:{number}: {exc}") from exc
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet benchmarks require pyarrow") from exc
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                yield from _text_fields(row)
    else:
        raise ValueError("Benchmark files must be JSONL or Parquet")


def _benchmark_grams(text: str):
    words = _WORDS.findall(text.casefold())
    for index in range(len(words) - 12):
        gram = "\0".join(words[index:index + 13]).encode("utf-8")
        yield hashlib.sha256(gram).digest()


_LEAKED_CHAT_TAGS = re.compile(
    r"(?:<\|im_start\|>(?:system|user|assistant)?|<\|im_end\|>|<\|endoftext\|>|"
    r"<\|eot_id\|>|<\|start_header_id\|>(?:system|user|assistant)?<\|end_header_id\|>|"
    r"</?s>|\[/?INST\])"
)
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_FENCE_LINE = re.compile(r"(?<!`)```(?!`)")


class DatasetCurator:
    """Filters accepted rows only; all deduplication state lives in SQLite."""

    def __init__(self, path: Path, options: dict[str, Any]):
        allowed = {
            "exact_duplicates",
            "near_duplicates",
            "similarity",
            "repetitive",
            "min_tokens",
            "benchmark_files",
            "drop_invalid",
            "sanitize_syntax",
            "semantic",
            "context",
        }
        if not isinstance(options, dict) or set(options) - allowed:
            raise ValueError("Unknown curation option")
        for key in ("exact_duplicates", "near_duplicates", "repetitive", "drop_invalid", "sanitize_syntax"):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
        self.exact = options.get("exact_duplicates", False)
        self.near = options.get("near_duplicates", False)
        self.repetitive = options.get("repetitive", False)
        self.drop_invalid = options.get("drop_invalid", False)
        self.sanitize_syntax = options.get("sanitize_syntax", False)
        self.similarity = options.get("similarity", 0.85)
        self.min_tokens = options.get("min_tokens", 0)
        if type(self.min_tokens) is not int or self.min_tokens < 0:
            raise ValueError("min_tokens must be a non-negative integer")
        if type(self.similarity) not in (int, float) or not math.isfinite(self.similarity) or not 0 < self.similarity <= 1:
            raise ValueError("similarity must be greater than 0 and at most 1")
        self.stats = {"exact_duplicate": 0, "near_duplicate": 0, "repetitive": 0, "too_short": 0, "benchmark_overlap": 0}
        if self.drop_invalid:
            self.stats['invalid_structure'] = 0
        if self.sanitize_syntax:
            self.stats['syntax_sanitized'] = 0
        self.reviewer = None
        self.context_filter = None
        if 'context' in options:
            from .dataset_context import NativeContextFilter
            context_opts = dict(options['context'])
            if 'min_tokens' not in context_opts and self.min_tokens > 0:
                context_opts['min_tokens'] = self.min_tokens
            self.context_filter = NativeContextFilter(context_opts)
            if self.context_filter.limit is not None:
                self.stats['too_long'] = 0
        self.token_counting = "native_tokenizer" if self.context_filter is not None else "lexical_fallback"
        self.sanitization_info = (
            {"enabled": True, "rules": ["leaked_chat_tags", "control_chars", "unclosed_code_fences"]}
            if self.sanitize_syntax
            else None
        )
        if 'semantic' in options:
            from .dataset_semantic import LocalSemanticReviewer
            self.reviewer = LocalSemanticReviewer(options['semantic'])
            self.stats['semantic_rejected'] = 0
        benchmark_files = options.get("benchmark_files", [])
        if not isinstance(benchmark_files, list) or len(benchmark_files) > 50 or any(not isinstance(item, str) or not item.strip() for item in benchmark_files):
            raise ValueError("benchmark_files must be an array of up to 50 paths")
        self.benchmark_info: list[dict[str, Any]] = []
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA temp_store=FILE;
            CREATE TABLE samples (id INTEGER PRIMARY KEY, digest BLOB UNIQUE, shingles TEXT NOT NULL);
            CREATE TABLE bands (band INTEGER NOT NULL, key BLOB NOT NULL, sample_id INTEGER NOT NULL);
            CREATE INDEX band_lookup ON bands (band, key);
            CREATE TABLE benchmark_grams (digest BLOB PRIMARY KEY) WITHOUT ROWID;
        """)
        try:
            for raw in benchmark_files:
                source = Path(raw).resolve(strict=True)
                if not source.is_file():
                    raise ValueError(f"Benchmark is not a file: {source}")
                file_hash = hashlib.sha256()
                with source.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        file_hash.update(chunk)
                before = self.db.total_changes
                batch = []
                for text in _benchmark_texts(source):
                    for digest in _benchmark_grams(text):
                        batch.append((digest,))
                        if len(batch) >= 1000:
                            self.db.executemany("INSERT OR IGNORE INTO benchmark_grams VALUES (?)", batch)
                            batch.clear()
                if batch:
                    self.db.executemany("INSERT OR IGNORE INTO benchmark_grams VALUES (?)", batch)
                inserted = self.db.total_changes - before
                self.benchmark_info.append({"path": str(source), "sha256": file_hash.hexdigest(), "new_13grams": inserted})
            if benchmark_files and not self.db.execute("SELECT 1 FROM benchmark_grams LIMIT 1").fetchone():
                raise ValueError("The selected benchmark files contain no text with 13 words")
        except Exception:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _tokens(row: dict[str, Any]) -> list[str]:
        words = []
        for message in row["messages"]:
            words.append("role:" + message["role"])
            words.extend(_TOKENS.findall(message["content"].casefold()))
        return words

    @staticmethod
    def _shingles(tokens: list[str]) -> set[str]:
        groups = zip(tokens, tokens[1:], tokens[2:]) if len(tokens) >= 3 else [tuple(tokens)]
        return {hashlib.blake2b("\0".join(group).encode("utf-8"), digest_size=8).hexdigest() for group in groups}

    @staticmethod
    def _bands(shingles: set[str]) -> list[bytes]:
        values = [bytes.fromhex(value) for value in shingles]
        signature = [min(hashlib.blake2b(value, digest_size=8, person=index.to_bytes(8, "big")).digest()
                         for value in values) for index in range(64)]
        return [b"".join(signature[index:index + 4]) for index in range(0, 64, 4)]

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = _CONTROL_CHARS.sub("", text)
        stripped_tags = _LEAKED_CHAT_TAGS.sub("", cleaned)
        if stripped_tags != cleaned:
            cleaned = stripped_tags.strip()
        if len(_FENCE_LINE.findall(cleaned)) % 2 == 1:
            cleaned = cleaned.rstrip() + ("\n```" if cleaned.strip() else "")
        return cleaned

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Normalize leaked syntax only when the user explicitly opted into sanitization."""
        if not self.sanitize_syntax:
            return row
        updated = []
        changed = False
        for message in row["messages"]:
            sanitized = self._sanitize_text(message["content"])
            if sanitized != message["content"]:
                changed = True
            updated.append({"role": message["role"], "content": sanitized})
        if not changed:
            return row
        self.stats["syntax_sanitized"] += 1
        return {"messages": updated}

    def accept(self, row: dict[str, Any], encoded: bytes) -> bool:
        if self.drop_invalid or self.sanitize_syntax:
            has_user = False
            messages = row["messages"]
            invalid = (not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip())
            if not invalid:
                for message in messages:
                    if not message["content"].strip() or (message["role"] == "assistant" and not has_user):
                        invalid = True
                        break
                    if message["role"] == "user":
                        has_user = True
                if not has_user:
                    invalid = True
            if invalid:
                if self.drop_invalid:
                    self.reject_invalid()
                    return False
                raise ValueError("Sanitization produced an invalid or empty conversation; enable drop_invalid to discard it")
        tokens = self._tokens(row) if (self.context_filter is None or self.near) else None
        if self.min_tokens > 0:
            token_count = (
                self.context_filter.count_message_tokens(row)
                if self.context_filter is not None
                else (len(tokens) - len(row["messages"]))
            )
            if token_count < self.min_tokens:
                self.stats["too_short"] += 1
                return False
        if self.context_filter is not None and not self.context_filter.accept(row):
            self.stats['too_long'] += 1
            return False
        if self.repetitive:
            for message in row["messages"]:
                if message["role"] != "assistant":
                    continue
                words = _TOKENS.findall(message["content"].casefold())
                if len(words) >= 24:
                    for size in (3, 4):
                        grams = list(zip(*(words[offset:] for offset in range(size))))
                        repeats = sum(count for count in Counter(grams).values() if count >= 3)
                        if repeats / len(grams) >= 0.6:
                            self.stats["repetitive"] += 1
                            return False
        if self.benchmark_info:
            for message in row["messages"]:
                for gram in _benchmark_grams(message["content"]):
                    if self.db.execute("SELECT 1 FROM benchmark_grams WHERE digest = ?", (gram,)).fetchone():
                        self.stats["benchmark_overlap"] += 1
                        return False
        if not (self.exact or self.near):
            return self._semantic_accept(row)
        digest = hashlib.sha256(encoded).digest()
        if self.db.execute("SELECT 1 FROM samples WHERE digest = ?", (digest,)).fetchone():
            self.stats["exact_duplicate"] += 1
            return False
        shingles = self._shingles(tokens) if self.near else set()
        bands = self._bands(shingles) if self.near else []
        if bands:
            conditions = " OR ".join("(bands.band = ? AND bands.key = ?)" for _ in bands)
            parameters = [part for number, key in enumerate(bands) for part in (number, key)]
            query = f"SELECT DISTINCT samples.id, samples.shingles FROM bands JOIN samples ON samples.id = bands.sample_id WHERE {conditions} ORDER BY samples.id"
            for _candidate, stored in self.db.execute(query, parameters):
                previous = set(json.loads(stored))
                if len(shingles & previous) / len(shingles | previous) >= self.similarity:
                    self.stats["near_duplicate"] += 1
                    return False
        if not self._semantic_accept(row):
            return False
        cursor = self.db.execute("INSERT INTO samples(digest, shingles) VALUES (?, ?)", (digest, json.dumps(sorted(shingles))))
        self.db.executemany("INSERT INTO bands(band, key, sample_id) VALUES (?, ?, ?)",
                            ((number, key, cursor.lastrowid) for number, key in enumerate(bands)))
        return True

    def _semantic_accept(self, row: dict[str, Any]) -> bool:
        if self.reviewer is None or self.reviewer.accept(row):
            return True
        self.stats['semantic_rejected'] += 1
        return False

    def reject_invalid(self) -> None:
        """Count a malformed conversation when the caller opted in to discard it."""
        if not self.drop_invalid:
            raise ValueError('Dropping malformed conversations is not enabled')
        self.stats['invalid_structure'] += 1
