"""Read GGUF metadata without loading tensors.

Architecture, expert counts, and layer counts come from the file header —
never from the model file name.
"""
from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"

# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12

FILE_TYPE_NAMES = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
}


class GgufMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class GgufMetadata:
    architecture: str
    quantization: str | None
    n_layer: int | None
    n_head: int | None
    n_head_kv: int | None
    n_embd: int | None
    n_expert: int | None
    n_expert_used: int | None
    context_length: int | None
    vocab_size: int | None
    parameter_count: int | None
    chat_template: str | None
    bos_token: str | None
    eos_token: str | None
    stop_tokens: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def is_moe(self) -> bool:
        return bool(self.n_expert and self.n_expert > 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_moe"] = self.is_moe
        return data

    def config_for_kv(self) -> dict[str, Any]:
        return {
            "num_hidden_layers": self.n_layer,
            "num_attention_heads": self.n_head,
            "num_key_value_heads": self.n_head_kv or self.n_head,
            "hidden_size": self.n_embd,
        }


def _read_u32(handle: BinaryIO) -> int:
    data = handle.read(4)
    if len(data) != 4:
        raise GgufMetadataError("truncated GGUF integer")
    return struct.unpack("<I", data)[0]


def _read_u64(handle: BinaryIO) -> int:
    data = handle.read(8)
    if len(data) != 8:
        raise GgufMetadataError("truncated GGUF integer")
    return struct.unpack("<Q", data)[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    if length > 16 * 1024 * 1024:
        raise GgufMetadataError("GGUF string exceeds 16 MiB")
    raw = handle.read(length)
    if len(raw) != length:
        raise GgufMetadataError("truncated GGUF string")
    return raw.decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int, *, depth: int = 0) -> Any:
    if depth > 4:
        raise GgufMetadataError("GGUF array nesting is too deep")
    if value_type == GGUF_UINT8:
        return handle.read(1)[0]
    if value_type == GGUF_INT8:
        return struct.unpack("<b", handle.read(1))[0]
    if value_type == GGUF_UINT16:
        return struct.unpack("<H", handle.read(2))[0]
    if value_type == GGUF_INT16:
        return struct.unpack("<h", handle.read(2))[0]
    if value_type == GGUF_UINT32:
        return _read_u32(handle)
    if value_type == GGUF_INT32:
        return struct.unpack("<i", handle.read(4))[0]
    if value_type == GGUF_FLOAT32:
        return struct.unpack("<f", handle.read(4))[0]
    if value_type == GGUF_BOOL:
        return handle.read(1)[0] != 0
    if value_type == GGUF_STRING:
        return _read_string(handle)
    if value_type == GGUF_UINT64:
        return _read_u64(handle)
    if value_type == GGUF_INT64:
        return struct.unpack("<q", handle.read(8))[0]
    if value_type == GGUF_FLOAT64:
        return struct.unpack("<d", handle.read(8))[0]
    if value_type == GGUF_ARRAY:
        item_type = _read_u32(handle)
        count = _read_u64(handle)
        if count > 2_000_000:
            raise GgufMetadataError("GGUF array is unreasonably large")
        return [_read_value(handle, item_type, depth=depth + 1) for _ in range(count)]
    raise GgufMetadataError(f"unsupported GGUF value type {value_type}")


def _positive(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def read_gguf_metadata(path: str | Path) -> GgufMetadata:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise GgufMetadataError(f"{file_path.name} is not a GGUF file")
        version = _read_u32(handle)
        if version not in {1, 2, 3}:
            raise GgufMetadataError(f"unsupported GGUF version {version}")
        tensor_count = _read_u64(handle) if version >= 2 else _read_u32(handle)
        kv_count = _read_u64(handle) if version >= 2 else _read_u32(handle)
        if tensor_count > 10_000_000 or kv_count > 100_000:
            raise GgufMetadataError("GGUF header counts are implausible")
        raw: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(handle)
            value_type = _read_u32(handle)
            raw[key] = _read_value(handle, value_type)

    architecture = str(raw.get("general.architecture") or "").strip() or "unknown"
    prefix = architecture if architecture != "unknown" else ""
    n_layer = _positive(raw, f"{prefix}.block_count", "llama.block_count")
    n_head = _positive(raw, f"{prefix}.attention.head_count", "llama.attention.head_count")
    n_head_kv = _positive(
        raw,
        f"{prefix}.attention.head_count_kv",
        "llama.attention.head_count_kv",
    )
    n_embd = _positive(raw, f"{prefix}.embedding_length", "llama.embedding_length")
    n_expert = _positive(
        raw,
        f"{prefix}.expert_count",
        f"{prefix}.expert.count",
        "llama.expert_count",
    )
    n_expert_used = _positive(
        raw,
        f"{prefix}.expert_used_count",
        f"{prefix}.expert.used_count",
        "llama.expert_used_count",
    )
    file_type = raw.get("general.file_type")
    quantization = FILE_TYPE_NAMES.get(int(file_type)) if isinstance(file_type, int) else None
    chat_template = raw.get("tokenizer.chat_template")
    if not isinstance(chat_template, str) or not chat_template.strip():
        chat_template = None
    bos = raw.get("tokenizer.ggml.bos_token_id")
    eos = raw.get("tokenizer.ggml.eos_token_id")
    tokens = raw.get("tokenizer.ggml.tokens")
    bos_token = tokens[bos] if isinstance(tokens, list) and isinstance(bos, int) and 0 <= bos < len(tokens) else None
    eos_token = tokens[eos] if isinstance(tokens, list) and isinstance(eos, int) and 0 <= eos < len(tokens) else None
    eot = raw.get("tokenizer.ggml.eot_token_id")
    stops: list[str] = []
    for token_id in (eos, eot):
        if isinstance(tokens, list) and isinstance(token_id, int) and 0 <= token_id < len(tokens):
            token = tokens[token_id]
            if isinstance(token, str) and token and token not in stops:
                stops.append(token)
    return GgufMetadata(
        architecture=architecture,
        quantization=quantization,
        n_layer=n_layer,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        context_length=_positive(raw, f"{prefix}.context_length", "llama.context_length"),
        vocab_size=len(raw["tokenizer.ggml.tokens"])
        if isinstance(raw.get("tokenizer.ggml.tokens"), list)
        else None,
        parameter_count=_positive(raw, "general.parameter_count"),
        chat_template=chat_template,
        bos_token=bos_token if isinstance(bos_token, str) else None,
        eos_token=eos_token if isinstance(eos_token, str) else None,
        stop_tokens=tuple(stops),
        raw=raw,
    )


def try_read_gguf_metadata(path: str | Path) -> GgufMetadata | None:
    try:
        return read_gguf_metadata(path)
    except (OSError, GgufMetadataError, struct.error, IndexError):
        return None
