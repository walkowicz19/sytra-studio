"""Build Sytra's byte-range expert index without rewriting model weights."""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .architecture_adapters import (
    AdapterCompatibilityError,
    adapter_by_id,
    build_architecture_contract,
    infer_architecture_adapter,
    validate_moe_config,
)


RUNTIME_MANIFEST = ".sytra-runtime.json"
DEFAULT_EXPERT_PATTERNS = (
    re.compile(
        r"(?:^|\.)layers\.(\d+)\.(?:(?:block_sparse_moe|mlp)\.)?experts\.(\d+)\."
    ),
    re.compile(r"(?:^|\.)blocks\.(\d+)\.(?:feed_forward\.)?experts\.(\d+)\."),
)
KNOWN_ADAPTER_PREFIXES = {
    "sytra-glm52": ("glm",),
    "sytra-kimi-k2.7-code": ("kimi_k25",),
    "sytra-kimi-k3": ("kimi_k3", "kimi-k3"),
    "sytra-inkling": ("inkling",),
}
WEIGHT_FORMATS = {
    "f32",
    "f16",
    "bf16",
    "int8",
    "int4_group",
    "packed_int4_group32",
    "fp8_e4m3",
    "nvfp4",
    "mxfp4",
    "gguf",
    "custom",
}

STACKED_LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)\.")
STACKED_EXPERT_MARKERS = (
    ".experts.",
    ".experts_",
    ".ffn.experts.",
    ".moe.experts.",
    ".block_sparse_moe.input_linear.weight",
    ".block_sparse_moe.output_linear.weight",
)


class MoEIndexError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MoEIndexError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MoEIndexError(f"{path} must contain a JSON object")
    return value


def _positive_int(config: dict[str, Any], *keys: str) -> int | None:
    sources = [config]
    for name in ("text_config", "language_config", "llm_config", "ffn_config"):
        nested = config.get(name)
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _bool(config: dict[str, Any], key: str) -> bool:
    if config.get(key):
        return True
    return any(
        isinstance(config.get(name), dict) and config[name].get(key)
        for name in ("text_config", "language_config", "llm_config", "ffn_config")
    )


def _detect_expert_format(config: dict[str, Any], adapter_id: str) -> str:
    if adapter_id == "sytra-kimi-k3":
        return "mxfp4"
    sources = [config]
    for key in ("text_config", "language_config", "llm_config"):
        if isinstance(config.get(key), dict):
            sources.append(config[key])
    for source in sources:
        quant = source.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        if quant.get("format") == "pack-quantized":
            groups = quant.get("config_groups")
            group = groups.get("group_0") if isinstance(groups, dict) else None
            weights = group.get("weights") if isinstance(group, dict) else None
            if (
                isinstance(weights, dict)
                and weights.get("num_bits") == 4
                and weights.get("group_size") == 32
                and weights.get("symmetric") is True
            ):
                return "packed_int4_group32"
        method = str(quant.get("quant_method") or quant.get("format") or "").lower()
        if "fp8" in method:
            return "fp8_e4m3"
        if "nvfp4" in method:
            return "nvfp4"
    dtype = str(config.get("dtype") or config.get("torch_dtype") or "").lower()
    if "float32" in dtype:
        return "f32"
    return "f16" if "float16" in dtype and "bfloat" not in dtype else "bf16"


def _stacked_expert_slices(
    tensor_name: str,
    metadata: dict[str, Any],
    *,
    experts_per_layer: int,
    allow_merged_rows: bool,
) -> tuple[int, str, list[tuple[int, int, list[int]]]] | None:
    """Return layer, layout, and expert-relative byte slices for packed tensors."""

    if not any(marker in tensor_name for marker in STACKED_EXPERT_MARKERS):
        return None
    layer_match = STACKED_LAYER_PATTERN.search(tensor_name)
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if (
        layer_match is None
        or not isinstance(shape, list)
        or not shape
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
    ):
        return None
    total = offsets[1] - offsets[0]
    if total <= 0 or total % experts_per_layer != 0:
        return None
    if shape[0] == experts_per_layer:
        layout = "stacked_axis0"
        expert_shape = shape[1:]
    elif allow_merged_rows and shape[0] % experts_per_layer == 0:
        layout = "merged_rows"
        expert_shape = [shape[0] // experts_per_layer, *shape[1:]]
    else:
        return None
    stride = total // experts_per_layer
    return (
        int(layer_match.group(1)),
        layout,
        [
            (expert, offsets[0] + expert * stride, expert_shape)
            for expert in range(experts_per_layer)
        ],
    )


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    size = path.stat().st_size
    if size < 12:
        raise MoEIndexError(f"Safetensors shard is too small: {path}")
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little", signed=False)
        if header_size <= 0 or header_size > min(size - 8, 100 * 1024 * 1024):
            raise MoEIndexError(f"Invalid Safetensors header length in {path}")
        try:
            header = json.loads(handle.read(header_size).decode("utf-8").rstrip())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MoEIndexError(f"Invalid Safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise MoEIndexError(f"Safetensors header is not an object: {path}")
    return 8 + header_size, header


def _expert_key(
    tensor_name: str,
    patterns: Iterable[re.Pattern[str]],
) -> tuple[int, int] | None:
    for pattern in patterns:
        match = pattern.search(tensor_name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _expert_signature(segments: list[dict[str, Any]], expert: int) -> set[tuple[Any, ...]]:
    """Canonical tensor signature used to reject incomplete/mixed experts."""

    marker = re.compile(rf"(?<=\.experts\.){expert}(?=\.)")
    return {
        (
            marker.sub("{expert}", segment["tensor"]),
            segment.get("dtype"),
            tuple(segment.get("shape") or ()),
            segment["length"],
        )
        for segment in segments
    }


def _validate_adapter(adapter: str, model_type: str) -> None:
    prefixes = KNOWN_ADAPTER_PREFIXES.get(adapter)
    if prefixes and not model_type.lower().startswith(prefixes):
        raise MoEIndexError(
            f"{adapter} cannot index model_type {model_type!r}; architecture adapters are exact"
        )


def build_runtime_manifest(
    model_root: str | Path,
    *,
    adapter: str = "auto",
    expert_format: str = "auto",
    expert_regex: str | None = None,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise MoEIndexError(f"Missing config.json under {root}")
    config = _json(config_path)
    model_type = str(config.get("model_type") or "")
    if not model_type:
        raise MoEIndexError("config.json is missing model_type")
    try:
        inferred = infer_architecture_adapter(config)
    except AdapterCompatibilityError as exc:
        raise MoEIndexError(str(exc)) from exc
    if adapter == "auto":
        if inferred is None:
            raise MoEIndexError("config.json does not describe a routed MoE model")
        selected_adapter = inferred
        adapter = selected_adapter.id
    else:
        selected_adapter = adapter_by_id(adapter)
        if selected_adapter is None:
            raise MoEIndexError(f"Unknown compiled adapter {adapter!r}")
        if (
            selected_adapter.id != "sytra-generic-moe"
            and (inferred is None or inferred.id != selected_adapter.id)
        ):
            raise MoEIndexError(
                f"{adapter} cannot index model_type {model_type!r}; architecture adapters are exact"
            )
    try:
        validate_moe_config(config, selected_adapter)
    except AdapterCompatibilityError as exc:
        raise MoEIndexError(str(exc)) from exc
    if expert_format == "auto":
        expert_format = _detect_expert_format(config, adapter)
    if adapter == "sytra-kimi-k2.7-code" and expert_format != "packed_int4_group32":
        raise MoEIndexError(
            "sytra-kimi-k2.7-code requires packed_int4_group32; generic INT4 layouts "
            "are not byte-compatible with compressed-tensors pack-quantized weights"
        )
    if expert_format not in WEIGHT_FORMATS:
        raise MoEIndexError(
            f"Unknown expert format {expert_format!r}; choose one of {sorted(WEIGHT_FORMATS)}"
        )

    if expert_regex:
        try:
            custom = re.compile(expert_regex)
        except re.error as exc:
            raise MoEIndexError(f"Invalid expert regex: {exc}") from exc
        if custom.groups != 2:
            raise MoEIndexError("Expert regex must capture exactly (layer, expert)")
        patterns = (custom,)
    else:
        patterns = DEFAULT_EXPERT_PATTERNS

    layers = _positive_int(config, "num_hidden_layers", "n_layer", "n_layers", "num_layers")
    per_layer = _positive_int(
        config,
        "num_local_experts",
        "n_routed_experts",
        "num_experts",
        "moe_num_experts",
    )
    top_k = _positive_int(
        config,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "top_k",
        "moe_top_k",
    )
    if not layers or not per_layer or not top_k:
        raise MoEIndexError(
            "config.json must declare layer count, routed experts per layer, and experts per token"
        )

    shards = sorted(
        path
        for path in root.rglob("*.safetensors")
        if ".cache" not in path.parts and not path.name.startswith(".")
    )
    if not shards:
        raise MoEIndexError(f"No Safetensors shards found under {root}")

    experts: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    dense_tensors: list[dict[str, Any]] = []
    dense_bytes = 0
    seen_tensors: set[str] = set()
    detected_layouts: set[str] = set()
    for shard in shards:
        data_start, header = read_safetensors_header(shard)
        relative = shard.relative_to(root).as_posix()
        data_size = shard.stat().st_size - data_start
        for tensor_name, metadata in header.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name in seen_tensors:
                raise MoEIndexError(f"Tensor {tensor_name!r} appears in multiple shards")
            seen_tensors.add(tensor_name)
            if not isinstance(metadata, dict):
                raise MoEIndexError(f"Tensor metadata is invalid for {tensor_name}")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[0] >= offsets[1]
                or offsets[1] > data_size
            ):
                raise MoEIndexError(f"Tensor offsets are invalid for {tensor_name}")
            length = offsets[1] - offsets[0]
            key = _expert_key(tensor_name, patterns)
            stacked = None
            if key is None and not expert_regex:
                stacked = _stacked_expert_slices(
                    tensor_name,
                    metadata,
                    experts_per_layer=per_layer,
                    allow_merged_rows="merged_rows" in selected_adapter.expert_layouts,
                )
            if stacked is not None:
                layer, layout, slices = stacked
                detected_layouts.add(layout)
                for expert, relative_offset, expert_shape in slices:
                    expert_length = length // per_layer
                    experts[(layer, expert)].append(
                        {
                            "tensor": tensor_name,
                            "dtype": (
                                str(metadata["dtype"])
                                if isinstance(metadata.get("dtype"), str)
                                else None
                            ),
                            "shape": [int(value) for value in expert_shape],
                            "shard": relative,
                            "offset": data_start + relative_offset,
                            "length": expert_length,
                        }
                    )
                continue
            if key is None:
                dense_bytes += length
                dense_tensors.append(
                    {
                        "tensor": tensor_name,
                        "dtype": (
                            str(metadata["dtype"])
                            if isinstance(metadata.get("dtype"), str)
                            else None
                        ),
                        "shape": (
                            [int(value) for value in metadata["shape"]]
                            if isinstance(metadata.get("shape"), list)
                            and all(
                                isinstance(value, int) and value >= 0
                                for value in metadata["shape"]
                            )
                            else []
                        ),
                        "shard": relative,
                        "offset": data_start + offsets[0],
                        "length": length,
                    }
                )
                continue
            detected_layouts.add("discrete")
            experts[key].append(
                {
                    "tensor": tensor_name,
                    "dtype": (
                        str(metadata["dtype"])
                        if isinstance(metadata.get("dtype"), str)
                        else None
                    ),
                    "shape": (
                        [int(value) for value in metadata["shape"]]
                        if isinstance(metadata.get("shape"), list)
                        and all(isinstance(value, int) and value >= 0 for value in metadata["shape"])
                        else []
                    ),
                    "shard": relative,
                    "offset": data_start + offsets[0],
                    "length": length,
                }
            )
    if not experts:
        raise MoEIndexError(
            "No routed experts matched the checkpoint. Supply --expert-regex with "
            "exact (layer, expert) capture groups for this architecture."
        )

    if top_k > per_layer:
        raise MoEIndexError("experts per token cannot exceed experts per layer")
    for layer, expert in experts:
        if layer >= layers or expert >= per_layer:
            raise MoEIndexError(
                f"Tensor index contains layer {layer} expert {expert}, outside config.json"
            )

    moe_layers = sorted({layer for layer, _ in experts})
    for layer in moe_layers:
        missing = [expert for expert in range(per_layer) if (layer, expert) not in experts]
        if missing:
            preview = ", ".join(map(str, missing[:8]))
            suffix = "..." if len(missing) > 8 else ""
            raise MoEIndexError(
                f"Layer {layer} is incomplete: missing routed experts {preview}{suffix}"
            )
        reference = _expert_signature(experts[(layer, 0)], 0)
        for expert in range(1, per_layer):
            signature = _expert_signature(experts[(layer, expert)], expert)
            if signature != reference:
                raise MoEIndexError(
                    f"Layer {layer} expert {expert} has a different tensor signature than expert 0"
                )

    if len(detected_layouts) > 1:
        # Discrete metadata tensors (for example weight_shape) may accompany
        # stacked matrices; the matrix-bearing layout is authoritative.
        detected_layouts.discard("discrete")
    expert_layout = next(iter(detected_layouts), "discrete")
    if expert_layout not in selected_adapter.expert_layouts:
        raise MoEIndexError(
            f"{adapter} does not support detected expert layout {expert_layout}"
        )

    indexed = []
    all_contiguous = True
    for (layer, expert), segments in sorted(experts.items()):
        segments.sort(key=lambda segment: (segment["shard"], segment["offset"], segment["tensor"]))
        contiguous = all(
            left["shard"] == right["shard"]
            and left["offset"] + left["length"] == right["offset"]
            for left, right in zip(segments, segments[1:])
        )
        all_contiguous = all_contiguous and contiguous
        indexed.append({"layer": layer, "expert": expert, "segments": segments})

    architecture = build_architecture_contract(
        config,
        selected_adapter,
        expert_format=expert_format,
        expert_layout=expert_layout,
    )
    architecture["moe_layers"] = moe_layers
    return {
        "schema_version": 1,
        "architecture": architecture,
        "dense_bytes": dense_bytes,
        "storage": {
            "contiguous_experts": all_contiguous,
            "experts": indexed,
            "dense_tensors": sorted(
                dense_tensors,
                key=lambda segment: (
                    segment["tensor"],
                    segment["shard"],
                    segment["offset"],
                ),
            ),
        },
    }


def write_runtime_manifest(model_root: str | Path, manifest: dict[str, Any]) -> Path:
    root = Path(model_root).resolve()
    destination = root / RUNTIME_MANIFEST
    handle, temporary = tempfile.mkstemp(
        prefix=f"{RUNTIME_MANIFEST}.",
        suffix=".tmp",
        dir=root,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination
