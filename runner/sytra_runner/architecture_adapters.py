"""Architecture adapters for correctness-gated out-of-core inference.

An adapter is deliberately narrower than a model family name.  It identifies
an exact runtime contract (tensor layout, router semantics, attention/KV
implementation) that an external engine has validated.  Unknown checkpoints
must fall back to a generic backend or fail preflight; they are never guessed
into an out-of-core kernel.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ADAPTER_MARKER = ".sytra-runtime.json"


class AdapterCompatibilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchitectureAdapter:
    id: str
    display_name: str
    engine: str
    engine_arch: str
    family: str = "generic"
    attention_kind: str = "standard"
    router_semantics: str = "top_k_softmax"
    router_score: str = "softmax"
    normalize_selected: bool = True
    expert_layouts: tuple[str, ...] = ("discrete", "stacked_axis0")
    activations: tuple[str, ...] = ("silu",)
    repo_patterns: tuple[str, ...] = ()
    model_type_patterns: tuple[str, ...] = ()
    architecture_patterns: tuple[str, ...] = ()
    file_patterns: tuple[str, ...] = ()
    supports_nvme_streaming: bool = True
    supports_gpu_expert_cache: bool = True
    manages_kv_cache: bool = True
    requires_tokenizer_json: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def matches(
        self,
        *,
        repo_id: str | None,
        model_type: str,
        architectures: Iterable[str],
        file_name: str = "",
    ) -> bool:
        candidates = (
            (repo_id or "", self.repo_patterns),
            (model_type, self.model_type_patterns),
        )
        for value, patterns in candidates:
            if value and any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns):
                if not self.file_patterns or any(
                    re.search(pattern, file_name, re.IGNORECASE)
                    for pattern in self.file_patterns
                ):
                    return True
        architecture_match = any(
            re.search(pattern, architecture, re.IGNORECASE)
            for architecture in architectures
            for pattern in self.architecture_patterns
        )
        return architecture_match and (
            not self.file_patterns
            or any(re.search(pattern, file_name, re.IGNORECASE) for pattern in self.file_patterns)
        )


# These IDs name Sytra-owned architecture contracts. They never select an
# external command. A runtime marker is required because a family name alone
# does not prove that the tensor, router, attention, and quantization contracts
# match a native kernel.
BUILTIN_ADAPTERS: tuple[ArchitectureAdapter, ...] = (
    ArchitectureAdapter(
        id="sytra-glm52",
        display_name="GLM-5.2 native Sytra contract",
        engine="sytra_moe",
        engine_arch="glm",
        family="glm_moe",
        attention_kind="mla",
        router_semantics="group_limited_top_k",
        router_score="sigmoid",
        model_type_patterns=(r"^glm.*moe", r"^glm_moe_dsa$"),
        architecture_patterns=(r"^Glm.*ForCausalLM$",),
        notes="MLA/DSA, routed-expert, quantization, and MTP semantics are adapter-owned.",
    ),
    ArchitectureAdapter(
        id="sytra-kimi-k2.7-code",
        display_name="Kimi K2.7 Code native Sytra contract",
        engine="sytra_moe",
        engine_arch="kimi_k27",
        family="kimi_k27",
        attention_kind="mla",
        router_semantics="no_aux_tc",
        router_score="sigmoid",
        repo_patterns=(r"^moonshotai/Kimi-K2\.7-Code$",),
        model_type_patterns=(r"^kimi_k25$",),
        architecture_patterns=(r"^KimiK25ForConditionalGeneration$",),
        notes=(
            "Text-only Kimi K2.7 Code contract: MLA with compact KV, noaux_tc routing, "
            "and compressed-tensors packed symmetric INT4 group-32 routed experts."
        ),
    ),
    ArchitectureAdapter(
        id="sytra-kimi-k3",
        display_name="Kimi K3 native Sytra contract",
        engine="sytra_moe",
        engine_arch="kimi",
        family="kimi_k3",
        attention_kind="mla",
        router_semantics="group_limited_top_k",
        router_score="sigmoid",
        model_type_patterns=(r"^kimi[-_]?k3$",),
        architecture_patterns=(r"^KimiK3.*ForCausalLM$",),
        notes="Kimi K3 MXFP4 experts and attention remain distinct from Kimi K2/K2.7.",
    ),
    ArchitectureAdapter(
        id="sytra-inkling",
        display_name="Inkling native Sytra contract",
        engine="sytra_moe",
        engine_arch="inkling",
        family="inkling",
        model_type_patterns=(r"inkling",),
        architecture_patterns=(r"Inkling.*ForCausalLM",),
        notes="Inkling dense-weight residency and routed-expert layout are adapter-owned.",
    ),
    ArchitectureAdapter(
        id="sytra-deepseek-v3",
        display_name="DeepSeek V2/V3 native Sytra contract",
        engine="sytra_moe",
        engine_arch="deepseek_v3",
        family="deepseek_v3",
        attention_kind="mla",
        router_semantics="no_aux_tc",
        router_score="sigmoid",
        model_type_patterns=(r"^deepseek_v[23]$",),
        architecture_patterns=(r"^DeepseekV[23].*ForCausalLM$",),
        notes="MLA plus sigmoid noaux_tc routing; dimensions are checkpoint-derived.",
    ),
    ArchitectureAdapter(
        id="sytra-qwen3-moe",
        display_name="Qwen3 MoE native Sytra contract",
        engine="sytra_moe",
        engine_arch="qwen3_moe",
        family="qwen3_moe",
        model_type_patterns=(r"^qwen3_moe$", r"^qwen3_next$"),
        architecture_patterns=(r"^Qwen3(?:Moe|Next).*ForCausalLM$",),
        notes="Standard/GQA or hybrid attention with normalized softmax top-k routing.",
    ),
    ArchitectureAdapter(
        id="sytra-qwen2-moe",
        display_name="Qwen2 MoE native Sytra contract",
        engine="sytra_moe",
        engine_arch="qwen2_moe",
        family="qwen2_moe",
        model_type_patterns=(r"^qwen2_moe$",),
        architecture_patterns=(r"^Qwen2Moe.*ForCausalLM$",),
    ),
    ArchitectureAdapter(
        id="sytra-mixtral",
        display_name="Mixtral native Sytra contract",
        engine="sytra_moe",
        engine_arch="mixtral",
        family="mixtral",
        model_type_patterns=(r"^mixtral$",),
        architecture_patterns=(r"^Mixtral.*ForCausalLM$",),
        expert_layouts=("discrete", "stacked_axis0"),
    ),
    ArchitectureAdapter(
        id="sytra-olmoe",
        display_name="OLMoE native Sytra contract",
        engine="sytra_moe",
        engine_arch="olmoe",
        family="olmoe",
        router_semantics="top_k_weighted",
        normalize_selected=False,
        model_type_patterns=(r"^olmoe$",),
        architecture_patterns=(r"^Olmoe.*ForCausalLM$",),
    ),
    ArchitectureAdapter(
        id="sytra-dbrx",
        display_name="DBRX native Sytra contract",
        engine="sytra_moe",
        engine_arch="dbrx",
        family="dbrx",
        model_type_patterns=(r"^dbrx$",),
        architecture_patterns=(r"^Dbrx.*ForCausalLM$",),
        expert_layouts=("merged_rows", "stacked_axis0"),
        activations=("silu", "gelu"),
    ),
    ArchitectureAdapter(
        id="sytra-granite-moe",
        display_name="Granite MoE native Sytra contract",
        engine="sytra_moe",
        engine_arch="granite_moe",
        family="granite_moe",
        model_type_patterns=(r"^granite[_-]?moe", r"^granitemoe"),
        architecture_patterns=(r"^GraniteMoe.*ForCausalLM$",),
        activations=("silu", "gelu"),
    ),
    ArchitectureAdapter(
        id="sytra-arctic",
        display_name="Snowflake Arctic native Sytra contract",
        engine="sytra_moe",
        engine_arch="arctic",
        family="arctic",
        model_type_patterns=(r"^arctic",),
        architecture_patterns=(r"^Arctic.*ForCausalLM$",),
        activations=("silu", "gelu"),
    ),
    ArchitectureAdapter(
        id="sytra-minimax-moe",
        display_name="MiniMax MoE native Sytra contract",
        engine="sytra_moe",
        engine_arch="minimax_moe",
        family="minimax_moe",
        model_type_patterns=(r"^minimax",),
        architecture_patterns=(r"^MiniMax.*ForCausalLM$",),
    ),
    ArchitectureAdapter(
        id="sytra-generic-moe",
        display_name="Generic MoE storage contract",
        engine="sytra_moe",
        engine_arch="generic_moe",
        family="generic",
        attention_kind="custom",
        router_semantics="custom",
        router_score="softmax",
        expert_layouts=("discrete", "stacked_axis0", "merged_rows"),
        activations=("silu", "gelu", "gelu_tanh", "relu", "relu2"),
        requires_tokenizer_json=False,
        notes="Indexes and streams unknown MoEs; storage-only until promoted to a compiled family.",
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterCompatibilityError(f"Invalid adapter metadata in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterCompatibilityError(f"Adapter metadata must be a JSON object: {path}")
    return value


def adapter_by_id(adapter_id: str) -> ArchitectureAdapter | None:
    return next((adapter for adapter in BUILTIN_ADAPTERS if adapter.id == adapter_id), None)


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the language-model config from common multimodal wrappers."""

    for key in ("text_config", "language_config", "llm_config"):
        nested = config.get(key)
        if isinstance(nested, dict):
            language = dict(nested)
            break
    else:
        language = dict(config)
    ffn = language.get("ffn_config") or config.get("ffn_config")
    if isinstance(ffn, dict):
        for key, value in ffn.items():
            language.setdefault(key, value)
    return language


def _positive(source: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def infer_architecture_adapter(
    config: dict[str, Any], *, repo_id: str | None = None
) -> ArchitectureAdapter | None:
    """Infer a trusted built-in profile while creating a runtime manifest.

    Serving never calls this function: it requires the generated marker and
    the native compiled allowlist. Unknown but structurally valid MoEs receive
    the storage-only generic profile.
    """

    language = _model_config(config)
    model_types = {
        str(value)
        for value in (config.get("model_type"), language.get("model_type"))
        if value
    }
    architecture_values: list[str] = []
    for source in (config, language):
        architectures = source.get("architectures") or ()
        if isinstance(architectures, (list, tuple)):
            architecture_values.extend(str(value) for value in architectures)
    matches = [
        adapter
        for adapter in BUILTIN_ADAPTERS
        if adapter.id != "sytra-generic-moe"
        and any(
            adapter.matches(
                repo_id=repo_id,
                model_type=model_type,
                architectures=architecture_values,
            )
            for model_type in model_types or {""}
        )
    ]
    if len(matches) > 1:
        raise AdapterCompatibilityError(
            "Checkpoint matches multiple compiled MoE profiles: "
            + ", ".join(adapter.id for adapter in matches)
        )
    if matches:
        validate_moe_config(config, matches[0])
        return matches[0]
    experts = _positive(
        language,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
    )
    top_k = _positive(
        language,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "moe_top_k",
    )
    if experts and top_k:
        return adapter_by_id("sytra-generic-moe")
    return None


def validate_moe_config(config: dict[str, Any], adapter: ArchitectureAdapter) -> None:
    language = _model_config(config)
    layers = _positive(language, "num_hidden_layers", "n_layers", "num_layers")
    hidden = _positive(language, "hidden_size", "d_model", "model_dim")
    experts = _positive(
        language,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
    )
    top_k = _positive(
        language,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "moe_top_k",
    )
    intermediate = _positive(
        language,
        "moe_intermediate_size",
        "expert_intermediate_size",
        "ffn_hidden_size",
        "intermediate_size",
    )
    missing = [
        name
        for name, value in (
            ("layers", layers),
            ("hidden size", hidden),
            ("expert count", experts),
            ("experts per token", top_k),
            ("expert intermediate size", intermediate),
        )
        if not value
    ]
    if missing:
        raise AdapterCompatibilityError(
            f"{adapter.id} config is missing " + ", ".join(missing)
        )
    if top_k > experts:
        raise AdapterCompatibilityError("experts per token exceeds the expert count")
    activation = str(language.get("hidden_act") or language.get("activation_function") or "silu")
    normalized_activation = {
        "swiglu": "silu",
        "gelu_pytorch_tanh": "gelu_tanh",
    }.get(activation, activation)
    if normalized_activation not in adapter.activations:
        raise AdapterCompatibilityError(
            f"{adapter.id} does not support activation {activation!r}"
        )
    if adapter.attention_kind == "mla" and not _positive(language, "kv_lora_rank"):
        raise AdapterCompatibilityError(f"{adapter.id} requires kv_lora_rank for MLA")
    if adapter.id == "sytra-kimi-k2.7-code":
        _validate_kimi_k27_config(config)


def build_architecture_contract(
    config: dict[str, Any],
    adapter: ArchitectureAdapter,
    *,
    expert_format: str,
    expert_layout: str,
) -> dict[str, Any]:
    """Build the versioned data-only contract consumed by the Rust engine."""

    language = _model_config(config)
    validate_moe_config(config, adapter)
    layers = _positive(language, "num_hidden_layers", "n_layers", "num_layers") or 0
    experts = _positive(
        language,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
    ) or 0
    top_k = _positive(
        language,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "moe_top_k",
    ) or 0
    hidden = _positive(language, "hidden_size", "d_model", "model_dim") or 0
    intermediate = _positive(
        language,
        "moe_intermediate_size",
        "expert_intermediate_size",
        "ffn_hidden_size",
        "intermediate_size",
    ) or 0
    activation = str(language.get("hidden_act") or language.get("activation_function") or "silu")
    activation = {"swiglu": "silu", "gelu_pytorch_tanh": "gelu_tanh"}.get(
        activation, activation
    )
    groups = _positive(language, "n_group", "num_expert_groups") or 1
    selected_groups = _positive(language, "topk_group", "num_limited_groups") or groups
    normalize = bool(language.get("norm_topk_prob", adapter.normalize_selected))
    scaling = language.get("routed_scaling_factor", 1.0)
    if not isinstance(scaling, (int, float)) or isinstance(scaling, bool):
        scaling = 1.0
    attention = adapter.attention_kind
    if attention == "standard" and bool(language.get("use_sliding_window")):
        attention = "sliding_window"
    if adapter.family == "qwen3_moe" and "qwen3_next" in str(language.get("model_type", "")):
        attention = "hybrid"
    heads = _positive(language, "num_attention_heads", "n_heads") or 0
    kv_heads = _positive(language, "num_key_value_heads", "n_kv_heads") or heads
    head_dim = _positive(language, "head_dim") or (hidden // heads if heads else 0)
    quant = _quantization_contract(language, expert_format)
    return {
        "adapter": adapter.id,
        "model_type": str(config.get("model_type") or language.get("model_type") or ""),
        "family": adapter.family,
        "attention": attention,
        "router": adapter.router_semantics,
        "expert_format": expert_format,
        "expert_layout": expert_layout,
        "activation": activation,
        "router_config": {
            "score": adapter.router_score,
            "normalize_selected": normalize,
            "scaling_factor": float(scaling),
            "correction_bias": adapter.router_semantics == "no_aux_tc",
            "groups": groups,
            "selected_groups": selected_groups,
        },
        "attention_config": {
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "q_lora_rank": _positive(language, "q_lora_rank") or 0,
            "kv_lora_rank": _positive(language, "kv_lora_rank") or 0,
            "qk_nope_head_dim": _positive(language, "qk_nope_head_dim") or 0,
            "qk_rope_head_dim": _positive(language, "qk_rope_head_dim") or 0,
            "value_head_dim": _positive(language, "v_head_dim") or head_dim,
        },
        "quantization": quant,
        "hidden_size": hidden,
        "expert_intermediate_size": intermediate,
        "num_layers": layers,
        "experts_per_layer": experts,
        "experts_per_token": top_k,
        "forward_verified": False,
    }


def _quantization_contract(config: dict[str, Any], expert_format: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bits": 0,
        "group_size": 0,
        "symmetric": False,
        "scale_dtype": None,
    }
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        groups = quant.get("config_groups")
        group = groups.get("group_0") if isinstance(groups, dict) else None
        weights = group.get("weights") if isinstance(group, dict) else None
        if isinstance(weights, dict):
            result.update(
                bits=int(weights.get("num_bits") or 0),
                group_size=int(weights.get("group_size") or 0),
                symmetric=bool(weights.get("symmetric")),
            )
    if expert_format == "packed_int4_group32":
        result.update(bits=4, group_size=32, symmetric=True, scale_dtype="bf16")
    elif expert_format == "int4_group":
        result["bits"] = result["bits"] or 4
    elif expert_format == "int8":
        result["bits"] = 8
    elif expert_format in {"fp8_e4m3", "mxfp4", "nvfp4"}:
        result["bits"] = 8 if expert_format == "fp8_e4m3" else 4
    return result


def resolve_architecture_adapter(
    model_root: str | Path,
    *,
    config: dict[str, Any],
    manifest: dict[str, Any] | None = None,
    file_name: str = "",
) -> ArchitectureAdapter | None:
    """Resolve a built-in adapter without trusting downloaded executable code."""

    root = Path(model_root)
    marker = root / ADAPTER_MARKER
    if marker.is_file():
        metadata = _load_json(marker)
        architecture = metadata.get("architecture")
        adapter_id = (
            architecture.get("adapter")
            if isinstance(architecture, dict)
            else metadata.get("adapter")
        )
        if not isinstance(adapter_id, str):
            raise AdapterCompatibilityError(
                f"{marker} must contain architecture.adapter as a string"
            )
        adapter = adapter_by_id(adapter_id)
        if adapter is None:
            raise AdapterCompatibilityError(
                f"Unknown architecture adapter {adapter_id!r}; supported adapters are "
                + ", ".join(item.id for item in BUILTIN_ADAPTERS)
            )
        model_type = str(config.get("model_type") or "")
        architectures = config.get("architectures") or ()
        if not isinstance(architectures, (list, tuple)):
            architectures = ()
        if adapter.id != "sytra-generic-moe" and (model_type or architectures) and not adapter.matches(
            repo_id=None,
            model_type=model_type,
            architectures=(str(value) for value in architectures),
            file_name=file_name,
        ):
            raise AdapterCompatibilityError(
                f"{adapter.id} does not match the checkpoint's config.json architecture"
            )
        validate_moe_config(config, adapter)
        return adapter
    return None


def _validate_kimi_k27_config(config: dict[str, Any]) -> None:
    """Reject lookalike Kimi checkpoints before the native engine sees bytes."""

    text = config.get("text_config")
    if not isinstance(text, dict):
        raise AdapterCompatibilityError("Kimi K2.7 Code requires text_config")
    exact: dict[str, Any] = {
        "model_type": "kimi_k2",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_attention_heads": 64,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 1,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
    }
    mismatches = [
        f"{key}={text.get(key)!r} (expected {expected!r})"
        for key, expected in exact.items()
        if text.get(key) != expected
    ]
    quant = text.get("quantization_config")
    weights: Any = None
    if isinstance(quant, dict):
        groups = quant.get("config_groups")
        group = groups.get("group_0") if isinstance(groups, dict) else None
        weights = group.get("weights") if isinstance(group, dict) else None
    if not isinstance(quant, dict) or quant.get("format") != "pack-quantized":
        mismatches.append("quantization format is not pack-quantized")
    expected_quant = {
        "num_bits": 4,
        "group_size": 32,
        "strategy": "group",
        "type": "int",
        "symmetric": True,
    }
    if not isinstance(weights, dict):
        mismatches.append("quantization group_0.weights is missing")
    else:
        mismatches.extend(
            f"weights.{key}={weights.get(key)!r} (expected {expected!r})"
            for key, expected in expected_quant.items()
            if weights.get(key) != expected
        )
    if mismatches:
        raise AdapterCompatibilityError(
            "Checkpoint is not the compiled Kimi K2.7 Code contract: "
            + "; ".join(mismatches)
        )


def validate_adapter_payload(
    model_root: str | Path,
    adapter: ArchitectureAdapter,
) -> tuple[int, list[str]]:
    """Perform engine-independent checks; the engine's doctor owns tensor checks."""

    root = Path(model_root)
    if not (root / "config.json").is_file():
        raise AdapterCompatibilityError(f"{adapter.display_name} requires config.json")
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tiktoken.model",
        "vocab.json",
    )
    if adapter.requires_tokenizer_json and not any(
        (root / name).is_file() for name in tokenizer_files
    ):
        raise AdapterCompatibilityError(
            f"{adapter.display_name} requires a supported tokenizer payload"
        )

    ignored = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        ".sytra-model.json",
        ADAPTER_MARKER,
    }
    payloads = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in ignored
        and ".cache" not in path.parts
        and not path.name.startswith(".")
        and path.stat().st_size > 0
        and (
            path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
            or path.name.startswith("out-")
        )
    ]
    if not payloads:
        raise AdapterCompatibilityError(
            f"{adapter.display_name} has no non-empty weight payloads under {root}"
        )
    return sum(path.stat().st_size for path in payloads), [str(path) for path in payloads]
