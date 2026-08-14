import json

import pytest

from sytra_runner.architecture_adapters import (
    ADAPTER_MARKER,
    AdapterCompatibilityError,
    resolve_architecture_adapter,
)
from sytra_runner.memory_hierarchy import estimate_kv_cache, plan_weight_placement


def _kimi_k27_config() -> dict:
    return {
        "model_type": "kimi_k25",
        "architectures": ["KimiK25ForConditionalGeneration"],
        "text_config": {
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
            "quantization_config": {
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "group_size": 32,
                            "strategy": "group",
                            "type": "int",
                            "symmetric": True,
                        }
                    }
                },
            },
        },
    }


def test_adapter_resolution_is_exact_and_does_not_guess_kimi_k2(tmp_path):
    kimi3_root = tmp_path / "kimi3"
    kimi3_root.mkdir()
    (kimi3_root / ADAPTER_MARKER).write_text(
        json.dumps({"architecture": {"adapter": "sytra-kimi-k3"}}),
        encoding="utf-8",
    )
    kimi3 = resolve_architecture_adapter(
        kimi3_root,
        config={
            "model_type": "kimi_k3",
            "num_hidden_layers": 1,
            "hidden_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "kv_lora_rank": 8,
        },
    )
    kimi2 = resolve_architecture_adapter(
        tmp_path,
        config={"model_type": "kimi_k2"},
    )
    unindexed_kimi3 = resolve_architecture_adapter(
        tmp_path,
        config={"model_type": "kimi_k3"},
    )

    assert kimi3 is not None
    assert kimi3.id == "sytra-kimi-k3"
    assert kimi2 is None
    assert unindexed_kimi3 is None


def test_explicit_marker_can_select_only_a_builtin_adapter(tmp_path):
    (tmp_path / ADAPTER_MARKER).write_text(
        json.dumps({"architecture": {"adapter": "remote-code-from-model"}}),
        encoding="utf-8",
    )

    with pytest.raises(AdapterCompatibilityError, match="Unknown architecture adapter"):
        resolve_architecture_adapter(tmp_path, config={})


def test_runtime_marker_cannot_apply_kimi_k3_kernel_to_kimi_k2(tmp_path):
    (tmp_path / ADAPTER_MARKER).write_text(
        json.dumps({"architecture": {"adapter": "sytra-kimi-k3"}}),
        encoding="utf-8",
    )

    with pytest.raises(AdapterCompatibilityError, match="does not match"):
        resolve_architecture_adapter(tmp_path, config={"model_type": "kimi_k2"})


def test_kimi_k27_marker_requires_exact_inner_and_quantization_contract(tmp_path):
    (tmp_path / ADAPTER_MARKER).write_text(
        json.dumps({"architecture": {"adapter": "sytra-kimi-k2.7-code"}}),
        encoding="utf-8",
    )
    config = _kimi_k27_config()
    adapter = resolve_architecture_adapter(tmp_path, config=config)
    assert adapter is not None
    assert adapter.id == "sytra-kimi-k2.7-code"

    config["text_config"]["quantization_config"]["config_groups"]["group_0"]["weights"][
        "group_size"
    ] = 64
    with pytest.raises(AdapterCompatibilityError, match="group_size"):
        resolve_architecture_adapter(tmp_path, config=config)


def test_standard_and_mla_kv_estimates_are_context_sensitive():
    standard = estimate_kv_cache(
        {
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
        },
        context_tokens=100,
        dtype="fp16",
        cpu_cache=False,
    )
    mla = estimate_kv_cache(
        {
            "num_hidden_layers": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
        },
        context_tokens=100,
        dtype="q8_0",
        cpu_cache=True,
        persistent=True,
    )

    assert standard.estimated_bytes == 7040
    assert standard.formulation == "standard K+V heads"
    assert mla.estimated_bytes == 1320
    assert mla.tier == "cpu"
    assert mla.persistent


def test_weight_placement_uses_nvme_only_when_adapter_allows_it():
    without_nvme = plan_weight_placement(
        weight_bytes=10 * 1024 * 1024,
        vram_budget_mb=1,
        ram_budget_mb=1,
        allow_cpu_offload=True,
        allow_nvme_streaming=False,
    )
    with_nvme = plan_weight_placement(
        weight_bytes=10 * 1024 * 1024,
        vram_budget_mb=1,
        ram_budget_mb=1,
        allow_cpu_offload=True,
        allow_nvme_streaming=True,
    )

    assert without_nvme.strategy == "insufficient-memory"
    assert with_nvme.strategy == "router-aware-vram-ram-nvme"
    assert with_nvme.estimated_nvme_weight_bytes > 0


@pytest.mark.parametrize(
    ("model_type", "architecture", "adapter_id"),
    [
        ("deepseek_v3", "DeepseekV3ForCausalLM", "sytra-deepseek-v3"),
        ("qwen3_moe", "Qwen3MoeForCausalLM", "sytra-qwen3-moe"),
        ("mixtral", "MixtralForCausalLM", "sytra-mixtral"),
        ("olmoe", "OlmoeForCausalLM", "sytra-olmoe"),
        ("dbrx", "DbrxForCausalLM", "sytra-dbrx"),
    ],
)
def test_major_moe_families_have_distinct_compiled_profiles(
    model_type, architecture, adapter_id
):
    from sytra_runner.architecture_adapters import infer_architecture_adapter

    config = {
        "model_type": model_type,
        "architectures": [architecture],
        "num_hidden_layers": 2,
        "hidden_size": 32,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
    }
    if model_type == "deepseek_v3":
        config.update(kv_lora_rank=8, n_routed_experts=8)
    adapter = infer_architecture_adapter(config)
    assert adapter is not None
    assert adapter.id == adapter_id
