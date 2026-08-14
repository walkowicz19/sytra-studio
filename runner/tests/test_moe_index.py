import json
from pathlib import Path

import pytest

from sytra_runner.moe_index import MoEIndexError, build_runtime_manifest


def _write_safetensors(path: Path, tensors: list[tuple[str, bytes]]) -> None:
    cursor = 0
    header = {}
    payload = bytearray()
    for name, data in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [cursor, cursor + len(data)],
        }
        payload.extend(data)
        cursor += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _write_shaped_safetensors(
    path: Path, tensors: list[tuple[str, str, list[int], bytes]]
) -> None:
    cursor = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + len(data)],
        }
        payload.extend(data)
        cursor += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


def _config(root: Path, model_type: str = "test_moe") -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "intermediate_size": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
            }
        ),
        encoding="utf-8",
    )


def test_index_groups_multiple_tensor_spans_without_rewriting_weights(tmp_path):
    _config(tmp_path)
    shard = tmp_path / "model.safetensors"
    tensors = [("model.embed_tokens.weight", b"DENSE")]
    for expert in range(4):
        tensors.extend(
            [
                (f"model.layers.0.mlp.experts.{expert}.gate_proj.weight", b"GATE"),
                (f"model.layers.0.mlp.experts.{expert}.up_proj.weight", b"UP"),
                (f"model.layers.0.mlp.experts.{expert}.down_proj.weight", b"DOWN"),
            ]
        )
    _write_safetensors(shard, tensors)
    before = shard.read_bytes()

    manifest = build_runtime_manifest(
        tmp_path,
        adapter="sytra-generic-moe",
        expert_format="int4_group",
    )

    assert shard.read_bytes() == before
    assert manifest["dense_bytes"] == 5
    assert manifest["storage"]["dense_tensors"][0]["tensor"] == "model.embed_tokens.weight"
    assert manifest["storage"]["dense_tensors"][0]["dtype"] == "U8"
    assert manifest["storage"]["dense_tensors"][0]["shape"] == [5]
    expert = next(
        entry for entry in manifest["storage"]["experts"] if entry["expert"] == 2
    )
    assert (expert["layer"], expert["expert"]) == (0, 2)
    assert {segment["tensor"].split(".")[-2] for segment in expert["segments"]} == {
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert manifest["architecture"]["forward_verified"] is False
    assert manifest["architecture"]["moe_layers"] == [0]


def test_index_splits_qwen_stacked_axis_zero_without_copying(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "moe_intermediate_size": 2,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        ),
        encoding="utf-8",
    )
    shard = tmp_path / "model.safetensors"
    _write_shaped_safetensors(
        shard,
        [("model.layers.0.mlp.experts.gate_proj.weight", "U8", [4, 2, 4], bytes(32))],
    )
    before = shard.read_bytes()

    manifest = build_runtime_manifest(tmp_path)

    assert shard.read_bytes() == before
    assert manifest["architecture"]["adapter"] == "sytra-qwen3-moe"
    assert manifest["architecture"]["expert_layout"] == "stacked_axis0"
    entries = manifest["storage"]["experts"]
    assert len(entries) == 4
    assert [entry["segments"][0]["length"] for entry in entries] == [8] * 4
    assert [entry["segments"][0]["shape"] for entry in entries] == [[2, 4]] * 4


def test_index_splits_legacy_granite_parallel_experts_and_detects_f32(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "granitemoe",
                "architectures": ["GraniteMoeForCausalLM"],
                "torch_dtype": "float32",
                "num_hidden_layers": 1,
                "hidden_size": 4,
                "intermediate_size": 4,
                "num_local_experts": 4,
                "num_experts_per_tok": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        ),
        encoding="utf-8",
    )
    _write_shaped_safetensors(
        tmp_path / "model.safetensors",
        [
            (
                "model.layers.0.block_sparse_moe.input_linear.weight",
                "F32",
                [4, 8, 4],
                bytes(4 * 8 * 4 * 4),
            ),
            (
                "model.layers.0.block_sparse_moe.output_linear.weight",
                "F32",
                [4, 4, 4],
                bytes(4 * 4 * 4 * 4),
            ),
            (
                "model.layers.0.block_sparse_moe.router.layer.weight",
                "F32",
                [4, 4],
                bytes(4 * 4 * 4),
            ),
        ],
    )

    manifest = build_runtime_manifest(tmp_path)

    assert manifest["architecture"]["adapter"] == "sytra-granite-moe"
    assert manifest["architecture"]["expert_format"] == "f32"
    assert manifest["architecture"]["expert_layout"] == "stacked_axis0"
    assert len(manifest["storage"]["experts"]) == 4
    assert {
        tuple(segment["shape"])
        for segment in manifest["storage"]["experts"][0]["segments"]
    } == {(8, 4), (4, 4)}


def test_index_preserves_exact_qwen_packed_int4_group32_triplets(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "architectures": ["Qwen3MoeForCausalLM"],
                "num_hidden_layers": 1,
                "hidden_size": 32,
                "moe_intermediate_size": 32,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
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
            }
        ),
        encoding="utf-8",
    )
    experts = 4
    tensors = [
        (
            "model.layers.0.self_attn.q_proj.weight_packed",
            "I32",
            [32, 4],
            bytes(32 * 4 * 4),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight_scale",
            "BF16",
            [32, 1],
            bytes(32 * 2),
        ),
        (
            "model.layers.0.self_attn.q_proj.weight_shape",
            "I32",
            [2],
            (32).to_bytes(4, "little", signed=True) * 2,
        ),
    ]
    for projection, rows, cols in (
        ("gate_up_proj", 64, 32),
        ("down_proj", 32, 32),
    ):
        prefix = f"model.layers.0.mlp.experts.{projection}"
        tensors.extend(
            [
                (
                    f"{prefix}.weight_packed",
                    "I32",
                    [experts, rows, cols // 8],
                    bytes(experts * rows * (cols // 8) * 4),
                ),
                (
                    f"{prefix}.weight_scale",
                    "BF16",
                    [experts, rows, cols // 32],
                    bytes(experts * rows * (cols // 32) * 2),
                ),
                (
                    f"{prefix}.weight_shape",
                    "I32",
                    [experts, 2],
                    (
                        rows.to_bytes(4, "little", signed=True)
                        + cols.to_bytes(4, "little", signed=True)
                    )
                    * experts,
                ),
            ]
        )
    _write_shaped_safetensors(tmp_path / "model.safetensors", tensors)

    manifest = build_runtime_manifest(tmp_path)

    architecture = manifest["architecture"]
    assert architecture["expert_format"] == "packed_int4_group32"
    assert architecture["expert_layout"] == "stacked_axis0"
    assert architecture["quantization"] == {
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "scale_dtype": "bf16",
    }
    assert len(manifest["storage"]["experts"]) == experts
    assert {
        tensor["tensor"] for tensor in manifest["storage"]["dense_tensors"]
    } == {
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.q_proj.weight_shape",
    }
    first = manifest["storage"]["experts"][0]["segments"]
    assert {segment["tensor"].rsplit(".", 1)[-1] for segment in first} == {
        "weight_packed",
        "weight_scale",
        "weight_shape",
    }
    assert sum(segment["length"] for segment in first) == 1744


def test_index_splits_dbrx_merged_rows(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dbrx",
                "architectures": ["DbrxForCausalLM"],
                "n_layers": 1,
                "d_model": 4,
                "n_heads": 2,
                "ffn_config": {
                    "moe_num_experts": 4,
                    "moe_top_k": 2,
                    "ffn_hidden_size": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    _write_shaped_safetensors(
        tmp_path / "model.safetensors",
        [("transformer.blocks.0.ffn.experts.mlp.w1", "U8", [8, 4], bytes(32))],
    )

    manifest = build_runtime_manifest(tmp_path)

    assert manifest["architecture"]["adapter"] == "sytra-dbrx"
    assert manifest["architecture"]["expert_layout"] == "merged_rows"
    assert [
        entry["segments"][0]["shape"] for entry in manifest["storage"]["experts"]
    ] == [[2, 4]] * 4


def test_known_adapter_rejects_similarly_named_wrong_architecture(tmp_path):
    _config(tmp_path, model_type="kimi_k2")
    _write_safetensors(
        tmp_path / "model.safetensors",
        [("model.layers.0.mlp.experts.0.gate_proj.weight", b"GATE")],
    )

    with pytest.raises(MoEIndexError, match="cannot index"):
        build_runtime_manifest(
            tmp_path,
            adapter="sytra-kimi-k3",
            expert_format="mxfp4",
        )


def test_index_rejects_partial_expert_layer(tmp_path):
    _config(tmp_path)
    _write_safetensors(
        tmp_path / "model.safetensors",
        [("model.layers.0.mlp.experts.0.gate_proj.weight", b"GATE")],
    )

    with pytest.raises(MoEIndexError, match="incomplete"):
        build_runtime_manifest(tmp_path, adapter="sytra-generic-moe")


def test_kimi_k27_rejects_generic_int4_before_scanning_weights(tmp_path):
    config = {
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
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(MoEIndexError, match="packed_int4_group32"):
        build_runtime_manifest(
            tmp_path,
            adapter="sytra-kimi-k2.7-code",
            expert_format="int4_group",
        )
