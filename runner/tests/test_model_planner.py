import json
from pathlib import Path

import pytest

from sytra_runner.model_planner import (
    ModelCompatibilityError,
    build_backend_plan,
    inspect_model,
)


def test_gguf_uses_llama_cpp_with_real_server_command(tmp_path, monkeypatch):
    model = tmp_path / "model-q4_k_m.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1024)
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")

    plan = build_backend_plan(model, vram_limit_mb=512, project_root=tmp_path)

    assert plan.compatible
    assert plan.backend == "llama_cpp"
    assert plan.command[0] == "fake-llama-server"
    assert "-ngl" in plan.command
    ngl = plan.command[plan.command.index("-ngl") + 1]
    assert ngl.isdigit()
    assert ngl != "auto"
    assert "-fit" not in plan.command
    assert plan.llama_offload is not None
    assert plan.llama_offload.mlock is False


def _write_safetensors_model(
    root: Path,
    *,
    missing_second: bool = False,
    config: dict | None = None,
    payload_padding_mb: int = 0,
) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            config or {
                "model_type": "mixtral",
                "architectures": ["MixtralForCausalLM"],
                "num_local_experts": 8,
            }
        ),
        encoding="utf-8",
    )
    def write_tensor(path: Path, tensor_name: str):
        header = json.dumps(
            {
                tensor_name: {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }
            },
            separators=(",", ":"),
        ).encode("utf-8")
        padding = (8 - len(header) % 8) % 8
        header += b" " * padding
        path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")

    write_tensor(root / "model-00001-of-00002.safetensors", "model.embed_tokens.weight")
    if not missing_second:
        write_tensor(root / "model-00002-of-00002.safetensors", "model.layers.0.experts.7.weight")
        if payload_padding_mb:
            with (root / "model-00002-of-00002.safetensors").open("ab") as handle:
                handle.write(b"\0" * payload_padding_mb * 1024 * 1024)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.layers.0.experts.7.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_runtime_manifest(
    root: Path,
    adapter: str,
    model_type: str,
    shard: str = "model-00002-of-00002.safetensors",
) -> None:
    (root / ".sytra-runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": {
                    "adapter": adapter,
                    "model_type": model_type,
                    "attention": "mla",
                    "router": "top_k_weighted",
                    "expert_format": "mxfp4",
                    "num_layers": 4,
                    "experts_per_layer": 8,
                    "experts_per_token": 2,
                    "forward_verified": False,
                },
                "dense_bytes": 1024,
                "storage": {
                    "contiguous_experts": True,
                    "experts": [
                        {
                            "layer": 0,
                            "expert": 0,
                            "segments": [
                                {
                                    "tensor": "gate_proj",
                                    "shard": shard,
                                    "offset": 0,
                                    "length": 4,
                                }
                            ],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_complete_safetensors_model_uses_vllm(tmp_path, monkeypatch):
    model = _write_safetensors_model(tmp_path / "model", payload_padding_mb=2)
    monkeypatch.setenv("SYTRA_VLLM_COMMAND", "fake-vllm")

    plan = build_backend_plan(model, vram_limit_mb=2048)

    assert plan.compatible
    assert plan.backend == "vllm"
    assert plan.artifact.is_moe
    assert plan.command[:2] == ["fake-vllm", "serve"]


def test_vllm_rejects_checkpoint_that_exceeds_budget(tmp_path, monkeypatch):
    model = _write_safetensors_model(tmp_path / "model", payload_padding_mb=2)
    monkeypatch.setenv("SYTRA_VLLM_COMMAND", "fake-vllm")

    plan = build_backend_plan(model, vram_limit_mb=1, ram_limit_mb=0)

    assert not plan.compatible
    assert plan.command == []
    assert any("conservative GPU+RAM budget" in reason for reason in plan.reasons)


def test_incomplete_safetensors_model_fails_before_launch(tmp_path):
    model = _write_safetensors_model(tmp_path / "model", missing_second=True)

    with pytest.raises(ModelCompatibilityError, match="incomplete"):
        inspect_model(model)


def test_vllm_uses_cpu_uva_for_model_that_fits_combined_memory(tmp_path, monkeypatch):
    model = _write_safetensors_model(tmp_path / "model", payload_padding_mb=2)
    monkeypatch.setenv("SYTRA_VLLM_COMMAND", "fake-vllm")

    plan = build_backend_plan(model, vram_limit_mb=1, ram_limit_mb=2048)

    assert plan.compatible
    assert plan.backend == "vllm"
    assert plan.weight_placement.strategy == "vllm-uva"
    assert "--cpu-offload-gb" in plan.command
    assert "--cpu-offload-params" in plan.command


def test_kimi_k3_indexed_checkpoint_routes_to_native_sytra(tmp_path, monkeypatch):
    model = _write_safetensors_model(
        tmp_path / "kimi",
        config={
            "model_type": "kimi_k3",
            "architectures": ["KimiK3ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 8,
            "moe_intermediate_size": 4,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "kv_lora_rank": 16,
            "qk_rope_head_dim": 8,
        },
        payload_padding_mb=2,
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_runtime_manifest(model, "sytra-kimi-k3", "kimi_k3")
    monkeypatch.setenv("SYTRA_ENGINE_COMMAND", "fake-sytra-engine")

    plan = build_backend_plan(
        model,
        vram_limit_mb=1,
        ram_limit_mb=1,
        draft_url="http://127.0.0.1:8081",
        draft_model="tiny-draft",
    )

    assert plan.compatible
    assert plan.backend == "sytra_moe"
    assert plan.artifact.adapter_id == "sytra-kimi-k3"
    assert plan.weight_placement.strategy == "router-aware-vram-ram-nvme"
    assert plan.kv_cache.formulation == "MLA compressed latent"
    assert plan.kv_cache.persistent
    assert plan.command[:2] == ["fake-sytra-engine", "serve"]
    assert plan.preflight_command[:2] == ["fake-sytra-engine", "doctor"]
    assert plan.command[plan.command.index("--ram-limit-mb") + 1] == "1"
    assert plan.command[plan.command.index("--accelerator-limit-mb") + 1] == "1"
    assert plan.command[plan.command.index("--dense-tile-mb") + 1] == "64"
    assert plan.command[plan.command.index("--verification-positions") + 1] == "8"
    assert plan.command[plan.command.index("--storage-bandwidth-mbps") + 1] == "3500"
    assert plan.command[plan.command.index("--target-tps") + 1] == "5.0"
    assert plan.command[plan.command.index("--draft-url") + 1] == "http://127.0.0.1:8081"
    assert plan.command[plan.command.index("--draft-model") + 1] == "tiny-draft"
    assert plan.preflight_command[plan.preflight_command.index("--context") + 1] == "4096"


def test_unknown_oversized_moe_is_not_guessed_into_native_sytra(tmp_path, monkeypatch):
    model = _write_safetensors_model(tmp_path / "unknown", payload_padding_mb=2)
    monkeypatch.setenv("SYTRA_VLLM_COMMAND", "fake-vllm")
    monkeypatch.setenv("SYTRA_ENGINE_COMMAND", "fake-sytra-engine")

    plan = build_backend_plan(model, vram_limit_mb=1, ram_limit_mb=0)

    assert not plan.compatible
    assert plan.backend == "vllm"
    assert plan.artifact.adapter_id is None
    assert any("No verified NVMe adapter" in reason for reason in plan.reasons)


def test_explicit_glm_native_container_is_inspected_without_safetensors(tmp_path, monkeypatch):
    model = tmp_path / "glm52"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmForCausalLM"],
                "num_hidden_layers": 4,
                "hidden_size": 8,
                "moe_intermediate_size": 4,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "kv_lora_rank": 16,
            }
        ),
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "out-layer-00.bin").write_bytes(b"weights")
    _write_runtime_manifest(model, "sytra-glm52", "glm_moe_dsa", "out-layer-00.bin")
    monkeypatch.setenv("SYTRA_ENGINE_COMMAND", "fake-sytra-engine")

    plan = build_backend_plan(model, vram_limit_mb=1024, ram_limit_mb=2048)

    assert plan.compatible
    assert plan.backend == "sytra_moe"
    assert plan.artifact.format == "sytra_moe"
    assert plan.artifact.adapter_id == "sytra-glm52"


def test_unverified_glm_gguf_quant_stays_on_llama_cpp(tmp_path, monkeypatch):
    model = tmp_path / "GLM-5.2-Q3_UNKNOWN.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 2048)
    (tmp_path / ".sytra-model.json").write_text(
        json.dumps({"repo_id": "unsloth/GLM-5.2-GGUF"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")

    plan = build_backend_plan(model, vram_limit_mb=1024, ram_limit_mb=1024)

    assert plan.backend == "llama_cpp"
    assert plan.artifact.adapter_id is None


def test_gguf_metadata_detects_moe_without_filename_guessing(tmp_path, monkeypatch):
    from gguf_test_file import write_metadata_gguf

    model = write_metadata_gguf(
        tmp_path / "weights.gguf",
        {
            "general.architecture": "qwen3moe",
            "qwen3moe.block_count": 12,
            "qwen3moe.expert_count": 64,
            "qwen3moe.expert_used_count": 8,
            "qwen3moe.embedding_length": 1024,
            "qwen3moe.attention.head_count": 16,
            "qwen3moe.attention.head_count_kv": 4,
            "general.file_type": 15,
            "general.parameter_count": 8_000_000_000,
        },
        payload_bytes=4 * 1024 * 1024,
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")

    artifact = inspect_model(model)
    assert artifact.architecture == "qwen3moe"
    assert artifact.is_moe
    assert artifact.n_expert == 64
    assert artifact.n_expert_used == 8
    assert artifact.quantization == "MOSTLY_Q4_K_M"

    plan = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=8192)
    assert plan.compatible
    assert plan.backend == "llama_cpp"
    assert plan.llama_offload is not None
    assert plan.llama_offload.gpu_layers >= 1
    assert plan.estimates["active_params_per_token"] is not None
    assert "qwen2" not in plan.artifact.architecture


def test_gguf_that_exceeds_ram_cap_is_rejected(tmp_path, monkeypatch):
    from gguf_test_file import write_metadata_gguf

    model = write_metadata_gguf(
        tmp_path / "huge.gguf",
        {"general.architecture": "llama", "llama.block_count": 32},
        payload_bytes=80 * 1024 * 1024,
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")

    plan = build_backend_plan(model, vram_limit_mb=64, ram_limit_mb=8)
    assert not plan.compatible
    assert plan.command == []
    assert any("exceed" in reason.lower() or "envelope" in reason.lower() for reason in plan.reasons)

