import json
import struct
from pathlib import Path

import pytest

from sytra_runner.helix_engine import (
    build_helix_resource_plan,
    classify_hardware_requirements,
    select_helix_hardware_profile,
)
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


def test_live_vram_guardrail_limits_budget_and_fails_closed(tmp_path, monkeypatch):
    import sytra_runner.model_planner as planner
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1024)
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setattr(planner, "_first_visible_gpu_free_mb", lambda: 2048)
    plan = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, vram_guardrail_mb=512)
    assert plan.available_vram_mb == 1536
    assert any("2048 MiB free" in warning for warning in plan.warnings)
    with pytest.raises(ModelCompatibilityError, match="does not exceed"):
        build_backend_plan(model, vram_limit_mb=4096, vram_guardrail_mb=2048)
    monkeypatch.setattr(planner, "_first_visible_gpu_free_mb", lambda: None)
    with pytest.raises(ModelCompatibilityError, match="Cannot read free NVIDIA VRAM"):
        build_backend_plan(model, vram_limit_mb=4096, vram_guardrail_mb=512)


def _draft_gguf(path, tokens):
    def string(value):
        value = value.encode()
        return struct.pack('<Q', len(value)) + value
    fields = [('general.architecture', 8, string('llama')),
              ('llama.block_count', 4, struct.pack('<I', 2)),
              ('tokenizer.ggml.model', 8, string('llama')),
              ('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, len(tokens)) + b''.join(map(string, tokens)))]
    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(fields)) +
                     b''.join(string(key) + struct.pack('<I', typ) + value for key, typ, value in fields))


def test_draft_model_requires_matching_tokenizer_and_supported_runtime(tmp_path, monkeypatch):
    import sytra_runner.model_planner as planner
    main, draft = tmp_path / 'main.gguf', tmp_path / 'draft.gguf'
    _draft_gguf(main, ['a', 'b'])
    _draft_gguf(draft, ['a', 'b'])
    monkeypatch.setenv('SYTRA_LLAMA_SERVER', 'fake-llama-server')
    monkeypatch.setattr(planner, '_visible_gpu_memory_mb', lambda: [])
    monkeypatch.setattr(planner, '_llama_server_help', lambda _: '--spec-draft-model --spec-draft-ngl --spec-type draft-simple')
    plan = build_backend_plan(main, draft_model_path=str(draft), vram_limit_mb=512, ram_limit_mb=4096)
    assert plan.compatible
    assert plan.command[plan.command.index('--spec-draft-model') + 1] == str(draft.resolve())
    assert plan.command[plan.command.index('--spec-draft-ngl') + 1] == '0'
    assert plan.command[plan.command.index('--spec-type') + 1] == 'draft-simple'
    assert plan.estimates['draft_ram_reserve_mb'] >= 1024
    with pytest.raises(ModelCompatibilityError, match='RAM'):
        build_backend_plan(main, draft_model_path=str(draft), vram_limit_mb=512, ram_limit_mb=500)
    _draft_gguf(draft, ['a', 'different'])
    with pytest.raises(ModelCompatibilityError, match='tokenizer'):
        build_backend_plan(main, draft_model_path=str(draft), vram_limit_mb=512, ram_limit_mb=4096)
    _draft_gguf(draft, ['a', 'b'])
    monkeypatch.setattr(planner, '_llama_server_help', lambda _: '--model-draft')
    with pytest.raises(ModelCompatibilityError, match='does not advertise'):
        build_backend_plan(main, draft_model_path=str(draft), vram_limit_mb=512, ram_limit_mb=4096)


def test_live_vram_queries_first_selected_device(monkeypatch):
    import sytra_runner.model_planner as planner
    from types import SimpleNamespace
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1")
    monkeypatch.setattr(planner.shutil, "which", lambda _: "nvidia-smi")
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="1234\n")
    monkeypatch.setattr(planner.subprocess, "run", run)
    assert planner._first_visible_gpu_free_mb() == 1234
    assert "--id=2" in calls[0]


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
                "num_experts_per_tok": 2,
                "num_hidden_layers": 32,
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
    assert plan.estimates is not None
    assert plan.estimates["helix_resource_plan"] is not None


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


def test_legacy_colibri_backend_is_rejected(tmp_path):
    model = _write_safetensors_model(tmp_path / "moe", payload_padding_mb=2)
    with pytest.raises(ModelCompatibilityError, match="Unknown inference backend"):
        build_backend_plan(model, backend="colibri")


def test_helix_profiles_cover_4gb_8gb_12gb_and_cpu():
    assert select_helix_hardware_profile(vram_limit_mb=4096, gpu_visible=True) == "vram_4gb_windowed"
    assert select_helix_hardware_profile(vram_limit_mb=8192, gpu_visible=True) == "vram_8gb_hybrid"
    assert select_helix_hardware_profile(vram_limit_mb=12288, gpu_visible=True) == "vram_12gb_balanced"
    assert select_helix_hardware_profile(vram_limit_mb=24576, gpu_visible=True) == "vram_16gb_plus_prosumer"
    assert select_helix_hardware_profile(vram_limit_mb=8192, gpu_visible=True, allow_cpu_only=True) == "cpu_only_native"


def test_helix_titan_700b_hardware_requirements_and_resource_plan():
    req = classify_hardware_requirements(
        params_b=744.0,
        weight_bytes=372 * 1024**3,
        n_expert=256,
        n_expert_used=8,
    )
    assert req.scale_tier == "titan_700b_moe"
    assert req.min_vram_mb == 4096
    assert req.recommended_vram_mb == 24576
    assert req.min_ram_mb == 32768
    assert req.recommended_ram_mb == 65536
    assert req.supports_4gb_vram is True
    assert req.supports_8gb_vram is True
    assert req.supports_12gb_vram is True
    assert req.supports_cpu_only is True

    # Test 4GB VRAM windowed execution on a 744B Titan MoE model
    plan_4gb = build_helix_resource_plan(
        weight_bytes=372 * 1024**3,
        vram_limit_mb=4096,
        ram_limit_mb=65536,
        nvme_weight_bytes=300 * 1024**3,
        storage_bandwidth_mbps=7000,
        target_tps=5.0,
        strategy="router-aware-vram-ram-nvme",
        io_max_tps=3.5,
        gpu_visible=True,
        n_layer=80,
        n_expert=256,
        n_expert_used=8,
        adapter_id="sytra-glm52",
        architecture="GlmForCausalLM",
        config={"num_hidden_layers": 80, "hidden_size": 7168, "n_routed_experts": 256, "num_experts_per_tok": 8},
    )
    assert plan_4gb.profile == "vram_4gb_windowed"
    assert plan_4gb.layer_window_size == 1
    assert plan_4gb.cpu_expert_coexecution is True
    assert plan_4gb.ngram_speculative_cache is True
    assert plan_4gb.mtp_draft_heads is True
    assert plan_4gb.cuda_streams == 3
    assert plan_4gb.meets_minimum_envelope is True
    assert plan_4gb.effective_estimated_tps is not None and plan_4gb.effective_estimated_tps > 3.5


def test_sytra_moe_rejects_unsustainable_storage_target_tps(tmp_path, monkeypatch):
    model = _write_safetensors_model(
        tmp_path / "deepseek",
        config={
            "model_type": "deepseek_v3",
            "architectures": ["DeepseekV3ForCausalLM"],
            "num_hidden_layers": 4,
            "hidden_size": 32,
            "moe_intermediate_size": 16,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "kv_lora_rank": 8,
        },
        payload_padding_mb=8,
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    _write_runtime_manifest(model, "sytra-deepseek-v3", "deepseek_v3")
    monkeypatch.setenv("SYTRA_ENGINE_COMMAND", "fake-sytra-engine")

    plan = build_backend_plan(
        model,
        vram_limit_mb=1,
        ram_limit_mb=1,
        storage_bandwidth_mbps=1,
        target_tps=50.0,
    )

    assert plan.backend == "sytra_moe"
    assert not plan.compatible
    assert plan.command == []
    assert any("below --target-tps" in reason for reason in plan.reasons)


def test_kv_cache_quant_backend_flags_and_flash_attention_contract(tmp_path, monkeypatch):
    import sytra_runner.model_planner as planner
    from gguf_test_file import write_metadata_gguf

    model = write_metadata_gguf(
        tmp_path / "dense.gguf",
        {
            "general.architecture": "llama",
            "llama.block_count": 4,
            "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 2,
            "llama.embedding_length": 512,
        },
        payload_bytes=2 * 1024 * 1024,
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setattr(planner, "_llama_server_help", lambda _: "-fa -ctk -ctv")

    fp16 = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="fp16", flash_attention=False)
    assert fp16.compatible
    assert fp16.command[fp16.command.index("-ctk") + 1] == "f16"
    assert fp16.command[fp16.command.index("-ctv") + 1] == "f16"

    q8_ok = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="q8_0", flash_attention=True)
    assert q8_ok.compatible
    assert q8_ok.command[q8_ok.command.index("-ctk") + 1] == "q8_0"

    q4_ok = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="q4_0", flash_attention=True)
    assert q4_ok.compatible
    assert q4_ok.command[q4_ok.command.index("-ctk") + 1] == "q4_0"
    assert q4_ok.kv_cache.estimated_bytes < fp16.kv_cache.estimated_bytes
    assert any("quality" in w.lower() or "perplexity" in w.lower() for w in q4_ok.warnings)

    q4_no_fa = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="q4_0", flash_attention=False)
    assert not q4_no_fa.compatible
    assert q4_no_fa.command == []
    assert any("Flash Attention" in r for r in q4_no_fa.reasons)

    with pytest.raises(ModelCompatibilityError, match="Unsupported KV cache"):
        build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="int2")


def test_vllm_honestly_reports_unsupported_ggml_kv_cache_and_unknown_hardware_fails_closed(tmp_path, monkeypatch):
    import sytra_runner.model_planner as planner

    model = _write_safetensors_model(tmp_path / "dense", payload_padding_mb=2)
    monkeypatch.setenv("SYTRA_VLLM_COMMAND", "fake-vllm")

    plan = build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=4096, kv_cache_quant="q4_0")
    assert plan.compatible
    assert plan.kv_cache.dtype == "fp16"
    assert any("does not support ggml q4_0" in w for w in plan.warnings)

    monkeypatch.setattr(planner, "_system_memory_mb", lambda: 0)
    with pytest.raises(ModelCompatibilityError, match="detect system RAM"):
        build_backend_plan(model, vram_limit_mb=4096, ram_limit_mb=None)

    monkeypatch.setattr(planner, "_system_memory_mb", lambda: 8192)
    monkeypatch.setattr(planner, "_visible_gpu_memory_mb", lambda: [])
    with pytest.raises(ModelCompatibilityError, match="detect GPU VRAM"):
        build_backend_plan(model, vram_limit_mb=None)

    monkeypatch.setattr(planner, "_visible_gpu_memory_mb", lambda: [4096])
    capped = build_backend_plan(model, vram_limit_mb=16384)
    assert capped.available_vram_mb == 4096
    assert any("capped to 4096 MiB" in w for w in capped.warnings)


def test_draft_model_rejects_mismatched_architecture(tmp_path, monkeypatch):
    import sytra_runner.model_planner as planner

    def write_arch_gguf(path, arch):
        def s(v):
            b = v.encode()
            return struct.pack("<Q", len(b)) + b
        fields = [
            ("general.architecture", 8, s(arch)),
            (f"{arch}.block_count", 4, struct.pack("<I", 2)),
            ("tokenizer.ggml.model", 8, s("llama")),
            ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + s("a") + s("b")),
        ]
        path.write_bytes(
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, len(fields))
            + b"".join(s(k) + struct.pack("<I", t) + v for k, t, v in fields)
        )

    main = tmp_path / "main.gguf"
    draft = tmp_path / "draft.gguf"
    write_arch_gguf(main, "llama")
    write_arch_gguf(draft, "qwen2")
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setattr(planner, "_visible_gpu_memory_mb", lambda: [])
    monkeypatch.setattr(planner, "_llama_server_help", lambda _: "--spec-draft-model --spec-draft-ngl")

    with pytest.raises(ModelCompatibilityError, match="architectures must match"):
        build_backend_plan(main, draft_model_path=str(draft), vram_limit_mb=512, ram_limit_mb=4096)
