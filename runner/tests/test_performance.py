import json
from pathlib import Path

import pytest
import yaml

from sytra_runner.config import RunConfig, ConfigError
from sytra_runner.adapters import resolve_training_source
from sytra_runner.performance.hardware import HardwareProfile, detect_hardware
from sytra_runner.performance.planner import training_plan, merge_plan
from sytra_runner.performance.planner import validate_performance
from sytra_runner.performance.moe import detect_moe
from sytra_runner.performance.checkpoints import checkpoint_fingerprint, checkpoint_memory_gb
from sytra_runner.memory_hierarchy import plan_llama_cpp_offload, MIB, GIB


def config_raw(tmp_path):
    raw = yaml.safe_load((Path(__file__).parents[2] / "fixtures/run.golden.yaml").read_text())
    raw["model"] = "mistralai/Mistral-7B-v0.1"
    raw["output"] = {"adapter_path": str(tmp_path / "new-run")}
    return raw


def test_dataset_slots_reject_overlap_and_multiple_sources(tmp_path):
    raw = config_raw(tmp_path)
    raw["train_dataset"] = str(tmp_path / "train.jsonl")
    raw["val_dataset"] = str(tmp_path / "train.jsonl")
    with pytest.raises(ConfigError, match="different files"):
        RunConfig.from_dict(raw)
    raw["val_dataset"] = str(tmp_path / "validation.parquet")
    raw["data"] = {"source": "multi", "datasets": []}
    with pytest.raises(ConfigError, match="one dataset"):
        RunConfig.from_dict(raw)


def test_helix_activation_budget_validation():
    assert validate_performance({'training_engine': 'helix', 'activation_memory_mb': 0})['activation_memory_mb'] == 0
    for value in (-1, 65537, True, 1.5):
        with pytest.raises(ValueError, match='activation_memory_mb'):
            validate_performance({'training_engine': 'helix', 'activation_memory_mb': value})
    with pytest.raises(ValueError, match='requires the Helix'):
        validate_performance({'activation_memory_mb': 512})
    assert validate_performance({'training_engine': 'helix', 'helix_prefetch': True})['helix_prefetch'] is True
    for options in ({'training_engine': 'helix', 'helix_prefetch': 'yes'}, {'helix_prefetch': True}):
        with pytest.raises(ValueError, match='helix_prefetch'):
            validate_performance(options)
    assert validate_performance({'training_engine': 'helix', 'helix_prefetch': True,
                                 'prefetch_pinned_memory_mb': 0})['prefetch_pinned_memory_mb'] == 0
    for value in (-1, 4097, True, 1.5):
        with pytest.raises(ValueError, match='prefetch_pinned_memory_mb'):
            validate_performance({'training_engine': 'helix', 'helix_prefetch': True, 'prefetch_pinned_memory_mb': value})
    with pytest.raises(ValueError, match='requires Helix prefetch'):
        validate_performance({'training_engine': 'helix', 'prefetch_pinned_memory_mb': 256})


def adapter(tmp_path, base="mistralai/Mistral-7B-v0.1"):
    source = tmp_path / "source" / "checkpoint-400"
    source.mkdir(parents=True)
    (source / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
        "r": 32, "lora_alpha": 64, "target_modules": ["q_proj"], "lora_dropout": 0,
    }))
    (source / "adapter_model.safetensors").write_bytes(b"source-weights")
    return source


def test_continuation_preserves_source_structure_and_fingerprint(tmp_path):
    source = adapter(tmp_path)
    raw = config_raw(tmp_path)
    raw["init_adapter"] = str(source)
    config = RunConfig.from_dict(raw)
    info = resolve_training_source(config)
    assert info["workflow"] == "continue_adapter"
    assert len(info["sha256"]) == 64
    assert config.adapter["rank"] == 32
    assert config.adapter["target_modules"] == ["q_proj"]
    assert not config.output.get("resume_from_checkpoint")
    before = info["sha256"]
    (source / "adapter_model.safetensors").write_bytes(b"updated-weights")
    assert resolve_training_source(config)["sha256"] != before


@pytest.mark.parametrize('saved', [True, False])
def test_continuation_inherits_rslora_from_adapter_metadata(tmp_path, saved):
    source = adapter(tmp_path)
    path = source / 'adapter_config.json'
    metadata = json.loads(path.read_text())
    metadata['use_rslora'] = saved
    path.write_text(json.dumps(metadata))
    raw = config_raw(tmp_path)
    raw['init_adapter'] = str(source)
    raw['adapter']['use_rslora'] = not saved
    config = RunConfig.from_dict(raw)
    resolve_training_source(config)
    assert config.adapter['use_rslora'] is saved


def test_local_adapter_as_model_and_base_mismatch(tmp_path):
    source = adapter(tmp_path)
    raw = config_raw(tmp_path)
    raw["model"] = str(source)
    assert resolve_training_source(RunConfig.from_dict(raw))["path"] == str(source.resolve())
    raw.update(model="other/base", init_adapter=str(source))
    with pytest.raises(ConfigError, match="does not match"):
        resolve_training_source(RunConfig.from_dict(raw))


def test_resume_never_silently_becomes_fresh_training(tmp_path):
    source = adapter(tmp_path)
    raw = config_raw(tmp_path)
    raw["output"]["resume_from"] = str(source)
    with pytest.raises(ConfigError, match="Full resume is missing"):
        resolve_training_source(RunConfig.from_dict(raw))
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (source / name).write_bytes(b"state")
    (source / "trainer_state.json").write_text('{"global_step": 400}')
    raw["train"]["max_steps"] = 500
    config = RunConfig.from_dict(raw)
    assert resolve_training_source(config)["workflow"] == "resume"
    assert config.output["resume_from_checkpoint"] == str(source.resolve())
    raw["train"]["max_steps"] = 300
    with pytest.raises(ConfigError, match="must exceed"):
        resolve_training_source(RunConfig.from_dict(raw))


def test_resume_preserves_saved_router_optimizer_groups(tmp_path):
    source = adapter(tmp_path)
    for name in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        (source / name).write_bytes(b'state')
    (source / 'trainer_state.json').write_text('{"global_step": 10}')
    (source.parent / 'sytra-provenance.json').write_text(json.dumps({'optim': {'router_lr_ratio': .1, 'router_aux_coef': .02}}))
    raw = config_raw(tmp_path)
    raw['output']['resume_from_checkpoint'] = str(source)
    raw['train']['max_steps'] = 500
    config = RunConfig.from_dict(raw)
    resolve_training_source(config)
    assert config.optim['router_lr_ratio'] == .1
    assert config.optim['router_aux_coef'] == .02
    raw['optim']['router_lr_ratio'] = .2
    with pytest.raises(ConfigError, match='saved router LR ratio'):
        resolve_training_source(RunConfig.from_dict(raw))
    raw['optim']['router_lr_ratio'] = .1
    raw['optim']['router_aux_coef'] = .03
    with pytest.raises(ConfigError, match='saved router auxiliary coefficient'):
        resolve_training_source(RunConfig.from_dict(raw))


def test_continuation_prevents_overwriting_source(tmp_path):
    source = adapter(tmp_path)
    raw = config_raw(tmp_path)
    raw["init_adapter"] = str(source)
    raw["output"]["adapter_path"] = str(source.parent)
    with pytest.raises(ConfigError, match="separate output"):
        resolve_training_source(RunConfig.from_dict(raw))


def test_aliases_preserved_and_conflicts_rejected(tmp_path):
    raw = config_raw(tmp_path)
    raw["optim"].pop("grad_accumulation_steps")
    raw["train"]["gradient_accumulation_steps"] = 32
    assert RunConfig.from_dict(raw).optim["grad_accumulation_steps"] == 32
    raw["optim"]["grad_accumulation_steps"] = 8
    with pytest.raises(ConfigError, match="Conflicting"):
        RunConfig.from_dict(raw)


@pytest.mark.parametrize("batch,accum", [(2, 8), (4, 16), (3, 7)])
def test_low_memory_preserves_effective_batch_and_context(batch, accum):
    hw = HardwareProfile(os_name="Windows", vram_total_gb=8, ram_total_gb=16, ram_available_gb=5)
    train = {"batch_size": batch, "max_seq_len": 8192, "max_steps": 300}
    plan = training_plan(hw, {}, train, {"grad_accumulation_steps": accum})
    assert plan.micro_batch_size * plan.gradient_accumulation_steps == batch * accum
    assert plan.dataloader_workers == 0 and not plan.pin_memory
    assert train == {"batch_size": batch, "max_seq_len": 8192, "max_steps": 300}


def workstation():
    return HardwareProfile(os_name="Linux", cpu_logical_cores=32, cpu_physical_cores=16,
        ram_total_gb=64, ram_available_gb=48, cuda_available=True, vram_total_gb=24,
        vram_free_gb=20, storage_kind="nvme")


def test_profiles_and_packing_require_boundaries():
    hw = workstation()
    fast = training_plan(hw, {"mode": "fast"}, {}, {}, lengths=[30, 100], packing_supported=True)
    assert fast.packing and fast.dataloader_workers == 4 and fast.pin_memory
    safe = training_plan(hw, {"mode": "safe"}, {}, {}, lengths=[30], packing_supported=True)
    assert not safe.packing and not safe.pin_memory and safe.dataloader_workers == 0
    auto = training_plan(hw, {}, {}, {}, lengths=[30], packing_supported=False)
    assert not auto.packing
    with pytest.raises(ValueError, match="boundary"):
        training_plan(hw, {"packing": True}, {}, {}, packing_supported=False)
    off = training_plan(hw, {"packing": False}, {}, {}, lengths=[30], packing_supported=True)
    assert not off.packing


def test_custom_worker_zero_disables_prefetch_and_persistence():
    plan = training_plan(workstation(), {"dataloader_workers": 0, "prefetch_factor": 4}, {}, {})
    assert plan.prefetch_factor is None and not plan.persistent_workers


def test_merge_cuda_free_memory_and_safe_fallback():
    auto = merge_plan(workstation(), {}, estimated_ram_gb=16, adapter_memory_gb=10)
    assert "--cuda" in auto.flags() and "--low-cpu-memory" not in auto.flags()
    assert auto.adapter_materialization_device == "cuda"
    assert auto.out_shard_size == "4B"
    safe = merge_plan(workstation(), {"mode": "safe"}, estimated_ram_gb=16, adapter_memory_gb=10)
    assert not safe.cuda and safe.low_cpu_memory and safe.out_shard_size == "1B"
    assert safe.adapter_materialization_device == "cpu"
    assert merge_plan(workstation(), {}, adapter_memory_gb=30).adapter_materialization_device == "cpu"
    with pytest.raises(ValueError, match="free VRAM"):
        merge_plan(workstation(), {"adapter_materialization_device": "cuda"}, adapter_memory_gb=30)
    assert not merge_plan(HardwareProfile(), {}).cuda


def test_detects_nested_moe_without_guessing_from_active_parameter_count():
    assert detect_moe({"text_config": {"model_type": "qwen3_moe", "num_experts": 128, "num_experts_per_tok": 8}})["is_moe"]
    assert not detect_moe({"num_experts": 1})["is_moe"]
    assert detect_moe({"n_routed_experts": 256})["num_experts"] == 256


def offload(**overrides):
    args = dict(weight_bytes=8 * GIB, vram_budget_mb=4096, ram_budget_mb=16384,
                n_layer=32, kv_bytes=512 * MIB, cpu_count=8, windows=True)
    return plan_llama_cpp_offload(**(args | overrides))


def test_kv_reserved_before_weights_and_forced_layers_rechecked():
    plan = offload()
    assert plan.peak_vram_mb <= 4096
    assert offload(kv_bytes=2 * GIB).gpu_layers < plan.gpu_layers
    assert offload(force_gpu_layers=32).strategy == "insufficient-memory"
    cpu = offload(force_gpu_layers=0, cpu_kv_cache=True)
    assert cpu.peak_ram_mb > plan.peak_ram_mb
    assert cpu.strategy == "cpu-only"
    assert offload(ram_budget_mb=100).strategy == "insufficient-memory"


def test_moe_experts_offload_by_actual_bytes_and_fit_in_ram():
    args = dict(layer_sizes=[GIB // 4] * 32, expert_sizes=[GIB * 3 // 16] * 32, cpu_moe=True)
    plan = offload(**args)
    assert plan.cpu_moe and plan.gpu_layers == 32
    assert plan.peak_vram_mb <= 4096 and plan.peak_ram_mb <= 16384
    assert plan.strategy == "moe-cpu-experts"
    assert offload(**args, ram_budget_mb=4096).strategy == "insufficient-memory"


def tiny_checkpoint(path):
    path.mkdir()
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"1234")
    (path / "config.json").write_text('{}')
    return path


def test_checkpoint_hash_changes_with_weights_and_dtype_size(tmp_path):
    base = tiny_checkpoint(tmp_path / "base")
    before = checkpoint_fingerprint(base)
    assert checkpoint_memory_gb(base, "float32") == 4 / GIB
    assert checkpoint_memory_gb(base, "float16") == 2 / GIB
    with (base / "model.safetensors").open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"5678")
    assert checkpoint_fingerprint(base) != before


def test_materialization_cache_invalidates_when_adapter_changes(tmp_path, monkeypatch):
    from sytra_runner import merge
    base = tiny_checkpoint(tmp_path / "base")
    source = adapter(tmp_path, str(base))
    calls = []

    def materialize(adapter_path, base_model, output_path, **kwargs):
        calls.append(kwargs)
        output = Path(output_path)
        (output / "config.json").write_text('{}')
        (output / "model.safetensors").write_bytes((base / "model.safetensors").read_bytes())

    monkeypatch.setattr(merge, "_merge_lora_adapter", materialize)
    config = tmp_path / "merge.yaml"
    config.write_text(yaml.safe_dump({"merge_method": "linear", "dtype": "float32", "models": [{"model": str(source)}]}))
    first = yaml.safe_load(Path(merge.write_mergekit_config(str(config), hardware=HardwareProfile())).read_text())
    second = yaml.safe_load(Path(merge.write_mergekit_config(str(config), hardware=HardwareProfile())).read_text())
    assert first == second and len(calls) == 1
    assert calls[0]["dtype"] == "float32"
    (source / "adapter_model.safetensors").write_bytes(b"changed")
    third = yaml.safe_load(Path(merge.write_mergekit_config(str(config), hardware=HardwareProfile())).read_text())
    assert third != first and len(calls) == 2


def test_real_gguf_tensor_layout_enables_cpu_expert_command(tmp_path, monkeypatch):
    import struct
    from gguf_test_file import _write_string, _write_value
    from sytra_runner.gguf_meta import read_gguf_layer_sizes
    from sytra_runner.model_planner import build_backend_plan

    metadata = {"general.architecture": "qwen3moe", "qwen3moe.block_count": 8,
                "qwen3moe.expert_count": 8, "qwen3moe.expert_used_count": 2}
    parts = [b"GGUF", struct.pack("<IQQ", 3, 16, len(metadata))]
    for name, value in metadata.items():
        _write_string(parts, name)
        encoded = []
        kind = _write_value(encoded, value)
        parts.extend([struct.pack("<I", kind), *encoded])
    offset = 0
    for layer in range(8):
        for name, size in (("attn_q.weight", 64 * 1024), ("ffn_up_exps.weight", 448 * 1024)):
            _write_string(parts, f"blk.{layer}.{name}")
            parts.append(struct.pack("<IQIQ", 1, size // 4, 0, offset))
            offset += size
    header = b"".join(parts)
    header += b"\0" * (-len(header) % 32)
    model = tmp_path / "moe.gguf"
    model.write_bytes(header + b"\0" * offset)
    layout = read_gguf_layer_sizes(model, 8)
    assert layout == ([512 * 1024] * 8, [448 * 1024] * 8)
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setattr("sytra_runner.model_planner._llama_server_help", lambda _: "--cpu-moe --cache-type-k -ctk --cache-type-v -ctv")
    plan = build_backend_plan(model, vram_limit_mb=2, ram_limit_mb=16)
    assert plan.compatible
    assert plan.llama_offload.cpu_moe
    assert "--cpu-moe" in plan.command
    assert plan.llama_offload.gpu_layers == 8
    assert plan.llama_offload.peak_vram_mb <= 2


@pytest.mark.parametrize("key", ["init_adapter", "resume_from", "resume_from_checkpoint"])
def test_empty_continuation_paths_fail_instead_of_starting_fresh(tmp_path, key):
    raw = config_raw(tmp_path)
    if key == "init_adapter":
        raw[key] = ""
    else:
        raw["output"][key] = ""
    with pytest.raises(ConfigError, match="non-empty"):
        RunConfig.from_dict(raw)
