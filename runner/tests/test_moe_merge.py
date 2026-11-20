import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sytra_runner.config import ConfigError, MergeConfig
from sytra_runner import merge


def moe_raw(tmp_path):
    return {
        "version": 1, "op_id": None, "merge_method": "moe", "base_model": "org/base",
        "dtype": "float16", "models": [
            {"model": "org/code", "positive_prompts": ["write code"], "negative_prompts": ["solve geometry"]},
            {"model": "org/math", "positive_prompts": ["solve geometry"]},
        ],
        "moe": {"architecture": "mixtral", "experts_per_token": 2, "shared_experts": []},
        "tokenizer": {"source": "base"}, "compat": {"verdict": "amber"},
        "output": {"model_path": str(tmp_path / "merged")},
    }


def test_moe_config_uses_mergekit_hidden_router_schema(tmp_path):
    config = MergeConfig.from_dict(moe_raw(tmp_path))
    path = merge.write_moe_config(config, str(tmp_path / "run.yaml"))
    generated = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert generated["gate_mode"] == "hidden"
    assert generated["base_model"] == "org/base"
    assert generated["architecture"] == "mixtral"
    assert generated["experts"] == [
        {"source_model": "org/code", "positive_prompts": ["write code"], "negative_prompts": ["solve geometry"]},
        {"source_model": "org/math", "positive_prompts": ["solve geometry"], "negative_prompts": []},
    ]
    assert "shared_experts" not in generated


def test_qwen_shared_expert_prompts_reach_mergekit(tmp_path):
    raw = moe_raw(tmp_path)
    raw['moe'] = {'architecture': 'qwen', 'experts_per_token': 2,
                  'shared_experts': [{'source_model': 'org/shared', 'positive_prompts': ['general questions'],
                                      'negative_prompts': ['avoid unrelated topics']}]}
    config = MergeConfig.from_dict(raw)
    generated = yaml.safe_load(Path(merge.write_moe_config(config, str(tmp_path / 'run.yaml'))).read_text(encoding='utf-8'))
    assert generated['shared_experts'] == raw['moe']['shared_experts']


@pytest.mark.parametrize('architecture,shared', [
    ('mixtral', [{'source_model': 'org/shared'}]),
    ('deepseek', [{'source_model': 'org/a'}, {'source_model': 'org/b'}]),
    ('deepseek', [{'source_model': 'org/a', 'positive_prompts': ['routed']}]),
    ('qwen', []),
    ('qwen', [{'source_model': 'org/a'}]),
])
def test_moe_rejects_shared_experts_unsupported_by_architecture(tmp_path, architecture, shared):
    raw = moe_raw(tmp_path)
    raw['moe'].update(architecture=architecture, shared_experts=shared)
    with pytest.raises(ConfigError):
        MergeConfig.from_dict(raw)


def test_moe_rejects_architecture_missing_from_installed_mergekit(tmp_path, monkeypatch):
    config = MergeConfig.from_dict(moe_raw(tmp_path))
    monkeypatch.setitem(sys.modules, "mergekit.moe", SimpleNamespace(
        ALL_OUTPUT_ARCHITECTURES=[SimpleNamespace(name=lambda: "DeepSeek MoE")]))
    with pytest.raises(ValueError, match="does not provide"):
        merge.validate_moe_architecture_available(config)


def test_moe_runtime_rejects_incompatible_transformers_and_accelerate(monkeypatch):
    import types
    transformers = types.ModuleType('transformers')
    utils = types.ModuleType('transformers.utils')
    transformers.__version__ = '5.12.1'
    utils.is_accelerate_available = lambda: True
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'transformers.utils', utils)
    with pytest.raises(ValueError, match='repair the merge environment'):
        merge.validate_moe_runtime()
    transformers.__version__ = '4.56.2'
    utils.is_accelerate_available = lambda: False
    with pytest.raises(ValueError, match='repair the merge environment'):
        merge.validate_moe_runtime()
    utils.is_accelerate_available = lambda: True
    merge.validate_moe_runtime()


@pytest.mark.parametrize("mutation", ["missing_base", "missing_prompt", "same_prompts", "bad_topk", "one_expert"])
def test_moe_rejects_invalid_configuration(tmp_path, mutation):
    raw = moe_raw(tmp_path)
    if mutation == "missing_base": raw["base_model"] = None
    if mutation == "missing_prompt": raw["models"][0]["positive_prompts"] = []
    if mutation == "same_prompts": raw["models"][1].update(positive_prompts=["write code"], negative_prompts=["solve geometry"])
    if mutation == "bad_topk": raw["moe"]["experts_per_token"] = 3
    if mutation == "one_expert": raw["models"] = raw["models"][:1]
    with pytest.raises(ConfigError):
        MergeConfig.from_dict(raw)


def test_moe_dispatch_uses_moe_cli_and_low_memory_flags(tmp_path, monkeypatch):
    raw = moe_raw(tmp_path)
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    Path(raw["output"]["model_path"]).mkdir()
    config = MergeConfig.from_dict(raw)
    hardware = SimpleNamespace(to_dict=lambda: {})
    plan = SimpleNamespace(cuda=False, cpu_threads=1, to_dict=lambda: {})
    monkeypatch.setattr(merge, "detect_hardware", lambda: hardware)
    monkeypatch.setattr(merge, "validate_moe_architecture_available", lambda _: None)
    monkeypatch.setattr(merge, "validate_moe_runtime", lambda: None)
    monkeypatch.setattr(merge, "merge_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(merge, "apply_cpu_threads", lambda *args: {})
    monkeypatch.setattr(merge, "_local_snapshot", lambda _: None)
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test")
    seen = {}
    def popen(cmd, **kwargs):
        seen["cmd"] = cmd
        out_dir = Path(cmd[4])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text('{"model_type": "mixtral"}', encoding="utf-8")
        (out_dir / "model.safetensors").write_bytes(b"stub-weights")
        return SimpleNamespace(stdout=[], wait=lambda: 0)
    monkeypatch.setattr(merge.subprocess, "Popen", popen)
    assert merge.run_real_merge(config, str(config_path)) == 0
    assert seen["cmd"][1:3] == ["-m", "sytra_runner.moe_exec"]
    assert "--low-cpu-memory" in seen["cmd"] and "--lazy-unpickle" in seen["cmd"]
    assert Path(seen["cmd"][3]).name.endswith(".mergekit-moe.yaml")
    assert json.loads((tmp_path / "merged" / "sytra-provenance.json").read_text())["versions"]["mergekit"] == "test"


def test_moe_exec_patches_shared_expert_loaders(monkeypatch):
    import types
    from sytra_runner import moe_exec

    calls = []

    class FakeModelRef:
        def __init__(self, name):
            self.name = name

        def lazy_loader(self, **kwargs):
            calls.append(self.name)
            return f"loader:{self.name}"

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return isinstance(other, FakeModelRef) and self.name == other.name

    base_ref = FakeModelRef("base")
    exp_ref = FakeModelRef("expert1")
    shared_ref = FakeModelRef("shared-distinct")

    def fake_orig_init(config, out_path, merge_options):
        return {base_ref: "loader:base", exp_ref: "loader:expert1"}, "loader:base", "writer"

    common_mod = types.ModuleType("mergekit.moe.common")
    common_mod.initialize_io = fake_orig_init
    qwen_mod = types.ModuleType("mergekit.moe.qwen")
    qwen_mod.initialize_io = fake_orig_init
    ds_mod = types.ModuleType("mergekit.moe.deepseek")
    ds_mod.initialize_io = fake_orig_init
    mix_mod = types.ModuleType("mergekit.moe.mixtral")
    mix_mod.initialize_io = fake_orig_init

    monkeypatch.setitem(sys.modules, "mergekit", types.ModuleType("mergekit"))
    monkeypatch.setitem(sys.modules, "mergekit.moe", types.ModuleType("mergekit.moe"))
    monkeypatch.setitem(sys.modules, "mergekit.moe.common", common_mod)
    monkeypatch.setitem(sys.modules, "mergekit.moe.qwen", qwen_mod)
    monkeypatch.setitem(sys.modules, "mergekit.moe.deepseek", ds_mod)
    monkeypatch.setitem(sys.modules, "mergekit.moe.mixtral", mix_mod)

    moe_exec.patch_mergekit_shared_loaders()

    cfg = SimpleNamespace(
        shared_experts=[SimpleNamespace(source_model=shared_ref), SimpleNamespace(source_model=base_ref)]
    )
    opts = SimpleNamespace(transformers_cache=None, lazy_unpickle=True)
    loaders, _, _ = qwen_mod.initialize_io(cfg, "/tmp/out", opts)
    assert shared_ref in loaders and loaders[shared_ref] == "loader:shared-distinct"
    assert calls == ["shared-distinct"]
    loaders_ds, _, _ = ds_mod.initialize_io(cfg, "/tmp/out", opts)
    assert shared_ref in loaders_ds
