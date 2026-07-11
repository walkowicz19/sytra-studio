from pathlib import Path

import pytest

from sytra_runner.config import ConfigError, MergeConfig, RunConfig

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_loads_golden_run_yaml():
    config = RunConfig.load(FIXTURES / "run.golden.yaml")
    assert config.version == 1
    assert config.train_mode == "sft"
    assert config.data.source == "local"
    assert config.data.params["format"] == "jsonl"
    assert config.data.params["mapping"] == {"prompt": "prompt", "completion": "completion"}
    assert config.train["max_steps"] == 4
    assert config.output["adapter_path"] == "fixtures/out/adapter"


def test_loads_golden_merge_yaml():
    config = MergeConfig.load(FIXTURES / "merge.golden.yaml")
    assert config.version == 1
    assert config.merge_method == "dare_ties"
    assert config.base_model == "mistralai/Mistral-7B-v0.1"
    assert len(config.models) == 2
    assert config.compat_verdict == "green"
    assert config.output["model_path"] == "fixtures/out/merged"


def test_rejects_unsupported_version():
    with pytest.raises(ConfigError):
        RunConfig.from_dict({"version": 99})


def test_rejects_red_compat_verdict():
    raw = {
        "version": 1,
        "op_id": None,
        "merge_method": "linear",
        "base_model": None,
        "dtype": "bfloat16",
        "models": [{"model": "a"}, {"model": "b"}],
        "tokenizer": {"source": "base"},
        "compat": {"verdict": "red", "fingerprint": None},
        "output": {"model_path": "out"},
    }
    with pytest.raises(ConfigError, match="red"):
        MergeConfig.from_dict(raw)


def test_rejects_slerp_with_more_than_two_models():
    raw = {
        "version": 1,
        "op_id": None,
        "merge_method": "slerp",
        "base_model": None,
        "dtype": "bfloat16",
        "models": [{"model": "a"}, {"model": "b"}, {"model": "c"}],
        "tokenizer": {"source": "base"},
        "compat": {"verdict": "green", "fingerprint": None},
        "output": {"model_path": "out"},
    }
    with pytest.raises(ConfigError, match="slerp"):
        MergeConfig.from_dict(raw)


def test_rejects_task_vector_method_without_base_model():
    raw = {
        "version": 1,
        "op_id": None,
        "merge_method": "ties",
        "base_model": None,
        "dtype": "bfloat16",
        "models": [{"model": "a"}, {"model": "b"}],
        "tokenizer": {"source": "base"},
        "compat": {"verdict": "green", "fingerprint": None},
        "output": {"model_path": "out"},
    }
    with pytest.raises(ConfigError, match="base_model"):
        MergeConfig.from_dict(raw)
