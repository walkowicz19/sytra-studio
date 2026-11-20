"""Exercise the complete runner handoff without downloading a production model."""
import dataclasses
import json
import sys
from types import SimpleNamespace

import pytest

from sytra_runner.backends import unsloth_sft as backend
from sytra_runner.config import RunConfig
from sytra_runner.performance.hardware import HardwareProfile
from test_performance import adapter, config_raw


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_source_is_loaded_without_reinitialization_and_resume_is_forwarded(tmp_path, monkeypatch, resume, validate, sharded):
    source = adapter(tmp_path)
    raw = config_raw(tmp_path)
    raw["data"]["jsonl_path"] = str(tmp_path / "data.jsonl")
    (tmp_path / "data.jsonl").write_text('{"prompt":"hi","completion":"hello"}\n')
    if sharded:
        (tmp_path / "second.jsonl").write_text('{"prompt":"hi","completion":"hello"}\n')
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"version": 1, "rows": 2, "shards": [
            {"path": "data.jsonl", "rows": 1}, {"path": "second.jsonl", "rows": 1}]}))
        raw["train_dataset"] = str(manifest)
    if validate:
        raw["val_dataset"] = str(tmp_path / "val.jsonl")
        (tmp_path / "val.jsonl").write_text('{"prompt":"test","completion":"answer"}\n')
    raw["train"].update(max_steps=500, packing=False, neftune_noise_alpha=5.0)
    raw["adapter"]["quant_bits"] = 4
    if resume:
        raw["output"]["resume_from_checkpoint"] = str(source)
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (source / name).write_bytes(b"state")
        (source / "trainer_state.json").write_text('{"global_step": 400}')
    else:
        raw["init_adapter"] = str(source)
    calls = {}

    class Model:
        peft_config = {"default": object()}
        is_loaded_in_4bit = True
        config = SimpleNamespace(to_dict=lambda: {}, _attn_implementation="sdpa")

        def parameters(self):
            return [SimpleNamespace(requires_grad=True, numel=lambda: 16)]

        def save_pretrained(self, path):
            calls["saved"] = path

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            roles = {"system": 10, "user": 11, "assistant": 12}
            tokens = [1]
            for message in messages:
                tokens.extend([roles[message["role"]], 20 + len(message["content"]), 2])
            if kwargs.get("add_generation_prompt"):
                tokens.append(12)
            return tokens

        def save_pretrained(self, path):
            pass

    class Loader:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return Model(), Tokenizer()

        @staticmethod
        def for_training(model, **kwargs):
            calls["for_training"] = kwargs

        @staticmethod
        def get_peft_model(*args, **kwargs):
            pytest.fail("Existing adapter must never be reinitialized")

    class Dataset(list):
        @property
        def column_names(self):
            return list(self[0]) if self else []

        def map(self, function, **kwargs):
            assert kwargs["remove_columns"] == self.column_names
            return Dataset(function(row) for row in self)

        def select(self, indices):
            return Dataset(self[index] for index in indices)

    fields = ["completion_only_loss", "remove_unused_columns", "packing", "per_device_train_batch_size", "gradient_accumulation_steps", "max_steps", "eval_strategy", "eval_steps", "neftune_noise_alpha"]
    config_type = dataclasses.make_dataclass("SFTConfig", [(name, object, dataclasses.field(default=None)) for name in fields])

    class Trainer:
        def __init__(self, model, train_dataset, args, callbacks, processing_class, eval_dataset=None):
            calls["args"] = args
            calls["eval_dataset"] = eval_dataset
            self.train_dataset = train_dataset
            self.args = args
            self.data_collator = lambda _: {"labels": SimpleNamespace(reshape=lambda *_: SimpleNamespace(tolist=lambda: [-100, 2, -100]))}
            self.state = SimpleNamespace(global_step=500, log_history=[{"loss": .25}])

        def training_step(self, model, inputs):
            return 0

        def train(self, **kwargs):
            calls["train"] = kwargs

    fake_cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None,
                               reset_peak_memory_stats=lambda: None, is_bf16_supported=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda, version=SimpleNamespace(cuda="test")))
    monkeypatch.setitem(sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=Loader))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object, TrainingArguments=object))
    monkeypatch.setitem(sys.modules, "trl", SimpleNamespace(SFTConfig=config_type, SFTTrainer=Trainer))
    dataset = Dataset([{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}])
    def load_dataset(*args, **kwargs):
        calls.setdefault("dataset_files", []).append(kwargs["data_files"])
        return dataset
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(backend, "detect_hardware", lambda *args: HardwareProfile(vram_total_gb=8))
    monkeypatch.setattr(backend, "apply_cpu_threads", lambda *args: None)
    assert backend.run_real_training(RunConfig.from_dict(raw)) == 0
    assert calls["load"]["model_name"] == str(source.resolve())
    assert calls["train"]["resume_from_checkpoint"] == (str(source.resolve()) if resume else None)
    assert calls["args"].completion_only_loss is False
    assert calls["args"].remove_unused_columns is False
    assert calls["args"].neftune_noise_alpha == 5.0
    assert (calls["eval_dataset"] is not None) == validate
    assert calls["args"].eval_strategy == ("steps" if validate else None)
    assert isinstance(calls["dataset_files"][0], list) == sharded
    if sharded:
        assert len(calls["dataset_files"][0]) == 2
    provenance = json.loads((tmp_path / "new-run/sytra-provenance.json").read_text())
    assert provenance["adapter_source"]["r"] == 32
