"""Numerical checks run in the ML environment; lightweight CI skips them."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sytra_runner.performance.metrics import TokenMeter, instrument_trainer


def test_token_meter_counts_consumed_nonpadding_and_completion_tokens(monkeypatch):
    meter = TokenMeter()
    batch = {"input_ids": torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
             "labels": torch.tensor([[-100, 2, 3, -100], [-100, 4, -100, -100]])}
    meter.add(batch)
    meter.add(batch)
    monkeypatch.setattr("sytra_runner.performance.metrics.time.perf_counter", lambda: meter.started + 2)
    result = meter.snapshot(1)
    assert result["input_tokens"] == 10
    assert result["completion_tokens"] == 6
    assert result["tokens_s"] == 5
    assert result["samples_per_second"] == 2
    assert result["padding_ratio"] == .375


def test_padding_free_counts_sample_boundaries():
    meter = TokenMeter()
    meter.add({"input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
               "position_ids": torch.tensor([[0, 1, 2, 0, 1]]),
               "labels": torch.tensor([[-100, 2, 3, -100, 5]])})
    assert int(meter.samples) == 2
    assert int(meter.tokens) == 5


def test_failed_training_step_is_not_counted():
    class Trainer:
        def training_step(self, model, inputs):
            raise RuntimeError("OOM")
    trainer, meter = Trainer(), TokenMeter()
    instrument_trainer(trainer, meter)
    with pytest.raises(RuntimeError, match="OOM"):
        trainer.training_step(None, {"input_ids": torch.tensor([[1]])})
    assert meter.tokens == 0


def test_materialized_lora_matches_peft_numerically(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from sytra_runner.merge import _merge_lora_adapter
    from sytra_runner.performance.hardware import HardwareProfile
    from sytra_runner.performance.planner import merge_plan

    torch.manual_seed(7)
    base = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
    ))
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    adapted = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.fill_(.15)
    source = tmp_path / "adapter"
    adapted.save_pretrained(source)
    expected = {name: tensor.detach().clone() for name, tensor in adapted.merge_and_unload(safe_merge=True).state_dict().items()}

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text('{}')
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    output = tmp_path / "merged"
    _merge_lora_adapter(str(source), str(base_path), str(output),
        plan=merge_plan(HardwareProfile(), {"mode": "safe"}), dtype="float32")
    actual = transformers.LlamaForCausalLM.from_pretrained(output).state_dict()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
