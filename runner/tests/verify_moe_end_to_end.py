"""Build and load synthetic MoE models (Mixtral, Qwen MoE, DeepSeek MoE) through Sytra/MergeKit.

Run with the merge environment and HF_HUB_OFFLINE=1. This verifies MergeKit config
schema, shared-expert rules, router projection initialization, tokenizer output,
and weight/forward numerical sanity. It is not a benchmark or evidence of model quality.
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

RUNNER_ROOT = Path(__file__).resolve().parents[1]
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))

from sytra_runner.config import MergeConfig
from sytra_runner.merge import run_real_merge


def _build_tokenizer() -> tuple[dict[str, int], PreTrainedTokenizerFast]:
    vocab = {
        "[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
        "code": 3, "function": 4, "python": 5,
        "math": 6, "equation": 7, "algebra": 8,
        "general": 9, "explain": 10,
    }
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )
    return vocab, fast


def _save_family_models(
    root: Path,
    prefix: str,
    config_cls,
    model_cls,
    vocab: dict[str, int],
    tokenizer: PreTrainedTokenizerFast,
    *,
    names: tuple[str, ...],
    extra_config: dict | None = None,
) -> tuple[list[Path], object]:
    cfg_kwargs = dict(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    if extra_config:
        cfg_kwargs.update(extra_config)
    config = config_cls(**cfg_kwargs)
    paths: list[Path] = []
    for idx, name in enumerate(names):
        path = root / f"{prefix}-{name}"
        torch.manual_seed(100 + idx)
        model = model_cls(config)
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
        paths.append(path)
        del model
    return paths, config


def _load_safetensors_dir(directory: Path) -> dict[str, torch.Tensor]:
    index_path = directory / "model.safetensors.index.json"
    tensors: dict[str, torch.Tensor] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        for shard in sorted(set(weight_map.values())):
            with safe_open(str(directory / shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)
    else:
        with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    return tensors


def verify_mixtral(root: Path, vocab: dict[str, int], tokenizer: PreTrainedTokenizerFast) -> None:
    paths, config = _save_family_models(
        root, "mixtral", LlamaConfig, LlamaForCausalLM, vocab, tokenizer,
        names=("base", "expert-code", "expert-math"),
    )
    out_dir = root / "merged-mixtral"
    raw = {
        "version": 1,
        "merge_method": "moe",
        "base_model": str(paths[0]),
        "dtype": "float32",
        "models": [
            {
                "model": str(paths[1]),
                "positive_prompts": ["write python code function"],
                "negative_prompts": ["solve math equation algebra"],
            },
            {
                "model": str(paths[2]),
                "positive_prompts": ["solve math equation algebra"],
                "negative_prompts": ["write python code function"],
            },
        ],
        "moe": {"architecture": "mixtral", "experts_per_token": 2, "shared_experts": []},
        "compat": {"verdict": "green"},
        "tokenizer": {"source": "base"},
        "performance": {"mode": "safe"},
        "output": {"model_path": str(out_dir)},
    }
    config_path = root / "mixtral.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert run_real_merge(MergeConfig.from_dict(raw), str(config_path)) == 0

    merged = AutoModelForCausalLM.from_pretrained(out_dir, local_files_only=True)
    assert merged.config.model_type == "mixtral"
    assert (out_dir / "tokenizer.json").is_file()
    with torch.no_grad():
        result = merged(input_ids=torch.tensor([[1, 3, 4, 5]]), use_cache=False)
    assert result.logits.shape == (1, 4, len(vocab)) and torch.isfinite(result.logits).all()

    gates = [
        param for name, param in merged.named_parameters()
        if "gate" in name and param.shape == (2, config.hidden_size)
    ]
    assert len(gates) == config.num_hidden_layers
    for gate in gates:
        assert torch.isfinite(gate).all() and torch.count_nonzero(gate) > 0
        assert not torch.allclose(gate[0], gate[1])
    assert (out_dir / "sytra-provenance.json").is_file()
    del result, merged, gates
    gc.collect()
    print("[PASS] Synthetic Mixtral MoE build, gate differentiation, and AutoModel forward pass.")


def verify_qwen_moe(root: Path, vocab: dict[str, int], tokenizer: PreTrainedTokenizerFast) -> None:
    paths, config = _save_family_models(
        root, "qwen", Qwen2Config, Qwen2ForCausalLM, vocab, tokenizer,
        names=("base", "expert-code", "expert-math", "shared-general"),
    )
    out_dir = root / "merged-qwen-moe"
    raw = {
        "version": 1,
        "merge_method": "moe",
        "base_model": str(paths[0]),
        "dtype": "float32",
        "models": [
            {
                "model": str(paths[1]),
                "positive_prompts": ["code function python"],
                "negative_prompts": ["math equation"],
            },
            {
                "model": str(paths[2]),
                "positive_prompts": ["math equation algebra"],
                "negative_prompts": ["code function"],
            },
        ],
        "moe": {
            "architecture": "qwen",
            "experts_per_token": 2,
            "shared_experts": [
                {
                    "source_model": str(paths[3]),
                    "positive_prompts": ["general assistance"],
                    "negative_prompts": ["unrelated noise"],
                }
            ],
        },
        "compat": {"verdict": "green"},
        "tokenizer": {"source": "base"},
        "performance": {"mode": "safe"},
        "output": {"model_path": str(out_dir)},
    }
    config_path = root / "qwen-moe.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert run_real_merge(MergeConfig.from_dict(raw), str(config_path)) == 0

    merged = AutoModelForCausalLM.from_pretrained(out_dir, local_files_only=True)
    assert merged.config.model_type == "qwen2_moe"
    assert (out_dir / "tokenizer.json").is_file()
    with torch.no_grad():
        result = merged(input_ids=torch.tensor([[1, 3, 6, 7]]), use_cache=False)
    assert result.logits.shape == (1, 4, len(vocab)) and torch.isfinite(result.logits).all()

    param_names = {name for name, _ in merged.named_parameters()}
    assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in param_names
    assert "model.layers.0.mlp.shared_expert_gate.weight" in param_names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in param_names
    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in param_names

    shared_src = _load_safetensors_dir(paths[3])["model.layers.0.mlp.gate_proj.weight"]
    merged_shared = dict(merged.named_parameters())["model.layers.0.mlp.shared_expert.gate_proj.weight"].detach().cpu()
    assert torch.allclose(merged_shared, shared_src, atol=1e-5)

    for layer_idx in range(config.num_hidden_layers):
        gate = dict(merged.named_parameters())[f"model.layers.{layer_idx}.mlp.gate.weight"]
        assert torch.isfinite(gate).all() and torch.count_nonzero(gate) > 0
        assert not torch.allclose(gate[0], gate[1])
    assert (out_dir / "sytra-provenance.json").is_file()
    del result, merged
    gc.collect()
    print("[PASS] Synthetic Qwen2 MoE build (with shared_expert & shared_expert_gate) and AutoModel forward pass.")


def verify_deepseek_moe(
    root: Path, vocab: dict[str, int], tokenizer: PreTrainedTokenizerFast, *, with_shared: bool
) -> None:
    prefix = "ds-shared" if with_shared else "ds-routed"
    names = ("base", "expert-code", "expert-math", "shared-general") if with_shared else ("base", "expert-code", "expert-math")
    paths, config = _save_family_models(
        root, prefix, LlamaConfig, LlamaForCausalLM, vocab, tokenizer, names=names
    )
    out_dir = root / f"merged-{prefix}"
    shared_cfg = [{"source_model": str(paths[3])}] if with_shared else []
    raw = {
        "version": 1,
        "merge_method": "moe",
        "base_model": str(paths[0]),
        "dtype": "float32",
        "models": [
            {
                "model": str(paths[1]),
                "positive_prompts": ["code function python"],
                "negative_prompts": ["math algebra"],
            },
            {
                "model": str(paths[2]),
                "positive_prompts": ["math equation algebra"],
                "negative_prompts": ["code python"],
            },
        ],
        "moe": {
            "architecture": "deepseek",
            "experts_per_token": 2,
            "shared_experts": shared_cfg,
        },
        "compat": {"verdict": "green"},
        "tokenizer": {"source": "base"},
        "performance": {"mode": "safe"},
        "output": {"model_path": str(out_dir)},
    }
    config_path = root / f"{prefix}.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert run_real_merge(MergeConfig.from_dict(raw), str(config_path)) == 0

    out_cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert out_cfg["model_type"] == "deepseek"
    assert out_cfg["n_routed_experts"] == 2
    assert out_cfg["num_experts_per_tok"] == 2
    assert (out_cfg.get("n_shared_experts") or 0) == (1 if with_shared else 0)
    assert (out_dir / "tokenizer.json").is_file()
    assert (out_dir / "sytra-provenance.json").is_file()

    weights = _load_safetensors_dir(out_dir)
    for layer_idx in range(config.num_hidden_layers):
        gate = weights[f"model.layers.{layer_idx}.mlp.gate.weight"]
        assert gate.shape == (2, config.hidden_size)
        assert torch.isfinite(gate).all() and torch.count_nonzero(gate) > 0
        assert not torch.allclose(gate[0], gate[1])
        for exp_idx in range(2):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"model.layers.{layer_idx}.mlp.experts.{exp_idx}.{proj}.weight" in weights

        shared_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
        if with_shared:
            assert shared_key in weights
            if layer_idx == 0:
                src_shared = _load_safetensors_dir(paths[3])["model.layers.0.mlp.gate_proj.weight"]
                assert torch.allclose(weights[shared_key], src_shared, atol=1e-5)
        else:
            assert shared_key not in weights

    # Numerical forward verification over layer 0 MoE block (routed top-2 + optional shared expert)
    hidden = torch.randn(1, 3, config.hidden_size)
    gate0 = weights["model.layers.0.mlp.gate.weight"].float()
    probs = F.softmax(hidden @ gate0.T, dim=-1)
    moe_out = torch.zeros_like(hidden)
    for exp_idx in range(2):
        w_gate = weights[f"model.layers.0.mlp.experts.{exp_idx}.gate_proj.weight"].float()
        w_up = weights[f"model.layers.0.mlp.experts.{exp_idx}.up_proj.weight"].float()
        w_down = weights[f"model.layers.0.mlp.experts.{exp_idx}.down_proj.weight"].float()
        exp_h = (F.silu(hidden @ w_gate.T) * (hidden @ w_up.T)) @ w_down.T
        moe_out = moe_out + probs[..., exp_idx : exp_idx + 1] * exp_h
    if with_shared:
        sg = weights["model.layers.0.mlp.shared_experts.gate_proj.weight"].float()
        su = weights["model.layers.0.mlp.shared_experts.up_proj.weight"].float()
        sd = weights["model.layers.0.mlp.shared_experts.down_proj.weight"].float()
        moe_out = moe_out + (F.silu(hidden @ sg.T) * (hidden @ su.T)) @ sd.T
    assert torch.isfinite(moe_out).all() and torch.count_nonzero(moe_out) > 0
    gc.collect()
    print(f"[PASS] Synthetic DeepSeek MoE build (shared_experts={int(with_shared)}), gate differentiation, and layer forward check.")


def main() -> None:
    torch.set_num_threads(2)
    vocab, tokenizer = _build_tokenizer()
    with tempfile.TemporaryDirectory(prefix="sytra-moe-e2e-") as directory:
        root = Path(directory).resolve()
        verify_mixtral(root, vocab, tokenizer)
        verify_qwen_moe(root, vocab, tokenizer)
        verify_deepseek_moe(root, vocab, tokenizer, with_shared=True)
        verify_deepseek_moe(root, vocab, tokenizer, with_shared=False)


if __name__ == "__main__":
    main()
