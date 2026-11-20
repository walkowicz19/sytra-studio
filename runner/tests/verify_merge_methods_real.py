"""End-to-end verification of non-MoE merge algorithms in merge-env.

Exercises real MergeKit execution (slerp, linear, task_arithmetic, ties, dare_ties,
dare_linear), Sytra spherical merges (multi_slerp, nuslerp), numerical checks against
explicit analytical formulas, AutoModelForCausalLM forward inference, and atomic failure
preservation. Run with .sytra-envs/merge-env and HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

RUNNER_ROOT = Path(__file__).resolve().parents[1]
if str(RUNNER_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNNER_ROOT))

from sytra_runner.config import MergeConfig
from sytra_runner.merge import run_real_merge


def _analytical_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Analytical MergeKit SLERP formula (sin((1-t)w)/sin(w))*v0 + (sin(tw)/sin(w))*v1."""
    a = v0.to(torch.float64).reshape(-1)
    b = v1.to(torch.float64).reshape(-1)
    na, nb = a.norm(), b.norm()
    ua, ub = a / na, b / nb
    dot = torch.clamp((ua * ub).sum(), -1.0, 1.0)
    if abs(dot.item()) > 0.9995:
        out = (1.0 - t) * a + t * b
    else:
        omega = torch.acos(dot)
        sin_o = torch.sin(omega)
        out = (torch.sin((1.0 - t) * omega) / sin_o) * a + (torch.sin(t * omega) / sin_o) * b
    return out.reshape(v0.shape)


def _analytical_ties(
    base: torch.Tensor, models: list[torch.Tensor], weights: list[float], *, normalize: bool = True
) -> torch.Tensor:
    deltas = [m.to(torch.float64) - base.to(torch.float64) for m in models]
    weighted = [w * d for w, d in zip(weights, deltas)]
    sign_sum = sum(weighted)
    majority = torch.where(sign_sum >= 0, 1.0, -1.0)
    acc = torch.zeros_like(base, dtype=torch.float64)
    weight_sum = torch.zeros_like(base, dtype=torch.float64)
    for w, w_d in zip(weights, weighted):
        agree = w_d.sign() == majority
        acc = acc + torch.where(agree, w_d, torch.zeros_like(w_d))
        weight_sum = weight_sum + torch.where(agree, float(w), 0.0)
    if normalize:
        mixed_delta = torch.where(weight_sum != 0, acc / weight_sum.clamp(min=1e-12), torch.zeros_like(acc))
    else:
        mixed_delta = acc
    return base.to(torch.float64) + mixed_delta


def _load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_path = model_dir / weight_map[name]
    else:
        shard_path = model_dir / "model.safetensors"
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()


def _create_kin_checkpoints(root: Path) -> tuple[Path, list[Path], PreTrainedTokenizerFast]:
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4, "test": 5, "sytra": 6, "merge": 7}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]"
    )
    cfg = LlamaConfig(
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
    torch.manual_seed(10)
    base = LlamaForCausalLM(cfg)
    base_dir = root / "base"
    base.save_pretrained(base_dir, safe_serialization=True)
    fast.save_pretrained(base_dir)

    children: list[Path] = []
    for idx in range(3):
        torch.manual_seed(10)
        child = LlamaForCausalLM(cfg)
        torch.manual_seed(20 + idx)
        with torch.no_grad():
            for param in child.parameters():
                param.add_(torch.randn_like(param) * 0.002)
        cdir = root / f"child-{idx}"
        child.save_pretrained(cdir, safe_serialization=True)
        fast.save_pretrained(cdir)
        children.append(cdir)
        del child
    del base
    return base_dir, children, fast


def _run_and_assert_forward(raw: dict, root: Path, label: str) -> Path:
    raw = {**raw, "compat": raw.get("compat") or {"verdict": "green"}}
    out_dir = Path(raw["output"]["model_path"])
    cfg_path = root / f"{label}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert run_real_merge(MergeConfig.from_dict(raw), str(cfg_path)) == 0
    model = AutoModelForCausalLM.from_pretrained(out_dir, local_files_only=True)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 3, 4, 5]]), use_cache=False).logits
    assert logits.shape == (1, 4, 8) and torch.isfinite(logits).all()
    assert (out_dir / "sytra-provenance.json").is_file()
    del model, logits
    gc.collect()
    return out_dir


def main() -> None:
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="sytra-merge-methods-") as directory:
        root = Path(directory).resolve()
        base_dir, children, _ = _create_kin_checkpoints(root)
        probe = "model.layers.0.self_attn.q_proj.weight"
        wb = _load_tensor(base_dir, probe)
        w0 = _load_tensor(children[0], probe)
        w1 = _load_tensor(children[1], probe)
        w2 = _load_tensor(children[2], probe)

        # 1. SLERP (auto t from weights 0.25 / 0.75 -> t = 0.75)
        out = _run_and_assert_forward(
            {
                "version": 1, "merge_method": "slerp", "dtype": "float32",
                "models": [
                    {"model": str(children[0]), "parameters": {"weight": 0.25}},
                    {"model": str(children[1]), "parameters": {"weight": 0.75}},
                ],
                "output": {"model_path": str(root / "out-slerp")},
            },
            root, "slerp",
        )
        assert torch.allclose(_load_tensor(out, probe), _analytical_slerp(w0, w1, 0.75).float(), atol=5e-4, rtol=5e-4)
        print("[PASS] SLERP matches analytical hyperspherical interpolation + AutoModel forward.")

        # 2. Linear
        out = _run_and_assert_forward(
            {
                "version": 1, "merge_method": "linear", "dtype": "float32",
                "models": [
                    {"model": str(children[0]), "parameters": {"weight": 0.3}},
                    {"model": str(children[1]), "parameters": {"weight": 0.7}},
                ],
                "parameters": {"normalize": True},
                "output": {"model_path": str(root / "out-linear")},
            },
            root, "linear",
        )
        assert torch.allclose(_load_tensor(out, probe), 0.3 * w0 + 0.7 * w1, atol=5e-4, rtol=5e-4)
        print("[PASS] Linear matches weighted average reference + AutoModel forward.")

        # 3. Task arithmetic
        out = _run_and_assert_forward(
            {
                "version": 1, "merge_method": "task_arithmetic", "base_model": str(base_dir), "dtype": "float32",
                "models": [
                    {"model": str(children[0]), "parameters": {"weight": 0.4}},
                    {"model": str(children[1]), "parameters": {"weight": 0.6}},
                ],
                "parameters": {"normalize": False},
                "output": {"model_path": str(root / "out-ta")},
            },
            root, "ta",
        )
        assert torch.allclose(_load_tensor(out, probe), wb + 0.4 * (w0 - wb) + 0.6 * (w1 - wb), atol=5e-4, rtol=5e-4)
        print("[PASS] Task Arithmetic matches base + weighted task deltas + AutoModel forward.")

        # 4. TIES (density=1.0 -> exact sign election + disjoint mean check)
        out = _run_and_assert_forward(
            {
                "version": 1, "merge_method": "ties", "base_model": str(base_dir), "dtype": "float32",
                "models": [
                    {"model": str(children[0]), "parameters": {"weight": 0.5, "density": 1.0}},
                    {"model": str(children[1]), "parameters": {"weight": 0.5, "density": 1.0}},
                    {"model": str(children[2]), "parameters": {"weight": 0.5, "density": 1.0}},
                ],
                "parameters": {"normalize": True},
                "output": {"model_path": str(root / "out-ties")},
            },
            root, "ties",
        )
        assert torch.allclose(
            _load_tensor(out, probe),
            _analytical_ties(wb, [w0, w1, w2], [0.5, 0.5, 0.5], normalize=True).float(),
            atol=5e-4,
            rtol=5e-4,
        )
        print("[PASS] TIES matches analytical sign-consensus & disjoint-mean reference + AutoModel forward.")

        # 5. DARE-TIES & DARE-Linear (seed pinned)
        for mname in ("dare_ties", "dare_linear"):
            out = _run_and_assert_forward(
                {
                    "version": 1, "merge_method": mname, "base_model": str(base_dir), "dtype": "float32",
                    "models": [
                        {"model": str(children[0]), "parameters": {"weight": 0.5, "density": 0.8}},
                        {"model": str(children[1]), "parameters": {"weight": 0.5, "density": 0.8}},
                    ],
                    "parameters": {"normalize": False, "int8_mask": True},
                    "output": {"model_path": str(root / f"out-{mname}")},
                },
                root, mname,
            )
            merged_w = _load_tensor(out, probe)
            assert torch.isfinite(merged_w).all() and (merged_w - wb).abs().max() < 0.25
        print("[PASS] DARE-TIES and DARE-Linear succeeded + AutoModel forward.")

        # 6. Multi-SLERP & NuSLERP (3-model task vector mode + 2-model direct)
        for mname in ("multi_slerp", "nuslerp"):
            out = _run_and_assert_forward(
                {
                    "version": 1, "merge_method": mname, "base_model": str(base_dir), "dtype": "float32",
                    "models": [
                        {"model": str(children[0]), "parameters": {"weight": 0.2}},
                        {"model": str(children[1]), "parameters": {"weight": 0.5}},
                        {"model": str(children[2]), "parameters": {"weight": 0.3}},
                    ],
                    "output": {"model_path": str(root / f"out-{mname}")},
                },
                root, mname,
            )
            assert torch.isfinite(_load_tensor(out, probe)).all()
        print("[PASS] Multi-SLERP and NuSLERP succeeded + AutoModel forward.")

        # 7. Real atomic failure: invalid model preserves sentinel & leaves no staging dirs
        bad_out = root / "preserved-out"
        bad_out.mkdir()
        (bad_out / "sentinel.json").write_text('{"keep": true}', encoding="utf-8")
        missing_model = root / "does-not-exist"
        bad_raw = {
            "version": 1, "merge_method": "linear", "dtype": "float32",
            "compat": {"verdict": "green"},
            "models": [
                {"model": str(children[0]), "parameters": {"weight": 0.5}},
                {"model": str(missing_model), "parameters": {"weight": 0.5}},
            ],
            "output": {"model_path": str(bad_out)},
        }
        bad_cfg = root / "bad.yaml"
        bad_cfg.write_text(yaml.safe_dump(bad_raw), encoding="utf-8")
        failed = False
        try:
            run_real_merge(MergeConfig.from_dict(bad_raw), str(bad_cfg))
        except Exception:
            failed = True
        assert failed
        assert json.loads((bad_out / "sentinel.json").read_text(encoding="utf-8")) == {"keep": True}
        assert not [p for p in root.iterdir() if p.name.startswith(".sytra-merge-")]
        print("[PASS] Atomic failure preserves prior output and cleans staging directory.")


if __name__ == "__main__":
    main()
