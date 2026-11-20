"""Numerical reference and atomic staging tests for merge algorithms and MergeKit dispatch."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from sytra_runner import merge
from sytra_runner.config import MergeConfig
from sytra_runner.spherical_merge import multi_slerp_tensor, nuslerp_tensor, run_spherical_merge


def _analytical_slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    a = v0.to(torch.float64).reshape(-1)
    b = v1.to(torch.float64).reshape(-1)
    na = a.norm()
    nb = b.norm()
    ua = a / na
    ub = b / nb
    dot = torch.clamp((ua * ub).sum(), -1.0, 1.0)
    if abs(dot.item()) > 0.9995:
        out = (1.0 - t) * a + t * b
    else:
        omega = torch.acos(dot)
        sin_o = torch.sin(omega)
        dir_vec = (torch.sin((1.0 - t) * omega) / sin_o) * ua + (torch.sin(t * omega) / sin_o) * ub
        out = dir_vec * ((1.0 - t) * na + t * nb)
    return out.reshape(v0.shape)


def _save_tiny_checkpoint(directory: Path, tensors: dict[str, torch.Tensor], *, vocab_size: int = 16) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
    cfg = {
        "model_type": "llama",
        "hidden_size": 16,
        "vocab_size": vocab_size,
        "num_hidden_layers": 1,
        "torch_dtype": "float32",
    }
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")
    (directory / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    return directory


def test_multi_slerp_matches_analytical_slerp_for_two_models():
    torch.manual_seed(7)
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    for t in (0.0, 0.25, 0.5, 0.85, 1.0):
        actual = multi_slerp_tensor([a, b], [1.0 - t, t])
        expected = _analytical_slerp(a, b, t)
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), f"mismatch at t={t}"


def test_multi_slerp_three_model_karcher_mean_and_task_vector():
    torch.manual_seed(12)
    base = torch.randn(16, 16)
    deltas = [torch.randn(16, 16) * 0.2 for _ in range(3)]
    models = [base + d for d in deltas]
    weights = [0.2, 0.5, 0.3]

    merged = multi_slerp_tensor(models, weights, base=base)
    merged_delta = merged - base.to(torch.float64)
    expected_norm = sum(w * d.to(torch.float64).norm() for w, d in zip(weights, deltas))
    assert math.isclose(float(merged_delta.norm()), float(expected_norm), rel_tol=1e-4)

    # Karcher mean first-order optimality: weighted sum of log-map tangent vectors vanishes
    mu = merged_delta / merged_delta.norm()
    tangent_sum = torch.zeros_like(mu)
    for w, d in zip(weights, deltas):
        u = d.to(torch.float64) / d.to(torch.float64).norm()
        dot = torch.clamp((mu * u).sum(), -1.0, 1.0)
        theta = torch.acos(dot)
        tangent = u - dot * mu
        if theta > 1e-4:
            tangent = tangent * (theta / torch.sin(theta))
        tangent_sum.add_(tangent, alpha=w)
    assert float(tangent_sum.norm()) < 1e-5


def test_nuslerp_rowwise_and_global_match_analytical_references():
    torch.manual_seed(99)
    a = torch.randn(12, 24)
    b = torch.randn(12, 24)
    t = 0.65

    row_actual = nuslerp_tensor([a, b], [1.0 - t, t], row_wise=True)
    for row in range(a.shape[0]):
        expected_row = _analytical_slerp(a[row], b[row], t)
        assert torch.allclose(row_actual[row], expected_row, atol=1e-5, rtol=1e-5)

    global_actual = nuslerp_tensor([a, b], [1.0 - t, t], row_wise=False)
    expected_global = _analytical_slerp(a, b, t)
    assert torch.allclose(global_actual, expected_global, atol=1e-5, rtol=1e-5)

    # Multi-model row-wise task vector convergence (zero nan, finite norms)
    c = torch.randn(12, 24)
    base = torch.randn(12, 24)
    tri = nuslerp_tensor([a, b, c], [0.25, 0.5, 0.25], base=base, row_wise=True)
    assert torch.isfinite(tri).all()
    assert tri.shape == a.shape


def test_slerp_config_derivation_from_model_weights(tmp_path):
    raw = {
        "version": 1,
        "merge_method": "slerp",
        "dtype": "float32",
        "models": [
            {"model": "org/model-a", "parameters": {"weight": 0.25}},
            {"model": "org/model-b", "parameters": {"weight": 0.75}},
        ],
        "output": {"model_path": str(tmp_path / "out")},
    }
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = yaml.safe_load(Path(merge.write_mergekit_config(str(cfg_path))).read_text(encoding="utf-8"))
    assert out["base_model"] == "org/model-a"
    assert math.isclose(out["parameters"]["t"], 0.75, rel_tol=1e-6)


@pytest.mark.parametrize("method", ["multi_slerp", "nuslerp"])
def test_spherical_merge_real_safetensors_sharded_and_vocab_aligned(tmp_path, method):
    torch.manual_seed(5)
    base_dir = _save_tiny_checkpoint(
        tmp_path / "base",
        {
            "model.embed_tokens.weight": torch.randn(16, 16),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
            "model.norm.weight": torch.ones(16),
            "lm_head.weight": torch.randn(16, 16),
        },
        vocab_size=16,
    )
    m1_dir = _save_tiny_checkpoint(
        tmp_path / "m1",
        {
            "model.embed_tokens.weight": torch.randn(16, 16) * 0.05 + torch.ones(16, 16),
            "model.layers.0.self_attn.q_proj.weight": _read_t(base_dir, "model.layers.0.self_attn.q_proj.weight") + torch.randn(16, 16) * 0.05,
            "model.norm.weight": torch.ones(16) * 1.02,
            "lm_head.weight": torch.randn(16, 16),
        },
        vocab_size=16,
    )
    # m2 has 20 vocab rows to test vocab alignment against 16-row base
    m2_dir = _save_tiny_checkpoint(
        tmp_path / "m2",
        {
            "model.embed_tokens.weight": torch.randn(20, 16),
            "model.layers.0.self_attn.q_proj.weight": _read_t(base_dir, "model.layers.0.self_attn.q_proj.weight") + torch.randn(16, 16) * 0.05,
            "model.norm.weight": torch.ones(16) * 0.98,
            "lm_head.weight": torch.randn(20, 16),
        },
        vocab_size=20,
    )

    out_dir = tmp_path / f"merged_{method}"
    raw = {
        "version": 1,
        "merge_method": method,
        "base_model": str(base_dir),
        "dtype": "float32",
        "models": [
            {"model": str(m1_dir), "parameters": {"weight": 0.3}},
            {"model": str(m2_dir), "parameters": {"weight": 0.7}},
        ],
        "compat": {"verdict": "green"},
        "tokenizer": {"source": "base"},
        "output": {"model_path": str(out_dir)},
    }
    cfg_path = tmp_path / f"{method}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = MergeConfig.from_dict(raw)

    # Exercise sharding on small tensors too via direct summary & full run_real_merge
    assert merge.run_real_merge(cfg, str(cfg_path)) == 0
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "tokenizer.json").is_file()
    assert json.loads((out_dir / "sytra-provenance.json").read_text(encoding="utf-8"))["versions"]["merge_backend"] == "sytra_spherical"

    q_out = _read_t(out_dir, "model.layers.0.self_attn.q_proj.weight")
    assert q_out.shape == (16, 16) and torch.isfinite(q_out).all()
    assert _read_t(out_dir, "model.embed_tokens.weight").shape == (16, 16)

    # Also test multi-shard output via tiny max_shard_bytes
    sharded_dir = tmp_path / f"sharded_{method}"
    summary = run_spherical_merge(
        method=method,
        model_dirs=[m1_dir, m2_dir],
        model_entries=raw["models"],
        base_dir=base_dir,
        tokenizer_dir=base_dir,
        dtype="float32",
        stage_dir=sharded_dir,
        max_shard_bytes=1200,
    )
    assert summary["shards"] >= 2
    assert (sharded_dir / "model.safetensors.index.json").is_file()
    index = json.loads((sharded_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    for rel in set(index["weight_map"].values()):
        assert (sharded_dir / rel).is_file()


def _read_t(directory: Path, name: str) -> torch.Tensor:
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def test_merge_failure_preserves_existing_output_and_cleans_stage_dir(tmp_path, monkeypatch):
    out_dir = tmp_path / "merged_existing"
    out_dir.mkdir()
    sentinel = out_dir / "sentinel.json"
    sentinel.write_text('{"preserved": true}', encoding="utf-8")

    raw = {
        "version": 1,
        "merge_method": "linear",
        "dtype": "float32",
        "compat": {"verdict": "green"},
        "models": [
            {"model": str(tmp_path / "m1"), "parameters": {"weight": 0.5}},
            {"model": str(tmp_path / "m2"), "parameters": {"weight": 0.5}},
        ],
        "output": {"model_path": str(out_dir)},
    }
    cfg_path = tmp_path / "merge.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = MergeConfig.from_dict(raw)

    monkeypatch.setattr(merge, "HAS_MERGEKIT", True)
    monkeypatch.setattr(merge, "detect_hardware", lambda: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(merge, "merge_plan", lambda *a, **k: SimpleNamespace(cuda=False, cpu_threads=1, to_dict=lambda: {}))
    monkeypatch.setattr(merge, "apply_cpu_threads", lambda *a: {})
    monkeypatch.setattr(merge, "_local_snapshot", lambda _: None)

    def failing_popen(cmd, **kwargs):
        # Write a partial invalid file in the stage dir and fail non-zero
        stage = Path(cmd[4])
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "partial.bin").write_bytes(b"corrupt")
        return SimpleNamespace(stdout=["50% merging\n"], wait=lambda: 2)

    monkeypatch.setattr(merge.subprocess, "Popen", failing_popen)

    with pytest.raises(RuntimeError, match="exited with code 2"):
        merge.run_real_merge(cfg, str(cfg_path))

    # Original output intact, partial file never promoted, staging dirs cleaned up
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"preserved": True}
    assert not (out_dir / "partial.bin").exists()
    leaked = [p for p in tmp_path.iterdir() if p.name.startswith(".sytra-merge-")]
    assert leaked == []
