"""Low-RAM streaming safetensors spherical merges (Multi-SLERP and NuSLERP)."""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import telemetry

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_shard_bytes(shard_size: str, dtype: str) -> int:
    """Translate a MergeKit shard parameter cap (e.g. '1B') into byte budget."""
    text = (shard_size or "1B").strip().upper()
    elem_bytes = 4 if dtype == "float32" else 2
    if text.endswith("B") and text[:-1].replace(".", "", 1).isdigit():
        return max(16 * 1024 * 1024, int(float(text[:-1]) * 1_000_000_000 * elem_bytes))
    return 2_000_000_000


class _CheckpointReader:
    """Lazy mmap reader over a local Hugging Face safetensors directory."""

    def __init__(self, directory: str | Path, stack: ExitStack):
        self.directory = Path(directory)
        self._stack = stack
        self._handles: dict[Path, Any] = {}
        self.tensor_to_file: dict[str, Path] = {}

        index_path = self.directory / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            for key, rel_file in weight_map.items():
                shard = (self.directory / rel_file).resolve()
                if not shard.is_relative_to(self.directory.resolve()) or not shard.is_file():
                    raise ValueError(f"Invalid shard path for tensor {key}: {rel_file}")
                self.tensor_to_file[key] = shard
        else:
            single = self.directory / "model.safetensors"
            if not single.is_file():
                raise ValueError(f"Checkpoint '{self.directory}' has no safetensors weights")
            handle = self._open_handle(single)
            for key in handle.keys():
                self.tensor_to_file[key] = single

    def _open_handle(self, path: Path):
        if path not in self._handles:
            self._handles[path] = self._stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        return self._handles[path]

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._open_handle(self.tensor_to_file[name]).get_tensor(name)


def _normalize_weights(models: list[dict[str, Any]]) -> list[float]:
    raw: list[float] = []
    for entry in models:
        params = entry.get("parameters") or {} if isinstance(entry, dict) else {}
        weight = params.get("weight", 1.0)
        if weight is None:
            weight = 1.0
        w = float(weight)
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"Invalid merge weight {weight!r}; weights must be finite and non-negative")
        raw.append(w)
    total = sum(raw)
    if total <= 0:
        raise ValueError("Sum of model weights must be positive")
    return [w / total for w in raw]


def _align_vocab_rows(name: str, tensors: list[torch.Tensor], base: torch.Tensor | None) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    shapes = {tuple(t.shape) for t in tensors}
    if base is not None:
        shapes.add(tuple(base.shape))
    if len(shapes) <= 1:
        return tensors, base
    is_vocab_matrix = (
        all(len(s) == 2 for s in shapes)
        and len({s[1] for s in shapes}) == 1
        and ("embed" in name.lower() or "lm_head" in name.lower())
    )
    if not is_vocab_matrix:
        raise ValueError(f"Incompatible tensor shapes for '{name}': {sorted(shapes)}")
    rows = min(s[0] for s in shapes)
    return [t[:rows] for t in tensors], (base[:rows] if base is not None else None)


def _karcher_mean_unit(units: list[torch.Tensor], weights: list[float], *, max_iter: int = 24, tol: float = 1e-6) -> torch.Tensor:
    """Weighted Riemannian Fréchet (Karcher) mean on the unit hypersphere S^{d-1}."""
    mu = torch.zeros_like(units[0])
    for w, u in zip(weights, units):
        mu.add_(u, alpha=w)
    mu_norm = mu.norm()
    if mu_norm < 1e-7:
        mu = units[0].clone()
    else:
        mu = mu / mu_norm

    for _ in range(max_iter):
        mean_tan = torch.zeros_like(mu)
        for w, u in zip(weights, units):
            if w <= 0:
                continue
            dot = torch.sum(mu * u).clamp(-1.0, 1.0)
            theta = torch.acos(dot)
            tan = u - dot * mu
            if theta > 1e-4:
                sin_theta = torch.sin(theta).clamp(min=1e-8)
                tan = tan * (theta / sin_theta)
            mean_tan.add_(tan, alpha=w)
        phi = mean_tan.norm()
        if phi < tol:
            break
        mu = torch.cos(phi) * mu + torch.sin(phi) * (mean_tan / phi)
        mu = mu / mu.norm().clamp(min=1e-12)
    return mu


def multi_slerp_tensor(tensors: list[torch.Tensor], weights: list[float], base: torch.Tensor | None = None) -> torch.Tensor:
    """Barycentric Multi-SLERP via weighted Karcher mean on the hypersphere."""
    work_dtype = torch.float64 if tensors[0].numel() <= 65536 else torch.float32
    vecs = [(t.to(work_dtype) - base.to(work_dtype)) if base is not None else t.to(work_dtype) for t in tensors]
    norms = [v.norm() for v in vecs]
    target_norm = sum(w * r for w, r in zip(weights, norms))

    active = [(w, v, r) for w, v, r in zip(weights, vecs, norms) if r > 1e-8 and w > 0]
    if not active:
        out = torch.zeros_like(vecs[0])
        return (base.to(work_dtype) + out) if base is not None else out
    if len(active) == 1:
        merged = (active[0][1] / active[0][2]) * target_norm
        return (base.to(work_dtype) + merged) if base is not None else merged

    aw_sum = sum(w for w, _, _ in active)
    aw = [w / aw_sum for w, _, _ in active]
    units = [v / r for _, v, r in active]
    mu = _karcher_mean_unit(units, aw)
    merged = mu * target_norm
    return (base.to(work_dtype) + merged) if base is not None else merged


def _nuslerp_pair_rows(u0: torch.Tensor, u1: torch.Tensor, t: float) -> torch.Tensor:
    """Vectorized row-wise Normalized Uniform SLERP between two 2D+ matrices."""
    shape = u0.shape
    a = u0.reshape(shape[0], -1)
    b = u1.reshape(shape[0], -1)
    na = a.norm(dim=1, keepdim=True)
    nb = b.norm(dim=1, keepdim=True)
    ua = a / na.clamp(min=1e-12)
    ub = b / nb.clamp(min=1e-12)

    dot = (ua * ub).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    target_norm = (1.0 - t) * na + t * nb

    colinear = (dot.abs() > 0.9995) | (na <= 1e-8) | (nb <= 1e-8) | (sin_theta <= 1e-6)
    safe_sin = torch.where(colinear, torch.ones_like(sin_theta), sin_theta)
    s0 = torch.sin((1.0 - t) * theta) / safe_sin
    s1 = torch.sin(t * theta) / safe_sin

    slerp_dir = s0 * ua + s1 * ub
    lerp_vec = (1.0 - t) * a + t * b
    slerp_vec = slerp_dir * target_norm
    out = torch.where(colinear, lerp_vec, slerp_vec)
    return out.reshape(shape)


def _nuslerp_multi_rows(vecs: list[torch.Tensor], weights: list[float], *, max_iter: int = 16) -> torch.Tensor:
    """Row-wise normalized barycentric spherical interpolation for 3+ models."""
    shape = vecs[0].shape
    mats = [v.reshape(shape[0], -1) for v in vecs]
    norms = [m.norm(dim=1, keepdim=True) for m in mats]
    target_norm = torch.zeros_like(norms[0])
    for w, n in zip(weights, norms):
        target_norm.add_(n, alpha=w)
    units = [m / n.clamp(min=1e-12) for m, n in zip(mats, norms)]

    mu = torch.zeros_like(units[0])
    for w, u in zip(weights, units):
        mu.add_(u, alpha=w)
    mu_n = mu.norm(dim=1, keepdim=True)
    mu = torch.where(mu_n > 1e-7, mu / mu_n.clamp(min=1e-12), units[0])

    for _ in range(max_iter):
        mean_tan = torch.zeros_like(mu)
        for w, u in zip(weights, units):
            if w <= 0:
                continue
            dot = (mu * u).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            scale = torch.where(theta > 1e-4, theta / sin_theta.clamp(min=1e-8), torch.ones_like(theta))
            mean_tan.add_((u - dot * mu) * scale, alpha=w)
        phi = mean_tan.norm(dim=1, keepdim=True)
        if torch.max(phi) < 1e-6:
            break
        step_dir = mean_tan / phi.clamp(min=1e-12)
        updated = torch.cos(phi) * mu + torch.sin(phi) * step_dir
        mu = torch.where(phi >= 1e-6, updated / updated.norm(dim=1, keepdim=True).clamp(min=1e-12), mu)

    return (mu * target_norm).reshape(shape)


def nuslerp_tensor(
    tensors: list[torch.Tensor],
    weights: list[float],
    base: torch.Tensor | None = None,
    *,
    row_wise: bool = True,
) -> torch.Tensor:
    """Normalized Uniform SLERP (per-row or global hyperspherical interpolation)."""
    work_dtype = torch.float64 if tensors[0].numel() <= 65536 else torch.float32
    vecs = [(t.to(work_dtype) - base.to(work_dtype)) if base is not None else t.to(work_dtype) for t in tensors]

    if not row_wise or vecs[0].ndim < 2 or vecs[0].shape[0] <= 1:
        merged = multi_slerp_tensor(tensors, weights, base=base)
        return merged

    if len(vecs) == 2:
        t = weights[1]
        merged = _nuslerp_pair_rows(vecs[0], vecs[1], t)
    else:
        merged = _nuslerp_multi_rows(vecs, weights)
    return (base.to(work_dtype) + merged) if base is not None else merged


def _copy_tokenizer_and_config(source_dir: Path, base_dir: Path | None, stage_dir: Path, dtype: str) -> None:
    cfg_src = (base_dir or source_dir) / "config.json"
    if not cfg_src.is_file():
        cfg_src = source_dir / "config.json"
    if cfg_src.is_file():
        cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
        cfg["torch_dtype"] = dtype
        (stage_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    tok_names = (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "added_tokens.json", "generation_config.json", "chat_template.jinja",
    )
    for name in tok_names:
        candidate = source_dir / name
        if candidate.is_file():
            shutil.copy2(candidate, stage_dir / name)


def _flush_shard(tensors: dict[str, torch.Tensor], shard_path: Path) -> None:
    save_file(tensors, str(shard_path), metadata={"format": "pt"})


def run_spherical_merge(
    *,
    method: str,
    model_dirs: list[Path],
    model_entries: list[dict[str, Any]],
    base_dir: Path | None,
    tokenizer_dir: Path,
    dtype: str,
    stage_dir: Path,
    parameters: dict[str, Any] | None = None,
    max_shard_bytes: int = 2_000_000_000,
) -> dict[str, Any]:
    """Stream tensors via safetensors mmap and write verified shards into stage_dir."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported merge dtype: {dtype!r}")
    out_dtype = _DTYPE_MAP[dtype]
    weights = _normalize_weights(model_entries)
    params = parameters or {}
    row_wise = bool(params.get("row_wise", True))

    stage_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        readers = [_CheckpointReader(path, stack) for path in model_dirs]
        base_reader = _CheckpointReader(base_dir, stack) if base_dir is not None else None
        key_order = list(base_reader.tensor_to_file.keys()) if base_reader is not None else list(readers[0].tensor_to_file.keys())

        current_shard: dict[str, torch.Tensor] = {}
        current_bytes = 0
        weight_map: dict[str, str] = {}
        total_tensors = len(key_order)

        # First stream tensors and write temporary sequential shard files, then rename cleanly
        temp_shard_files: list[tuple[Path, list[str]]] = []
        for idx, name in enumerate(key_order):
            if not all(name in r.tensor_to_file for r in readers):
                if "lm_head" in name:
                    continue
                raise ValueError(f"Tensor '{name}' missing from one of the source checkpoints")

            inputs = [r.get_tensor(name) for r in readers]
            base_tensor = base_reader.get_tensor(name) if base_reader is not None else None
            inputs, base_tensor = _align_vocab_rows(name, inputs, base_tensor)

            if not inputs[0].is_floating_point():
                out = inputs[0].clone()
            elif method == "multi_slerp":
                out = multi_slerp_tensor(inputs, weights, base=base_tensor).to(out_dtype)
            elif method == "nuslerp":
                out = nuslerp_tensor(inputs, weights, base=base_tensor, row_wise=row_wise).to(out_dtype)
            else:
                raise ValueError(f"Unknown spherical merge method: {method!r}")

            if not torch.isfinite(out.float()).all():
                raise ValueError(f"Non-finite values produced while merging tensor '{name}'")

            tensor_bytes = out.numel() * out.element_size()
            if current_shard and current_bytes + tensor_bytes > max_shard_bytes:
                tmp_name = f".tmp-shard-{len(temp_shard_files):05d}.safetensors"
                tmp_path = stage_dir / tmp_name
                _flush_shard(current_shard, tmp_path)
                temp_shard_files.append((tmp_path, list(current_shard.keys())))
                current_shard.clear()
                current_bytes = 0

            current_shard[name] = out.contiguous()
            current_bytes += tensor_bytes

            if (idx + 1) % max(1, total_tensors // 10) == 0 or idx + 1 == total_tensors:
                telemetry.emit_metric(progress=round(0.55 + 0.30 * ((idx + 1) / max(1, total_tensors)), 3), stage="merging_tensors")

        if current_shard:
            tmp_name = f".tmp-shard-{len(temp_shard_files):05d}.safetensors"
            tmp_path = stage_dir / tmp_name
            _flush_shard(current_shard, tmp_path)
            temp_shard_files.append((tmp_path, list(current_shard.keys())))
            current_shard.clear()

        total_shards = len(temp_shard_files)
        total_size = 0
        for s_idx, (tmp_path, keys) in enumerate(temp_shard_files, 1):
            final_name = "model.safetensors" if total_shards == 1 else f"model-{s_idx:05d}-of-{total_shards:05d}.safetensors"
            final_path = stage_dir / final_name
            tmp_path.rename(final_path)
            total_size += final_path.stat().st_size
            for k in keys:
                weight_map[k] = final_name

        if total_shards > 1:
            (stage_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
                encoding="utf-8",
            )

        _copy_tokenizer_and_config(tokenizer_dir, base_dir, stage_dir, dtype)
        return {"tensors": total_tensors, "shards": total_shards, "bytes": total_size, "weights": weights}
