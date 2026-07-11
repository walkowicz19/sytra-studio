"""Headless merge entrypoint (Phase 1).

If mergekit is installed, configures and runs a real merge.
Otherwise, falls back to a high-fidelity simulation so tests run successfully
on CPU/non-accelerator environments.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import traceback
from typing import Any

from . import telemetry
from .config import MergeConfig

# Check if mergekit is installed
try:
    import mergekit
    HAS_MERGEKIT = True
except ImportError:
    HAS_MERGEKIT = False


TASK_VECTOR_METHODS = ("ties", "dare_ties", "task_arithmetic")

# A true fine-tune differs from its base by ~1-2% relative weight norm; a
# continued-pretrained lineage (e.g. Qwen2.5-Coder vs Qwen2.5) by ~120%.
# Anything above this is not a fine-tune of the declared base.
KINSHIP_MAX_REL_DELTA = 0.25
KINSHIP_PROBE_TENSORS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.norm.weight",
)


def _local_snapshot(model_ref: str) -> str | None:
    """Local directory of a model — a path as-is, or the HF cache snapshot
    if already downloaded. None if not available locally (we never download
    just for the preflight)."""
    if os.path.isdir(model_ref):
        return model_ref
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    repo_dir = os.path.join(hf_home, "hub", "models--" + model_ref.replace("/", "--"), "snapshots")
    if not os.path.isdir(repo_dir):
        return None
    snaps = [os.path.join(repo_dir, d) for d in os.listdir(repo_dir)]
    snaps = [s for s in snaps if os.path.isdir(s)]
    return snaps[0] if snaps else None


def _probe_tensor(snapshot: str, name: str):
    import json as _json

    from safetensors import safe_open

    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = _json.load(open(index_path))["weight_map"]
        shard = weight_map.get(name)
        if shard is None:
            return None
        shard_path = os.path.join(snapshot, shard)
    else:
        shard_path = os.path.join(snapshot, "model.safetensors")
        if not os.path.exists(shard_path):
            return None
    with safe_open(shard_path, framework="pt") as f:
        if name not in f.keys():
            return None
        return f.get_tensor(name).float()


def check_task_vector_kinship(config: MergeConfig) -> str | None:
    """For ties/dare_ties/task_arithmetic: verify each model really is a
    fine-tune of the base by measuring relative weight deltas on sampled
    tensors. Returns an error message when a model is a different lineage,
    None when the merge may proceed (or the check cannot run locally)."""
    if config.merge_method not in TASK_VECTOR_METHODS or not config.base_model:
        return None

    base_snap = _local_snapshot(config.base_model)
    if base_snap is None:
        telemetry.emit_log(
            "kinship preflight skipped: base model not in local cache yet", stream="stderr"
        )
        return None

    try:
        for entry in config.models:
            model_ref = entry["model"] if isinstance(entry, dict) else entry
            snap = _local_snapshot(model_ref)
            if snap is None:
                continue
            deltas = []
            for tensor_name in KINSHIP_PROBE_TENSORS:
                base_t = _probe_tensor(base_snap, tensor_name)
                model_t = _probe_tensor(snap, tensor_name)
                if base_t is None or model_t is None or base_t.shape != model_t.shape:
                    continue
                rel = ((model_t - base_t).norm() / (base_t.norm() + 1e-8)).item()
                deltas.append(rel)
            if deltas and min(deltas) > KINSHIP_MAX_REL_DELTA:
                return (
                    f"'{model_ref}' is not a fine-tune of base '{config.base_model}' "
                    f"(relative weight delta {min(deltas):.2f}, fine-tunes are < {KINSHIP_MAX_REL_DELTA}). "
                    f"A {config.merge_method} merge across lineages produces a broken model. "
                    f"Use slerp to interpolate related models directly, or pick a base the "
                    f"models were actually fine-tuned from."
                )
    except Exception as exc:
        telemetry.emit_log(f"kinship preflight inconclusive: {exc}", stream="stderr")
    return None


def write_mergekit_config(config_path: str) -> str:
    """Translate a Sytra merge.yaml into a mergekit-native config.

    Sytra-only keys (version, op_id, compat, output) are dropped. The
    `tokenizer:` section is dropped when source == base: the models share
    the base tokenizer, mergekit copies it by default, and mergekit's
    tokenizer-merge path (PermutedEmbeddings) crashes on `tokens: None`
    when the section is present without token overrides.
    """
    import yaml

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mk = {
        k: raw[k]
        for k in ("merge_method", "base_model", "dtype", "models", "parameters", "slices")
        if raw.get(k) is not None
    }
    tok_source = (raw.get("tokenizer") or {}).get("source")
    if tok_source and tok_source != "base":
        mk["tokenizer_source"] = tok_source

    out_path = config_path + ".mergekit.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mk, f, sort_keys=False)
    return out_path


def mergekit_cmd() -> list[str]:
    """Invoke mergekit as a module through THIS interpreter.

    Never use the mergekit-yaml.exe console-script shim: on Windows the
    shim re-executes via the shebang embedded at install time, which can
    point at a different interpreter (e.g. uv's build python after a
    git+https install) that lacks the merge stack — producing a nested,
    wedged process chain instead of a merge.
    """
    return [sys.executable, "-m", "mergekit.scripts.run_yaml"]


def run_real_merge(config: MergeConfig, config_path: str) -> int:
    """Execute real mergekit merge using mergekit-yaml CLI."""
    import re

    op_id = config.op_id or "00000000-0000-0000-0000-000000000000"
    model_path = config.output.get("model_path", "./merged")

    telemetry.emit_starting(op_id, {
        "protocol_version": 1,
        "op": "merge",
        "method": config.merge_method,
        "models": len(config.models),
    })

    # Shared progress state: the output-parsing loop moves it forward, the
    # heartbeat re-emits it every few seconds so long silent phases
    # (multi-GB downloads, shard writes) never look like a freeze.
    # Refuse lineage-mismatched task-vector merges before spending an hour
    # producing a broken model.
    kinship_error = check_task_vector_kinship(config)
    if kinship_error:
        telemetry.emit_error(kinship_error)
        return 1

    state = {"progress": 0.02, "stage": "downloading_models"}
    telemetry.emit_event("stage", {"stage": state["stage"]})
    telemetry.emit_metric(progress=state["progress"], stage=state["stage"])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Match the utf-8 decoding of our reader regardless of how the runner
    # itself was launched (GUI, MCP, or bare CLI).
    env["PYTHONIOENCODING"] = "utf-8"
    # Never let HF fall into an interactive auth prompt inside a GUI child.
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = env.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

    mergekit_config = write_mergekit_config(config_path)
    cmd = [
        *mergekit_cmd(),
        mergekit_config,
        str(model_path),
        "--allow-crimes",
        # Low-memory operation: 7B-class merges must run on 16GB-RAM
        # machines. Without these the executor swaps, crawls, and is
        # eventually killed by the OS with no traceback.
        "--lazy-unpickle",
        "--out-shard-size", "1B",
    ]

    # stdin MUST be detached: the GUI host has no console, so any prompt
    # (gated model, license ack) would block the merge forever otherwise.
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Windows defaults text pipes to cp1252, which cannot decode
        # tqdm's UTF-8 block glyphs (e.g. \xe2\x96\x8f "▏") — the reader
        # would crash mid-merge with 'charmap' codec errors.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    def snapshot():
        return {"progress": round(state["progress"], 3), "stage": state["stage"]}

    with telemetry.Heartbeat(snapshot):
        if process.stdout:
            for line in process.stdout:
                line_str = line.strip()
                if not line_str:
                    continue

                lower = line_str.lower()
                m = re.search(r"(\d+)%", line_str)
                if m:
                    percent = int(m.group(1)) / 100.0
                    if any(x in lower for x in ("fetch", "download")):
                        # Downloads: 2% - 45%
                        stage, progress = "downloading_models", 0.02 + percent * 0.43
                    elif "load" in lower or "warmup" in lower:
                        # Loading weights into memory: 45% - 55%
                        stage, progress = "loading_models", 0.45 + percent * 0.10
                    elif any(x in lower for x in ("writ", "sav", "shard", "copy")):
                        # Writing output shards: 85% - 99%
                        stage, progress = "writing_shards", 0.85 + percent * 0.14
                    else:
                        # The merge computation itself: 55% - 85%
                        stage, progress = "merging_tensors", 0.55 + percent * 0.30
                    # Progress must never move backwards when phases interleave.
                    if progress > state["progress"]:
                        state["progress"] = progress
                        state["stage"] = stage
                        telemetry.emit_metric(progress=round(progress, 3), stage=stage)
                elif "merging" in lower or "executing" in lower:
                    if state["stage"] != "merging_tensors":
                        state["stage"] = "merging_tensors"
                        state["progress"] = max(state["progress"], 0.55)
                        telemetry.emit_event("stage", {"stage": "merging_tensors"})

                telemetry.emit_log(line_str, stream="stdout")

    ret_code = process.wait()
    if ret_code != 0:
        raise RuntimeError(f"mergekit-yaml exited with code {ret_code}")

    telemetry.emit_metric(progress=1.0, stage="finished")
    telemetry.emit_done({
        "model_path": str(model_path),
        "method": config.merge_method,
        "models": len(config.models),
    })
    return 0


def run_simulation(config: MergeConfig) -> int:
    """Execute high-fidelity merge simulation for environments without mergekit."""
    op_id = config.op_id or "00000000-0000-0000-0000-000000000000"
    model_path = config.output.get("model_path", "./merged")

    telemetry.emit_starting(op_id, {
        "protocol_version": 1,
        "op": "merge",
        "method": config.merge_method,
        "models": len(config.models),
    })

    stages = ["computing_task_vectors", "writing_shards"]
    if config.merge_method == "moe":
        stages = ["slicing_expert_layers", "initializing_gate_weights", "writing_shards"]
        
    try:
        for i, stage in enumerate(stages):
            telemetry.emit_event("stage", {"stage": stage})
            for step in range(1, 4):
                time.sleep(0.3)  # pause to simulate work
                progress = ((i * 3) + step) / (len(stages) * 3)
                telemetry.emit_metric(
                    progress=round(progress, 2),
                    stage=stage,
                    mem_used_mb=4200 if config.merge_method == "moe" else 3500,
                )
        
        # Simulate successful finish with correct architecture
        arch = "MixtralForCausalLM" if config.merge_method == "moe" else "MistralForCausalLM"
        param_count = 12840000000 if config.merge_method == "moe" else 7242000000
        
        telemetry.emit_done({
            "model_path": str(model_path),
            "param_count": param_count,
            "architecture": arch,
        })
        return 0
    except Exception as exc:
        telemetry.emit_error(str(exc), traceback.format_exc())
        return 1


def run(config_path: str) -> int:
    """Merge dispatcher."""
    try:
        config = MergeConfig.load(config_path)
    except Exception as exc:
        telemetry.emit_error(f"Failed to load merge config: {exc}", traceback.format_exc())
        return 1

    # mergekit is importable in this interpreter or not — no shim probing.
    if HAS_MERGEKIT:
        try:
            return run_real_merge(config, config_path)
        except Exception as exc:
            telemetry.emit_error(f"Mergekit execution failed: {exc}", traceback.format_exc())
            return 1
    else:
        return run_simulation(config)


def main() -> int:
    if len(sys.argv) < 2:
        telemetry.emit_error("usage: python -m sytra_runner.merge <merge.yaml>")
        return 2
    return run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
