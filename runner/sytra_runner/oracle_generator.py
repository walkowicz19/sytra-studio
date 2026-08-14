"""Create a checkpoint-bound Sytra correctness oracle with Transformers.

This module intentionally imports torch/transformers only while generating an
oracle. They are reference-machine dependencies, not serving dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


ORACLE_FILE = ".sytra-oracle.json"
SUPPORTED_ADAPTERS = {
    "sytra-kimi-k2.7-code",
    "sytra-mixtral",
    "sytra-qwen3-moe",
    "sytra-qwen2-moe",
    "sytra-olmoe",
    "sytra-granite-moe",
}
DEFAULT_CASES = (
    "The capital of France is",
    "Write one safe Rust function that adds two integers:",
)


class OracleGenerationError(RuntimeError):
    """The reference oracle could not be produced safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleGenerationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OracleGenerationError(f"{path} must contain a JSON object")
    return value


def _engine_command(raw: str | None) -> list[str]:
    if raw:
        command = shlex.split(raw, posix=os.name != "nt")
        if command:
            return command
    executable = shutil.which("sytra-engine") or shutil.which("sytra-engine.exe")
    if executable:
        return [executable]
    raise OracleGenerationError(
        "sytra-engine is not on PATH; pass --engine-command or set SYTRA_ENGINE_COMMAND"
    )


def _checkpoint_fingerprint(model: Path, engine: Sequence[str]) -> str:
    completed = subprocess.run(
        [*engine, "fingerprint", "--model", str(model)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OracleGenerationError(f"sytra-engine fingerprint failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
        fingerprint = payload["model_fingerprint"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OracleGenerationError("sytra-engine returned an invalid fingerprint") from exc
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise OracleGenerationError("sytra-engine returned an invalid SHA-256 fingerprint")
    return fingerprint


def build_oracle_case(
    name: str,
    input_tokens: Sequence[int],
    predictions: Sequence[int],
    final_logits: Sequence[float],
    probe_tokens: Sequence[int],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Build the JSON-compatible part independent of torch for unit testing."""
    if len(input_tokens) < 2 or len(input_tokens) != len(predictions):
        raise OracleGenerationError(
            "each oracle case needs at least two inputs and one prediction per position"
        )
    probes = []
    for token in probe_tokens:
        if token < 0 or token >= len(final_logits):
            raise OracleGenerationError(f"probe token {token} is outside the vocabulary")
        probes.append(
            {
                "token": int(token),
                "expected": float(final_logits[token]),
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
            }
        )
    if not probes:
        raise OracleGenerationError("each oracle case needs at least one logit probe")
    return {
        "name": name,
        "input_tokens": [int(token) for token in input_tokens],
        "teacher_forced_predictions": [int(token) for token in predictions],
        "final_logit_probes": probes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Sytra's reference-logit and teacher-forced checkpoint oracle"
    )
    parser.add_argument("--model", required=True, help="Complete Sytra model directory")
    parser.add_argument(
        "--engine-command",
        default=os.environ.get("SYTRA_ENGINE_COMMAND"),
        help="Command used to run sytra-engine (also reads SYTRA_ENGINE_COMMAND)",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Reference prompt; repeat at least twice (two stable defaults are used otherwise)",
    )
    parser.add_argument("--probe-count", type=int, default=16)
    parser.add_argument("--absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Transformers device map; use 'none' for a normal single-device load "
            "that does not require accelerate"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--max-memory-json",
        help='Optional Transformers device memory map, e.g. {"0":"70GiB","cpu":"200GiB"}',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Explicitly allow the checkpoint repository's custom Python model code",
    )
    parser.add_argument("--reference-revision", help="Immutable reference/model commit")
    parser.add_argument("--force", action="store_true", help="Replace an existing oracle")
    return parser


def generate_oracle(args: argparse.Namespace) -> Path:
    model_root = Path(args.model).resolve()
    runtime = _read_json(model_root / ".sytra-runtime.json")
    adapter = runtime.get("architecture", {}).get("adapter")
    if adapter not in SUPPORTED_ADAPTERS:
        raise OracleGenerationError(
            "oracle execution is currently implemented only for "
            f"{sorted(SUPPORTED_ADAPTERS)}, got {adapter!r}"
        )
    output = model_root / ORACLE_FILE
    if output.exists() and not args.force:
        raise OracleGenerationError(f"{output} already exists; pass --force to replace it")
    cases = args.cases or list(DEFAULT_CASES)
    if len(cases) < 2:
        raise OracleGenerationError("at least two independent --case prompts are required")
    if not 1 <= args.probe_count <= 128:
        raise OracleGenerationError("--probe-count must be between 1 and 128")
    if not 0 <= args.absolute_tolerance <= 0.1:
        raise OracleGenerationError("--absolute-tolerance must be between 0 and 0.1")
    if not 0 <= args.relative_tolerance <= 0.05:
        raise OracleGenerationError("--relative-tolerance must be between 0 and 0.05")

    download = _read_json(model_root / ".sytra-model.json")
    reference_revision = args.reference_revision or download.get("resolved_revision")
    if not isinstance(reference_revision, str) or len(reference_revision) < 7:
        raise OracleGenerationError("an immutable --reference-revision is required")
    fingerprint = _checkpoint_fingerprint(model_root, _engine_command(args.engine_command))

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise OracleGenerationError(
            "oracle generation needs torch, transformers, and accelerate on the reference machine"
        ) from exc

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    load_options: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": True,
    }
    if args.device_map.lower() != "none":
        load_options["device_map"] = args.device_map
    if args.max_memory_json:
        try:
            load_options["max_memory"] = json.loads(args.max_memory_json)
        except json.JSONDecodeError as exc:
            raise OracleGenerationError("--max-memory-json is invalid JSON") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    reference = AutoModelForCausalLM.from_pretrained(model_root, **load_options)
    reference.eval()
    oracle_cases: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, prompt in enumerate(cases):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] < 2:
                raise OracleGenerationError(f"case {index} tokenized to fewer than two positions")
            model_device = reference.get_input_embeddings().weight.device
            encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}
            logits = reference(**encoded, use_cache=False).logits[0].float().cpu()
            predictions = logits.argmax(dim=-1).tolist()
            final = logits[-1]
            count = min(args.probe_count, final.numel())
            probe_tokens = torch.topk(final.abs(), k=count).indices.tolist()
            oracle_cases.append(
                build_oracle_case(
                    f"reference-{index + 1}",
                    input_ids[0].tolist(),
                    predictions,
                    final.tolist(),
                    probe_tokens,
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                )
            )

    payload = {
        "schema_version": 1,
        "adapter": adapter,
        "model_fingerprint": fingerprint,
        "reference_implementation": f"transformers@{transformers.__version__}",
        "reference_revision": reference_revision,
        "cases": oracle_cases,
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=model_root, prefix=".sytra-oracle-", suffix=".tmp", delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(output)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = generate_oracle(args)
    except OracleGenerationError as exc:
        print(f"sytra oracle generation failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps({"oracle": str(output), "next": "sytra-engine oracle-check --model " + str(Path(args.model).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
