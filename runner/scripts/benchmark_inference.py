"""Measure a real llama.cpp server. Refuses to invent numbers.

Usage:
  python scripts/benchmark_inference.py --model path.gguf --launch
Writes JSON under runs/benchmarks/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.model_planner import ModelCompatibilityError, build_backend_plan
from sytra_runner.runtime_detect import prepend_runtime_path
from sytra_runner.serve_ports import port_in_use, require_free_port


def _post(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _nvidia_query() -> dict | None:
    exe = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [
                exe,
                "--query-gpu=memory.used,utilization.gpu,utilization.memory,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (result.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "vram_used_mb": int(float(parts[0])),
            "gpu_util_pct": int(float(parts[1])),
            "mem_util_pct": int(float(parts[2])),
            "gpu_temp_c": int(float(parts[3])),
        }
    except ValueError:
        return None


def _wait_http(url: str, timeout_sec: float) -> None:
    deadline = time.perf_counter() + timeout_sec
    last = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(0.4)
    raise RuntimeError(f"llama-server did not become ready at {url}: {last}")


def _launch_server(command: list[str], host: str, port: int) -> subprocess.Popen:
    require_free_port(host, port)
    env = prepend_runtime_path(os.environ.copy(), command)
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc


def _completion(base_url: str, prompt: str, max_tokens: int, timeout: int) -> tuple[dict, float, float]:
    """Returns body, time-to-first-byte, total elapsed."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        first = time.perf_counter()
        body = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    return body, (first - started if first else elapsed), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real llama.cpp generation benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--vram-limit", type=int, default=12288)
    parser.add_argument("--ram-limit", type=int, default=12288)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--ngl", type=int, default=None)
    parser.add_argument("--allow-cpu-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    if args.launch:
        args.base_url = f"http://{args.host}:{args.port}"
    try:
        plan = build_backend_plan(
            args.model,
            host=args.host,
            port=args.port,
            vram_limit_mb=args.vram_limit,
            ram_limit_mb=args.ram_limit,
            context=args.context,
            force_gpu_layers=args.ngl,
            allow_cpu_only=args.allow_cpu_only,
            project_root=args.project_root,
        )
    except ModelCompatibilityError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    if not plan.compatible:
        print(json.dumps({"ok": False, "plan": plan.to_dict()}), file=sys.stderr)
        return 2

    proc = None
    base = args.base_url.rstrip("/")
    if args.launch:
        if port_in_use(args.host, args.port):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": f"Port {args.host}:{args.port} is occupied; refusing to attach to a stale server.",
                    }
                ),
                file=sys.stderr,
            )
            return 4
        proc = _launch_server(plan.command, args.host, args.port)
        try:
            _wait_http(base + "/health", timeout_sec=180)
        except RuntimeError as exc:
            if proc.poll() is None:
                proc.terminate()
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
            return 3

    gpu_before = _nvidia_query()
    try:
        body, ttft, elapsed = _completion(base, args.prompt, args.max_tokens, timeout=300)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        if proc is not None and proc.poll() is None:
            proc.terminate()
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"llama.cpp HTTP call failed: {exc}",
                    "hint": "Start llama-server with the command in plan.command first, or pass --launch.",
                    "plan": plan.to_dict(),
                }
            ),
            file=sys.stderr,
        )
        return 3
    gpu_after = _nvidia_query()
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    timings = body.get("timings") or {}
    prompt_n = timings.get("prompt_n") or body.get("tokens_evaluated") or 0
    predicted_n = timings.get("predicted_n") or 0
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    prompt_tok_s = timings.get("prompt_per_second")
    gen_tok_s = timings.get("predicted_per_second")
    text = body.get("content") or ""
    record = {
        "ok": True,
        "simulated": False,
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "output": text,
        "time_to_first_token_sec": round(ttft, 3),
        "elapsed_sec": round(elapsed, 3),
        "prompt_tokens": prompt_n,
        "completion_tokens": predicted_n,
        "prompt_tok_s": prompt_tok_s,
        "generation_tok_s": gen_tok_s,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "load_time_ms": timings.get("model_load_ms"),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "peak_vram_mb_observed": (gpu_after or {}).get("vram_used_mb"),
        "plan": plan.to_dict(),
        "runtime_version": plan.runtime_version,
        "backend": plan.backend,
        "gpu_layers": plan.llama_offload.gpu_layers if plan.llama_offload else None,
        "quantization": plan.artifact.quantization,
        "architecture": plan.artifact.architecture,
        "context": args.context,
        "command": plan.command,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = args.label or f"ngl{record['gpu_layers']}"
    out = (
        Path(args.out)
        if args.out
        else Path("runs") / "benchmarks" / f"bench-{stamp}-{label}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "path": str(out),
                "generation_tok_s": gen_tok_s,
                "prompt_tok_s": prompt_tok_s,
                "time_to_first_token_sec": record["time_to_first_token_sec"],
                "peak_vram_mb_observed": record["peak_vram_mb_observed"],
                "gpu_layers": record["gpu_layers"],
                "output": text[:240],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
