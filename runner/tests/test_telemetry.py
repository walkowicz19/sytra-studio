import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "fixtures"


def _run_module(module: str, config_path: Path) -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-m", module, str(config_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        # The runner probes heavy ML imports (torch/transformers) before
        # falling back to simulation; a cold import scan alone can take
        # >30s on an HDD.
        timeout=180,
    )
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    return [json.loads(l) for l in lines]


def _terminal_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "event" and e["event"] in ("done", "error")]


def test_train_subprocess_fails_closed_without_cuda_training_stack():
    events = _run_module("sytra_runner", FIXTURES / "run.golden.yaml")
    terminal = _terminal_events(events)
    assert len(terminal) == 1
    assert terminal[0]["event"] == "error"
    assert not any(e.get("event") == "done" for e in events)


def test_merge_subprocess_fails_closed_without_mergekit_or_on_error():
    events = _run_module("sytra_runner.merge", FIXTURES / "merge.golden.yaml")
    terminal = _terminal_events(events)
    assert len(terminal) == 1
    assert terminal[0]["event"] == "error"
    assert not any(e.get("event") == "done" for e in events)


def test_merge_subprocess_with_red_verdict_emits_error_and_no_done():
    bad_config = FIXTURES / "merge.red_verdict.tmp.yaml"
    raw = (FIXTURES / "merge.golden.yaml").read_text(encoding="utf-8")
    bad_config.write_text(raw.replace("verdict: green", "verdict: red"), encoding="utf-8")
    try:
        events = _run_module("sytra_runner.merge", bad_config)
        assert events[0]["event"] == "error"
        assert not any(e.get("event") == "done" for e in events)
    finally:
        bad_config.unlink(missing_ok=True)
