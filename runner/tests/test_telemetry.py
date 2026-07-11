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


def test_train_subprocess_emits_well_formed_transcript():
    events = _run_module("sytra_runner", FIXTURES / "run.golden.yaml")
    assert events[0]["type"] == "event"
    assert events[0]["event"] == "starting"

    terminal = [e for e in events if e.get("type") == "event" and e["event"] in ("done", "error")]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "done"

    metrics = [e for e in events if e["type"] == "metric"]
    assert all("step" in m for m in metrics)
    assert [m["step"] for m in metrics] == [1, 2, 3, 4]


def test_merge_subprocess_emits_well_formed_transcript():
    events = _run_module("sytra_runner.merge", FIXTURES / "merge.golden.yaml")
    assert events[0]["type"] == "event"
    assert events[0]["event"] == "starting"

    terminal = [e for e in events if e.get("type") == "event" and e["event"] in ("done", "error")]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "done"

    metrics = [e for e in events if e["type"] == "metric"]
    assert all(0.0 <= m["progress"] <= 1.0 for m in metrics)


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
