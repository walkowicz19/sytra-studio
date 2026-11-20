"""Exercise the host-to-runner serving contract without loading a model."""
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location("sytra_serve_cli", Path(__file__).parents[1] / "scripts" / "serve_model.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


@dataclass
class FakePlan:
    compatible: bool = True
    preflight_command: list[str] = field(default_factory=list)


def test_host_arguments_reach_planner_and_dry_run_never_launches(monkeypatch, capsys):
    captured = {}
    def build(model, **kwargs):
        captured.update(model=model, **kwargs)
        return FakePlan()
    monkeypatch.setattr(serve, "build_backend_plan", build)
    monkeypatch.setattr(serve, "launch_backend_plan", lambda _: pytest.fail("dry run must not launch"))
    assert serve.main(["--model", "test.gguf", "--context", "8192", "--vram-limit", "4096",
                       "--ram-limit", "16000", "--kv-cache-quant", "q4_0", "--cpu-kv-cache",
                       "--no-flash-attention", "--project-root", ".", "--port", "8080",
                       "--vram-guardrail-mb", "1024",
                       "--draft-model-path", "draft.gguf",
                       "--dry-run", "--verify-engine"]) == 0
    assert captured["model"] == "test.gguf"
    assert captured["context_length"] == 8192
    assert captured["vram_limit_mb"] == 4096
    assert captured["kv_cache_quant"] == "q4_0"
    assert captured["vram_guardrail_mb"] == 1024
    assert captured["draft_model_path"] == 'draft.gguf'
    assert captured["cpu_kv_cache"] is True and captured["flash_attention"] is False
    assert '"compatible": true' in capsys.readouterr().out


def test_launch_is_explicit_and_engine_failure_blocks_it(monkeypatch):
    monkeypatch.setattr(serve, "build_backend_plan", lambda *args, **kwargs: FakePlan(preflight_command=["engine", "doctor"]))
    monkeypatch.setattr(serve.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="bad engine", stdout=""))
    monkeypatch.setattr(serve, "launch_backend_plan", lambda _: pytest.fail("failed preflight must not launch"))
    with pytest.raises(serve.ModelCompatibilityError, match="bad engine"):
        serve.main(["test.gguf", "--verify-engine", "--launch"])


def test_launch_calls_backend_and_returns_exit_code(monkeypatch):
    monkeypatch.setattr(serve, "build_backend_plan", lambda *args, **kwargs: FakePlan())
    monkeypatch.setattr(serve, "launch_backend_plan", lambda _: SimpleNamespace(wait=lambda: 7))
    assert serve.main(["test.gguf", "--launch"]) == 7


def test_idle_launch_uses_private_backend_port(monkeypatch):
    import sytra_runner.idle_proxy as proxy
    plans = []
    def build(_model, **kwargs):
        plans.append(kwargs['port'])
        return FakePlan()
    monkeypatch.setattr(serve, 'build_backend_plan', build)
    monkeypatch.setattr(serve, 'launch_backend_plan', lambda plan: 'backend')
    def run(host, port, ttl, launch, *, temperature, context_window=None, model_name=None):
        assert (host, port, ttl) == ('127.0.0.1', 8080, 1800)
        assert temperature == .25
        assert context_window == 4096
        assert model_name == 'test.gguf'
        assert launch(43210) == 'backend'
        return 0
    monkeypatch.setattr(proxy, 'serve_with_idle_timeout', run)
    assert serve.main(['test.gguf', '--port', '8080', '--idle-ttl-seconds', '1800', '--temperature', '.25', '--launch']) == 0
    assert plans == [8080, 43210]
