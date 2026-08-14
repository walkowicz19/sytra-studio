from pathlib import Path

from sytra_runner.runtime_detect import find_llama_server, find_lm_studio, find_ollama, prepend_runtime_path
from sytra_runner.serve_ports import port_in_use, require_free_port
from sytra_runner.model_planner import ModelCompatibilityError, inspect_model, build_backend_plan
from gguf_test_file import write_metadata_gguf


def test_find_llama_server_in_tools_release(tmp_path, monkeypatch):
    monkeypatch.delenv("SYTRA_LLAMA_SERVER", raising=False)
    exe = tmp_path / ".tools" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake")
    found = find_llama_server(tmp_path)
    assert found == [str(exe.resolve())]


def test_prepend_runtime_path_puts_dll_dir_first(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")
    env = prepend_runtime_path({"PATH": "C:\\Windows\\System32"}, [str(exe)])
    assert env["PATH"].startswith(str(exe.parent))


def test_require_free_port_errors_when_bound():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert port_in_use("127.0.0.1", port)
        try:
            require_free_port("127.0.0.1", port)
            raised = False
        except RuntimeError as exc:
            raised = True
            assert str(port) in str(exc)
        assert raised
    finally:
        sock.close()


def test_qwen35_filename_is_not_treated_as_qwen2(tmp_path):
    model = write_metadata_gguf(
        tmp_path / "Qwen3.5-9B-Q4_K_M.gguf",
        {"general.architecture": "qwen2", "qwen2.block_count": 32},
    )
    try:
        inspect_model(model)
        raised = False
    except ModelCompatibilityError as exc:
        raised = True
        assert "Qwen3.5" in str(exc) or "qwen2" in str(exc).lower()
    assert raised


def test_explicit_cpu_only_allowed_when_gpu_visible(tmp_path, monkeypatch):
    model = write_metadata_gguf(
        tmp_path / "tiny.gguf",
        {"general.architecture": "llama", "llama.block_count": 8},
        payload_bytes=1024 * 1024,
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setenv("SYTRA_GPU_MEMORY_MB", "12288")
    plan = build_backend_plan(
        model,
        vram_limit_mb=4096,
        ram_limit_mb=8192,
        force_gpu_layers=0,
        allow_cpu_only=True,
    )
    assert plan.compatible
    assert plan.llama_offload is not None
    assert plan.llama_offload.gpu_layers == 0


def test_silent_cpu_fallback_rejected(tmp_path, monkeypatch):
    model = write_metadata_gguf(
        tmp_path / "tiny.gguf",
        {"general.architecture": "llama", "llama.block_count": 8},
        payload_bytes=1024 * 1024,
    )
    monkeypatch.setenv("SYTRA_LLAMA_SERVER", "fake-llama-server")
    monkeypatch.setenv("SYTRA_GPU_MEMORY_MB", "12288")
    plan = build_backend_plan(
        model,
        vram_limit_mb=4096,
        ram_limit_mb=8192,
        force_gpu_layers=0,
        allow_cpu_only=False,
    )
    assert not plan.compatible
    assert plan.command == []
    assert any("silently" in reason.lower() or "CPU-only" in reason for reason in plan.reasons)
