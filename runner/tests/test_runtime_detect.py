from pathlib import Path

from sytra_runner.runtime_detect import (
    extra_runtime_roots,
    find_colibri,
    find_llama_server,
    find_lm_studio,
    find_ollama,
    find_sytra_engine,
    prepend_runtime_path,
    project_roots,
)
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


def test_project_roots_includes_source_root_env(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("SYTRA_SOURCE_ROOT", str(checkout))
    monkeypatch.delenv("SYTRA_WORKSPACE", raising=False)
    roots = project_roots(tmp_path / "workspace")
    assert checkout.resolve() in roots
    assert extra_runtime_roots()[0].resolve() == checkout.resolve()


def test_find_llama_server_searches_extra_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("SYTRA_LLAMA_SERVER", raising=False)
    source = tmp_path / "checkout"
    exe = source / ".tools" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake")
    monkeypatch.setattr(
        "sytra_runner.runtime_detect.project_roots",
        lambda project_root=None: [source],
    )
    found = find_llama_server(tmp_path)
    assert found == [str(exe.resolve())]


def test_find_sytra_engine_in_target_build_debug(tmp_path, monkeypatch):
    monkeypatch.delenv("SYTRA_ENGINE_COMMAND", raising=False)
    monkeypatch.delenv("SYTRA_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("SYTRA_WORKSPACE", raising=False)
    exe = tmp_path / "target-build" / "debug" / "sytra-engine.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"fake")
    found = find_sytra_engine(tmp_path)
    assert found == [str(exe.resolve())]


def test_find_colibri_in_tools_and_home_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SYTRA_COLIBRI_COMMAND", raising=False)
    home = tmp_path / "colibri-home"
    exe = home / "coli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("SYTRA_COLIBRI_HOME", str(home))
    found = find_colibri(tmp_path)
    assert found == [str(exe.resolve())]


def test_find_colibri_wraps_python_launcher(tmp_path, monkeypatch):
    monkeypatch.delenv("SYTRA_COLIBRI_COMMAND", raising=False)
    monkeypatch.delenv("SYTRA_COLIBRI_HOME", raising=False)
    monkeypatch.setattr("sytra_runner.runtime_detect.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "sytra_runner.runtime_detect.project_roots",
        lambda project_root=None: [tmp_path],
    )
    script = tmp_path / ".tools" / "colibri" / "coli"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env python3\nprint('coli')\n", encoding="utf-8")
    found = find_colibri(tmp_path)
    assert found is not None
    assert found[-1] == str(script.resolve())
    assert found[0].lower().endswith("python.exe") or "python" in Path(found[0]).name.lower()


def test_prepend_runtime_path_uses_coli_dir_not_python(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "C:\\Windows\\System32")
    script = tmp_path / "coli"
    script.write_text("print('x')\n", encoding="utf-8")
    env = prepend_runtime_path(
        {"PATH": "C:\\Windows\\System32"},
        ["C:\\Python\\python.exe", str(script)],
    )
    assert env["PATH"].startswith(str(script.parent))


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
