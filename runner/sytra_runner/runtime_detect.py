"""Locate llama.cpp, Ollama, and LM Studio without relying on PATH alone."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def _command_override(env_name: str) -> list[str] | None:
    import shlex

    value = os.environ.get(env_name)
    return shlex.split(value, posix=os.name != "nt") if value else None


def _existing_file(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def project_roots(project_root: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if project_root is not None:
        roots.append(Path(project_root).resolve())
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[2])
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def find_llama_server(project_root: Path | None = None) -> list[str] | None:
    override = _command_override("SYTRA_LLAMA_SERVER")
    if override:
        return override
    executable = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if executable:
        return [executable]

    names = ("llama-server.exe", "llama-server")
    suffixes = (
        Path(".tools/llama.cpp/build/bin/Release"),
        Path(".tools/llama.cpp/build/bin/Debug"),
        Path(".tools/llama.cpp/build/bin"),
        Path(".tools/llama.cpp-bin"),
    )
    for root in project_roots(project_root):
        for suffix in suffixes:
            for name in names:
                found = _existing_file(root / suffix / name)
                if found:
                    return [str(found)]
    return None


def llama_server_lib_dir(launcher: list[str] | None) -> Path | None:
    if not launcher:
        return None
    path = Path(launcher[0])
    if path.is_file():
        return path.parent
    return None


def prepend_runtime_path(env: dict[str, str], launcher: list[str] | None) -> dict[str, str]:
    """Put CUDA runtime DLLs next to llama-server on PATH (Windows)."""
    updated = dict(env)
    lib_dir = llama_server_lib_dir(launcher)
    if lib_dir is None:
        return updated
    current = updated.get("PATH", "")
    prefix = str(lib_dir)
    if prefix.lower() not in current.lower():
        updated["PATH"] = prefix + os.pathsep + current
    return updated


def _well_known_ollama() -> list[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    home = Path.home()
    return [
        local / "Programs" / "Ollama" / "ollama.exe",
        Path(r"C:\Program Files\Ollama\ollama.exe"),
        home / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
    ]


def _well_known_lms() -> list[Path]:
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    return [
        home / ".lmstudio" / "bin" / "lms.exe",
        local / "LM-Studio" / "lms.exe",
        Path(r"C:\Program Files\LM Studio\lms.exe"),
    ]


def find_ollama() -> str | None:
    exe = shutil.which("ollama") or shutil.which("ollama.exe")
    if exe:
        return exe
    found = _existing_file(*_well_known_ollama())
    return str(found) if found else None


def find_lm_studio() -> str | None:
    exe = shutil.which("lms") or shutil.which("lms.exe")
    if exe:
        return exe
    found = _existing_file(*_well_known_lms())
    return str(found) if found else None
