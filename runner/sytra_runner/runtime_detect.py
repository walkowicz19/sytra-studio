"""Locate llama.cpp, Ollama, LM Studio, Sytra engine, and Colibri."""
from __future__ import annotations

import os
import shutil
import sys
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


def extra_runtime_roots() -> list[Path]:
    """Search the MCP install home and an optional source checkout.

    MCP runs with workspace ``~/.sytra``, which does not contain
    ``.tools/llama.cpp``, ``.tools/colibri``, or ``target-build/sytra-engine``. Point
    ``SYTRA_SOURCE_ROOT`` or ``~/.sytra/.sytra-source-root`` at the git
    checkout so planning still finds those binaries.
    """
    roots: list[Path] = []
    for env_name in ("SYTRA_SOURCE_ROOT", "SYTRA_WORKSPACE"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    home_sytra = Path.home() / ".sytra"
    roots.append(home_sytra)
    pointer = home_sytra / ".sytra-source-root"
    try:
        text = pointer.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        text = ""
    if text and not text.startswith("#"):
        roots.append(Path(text))
    return roots


def project_roots(project_root: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if project_root is not None:
        roots.append(Path(project_root).resolve())
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[2])
    roots.extend(extra_runtime_roots())
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
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
    for token in launcher[:2]:
        path = Path(token)
        if not path.is_file():
            continue
        if path.name.lower().startswith("python"):
            continue
        return path.parent
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


def find_sytra_engine(project_root: Path | None = None) -> list[str] | None:
    override = _command_override("SYTRA_ENGINE_COMMAND")
    if override:
        return override
    executable = shutil.which("sytra-engine") or shutil.which("sytra-engine.exe")
    if executable:
        return [executable]

    candidates = (
        Path("target-build/release/sytra-engine.exe"),
        Path("target-build/release/sytra-engine"),
        Path("target-build/debug/sytra-engine.exe"),
        Path("target-build/debug/sytra-engine"),
        Path("target/release/sytra-engine"),
        Path("target/debug/sytra-engine"),
        Path("bin/sytra-engine.exe"),
        Path("bin/sytra-engine"),
    )
    for root in project_roots(project_root):
        for suffix in candidates:
            found = _existing_file(root / suffix)
            if found:
                return [str(found)]
    return None


def _is_native_executable(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in {".exe", ".cmd", ".bat"}:
        return True
    try:
        header = path.read_bytes()[:4]
    except OSError:
        return False
    return header[:2] == b"MZ" or header[:4] in {
        b"\x7fELF",
        b"\xcf\xfa\xed\xfe",
        b"\xce\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xfe\xed\xfa\xcf",
    }


def coli_command_for(path: Path) -> list[str]:
    resolved = path.resolve()
    if _is_native_executable(resolved):
        return [str(resolved)]
    return [sys.executable, str(resolved)]


def find_colibri_in_dir(directory: Path) -> list[str] | None:
    names = ("coli.exe", "coli.cmd", "coli.py", "coli")
    found = _existing_file(
        *(directory / name for name in names),
        *(directory / "c" / name for name in names),
    )
    if found:
        return coli_command_for(found)
    if not directory.is_dir():
        return None
    try:
        children = list(directory.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_dir():
            nested = _existing_file(
                *(child / name for name in names),
                *(child / "c" / name for name in names),
            )
            if nested:
                return coli_command_for(nested)
    return None


def find_colibri(project_root: Path | None = None) -> list[str] | None:
    override = _command_override("SYTRA_COLIBRI_COMMAND")
    if override:
        return override
    home = os.environ.get("SYTRA_COLIBRI_HOME")
    if home:
        override_home = find_colibri_in_dir(Path(home))
        if override_home:
            return override_home
    executable = (
        shutil.which("coli")
        or shutil.which("coli.exe")
        or shutil.which("coli.cmd")
        or shutil.which("coli.py")
    )
    if executable:
        return coli_command_for(Path(executable))
    suffixes = (
        Path(".tools/colibri"),
        Path(".tools/colibri/c"),
        Path("colibri"),
        Path("colibri/c"),
    )
    for root in project_roots(project_root):
        for suffix in suffixes:
            found = find_colibri_in_dir(root / suffix)
            if found:
                return found
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    extras = (
        Path.home() / "colibri",
        Path.home() / "colibri" / "c",
        local / "Programs" / "colibri",
    )
    for directory in extras:
        found = find_colibri_in_dir(directory)
        if found:
            return found
    return None


def llama_server_provision_hint(project_root: Path | None = None) -> str:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    script = Path(__file__).resolve().parents[1] / "scripts" / "provision_llama_cpp.py"
    return (
        "llama-server was not found. Run "
        f"`python {script} --project-root {root}`, install llama.cpp on PATH, "
        "set SYTRA_LLAMA_SERVER, or set SYTRA_SOURCE_ROOT / ~/.sytra/.sytra-source-root "
        "to a checkout that already has .tools/llama.cpp."
    )
