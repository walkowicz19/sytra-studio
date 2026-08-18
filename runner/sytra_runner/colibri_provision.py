"""Download a pinned Colibri (`coli`) release into `.tools/colibri` (gitignored).

Frontier streamed-MoE serving is Colibri's job. Sytra unpacks the official
Windows/Linux/macOS archive next to llama.cpp so `plan_inference` and
`serve_model` can find `coli` without a manual PATH install.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .runtime_detect import find_colibri, find_colibri_in_dir

RELEASE = "v1.6.2"
BASE = f"https://github.com/JustVugg/colibri/releases/download/{RELEASE}"
VERSION_STAMP = ".sytra-colibri-version"

ASSETS: dict[str, tuple[str, str]] = {
    "win32": (
        "colibri-v1.6.2-windows-x86_64.zip",
        "12d4cb059a8d3a4f7700eaf16a2cd605de78099e48ddcd756e6b67b1043a1596",
    ),
    "linux": (
        "colibri-v1.6.2-linux-x86_64.tar.gz",
        "a76601d781fae4bd48e6be2a73ca3c4b0f5179e7bf008429ad4edd03d55872d0",
    ),
    "darwin": (
        "colibri-v1.6.2-macos-arm64.tar.gz",
        "07a30190763c25e04abea33cad92e3dc23984b1792830f61d3513ed8fcdd7621",
    ),
}


class ColibriProvisionError(RuntimeError):
    """Pinned Colibri archive could not be downloaded or unpacked."""


def colibri_provision_allowed() -> bool:
    if os.environ.get("SYTRA_SKIP_COLIBRI_PROVISION") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


def colibri_install_root(project_root: Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()
    source = os.environ.get("SYTRA_SOURCE_ROOT")
    if source:
        return Path(source).resolve()
    return Path(__file__).resolve().parents[2]


def colibri_tools_dir(project_root: Path | None = None) -> Path:
    return colibri_install_root(project_root) / ".tools" / "colibri"


def current_asset(platform: str | None = None) -> tuple[str, str]:
    key = platform or sys.platform
    if key.startswith("linux"):
        key = "linux"
    elif key.startswith("darwin"):
        key = "darwin"
    elif key.startswith("win"):
        key = "win32"
    asset = ASSETS.get(key)
    if asset is None:
        raise ColibriProvisionError(f"No pinned Colibri release for platform {key!r}")
    return asset


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "sytra-studio-colibri-provision",
            "Accept": "application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(dest)
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tarred:
            tarred.extractall(dest)
        return
    raise ColibriProvisionError(f"Unsupported Colibri archive: {archive.name}")


def _flatten_single_root(dest: Path) -> None:
    children = [child for child in dest.iterdir() if child.name != VERSION_STAMP]
    if len(children) != 1 or not children[0].is_dir():
        return
    nested = children[0]
    for item in nested.iterdir():
        target = dest / item.name
        if target.exists():
            continue
        item.rename(target)
    try:
        nested.rmdir()
    except OSError:
        pass


def provision_colibri(
    project_root: Path | None = None,
    *,
    download: bool = True,
    platform: str | None = None,
) -> list[str]:
    dest = colibri_tools_dir(project_root)
    dest.mkdir(parents=True, exist_ok=True)
    stamp = dest / VERSION_STAMP
    local = find_colibri_in_dir(dest)
    if (
        local
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8").strip() == RELEASE
    ):
        return local
    if not download:
        raise ColibriProvisionError(f"Colibri {RELEASE} is not installed at {dest}")

    filename, expected_sha = current_asset(platform)
    print(f"Provisioning Colibri {RELEASE} into {dest}", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="sytra-colibri-") as tmp:
        archive = Path(tmp) / filename
        download_file(f"{BASE}/{filename}", archive)
        digest = sha256_file(archive)
        if digest != expected_sha:
            raise ColibriProvisionError(
                f"Colibri archive SHA256 mismatch for {filename}: got {digest}, expected {expected_sha}"
            )
        extract_archive(archive, dest)
    _flatten_single_root(dest)
    stamp.write_text(RELEASE + "\n", encoding="utf-8")
    found = find_colibri_in_dir(dest)
    if not found:
        raise ColibriProvisionError(
            f"Unpacked Colibri {RELEASE} into {dest} but could not find the `coli` launcher"
        )
    return found


def maybe_provision_colibri(
    project_root: Path | None,
    *,
    requested_backend: str,
    colibri_family: str | None,
) -> list[str] | None:
    """Unpack coli only for the streamed-MoE / explicit Colibri path."""
    if requested_backend not in {"auto", "colibri"}:
        return None
    if requested_backend == "auto" and not colibri_family:
        return None
    if not colibri_provision_allowed():
        return None
    existing = find_colibri(project_root)
    if existing:
        return existing
    try:
        return provision_colibri(project_root)
    except ColibriProvisionError as exc:
        print(f"Colibri auto-install failed: {exc}", file=sys.stderr, flush=True)
        return None
