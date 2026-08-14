"""Download a pinned llama.cpp CUDA build into .tools (gitignored)."""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "b10423"
CUDA = "13.3"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}"
SERVER_ZIP = f"llama-{RELEASE}-bin-win-cuda-{CUDA}-x64.zip"
CUDART_ZIP = f"cudart-llama-bin-win-cuda-{CUDA}-x64.zip"


def _download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    dest = root / ".tools" / "llama.cpp" / "build" / "bin" / "Release"
    dest.mkdir(parents=True, exist_ok=True)
    server = dest / "llama-server.exe"
    with tempfile.TemporaryDirectory(prefix="sytra-llama-") as tmp:
        tmp_path = Path(tmp)
        if not server.is_file():
            zip_path = tmp_path / SERVER_ZIP
            _download_file(f"{BASE}/{SERVER_ZIP}", zip_path)
            _extract(zip_path, dest)
        has_cudart = any(dest.glob("cudart64*.dll"))
        if not has_cudart:
            zip_path = tmp_path / CUDART_ZIP
            _download_file(f"{BASE}/{CUDART_ZIP}", zip_path)
            _extract(zip_path, dest)
    if not server.is_file():
        print("llama-server.exe missing after extract", file=sys.stderr)
        return 1
    print(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
