"""Verified, resumable Hugging Face model downloads.

The Hub's Xet transport owns chunking, retries, resumption, and content
validation. Sytra owns model-file selection, commit pinning, aggregate status,
and a local manifest describing exactly what was downloaded.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import telemetry
from .xet_safety import apply_xet_safety

apply_xet_safety()


_METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "NOTICE",
    "Modelfile",
}
_MODEL_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_QUANT_ORDER = (
    "Q4_K_M",
    "UD-Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
    "BF16",
    "FP16",
    "F16",
)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None = None
    blob_id: str | None = None


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Model downloads require huggingface-hub and hf-xet. "
            "Install them with: uv pip install 'huggingface-hub>=0.34' 'hf-xet>=1.1.5'"
        ) from exc
    return HfApi, hf_hub_download


def _format_eta(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds}s"


class _DownloadProgress:
    """Poll local files while hf-xet downloads in worker threads."""

    def __init__(
        self,
        repo_id: str,
        target_dir: Path,
        files: list[RemoteFile],
        status_file: Path,
        progress_cb: Callable[[int, int], None] | None,
    ):
        self.repo_id = repo_id
        self.target_dir = target_dir
        self.files = files
        self.status_file = status_file
        self.progress_cb = progress_cb
        self.total = sum(item.size or 0 for item in files)
        self.completed: set[str] = set()
        self.active: set[str] = set()
        self.failed: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sytra-download-progress")
        self._start = time.monotonic()
        self._last_bytes = 0
        self._last_time = self._start
        self._speed = 0.0

    def start(self) -> None:
        self._write("resolving", 0)
        self._thread.start()

    def mark_active(self, name: str) -> None:
        with self._lock:
            self.active.add(name)

    def mark_completed(self, name: str) -> None:
        with self._lock:
            self.active.discard(name)
            self.completed.add(name)

    def mark_failed(self, message: str) -> None:
        with self._lock:
            self.failed = message

    def stop(self, status: str) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        downloaded = self._downloaded_bytes(include_incomplete=False)
        if status == "completed" and self.total:
            downloaded = self.total
        self._write(status, downloaded)

    def _downloaded_bytes(self, *, include_incomplete: bool = True) -> int:
        completed_bytes = 0
        with self._lock:
            completed = set(self.completed)
            active = set(self.active)
        sizes = {item.name: item.size for item in self.files}
        for name in completed:
            expected = sizes.get(name)
            path = self.target_dir / name
            completed_bytes += expected if expected is not None else (path.stat().st_size if path.exists() else 0)

        if include_incomplete and active:
            for name in active:
                path = self.target_dir / name
                if path.exists():
                    try:
                        completed_bytes += path.stat().st_size
                    except OSError:
                        pass
        return min(completed_bytes, self.total) if self.total else completed_bytes

    def _write(self, status: str, downloaded: int) -> None:
        now = time.monotonic()
        dt = max(now - self._last_time, 0.001)
        instant_speed = max(downloaded - self._last_bytes, 0) / dt
        if instant_speed:
            self._speed = instant_speed if not self._speed else self._speed * 0.7 + instant_speed * 0.3
        self._last_bytes = downloaded
        self._last_time = now

        with self._lock:
            active = sorted(self.active)
            failed = self.failed
            completed_count = len(self.completed)

        remaining = max(self.total - downloaded, 0)
        eta = int(remaining / self._speed) if self._speed > 1024 else 0
        payload = {
            "repo_id": self.repo_id,
            "status": status,
            "downloaded_gb": round(downloaded / (1024**3), 2),
            "total_gb": round(self.total / (1024**3), 2),
            "pct": 100.0 if status == "completed" else (round(downloaded / self.total * 100, 1) if self.total else 0.0),
            "speed_mbps": round(self._speed / (1024**2), 1),
            "eta_seconds": eta,
            "eta_formatted": "Done" if status == "completed" else ("Failed" if status == "error" else _format_eta(eta)),
            "current_file": (
                "Completed"
                if status == "completed"
                else "Failed"
                if status == "error"
                else active[0]
                if len(active) == 1
                else f"{len(active)} files active"
            ),
            "shard_index": completed_count,
            "total_shards": len(self.files),
            "timestamp": time.time(),
            "error": failed,
        }
        try:
            temp = self.status_file.with_suffix(".json.tmp")
            temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temp, self.status_file)
        except OSError:
            pass
        if self.progress_cb:
            self.progress_cb(downloaded, self.total)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            self._write("downloading", self._downloaded_bytes())


class FastHFDownloader:
    """Download a commit-pinned, format-consistent model snapshot via hf-xet."""

    def __init__(
        self,
        repo_id: str,
        cache_dir: str | Path | None = None,
        tokenless: bool = True,
        max_workers: int = 4,
    ):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir or os.environ.get("HF_HOME", "./.hf-cache"))
        self.tokenless = tokenless
        # hf-xet already parallelizes chunks within a file. Extra file
        # workers each reserve reconstruction buffers, which pages every OS.
        self.max_workers = 1
        self.repo_dir = self.cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}"
        self._resolved_revision: str | None = None

        os.environ["HF_HOME"] = str(self.cache_dir.resolve())
        os.environ["HF_XET_CACHE"] = str((self.cache_dir / "xet").resolve())
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        apply_xet_safety()

    @property
    def token(self) -> str | bool | None:
        explicit = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if explicit:
            return explicit
        return False if self.tokenless else None

    @property
    def resolved_revision(self) -> str | None:
        return self._resolved_revision

    @staticmethod
    def _is_metadata(name: str) -> bool:
        base = Path(name).name
        return (
            base in _METADATA_NAMES
            or ("/" not in name and name.endswith((".json", ".txt", ".model", ".tiktoken", ".jinja", ".py")))
        )

    @staticmethod
    def _quant_key(name: str) -> str:
        upper = name.upper()
        for quant in _QUANT_ORDER:
            if quant in upper:
                return quant
        return "OTHER"

    @classmethod
    def select_files(
        cls,
        filenames: Iterable[str],
        *,
        purpose: str = "inference",
        quant: str = "auto",
    ) -> list[str]:
        """Select one complete weight format without changing model semantics."""
        names = sorted({name for name in filenames if name and not name.startswith((".", "git"))})
        metadata = [name for name in names if cls._is_metadata(name)]
        ggufs = [name for name in names if name.lower().endswith(".gguf")]
        auxiliary_ggufs = [
            name
            for name in ggufs
            if "mmproj" in Path(name).name.lower() or "projector" in Path(name).name.lower()
        ]
        model_ggufs = [name for name in ggufs if name not in auxiliary_ggufs]
        tensor_weights = [name for name in names if name.lower().endswith(_MODEL_SUFFIXES)]

        selected_weights: list[str]
        if purpose == "inference" and model_ggufs:
            groups: dict[str, list[str]] = {}
            for name in model_ggufs:
                groups.setdefault(cls._quant_key(name), []).append(name)
            requested = (quant or "auto").strip().upper()
            chosen = None
            if requested != "AUTO":
                chosen = next((key for key in groups if requested in key), None)
                if chosen is None:
                    available = ", ".join(groups)
                    raise ValueError(f"Quantization {quant!r} is unavailable; found: {available}")
            else:
                chosen = next((key for key in _QUANT_ORDER if key in groups), None)
                chosen = chosen or next(iter(groups))
            selected_weights = groups[chosen] + auxiliary_ggufs
        elif tensor_weights:
            # A quality-preserving snapshot needs every tensor shard. Never
            # guess that router top-k means only expert IDs 0..k are required.
            selected_weights = tensor_weights
        else:
            expected = "GGUF or Safetensors/PyTorch weights"
            raise ValueError(f"Repository contains no {expected} for purpose={purpose!r}")

        selected_dirs = {str(Path(name).parent) for name in selected_weights}
        selected_metadata = [
            name
            for name in metadata
            if str(Path(name).parent) in selected_dirs or str(Path(name).parent) == "."
        ]
        return list(dict.fromkeys(selected_metadata + selected_weights))

    def _model_info(self, revision: str):
        HfApi, _ = _require_huggingface_hub()
        return HfApi().model_info(
            self.repo_id,
            revision=revision,
            files_metadata=True,
            token=self.token,
        )

    @staticmethod
    def _remote_files(info: Any) -> list[RemoteFile]:
        result = []
        for sibling in info.siblings or []:
            size = getattr(sibling, "size", None)
            lfs = getattr(sibling, "lfs", None)
            if size is None and lfs:
                size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
            result.append(
                RemoteFile(
                    name=sibling.rfilename,
                    size=size,
                    blob_id=getattr(sibling, "blob_id", None),
                )
            )
        return result

    def download_file(
        self,
        filename: str,
        revision: str = "main",
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download one file atomically through huggingface_hub/hf-xet."""
        _, hf_hub_download = _require_huggingface_hub()
        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            revision=revision,
            local_dir=str(self.repo_dir),
            token=self.token,
        )
        result = Path(path)
        if not result.is_file() or result.stat().st_size <= 0:
            raise RuntimeError(f"Hub returned an invalid local file for {filename}")
        if progress_cb:
            progress_cb(result.stat().st_size, result.stat().st_size)
        return result

    def fetch_manifest(self, revision: str = "main") -> dict[str, Any]:
        """Fetch config plus the optional Safetensors index."""
        config_path = self.download_file("config.json", revision=revision)
        with config_path.open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = {"config": json.load(handle)}
        try:
            index_path = self.download_file("model.safetensors.index.json", revision=revision)
        except Exception:
            return manifest
        with index_path.open("r", encoding="utf-8") as handle:
            manifest["index"] = json.load(handle)
        return manifest

    def download_shards(
        self,
        shards: list[str],
        revision: str = "main",
        op_id: str | None = None,
    ) -> list[Path]:
        """Download a caller-supplied complete shard set and fail on any error."""
        telemetry.emit_event(
            "fast_download_start",
            {"repo_id": self.repo_id, "shard_count": len(shards)},
            op_id=op_id,
        )
        downloaded: list[Path] = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(shards) or 1)) as pool:
            futures = {pool.submit(self.download_file, shard, revision): shard for shard in shards}
            for future in as_completed(futures):
                shard = futures[future]
                try:
                    downloaded.append(future.result())
                except Exception as exc:
                    telemetry.emit_error(
                        f"Failed shard download {shard}: {exc}",
                        op="verified_download",
                        op_id=op_id,
                    )
                    raise RuntimeError(f"Failed shard download {shard}: {exc}") from exc
        telemetry.emit_event(
            "fast_download_complete",
            {"repo_id": self.repo_id, "downloaded_count": len(downloaded)},
            op_id=op_id,
        )
        return downloaded

    def download_all(
        self,
        dest_dir: str | Path | None = None,
        quant: str = "auto",
        progress_cb: Callable[[int, int], None] | None = None,
        *,
        purpose: str = "inference",
        revision: str = "main",
        status_dir: str | Path | None = None,
    ) -> list[Path]:
        """Resolve, pin, download, and verify a model snapshot."""
        target_dir = Path(dest_dir) if dest_dir else self.repo_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        status_root = Path(status_dir) if status_dir else target_dir
        status_root.mkdir(parents=True, exist_ok=True)
        status_file = status_root / ".download_status.json"

        try:
            info = self._model_info(revision)
            resolved_revision = info.sha
            if not resolved_revision:
                raise RuntimeError(f"Hugging Face did not resolve revision {revision!r} to a commit")
            self._resolved_revision = resolved_revision
            remote = self._remote_files(info)
            selected_names = self.select_files(
                (item.name for item in remote),
                purpose=purpose,
                quant=quant,
            )
            by_name = {item.name: item for item in remote}
            selected = [by_name[name] for name in selected_names]
        except Exception as exc:
            payload = {
                "repo_id": self.repo_id,
                "status": "error",
                "downloaded_gb": 0.0,
                "total_gb": 0.0,
                "pct": 0.0,
                "speed_mbps": 0.0,
                "eta_seconds": 0,
                "eta_formatted": "Failed",
                "current_file": "Preflight failed",
                "shard_index": 0,
                "total_shards": 0,
                "timestamp": time.time(),
                "error": str(exc),
            }
            status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            raise RuntimeError(f"Model download preflight failed for {self.repo_id}: {exc}") from exc

        old_repo_dir = self.repo_dir
        self.repo_dir = target_dir
        monitor = _DownloadProgress(self.repo_id, target_dir, selected, status_file, progress_cb)
        monitor.start()
        downloaded: list[Path] = []
        errors: list[str] = []

        def fetch(item: RemoteFile) -> Path:
            monitor.mark_active(item.name)
            try:
                path = self.download_file(item.name, revision=resolved_revision)
                if item.size is not None and path.stat().st_size != item.size:
                    raise RuntimeError(
                        f"size mismatch for {item.name}: expected {item.size}, got {path.stat().st_size}"
                    )
                monitor.mark_completed(item.name)
                return path
            except Exception as exc:
                message = f"{item.name}: {exc}"
                monitor.mark_failed(message)
                raise RuntimeError(message) from exc

        try:
            # Xet handles chunk parallelism; a small number of file workers is
            # enough to keep sharded checkpoints busy without connection storms.
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(selected) or 1)) as pool:
                futures = {pool.submit(fetch, item): item for item in selected}
                for future in as_completed(futures):
                    try:
                        downloaded.append(future.result())
                    except Exception as exc:
                        errors.append(str(exc))
            if errors:
                raise RuntimeError("; ".join(errors))

            manifest = {
                "schema_version": 1,
                "repo_id": self.repo_id,
                "requested_revision": revision,
                "resolved_revision": resolved_revision,
                "purpose": purpose,
                "quantization": quant,
                "transport": "huggingface_hub+hf_xet",
                "files": [
                    {
                        "path": item.name,
                        "size": item.size,
                        "blob_id": item.blob_id,
                    }
                    for item in selected
                ],
                "total_bytes": sum(item.size or 0 for item in selected),
                "completed_at": time.time(),
            }
            (target_dir / ".sytra-model.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            monitor.stop("completed")
            return sorted(downloaded)
        except Exception as exc:
            monitor.mark_failed(str(exc))
            monitor.stop("error")
            raise RuntimeError(f"Verified download failed for {self.repo_id}: {exc}") from exc
        finally:
            self.repo_dir = old_repo_dir
