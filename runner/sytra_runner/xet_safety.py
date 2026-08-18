"""Overwrite Hugging Face Xet defaults so Hub I/O cannot page the OS.

hf-xet's stock reconstruction buffers are 2 GiB / 8 GiB / 1 GiB prefetch
with adaptive download concurrency up to 64 streams. Those reservations
are allocated even for small files, on Windows, macOS, and Linux. Call
``apply_xet_safety`` before any huggingface_hub import or download, and
``child_process_env`` for every mergekit / llama.cpp / vLLM grandchild.
"""
from __future__ import annotations

import os
from typing import Mapping

XET_SAFETY_ENV: dict[str, str] = {
    "HF_XET_HIGH_PERFORMANCE": "0",
    "HF_XET_HP": "0",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY": "false",
    "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "2",
    "HF_XET_FIXED_UPLOAD_CONCURRENCY": "1",
    "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "1",
    "HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION": "1",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": "64mb",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": "32mb",
    "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "128mb",
    "HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE": "16mb",
    "HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE": "128mb",
    "HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER": "16mb",
    "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
}

THREAD_CAP_ENV: dict[str, str] = {
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


def apply_xet_safety() -> None:
    """Force low-RAM Xet settings. Overwrites, never setdefault."""
    os.environ.update(XET_SAFETY_ENV)


def apply_runtime_safety() -> None:
    """Xet caps plus scheduler niceness so merge/train/download cannot starve the desktop."""
    apply_xet_safety()
    for key, value in THREAD_CAP_ENV.items():
        os.environ.setdefault(key, value)
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00004000,  # BELOW_NORMAL_PRIORITY_CLASS
            )
        except Exception:
            pass


def child_process_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for mergekit / engine grandchildren. Overwrites Xet defaults."""
    apply_xet_safety()
    env = dict(os.environ)
    env.update(XET_SAFETY_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for key, value in THREAD_CAP_ENV.items():
        env.setdefault(key, value)
    if extra:
        env.update(extra)
    return env
