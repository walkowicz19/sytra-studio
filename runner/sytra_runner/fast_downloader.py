"""Ultra-fast, zero-token, memory-efficient Hugging Face model and shard downloader.

Bypasses HF_TOKEN requirement for public repositories, streams files directly via
concurrent HTTP range requests, and supports selective index/shard fetching.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from . import telemetry


class FastHFDownloader:
    """Downloader that bypasses HF auth requirements for open models and fetches
    shards/manifests with parallel range requests and zero RAM allocation.
    """

    def __init__(
        self,
        repo_id: str,
        cache_dir: str | Path | None = None,
        tokenless: bool = True,
        max_workers: int = 8,
    ):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir or os.environ.get("HF_HOME", "./.hf-cache"))
        self.tokenless = tokenless
        self.max_workers = max_workers
        self.repo_dir = self.cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}"

        # Ensure HF environment is set to avoid auth prompts / warnings
        if tokenless:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ.pop("HF_TOKEN", None)
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # standard fast fallback

    def _get_hf_url(self, filename: str, revision: str = "main") -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{revision}/{filename}"

    def download_file(
        self,
        filename: str,
        revision: str = "main",
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Download a file using 16 parallel HTTP Range streams for maximum bandwidth."""
        target_dir = self.repo_dir / "snapshots" / revision
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if dest_path.exists() and dest_path.stat().st_size > 0:
            return dest_path

        url = self._get_hf_url(filename, revision)
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
        if not self.tokenless and os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

        try:
            with urllib.request.urlopen(req) as resp:
                total_size = int(resp.headers.get("Content-Length", 0))
        except Exception:
            total_size = 0

        temp_path = dest_path.with_suffix(".tmp")
        
        # If total_size is known and > 50 MB, use 32 parallel range workers
        if total_size > 50 * 1024 * 1024:
            num_workers = 32
            chunk_size = 32 * 1024 * 1024 # 32 MB chunks
            
            # Pre-allocate sparse file
            with open(temp_path, "wb") as f:
                f.truncate(total_size)

            downloaded_bytes = 0
            # Open single shared file handle to avoid Windows file locking issues
            with open(temp_path, "rb+") as out:
                lock = __import__("threading").Lock()

                def download_range(start: int, end: int):
                    nonlocal downloaded_bytes
                    r = urllib.request.Request(url)
                    r.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
                    r.add_header("Range", f"bytes={start}-{end}")
                    if not self.tokenless and os.environ.get("HF_TOKEN"):
                        r.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

                    try:
                        with urllib.request.urlopen(r) as stream:
                            buf_size = 2 * 1024 * 1024
                            offset = start
                            while True:
                                data = stream.read(buf_size)
                                if not data:
                                    break
                                with lock:
                                    out.seek(offset)
                                    out.write(data)
                                    downloaded_bytes += len(data)
                                    if progress_cb:
                                        progress_cb(downloaded_bytes, total_size)
                                offset += len(data)
                    except Exception as e:
                        print(f"\n[Worker Warning] Range {start}-{end} failed: {e}")

                # Build list of byte ranges
                ranges = []
                cur = 0
                while cur < total_size:
                    r_end = min(cur + chunk_size - 1, total_size - 1)
                    ranges.append((cur, r_end))
                    cur += chunk_size

                with ThreadPoolExecutor(max_workers=num_workers) as pool:
                    futures = [pool.submit(download_range, r[0], r[1]) for r in ranges]
                    for fut in as_completed(futures):
                        fut.result()

            temp_path.rename(dest_path)
            return dest_path

        # Sequential fallback for small files
        try:
            r = urllib.request.Request(url)
            r.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
            if not self.tokenless and os.environ.get("HF_TOKEN"):
                r.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            with urllib.request.urlopen(r) as resp, open(temp_path, "wb") as out_file:
                total_size = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 2 * 1024 * 1024
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total_size)

            temp_path.rename(dest_path)
            return dest_path
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Failed to download {filename} from {self.repo_id}: {exc}") from exc

    def fetch_manifest(self, revision: str = "main") -> dict[str, Any]:
        """Fetch model index manifest (model.safetensors.index.json or config.json)."""
        manifest = {}
        try:
            config_path = self.download_file("config.json", revision=revision)
            with open(config_path, "r", encoding="utf-8") as f:
                manifest["config"] = json.load(f)
        except Exception:
            pass

        try:
            index_path = self.download_file("model.safetensors.index.json", revision=revision)
            with open(index_path, "r", encoding="utf-8") as f:
                manifest["index"] = json.load(f)
        except Exception:
            pass

        return manifest

    def download_shards(
        self,
        shards: list[str],
        revision: str = "main",
        op_id: str | None = None,
    ) -> list[Path]:
        """Parallel download of specified safetensor shards without loading into system RAM."""
        downloaded_paths = []
        total_shards = len(shards)

        telemetry.emit_event(
            "fast_download_start",
            {"repo_id": self.repo_id, "shard_count": total_shards, "shards": shards},
        )

        with ThreadPoolExecutor(max_workers=min(self.max_workers, total_shards or 1)) as executor:
            future_to_shard = {
                executor.submit(self.download_file, shard, revision): shard
                for shard in shards
            }
            completed_count = 0

            for future in as_completed(future_to_shard):
                shard_name = future_to_shard[future]
                try:
                    path = future.result()
                    downloaded_paths.append(path)
                    completed_count += 1
                    telemetry.emit_metric(
                        op="fast_download",
                        progress=round(completed_count / total_shards, 4),
                        completed_shards=completed_count,
                        total_shards=total_shards,
                        last_downloaded=shard_name,
                    )
                except Exception as exc:
                    telemetry.emit_error(
                        f"Failed shard download {shard_name}: {exc}",
                        traceback="",
                    )
                    raise

        telemetry.emit_event(
            "fast_download_complete",
            {"repo_id": self.repo_id, "downloaded_count": len(downloaded_paths)},
        )
        return downloaded_paths

    def download_all(
        self,
        dest_dir: str | Path | None = None,
        quant: str = "auto",
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        """Download all files from a model repository directly into target destination directory."""
        target_dir = Path(dest_dir) if dest_dir else self.repo_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        old_repo_dir = self.repo_dir
        self.repo_dir = target_dir

        files_to_download = []
        
        # 1. Try fetching file list via Hugging Face model API
        api_url = f"https://huggingface.co/api/models/{self.repo_id}"
        try:
            req = urllib.request.Request(api_url)
            req.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
            if not self.tokenless and os.environ.get("HF_TOKEN"):
                req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                siblings = data.get("siblings", [])
                files_to_download = [s["rfilename"] for s in siblings if "rfilename" in s]
        except Exception as exc:
            print(f"[NOTE] HF API query notice ({exc}), trying huggingface_hub fallback...")

        # 2. Filter files so user's selected quantization (or auto-selected best GGUF/FP16) is downloaded
        if files_to_download:
            meta_files = [f for f in files_to_download if f in ('.gitattributes', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model', 'special_tokens_map.json', 'README.md', 'LICENSE', 'Modelfile') or f.endswith('.json') or f.endswith('.txt')]
            
            # Check if repo organizes quantizations into subfolders (e.g. UD-Q4_K_M/...)
            folder_groups = {}
            root_files = []
            for f in files_to_download:
                if f in meta_files:
                    continue
                if '/' in f:
                    folder = f.split('/')[0]
                    folder_groups.setdefault(folder, []).append(f)
                else:
                    root_files.append(f)

            selected_model_files = []
            req_q = (quant or "auto").strip().lower()

            if folder_groups:
                chosen_folder = None
                # If explicit quant requested (e.g. "fp16", "bf16", "q8_0", "q5_k_m")
                if req_q != "auto":
                    for folder in folder_groups:
                        if req_q in folder.lower():
                            chosen_folder = folder
                            break
                if not chosen_folder:
                    preferred = ['q4_k_m', 'ud-q4_k_m', 'q4_k_s', 'q4_0', 'q5_k_m', 'q8_0', 'bf16', 'f16', 'fp16']
                    for pref in preferred:
                        for folder in folder_groups:
                            if pref in folder.lower():
                                chosen_folder = folder
                                break
                        if chosen_folder:
                            break
                if not chosen_folder:
                    chosen_folder = list(folder_groups.keys())[0]
                print(f"[FastHFDownloader] Selected quantization folder: {chosen_folder} ({len(folder_groups[chosen_folder])} files)")
                selected_model_files = folder_groups[chosen_folder]
            else:
                # Root files (e.g. qwen2.5-coder-7b-instruct-fp16.gguf vs q4_k_m, q8_0, etc.)
                ggufs = [f for f in root_files if f.endswith('.gguf')]
                if ggufs:
                    variant_groups = {}
                    for f in ggufs:
                        name_upper = f.upper()
                        key = 'OTHER'
                        for q in ['Q4_K_M', 'Q4_K_S', 'Q4_0', 'Q5_K_M', 'Q5_K_S', 'Q6_K', 'Q8_0', 'Q2_K', 'Q3_K', 'FP16', 'BF16', 'F16']:
                            if q in name_upper:
                                key = q
                                break
                        variant_groups.setdefault(key, []).append(f)
                    
                    chosen_variant = None
                    if req_q != "auto":
                        for key in variant_groups:
                            if req_q in key.lower():
                                chosen_variant = key
                                break
                    if not chosen_variant:
                        for pref in ['Q4_K_M', 'Q4_K_S', 'Q4_0', 'Q5_K_M', 'Q8_0', 'FP16', 'BF16']:
                            if pref in variant_groups:
                                chosen_variant = pref
                                break
                    if not chosen_variant:
                        chosen_variant = list(variant_groups.keys())[0]
                    print(f"[FastHFDownloader] Selected quantization variant: {chosen_variant} ({len(variant_groups[chosen_variant])} files)")
                    selected_model_files = variant_groups[chosen_variant]
                else:
                    selected_model_files = root_files

            # Deduplicate preserving order
            seen = set()
            files_to_download = []
            for f in meta_files + selected_model_files:
                if f not in seen:
                    seen.add(f)
                    files_to_download.append(f)

        # 3. Fallback to snapshot_download if API query didn't return siblings
        if not files_to_download:
            try:
                from huggingface_hub import snapshot_download
                print(f"[FastHFDownloader] Downloading {self.repo_id} via snapshot_download to {target_dir}...")
                dl_path = snapshot_download(
                    repo_id=self.repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                )
                self.repo_dir = old_repo_dir
                return [Path(dl_path)]
            except Exception as e:
                print(f"[ERROR] Could not fetch model files: {e}")
                self.repo_dir = old_repo_dir
                raise RuntimeError(f"Failed to fetch file list for {self.repo_id}: {e}") from e

        downloaded_files = []
        total_shards = len(files_to_download)
        total_model_bytes = 0
        start_time = time.time()
        status_file = target_dir / ".download_status.json"

        # Pre-sum total size if HEAD responses return Content-Length
        for idx, filename in enumerate(files_to_download, 1):
            if filename.startswith(".") or filename.startswith("git"):
                continue
            print(f"\n[Sytra Downloader] Fetching: {filename} ({idx}/{total_shards})")
            try:
                target_file = target_dir / filename
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                url = self._get_hf_url(filename)
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
                if not self.tokenless and os.environ.get("HF_TOKEN"):
                    req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

                try:
                    with urllib.request.urlopen(req) as resp:
                        total_size = int(resp.headers.get("Content-Length", 0))
                except Exception:
                    total_size = 0

                if target_file.exists() and total_size > 0 and target_file.stat().st_size == total_size:
                    print(f"[OK] File {filename} already downloaded ({target_file.stat().st_size} bytes). Skipping.")
                    downloaded_files.append(target_file)
                    continue

                def write_status_file(cur_file_bytes: int, file_tot_bytes: int):
                    now = time.time()
                    elapsed = max(now - start_time, 0.1)
                    speed_mbps = (cur_file_bytes / (1024 * 1024)) / elapsed
                    pct = round((cur_file_bytes / max(file_tot_bytes, 1)) * 100, 1) if file_tot_bytes > 0 else 0.0
                    rem_bytes = max(file_tot_bytes - cur_file_bytes, 0)
                    speed_bytes_sec = cur_file_bytes / elapsed
                    eta_sec = int(rem_bytes / speed_bytes_sec) if speed_bytes_sec > 1024 else 0
                    
                    if eta_sec >= 3600:
                        eta_str = f"{eta_sec // 3600}h {(eta_sec % 3600) // 60}m"
                    elif eta_sec >= 60:
                        eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"
                    else:
                        eta_str = f"{eta_sec}s"

                    status_payload = {
                        "repo_id": self.repo_id,
                        "status": "downloading",
                        "downloaded_gb": round(cur_file_bytes / (1024 * 1024 * 1024), 2),
                        "total_gb": round(file_tot_bytes / (1024 * 1024 * 1024), 2),
                        "pct": pct,
                        "speed_mbps": round(speed_mbps, 1),
                        "eta_seconds": eta_sec,
                        "eta_formatted": eta_str,
                        "current_file": filename,
                        "shard_index": idx,
                        "total_shards": total_shards,
                        "timestamp": now,
                    }
                    try:
                        status_file.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                if total_size > 50 * 1024 * 1024:
                    num_workers = min(self.max_workers, 64)
                    chunk_size = 16 * 1024 * 1024
                    temp_path = target_file.with_suffix(".tmp")
                    
                    with open(temp_path, "wb") as f:
                        f.truncate(total_size)

                    downloaded_bytes = 0
                    with open(temp_path, "rb+") as out:
                        lock = __import__("threading").Lock()

                        def download_range(start: int, end: int):
                            nonlocal downloaded_bytes
                            r = urllib.request.Request(url)
                            r.add_header("User-Agent", "SytraStudio-FastDownloader/1.0 (ZeroCPU-Parallel)")
                            r.add_header("Range", f"bytes={start}-{end}")
                            if not self.tokenless and os.environ.get("HF_TOKEN"):
                                r.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")

                            try:
                                with urllib.request.urlopen(r) as stream:
                                    # Tune socket receive buffer to 2MB for maximum TCP window throughput
                                    try:
                                        sock = getattr(stream.fp, "raw", None)
                                        if sock and hasattr(sock, "_sock"):
                                            import socket
                                            sock._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
                                            sock._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                                    except Exception:
                                        pass

                                    buf_size = 4 * 1024 * 1024
                                    offset = start
                                    while True:
                                        data = stream.read(buf_size)
                                        if not data:
                                            break
                                        with lock:
                                            out.seek(offset)
                                            out.write(data)
                                            downloaded_bytes += len(data)
                                            write_status_file(downloaded_bytes, total_size)
                                            if progress_cb:
                                                progress_cb(downloaded_bytes, total_size)
                                        offset += len(data)
                            except Exception as e:
                                print(f"\n[Worker Warning] Range {start}-{end} failed: {e}")

                        ranges = []
                        cur = 0
                        while cur < total_size:
                            r_end = min(cur + chunk_size - 1, total_size - 1)
                            ranges.append((cur, r_end))
                            cur += chunk_size

                        with ThreadPoolExecutor(max_workers=num_workers) as pool:
                            futures = [pool.submit(download_range, r[0], r[1]) for r in ranges]
                            for fut in as_completed(futures):
                                fut.result()

                    temp_path.rename(target_file)
                    downloaded_files.append(target_file)
                else:
                    temp_path = target_file.with_suffix(".tmp")
                    r = urllib.request.Request(url)
                    r.add_header("User-Agent", "SytraStudio-FastDownloader/1.0")
                    if not self.tokenless and os.environ.get("HF_TOKEN"):
                        r.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
                    with urllib.request.urlopen(r) as resp, open(temp_path, "wb") as out_file:
                        block_size = 2 * 1024 * 1024
                        downloaded = 0
                        while True:
                            chunk = resp.read(block_size)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            write_status_file(downloaded, total_size)
                            if progress_cb:
                                progress_cb(downloaded, total_size)
                    temp_path.rename(target_file)
                    downloaded_files.append(target_file)
            except Exception as exc:
                print(f"[ERROR] Failed downloading {filename}: {exc}")

        # Final completed status payload
        try:
            status_file.write_text(json.dumps({
                "repo_id": self.repo_id,
                "status": "completed",
                "downloaded_gb": round(sum(f.stat().st_size for f in downloaded_files if f.exists()) / (1024 * 1024 * 1024), 2),
                "total_gb": round(sum(f.stat().st_size for f in downloaded_files if f.exists()) / (1024 * 1024 * 1024), 2),
                "pct": 100.0,
                "speed_mbps": 0.0,
                "eta_seconds": 0,
                "eta_formatted": "Done",
                "current_file": "Completed",
                "shard_index": total_shards,
                "total_shards": total_shards,
                "timestamp": time.time(),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

        self.repo_dir = old_repo_dir
        return downloaded_files
