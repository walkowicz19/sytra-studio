"""Verified Hugging Face/Xet model downloader for Sytra Studio."""
import os
import sys
import argparse
import traceback
from pathlib import Path

# Add runner/ directory to sys.path so sytra_runner can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.fast_downloader import FastHFDownloader

def download_model_fast(
    repo_id: str,
    dest_dir: str = None,
    workers: int = 4,
    quant: str = "auto",
    purpose: str = "inference",
    revision: str = "main",
    tokenless: bool = False,
):
    if not repo_id:
        print("[ERROR] Please provide a HuggingFace repository ID using --model (e.g. --model owner/repo-name)")
        sys.exit(1)

    print("=== Sytra 1.2.0: Verified Hugging Face/Xet Download ===")
    print(f"Repository: {repo_id}")
    print(f"Quantization Mode: {quant}")
    print(f"Purpose: {purpose}")
    print(f"Requested Revision: {revision}")

    if not dest_dir or not dest_dir.strip():
        dest_dir = str((Path.home() / "lm-studio models").resolve())

    storage_dir = Path(dest_dir)
    target_dir = storage_dir / repo_id.replace("/", "--")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Storage Path: {storage_dir.resolve()}")
    print(f"Model Path: {target_dir.resolve()}")

    downloader = FastHFDownloader(repo_id, tokenless=tokenless, max_workers=workers)

    def progress(current, total):
        if total > 0:
            pct = (current / total) * 100
            gb_cur = current / (1024 * 1024 * 1024)
            gb_tot = total / (1024 * 1024 * 1024)
            print(f"\rVerified download: {gb_cur:.2f} GB / {gb_tot:.2f} GB ({pct:.1f}%)", end="", flush=True)

    print(f"\nDownloading a commit-pinned snapshot to {target_dir.resolve()} via hf-xet...")
    try:
        files = downloader.download_all(
            dest_dir=str(target_dir),
            quant=quant,
            progress_cb=progress,
            purpose=purpose,
            revision=revision,
            status_dir=storage_dir,
        )
        print("\n\n[OK] Model download completed and verified!")
        print(f"[OK] Saved {len(files)} files to: {target_dir.resolve()}")
        print(f"[OK] Resolved commit: {downloader.resolved_revision}")
    except Exception as exc:
        print(f"\n[ERROR] Downloader error: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== SUCCESS: Download completed for " + repo_id + " ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sytra verified Hugging Face/Xet model downloader")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model repository ID (e.g. org/repo)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory on any drive")
    parser.add_argument("--workers", type=int, default=4, help="Number of files downloaded concurrently (hf-xet parallelizes chunks)")
    parser.add_argument("--quant", type=str, default="auto", help="Quantization target (auto, Q4_K_M, Q5_K_M, Q8_0, FP16, BF16)")
    parser.add_argument("--purpose", choices=("inference", "finetune", "merge"), default="inference")
    parser.add_argument("--revision", default="main", help="Branch, tag, or commit to resolve and pin")
    parser.add_argument("--tokenless", action="store_true", help="Do not use cached Hub credentials unless HF_TOKEN is explicitly set")
    args = parser.parse_args()

    download_model_fast(
        repo_id=args.model,
        dest_dir=args.dest,
        workers=args.workers,
        quant=args.quant,
        purpose=args.purpose,
        revision=args.revision,
        tokenless=args.tokenless,
    )
    sys.exit(0)
