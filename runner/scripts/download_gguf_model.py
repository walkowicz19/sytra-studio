"""Sytra 1.2.0 General-Purpose Parallel Model Downloader for LM Studio, Ollama & Sytra Studio."""
import os
import sys
import argparse
import traceback
from pathlib import Path

# Add runner/ directory to sys.path so sytra_runner can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sytra_runner.fast_downloader import FastHFDownloader

def download_model_fast(repo_id: str, dest_dir: str = None, workers: int = 64, quant: str = "auto"):
    if not repo_id:
        print("[ERROR] Please provide a HuggingFace repository ID using --model (e.g. --model owner/repo-name)")
        sys.exit(1)

    print(f"=== Sytra 1.2.0: Fast {workers}-Worker Download ===")
    print(f"Repository: {repo_id}")
    print(f"Quantization Mode: {quant}")

    if not dest_dir or not dest_dir.strip():
        dest_dir = str((Path.home() / "lm-studio models").resolve())
    
    target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Destination Path: {target_dir.resolve()}")

    downloader = FastHFDownloader(repo_id, tokenless=True, max_workers=workers)

    def progress(current, total):
        if total > 0:
            pct = (current / total) * 100
            gb_cur = current / (1024 * 1024 * 1024)
            gb_tot = total / (1024 * 1024 * 1024)
            print(f"\rDownloading model ({workers} Workers): {gb_cur:.2f} GB / {gb_tot:.2f} GB ({pct:.1f}%)", end="", flush=True)

    print(f"\nStreaming repository files to {target_dir.resolve()} via {workers} parallel streams...")
    try:
        files = downloader.download_all(dest_dir=str(target_dir), quant=quant, progress_cb=progress)
        print(f"\n\n[OK] Model download completed successfully!")
        print(f"[OK] Saved {len(files)} files to: {target_dir.resolve()}")
    except Exception as exc:
        print(f"\n[ERROR] Downloader error: {exc}")
        traceback.print_exc()
        sys.exit(1)

    print("\n=== SUCCESS: Download completed for " + repo_id + " ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sytra 1.2.0 General Model Downloader")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model repository ID (e.g. org/repo)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory on any drive")
    parser.add_argument("--workers", type=int, default=64, help="Number of parallel HTTP streams")
    parser.add_argument("--quant", type=str, default="auto", help="Quantization target (auto, Q4_K_M, Q5_K_M, Q8_0, FP16, BF16)")
    args = parser.parse_args()

    download_model_fast(repo_id=args.model, dest_dir=args.dest, workers=args.workers, quant=args.quant)
    sys.exit(0)
