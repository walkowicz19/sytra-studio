"""Merge a PEFT adapter into its base model for GGUF conversion."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading base model on CPU in bfloat16...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        cache_dir=args.cache_dir,
    )
    gc.collect()

    print("Loading and merging adapter...", flush=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    gc.collect()

    print(f"Saving merged model to {args.output}...", flush=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    AutoTokenizer.from_pretrained(
        args.adapter,
        cache_dir=args.cache_dir,
    ).save_pretrained(args.output)
    print("Merge complete.", flush=True)


if __name__ == "__main__":
    main()
