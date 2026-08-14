"""Synthetic data generation runner.

Generates instruction/response data using a generator model on CUDA.
Missing CUDA or transformers is an error — never a template-fabricated dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def generate_real(generator: str, judge: str, mode: str, count: int, topic: str) -> list[dict[str, str]]:
    """Generate real data using Hugging Face transformers on CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; synthetic generation requires a GPU.")

    tokenizer = AutoTokenizer.from_pretrained(generator)
    model = AutoModelForCausalLM.from_pretrained(
        generator,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    generator_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    results = []
    for i in range(count):
        sys.stderr.write(f"Generating sample {i+1}/{count}...\n")
        sys.stderr.flush()

        prompt_gen = f"Write a diverse question or prompt about the topic: {topic}. Output ONLY the prompt itself and nothing else."
        out = generator_pipeline(prompt_gen, max_new_tokens=64, num_return_sequences=1)
        prompt = out[0]["generated_text"][len(prompt_gen):].strip()

        if mode == "prompts":
            results.append({"prompt": prompt})
            continue

        comp_gen = f"Answer the following prompt comprehensively: {prompt}"
        out_comp = generator_pipeline(comp_gen, max_new_tokens=256, num_return_sequences=1)
        completion = out_comp[0]["generated_text"][len(comp_gen):].strip()

        if mode == "dpo":
            bad_gen = f"Write a short, incorrect, or low-quality answer to the prompt: {prompt}"
            out_bad = generator_pipeline(bad_gen, max_new_tokens=128, num_return_sequences=1)
            rejected = out_bad[0]["generated_text"][len(bad_gen):].strip()
            results.append({
                "prompt": prompt,
                "chosen": completion,
                "rejected": rejected,
            })
        else:
            results.append({
                "prompt": prompt,
                "completion": completion,
            })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sytra Studio Synthetic Data Generator")
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--judge", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["prompts", "sft", "dpo"], default="sft")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--topic", type=str, default="general")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    print(json.dumps({
        "type": "event",
        "event": "starting",
        "ts": time.time(),
        "payload": {
            "generator": args.generator,
            "mode": args.mode,
            "count": args.count,
            "topic": args.topic,
        },
    }))
    sys.stdout.flush()

    try:
        rows = generate_real(args.generator, args.judge, args.mode, args.count, args.topic)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        print(json.dumps({
            "type": "event",
            "event": "done",
            "ts": time.time(),
            "payload": {
                "output_path": str(output_path),
                "row_count": len(rows),
            },
        }))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        print(json.dumps({
            "type": "event",
            "event": "error",
            "ts": time.time(),
            "payload": {
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }))
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
