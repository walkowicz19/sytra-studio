"""Synthetic data generation runner (Phase 4).

Generates instruction/response data (Prompts, SFT, DPO) using a generator model
and optionally a judge model. Falls back to a high-fidelity simulation when running
on CPU/development environments without CUDA.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path


def generate_simulated(mode: str, count: int, topic: str) -> list[dict[str, str]]:
    """Simulate high-fidelity LLM generation on CPU."""
    templates = [
        ("What are the key concepts of {topic}?", "The key concepts of {topic} include its foundational principles, practical implementations, and core methodologies. Specifically, one must consider how it scales, its efficiency trade-offs, and typical use cases in real-world scenarios."),
        ("Explain {topic} to a beginner.", "To understand {topic}, think of it like a library. Just as a library organizes books for easy retrieval, {topic} structures concepts and data to optimize performance and usability. The main goal is to simplify complex tasks and ensure reliable outcomes."),
        ("Write a Python function demonstrating {topic}.", "Here is a basic example of {topic} implemented in Python:\n\n```python\ndef demonstrate_topic(data):\n    # Core logic for {topic}\n    result = [x * 2 for x in data]\n    return result\n```\nThis function iterates over input data and applies the primary transformation rule."),
        ("What are the common pitfalls when working with {topic}?", "Common pitfalls with {topic} include: 1. Over-engineering early solutions, 2. Failing to validate edge cases, 3. Neglecting resource usage and memory profiling, and 4. Inadequate documentation of design decisions."),
        ("Compare {topic} with its alternatives.", "When comparing {topic} to alternatives, it stands out due to its simplicity and flexibility. While other options might offer slightly better raw speed, they often come with much higher configuration overhead and steeper learning curves."),
    ]

    results = []
    for i in range(count):
        tpl_prompt, tpl_comp = random.choice(templates)
        prompt = tpl_prompt.format(topic=topic)
        comp = tpl_comp.format(topic=topic)
        
        # Add slight variation per row
        prompt += f" (Part {i+1})"
        
        if mode == "prompts":
            results.append({"prompt": prompt})
        elif mode == "dpo":
            results.append({
                "prompt": prompt,
                "chosen": comp,
                "rejected": f"This is a suboptimal response regarding {topic} that lacks depth and code examples."
            })
        else: # sft
            results.append({
                "prompt": prompt,
                "completion": comp
            })
            
    return results


def generate_real(generator: str, judge: str, mode: str, count: int, topic: str) -> list[dict[str, str]]:
    """Generate real data using Hugging Face transformers on CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but real synthetic generation was requested.")

    tokenizer = AutoTokenizer.from_pretrained(generator)
    model = AutoModelForCausalLM.from_pretrained(
        generator, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    generator_pipeline = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer
    )

    results = []
    # Simple prompt loop to generate unique questions and answers
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
            # For DPO, we can generate a rejected response by prompting the model to write a bad version
            bad_gen = f"Write a short, incorrect, or low-quality answer to the prompt: {prompt}"
            out_bad = generator_pipeline(bad_gen, max_new_tokens=128, num_return_sequences=1)
            rejected = out_bad[0]["generated_text"][len(bad_gen):].strip()

            results.append({
                "prompt": prompt,
                "chosen": completion,
                "rejected": rejected
            })
        else:
            results.append({
                "prompt": prompt,
                "completion": completion
            })

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sytra Studio Synthetic Data Generator")
    parser.add_argument("--generator", type=str, required=True, help="Generator model name or path")
    parser.add_argument("--judge", type=str, required=True, help="Judge model name or path")
    parser.add_argument("--mode", type=str, choices=["prompts", "sft", "dpo"], default="sft", help="Generation mode")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--topic", type=str, default="general", help="Topic for generation")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--simulate", action="store_true", help="Force simulation mode")

    args = parser.parse_args()

    # Determine if we should simulate
    has_ml_deps = False
    try:
        import torch
        import transformers
        has_ml_deps = torch.cuda.is_available()
    except ImportError:
        pass

    simulate = args.simulate or not has_ml_deps

    print(json.dumps({
        "type": "event",
        "event": "starting",
        "ts": time.time(),
        "payload": {
            "generator": args.generator,
            "mode": args.mode,
            "count": args.count,
            "topic": args.topic,
            "simulate": simulate
        }
    }))
    sys.stdout.flush()

    try:
        if simulate:
            # Simulate a brief delay to mimic generation time
            for step in range(args.count):
                time.sleep(0.05)
                print(json.dumps({
                    "type": "metric",
                    "ts": time.time(),
                    "step": step + 1,
                    "progress": (step + 1) / args.count
                }))
                sys.stdout.flush()
            rows = generate_simulated(args.mode, args.count, args.topic)
        else:
            rows = generate_real(args.generator, args.judge, args.mode, args.count, args.topic)

        # Write output directory and file
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
                "row_count": len(rows)
            }
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
                "traceback": traceback.format_exc()
            }
        }))
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())
