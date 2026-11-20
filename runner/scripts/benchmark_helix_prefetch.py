"""Compare Helix CUDA prefetch on a small, synthetic Mixtral checkpoint.

Run from the project root with PYTHONPATH=runner. This is a microbenchmark,
not a capacity or throughput prediction for a production model.
"""
import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import torch
from transformers import MixtralConfig, MixtralForCausalLM

from sytra_runner.helix_checkpoint import load_mapped_mixtral
from sytra_runner.helix_mixtral import mixtral_lora_backward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=3, choices=range(1, 21))
    parser.add_argument('--nf4', action='store_true', help='Also convert the synthetic checkpoint to Sytra NF4')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this comparison')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = MixtralConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                           num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                           num_local_experts=4, num_experts_per_tok=2,
                           max_position_embeddings=128, use_cache=False,
                           output_router_logits=False, attention_dropout=0.0,
                           router_jitter_noise=0.0, sliding_window=None)
    with tempfile.TemporaryDirectory(prefix='helix-bench-', dir='.') as directory:
        root = Path(directory)
        MixtralForCausalLM(config).save_pretrained(root / 'model')
        model, pager = load_mapped_mixtral(root / 'model', rank=4, alpha=8,
                                           target_modules=('q_proj', 'w1', 'w2', 'w3'),
                                           quant_cache_dir=root / 'nf4' if args.nf4 else None)
        # Exercise both LoRA matrices; PEFT normally initializes B to zero.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and 'lora_B' in name:
                    parameter.normal_(0, 0.01)
        model.train()
        ids = torch.randint(0, config.vocab_size, (1, 64))
        mask = torch.ones_like(ids)
        labels = ids.clone()
        labels[:, :16] = -100
        reports, expected = {}, None
        for enabled in (False, True):
            times, losses = [], []
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(args.iterations + 1):
                for parameter in model.parameters():
                    parameter.grad = None
                torch.manual_seed(123)
                torch.cuda.manual_seed_all(123)
                torch.cuda.synchronize()
                if iteration == 1:
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = mixtral_lora_backward(model, ids, mask, labels, device='cuda',
                                                weight_pager=pager, prefetch_layers=enabled,
                                                prefetch_pinned_memory_bytes=16 * 1024 * 1024)
                torch.cuda.synchronize()
                duration = time.perf_counter() - start
                if iteration:
                    times.append(duration)
                    losses.append(result.loss)
            gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()
                         if p.requires_grad and p.grad is not None}
            if expected is None:
                expected = (losses[-1], gradients)
            else:
                torch.testing.assert_close(torch.tensor(losses[-1]), torch.tensor(expected[0]), rtol=1e-5, atol=1e-6)
                if gradients.keys() != expected[1].keys():
                    raise AssertionError('Prefetch changed the set of gradients')
                for name, value in gradients.items():
                    torch.testing.assert_close(value, expected[1][name], rtol=1e-4, atol=1e-6, msg=name)
            reports['prefetch_on' if enabled else 'prefetch_off'] = {
                'seconds': times, 'median_seconds': statistics.median(times),
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / (1024 * 1024),
                'loss': losses[-1],
            }
        print(json.dumps({'gpu': torch.cuda.get_device_name(), 'synthetic': True,
                          'quantization': 'sytra_nf4' if args.nf4 else 'none',
                          'nf4_decoder': ('triton_matmul' if os.environ.get('SYTRA_HELIX_TRITON_NF4_MATMUL') == '1'
                                          else 'triton_decode' if os.environ.get('SYTRA_HELIX_TRITON_NF4') == '1'
                                          else 'torch') if args.nf4 else None,
                          'sequence_length': 64, 'warmup_steps': 1, 'iterations': args.iterations,
                          'gradients_match': True, 'gradient_tensors': len(expected[1]),
                          'results': reports}, indent=2))


if __name__ == '__main__':
    main()
