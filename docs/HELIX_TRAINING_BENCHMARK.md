# Helix training microbenchmark

## LoRA training versus a CUDA-resident baseline

On an RTX 3060 (12 GiB), `benchmark_helix_training.py` compared Helix layer streaming
with Transformers + PEFT keeping the base model on CUDA. Both performed LoRA
updates on the same randomly initialized 28,848,640-parameter Mixtral checkpoint
and 64 synthetic arithmetic examples. Both used FP32, an attention-only rank-8
adapter (114,688 trainable parameters), AdamW, and batches of four pre-tokenized
examples with answer-only loss. Quantization and packing were disabled.

The trial order was Helix, Transformers, Transformers, Helix. Each trial ran
four warmup and 16 measured steps. The table pools step times and token counts
across the two trials for each method.

| Method | Median step | Supervised tokens/s | Peak PyTorch CUDA allocation |
| --- | ---: | ---: | ---: |
| Helix | 198.4 ms | 272 | 50.3 MiB |
| Transformers + PEFT | 56.6 ms | 968 | 150.9 MiB |

The baseline processed about 3.6× as many supervised tokens per second, while
Helix used about 67% less PyTorch-allocated GPU memory. Both ended at a training
loss of approximately 4.86015. Timings cover synchronized optimizer updates with
prepared batches and exclude loading, tokenization and artifact export. Peak
allocation includes the loaded model where resident but excludes CUDA context
memory. This is a small, single-GPU throughput test with PyTorch 2.10.0+cu128 and
Transformers 5.2.0. Evaluation quality and throughput on pretrained models or with
the Unsloth backend were outside this comparison.

```powershell
$env:PYTHONPATH = 'runner'
python runner/scripts/benchmark_helix_training.py --output runs/helix-training-benchmark.json
```

## Scaling to a 1.02B-parameter Mixtral

On the same RTX 3060, both paths trained with the same 64 examples, FP32
weights, and rank-8 LoRA configuration on a randomly initialized Mixtral with
1,020,937,728 parameters. The base weights alone occupy 3.80 GiB. The benchmark
used two warmup steps and eight measured steps per trial, with two trials per
method in the same alternating order as the smaller run.

| Method | Median step | Supervised tokens/s | Peak PyTorch CUDA allocation |
| --- | ---: | ---: | ---: |
| Helix | 1.965 s | 26.4 | 283.0 MiB |
| Transformers + PEFT | 0.225 s | 244.8 | 4,187.5 MiB |

With enough VRAM to keep the full model resident, Transformers processed
9.3× as many supervised tokens per second. Helix used 93% less allocated GPU
memory. Both ended at a training loss of approximately 4.00487. The first
attempt to load the larger fused expert tensors in Helix raised a native
access violation on Windows. The timed Helix run uses the same weight values
stored as separate expert tensors. Checkpoint preparation remains outside
the timed steps.

```powershell
$env:PYTHONPATH = 'runner'
python runner/scripts/benchmark_helix_training.py --size 1b --warmup 2 --steps 8 --output runs/helix-training-benchmark-1b.json
```

A separate run limited the PyTorch CUDA allocator to **1 GiB** on the same
RTX 3060. Helix completed two warmup and four measured steps, with a 1.732 s
median step and 282.6 MiB peak allocation. Transformers + PEFT raised a CUDA
out-of-memory error while moving the model to CUDA, before its first update.
This run tests full-resident FP32 under an imposed memory budget.

```powershell
$env:PYTHONPATH = 'runner'
python runner/scripts/benchmark_helix_training.py --size 1b --vram-budget-mib 1024 --warmup 2 --steps 4 --output runs/helix-training-benchmark-1b-cap.json
```

## Layer prefetch

`runner/scripts/benchmark_helix_prefetch.py` compares a Helix backward pass with
prefetch disabled and enabled. It generates a small local Mixtral checkpoint,
uses mapped frozen weights and LoRA on attention and expert projections, and
checks the loss and every populated adapter gradient for agreement. It does not
download models or update model weights. Add `--nf4` to convert the checkpoint
to Sytra's NF4 cache before measuring.

From the project root, using an environment with the runner dependencies and
CUDA-enabled PyTorch:

```powershell
$env:PYTHONPATH = 'runner'
python runner/scripts/benchmark_helix_prefetch.py
python runner/scripts/benchmark_helix_prefetch.py --nf4
```

Each mode has one warmup and three measured steps by default. The JSON output
contains each wall-clock time, median, peak PyTorch-allocated GPU memory, loss,
and the number of gradient tensors compared. Timing includes CPU work and CUDA
synchronization. Model creation and NF4 conversion are outside the measurement.

An exploratory run on an RTX 3060 with 12 GiB on 2026-11-08 used a three-layer,
128-hidden-size model with four experts and a 64-token sequence. All 78 populated
adapter gradients matched between modes. Median times were 0.206 s without and
0.167 s with prefetch for unquantized weights; the NF4 run measured 0.819 s and
0.807 s respectively. Peak allocated GPU memory was approximately 19.1 MiB for
the unquantized run and 17.9 MiB for NF4. A prior short unquantized run showed
the opposite prefetch result, so these numbers do not establish a speedup.

This synthetic comparison measures short, fixed-order runs on the small model.
Its scope is optimizer-step throughput and peak CUDA allocation on the tested
GPU.

## Experimental fused NF4 decoding

With a compatible local Triton installation, set
`$env:SYTRA_HELIX_TRITON_NF4 = '1'` before launching the training process to
use the experimental CUDA decoder. Unset the variable to use the default
PyTorch decoder. An enabled but unavailable or incompatible Triton installation
causes an error; it is not silently substituted. CPU decoding is unchanged.
Triton may compile kernels and populate its configured cache on first use.

This kernel fuses the unpacking, lookup, and scaling of NF4 values. It does not
fuse the following matrix multiplication or the LoRA forward and backward.
On the same RTX 3060 and synthetic configuration described above, a separate
five-step comparison with prefetch disabled measured median steps of 0.734 s
with the PyTorch decoder and 0.404 s with the Triton decoder. Both produced the
same displayed loss; decoder outputs and gradients were also compared in tests.
The runs were sequential, and these numbers are not evidence of performance on
other GPUs, real large models, or a fused LoRA training kernel.

An additional experimental path fuses NF4 decoding with the frozen base matrix
multiplication in both forward and input-gradient computation. Enable it with
`$env:SYTRA_HELIX_TRITON_NF4_MATMUL = '1'`; it takes precedence over the
decoder-only switch. The adapter's LoRA operations remain separate PyTorch
operations. This path also requires a working CUDA-compatible Triton install
and is disabled by default.

In a separate five-step run of the same small synthetic benchmark with prefetch
disabled, the fused matrix multiplication measured a median of 0.239 s per
step and 17.7 MiB peak PyTorch-allocated GPU memory. Runs were sequential and
measure only one hardware and software configuration. FP32, FP16 and BF16
forward and input gradients were checked against a dequantized reference,
including non-contiguous inputs and sizes that are not multiples of tile sizes.

## Packed attention mask

Helix's optional packed path constructs FlexAttention metadata directly in
128-token blocks, avoiding construction of a dense token-by-token mask.
The current FlexAttention block index tensors still scale quadratically with
the number of blocks; this is not a constant-memory or linear-memory guarantee.
Tiles touching multiple conversations can be partially evaluated and masked at
token granularity. Correctness tests cover boundaries crossing blocks, causal
and sliding-window masks, compiled CUDA gradients, and checkpoint resume.

Helix also prefetches at most one pre-tokenized batch on a background thread
while the current batch runs. This keeps dataset order and avoids duplicating a
CUDA process, but does not prove the GPU will never wait for data. The host
memory estimate includes the current and next collated input, mask and label
tensors. Tokenization and dataset preparation still occur before training.

## Sytra 1.3.0 Update (Observed on RTX 3060 12 GB / 16 GiB RAM Host)

- **Multi-architecture & pre-quantized checkpoints**: Helix 1.3.0 extends layer-streamed SFT (`helix_checkpoint.py`, `helix_quant.py`, `helix_mixtral.py`) beyond unquantized Mixtral to `qwen2_moe`, `qwen3_moe`, `deepseek_v2`, and `deepseek_v3` (including `shared_expert`/`shared_experts`, MLA projections, and transformers 5.x fused 3D MoE experts via safetensors slicing), plus `bitsandbytes` 4-bit (`nf4`/`fp4` single and double quant, high-nibble-first compatible) and 8-bit (`SCB`) checkpoints and `FrozenInt8Linear` local 8-bit quantization.
- **Triton fused NF4+LoRA & chunked cross-entropy (`helix_triton_ops.py`)**:
  - `fused_nf4_lora_linear` CUDA parity vs PyTorch reference (`low_first=True/False`): `fp32` `dy=1.91e-6`, `dx=da=db=0.0`; `fp16` `dy<=1.83e-3`; `bf16` `dy<=1.34e-2` (with finite grads at `dropout_p=0.25`).
  - `chunked_cross_entropy_sum` (`V=151936` on RTX 3060 12 GB CUDA): extra peak VRAM **148.38 MB** vs **890.24 MB** for `torch_fp32_cast`.
- **Packing, memmap & 8-bit paged optimizer**:
  - `helix_packing.py` replaces quadratic FlexAttention block indices with `O(blocks * W)` rectangular sliding window indexing and provides CPU/eval `packed_causal_4d_mask`.
  - `helix_dataset_mmap.py` persists pre-tokenized conversations and assistant masks via memory-mapped arrays (`MmapTokenDataset`).
  - `backends/helix_sft.py` uses `HelixPagedAdamW8bit` (`bitsandbytes.optim.PagedAdamW8bit`) when CUDA is available (staging CPU-resident LoRA tensors through CUDA with verified `16.32 MB` test-fixture peak VRAM and `atol=1e-7` step-2 resume parity in `test_runner_training_and_checkpoint_resume[attention-False-cuda]`) and falls back safely to `torch.optim.AdamW` on CPU.
- **Hardware limitation notice**: All CUDA runs on this machine used a 12 GB RTX 3060; 8 GB VRAM GPUs were not physically available for measurement, and no `3x–8x` speedup or zero-forgetting claims are asserted.
