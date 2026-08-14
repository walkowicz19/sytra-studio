# Correctness-gated benchmark records

These records are evidence for one exact checkpoint, machine, command, and build. They are not throughput projections for larger models.

## 2026-08-14 — Qwen2.5-0.5B Instruct Q4_K_M on RTX 3060 12 GB (llama.cpp b10423)

- Artifact: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` file `qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes)
- Architecture from GGUF header: `qwen2`, 24 layers, `MOSTLY_Q4_K_M` (not inferred from the filename)
- Runtime: llama.cpp `llama-server` **b10423** / commit `a94d563ed`, Windows CUDA 13.3, RTX 3060 12 GB, Sytra RAM cap 12 GB
- Command planner: GPU-first `-ngl 24`, mmap on, mlock off, flash attention on, KV `q8_0`
- Records (not simulated): `runs/benchmarks/bench-20260814T013627Z-tiny-qwen-gpu.json`, `...-tiny-qwen-cpu.json`, `...-tiny-qwen-gpu-sustained.json`

| Label | n-gpu-layers | gen tok/s | prompt tok/s | TTFT s | nvidia-smi VRAM used MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU | 24 | 145.75 | 98.42 | 0.542 | 1086 |
| CPU explicit | 0 | 42.14 | 91.21 | 0.792 | 718 (desktop, not llama) |
| GPU sustained 128 tok | 24 | 138.75 | 162.18 | 0.949 | 1196 |

GPU generation is about **3.5×** the explicit CPU-only baseline. Peak observed VRAM stayed far below 12 GB. Output began with “Paris.” Repeat start/stop on ports 8090/8091/8092 succeeded.

Ollama 0.32.9 `create` from the generated Modelfile succeeded; `/api/generate` returned `Paris.` with `done_reason: stop`. LM Studio CLI `lms import -y -c` copied the same GGUF into `~/.lmstudio/models/sytra/...`.

A real MoE GGUF larger than VRAM (OLMoE-1B-7B Q4_K_M, 4.21 GB) was downloading at session end; do not treat the dense 0.5B numbers as MoE expert-paging throughput.

## 2026-08-02 — tiny random Mixtral on RTX 3060 12 GB

- Checkpoint: `hf-internal-testing/tiny-random-MixtralForCausalLM`
- Immutable revision: `ccb12fe2fc142cb752085506c3db22572290e90c`
- Sytra fingerprint: `01335348e4d935fcc40408cad5e840836fb7c5c7bab6c5670dd735b204a1671c`
- Reference: `transformers@5.12.1`, CPU float32
- Serving weights: BF16 SafeTensors, 2 layers, 4 routed experts, top-2
- Envelope: 2 GiB RAM, 1 GiB accelerator, 128-token context, 1 MiB dense tile
- Workload: 6-token prompt, 2-token warmup, 3 runs of 16 generated tokens

The immutable oracle passed two teacher-forced cases with 16 final-logit probes each. Maximum absolute probe errors were `0.0005147` and `0.0009729`.

| KV placement | Aggregate tok/s | Target | Result |
| --- | ---: | ---: | --- |
| Compact BF16 host KV | 172.75 | 5 | pass |
| Compact BF16 CUDA KV | 161.72 | 5 | pass |

The CUDA-KV run reported `kv_tier: accelerator`, no planner notes or CUDA budget denials, and a 1,903,872-byte peak CUDA allocation during oracle verification. The tiny fixture is dominated by launch overhead, so CUDA KV is not expected to improve its speed; the path exists to bound host memory and avoid moving a long-context cache between host and device.

Reproduce the serving-side measurement after creating `.sytra-runtime.json` and `.sytra-oracle.json` for the pinned checkpoint:

```powershell
target-build\debug\sytra-engine.exe benchmark `
  --model .test-models\hf-internal-testing--tiny-random-MixtralForCausalLM `
  --ram-limit-mb 2048 --accelerator-limit-mb 1024 `
  --ram-dense-mb 256 --ram-expert-mb 64 --accelerator-expert-mb 64 `
  --context 128 --verification-positions 4 --dense-tile-mb 1 `
  --cuda-device 0 --prompt "The capital of France is" `
  --max-tokens 16 --iterations 3 --warmup-tokens 2 --target-tps 5
```

A representative 700B-class result remains unmeasured. It requires the exact target checkpoint, its promoted adapter and oracle, enough local storage for every routed expert, and a benchmark on the intended low-end machine.

## 2026-08-02 — tiny random GraniteMoE on RTX 3060 12 GB

- Checkpoint: `hf-internal-testing/tiny-random-GraniteMoeForCausalLM`
- Immutable revision: `4037537bb31c7b3c0fd9ea27945f11089d15fcc8`
- Sytra fingerprint: `e9c4f5cad6ae6f976ea39d81dbbf60bdbdba9443e00948904ae93889cadc6763`
- Reference: `transformers@5.12.1`, CPU float32
- Serving weights: F32 SafeTensors, 2 layers, 8 routed experts, top-2
- Placement: bounded CPU-streamed dense/experts plus compact BF16 CUDA KV
- Envelope/workload: same 2 GiB RAM, 1 GiB accelerator, 128 context, 1 MiB tile, and 3 × 16-token benchmark shape used above

The two-case immutable oracle passed with maximum probe errors `0.0003482` and `0.0001309`. Aggregate throughput was **5.79 tok/s**, so `target_met` was true for the 5 tok/s target. The planner correctly reports the F32 dense and expert CPU fallbacks; this fixture is architecture-correctness evidence, not evidence that a large Granite checkpoint will reach the same rate.
