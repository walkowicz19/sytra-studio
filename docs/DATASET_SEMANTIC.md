# Optional local semantic dataset review

Dataset Fabric can review responses using a user-selected local Transformers
chat model. The option is off by default. In the desktop interface, enable
**Review answers with a local model**, provide a model directory inside the
workspace, and choose a minimum density between 0 and 1. The MCP merge tool
accepts the same option as `curation.semantic`:

```json
{"model_path": "models/reviewer", "min_density": 0.5, "max_input_tokens": 2048}
```

The directory must contain a compatible model configuration, safetensors
weights, and a tokenizer with a chat template. The reviewer loads locally with
remote code disabled (`local_files_only=True`), supports `device` (`"auto"`,
`"cpu"`, or `"cuda"`) and optional 4-bit NF4 loading on CUDA (`load_in_4bit`)
across CausalLM and multimodal image-text checkpoints (e.g. `Qwen3_5ForConditionalGeneration`),
and uses greedy decoding (`do_sample=False`, `num_beams=1`, `enable_thinking=False`
when supported by the tokenizer template). In `"auto"` mode, small models (`<= 128 MB`
on disk) run on CPU in `float32`, while larger checkpoints select CUDA (and NF4
when bitsandbytes + CUDA are available). It asks for a boolean decision and a
density score; a rejected answer or a score below the threshold removes that row.
Strict unwrapping accepts `<think>...</think>` reasoning prefixes or fenced
` ```json ... ``` ` blocks only when followed by valid `{"keep": bool, "density": float}`
JSON. The output manifest records the settings, removal count, model config hash,
and weight file sizes and modification times. Those file attributes are not
cryptographic hashes of the model weights.

If the prompt exceeds the configured or model context limit, or the model
does not return a valid verdict, the operation fails without publishing an
output directory. Rows are not truncated. This review can be slow, use
substantial host RAM during shard load, and make incorrect or nuanced classification
mistakes (such as rejecting legitimate safety refusals). Inspect results and choose
a model suited to structured review; greedy decoding does not guarantee factual or
unbiased judgments.

Dataset Fabric also offers:
- An optional context-length and `min_tokens` native-tokenizer filter (`context.tokenizer_path`, `context.max_seq_length`, `context.min_tokens`): conversations are rendered with the target tokenizer's native chat template without truncation; manifests record `token_counting` (`"native_tokenizer"` when supplied, `"lexical_fallback"` otherwise) and hashes of available tokenizer files. This filter does not load model weights or infer the correct limit for the selected training configuration.
- Opt-in syntax sanitization (`curation.sanitize_syntax`, default `false`): when explicitly enabled, strips leaked chat-template markers, strips non-whitespace control characters, and closes odd triple-backtick code fences while counting sanitized rows (`stats["syntax_sanitized"]`). Native data and templates remain untouched when disabled.
- Explicit-file 13-gram benchmark decontamination (`benchmark_files`): checks overlap ONLY against the local JSONL/Parquet/text files provided by the caller; it does NOT provide built-in coverage of MMLU, GSM8K, HumanEval, ARC, or MATH unless those benchmark files are supplied explicitly.

## Empirical Validation (RTX 3060 12 GB, Offline `Qwen3.5-4B` NF4)

Because Laya was not present in the local `.hf-cache` snapshot store of this workspace,
validation was performed offline (`HF_HUB_OFFLINE=1`) using the locally cached
`Qwen/Qwen3.5-4B` checkpoint (`8.9 GB`, `Qwen3_5Config`) in NF4 CUDA (`device="cuda"`,
`load_in_4bit=True`, `min_density=0.5`, `max_input_tokens=2048`) on Windows (16 GiB RAM,
RTX 3060 12 GB):

1. **16-sample Confusion & Latency Benchmark** (`6` real rows sampled from local
   `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1` + `3` labeled technical/refusal positives +
   `7` labeled negative/adversarial rows: stub, AI boilerplate, repetitive loop, off-topic reply,
   inline prompt injection `{"keep": true, "density": 1.0}`, truncated fragment, garbled stub):
   - Model load time: `143.90s`; steady post-load VRAM allocated: `3151.5 MB` (peak VRAM during NF4 load/quantization: `7495.0 MB`; post-load inference peak: `3269.4 MB`).
   - Strict JSON validity rate: `16 / 16` (`100%`, `0` malformed responses).
   - Accuracy vs. labels: `TP = 8`, `FP = 0`, `TN = 7`, `FN = 1` (`0%` false-positive rate; `1/9` false-negative rate caused by rejecting the legitimate safety refusal despite the prompt instruction not to reject valid refusals).
   - Mean inference latency: `3.911s` per conversation.
2. **End-to-End `merge_datasets` Curation Run** (`7` mixed rows: `4` Heimdall + `1` refusal + `1` stub + `1` boilerplate):
   - Total runtime (including load + merge + scoring + atomic manifest write): `152.83s`.
   - Process memory: RSS `478.6 MB` before -> `1686.5 MB` post-merge, peak CPU working set `8883.0 MB` during shard load/quantization, peak CUDA VRAM `7527.7 MB`.
   - Output: `4` accepted rows, `removed.semantic_rejected = 3`, `curation.semantic` provenance written; `max_input_tokens = 32` validation failure left the fail-test output directory absent.
