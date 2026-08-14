"""Emit crates/sytra-contracts/src/catalog.json with downloadable HF repos + risk metadata."""
from __future__ import annotations

import json
from pathlib import Path

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ATTN + ["gate_proj", "up_proj", "down_proj"]


def entry(
    model_id: str,
    name: str,
    param_count: int,
    architecture: str,
    tags: list[str],
    *,
    license: str = "apache-2.0",
    dtype: str = "bfloat16",
    tokenizer_id: str | None = None,
    modules: list[str] | None = None,
    moe_active: int | None = None,
    format: str = "",
    downloadable: bool = True,
    workflows: list[str] | None = None,
    gated: bool = False,
    approx_gb: float | None = None,
    hint: str = "medium",
    risks: list[str] | None = None,
) -> dict:
    fmt = format or ("gguf" if "gguf" in model_id.lower() else "safetensors")
    flows = workflows
    if flows is None:
        flows = ["inference"] if fmt == "gguf" else ["finetune", "merge", "inference"]
    return {
        "model_id": model_id,
        "name": name,
        "param_count": param_count,
        "architecture": architecture,
        "dtype": dtype,
        "moe_active_params": moe_active,
        "license": license,
        "default_target_modules": modules or (MLP if fmt != "gguf" else ATTN),
        "tokenizer_id": tokenizer_id or model_id.replace("-GGUF", "").replace("-gguf", ""),
        "use_case_tags": tags,
        "benchmark_hint": hint,
        "format": fmt,
        "downloadable": downloadable,
        "workflows": flows,
        "gated": gated,
        "approx_download_gb": approx_gb,
        "explicit_risks": risks or [],
    }


models: list[dict] = []

# Existing contracts (keep IDs used by tests)
models += [
    entry(
        "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        "Llama 3 8B Instruct (MLX 4-bit)",
        8_030_000_000,
        "LlamaForCausalLM",
        ["chat", "general", "instruction-following", "mlx"],
        license="llama3",
        dtype="float16",
        tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
        modules=ATTN,
        format="mlx",
        downloadable=False,
        workflows=[],
        gated=True,
        approx_gb=4.5,
        hint="high",
        risks=["Apple MLX package — not for Windows CUDA or llama.cpp."],
    ),
    entry(
        "mistralai/Mistral-7B-v0.1",
        "Mistral 7B v0.1",
        7_242_000_000,
        "MistralForCausalLM",
        ["general", "completion", "fine-tuning"],
        tokenizer_id="mistralai/Mistral-7B-v0.1",
        hint="medium",
        approx_gb=14.5,
    ),
    entry(
        "org/knowledge-ft",
        "Knowledge Fine-tuned Mistral",
        7_242_000_000,
        "MistralForCausalLM",
        ["knowledge"],
        tokenizer_id="mistralai/Mistral-7B-v0.1",
        downloadable=False,
        workflows=["merge"],
        modules=ATTN,
    ),
    entry(
        "org/toolcalling-ft",
        "Toolcalling Fine-tuned Mistral",
        7_242_000_000,
        "MistralForCausalLM",
        ["tool-use"],
        tokenizer_id="mistralai/Mistral-7B-v0.1",
        downloadable=False,
        workflows=["merge"],
        modules=ATTN,
    ),
    entry(
        "org/behavior-ft",
        "Behavior Fine-tuned Mistral",
        7_242_000_000,
        "MistralForCausalLM",
        ["chat"],
        tokenizer_id="mistralai/Mistral-7B-v0.1",
        downloadable=False,
        workflows=["merge"],
        modules=ATTN,
    ),
    entry(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5 Coder 7B Instruct",
        7_620_000_000,
        "Qwen2ForCausalLM",
        ["code", "frontend", "instruction-following", "fine-tuning"],
        hint="high",
        approx_gb=15.2,
    ),
    entry(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct",
        7_620_000_000,
        "Qwen2ForCausalLM",
        ["chat", "general", "instruction-following", "fine-tuning"],
        hint="high",
        approx_gb=15.2,
    ),
    entry(
        "Qwen/Qwen3.5-9B-Base",
        "Qwen3.5 9B Base",
        10_000_000_000,
        "Qwen3_5ForConditionalGeneration",
        ["base", "code", "fine-tuning", "multimodal"],
        modules=ATTN,
        hint="high",
        approx_gb=20.0,
        risks=["Never treat as Qwen2. Vision tensors must not receive text LoRA."],
    ),
]

# Qwen2.5 dense (finetune)
for size, params, gb in [
    ("0.5B", 494_000_000, 1.0),
    ("1.5B", 1_540_000_000, 3.1),
    ("3B", 3_090_000_000, 6.2),
    ("14B", 14_770_000_000, 29.5),
]:
    models.append(
        entry(
            f"Qwen/Qwen2.5-{size}-Instruct",
            f"Qwen2.5 {size} Instruct",
            params,
            "Qwen2ForCausalLM",
            ["chat", "instruction-following", "fine-tuning"]
            + (["small"] if "0.5" in size or "1.5" in size or size == "3B" else ["large"]),
            hint="high" if size in {"7B", "14B"} else "medium",
            approx_gb=gb,
        )
    )

models.append(
    entry(
        "Qwen/Qwen2.5-7B",
        "Qwen2.5 7B Base",
        7_620_000_000,
        "Qwen2ForCausalLM",
        ["base", "fine-tuning"],
        approx_gb=15.2,
    )
)

# Official Qwen GGUF (Sytra Xet download)
qwen_gguf = [
    ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5 0.5B Instruct GGUF", 494_000_000, 0.47, ["small", "fast", "chat"]),
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5 1.5B Instruct GGUF", 1_540_000_000, 1.0, ["small", "fast", "chat"]),
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5 3B Instruct GGUF", 3_090_000_000, 2.0, ["small", "chat"]),
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct GGUF", 7_620_000_000, 4.7, ["chat", "general"]),
    ("Qwen/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5 14B Instruct GGUF", 14_770_000_000, 9.0, ["chat", "general"]),
    ("Qwen/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5 32B Instruct GGUF", 32_800_000_000, 20.0, ["chat", "large"]),
    ("Qwen/Qwen2.5-72B-Instruct-GGUF", "Qwen2.5 72B Instruct GGUF", 72_700_000_000, 43.0, ["chat", "large", "unverified"]),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", "Qwen2.5 Coder 1.5B GGUF", 1_540_000_000, 1.0, ["code", "small"]),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5 Coder 7B GGUF", 7_620_000_000, 4.7, ["code"]),
    ("Qwen/Qwen2.5-Coder-14B-Instruct-GGUF", "Qwen2.5 Coder 14B GGUF", 14_770_000_000, 9.0, ["code"]),
    ("Qwen/Qwen2.5-Math-7B-Instruct-GGUF", "Qwen2.5 Math 7B GGUF", 7_620_000_000, 4.7, ["math"]),
]
for mid, name, params, gb, tags in qwen_gguf:
    models.append(
        entry(mid, name, params, "Qwen2ForCausalLM", tags, format="gguf", approx_gb=gb, modules=ATTN)
    )

# MoE GGUF / ST that can actually be named
models += [
    entry(
        "allenai/OLMoE-1B-7B-0125-Instruct-GGUF",
        "OLMoE 1B-active 7B Instruct GGUF",
        6_900_000_000,
        "OlmoeForCausalLM",
        ["moe", "chat", "instruct"],
        format="gguf",
        moe_active=1_300_000_000,
        approx_gb=4.21,
        hint="high",
        modules=ATTN,
    ),
    entry(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF",
        "Qwen1.5 MoE A2.7B Chat GGUF",
        14_300_000_000,
        "Qwen2MoeForCausalLM",
        ["moe", "chat", "unverified"],
        format="gguf",
        moe_active=2_700_000_000,
        approx_gb=8.2,
        modules=ATTN,
    ),
    entry(
        "bartowski/Qwen2-57B-A14B-Instruct-GGUF",
        "Qwen2 57B-A14B Instruct GGUF",
        57_000_000_000,
        "Qwen2MoeForCausalLM",
        ["moe", "large", "unverified"],
        format="gguf",
        moe_active=14_000_000_000,
        approx_gb=32.0,
        modules=ATTN,
        license="tongyi-qianwen",
    ),
]

# Small / practical GGUF
ggufs = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF", "SmolLM2 135M Instruct", 135_000_000, 0.09, "LlamaForCausalLM", ["small", "edge"]),
    ("HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", "SmolLM2 360M Instruct", 360_000_000, 0.22, "LlamaForCausalLM", ["small", "edge"]),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2 1.7B Instruct", 1_700_000_000, 1.1, "LlamaForCausalLM", ["small", "fast"]),
    ("microsoft/Phi-3-mini-4k-instruct-gguf", "Phi-3 Mini 4k Instruct GGUF", 3_800_000_000, 2.3, "Phi3ForCausalLM", ["small", "chat"]),
    ("microsoft/phi-4-gguf", "Phi-4 14B Instruct GGUF", 14_700_000_000, 9.1, "Phi3ForCausalLM", ["reasoning", "math"]),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama 3.2 1B Instruct GGUF", 1_240_000_000, 0.8, "LlamaForCausalLM", ["small", "chat"]),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B Instruct GGUF", 3_210_000_000, 2.0, "LlamaForCausalLM", ["small", "chat"]),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Llama 3.1 8B Instruct GGUF", 8_030_000_000, 4.9, "LlamaForCausalLM", ["chat", "general"]),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral 7B Instruct v0.3 GGUF", 7_250_000_000, 4.4, "MistralForCausalLM", ["chat", "general"]),
    ("bartowski/gemma-2-2b-it-GGUF", "Gemma 2 2B IT GGUF", 2_610_000_000, 1.7, "Gemma2ForCausalLM", ["chat", "small"]),
    ("bartowski/gemma-2-9b-it-GGUF", "Gemma 2 9B IT GGUF", 9_240_000_000, 5.8, "Gemma2ForCausalLM", ["chat", "general"]),
    ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5 0.5B Instruct (bartowski)", 494_000_000, 0.4, "Qwen2ForCausalLM", ["small", "fast"]),
    ("unsloth/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct (Unsloth GGUF)", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["chat", "unsloth"]),
    ("unsloth/Llama-3.1-8B-Instruct-GGUF", "Llama 3.1 8B Instruct (Unsloth GGUF)", 8_030_000_000, 4.9, "LlamaForCausalLM", ["chat", "unsloth"]),
    ("ggml-org/gemma-3-270m-it-GGUF", "Gemma 3 270M IT GGUF", 270_000_000, 0.2, "Gemma3ForCausalLM", ["small", "edge", "unverified"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", "DeepSeek R1 Distill Qwen 1.5B", 1_540_000_000, 1.1, "Qwen2ForCausalLM", ["reasoning", "small"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek R1 Distill Qwen 7B", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["reasoning"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF", "DeepSeek R1 Distill Qwen 14B", 14_770_000_000, 9.0, "Qwen2ForCausalLM", ["reasoning"]),
    ("bartowski/OLMo-2-0425-1B-Instruct-GGUF", "OLMo 2 1B Instruct GGUF", 1_000_000_000, 0.7, "Olmo2ForCausalLM", ["chat", "small"]),
    ("bartowski/OLMo-2-1124-7B-Instruct-GGUF", "OLMo 2 7B Instruct GGUF", 7_000_000_000, 4.4, "Olmo2ForCausalLM", ["chat"]),
]
gated_llama = {
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "unsloth/Llama-3.1-8B-Instruct-GGUF",
}
gated_gemma = {
    "bartowski/gemma-2-2b-it-GGUF",
    "bartowski/gemma-2-9b-it-GGUF",
    "ggml-org/gemma-3-270m-it-GGUF",
}
for mid, name, params, gb, arch, tags in ggufs:
    lic = "apache-2.0"
    gated = False
    if mid in gated_llama:
        lic, gated = "llama3.1", True
    if mid in gated_gemma:
        lic, gated = "gemma", True
    if "Phi" in name or "phi" in mid:
        lic = "mit"
    models.append(
        entry(mid, name, params, arch, tags, format="gguf", approx_gb=gb, modules=ATTN, license=lic, gated=gated)
    )

# Large / danger (catalogued so MCP can warn instead of guessing)
models += [
    entry(
        "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF",
        "Llama 3.1 70B Instruct GGUF",
        70_600_000_000,
        "LlamaForCausalLM",
        ["large", "chat", "unverified"],
        format="gguf",
        license="llama3.1",
        gated=True,
        approx_gb=40.0,
        modules=ATTN,
    ),
    entry(
        "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "Mixtral 8x7B Instruct GGUF",
        46_700_000_000,
        "MixtralForCausalLM",
        ["moe", "large", "unverified"],
        format="gguf",
        moe_active=12_900_000_000,
        approx_gb=26.0,
        modules=ATTN,
        license="apache-2.0",
    ),
    entry(
        "bartowski/Mixtral-8x22B-Instruct-v0.1-GGUF",
        "Mixtral 8x22B Instruct GGUF",
        141_000_000_000,
        "MixtralForCausalLM",
        ["moe", "large", "unverified"],
        format="gguf",
        moe_active=39_000_000_000,
        approx_gb=80.0,
        modules=ATTN,
    ),
    entry(
        "unsloth/Kimi-K2.7-Code-GGUF",
        "Kimi K2.7 Code GGUF",
        1_000_000_000_000,
        "KimiK25ForConditionalGeneration",
        ["coding", "moe", "large", "unverified"],
        format="gguf",
        approx_gb=295.0,
        modules=ATTN,
    ),
    entry(
        "THUDM/GLM-5.2-744B-A40B-GGUF",
        "GLM-5.2 744B MoE GGUF",
        744_000_000_000,
        "Glm4MoeForCausalLM",
        ["moe", "frontier", "large", "unverified"],
        format="gguf",
        moe_active=40_000_000_000,
        approx_gb=370.0,
        modules=ATTN,
    ),
]

# Finetune safetensors (small/medium)
st = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", 494_000_000, "Qwen2ForCausalLM", ["chat", "small", "fine-tuning"], 1.0),
    ("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", 1_240_000_000, "LlamaForCausalLM", ["chat", "small", "fine-tuning"], 2.5),
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", 3_210_000_000, "LlamaForCausalLM", ["chat", "small", "fine-tuning"], 6.4),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", 8_030_000_000, "LlamaForCausalLM", ["chat", "fine-tuning"], 16.1),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3", 7_250_000_000, "MistralForCausalLM", ["chat", "fine-tuning"], 14.5),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini 4k Instruct", 3_800_000_000, "Phi3ForCausalLM", ["chat", "small", "fine-tuning"], 7.6),
    ("microsoft/phi-4", "Phi-4 14B", 14_700_000_000, "Phi3ForCausalLM", ["reasoning", "fine-tuning", "large"], 29.0),
    ("google/gemma-2-2b-it", "Gemma 2 2B IT", 2_610_000_000, "Gemma2ForCausalLM", ["chat", "small", "fine-tuning"], 5.2),
    ("google/gemma-2-9b-it", "Gemma 2 9B IT", 9_240_000_000, "Gemma2ForCausalLM", ["chat", "fine-tuning"], 18.5),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B Instruct", 1_700_000_000, "LlamaForCausalLM", ["small", "fine-tuning"], 3.4),
    ("unsloth/mistral-7b-v0.3-bnb-4bit", "Mistral 7B v0.3 (Unsloth 4-bit)", 7_250_000_000, "MistralForCausalLM", ["finetune", "4-bit", "unsloth"], 4.1),
    ("unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "Qwen2.5 7B Instruct (Unsloth 4-bit)", 7_620_000_000, "Qwen2ForCausalLM", ["finetune", "4-bit", "unsloth"], 4.5),
    ("unsloth/Llama-3.1-8B-Instruct-bnb-4bit", "Llama 3.1 8B Instruct (Unsloth 4-bit)", 8_030_000_000, "LlamaForCausalLM", ["finetune", "4-bit", "unsloth"], 5.0),
]
# Qwen 0.5B ST already added in loop - skip duplicate
seen = {m["model_id"] for m in models}
for mid, name, params, arch, tags, gb in st:
    if mid in seen:
        continue
    lic, gated = "apache-2.0", False
    if "llama" in mid.lower() or "Llama" in name:
        lic, gated = "llama3.1", True
    if "gemma" in mid:
        lic, gated = "gemma", True
    if "Phi" in name or "phi" in mid:
        lic = "mit"
    models.append(
        entry(mid, name, params, arch, tags, license=lic, gated=gated, approx_gb=gb, modules=MLP if "4-bit" not in name else ATTN)
    )

# Additional popular HF GGUF + SafeTensors (pinned IDs for Sytra Xet downloads)
extra_gguf = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF", "TinyLlama 1.1B Chat GGUF", 1_100_000_000, 0.7, "LlamaForCausalLM", ["small", "edge", "chat"]),
    ("Qwen/Qwen3-0.6B-GGUF", "Qwen3 0.6B GGUF", 600_000_000, 0.5, "Qwen3ForCausalLM", ["small", "chat", "unverified"]),
    ("Qwen/Qwen3-1.7B-GGUF", "Qwen3 1.7B GGUF", 1_700_000_000, 1.1, "Qwen3ForCausalLM", ["small", "chat", "unverified"]),
    ("Qwen/Qwen3-4B-GGUF", "Qwen3 4B GGUF", 4_000_000_000, 2.5, "Qwen3ForCausalLM", ["chat", "unverified"]),
    ("Qwen/Qwen3-8B-GGUF", "Qwen3 8B GGUF", 8_000_000_000, 5.0, "Qwen3ForCausalLM", ["chat", "unverified"]),
    ("Qwen/Qwen3-14B-GGUF", "Qwen3 14B GGUF", 14_800_000_000, 9.0, "Qwen3ForCausalLM", ["chat", "unverified"]),
    ("Qwen/Qwen3-32B-GGUF", "Qwen3 32B GGUF", 32_800_000_000, 20.0, "Qwen3ForCausalLM", ["large", "unverified"]),
    ("Qwen/QwQ-32B-GGUF", "QwQ 32B GGUF", 32_800_000_000, 20.0, "Qwen2ForCausalLM", ["reasoning", "large"]),
    ("Qwen/Qwen2.5-Coder-32B-Instruct-GGUF", "Qwen2.5 Coder 32B Instruct GGUF", 32_800_000_000, 20.0, "Qwen2ForCausalLM", ["code", "large"]),
    ("unsloth/Phi-4-mini-instruct-GGUF", "Phi-4 Mini Instruct GGUF", 3_800_000_000, 2.4, "Phi3ForCausalLM", ["small", "chat"]),
    ("microsoft/Phi-3.5-mini-instruct-gguf", "Phi-3.5 Mini Instruct GGUF", 3_800_000_000, 2.4, "Phi3ForCausalLM", ["small", "chat"]),
    ("bartowski/Phi-4-GGUF", "Phi-4 Instruct (bartowski) GGUF", 14_700_000_000, 9.1, "Phi3ForCausalLM", ["reasoning", "math"]),
    ("bartowski/Llama-3.3-70B-Instruct-GGUF", "Llama 3.3 70B Instruct GGUF", 70_600_000_000, 42.5, "LlamaForCausalLM", ["large", "chat", "unverified"]),
    ("bartowski/Llama-3.2-11B-Vision-Instruct-GGUF", "Llama 3.2 11B Vision Instruct GGUF", 10_600_000_000, 7.0, "MllamaForConditionalGeneration", ["vision", "multimodal", "unverified"]),
    ("ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF", "Kimi VL A3B Thinking GGUF", 16_000_000_000, 9.8, "KimiVLForConditionalGeneration", ["vision", "coding", "thinking", "moe", "unverified"]),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek R1 Distill Llama 8B GGUF", 8_030_000_000, 4.9, "LlamaForCausalLM", ["reasoning"]),
    ("bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF", "DeepSeek R1 Distill Qwen 32B GGUF", 32_800_000_000, 20.0, "Qwen2ForCausalLM", ["reasoning", "large", "unverified"]),
    ("bartowski/gemma-2-27b-it-GGUF", "Gemma 2 27B IT GGUF", 27_200_000_000, 16.0, "Gemma2ForCausalLM", ["chat", "large"]),
    ("bartowski/gemma-3-1b-it-GGUF", "Gemma 3 1B IT GGUF", 1_000_000_000, 0.7, "Gemma3ForCausalLM", ["small", "chat", "unverified"]),
    ("bartowski/gemma-3-4b-it-GGUF", "Gemma 3 4B IT GGUF", 4_000_000_000, 2.6, "Gemma3ForCausalLM", ["chat", "unverified"]),
    ("bartowski/gemma-3-12b-it-GGUF", "Gemma 3 12B IT GGUF", 12_200_000_000, 7.5, "Gemma3ForCausalLM", ["chat", "unverified"]),
    ("bartowski/gemma-3-27b-it-GGUF", "Gemma 3 27B IT GGUF", 27_400_000_000, 16.5, "Gemma3ForCausalLM", ["large", "unverified"]),
    ("unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5 Coder 7B (Unsloth GGUF)", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["code", "unsloth"]),
    ("unsloth/gemma-2-9b-it-GGUF", "Gemma 2 9B IT (Unsloth GGUF)", 9_240_000_000, 5.8, "Gemma2ForCausalLM", ["chat", "unsloth"]),
    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral Nemo Instruct GGUF", 12_200_000_000, 7.5, "MistralForCausalLM", ["chat", "general"]),
    ("bartowski/Mistral-Small-Instruct-2409-GGUF", "Mistral Small Instruct GGUF", 22_000_000_000, 13.0, "MistralForCausalLM", ["chat", "large", "unverified"]),
    ("bartowski/Codestral-22B-v0.1-GGUF", "Codestral 22B GGUF", 22_200_000_000, 13.0, "MistralForCausalLM", ["code", "large", "unverified"]),
    ("ibm-granite/granite-3.1-8b-instruct-GGUF", "Granite 3.1 8B Instruct GGUF", 8_000_000_000, 4.9, "GraniteForCausalLM", ["chat", "unverified"]),
    ("ibm-granite/granite-3.1-2b-instruct-GGUF", "Granite 3.1 2B Instruct GGUF", 2_500_000_000, 1.6, "GraniteForCausalLM", ["small", "chat", "unverified"]),
    ("NousResearch/Hermes-3-Llama-3.1-8B-GGUF", "Hermes 3 Llama 3.1 8B GGUF", 8_030_000_000, 4.9, "LlamaForCausalLM", ["chat", "tool-use"]),
    ("bartowski/Yi-1.5-9B-Chat-GGUF", "Yi 1.5 9B Chat GGUF", 8_830_000_000, 5.5, "LlamaForCausalLM", ["chat", "multilingual"]),
    ("internlm/internlm2_5-7b-chat-gguf", "InternLM2.5 7B Chat GGUF", 7_700_000_000, 4.7, "InternLM2ForCausalLM", ["chat", "unverified"]),
    ("HuggingFaceH4/zephyr-7b-beta-GGUF", "Zephyr 7B Beta GGUF", 7_240_000_000, 4.4, "MistralForCausalLM", ["chat"]),
    ("bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF", "Qwen2.5 Math 1.5B Instruct GGUF", 1_540_000_000, 1.0, "Qwen2ForCausalLM", ["math", "small"]),
    ("Qwen/Qwen2.5-Math-72B-Instruct-GGUF", "Qwen2.5 Math 72B Instruct GGUF", 72_700_000_000, 43.0, "Qwen2ForCausalLM", ["math", "large", "unverified"]),
    ("unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek R1 Distill Qwen 7B (Unsloth)", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["reasoning", "unsloth"]),
    ("bartowski/OLMo-2-0325-32B-Instruct-GGUF", "OLMo 2 32B Instruct GGUF", 32_000_000_000, 19.0, "Olmo2ForCausalLM", ["chat", "large", "unverified"]),
    ("allenai/OLMo-2-1124-13B-Instruct-GGUF", "OLMo 2 13B Instruct GGUF", 13_000_000_000, 8.0, "Olmo2ForCausalLM", ["chat"]),
    ("Qwen/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5 VL 7B Instruct GGUF", 8_290_000_000, 5.5, "Qwen2VLForConditionalGeneration", ["vision", "multimodal", "unverified"]),
    ("Qwen/Qwen2.5-VL-3B-Instruct-GGUF", "Qwen2.5 VL 3B Instruct GGUF", 3_750_000_000, 2.5, "Qwen2VLForConditionalGeneration", ["vision", "multimodal", "small", "unverified"]),
    ("bartowski/Ministral-8B-Instruct-2410-GGUF", "Ministral 8B Instruct GGUF", 8_000_000_000, 4.9, "MistralForCausalLM", ["chat"]),
    ("unsloth/Llama-3.2-1B-Instruct-GGUF", "Llama 3.2 1B Instruct (Unsloth GGUF)", 1_240_000_000, 0.8, "LlamaForCausalLM", ["small", "unsloth"]),
    ("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B Instruct (Unsloth GGUF)", 3_210_000_000, 2.0, "LlamaForCausalLM", ["small", "unsloth"]),
    ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct (bartowski)", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["chat"]),
    ("lmstudio-community/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct (LM Studio)", 7_620_000_000, 4.7, "Qwen2ForCausalLM", ["chat"]),
    ("lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", "Llama 3.1 8B Instruct (LM Studio)", 8_030_000_000, 4.9, "LlamaForCausalLM", ["chat"]),
]
extra_llama = {
    "bartowski/Llama-3.3-70B-Instruct-GGUF",
    "bartowski/Llama-3.2-11B-Vision-Instruct-GGUF",
    "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF",
    "unsloth/Llama-3.2-1B-Instruct-GGUF",
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
}
extra_gemma = {
    "bartowski/gemma-2-27b-it-GGUF",
    "bartowski/gemma-3-1b-it-GGUF",
    "bartowski/gemma-3-4b-it-GGUF",
    "bartowski/gemma-3-12b-it-GGUF",
    "bartowski/gemma-3-27b-it-GGUF",
    "unsloth/gemma-2-9b-it-GGUF",
}
for mid, name, params, gb, arch, tags in extra_gguf:
    lic, gated = "apache-2.0", False
    if mid in extra_llama:
        lic, gated = "llama3.1", True
    if mid in extra_gemma:
        lic, gated = "gemma", True
    if "Phi" in name or "phi" in mid.lower():
        lic, gated = "mit", False
    models.append(
        entry(mid, name, params, arch, tags, format="gguf", approx_gb=gb, modules=ATTN, license=lic, gated=gated)
    )

extra_st = [
    ("Qwen/Qwen3-8B", "Qwen3 8B", 8_000_000_000, "Qwen3ForCausalLM", ["chat", "fine-tuning", "unverified"], 16.0),
    ("Qwen/Qwen3-4B", "Qwen3 4B", 4_000_000_000, "Qwen3ForCausalLM", ["chat", "fine-tuning", "unverified"], 8.0),
    ("Qwen/Qwen3-1.7B", "Qwen3 1.7B", 1_700_000_000, "Qwen3ForCausalLM", ["small", "fine-tuning", "unverified"], 3.4),
    ("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen2.5 Coder 32B Instruct", 32_800_000_000, "Qwen2ForCausalLM", ["code", "fine-tuning", "large"], 65.0),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5 32B Instruct", 32_800_000_000, "Qwen2ForCausalLM", ["chat", "fine-tuning", "large"], 65.0),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5 14B Instruct", 14_770_000_000, "Qwen2ForCausalLM", ["chat", "fine-tuning"], 29.5),
    ("microsoft/Phi-4-mini-instruct", "Phi-4 Mini Instruct", 3_800_000_000, "Phi3ForCausalLM", ["chat", "small", "fine-tuning"], 7.6),
    ("microsoft/Phi-3.5-mini-instruct", "Phi-3.5 Mini Instruct", 3_800_000_000, "Phi3ForCausalLM", ["chat", "small", "fine-tuning"], 7.6),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B Chat", 1_100_000_000, "LlamaForCausalLM", ["small", "fine-tuning"], 2.2),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2 135M Instruct", 135_000_000, "LlamaForCausalLM", ["small", "fine-tuning", "edge"], 0.3),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M Instruct", 360_000_000, "LlamaForCausalLM", ["small", "fine-tuning"], 0.7),
    ("google/gemma-2-27b-it", "Gemma 2 27B IT", 27_200_000_000, "Gemma2ForCausalLM", ["chat", "fine-tuning", "large"], 54.0),
    ("google/gemma-3-1b-it", "Gemma 3 1B IT", 1_000_000_000, "Gemma3ForCausalLM", ["small", "fine-tuning", "unverified"], 2.0),
    ("google/gemma-3-4b-it", "Gemma 3 4B IT", 4_000_000_000, "Gemma3ForCausalLM", ["chat", "fine-tuning", "unverified"], 8.0),
    ("mistralai/Mistral-Nemo-Instruct-2407", "Mistral Nemo Instruct", 12_200_000_000, "MistralForCausalLM", ["chat", "fine-tuning"], 24.0),
    ("mistralai/Ministral-8B-Instruct-2410", "Ministral 8B Instruct", 8_000_000_000, "MistralForCausalLM", ["chat", "fine-tuning"], 16.0),
    ("ibm-granite/granite-3.1-8b-instruct", "Granite 3.1 8B Instruct", 8_000_000_000, "GraniteForCausalLM", ["chat", "fine-tuning", "unverified"], 16.0),
    ("ibm-granite/granite-3.1-2b-instruct", "Granite 3.1 2B Instruct", 2_500_000_000, "GraniteForCausalLM", ["small", "fine-tuning", "unverified"], 5.0),
    ("NousResearch/Hermes-3-Llama-3.1-8B", "Hermes 3 Llama 3.1 8B", 8_030_000_000, "LlamaForCausalLM", ["chat", "tool-use", "fine-tuning"], 16.1),
    ("01-ai/Yi-1.5-9B-Chat", "Yi 1.5 9B Chat", 8_830_000_000, "LlamaForCausalLM", ["chat", "fine-tuning"], 17.7),
    ("internlm/internlm2_5-7b-chat", "InternLM2.5 7B Chat", 7_700_000_000, "InternLM2ForCausalLM", ["chat", "fine-tuning", "unverified"], 15.4),
    ("HuggingFaceH4/zephyr-7b-beta", "Zephyr 7B Beta", 7_240_000_000, "MistralForCausalLM", ["chat", "fine-tuning"], 14.5),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen2.5 VL 7B Instruct", 8_290_000_000, "Qwen2VLForConditionalGeneration", ["vision", "multimodal", "fine-tuning", "unverified"], 16.6),
    ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2.5 VL 3B Instruct", 3_750_000_000, "Qwen2VLForConditionalGeneration", ["vision", "multimodal", "fine-tuning", "unverified"], 7.5),
    ("unsloth/Phi-4-mini-instruct-bnb-4bit", "Phi-4 Mini Instruct (Unsloth 4-bit)", 3_800_000_000, "Phi3ForCausalLM", ["finetune", "4-bit", "unsloth"], 2.5),
    ("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", "Qwen2.5 Coder 7B (Unsloth 4-bit)", 7_620_000_000, "Qwen2ForCausalLM", ["finetune", "4-bit", "unsloth", "code"], 4.5),
    ("unsloth/gemma-2-9b-it-bnb-4bit", "Gemma 2 9B IT (Unsloth 4-bit)", 9_240_000_000, "Gemma2ForCausalLM", ["finetune", "4-bit", "unsloth"], 6.0),
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct", 70_600_000_000, "LlamaForCausalLM", ["large", "fine-tuning", "unverified"], 140.0),
    ("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B Instruct", 72_700_000_000, "Qwen2ForCausalLM", ["large", "fine-tuning", "unverified"], 145.0),
]
seen_extra = {m["model_id"] for m in models}
for mid, name, params, arch, tags, gb in extra_st:
    if mid in seen_extra:
        continue
    lic, gated = "apache-2.0", False
    if mid.startswith("meta-llama/") or "Llama-3" in mid:
        lic, gated = "llama3.1", True
    if "gemma" in mid.lower():
        lic, gated = "gemma", True
    if "Phi" in name or "phi" in mid.lower():
        lic, gated = "mit", False
    models.append(
        entry(mid, name, params, arch, tags, license=lic, gated=gated, approx_gb=gb)
    )

# Dedup preserve order
uniq = []
seen = set()
for m in models:
    if m["model_id"] in seen:
        continue
    seen.add(m["model_id"])
    uniq.append(m)

out = Path(__file__).resolve().parents[2] / "crates" / "sytra-contracts" / "src" / "catalog.json"
# script lives in runner/scripts -> parents[2] is repo root
if not out.parent.exists():
    out = Path(__file__).resolve().parents[1] / "src" / "catalog.json"
out.write_text(json.dumps(uniq, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(uniq)} models to {out}")
