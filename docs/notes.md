# Chronos-Coder-7B — Dataset Plan & Training Objective

Base model: **Qwen2.5-Coder-7B-Instruct**
Method: **QLoRA** (RTX 3060 12GB) → merge → optional DPO pass
Owner: Matheus Nery Walkowicz

---

## 1. Overall Objective

The goal is not "more data" — it's a **7B model that behaves like a much larger one** on three axes:

1. **Breadth**: solid, idiomatic code generation across 15 languages/frameworks + general reasoning.
2. **Honesty**: the model says "I don't know" or asks for clarification instead of hallucinating APIs, functions, or facts — especially under its own knowledge boundary.
3. **Agentic reliability**: the model calls tools _only_ when appropriate, follows the user's actual intent/context instead of drifting, and doesn't hallucinate tool calls or parameters — even at small scale, where this failure mode is most common.

A 7B model can't out-know a 70B model. What it _can_ do is **know its own limits and stay grounded** — that's the actual lever for making it feel "intelligent" in practice.

### Training stages

| Stage | Method                | Purpose                                                                               |
| ----- | --------------------- | ------------------------------------------------------------------------------------- |
| 1     | SFT (QLoRA)           | Code capability across all languages + general instruction following                  |
| 2     | SFT (QLoRA)           | Tool-use — classic single-call fluency (§4.1), then programmatic orchestration (§4.2) |
| 3     | SFT or self-generated | Refusal/honesty calibration (R-Tuning style, model-specific)                          |
| 4     | DPO                   | Preference alignment — honesty, tool-call precision, helpfulness                      |

Stages 1–2 can be merged into one SFT mix. Stage 3 should be generated _after_ Stage 1–2, using your own model's outputs (see §5). Stage 4 is optional but recommended given the tool-hallucination goal.

---

## 2. Code Capability Datasets

### 2.1 General / multi-language instruction data

| Dataset ID                              | Notes                                                                                                   |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | Decontaminated multi-language evol-instruct set                                                         |
| `theblackcat102/evol-codealpaca-v1`     | Multi-language, GPT-4 augmented via 10 evolution strategies                                             |
| `nickrosh/Evol-Instruct-Code-80k-v1`    | 80k evolved instruction-response pairs                                                                  |
| `HuggingFaceH4/CodeAlpaca_20K`          | Base instruction-following code tasks                                                                   |
| `glaiveai/glaive-code-assistant-v2`     | ~950K problems; concentrated in C, C++, C#, Go, Java, JavaScript, Python, Rust — filter by language tag |
| `nvidia/OpenCodeReasoning`              | Reasoning-focused code tasks                                                                            |
| `microsoft/orca-agentinstruct-1M-v1`    | General instruction-following (non-code) — use for the "10 general" bucket                              |
| `codeparrot/apps`                       | Python-heavy, algorithmic/general reasoning                                                             |

### 2.2 Language-specific / raw source

| Language(s)                                   | Dataset ID                                          | Notes                                                                        |
| --------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------- |
| Python                                        | `iamtarun/python_code_instructions_18k_alpaca`      | Direct instruction match                                                     |
| C, C++, C#, Go, Java, JS, Python, Rust        | `glaiveai/glaive-code-assistant-v2`                 | Filter/split by language                                                     |
| 26 languages (incl. Kotlin, Swift, PHP, Ruby) | `cakiki/rosetta-code`                               | Small — reference/eval more than volume training                             |
| 358 languages, raw source                     | `bigcode/the-stack-dedup` or `bigcode/the-stack-v2` | Not instruction-formatted — needs conversion to instruction pairs before use |

### 2.3 Known gaps (no strong dedicated HF instruct dataset exists)

**Swift, Kotlin, PHP, HTML/CSS, Bash/Shell** — these need to be filled with:

- Filtered subsets from `glaive-code-assistant-v2` / `the-stack` where available, converted to instruction format, **and**

---

## 3. Honesty / Calibration Datasets (preference + refusal)

| Dataset ID          | Format                                                                        | Notes                                                                                            |
| ------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `Anthropic/hh-rlhf` | Preference pairs (chosen/rejected)                                            | ~169K helpfulness/harmlessness comparisons — standard DPO base                                   |
| `nvidia/When2Call`  | SFT (`when2call_train_sft.jsonl`) + preference (`when2call_train_pref.jsonl`) | Directly teaches when **not** to call a tool — most relevant single dataset for your stated goal |

**Not on HF, generate yourself:** R-Tuning-style refusal data. The method (from the R-Tuning paper) is model-specific by construction — you probe your own post-SFT checkpoint to find its actual knowledge boundary, then label accordingly. A generic downloaded refusal set won't transfer well because it reflects a _different_ model's knowledge gaps, not Chronos-Coder-7B's. This is a generation task for your existing pipeline, not a download.

---

## 4. Agentic / Tool-Use Datasets — targeting a Programmatic Tool-Calling paradigm

**Context for this rewrite:** OpenAI's GPT-5.6 Sol (flagship, GA July 9, 2026) introduced Programmatic Tool Calling — instead of one JSON tool call per round trip, the model writes a short program (JS in Sol's case) that orchestrates multiple tool calls, loops, filters intermediate results, and returns only the compact useful state to the model context. This collapses N round trips into one and is the direction agentic tool-use is heading for coding-focused models. Since Chronos-Coder-7B is code-native, this paradigm is a _better fit_ for it than classic single-call function calling — it's asking the model to do what it's already good at (write correct code) as the orchestration layer itself.

**Reality check:** this capability shipped days before this document was written. There is no dedicated HF dataset yet that trains a model to _write_ tool-orchestration code rather than emit a single JSON call — you will need to build this layer yourself, using the classic datasets below as the grounding layer (tool schemas, when/when-not-to-call, parameter accuracy) and then synthesizing a second layer on top (multi-call orchestration written as code).

### 4.1 Foundation layer — classic single-call function calling (train first)

| Dataset ID                                | Notes                                                                                                     |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `nvidia/When2Call`                        | Core dataset for tool-call precision and abstention — keep as your base for "should I call a tool at all" |
| `NousResearch/hermes-function-calling-v1` | Multi-turn + single-turn function calling, JSON-mode structured output                                    |
| `glaiveai/glaive-function-calling-v2`     | ~113K samples; deliberately includes no-function-needed examples to prevent over-triggering               |
| `Salesforce/xlam-function-calling-60k`    | 60K entries from Salesforce's APIGen pipeline, strong on multi-parameter accuracy                         |

Blend these rather than picking one — cross-format exposure (different JSON schemas, different system-prompt conventions) is what generalizes tool-use behavior instead of overfitting to one API style.

### 4.2 Orchestration layer — build yourself (no HF dataset exists yet)

Structure each synthetic example as:

1. A system prompt exposing 2+ tool schemas (reuse schemas from your existing tool-calling dataset).
2. A user task that requires calling more than one tool, filtering/aggregating results, or looping over a collection.
3. The target output: a short script (JS or Python, match whatever your inference harness executes) that calls the tools programmatically, filters intermediate data, and returns only the final compact result — not a single flat JSON call.
4. A negative-example subset: tasks where a single direct call is _correct_ and writing an orchestration script would be overkill — so the model learns to choose the lighter-weight approach when it's actually better, not default to code-orchestration for everything.

Run every generated script through your compile-verified harness before it enters the training set, same as your code-generation data — an orchestration script with a runtime bug is worse than a wrong JSON call, since it can silently loop or discard data.

---

## 5. Post-Stage-1 Self-Generated Data (do this after initial SFT)

Once Chronos-Coder-7B has a baseline checkpoint:

1. Probe it with in-domain and out-of-domain questions.
2. Label responses correct/incorrect against ground truth (compile-verified for code, fact-checked for general knowledge).
3. Build refusal-labeled pairs: known → answer normally; unknown → "I don't know" / clarifying question.
4. Feed this back as Stage 3 SFT data, then optionally pair with DPO in Stage 4.

This closes the loop with your compile-verified evaluation harness and failure taxonomy work — the honesty layer should be measured with the same rigor as the code-correctness layer.

---

## 6. Practical Notes

- Filter `glaive-code-assistant-v2` and `the-stack` by language **before** downloading in full — both are large, and you only need your 15 target languages.
- Convert any raw-source datasets (`the-stack`) into instruction format using your own template before mixing with instruction-native sets — don't train directly on raw code.
- Keep `When2Call` and `glaive-function-calling-v2`'s "no function needed" examples in proportion — dropping them is the most common cause of tool-happy models that call functions when they shouldn't.
- Track dataset provenance per example (source dataset, license, conversion method) — this feeds directly into your `kl-train` provenance-gated collapse guard work.
