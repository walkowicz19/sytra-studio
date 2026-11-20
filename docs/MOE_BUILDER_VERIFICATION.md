# MoE Builder & Model Merger Verification

`runner/tests/verify_moe_end_to_end.py` and `runner/tests/verify_merge_methods_real.py`
exercise real end-to-end merges in `.sytra-envs/merge-env` against tiny synthetic
local Llama and Qwen2 checkpoints with `HF_HUB_OFFLINE=1` (no Hub downloads).

## Reproducible Commands

```powershell
$env:PYTHONPATH = 'runner'
$env:HF_HUB_OFFLINE = '1'
.\.sytra-envs\merge-env\Scripts\python.exe -u runner/tests/verify_moe_end_to_end.py
.\.sytra-envs\merge-env\Scripts\python.exe -u runner/tests/verify_merge_methods_real.py
```

And unit/numerical suites in `.sytra-envs/train-env` and Rust host/contracts/MCP:

```powershell
$env:PYTHONPATH = 'runner'
.\.sytra-envs\train-env\Scripts\python.exe -u -m pytest runner/tests/test_moe_merge.py runner/tests/test_merge_algorithms.py -v --tb=short
cargo test -j 1 -p sytra-host -p sytra-contracts -p sytra-mcp
```

## Verified Results (2026-11-13)

- **Environment provisioner (`.sytra-envs/merge-env`)**:
  - Provisioned with `transformers==4.57.6`, `huggingface-hub==0.36.2`, `accelerate==1.0.1`,
    `peft==0.19.1`, `pydantic==2.9.2`, and `mergekit==0.0.5.1` matching
    `MERGE_RUNTIME_REVISION = 2` and `PROVISION_SCHEMA = 2`.
  - `uv pip check` exited `0` (67 packages compatible); `sytra-mcp` `get_status`
    reported `envs.merge.provisioned = true` and `stamp_ok = true`.
- **MoE Builder (`runner/tests/verify_moe_end_to_end.py`, exit 0)**:
  - Fixed an upstream MergeKit `0.0.5.1` bug via `sytra_runner.moe_exec`:
    MergeKit's `initialize_io()` omitted `config.shared_experts`, which caused
    `shared_loader.get_tensor(...)` to fail with `AttributeError` when a distinct
    shared expert model was configured for Qwen MoE or DeepSeek MoE.
  - **Mixtral MoE**: verified config, tokenizer copy, distinct hidden-state router
    weights, `AutoModelForCausalLM.from_pretrained`, finite forward logits, and
    `sytra-provenance.json`.
  - **Qwen MoE (`qwen2_moe`)**: verified config, `shared_expert` weights,
    `shared_expert_gate` router weights, distinct expert gates, `AutoModelForCausalLM`
    load, and finite forward logits.
  - **DeepSeek MoE (`shared_experts=1` and `shared_experts=0`)**: verified config
    (`model_type == "deepseek"`, `n_routed_experts == 2`, `n_shared_experts`),
    expert/shared tensor layout, distinct gate weights, and layerwise routed+shared
    MLP forward pass. Note: MergeKit `0.0.5.1` emits HF `auto_map` entries
    (`deepseek-ai/deepseek-moe-16b-base--modeling_deepseek`) rather than offline-native
    `transformers` `deepseek_v2`, and MergeKit `0.0.5.1` ships Mixtral, Qwen MoE
    (`qwen2_moe`), and DeepSeek MoE (not `qwen3` MoE).
- **Merge methods & atomic publishing (`runner/tests/verify_merge_methods_real.py` + `pytest`, 23 passed)**:
  - Verified `slerp`, `linear`, `task_arithmetic`, `ties`, `dare_ties`, and `dare_linear`
    via MergeKit `0.0.5.1`, plus `multi_slerp` (Karcher mean on unit sphere + task vector
    mode) and `nuslerp` via `runner/sytra_runner/spherical_merge.py`.
  - Verified MergeKit and spherical outputs match analytical reference formulas,
    pass `AutoModelForCausalLM` forward checks, clean up temporary staging dirs, and
    preserve existing outputs untouched on simulated/missing-checkpoint failures.

## Scope & Limitations

- End-to-end checks exercise small synthetic checkpoints locally (`16 GiB RAM`,
  CPU execution in `merge-env`) and verify structural, numerical, and contract
  correctness rather than perplexity, downstream benchmark accuracy, or production
  7B+ throughput.
