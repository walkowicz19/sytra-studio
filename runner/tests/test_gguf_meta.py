from pathlib import Path

from sytra_runner.gguf_meta import read_gguf_metadata
from sytra_runner.memory_hierarchy import plan_llama_cpp_offload
from gguf_test_file import write_metadata_gguf


def test_reads_architecture_and_experts(tmp_path):
    path = write_metadata_gguf(
        tmp_path / "moe.gguf",
        {
            "general.architecture": "mixtral",
            "mixtral.block_count": 32,
            "mixtral.expert_count": 8,
            "mixtral.expert_used_count": 2,
            "general.file_type": 15,
        },
    )
    meta = read_gguf_metadata(path)
    assert meta.architecture == "mixtral"
    assert meta.is_moe
    assert meta.n_layer == 32
    assert meta.quantization == "MOSTLY_Q4_K_M"


def test_gpu_first_offload_keeps_mlock_off_on_windows():
    plan = plan_llama_cpp_offload(
        weight_bytes=18 * 1024 * 1024 * 1024,
        vram_budget_mb=12 * 1024,
        ram_budget_mb=12 * 1024,
        n_layer=32,
        kv_bytes=512 * 1024 * 1024,
        cpu_count=8,
        windows=True,
    )
    assert plan.mlock is False
    assert plan.mmap is True
    assert plan.gpu_layers >= 1
    assert plan.gpu_layers < 32
    assert plan.strategy == "gpu-first-hybrid"
