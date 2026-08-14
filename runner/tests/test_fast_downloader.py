"""Tests for verified file selection, shard safety, and expert pager plumbing."""
import tempfile
from pathlib import Path

import pytest

from sytra_runner.fast_downloader import FastHFDownloader
from sytra_runner.shard_manager import ShardManager
from sytra_runner.backends.expert_pager import GPUExpertPager


def test_fast_downloader_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("Qwen/Qwen2.5-Coder-7B", cache_dir=tmpdir, tokenless=True)
        assert downloader.tokenless is True
        assert downloader.repo_id == "Qwen/Qwen2.5-Coder-7B"


def test_selects_every_split_file_for_one_gguf_quantization():
    files = [
        "README.md",
        "config.json",
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q8_0-00001-of-00002.gguf",
        "model-Q8_0-00002-of-00002.gguf",
        "mmproj-model-f16.gguf",
    ]
    selected = FastHFDownloader.select_files(files, purpose="inference", quant="Q4_K_M")
    assert "model-Q4_K_M-00001-of-00002.gguf" in selected
    assert "model-Q4_K_M-00002-of-00002.gguf" in selected
    assert "mmproj-model-f16.gguf" in selected
    assert not any("Q8_0" in name for name in selected)


def test_safetensors_selection_keeps_all_shards():
    files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    selected = FastHFDownloader.select_files(files, purpose="inference", quant="auto")
    assert {name for name in selected if name.endswith(".safetensors")} == set(files[2:])


def test_shard_manager_mock_manifest():
    class MockDownloader:
        def fetch_manifest(self, revision="main"):
            return {
                "config": {
                    "num_local_experts": 8,
                    "num_experts_per_tok": 2,
                },
                "index": {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.0.w1.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.1.w1.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.7.w1.weight": "model-00003-of-00003.safetensors",
                    }
                }
            }

    sm = ShardManager("mock/moe-model", downloader=MockDownloader())
    shards = sm.select_active_shards(auto_top_k=True)
    assert shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model-00003-of-00003.safetensors",
    ]


def test_python_expert_pager_cannot_bypass_the_native_runtime():
    with pytest.raises(RuntimeError, match="native Rust runtime"):
        GPUExpertPager(num_experts=8, vram_limit_mb=2048, quant_bits=4)
