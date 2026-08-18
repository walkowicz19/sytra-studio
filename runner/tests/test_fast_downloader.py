"""Tests for verified file selection, shard safety, and expert pager plumbing."""
import os
import tempfile
from pathlib import Path

import pytest

from sytra_runner.fast_downloader import FastHFDownloader
from sytra_runner.shard_manager import ShardManager
from sytra_runner.backends.expert_pager import GPUExpertPager
from sytra_runner.xet_safety import XET_SAFETY_ENV, apply_xet_safety


def test_fast_downloader_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("Qwen/Qwen2.5-Coder-7B", cache_dir=tmpdir, tokenless=True)
        assert downloader.tokenless is True
        assert downloader.repo_id == "Qwen/Qwen2.5-Coder-7B"


def test_xet_safety_overwrites_high_ram_defaults():
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "8gb"
    os.environ["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] = "1gb"
    apply_xet_safety()
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"] == "false"
    assert os.environ["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "1"
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"
    assert os.environ["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] == "16mb"
    for key, expected in XET_SAFETY_ENV.items():
        assert os.environ[key] == expected


def test_child_process_env_caps_xet_for_mergekit_grandchildren():
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "8gb"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from sytra_runner.xet_safety import child_process_env

    env = child_process_env({"OMP_NUM_THREADS": "2"})
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"
    assert env["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] == "16mb"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["OMP_NUM_THREADS"] == "2"


def test_downloader_is_sequential_on_every_os():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("org/tiny", cache_dir=tmpdir, max_workers=8)
    assert downloader.max_workers == 1


def test_downloader_caps_xet_buffers_even_when_cache_is_custom():
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] = "2gb"
    with tempfile.TemporaryDirectory() as tmpdir:
        FastHFDownloader("org/tiny", cache_dir=tmpdir)
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] == "64mb"
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"


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
