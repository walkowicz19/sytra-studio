"""Tests for fast_downloader, shard_manager, and expert_pager."""
import tempfile
from pathlib import Path
from sytra_runner.fast_downloader import FastHFDownloader
from sytra_runner.shard_manager import ShardManager
from sytra_runner.backends.expert_pager import GPUExpertPager


def test_fast_downloader_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("Qwen/Qwen2.5-Coder-7B", cache_dir=tmpdir, tokenless=True)
        assert downloader.tokenless is True
        assert downloader.repo_id == "Qwen/Qwen2.5-Coder-7B"


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
                    }
                }
            }

    sm = ShardManager("mock/moe-model", downloader=MockDownloader())
    shards = sm.select_active_shards(auto_top_k=True)
    assert len(shards) > 0


def test_expert_pager():
    pager = GPUExpertPager(num_experts=8, vram_limit_mb=2048, quant_bits=4)
    assert pager.num_experts == 8

    # Mock load fn
    loaded = []
    def load_fn(exp_id):
        loaded.append(exp_id)
        return f"expert_{exp_id}"

    weights = pager.get_expert_weights(0, load_fn)
    assert weights == "expert_0"
    assert pager.is_expert_resident(0)
