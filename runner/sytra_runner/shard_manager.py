"""Model shard manifest inspection for Sytra.

Inspects model manifest structures (model.safetensors.index.json) to extract tensor
locations. Quality-preserving downloads always retain the complete shard set.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .fast_downloader import FastHFDownloader


class ShardManager:
    """Manages model shard selection and MoE expert filtering."""

    def __init__(self, repo_id_or_path: str | Path, downloader: FastHFDownloader | None = None):
        self.target = str(repo_id_or_path)
        self.downloader = downloader or FastHFDownloader(self.target)

    def analyze_model_manifest(self) -> dict[str, Any]:
        """Fetch and analyze model structure manifest."""
        manifest = self.downloader.fetch_manifest()
        config = manifest.get("config", {})
        index = manifest.get("index", {})

        weight_map = index.get("weight_map", {})
        all_shards = sorted(list(set(weight_map.values()))) if weight_map else []

        # Detect MoE structures (support flat and nested text_config like Kimi/DeepSeek-V3)
        text_cfg = config.get("text_config", {}) if isinstance(config.get("text_config"), dict) else {}
        num_experts = (
            config.get("num_local_experts")
            or config.get("n_routed_experts")
            or text_cfg.get("num_local_experts")
            or text_cfg.get("n_routed_experts")
            or 0
        )
        num_experts_per_tok = (
            config.get("num_experts_per_tok")
            or config.get("num_selected_experts")
            or text_cfg.get("num_experts_per_tok")
            or text_cfg.get("num_selected_experts")
            or 2
        )
        is_moe = num_experts > 0

        expert_tensors: dict[int, list[str]] = {}
        backbone_tensors: list[str] = []

        if is_moe and weight_map:
            for tensor_name in weight_map.keys():
                # Check pattern like layers.X.block_sparse_moe.experts.Y.z or experts.Y
                match = re.search(r"experts\.(\d+)", tensor_name)
                if match:
                    expert_id = int(match.group(1))
                    expert_tensors.setdefault(expert_id, []).append(tensor_name)
                else:
                    backbone_tensors.append(tensor_name)

        return {
            "config": config,
            "weight_map": weight_map,
            "all_shards": all_shards,
            "is_moe": is_moe,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "expert_tensors": expert_tensors,
            "backbone_tensors": backbone_tensors,
        }

    def select_active_shards(
        self,
        auto_top_k: bool = True,
        top_k_experts: int | None = None,
        target_domain: str | None = None,
    ) -> list[str]:
        """Return every weight shard required by the checkpoint.

        ``num_experts_per_tok`` is the number chosen by the router for one
        token, not a fixed subset of experts that can be deleted. Older Sytra
        releases incorrectly treated experts ``0..K-1`` as the active set.
        Keep the legacy arguments for API compatibility, but never use them to
        change model semantics.
        """
        analysis = self.analyze_model_manifest()
        return analysis["all_shards"]
