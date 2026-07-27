"""Selective Shard & MoE Expert Pruning Manager for Sytra.

Inspects model manifest structures (model.safetensors.index.json) to extract tensor
locations and map required backbone + active expert shards for automatic MoE pruning.
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
        """Select required safetensor shards based on backbone + active expert selection."""
        analysis = self.analyze_model_manifest()
        weight_map = analysis["weight_map"]
        all_shards = analysis["all_shards"]

        if not weight_map or not analysis["is_moe"]:
            # Standard dense model or single shard
            return all_shards

        k = top_k_experts or analysis["num_experts_per_tok"]
        num_total_experts = analysis["num_experts"]

        # Automatic expert selection strategy: select top-K active experts
        # (Default: keep active router experts 0..K-1 or domain-routed expert subset)
        active_expert_ids = list(range(min(k, num_total_experts)))

        required_tensors = set(analysis["backbone_tensors"])
        for exp_id in active_expert_ids:
            for t_name in analysis["expert_tensors"].get(exp_id, []):
                required_tensors.add(t_name)

        required_shards = set()
        for t_name in required_tensors:
            if t_name in weight_map:
                required_shards.add(weight_map[t_name])

        return sorted(list(required_shards))
