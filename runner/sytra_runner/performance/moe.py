from __future__ import annotations


def detect_moe(config: dict) -> dict:
    text = config.get("text_config") or config
    experts = next((text[k] for k in ("num_experts", "num_local_experts", "n_routed_experts") if isinstance(text.get(k), int) and text[k] > 1), None)
    active = next((text[k] for k in ("num_experts_per_tok", "num_selected_experts", "num_experts_per_token") if isinstance(text.get(k), int) and text[k] > 0), None)
    architecture = text.get("model_type") or config.get("model_type")
    return {"is_moe": experts is not None, "num_experts": experts, "experts_per_token": active,
            "architecture": architecture, "execution": "model-native", "expert_pruning": False}
