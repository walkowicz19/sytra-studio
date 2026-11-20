"""Resolve exact LoRA targets for supported, module-based MoE layouts."""
from __future__ import annotations

import re
from typing import Any

_ATTENTION = re.compile(r'(?:^|\.)layers\.(\d+)\..*self_attn\.(q_proj|k_proj|v_proj|o_proj)$')
_ROUTER = re.compile(r'(?:^|\.)layers\.(\d+)\.(?:block_sparse_moe|mlp)\.(?:gate|router)$')
_EXPERT = re.compile(r'(?:^|\.)layers\.(\d+)\..*?experts\.(\d+)\.(w1|w2|w3|gate_proj|up_proj|down_proj)$')
_SHARED = re.compile(r'(?:^|\.)layers\.(\d+)\..*?shared_experts?\.(?:(\d+)\.)?(w1|w2|w3|gate_proj|up_proj|down_proj)$')
_SHARED_ROOT = re.compile(r'(?:^|\.)layers\.(\d+)\..*?shared_experts?$')
_EXPERT_ROOT = re.compile(r'(?:^|\.)layers\.(\d+)\.(?:block_sparse_moe|mlp)\.experts$')
_ROUTER_LORA = re.compile(r'(?:^|\.)layers\.\d+\.(?:block_sparse_moe|mlp)\.(?:gate|router)\.lora_(?:A|B|magnitude_vector)\.[^.]+\.(?:weight|bias)$')


def resolve_moe_targets(model: Any, profile: dict[str, Any], topology: str) -> list[str]:
    """Select exact Linear modules; refuse fused/unknown or incomplete layouts.

    This is regular PEFT training. It does not claim expert paging or an
    out-of-core memory envelope.
    """
    import torch

    if topology not in {'moe_hybrid', 'moe_full'} or not profile.get('is_moe'):
        raise ValueError('MoE topology requires a recognized MoE model')
    attention: dict[int, set[str]] = {}
    routers: dict[int, list[str]] = {}
    experts: dict[int, dict[int, set[str]]] = {}
    shared: dict[tuple[int, str], set[str]] = {}
    shared_roots: set[int] = set()
    expert_roots: set[int] = set()
    selected: list[str] = []
    for name, module in model.named_modules():
        if _ROUTER.search(name) and not isinstance(module, torch.nn.Linear):
            raise ValueError('MoE topology requires identifiable Linear routers; fused parameter routers are not supported')
        root = _SHARED_ROOT.search(name)
        if root:
            shared_roots.add(int(root[1]))
        root = _EXPERT_ROOT.search(name)
        if root:
            expert_roots.add(int(root[1]))
        if not isinstance(module, torch.nn.Linear):
            continue
        match = _ATTENTION.search(name)
        if match:
            attention.setdefault(int(match[1]), set()).add(match[2])
            selected.append(name)
            continue
        match = _ROUTER.search(name)
        if match:
            routers.setdefault(int(match[1]), []).append(name)
            selected.append(name)
            continue
        match = _SHARED.search(name)
        if match:
            shared.setdefault((int(match[1]), match[2] or ''), set()).add(match[3])
            selected.append(name)
            continue
        match = _EXPERT.search(name)
        if match and topology == 'moe_full':
            experts.setdefault(int(match[1]), {}).setdefault(int(match[2]), set()).add(match[3])
            selected.append(name)

    config = getattr(model, 'config', None)
    config = getattr(config, 'text_config', config)
    layer_count = getattr(config, 'num_hidden_layers', None)
    if (not isinstance(layer_count, int) or layer_count <= 0 or set(attention) != set(range(layer_count)) or
            any(parts != {'q_proj', 'k_proj', 'v_proj', 'o_proj'} for parts in attention.values())):
        raise ValueError('MoE topology requires complete, identifiable attention projections')
    if (not routers or set(routers) != expert_roots or any(len(items) != 1 for items in routers.values())
            or not set(routers).issubset(attention)):
        raise ValueError('MoE topology requires identifiable Linear routers; fused parameter routers are not supported')
    patterns = ({'w1', 'w2', 'w3'}, {'gate_proj', 'up_proj', 'down_proj'})
    if shared_roots != {layer for layer, _ in shared} or any(parts not in patterns for parts in shared.values()):
        raise ValueError('Shared expert projections are incomplete')
    if topology == 'moe_full':
        count = profile.get('num_experts')
        if not isinstance(count, int) or count < 2:
            raise ValueError('MoE expert count is unavailable')
        for layer in routers:
            found = experts.get(layer, {})
            if set(found) != set(range(count)) or any(parts not in patterns for parts in found.values()):
                raise ValueError('Full MoE LoRA requires separate complete Linear experts in every routed layer; fused expert parameters are not supported')
    return sorted(selected)


def verify_moe_targets(model: Any, expected: list[str]) -> None:
    actual = [name for name, module in model.named_modules() if len(getattr(module, 'lora_A', {}))]
    matched = {target for target in expected if any(name == target or name.endswith('.' + target) for name in actual)}
    if matched != set(expected) or any(not any(name == target or name.endswith('.' + target) for target in expected) for name in actual):
        raise RuntimeError('Installed PEFT stack changed the selected MoE LoRA targets; refusing to train')


def router_adapter_parameters(model: Any) -> dict[int, str]:
    return {id(parameter): name for name, parameter in model.named_parameters()
            if parameter.requires_grad and _ROUTER_LORA.search(name)}


def apply_router_lr_ratio(trainer: Any, model: Any, ratio: float) -> int:
    """Split optimizer groups before scheduling while preserving weight decay."""
    if not 0 < ratio <= 1:
        raise ValueError('Router learning-rate ratio must be greater than 0 and at most 1')
    router = router_adapter_parameters(model)
    if not router:
        raise ValueError('Router learning-rate ratio requested, but no supported trainable router adapters were found')
    if getattr(trainer, 'optimizer', None) is not None or getattr(trainer, 'lr_scheduler', None) is not None:
        raise RuntimeError('Router learning-rate groups must be created before optimizer or scheduler initialization')
    optimizer = trainer.create_optimizer()
    if optimizer is None or getattr(optimizer, 'state', None):
        raise RuntimeError('Cannot safely configure router learning rate on an initialized optimizer')
    groups = []
    found: set[int] = set()
    for group in optimizer.param_groups:
        if 'param_names' in group:
            raise RuntimeError('Named optimizer groups are not supported for router LR scaling')
        normal, routed = [], []
        for parameter in group['params']:
            (routed if id(parameter) in router else normal).append(parameter)
        for parameter in routed:
            if id(parameter) in found:
                raise RuntimeError('Router parameter appeared in more than one optimizer group')
            found.add(id(parameter))
        if normal or not routed:
            groups.append({**group, 'params': normal})
        if routed:
            scaled = {**group, 'params': routed, 'lr': group['lr'] * ratio}
            if 'initial_lr' in scaled:
                scaled['initial_lr'] *= ratio
            groups.append(scaled)
    if found != set(router):
        raise RuntimeError('The optimizer did not include every router adapter parameter')
    optimizer.param_groups[:] = groups
    return len(router)
