"""Auxiliary load balancing for explicitly supported softmax MoE routers."""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F

_ROUTER = re.compile(r'(?:^|\.)layers\.\d+\.(?:block_sparse_moe|mlp)\.(?:gate|router)$')
_SOFTMAX_MODELS = {'mixtral', 'qwen2_moe'}


def balance_loss(logits: list[torch.Tensor], attention_mask: torch.Tensor,
                 num_experts: int, top_k: int, *, score_mode: str = 'softmax') -> torch.Tensor:
    """Return mean Switch-style load balance, excluding padding tokens."""
    if not logits or not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError('Router balance requires logits and a two-dimensional attention mask')
    if type(num_experts) is not int or type(top_k) is not int or not 0 < top_k <= num_experts:
        raise ValueError('Router balance requires a valid expert count and top-k')
    if score_mode not in {'softmax', 'sigmoid'}:
        raise ValueError('Unsupported router score_mode')
    mask = attention_mask.reshape(-1)
    if not torch.all((mask == 0) | (mask == 1)) or not torch.any(mask):
        raise ValueError('Router attention mask must contain valid tokens')
    losses = []
    for scores in logits:
        if not isinstance(scores, torch.Tensor) or scores.ndim not in (2, 3) or scores.shape[-1] != num_experts:
            raise ValueError('Unsupported router logits shape')
        scores = scores.reshape(-1, num_experts)
        if scores.shape[0] != mask.numel():
            raise ValueError('Router logits do not align with the attention mask')
        valid = scores[mask.to(scores.device, dtype=torch.bool)].float()
        if score_mode == 'sigmoid':
            raw = torch.sigmoid(valid)
            probabilities = raw / (raw.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            probabilities = F.softmax(valid, dim=-1)
        indices = torch.topk(probabilities, top_k, dim=-1).indices
        usage = torch.bincount(indices.reshape(-1), minlength=num_experts).to(probabilities.dtype)
        usage = usage / (valid.shape[0] * top_k)
        losses.append(num_experts * torch.sum(usage * probabilities.mean(dim=0)))
    return torch.stack(losses).mean()


class RouterBalance:
    """Capture router logits during a single trainer forward pass."""

    def __init__(self, model: Any, profile: dict[str, Any], coefficient: float):
        if profile.get('architecture') not in _SOFTMAX_MODELS or not profile.get('is_moe'):
            raise ValueError('Router balance supports only recognized Mixtral or Qwen2 MoE softmax routers')
        self.num_experts = profile.get('num_experts')
        self.top_k = profile.get('experts_per_token')
        if type(self.num_experts) is not int or type(self.top_k) is not int or not 0 < self.top_k <= self.num_experts:
            raise ValueError('Router balance requires model expert count and top-k metadata')
        if not 0 < coefficient <= 1:
            raise ValueError('Router balance coefficient must be greater than 0 and at most 1')
        config = getattr(model, 'config', None)
        text_config = getattr(config, 'text_config', config)
        if any(getattr(candidate, 'output_router_logits', False) for candidate in (config, text_config)):
            raise ValueError('Disable native router auxiliary loss before enabling Sytra router balance')
        routers = [(name, module) for name, module in model.named_modules()
                   if _ROUTER.search(name) and len(getattr(module, 'lora_A', {}))]
        if not routers:
            raise ValueError('Router balance requires supported trainable LoRA router modules')
        self.coefficient = coefficient
        self.active = False
        self.captured: dict[str, torch.Tensor] = {}
        self.handles = [module.register_forward_hook(self._hook(name)) for name, module in routers]
        self.names = {name for name, _ in routers}

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            if self.active:
                if name in self.captured:
                    raise RuntimeError('Router ran more than once in one forward; auxiliary loss cannot be verified')
                self.captured[name] = output
        return capture

    def attach(self, trainer: Any) -> None:
        original = trainer.compute_loss

        def compute_loss(model, inputs, return_outputs=False, **kwargs):
            mask = inputs.get('attention_mask')
            if mask is None:
                raise RuntimeError('Router balance requires a non-padding-free attention mask')
            self.captured.clear()
            self.active = True
            try:
                result = original(model, inputs, return_outputs=True, **kwargs)
                self.active = False
                if set(self.captured) != self.names:
                    raise RuntimeError('Not all router outputs were observed; auxiliary loss cannot be verified')
                if any(not isinstance(value, torch.Tensor) for value in self.captured.values()):
                    raise RuntimeError('Router returned unsupported output for auxiliary loss')
                if model.training and torch.is_grad_enabled() and any(not value.requires_grad for value in self.captured.values()):
                    raise RuntimeError('Router logits have no gradient; use non-reentrant checkpointing or disable it')
                if not isinstance(result, tuple) or len(result) != 2:
                    raise RuntimeError('Trainer did not return loss and outputs')
                loss, outputs = result
                auxiliary = balance_loss(list(self.captured.values()), mask, self.num_experts, self.top_k)
                total = loss + self.coefficient * auxiliary
                return (total, outputs) if return_outputs else total
            finally:
                self.active = False
                self.captured.clear()

        trainer.compute_loss = compute_loss

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.captured.clear()
