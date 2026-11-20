"""Multi-architecture MoE adapter for layer-wise LoRA backpropagation and evaluation."""
from __future__ import annotations

import importlib
import inspect
import math
from typing import Any

import torch

from .helix_checkpoint import MOE_FAMILY_SPECS
from .helix_training import WindowedEvaluationResult, WindowedStepResult, windowed_lora_backward, windowed_lora_evaluate


def _resolve_moe_layer_routers(core, spec) -> tuple[list[tuple[int, torch.nn.Module]], int, int]:
    num_experts = getattr(core.config, spec.experts_attr, None)
    top_k = getattr(core.config, 'num_experts_per_tok', None)
    if type(num_experts) is not int or type(top_k) is not int or not 0 < top_k <= num_experts:
        raise ValueError(f'Router balance requires valid {spec.model_type} expert metadata')
    moe_routers: list[tuple[int, torch.nn.Module]] = []
    for index, layer in enumerate(core.layers):
        mlp = getattr(layer, 'mlp', None)
        if mlp is None or not hasattr(mlp, 'experts'):
            continue
        projection = getattr(getattr(mlp, 'gate', None), 'router_projection', None)
        if not isinstance(projection, torch.nn.Module):
            raise ValueError(f'Router balance requires Helix split {spec.model_type} routers')
        moe_routers.append((index, projection))
    if not moe_routers:
        raise ValueError('Router balance requires at least one sparse MoE layer')
    return moe_routers, num_experts, top_k


def moe_lora_backward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: str | torch.device = 'cpu',
    loss_chunk_tokens: int = 256,
    weight_pager: object | None = None,
    gradient_scale: float = 1.0,
    router_aux_coef: float = 0.0,
    activation_memory_bytes: int = 512 * 1024 * 1024,
    activation_spill_dir: str | None = None,
    prefetch_layers: bool = False,
    prefetch_pinned_memory_bytes: int = 256 * 1024 * 1024,
    neftune_noise_alpha: float = 0.0,
    cu_seqlens: torch.Tensor | None = None,
    _evaluate: bool = False,
) -> WindowedStepResult | WindowedEvaluationResult:
    """Run causal SFT forward/backward or evaluation for supported Hugging Face MoE models."""
    model = model.get_base_model() if callable(getattr(model, 'get_base_model', None)) else model
    config = getattr(model, 'config', None)
    core = getattr(model, 'model', None)
    model_type = getattr(config, 'model_type', None) if config is not None else None
    spec = MOE_FAMILY_SPECS.get(model_type) if model_type is not None else None
    if spec is None or core is None:
        raise ValueError('Only Mixtral, Qwen-MoE, and DeepSeek-MoE models are supported')
    if getattr(config, 'output_router_logits', False):
        raise ValueError('Disable the native router auxiliary loss for windowed training')
    if (isinstance(router_aux_coef, bool) or not isinstance(router_aux_coef, (int, float)) or
            not math.isfinite(router_aux_coef) or not 0 <= router_aux_coef <= 1):
        raise ValueError('router_aux_coef must be between 0 and 1')
    if (isinstance(neftune_noise_alpha, bool) or not isinstance(neftune_noise_alpha, (int, float)) or
            not math.isfinite(neftune_noise_alpha) or not 0 <= neftune_noise_alpha <= 100):
        raise ValueError('neftune_noise_alpha must be between 0 and 100')
    if getattr(config, '_attn_implementation', None) not in {'eager', 'sdpa'}:
        raise ValueError('MoE windowed training requires eager or SDPA attention')
    if input_ids.ndim != 2 or input_ids.dtype != torch.long or attention_mask.shape != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError('Expected matching [batch, sequence] input IDs, attention mask and labels')
    if any(parameter.requires_grad for parameter in core.embed_tokens.parameters()):
        raise ValueError('MoE embeddings must be frozen')
    if any(parameter.device.type != 'cpu' for parameter in core.embed_tokens.parameters()):
        raise ValueError('MoE embeddings must start on CPU')
    if any(getattr(layer, 'gradient_checkpointing', False) for layer in core.layers):
        raise ValueError('Disable model-level gradient checkpointing for windowed replay')
    if input_ids.device.type != 'cpu' or attention_mask.device.type != 'cpu' or labels.device.type != 'cpu':
        raise ValueError('Input IDs, attention mask and labels must start on CPU')
    if not torch.all((attention_mask == 0) | (attention_mask == 1)) or torch.any((attention_mask == 0) & (labels != -100)):
        raise ValueError('Padding must be masked from the loss with label -100')

    device = torch.device(device)
    with torch.no_grad():
        embeddings = core.embed_tokens(input_ids)
        if model.training and neftune_noise_alpha:
            magnitude = neftune_noise_alpha / math.sqrt(embeddings.shape[1] * embeddings.shape[2])
            embeddings = embeddings + torch.empty_like(embeddings).uniform_(-magnitude, magnitude)

    previous_attention = config._attn_implementation
    use_flex = False
    if cu_seqlens is not None:
        if (input_ids.shape[0] != 1 or not torch.all(attention_mask == 1) or
                not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dtype != torch.long or
                cu_seqlens.device.type != 'cpu' or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or
                cu_seqlens[-1].item() != input_ids.shape[1]):
            raise ValueError('Packed Helix training requires one unpadded row and valid conversation boundaries')
        if torch.any(labels[0, cu_seqlens[:-1]] != -100):
            raise ValueError('Packed conversation boundaries and labels do not match input')
        can_use_flex = (
            device.type == 'cuda' and
            getattr(config, 'attention_dropout', 0) == 0 and
            not (model_type in {'deepseek_v2', 'deepseek_v3'} and getattr(config, 'v_head_dim', 16) < 16)
        )
        if can_use_flex:
            from .helix_packing import packed_attention_layout
            positions, mask = packed_attention_layout(cu_seqlens, device, sliding_window=getattr(config, 'sliding_window', None))
            use_flex = True
        else:
            if not _evaluate and device.type != 'cuda':
                raise ValueError('Packed Helix training requires CUDA FlexAttention')
            if getattr(config, 'attention_dropout', 0) and model.training:
                raise ValueError('Packed FlexAttention does not support attention dropout')
            from .helix_packing import packed_causal_4d_mask
            positions, mask = packed_causal_4d_mask(
                cu_seqlens,
                dtype=embeddings.dtype,
                device=device,
                sliding_window=getattr(config, 'sliding_window', None),
            )
    else:
        positions = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        modeling_mod = importlib.import_module(spec.modeling_module)
        sliding = getattr(config, 'sliding_window', None)
        mask_fn = (getattr(modeling_mod, 'create_sliding_window_causal_mask', None)
                   if sliding is not None else None) or modeling_mod.create_causal_mask
        kwargs = {
            'config': config,
            'inputs_embeds': embeddings,
            'attention_mask': attention_mask,
            'past_key_values': None,
            'position_ids': positions,
        }
        if 'cache_position' in inspect.signature(mask_fn).parameters:
            kwargs['cache_position'] = positions[0]
        mask = mask_fn(**kwargs)
        if mask is not None and not isinstance(mask, torch.Tensor):
            raise ValueError('The selected attention backend returned an unsupported mask')

    layers = [*core.layers, core.norm]
    decoder_count = len(core.layers)

    def run_layer(layer, hidden, index):
        if index == decoder_count:
            return layer(hidden)
        current_positions = positions.to(hidden.device)
        core.rotary_emb.to(hidden.device)
        try:
            with torch.no_grad():
                positional = core.rotary_emb(hidden, current_positions)
            arguments = {
                'attention_mask': mask.to(hidden.device) if mask is not None else None,
                'position_ids': current_positions,
                'past_key_values': None,
                'use_cache': False,
                'position_embeddings': positional,
            }
            if 'cache_position' in inspect.signature(layer.forward).parameters:
                arguments['cache_position'] = current_positions[0]
            output = layer(hidden, **arguments)
            return output[0] if isinstance(output, tuple) else output
        finally:
            core.rotary_emb.to('cpu')

    handles = []
    captured = {}
    auxiliary = None
    if router_aux_coef:
        from .performance.moe_balance import balance_loss
        core.config = config
        moe_routers, num_experts, top_k = _resolve_moe_layer_routers(core, spec)
        score_mode = 'sigmoid' if spec.router_style == 'deepseek_v3' else 'softmax'

        def capture(index):
            def hook(_module, _inputs, output):
                if index in captured:
                    raise RuntimeError('Router executed more than once in a window')
                captured[index] = output
            return hook

        def auxiliary(index):
            if index not in moe_layer_set:
                return None
            if index not in captured:
                raise RuntimeError('Router output was not observed in a window')
            scores = captured.pop(index)
            return balance_loss([scores], attention_mask, num_experts, top_k, score_mode=score_mode) * (router_aux_coef / len(moe_routers))

        moe_layer_set = {idx for idx, _ in moe_routers}
        handles = [projection.register_forward_hook(capture(idx)) for idx, projection in moe_routers]

    try:
        if use_flex:
            config._attn_implementation = 'flex_attention'
        if _evaluate:
            return windowed_lora_evaluate(
                layers,
                model.lm_head,
                embeddings,
                labels,
                device=device,
                loss_chunk_tokens=loss_chunk_tokens,
                forward_layer=run_layer,
                weight_pager=weight_pager,
            )
        return windowed_lora_backward(
            layers,
            model.lm_head,
            embeddings,
            labels,
            device=device,
            loss_chunk_tokens=loss_chunk_tokens,
            forward_layer=run_layer,
            weight_pager=weight_pager,
            gradient_scale=gradient_scale,
            auxiliary_loss=auxiliary,
            activation_memory_bytes=activation_memory_bytes,
            activation_spill_dir=activation_spill_dir,
            prefetch_layers=prefetch_layers,
            prefetch_pinned_memory_bytes=prefetch_pinned_memory_bytes,
        )
    finally:
        config._attn_implementation = previous_attention
        for handle in handles:
            handle.remove()
        captured.clear()


mixtral_lora_backward = moe_lora_backward


def moe_lora_evaluate(
    model,
    input_ids,
    attention_mask,
    labels,
    *,
    device: str | torch.device = 'cpu',
    loss_chunk_tokens: int = 256,
    weight_pager=None,
    cu_seqlens: torch.Tensor | None = None,
):
    """Evaluate without changing model modes, gradients, or PyTorch RNG state."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            return moe_lora_backward(
                model,
                input_ids,
                attention_mask,
                labels,
                device=device,
                loss_chunk_tokens=loss_chunk_tokens,
                weight_pager=weight_pager,
                cu_seqlens=cu_seqlens,
                _evaluate=True,
            )
    finally:
        for module, training in modes:
            module.training = training


mixtral_lora_evaluate = moe_lora_evaluate
