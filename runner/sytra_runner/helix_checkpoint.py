"""Load local MoE checkpoints (Mixtral, Qwen2-MoE, Qwen3-MoE, DeepSeek-V2/V3) via mmap and quantization."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

from .helix_quant import (
    FrozenInt8Linear,
    Int8PeftLinear,
    detect_checkpoint_quantization,
    load_quantized_linear_from_sources,
)
from .helix_weights import MappedFrozenWeights, TensorSlice

_PEFT_LOADING_LOCK = threading.Lock()
_NF4_FORMAT = {'format': 'sytra_nf4_v1', 'block_size': 64, 'scope': 'attention_experts_head'}
_QUANT_FORMAT = 'sytra_helix_quant_v1'
_BNB_SUFFIXES = (
    '.weight.absmax',
    '.weight.quant_map',
    '.weight.nested_absmax',
    '.weight.nested_quant_map',
    '.SCB',
    '.weight_format',
)


@dataclass(frozen=True)
class MoEFamilySpec:
    model_type: str
    model_cls_name: str
    modeling_module: str
    experts_attr: str
    intermediate_attr: str
    proj_names: tuple[str, str, str]
    router_style: str
    default_attn_targets: tuple[str, ...]
    shared_attr_names: tuple[str, ...]


MOE_FAMILY_SPECS: dict[str, MoEFamilySpec] = {
    'mixtral': MoEFamilySpec(
        model_type='mixtral',
        model_cls_name='MixtralForCausalLM',
        modeling_module='transformers.models.mixtral.modeling_mixtral',
        experts_attr='num_local_experts',
        intermediate_attr='intermediate_size',
        proj_names=('w1', 'w2', 'w3'),
        router_style='mixtral',
        default_attn_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
        shared_attr_names=(),
    ),
    'qwen2_moe': MoEFamilySpec(
        model_type='qwen2_moe',
        model_cls_name='Qwen2MoeForCausalLM',
        modeling_module='transformers.models.qwen2_moe.modeling_qwen2_moe',
        experts_attr='num_experts',
        intermediate_attr='moe_intermediate_size',
        proj_names=('gate_proj', 'up_proj', 'down_proj'),
        router_style='qwen',
        default_attn_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
        shared_attr_names=('shared_expert', 'shared_expert_gate'),
    ),
    'qwen3_moe': MoEFamilySpec(
        model_type='qwen3_moe',
        model_cls_name='Qwen3MoeForCausalLM',
        modeling_module='transformers.models.qwen3_moe.modeling_qwen3_moe',
        experts_attr='num_experts',
        intermediate_attr='moe_intermediate_size',
        proj_names=('gate_proj', 'up_proj', 'down_proj'),
        router_style='qwen',
        default_attn_targets=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
        shared_attr_names=(),
    ),
    'deepseek_v2': MoEFamilySpec(
        model_type='deepseek_v2',
        model_cls_name='DeepseekV2ForCausalLM',
        modeling_module='transformers.models.deepseek_v2.modeling_deepseek_v2',
        experts_attr='n_routed_experts',
        intermediate_attr='moe_intermediate_size',
        proj_names=('gate_proj', 'up_proj', 'down_proj'),
        router_style='deepseek_v2',
        default_attn_targets=('q_a_proj', 'q_b_proj', 'q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'),
        shared_attr_names=('shared_experts',),
    ),
    'deepseek_v3': MoEFamilySpec(
        model_type='deepseek_v3',
        model_cls_name='DeepseekV3ForCausalLM',
        modeling_module='transformers.models.deepseek_v3.modeling_deepseek_v3',
        experts_attr='n_routed_experts',
        intermediate_attr='moe_intermediate_size',
        proj_names=('gate_proj', 'up_proj', 'down_proj'),
        router_style='deepseek_v3',
        default_attn_targets=('q_a_proj', 'q_b_proj', 'q_proj', 'kv_a_proj_with_mqa', 'kv_b_proj', 'o_proj'),
        shared_attr_names=('shared_experts',),
    ),
}
SUPPORTED_MOE_TYPES = frozenset(MOE_FAMILY_SPECS)


class _SeparateExpert(torch.nn.Module):
    """Single routed MoE expert supporting both Mixtral (w1/w2/w3) and SwiGLU (gate/up/down) naming."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, proj_names: tuple[str, str, str]):
        super().__init__()
        from transformers.activations import ACT2FN
        if proj_names == ('w1', 'w2', 'w3'):
            gate_name, down_name, up_name = 'w1', 'w2', 'w3'
        else:
            gate_name, up_name, down_name = proj_names
        self._gate_name, self._up_name, self._down_name = gate_name, up_name, down_name
        setattr(self, gate_name, torch.nn.Linear(hidden_size, intermediate_size, bias=False))
        setattr(self, down_name, torch.nn.Linear(intermediate_size, hidden_size, bias=False))
        setattr(self, up_name, torch.nn.Linear(hidden_size, intermediate_size, bias=False))
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = getattr(self, self._gate_name)(value)
        up = getattr(self, self._up_name)(value)
        return getattr(self, self._down_name)(self.act_fn(gate) * up)


class _HelixRouter(torch.nn.Module):
    """Softmax top-k router exposing ``router_projection`` for LoRA adaptation."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, *, normalize_weights: bool = True):
        super().__init__()
        self.router_projection = torch.nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.normalize_weights = normalize_weights

    def forward(self, hidden: torch.Tensor):
        flat = hidden.reshape(-1, hidden.shape[-1])
        if flat.dtype != self.router_projection.weight.dtype:
            flat = flat.to(self.router_projection.weight.dtype)
        logits = self.router_projection(flat)
        probabilities = torch.nn.functional.softmax(logits.float(), dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return probabilities, weights, indices


class _HelixLogitRouter(torch.nn.Module):
    """Shape-preserving router projection for DeepSeek-V2 MoE blocks."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.router_projection = torch.nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = hidden if hidden.dtype == self.router_projection.weight.dtype else hidden.to(self.router_projection.weight.dtype)
        return self.router_projection(value).float()


class _HelixDeepseekV3Router(torch.nn.Module):
    """DeepSeek-V3 group-limited sigmoid router preserving ``e_score_correction_bias``."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.router_projection = torch.nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.register_buffer('e_score_correction_bias', torch.zeros(config.n_routed_experts))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        flat = hidden.view(-1, self.hidden_size)
        if flat.dtype != self.router_projection.weight.dtype:
            flat = flat.to(self.router_projection.weight.dtype)
        return self.router_projection(flat).float()


class _SeparateExperts(torch.nn.ModuleList):
    """Per-expert CPU container whose active experts move to compute device only after routing."""

    _helix_deferred_experts = True

    def __init__(self, modules):
        super().__init__(modules)
        self.pager = None
        self.loaded = set()
        self.last_active_experts = 0
        self.total_expert_loads = 0

    def _apply(self, fn, recurse=True):
        return self

    def _helix_release_experts(self):
        for index in self.loaded:
            expert = self[index]
            self.pager.restore(expert)
            expert.to('cpu')
        self.loaded.clear()

    def forward(self, hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        active = 0
        for index, expert in enumerate(self):
            tokens, slots = torch.where(indices == index)
            if tokens.numel():
                active += 1
                if self.pager is None:
                    raise RuntimeError('Expert pager must be attached before executing the model')
                if index not in self.loaded:
                    self.pager.prepare(expert, hidden.device)
                    expert.to(hidden.device)
                    self.loaded.add(index)
                    self.total_expert_loads += 1
                values = expert(hidden[tokens]) * weights[tokens, slots, None]
                output.index_add_(0, tokens, values.to(output.dtype))
        self.last_active_experts = active
        return output


def _patch_deepseek_v2_forward(block: torch.nn.Module) -> None:
    base_cls = block.__class__
    if getattr(base_cls, '_helix_v2_moe_patched', False):
        return

    class _HelixDeepseekV2Moe(base_cls):
        _helix_v2_moe_patched = True

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            residuals = hidden_states
            orig_shape = hidden_states.shape
            router_logits = self.gate(hidden_states)
            topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            routed = self.experts(flat, topk_indices, topk_weights).view(*orig_shape)
            shared = getattr(self, 'shared_experts', None)
            return routed + shared(residuals) if shared is not None else routed

    block.__class__ = _HelixDeepseekV2Moe


def _adapt_moe_experts(model, config, spec: MoEFamilySpec) -> bool:
    """Replace fused/discrete MoE blocks with on-demand ``_SeparateExperts`` and LoRA routers."""
    from accelerate import init_empty_weights
    num_experts = int(getattr(config, spec.experts_attr))
    intermediate = int(getattr(config, spec.intermediate_attr))
    replaced = False
    with init_empty_weights():
        for layer in model.model.layers:
            mlp = getattr(layer, 'mlp', None)
            if mlp is None or not hasattr(mlp, 'experts'):
                continue
            replaced = True
            mlp.experts = _SeparateExperts([
                _SeparateExpert(config.hidden_size, intermediate, config.hidden_act, spec.proj_names)
                for _ in range(num_experts)
            ])
            if spec.router_style == 'mixtral':
                mlp.gate = _HelixRouter(config.hidden_size, num_experts, config.num_experts_per_tok, normalize_weights=True)
            elif spec.router_style == 'qwen':
                norm = bool(getattr(config, 'norm_topk_prob', False))
                mlp.gate = _HelixRouter(config.hidden_size, num_experts, config.num_experts_per_tok, normalize_weights=norm)
            elif spec.router_style == 'deepseek_v2':
                _patch_deepseek_v2_forward(mlp)
                mlp.gate = _HelixLogitRouter(config.hidden_size, num_experts)
            elif spec.router_style == 'deepseek_v3':
                mlp.gate = _HelixDeepseekV3Router(config)
    return replaced


def _is_bnb_aux_key(key: str) -> bool:
    return '.weight.quant_state.bitsandbytes__' in key or any(key.endswith(suffix) for suffix in _BNB_SUFFIXES)


def _checkpoint_sources(directory: Path) -> dict[str, tuple[Path, str]]:
    root = directory.resolve(strict=True)
    index = root / 'model.safetensors.index.json'
    single = root / 'model.safetensors'
    if index.is_file():
        data = json.loads(index.read_text(encoding='utf-8'))
        weight_map = data.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Safetensors index must contain a non-empty weight_map')
        sources = {}
        for key, filename in weight_map.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise ValueError('Invalid safetensors index entry')
            path = (root / filename).resolve(strict=True)
            if not path.is_relative_to(root) or path.suffix != '.safetensors' or not path.is_file():
                raise ValueError('Safetensors shard must be a file inside the checkpoint directory')
            sources[key] = (path, key)
        return sources
    if single.is_file():
        single = single.resolve(strict=True)
        if not single.is_relative_to(root):
            raise ValueError('Safetensors checkpoint must be inside the checkpoint directory')
        with safe_open(str(single), framework='pt', device='cpu') as file:
            return {key: (single, key) for key in file.keys()}
    raise ValueError('Local checkpoint requires model.safetensors or its sharded index')


def _candidate_linear_prefixes(module_name: str, spec: MoEFamilySpec) -> list[str]:
    prefixes = [module_name]
    if module_name.endswith('.mlp.gate.router_projection'):
        base = module_name[:-len('.router_projection')]
        prefixes.append(base)
        if spec.model_type == 'mixtral':
            prefixes.append(re.sub(r'^(model\.layers\.\d+)\.mlp\.gate$', r'\1.block_sparse_moe.gate', base))
    elif spec.model_type == 'mixtral' and '.mlp.' in module_name:
        prefixes.append(re.sub(r'^(model\.layers\.\d+)\.mlp\.', r'\1.block_sparse_moe.', module_name))
    return list(dict.fromkeys(prefixes))


def _mark_bnb_keys(sources: Mapping[str, Any], prefix: str, consumed: set[str]) -> None:
    weight_key = f'{prefix}.weight'
    if weight_key in sources:
        consumed.add(weight_key)
    for key in sources:
        if key.startswith(f'{prefix}.weight.') or key in {f'{prefix}.SCB', f'{prefix}.weight_format'}:
            consumed.add(key)


def _resolve_expert_slice(
    param_name: str,
    sources: Mapping[str, tuple[Path, str]],
    config,
    spec: MoEFamilySpec,
) -> tuple[TensorSlice, str] | None:
    match = re.fullmatch(r'(model\.layers\.\d+)\.mlp\.experts\.(\d+)\.([A-Za-z0-9_]+)\.weight', param_name)
    if not match:
        return None
    layer_prefix, expert_idx_str, proj = match.groups()
    expert_idx = int(expert_idx_str)
    inter_size = int(getattr(config, spec.intermediate_attr))
    gate_name, up_name, down_name = (('w1', 'w3', 'w2') if spec.proj_names == ('w1', 'w2', 'w3') else spec.proj_names)
    gu_key = f'{layer_prefix}.mlp.experts.gate_up_proj'
    dn_key = f'{layer_prefix}.mlp.experts.down_proj'
    if proj == gate_name and gu_key in sources:
        return TensorSlice(sources[gu_key][0], gu_key, expert_idx, 0, inter_size), gu_key
    if proj == up_name and gu_key in sources:
        return TensorSlice(sources[gu_key][0], gu_key, expert_idx, inter_size, 2 * inter_size), gu_key
    if proj == down_name and dn_key in sources:
        return TensorSlice(sources[dn_key][0], dn_key, expert_idx), dn_key
    return None


def _resolve_param_source(
    param_name: str,
    sources: Mapping[str, tuple[Path, str]],
    config,
    spec: MoEFamilySpec,
) -> tuple[tuple[Path, str] | TensorSlice, str] | None:
    candidates = [param_name]
    if '.mlp.gate.router_projection.' in param_name:
        replaced = param_name.replace('.mlp.gate.router_projection.', '.mlp.gate.')
        candidates.append(replaced)
        if spec.model_type == 'mixtral':
            candidates.append(re.sub(r'^(model\.layers\.\d+)\.mlp\.gate\.', r'\1.block_sparse_moe.gate.', replaced))
    elif spec.model_type == 'mixtral':
        candidates.append(re.sub(r'^(model\.layers\.\d+)\.mlp\.', r'\1.block_sparse_moe.', param_name))
    for key in candidates:
        if key in sources:
            return sources[key], key
    return _resolve_expert_slice(param_name, sources, config, spec)


def _load_source_value(source: tuple[Path, str] | TensorSlice) -> torch.Tensor:
    if isinstance(source, TensorSlice):
        with safe_open(str(source.path), framework='pt', device='cpu') as file:
            tensor = file.get_slice(source.key)[source.dim0_index]
            if source.row_start is not None or source.row_end is not None:
                tensor = tensor[source.row_start:source.row_end]
            return tensor.contiguous()
    path, key = source
    with safe_open(str(path), framework='pt', device='cpu') as file:
        return file.get_tensor(key)


def _should_quantize_linear(module_name: str, *, strict_mixtral_nf4: bool, num_layers: int) -> bool:
    if re.search(r'^model\.layers\.\d+\.self_attn\.[^.]+$', module_name):
        return True
    if re.search(r'^model\.layers\.\d+\.mlp\.experts\.\d+\.(?:w1|w2|w3|gate_proj|up_proj|down_proj)$', module_name):
        return True
    if module_name == 'lm_head':
        return True
    if not strict_mixtral_nf4:
        if re.search(r'^model\.layers\.\d+\.mlp\.(?:shared_experts?\.|)(?:gate_proj|up_proj|down_proj)$', module_name):
            return True
    return False


def helix_target_modules(
    model_or_config,
    topology: str = 'attention',
    requested: Sequence[str] | str | None = None,
) -> tuple[str, ...] | str:
    """Resolve PEFT ``target_modules`` safely for attention, Hybrid (shared+router), or Full MoE."""
    if isinstance(requested, str):
        return requested
    config = getattr(model_or_config, 'config', model_or_config)
    model_type = getattr(config, 'model_type', 'mixtral')
    spec = MOE_FAMILY_SPECS.get(model_type)
    base_targets = list(requested or ())
    if not base_targets and spec is not None:
        base_targets = list(spec.default_attn_targets)
        if model_type in {'deepseek_v2', 'deepseek_v3'}:
            excluded = {'q_proj'} if getattr(config, 'q_lora_rank', None) else {'q_a_proj', 'q_b_proj'}
            base_targets = [target for target in base_targets if target not in excluded]
    elif model_or_config is not model_or_config.__class__ and hasattr(model_or_config, 'named_modules'):
        present = {name.rsplit('.', 1)[-1] for name, m in model_or_config.named_modules()
                   if isinstance(m, (torch.nn.Linear, FrozenInt8Linear)) or m.__class__.__name__ == 'FrozenNF4Linear'}
        if base_targets and not (set(base_targets) & present) and spec is not None:
            base_targets = [t for t in spec.default_attn_targets if t in present]
    if topology == 'attention':
        return tuple(dict.fromkeys(base_targets))
    if topology == 'moe_full':
        extra = ['router_projection', *(spec.proj_names if spec else ('w1', 'w2', 'w3'))]
        if spec and spec.shared_attr_names:
            extra.extend(('gate_proj', 'up_proj', 'down_proj'))
        return tuple(dict.fromkeys([*base_targets, *extra]))
    if topology == 'moe_hybrid':
        if spec is None or not spec.shared_attr_names:
            return tuple(dict.fromkeys([*base_targets, 'router_projection']))
        attn_alt = '|'.join(re.escape(t) for t in base_targets)
        return (
            rf'(?:.*\.self_attn\.(?:{attn_alt})'
            r'|.*\.mlp\.gate\.router_projection'
            r'|.*\.mlp\.shared_experts?\.(?:gate_proj|up_proj|down_proj)'
            r'|model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj))$'
        )
    raise ValueError(f'Unsupported Helix MoE topology: {topology!r}')


def _verify_matched_targets(model, target_modules: Sequence[str] | str) -> None:
    lora_module_names = [name for name, module in model.named_modules() if getattr(module, 'lora_A', None)]
    if not lora_module_names or not any(p.requires_grad and 'lora_' in n for n, p in model.named_parameters()):
        raise ValueError('No LoRA parameters were attached')
    if isinstance(target_modules, str):
        return
    leaves = {name.rsplit('.', 1)[-1] for name in lora_module_names}
    missing = {t for t in target_modules if t not in leaves and not any(re.fullmatch(t, n) for n in lora_module_names)}
    if missing:
        raise ValueError(f'LoRA targets did not match supported modules: {sorted(missing)}')


def load_mapped_moe(
    directory: str | Path,
    *,
    rank: int = 8,
    alpha: int = 16,
    target_modules: tuple[str, ...] | str = ('q_proj', 'k_proj', 'v_proj', 'o_proj'),
    dropout: float = 0.0,
    use_rslora: bool = False,
    quant_cache_dir: str | Path | None = None,
    quant_bits: int | None = None,
):
    """Create a CPU MoE model with mapped or pre-quantized frozen weights and trainable LoRA adapters."""
    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    import peft.tuners.lora.model as lora_model
    import transformers
    from .helix_nf4 import FrozenNF4Linear
    from .helix_nf4_peft import NF4PeftLinear

    root = Path(directory).resolve(strict=True)
    config_json = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    ckpt_quant = detect_checkpoint_quantization(config_json)
    sources = _checkpoint_sources(root)
    config = transformers.AutoConfig.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    model_type = getattr(config, 'model_type', None)
    spec = MOE_FAMILY_SPECS.get(model_type)
    if spec is None or getattr(config, 'tie_word_embeddings', False):
        raise ValueError('Mapped Helix loading requires an untied Mixtral, Qwen-MoE, or DeepSeek-MoE checkpoint')
    if ckpt_quant is not None and quant_bits is not None and ckpt_quant['bits'] != quant_bits:
        raise ValueError(f"Checkpoint is {ckpt_quant['bits']}-bit quantized and does not match requested {quant_bits}-bit mode")
    effective_bits = ckpt_quant['bits'] if ckpt_quant is not None else (4 if quant_cache_dir is not None and quant_bits is None else quant_bits)
    if effective_bits not in (None, 16, 4, 8):
        raise ValueError(f'Unsupported Helix quantization width: {effective_bits}')

    config._attn_implementation = 'eager'
    config.use_cache = False
    model_cls = getattr(transformers, spec.model_cls_name)
    with init_empty_weights():
        model = model_cls(config)
    split = _adapt_moe_experts(model, config, spec)
    num_experts = int(getattr(config, spec.experts_attr))
    if quant_cache_dir is not None and model_type == 'mixtral' and not split:
        raise ValueError('NF4 loading requires separate Mixtral expert tensors')

    consumed: set[str] = set()
    linear_names = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)]
    for linear_name in linear_names:
        for prefix in _candidate_linear_prefixes(linear_name, spec):
            qmodule = load_quantized_linear_from_sources(sources, prefix)
            if qmodule is not None:
                _mark_bnb_keys(sources, prefix, consumed)
                parent, _, leaf = linear_name.rpartition('.')
                container = model.get_submodule(parent) if parent else model
                setattr(container, leaf, qmodule)
                break

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    mapped_sources: dict[str, tuple[Path, str] | TensorSlice] = {}
    staged_params: list[tuple[str, torch.nn.Parameter, torch.Tensor]] = []
    for name, parameter in parameters.items():
        if '.weight_scales' in name or '.packed_weight' in name:
            continue
        resolved = _resolve_param_source(name, sources, config, spec)
        if resolved is None:
            raise ValueError(f'Checkpoint layout does not match {spec.model_type}: missing parameter {name}')
        src_obj, consumed_key = resolved
        consumed.add(consumed_key)
        tensor = _load_source_value(src_obj)
        if tensor.shape != parameter.shape or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f'Unsupported checkpoint tensor shape or dtype: {name}')
        staged_params.append((name, parameter, tensor))
        mapped_sources[name] = src_obj

    staged_buffers: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for bname, buf in buffers.items():
        if bname.endswith('.packed_weight') or bname.endswith('.weight_scales') or bname.endswith('.weight_codebook') or bname.endswith('.nested_absmax') or bname.endswith('.nested_quant_map') or bname.endswith('.weight_i8'):
            continue
        if bname in sources:
            consumed.add(bname)
            tensor = _load_source_value(sources[bname]).to(dtype=buf.dtype)
            if tensor.shape != buf.shape:
                raise ValueError(f'Checkpoint buffer shape mismatch: {bname}')
            staged_buffers.append((bname, buf, tensor))

    unexpected = set(sources) - consumed
    if unexpected:
        raise ValueError(f'Checkpoint layout does not match {spec.model_type}: unexpected={sorted(unexpected)[:3]}')

    for name, _parameter, tensor in staged_params:
        parent, _, leaf = name.rpartition('.')
        module = model.get_submodule(parent) if parent else model
        setattr(module, leaf, torch.nn.Parameter(tensor, requires_grad=False))
    for bname, _buf, tensor in staged_buffers:
        parent, _, leaf = bname.rpartition('.')
        module = model.get_submodule(parent) if parent else model
        module.register_buffer(leaf, tensor)

    quant_sources: list[tuple[str, str]] = []
    strict_mixtral_nf4 = model_type == 'mixtral' and quant_cache_dir is not None and ckpt_quant is None
    if effective_bits in (4, 8):
        cache = Path(quant_cache_dir).resolve() if quant_cache_dir is not None else None
        if cache is not None and (cache == root or cache in root.parents or root in cache.parents):
            raise ValueError('NF4 cache must be separate from the base checkpoint directory')
        targets = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Linear) and _should_quantize_linear(name, strict_mixtral_nf4=strict_mixtral_nf4, num_layers=config.num_hidden_layers)
        ]
        if strict_mixtral_nf4:
            expected_count = config.num_hidden_layers * (4 + 3 * num_experts) + 1
            if len(targets) != expected_count:
                raise ValueError('NF4 loading requires complete separate attention and expert projections')
        for name, mod in targets:
            w = mod.weight.detach()
            bias = mod.bias.detach() if mod.bias is not None else None
            if effective_bits == 4:
                if cache is not None:
                    from .helix_nf4_store import cached_nf4_matrix
                    quantized = cached_nf4_matrix(w, cache)
                    quant_sources.append((name, quantized.sytra_source_digest))
                else:
                    quantized = FrozenNF4Linear.from_float(w, bias=bias)
                    quant_sources.append((name, hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
            else:
                quantized = FrozenInt8Linear.from_float(w, bias=bias)
                quant_sources.append((name, hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
            if bias is not None and quantized.bias is None:
                quantized.register_parameter('bias', torch.nn.Parameter(bias.contiguous(), requires_grad=False))
            parent, _, leaf = name.rpartition('.')
            container = model.get_submodule(parent) if parent else model
            setattr(container, leaf, quantized)
            mapped_sources.pop(name + '.weight', None)

    if isinstance(target_modules, (list, tuple)) and not isinstance(target_modules, str):
        present_leaves = {name.rsplit('.', 1)[-1] for name, m in model.named_modules()
                          if isinstance(m, (torch.nn.Linear, FrozenNF4Linear, FrozenInt8Linear))}
        if (tuple(target_modules) == ('q_proj', 'k_proj', 'v_proj', 'o_proj') and
                spec.default_attn_targets != ('q_proj', 'k_proj', 'v_proj', 'o_proj') and
                not set(target_modules).issubset(present_leaves)):
            target_modules = tuple(t for t in spec.default_attn_targets if t in present_leaves)

    pager = MappedFrozenWeights(model, mapped_sources, allow_empty=not mapped_sources)
    pager.register_quant_buffers(model)
    for module in model.modules():
        if isinstance(module, _SeparateExperts):
            module.pager = pager

    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules) if isinstance(target_modules, (list, tuple)) else target_modules,
        use_rslora=use_rslora,
        task_type='CAUSAL_LM',
        bias='none',
    )
    has_nf4 = any(isinstance(m, FrozenNF4Linear) for m in model.modules())
    has_int8 = any(isinstance(m, FrozenInt8Linear) for m in model.modules())
    if has_nf4 or has_int8:
        mapping = {}
        if has_nf4:
            mapping[FrozenNF4Linear] = NF4PeftLinear
        if has_int8:
            mapping[FrozenInt8Linear] = Int8PeftLinear
        lora._register_custom_module(mapping)

    original_type = config.model_type
    with _PEFT_LOADING_LOCK:
        bnb_available, bnb_4bit = lora_model.is_bnb_available, lora_model.is_bnb_4bit_available
        try:
            if split:
                config.model_type = 'sytra_mixtral_split'
            lora_model.is_bnb_available = lambda: False
            lora_model.is_bnb_4bit_available = lambda: False
            model = get_peft_model(model, lora)
        finally:
            lora_model.is_bnb_available = bnb_available
            lora_model.is_bnb_4bit_available = bnb_4bit
            config.model_type = original_type

    _verify_matched_targets(model, target_modules)

    if strict_mixtral_nf4:
        model.sytra_nf4_format = dict(_NF4_FORMAT)
        model.sytra_nf4_source_digest = hashlib.sha256(
            json.dumps(sorted(quant_sources), separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        model.sytra_quant_format = None
    elif effective_bits in (4, 8):
        model.sytra_nf4_format = None
        model.sytra_nf4_source_digest = None
        bnb_keys = sorted(k for k in sources if _is_bnb_aux_key(k))
        digest_payload = {'quant_sources': sorted(quant_sources), 'bnb_keys': bnb_keys, 'bits': effective_bits}
        model.sytra_quant_format = {
            'format': _QUANT_FORMAT,
            'model_type': model_type,
            'quant_bits': effective_bits,
            'ckpt_method': ckpt_quant['method'] if ckpt_quant else 'sytra',
            'source_digest': hashlib.sha256(json.dumps(digest_payload, separators=(',', ':')).encode('utf-8')).hexdigest(),
        }
    else:
        model.sytra_nf4_format = None
        model.sytra_nf4_source_digest = None
        model.sytra_quant_format = None
    model.train()
    return model, pager


load_mapped_mixtral = load_mapped_moe


def save_mapped_adapter(model, directory: str | Path) -> None:
    """Save the PEFT adapter together with Helix's explicit base quantization marker."""
    target = Path(directory)
    nf4_marker = target / 'sytra-helix-nf4.json'
    quant_marker = target / 'sytra-helix-quant.json'
    quantization = getattr(model, 'sytra_nf4_format', None)
    generic_quant = getattr(model, 'sytra_quant_format', None)
    if quantization is None and nf4_marker.exists():
        raise ValueError('Output already contains an NF4 adapter marker')
    if quantization is not None and quantization != _NF4_FORMAT:
        raise ValueError('Unsupported Helix NF4 configuration')
    digest = getattr(model, 'sytra_nf4_source_digest', None)
    if quantization is not None and (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('Missing Helix NF4 source fingerprint')
    model.save_pretrained(str(target), safe_serialization=True, save_embedding_layers=False)
    if quantization is not None:
        nf4_marker.write_text(json.dumps({**quantization, 'source_digest': digest}, sort_keys=True, indent=2), encoding='utf-8')
    elif generic_quant is not None:
        quant_marker.write_text(json.dumps(generic_quant, sort_keys=True, indent=2), encoding='utf-8')


def load_mapped_adapter(
    directory: str | Path,
    adapter_directory: str | Path,
    *,
    quant_cache_dir: str | Path | None = None,
    quant_bits: int | None = None,
):
    """Load a Helix PEFT adapter onto the matching mapped local base model."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from .adapters import adapter_metadata

    base = Path(directory).resolve(strict=True)
    metadata = adapter_metadata(adapter_directory)
    nf4_marker = Path(adapter_directory) / 'sytra-helix-nf4.json'
    quant_marker = Path(adapter_directory) / 'sytra-helix-quant.json'
    saved_digest = None
    saved_quant_meta = None
    if nf4_marker.exists():
        saved = json.loads(nf4_marker.read_text(encoding='utf-8'))
        if not isinstance(saved, dict) or {key: saved.get(key) for key in _NF4_FORMAT} != _NF4_FORMAT or set(saved) != {*_NF4_FORMAT, 'source_digest'}:
            raise ValueError('Unsupported Helix NF4 adapter marker')
        saved_digest = saved.get('source_digest')
        if not isinstance(saved_digest, str) or not re.fullmatch(r'[0-9a-f]{64}', saved_digest):
            raise ValueError('Invalid Helix NF4 source fingerprint')
        if quant_cache_dir is None:
            raise ValueError('This adapter requires Helix NF4 loading')
    elif quant_marker.exists():
        saved_quant_meta = json.loads(quant_marker.read_text(encoding='utf-8'))
        if not isinstance(saved_quant_meta, dict) or saved_quant_meta.get('format') != _QUANT_FORMAT:
            raise ValueError('Unsupported Helix quantized adapter marker')
        if quant_bits is None:
            quant_bits = saved_quant_meta.get('quant_bits')
        elif quant_bits != saved_quant_meta.get('quant_bits'):
            raise ValueError('Helix adapter quantization width does not match requested quant_bits')
    saved_base = Path(metadata['base_model_name_or_path']).resolve(strict=True)
    if saved_base != base:
        raise ValueError('Adapter base model does not match the requested checkpoint')
    if metadata.get('use_dora') or metadata.get('bias', 'none') != 'none':
        raise ValueError('Only ordinary LoRA and rsLoRA adapters are supported by Helix')
    targets = metadata['target_modules']
    if isinstance(targets, list):
        if any(not isinstance(name, str) or not name for name in targets):
            raise ValueError('Adapter target_modules must be a list of module names')
        resolved_targets: tuple[str, ...] | str = tuple(targets)
    elif isinstance(targets, str) and targets:
        resolved_targets = targets
    else:
        raise ValueError('Adapter target_modules must be a module name list or regex pattern')

    model, pager = load_mapped_moe(
        base,
        rank=metadata['r'],
        alpha=metadata['lora_alpha'],
        target_modules=resolved_targets,
        dropout=metadata.get('lora_dropout', 0),
        use_rslora=metadata.get('use_rslora', False),
        quant_cache_dir=quant_cache_dir,
        quant_bits=quant_bits,
    )
    if saved_digest is not None and model.sytra_nf4_source_digest != saved_digest:
        raise ValueError('Helix NF4 adapter source weights changed')
    if saved_quant_meta is not None and model.sytra_quant_format != saved_quant_meta:
        raise ValueError('Helix quantized adapter source configuration changed')

    actual = load_file(str(Path(adapter_directory) / 'adapter_model.safetensors'), device='cpu')
    expected = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(actual) != set(expected) or any(actual[key].shape != expected[key].shape for key in expected):
        raise ValueError('Adapter tensors do not exactly match the selected Helix topology')
    config = model.config
    original_type = config.model_type
    try:
        if any(isinstance(module, _SeparateExperts) for module in model.modules()):
            config.model_type = 'sytra_mixtral_split'
        result = set_peft_model_state_dict(model, actual)
    finally:
        config.model_type = original_type
    if result.unexpected_keys or any('lora_' in key for key in result.missing_keys):
        raise ValueError('PEFT did not restore every Helix adapter tensor')
    return model, pager
