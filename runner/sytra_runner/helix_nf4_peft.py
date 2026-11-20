"""PEFT LoRA dispatch for frozen Sytra NF4 linear modules."""
from peft.tuners.lora.layer import Linear

from .helix_nf4 import FrozenNF4Linear
from .helix_triton_ops import fused_nf4_lora_linear, triton_kernels_available


class NF4PeftLinear(Linear):
    def _get_in_out_features(self, module):
        if not isinstance(module, FrozenNF4Linear):
            raise TypeError('NF4 PEFT adapter requires a frozen NF4 linear base')
        return module.in_features, module.out_features

    def forward(self, x, *args, **kwargs):
        base = self.get_base_layer()
        active = list(getattr(self, 'active_adapters', ()))
        if (not args and not kwargs and isinstance(base, FrozenNF4Linear) and not getattr(self, 'merged', False) and
                not getattr(self, 'disable_adapters', False) and len(active) == 1 and x.device.type == 'cuda' and
                triton_kernels_available(x.device)):
            adapter = active[0]
            if (adapter in self.lora_A and adapter in self.lora_B and
                    not getattr(self, 'use_dora', {}).get(adapter, False) and
                    self.lora_B[adapter].bias is None):
                dropout_mod = self.lora_dropout[adapter] if adapter in self.lora_dropout else None
                dropout_p = float(getattr(dropout_mod, 'p', 0.0)) if dropout_mod is not None else 0.0
                out = fused_nf4_lora_linear(
                    x,
                    self.lora_A[adapter].weight,
                    self.lora_B[adapter].weight,
                    base.packed_weight,
                    base.resolve_scales(),
                    base.resolve_codebook(x.device),
                    base.out_features,
                    base.in_features,
                    base.block_size,
                    low_first=base.low_first,
                    lora_scale=float(self.scaling[adapter]),
                    dropout_p=dropout_p,
                    training=self.training,
                )
                if base.bias is not None:
                    out = out + base.bias.to(dtype=out.dtype)
                return out
        return super().forward(x, *args, **kwargs)

    def merge(self, *args, **kwargs):
        raise RuntimeError('Merging an adapter into NF4 weights is not supported')

    def unmerge(self):
        raise RuntimeError('Merging an adapter into NF4 weights is not supported')
