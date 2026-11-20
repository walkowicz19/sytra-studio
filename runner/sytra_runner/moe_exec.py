"""MergeKit MoE launcher that fixes shared-expert loader initialization.

MergeKit 0.0.5.1's ``initialize_io()`` warms up tensor loaders only for
``[base_model] + [e.source_model for e in config.experts]``, omitting
``config.shared_experts``. When a shared expert points at a checkpoint distinct
from ``base_model`` and the routed ``experts``, ``loaders.get(...)`` returns
``None`` and crashes both ``qwen.py`` and ``deepseek.py``.
"""
from __future__ import annotations


def patch_mergekit_shared_loaders() -> None:
    """Wrap ``initialize_io`` so ``config.shared_experts`` loaders are initialized."""
    import mergekit.moe.common as moe_common

    orig_init = getattr(moe_common.initialize_io, "_sytra_orig", moe_common.initialize_io)

    def _patched_initialize_io(config, out_path, merge_options):
        loaders, base_loader, writer = orig_init(config, out_path, merge_options)
        for shared in getattr(config, "shared_experts", None) or []:
            model = shared.source_model
            if model not in loaders:
                loaders[model] = model.lazy_loader(
                    cache_dir=merge_options.transformers_cache,
                    lazy_unpickle=merge_options.lazy_unpickle,
                )
        return loaders, base_loader, writer

    _patched_initialize_io._sytra_orig = orig_init  # type: ignore[attr-defined]
    moe_common.initialize_io = _patched_initialize_io

    for mod_name in ("mergekit.moe.deepseek", "mergekit.moe.mixtral", "mergekit.moe.qwen"):
        try:
            mod = __import__(mod_name, fromlist=["initialize_io"])
        except ImportError:
            continue
        if hasattr(mod, "initialize_io"):
            mod.initialize_io = _patched_initialize_io


def main() -> None:
    patch_mergekit_shared_loaders()
    from mergekit.scripts.moe import main as mergekit_moe_main

    mergekit_moe_main()


if __name__ == "__main__":
    main()
