"""Compatibility guard for the removed placeholder MoE engine."""
from __future__ import annotations


class MoEInferenceEngine:
    """Fail explicitly instead of returning placeholder tokens.

    Use ``runner/scripts/serve_model.py`` to launch llama.cpp, vLLM, or a
    forward-verified ``sytra-engine`` adapter. The native shared streaming core
    exists, but each model still requires architecture-specific kernels and
    reference correctness tests before token serving is enabled.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Sytra's placeholder MoEInferenceEngine was removed because it did not perform "
            "real inference. Start runner/scripts/serve_model.py with a verified GGUF, "
            "Safetensors, or native Sytra runtime container instead."
        )
