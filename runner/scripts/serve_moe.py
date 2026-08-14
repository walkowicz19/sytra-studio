"""Backward-compatible entry point for the verified model server launcher.

The old Python MoE placeholder never performed a model forward pass. Keep this
script name for existing integrations, but route every request through the
capability-gated llama.cpp, vLLM, or native Sytra launcher.
"""
from serve_model import main


if __name__ == "__main__":
    raise SystemExit(main())
