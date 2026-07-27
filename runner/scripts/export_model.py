"""Sytra 1.2.0 General-Purpose Downstream Exporter for Ollama & LM Studio."""
import os
import sys
import argparse
import subprocess
from pathlib import Path

def export_model(model_path: str, model_name: str = None, context_length: int = 4096):
    path = Path(model_path)
    if not path.exists():
        print(f"[ERROR] Model file or directory does not exist: {model_path}")
        return

    if not model_name:
        model_name = path.stem.lower().replace("_", "-").replace(" ", "-")

    print(f"=== Sytra 1.2.0: General Model Exporter ===")
    print(f"Model Path: {path.resolve()}")
    print(f"Model Identifier: {model_name}")

    target_dir = path.parent if path.is_file() else path
    modelfile_path = target_dir / "Modelfile"

    modelfile_content = f"""FROM ./{path.name if path.is_file() else '*'}
PARAMETER num_ctx {context_length}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"[OK] Created Modelfile at: {modelfile_path.resolve()}")

    # Register in Ollama if ollama CLI is available
    print(f"\n--- Registering '{model_name}' in Ollama ---")
    try:
        res = subprocess.run(
            ["ollama", "create", model_name, "-f", "Modelfile"],
            cwd=str(target_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=180
        )
        if res.returncode == 0:
            print(f"[OK] Successfully registered '{model_name}' model in Ollama!")
        else:
            print(f"[LOG] Ollama CLI output: {res.stdout or res.stderr}")
    except Exception as exc:
        print(f"[NOTE] Ollama step skipped: {exc}")

    print(f"\n=== SUCCESS: Model '{model_name}' is ready for LM Studio & Ollama! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sytra 1.2.0 General Model Exporter")
    parser.add_argument("--model", required=True, help="Path to downloaded GGUF/Safetensors model file or folder")
    parser.add_argument("--name", default=None, help="Custom model identifier for Ollama registration")
    parser.add_argument("--context", type=int, default=4096, help="Context length limit (default: 4096)")
    args = parser.parse_args()

    export_model(model_path=args.model, model_name=args.name, context_length=args.context)
    sys.exit(0)
