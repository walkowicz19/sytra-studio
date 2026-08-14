import os
import sys
import argparse
import json
import yaml
from sytra_runner.telemetry import emit_metric, emit_event, emit_log

def parse_args():
    parser = argparse.ArgumentParser(description="Sytra Studio HF Publisher")
    parser.add_argument("artifact_path", type=str, help="Path to the model directory to upload")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repository ID")
    parser.add_argument("--private", type=str, default="false", help="Whether repo is private ('true' or 'false')")
    parser.add_argument("--token", type=str, default="", help="Hugging Face API token")
    return parser.parse_args()

def generate_model_card(artifact_path, repo_id):
    # Try to load config files
    run_yaml_path = os.path.join(artifact_path, "run.yaml")
    merge_yaml_path = os.path.join(artifact_path, "merge.yaml")
    
    card_content = ""
    
    if os.path.exists(run_yaml_path):
        try:
            with open(run_yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            base_model = config.get("model", "unknown")
            adapter = config.get("adapter", {})
            adapter_kind = adapter.get("kind", "lora").upper()
            rank = adapter.get("rank", 16)
            alpha = adapter.get("alpha", 32)
            
            optim = config.get("optim", {})
            lr = optim.get("learning_rate", 2e-4)
            
            train = config.get("train", {})
            max_steps = train.get("max_steps", 200)
            
            card_content = f"""---
library_name: peft
base_model: {base_model}
tags:
- sytra-studio
- fine-tuned
---

# {repo_id}

This model is a fine-tuned adapter trained with **Sytra Studio**.

## Training Hyperparameters
- **Base Model**: `{base_model}`
- **Adapter Type**: `{adapter_kind}` (r={rank}, alpha={alpha})
- **Learning Rate**: `{lr}`
- **Max Steps**: `{max_steps}`
"""
        except Exception as e:
            card_content = f"# {repo_id}\n\nFine-tuned adapter uploaded via Sytra Studio.\n\nError parsing metadata: {e}"
            
    elif os.path.exists(merge_yaml_path):
        try:
            with open(merge_yaml_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                
            merge_method = config.get("merge_method", "slerp").upper()
            base_model = config.get("base_model", "none")
            models = config.get("models", [])
            models_list = "\n".join([f"- `{m.get('model')}` (weight: {m.get('weight')})" for m in models])
            
            card_content = f"""---
tags:
- sytra-studio
- merged
---

# {repo_id}

This model is a merged checkpoint created with **Sytra Studio** using the `{merge_method}` method.

## Merge Recipe
- **Merge Method**: `{merge_method}`
- **Base Model**: `{base_model}`
- **Models Combined**:
{models_list}
"""
        except Exception as e:
            card_content = f"# {repo_id}\n\nMerged checkpoint uploaded via Sytra Studio.\n\nError parsing recipe metadata: {e}"
            
    else:
        card_content = f"""---
tags:
- sytra-studio
---

# {repo_id}

Model checkpoint uploaded via **Sytra Studio**.
"""
    return card_content

def main():
    args = parse_args()
    artifact_path = args.artifact_path
    repo_id = args.repo_id
    is_private = args.private.lower() == "true"
    token = args.token
    
    emit_log(f"Initializing HF upload for {artifact_path} to {repo_id}...")
    
    # Generate README.md if not already present
    readme_path = os.path.join(artifact_path, "README.md")
    if not os.path.exists(readme_path):
        emit_log("Generating model card README.md from configuration...")
        try:
            card_content = generate_model_card(artifact_path, repo_id)
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(card_content)
        except Exception as e:
            emit_log(f"Warning: failed to create README.md: {e}")

    # Check if huggingface_hub is installed and we have a token to upload
    huggingface_hub_installed = False
    try:
        from huggingface_hub import HfApi
        huggingface_hub_installed = True
    except ImportError:
        pass
        
    if not huggingface_hub_installed:
        emit_event("error", error="huggingface_hub is not installed; refusing to fake an upload")
        sys.exit(1)
    if not token:
        emit_event("error", error="Hugging Face token is required; refusing to fake an upload")
        sys.exit(1)

    try:
        emit_log("Connecting to Hugging Face Hub...")
        api = HfApi()
        emit_log(f"Ensuring repository '{repo_id}' exists...")
        api.create_repo(
            repo_id=repo_id,
            private=is_private,
            token=token,
            exist_ok=True
        )
        emit_log("Uploading folder contents...")
        api.upload_folder(
            folder_path=artifact_path,
            repo_id=repo_id,
            token=token
        )
        emit_metric(step=100, progress=1.0)
        target_url = f"https://huggingface.co/{repo_id}"
        emit_log(f"Upload completed successfully. Available at: {target_url}")
        emit_event("done", url=target_url)
    except Exception as e:
        emit_log(f"HF Upload error: {e}")
        emit_event("error", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
