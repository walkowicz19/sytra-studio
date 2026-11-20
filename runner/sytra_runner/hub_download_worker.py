"""Isolated Hub transfer; the parent supervises and may terminate this process."""
import json
import os
import sys


def main() -> None:
    repo_id, filename, revision, local_dir, tokenless = sys.argv[1:]
    # Import only after the parent has configured HF_HUB_DISABLE_XET: Hub reads
    # that setting during import, and changing it later does not disable Xet.
    from huggingface_hub import hf_hub_download

    explicit_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token = explicit_token if explicit_token else (False if tokenless == "1" else None)
    result = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                             local_dir=local_dir, token=token)
    print("SYTRA_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
