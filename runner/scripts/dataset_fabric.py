"""JSON stdin/stdout bridge for Dataset Fabric, used by the desktop and MCP host."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sytra_runner.dataset_fabric import merge_datasets, split_dataset


def main():
    request = json.load(sys.stdin)
    if request["operation"] == "merge":
        result = merge_datasets(request["inputs"], request["output_dir"],
                                weights=request.get("weights"), total_samples=request.get("total_samples"),
                                shard_rows=request.get("shard_rows", 0), seed=request.get("seed", 3407),
                                curation=request.get("curation"))
    elif request["operation"] == "split":
        result = split_dataset(request["input"], request["output_dir"],
                               validation_fraction=request.get("validation_fraction", 0.1),
                               seed=request.get("seed", 3407))
    else:
        raise ValueError("Unknown dataset operation")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
