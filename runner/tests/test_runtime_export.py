import json
from pathlib import Path

import pytest

from sytra_runner.model_planner import ModelCompatibilityError
from sytra_runner.runtime_export import export_runtime_configs
from gguf_test_file import write_metadata_gguf


def test_export_writes_modelfile_from_gguf(tmp_path):
    model = write_metadata_gguf(
        tmp_path / "chat.gguf",
        {
            "general.architecture": "qwen3",
            "tokenizer.chat_template": "{{ bos }}{{ message }}",
            "tokenizer.ggml.tokens": ["<s>", "</s>"],
            "tokenizer.ggml.bos_token_id": 0,
            "tokenizer.ggml.eos_token_id": 1,
        },
    )
    result = export_runtime_configs(model, context=2048, dest_dir=tmp_path)
    modelfile = Path(result["ollama"]["path"]).read_text(encoding="utf-8")
    assert "FROM " in modelfile
    assert "PARAMETER num_ctx 2048" in modelfile
    assert "TEMPLATE" in modelfile
    sidecar = json.loads(Path(result["lm_studio"]["path"]).read_text(encoding="utf-8"))
    assert sidecar["gguf_path"].endswith("chat.gguf")
    assert sidecar["recommended"]["mlock"] is False


def test_export_refuses_safetensors(tmp_path):
    root = tmp_path / "st"
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"llama","architectures":["LlamaForCausalLM"]}', encoding="utf-8")
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    padding = (8 - len(header) % 8) % 8
    header += b" " * padding
    (root / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
    with pytest.raises(ModelCompatibilityError, match="GGUF"):
        export_runtime_configs(root)
