"""Export llama.cpp-compatible configs for Ollama and LM Studio.

Ollama is never pointed at raw SafeTensors — that path silently corrupts
some architectures. LM Studio loads GGUF files from disk; Sytra writes a
sidecar JSON of recommended runtime flags rather than guessing LM Studio's
internal schema across versions.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .gguf_meta import GgufMetadata, try_read_gguf_metadata
from .model_planner import ModelCompatibilityError, inspect_model
from .runtime_detect import find_lm_studio, find_ollama


@dataclass(frozen=True)
class RuntimeExport:
    format: str
    compatible: bool
    path: str | None
    command: list[str]
    reasons: list[str]
    runtime_version: str | None
    contents: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _runtime_version(executable: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lower().startswith("warning")
        and not line.startswith("\x1b")
    ]
    return lines[-1][:200] if lines else None


def detect_ollama() -> tuple[str | None, str | None]:
    exe = find_ollama()
    if not exe:
        return None, None
    return exe, _runtime_version(exe, "--version")


def detect_lm_studio() -> tuple[str | None, str | None]:
    exe = find_lm_studio()
    if not exe:
        return None, None
    return exe, _runtime_version(exe, "version")


def _modelfile_text(
    model_path: Path,
    *,
    context: int,
    metadata: GgufMetadata | None,
) -> str:
    lines = [f"FROM {model_path.resolve().as_posix()}", f"PARAMETER num_ctx {context}"]
    if metadata and metadata.chat_template:
        escaped = metadata.chat_template.replace("'''", "’''")
        lines.append(f"TEMPLATE '''{escaped}'''")
    stops = list(metadata.stop_tokens) if metadata else []
    if not stops:
        stops = ["<|im_end|>", "<|endoftext|>"]
    for stop in stops:
        lines.append(f'PARAMETER stop "{stop}"')
    return "\n".join(lines) + "\n"


def export_runtime_configs(
    model_path: str | Path,
    *,
    context: int = 4096,
    dest_dir: str | Path | None = None,
) -> dict:
    artifact = inspect_model(model_path)
    if artifact.format != "gguf":
        raise ModelCompatibilityError(
            "Ollama and LM Studio exports require a GGUF file. "
            "Convert SafeTensors with the bundled llama.cpp converter first; "
            "never `ollama create` from a raw safetensors directory."
        )
    gguf = Path(artifact.model_path)
    metadata = try_read_gguf_metadata(gguf)
    out_dir = Path(dest_dir) if dest_dir else gguf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ollama_exe, ollama_version = detect_ollama()
    lms_exe, lms_version = detect_lm_studio()
    modelfile = _modelfile_text(gguf, context=context, metadata=metadata)
    modelfile_path = out_dir / "Modelfile"
    modelfile_path.write_text(modelfile, encoding="utf-8")
    ollama_name = gguf.stem.lower().replace(" ", "-")
    ollama_cmd = [ollama_exe or "ollama", "create", ollama_name, "-f", str(modelfile_path)]
    ollama_reasons = [
        f"Wrote {modelfile_path} FROM the GGUF path (not SafeTensors).",
        "Run the create command in a real terminal; `ollama run` hangs without a TTY.",
    ]
    if ollama_exe is None:
        ollama_reasons.append("ollama was not found on PATH; install Ollama or add it to PATH.")

    lmstudio = {
        "gguf_path": str(gguf.resolve()),
        "recommended": {
            "context_length": context,
            "gpu_offload": "follow Sytra plan_inference n-gpu-layers",
            "mmap": True,
            "mlock": False,
            "flash_attention": True,
        },
        "notes": [
            "LM Studio loads GGUF from disk; point it at this exact file.",
            "LM Studio does not expose the same flags as llama-server; apply GPU offload in its UI.",
            "Do not import SafeTensors into LM Studio for architectures Sytra has not converted.",
        ],
    }
    lmstudio_path = out_dir / "lmstudio.sytra.json"
    lmstudio_path.write_text(json.dumps(lmstudio, indent=2), encoding="utf-8")
    lms_reasons = [f"Wrote {lmstudio_path} with the GGUF path and recommended mmap/mlock policy."]
    if lms_exe is None:
        lms_reasons.append(
            "LM Studio CLI (`lms`) was not found. Open LM Studio and load the GGUF path above."
        )

    return {
        "artifact": artifact.model_path,
        "architecture": artifact.architecture,
        "quantization": artifact.quantization,
        "is_moe": artifact.is_moe,
        "ollama": RuntimeExport(
            format="modelfile",
            compatible=True,
            path=str(modelfile_path),
            command=ollama_cmd,
            reasons=ollama_reasons,
            runtime_version=ollama_version,
            contents=modelfile,
        ).to_dict(),
        "lm_studio": RuntimeExport(
            format="lmstudio-sidecar",
            compatible=True,
            path=str(lmstudio_path),
            command=[lms_exe, "load", str(gguf)] if lms_exe else [],
            reasons=lms_reasons,
            runtime_version=lms_version,
            contents=json.dumps(lmstudio, indent=2),
        ).to_dict(),
    }
