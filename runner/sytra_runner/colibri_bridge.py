"""Map Sytra planning onto Colibri's published CLI (coli v1.6).

Colibri owns token generation for disk-streamed frontier MoE. Sytra owns
catalog, download, GGUF/llama.cpp, and the hardware envelope. Flags follow
https://github.com/JustVugg/colibri/blob/main/docs/SETTINGS.md
"""
from __future__ import annotations

import re
from typing import Any
from pathlib import Path

COLIBRI_RELEASES = "https://github.com/JustVugg/colibri/releases"

# Families Colibri actually serves today (one C file each). Kimi K2.7 is Sytra's
# native kernel, not a Colibri target.
COLIBRI_PREFERRED_FAMILIES = frozenset({"glm_moe", "kimi_k3", "inkling"})
COLIBRI_SUPPORTED_FAMILIES = frozenset({"glm_moe", "kimi_k3", "inkling", "olmoe"})

_MODEL_TYPE = (
    re.compile(r"glm.*moe", re.I),
    re.compile(r"^glm_moe_dsa$", re.I),
    re.compile(r"^olmoe$", re.I),
    re.compile(r"inkling", re.I),
    re.compile(r"^kimi[-_]?k3$", re.I),
    re.compile(r"deepseek.*v4", re.I),
)
_ARCHITECTURE = (
    re.compile(r"^Glm.*ForCausalLM$", re.I),
    re.compile(r"^Olmoe", re.I),
    re.compile(r"Inkling", re.I),
    re.compile(r"^KimiK3", re.I),
    re.compile(r"DeepseekV4", re.I),
)
_REPO = (
    re.compile(r"GLM-5\.2", re.I),
    re.compile(r"Kimi-K3", re.I),
    re.compile(r"OLMoE", re.I),
    re.compile(r"Inkling", re.I),
    re.compile(r"DeepSeek-V4", re.I),
    re.compile(r"colibri-int4", re.I),
)

COLIBRI_STREAMING_FLOOR_TPS = 0.05


def mb_to_coli_gb(mb: int) -> int:
    """Colibri `--ram` / `--vram` are integer GB. 0 means engine auto."""
    if mb <= 0:
        return 0
    return mb // 1024


def detect_colibri_family(
    *,
    adapter_family: str | None,
    adapter_id: str | None,
    model_type: str,
    architecture: str,
    repo_id: str | None,
) -> str | None:
    if adapter_id == "sytra-kimi-k2.7-code":
        return None
    if adapter_family in COLIBRI_SUPPORTED_FAMILIES:
        return adapter_family
    blob = " ".join(part for part in (model_type, architecture, repo_id or "") if part)
    if any(pattern.search(model_type) for pattern in _MODEL_TYPE):
        return _family_from_text(blob)
    if any(pattern.search(architecture) for pattern in _ARCHITECTURE):
        return _family_from_text(blob)
    if repo_id and any(pattern.search(repo_id) for pattern in _REPO):
        return _family_from_text(blob)
    return None


def _family_from_text(text: str) -> str:
    lowered = text.lower()
    if "kimi" in lowered and "k3" in lowered:
        return "kimi_k3"
    if "olmoe" in lowered:
        return "olmoe"
    if "inkling" in lowered:
        return "inkling"
    if "deepseek" in lowered:
        return "deepseek_v4"
    return "glm_moe"


def colibri_model_id(*, repo_id: str | None, family: str | None, architecture: str) -> str:
    if family == "kimi_k3" or (repo_id and "Kimi-K3" in repo_id):
        return "kimi-k3-colibri"
    if family == "olmoe" or "olmoe" in architecture.lower():
        return "olmoe-colibri"
    if family == "inkling" or "inkling" in architecture.lower():
        return "inkling-colibri"
    if family == "deepseek_v4":
        return "deepseek-v4-colibri"
    return "glm-5.2-colibri"


def colibri_install_hint(project_root: Path | str | None = None) -> str:
    root = Path(project_root).resolve() if project_root is not None else Path.cwd()
    script = Path(__file__).resolve().parents[1] / "scripts" / "provision_colibri.py"
    return (
        "Colibri (`coli`) was not found. Sytra will keep catalog/download/GGUF planning; "
        "frontier MoE token generation needs the Colibri engine. Run "
        f"`python {script} --project-root {root}` to unpack the pinned Colibri release "
        "into `.tools/colibri` (also auto-installed on `plan_inference` / "
        "`serve_model --backend colibri` and on `auto` for GLM-5.2, Kimi K3, Inkling, and "
        "OLMoE safetensors). Or put `coli` on PATH / set SYTRA_COLIBRI_COMMAND / "
        f"SYTRA_COLIBRI_HOME. Releases: {COLIBRI_RELEASES}. "
        "Set SYTRA_SKIP_COLIBRI_PROVISION=1 to disable the download."
    )


def build_colibri_serve_command(
    launcher: list[str],
    *,
    model_path: str,
    host: str,
    port: int,
    context: int,
    vram_limit_mb: int,
    ram_limit_mb: int,
    gpu_visible: bool,
    model_id: str,
    policy: str = "quality",
) -> list[str]:
    ram_gb = mb_to_coli_gb(ram_limit_mb)
    vram_gb = mb_to_coli_gb(vram_limit_mb)
    command = launcher + [
        "serve",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--model-id",
        model_id,
        "--ctx",
        str(context),
        "--policy",
        policy,
        "--auto-tier",
        "--gpu",
        "auto" if gpu_visible else "none",
    ]
    if ram_gb:
        command.extend(["--ram", str(ram_gb)])
    if vram_gb:
        command.extend(["--vram", str(vram_gb)])
    return command


def build_colibri_preflight_command(launcher: list[str], *, model_path: str) -> list[str]:
    return launcher + ["doctor", "--model", model_path, "--json"]


def colibri_resource_plan(
    *,
    vram_limit_mb: int,
    ram_limit_mb: int,
    nvme_weight_bytes: int,
    storage_bandwidth_mbps: int,
    target_tps: float,
    strategy: str,
    io_max_tps: float | None,
    family: str | None,
    model_id: str,
) -> dict[str, Any]:
    return {
        "vram_mb": vram_limit_mb,
        "ram_mb": ram_limit_mb,
        "vram_gb": mb_to_coli_gb(vram_limit_mb),
        "ram_gb": mb_to_coli_gb(ram_limit_mb),
        "nvme_weight_bytes": nvme_weight_bytes,
        "storage_bandwidth_mbps": storage_bandwidth_mbps,
        "target_tps": target_tps,
        "strategy": strategy,
        "io_max_tps": io_max_tps,
        "auto_tier": True,
        "policy": "quality",
        "family": family,
        "model_id": model_id,
        "sytra_role": "catalog-download-plan",
        "colibri_role": "serve-streamed-experts",
    }
