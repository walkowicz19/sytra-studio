"""Sytra 1.2.0 — OpenAI-compatible REST server for MoE inference.

Wraps MoEInferenceEngine in a FastAPI server that LM Studio or any OpenAI
client can proxy to. Expert streaming keeps VRAM under the configured budget
so the PC never freezes.

Usage:
    python runner/scripts/serve_moe.py --model "D:/lm-studio models/moonshotai/Kimi-GGUF/model.gguf"
    python runner/scripts/serve_moe.py --model path/to/model.gguf --vram-limit 10000 --port 8080
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sytra.serve_moe")

# Add runner/ to path so we can import sytra_runner
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from sytra_runner.backends.moe_inference import MoEInferenceEngine
from sytra_runner.tool_engine import UniversalToolEngine

app = FastAPI(
    title="Sytra MoE Inference Server",
    description="OpenAI-compatible endpoint backed by GPU Expert Streaming Pager and Universal Tool Engine",
    version="1.2.0",
)

# Global engine (initialised at startup)
_engine: MoEInferenceEngine | None = None
_model_name: str = "kimi"
_tool_engine = UniversalToolEngine()


# ─── Request / response schemas ──────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "kimi"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False
    stop: list[str] | None = None
    enable_tools: bool = True


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if _engine is None:
        return {"status": "loading"}
    stats = _engine.pager.stats()
    return {
        "status": "ok",
        "model": _model_name,
        "vram_used_mb": stats["vram_used_mb"],
        "vram_limit_mb": stats["vram_limit_mb"],
        "vram_hit_rate": stats["vram_hit_rate"],
        "disk_loads": stats["disk_loads"],
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "created": int(time.time()), "owned_by": "sytra"}]
    }

@app.get("/v1/local_models")
def scan_local_models():
    """Scans storage directories for downloaded, fine-tuned, and merged models."""
    results = []
    search_paths = [
        ("downloaded", Path.home() / "lm-studio models"),
        ("downloaded", Path.home() / ".cache" / "lm-studio" / "models"),
        ("finetuned", Path("runs")),
        ("merged", Path("runs/merged")),
    ]
    seen = set()
    for cat, base_dir in search_paths:
        if not base_dir.exists():
            continue
        for file in base_dir.rglob("*"):
            if file.is_file() and (file.suffix in (".gguf", ".safetensors", ".bin")):
                if file.name.startswith(".") or file.name.endswith(".tmp"):
                    continue
                file_str = str(file.resolve())
                if file_str not in seen:
                    seen.add(file_str)
                    size_gb = round(file.stat().st_size / (1024 * 1024 * 1024), 2)
                    results.append({
                        "id": file.stem,
                        "name": file.name,
                        "category": cat,
                        "path": file_str,
                        "size_gb": size_gb,
                        "format": file.suffix.lstrip("."),
                    })
    return {"models": results}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prompt = _build_prompt(req.messages, enable_tools=req.enable_tools)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Non-streaming: collect all tokens and process ReAct tool loop if triggered
    full_text = "".join(_engine.generate(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop_sequences=req.stop or [],
    ))

    # Check for Universal Tool Call output
    if req.enable_tools:
        tool_call = _tool_engine.extract_tool_call(full_text)
        if tool_call:
            t_name = tool_call.get("tool_call")
            t_args = tool_call.get("arguments", {})
            logger.info("Executing Universal Tool Call: %s(%s)", t_name, t_args)
            t_res = _tool_engine.execute_tool(t_name, t_args)
            
            # Feed tool observation back into model for final answer
            followup_prompt = prompt + f"\n<|assistant|>\n{full_text}\n<|system|>\nTool Output for '{t_name}':\n{t_res}\n<|assistant|>\n"
            final_answer = "".join(_engine.generate(followup_prompt, max_tokens=512, temperature=req.temperature))
            full_text = f"{full_text}\n\n🛠️ **[Tool Call: {t_name}]**\n```json\n{json.dumps(t_args, indent=2)}\n```\n\n**Output:** {t_res}\n\n{final_answer}"

    stats = _engine.pager.stats()
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(full_text.split()), "total_tokens": 0},
        "_sytra": {
            "vram_used_mb": stats["vram_used_mb"],
            "vram_limit_mb": stats["vram_limit_mb"],
            "vram_hit_rate": stats["vram_hit_rate"],
            "disk_loads": stats["disk_loads"],
        },
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_prompt(messages: list[Message], enable_tools: bool = True) -> str:
    parts = []
    has_system = any(m.role == "system" for m in messages)

    if enable_tools and not has_system:
        parts.append(f"<|system|>\n{_tool_engine.get_system_prompt_wrapper()}\n")

    for msg in messages:
        if msg.role == "system":
            sys_content = msg.content
            if enable_tools and "Available Tools" not in sys_content:
                sys_content = f"{_tool_engine.get_system_prompt_wrapper()}\n\n{sys_content}"
            parts.append(f"<|system|>\n{sys_content}\n")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    if not HAS_FASTAPI:
        print("[ERROR] FastAPI and uvicorn are required. Install with:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Sytra MoE Inference Server")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--vram-limit", type=int, default=8192, help="VRAM budget in MB (default: 8192)")
    parser.add_argument("--context", type=int, default=4096, help="Context window length limit (default: 4096)")
    parser.add_argument("--cpu-kv-cache", action="store_true", help="Offload 100% of KV Cache to CPU RAM to prevent GPU VRAM overflow")
    parser.add_argument("--ram-limit", type=int, default=4096, help="Pinned RAM buffer in MB (default: 4096)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--no-pilot", action="store_true", help="Disable PILOT prefetch thread")
    parser.add_argument("--dry-run", action="store_true", help="Load metadata only, do not start server")
    args = parser.parse_args()

    global _engine, _model_name

    logger.info(
        "Loading model: %s (Context: %d tokens, VRAM limit: %d MB, CPU KV Cache: %s)",
        args.model, args.context, args.vram_limit, "ENABLED" if args.cpu_kv_cache else "DISABLED"
    )
    _model_name = Path(args.model).stem
    _engine = MoEInferenceEngine(
        gguf_path=args.model,
        vram_limit_mb=args.vram_limit,
        ram_limit_mb=args.ram_limit,
        pilot_prefetch=not args.no_pilot,
    )

    if args.dry_run:
        logger.info("Dry run complete. Metadata: %s", dict(list(_engine.reader.metadata.items())[:10]))
        _engine.close()
        return

    logger.info("Starting Sytra MoE server on %s:%d", args.host, args.port)
    logger.info("OpenAI-compatible endpoint: http://%s:%d/v1/chat/completions", args.host, args.port)
    logger.info("Health check: http://%s:%d/health", args.host, args.port)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if _engine:
            _engine.close()


if __name__ == "__main__":
    main()
