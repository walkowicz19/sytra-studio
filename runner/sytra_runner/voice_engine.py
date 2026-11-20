"""Local voice runtime for Sytra Studio: Kokoro-82M ONNX TTS + Faster-Whisper STT."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import threading
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

KOKORO_REPO_ID = "fastrtc/kokoro-onnx"
KOKORO_MODEL_INT8_FILE = "kokoro-v1.0.int8.onnx"
KOKORO_MODEL_FP32_FILE = "kokoro-v1.0.onnx"
KOKORO_MODEL_FILE = KOKORO_MODEL_INT8_FILE
KOKORO_VOICES_FILE = "voices-v1.0.bin"
KOKORO_MODEL_MIRRORS = (
    (
        KOKORO_MODEL_INT8_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    ),
    (
        KOKORO_MODEL_FP32_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    ),
    (
        KOKORO_MODEL_FP32_FILE,
        "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/kokoro-v1.0.onnx",
    ),
)
KOKORO_VOICES_MIRRORS = (
    (
        KOKORO_VOICES_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ),
    (
        KOKORO_VOICES_FILE,
        "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/voices-v1.0.bin",
    ),
)
WHISPER_REPO_ID = "Systran/faster-whisper-tiny"

DEFAULT_KOKORO_VOICES = [
    {"id": "auto", "label": "Auto (EN / PT-BR)", "lang": "auto"},
    {"id": "af_heart", "label": "Kokoro Heart (US F)", "lang": "en-us"},
    {"id": "af_bella", "label": "Kokoro Bella (US F)", "lang": "en-us"},
    {"id": "af_sky", "label": "Kokoro Sky (US F)", "lang": "en-us"},
    {"id": "am_adam", "label": "Kokoro Adam (US M)", "lang": "en-us"},
    {"id": "bf_emma", "label": "Kokoro Emma (UK F)", "lang": "en-gb"},
    {"id": "bm_george", "label": "Kokoro George (UK M)", "lang": "en-gb"},
    {"id": "pf_dora", "label": "Kokoro Dora (PT-BR F)", "lang": "pt-br"},
    {"id": "pm_alex", "label": "Kokoro Alex (PT-BR M)", "lang": "pt-br"},
]

_PT_HINT_RE = re.compile(
    r"(\b(ol[aá]|voc[eê]s?|n[aã]o|sim|obrigad[oa]|tudo|bem|como|para|com|isso|aqui|agora|sistema|modelo|relat[oó]rio|arquivo|ajuda|falar|ouvir)\b|[ãõç])",
    re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"https?://[^\s)]+")
_MD_SYNTAX_RE = re.compile(r"(^#+\s*|`+|[*_~]{1,3}|^>\s*)", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$", re.MULTILINE)


def clean_speech_text(raw: str, max_chars: int = 2400) -> str:
    """Normalize LLM markdown replies into crisp spoken prose for TTS."""
    text = (raw or "").replace("\r\n", "\n")
    text = _CODE_FENCE_RE.sub(" Code block shown on screen. ", text)
    text = _TABLE_SEP_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" link ", text)
    text = _MD_SYNTAX_RE.sub(" ", text)
    text = text.replace("|", ", ")
    text = re.sub(r"[ \t\n]+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def split_speech_sentences(text: str) -> list[str]:
    """Split normalized speech prose into bounded, natural sentence chunks."""
    cleaned = clean_speech_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    cur = ""
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if len(cur) + len(p) + 1 <= 260:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def infer_voice_and_lang(voice: str, text: str, explicit_lang: str = "auto") -> tuple[str, str]:
    """Resolve Kokoro voice ID and G2P language code from explicit preference or text."""
    v = (voice or "auto").strip()
    lang = (explicit_lang or "auto").strip().lower()

    if v in ("", "auto"):
        if _PT_HINT_RE.search(text or ""):
            return "pf_dora", "pt-br"
        return "af_heart", "en-us"

    prefix_lang = {
        "af_": "en-us",
        "am_": "en-us",
        "bf_": "en-gb",
        "bm_": "en-gb",
        "pf_": "pt-br",
        "pm_": "pt-br",
        "ef_": "es",
        "em_": "es",
        "ff_": "fr-fr",
        "if_": "it",
        "im_": "it",
        "jf_": "ja",
        "jm_": "ja",
    }
    inferred = prefix_lang.get(v[:3], "en-us")
    if lang in ("", "auto"):
        return v, inferred
    return v, lang


def encode_pcm16_wav(samples: Any, sample_rate: int = 24000) -> bytes:
    """Encode float32 [-1..1] or sequence samples into mono 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        if hasattr(samples, "tolist"):
            # Faster path when numpy array is available
            try:
                import numpy as np

                arr = np.asarray(samples, dtype=np.float32).reshape(-1)
                clipped = np.clip(arr, -1.0, 1.0)
                pcm = (clipped * 32767.0).astype(np.int16)
                wf.writeframes(pcm.tobytes())
                return buf.getvalue()
            except Exception:
                samples = samples.tolist()

        raw_ints = [
            max(-32767, min(32767, int(float(s) * 32767.0)))
            for s in samples
        ]
        wf.writeframes(struct.pack(f"<{len(raw_ints)}h", *raw_ints))
    return buf.getvalue()


def _wav_rms(wav_bytes: bytes) -> float:
    """Compute normalized RMS amplitude of a 16-bit PCM WAV payload to gate silence."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            if wf.getsampwidth() != 2:
                return 1.0
            frames = wf.readframes(wf.getnframes())
        if not frames:
            return 0.0
        count = len(frames) // 2
        if count == 0:
            return 0.0
        samples = struct.unpack(f"<{count}h", frames[: count * 2])
        sq = sum((s / 32768.0) ** 2 for s in samples)
        return (sq / count) ** 0.5
    except Exception:
        return 1.0


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def resolve_kokoro_model_path(model_dir: Path) -> Path | None:
    """Return an existing non-empty Kokoro ONNX path (int8 preferred, fp32 fallback)."""
    for filename in (KOKORO_MODEL_INT8_FILE, KOKORO_MODEL_FP32_FILE):
        candidate = model_dir / filename
        if _is_nonempty_file(candidate):
            return candidate
    return None


def _ssl_context() -> Any:
    """Use OS certificate store via truststore when available so uv Python validates TLS cleanly."""
    import ssl

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return ssl.create_default_context()


def _stream_download_url(url: str, dest_path: Path) -> None:
    tmp_path = dest_path.parent / f".tmp.{dest_path.name}"
    req = urllib.request.Request(url, headers={"User-Agent": "sytra-studio/1.3"})
    try:
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=180) as resp:
            if getattr(resp, "status", 200) >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            raw_len = resp.headers.get("Content-Length")
            expected = int(raw_len) if raw_len and raw_len.isdigit() else None
            downloaded = 0
            with open(tmp_path, "wb") as fp:
                while True:
                    chunk = resp.read(65_536)
                    if not chunk:
                        break
                    fp.write(chunk)
                    downloaded += len(chunk)
        if downloaded <= 0:
            raise RuntimeError(f"Empty download from {url}")
        if expected and downloaded != expected:
            raise RuntimeError(f"Truncated download from {url}: {downloaded}/{expected}")
        os.replace(tmp_path, dest_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def _download_with_mirrors(model_dir: Path, mirrors: tuple[tuple[str, str], ...]) -> Path:
    errors: list[str] = []
    for filename, url in mirrors:
        dest_path = model_dir / filename
        try:
            _stream_download_url(url, dest_path)
            return dest_path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Kokoro download failed across mirrors: " + " | ".join(errors))


def download_kokoro_models(model_dir: Path) -> tuple[Path, Path]:
    """Download Kokoro-82M int8 ONNX model and v1.0 voice bank directly over HTTPS."""
    model_dir.mkdir(parents=True, exist_ok=True)
    stale_cache = model_dir / ".cache"
    if stale_cache.exists():
        shutil.rmtree(stale_cache, ignore_errors=True)

    voices_path = model_dir / KOKORO_VOICES_FILE
    if not _is_nonempty_file(voices_path):
        voices_path = _download_with_mirrors(model_dir, KOKORO_VOICES_MIRRORS)

    model_path = resolve_kokoro_model_path(model_dir)
    if model_path is None:
        model_path = _download_with_mirrors(model_dir, KOKORO_MODEL_MIRRORS)

    if model_path is None or not _is_nonempty_file(model_path) or not _is_nonempty_file(voices_path):
        raise RuntimeError("Kokoro model download did not produce expected ONNX artifacts.")
    return model_path, voices_path


class VoiceRuntime:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._kokoro: Any = None
        self._whisper: Any = None
        self.shutdown_event = threading.Event()

    def status(self) -> dict[str, Any]:
        model_ok = resolve_kokoro_model_path(self.model_dir) is not None
        voices_ok = _is_nonempty_file(self.model_dir / KOKORO_VOICES_FILE)
        return {
            "ok": True,
            "kokoro_model_ready": model_ok and voices_ok,
            "kokoro_loaded": self._kokoro is not None,
            "whisper_loaded": self._whisper is not None,
            "model_dir": str(self.model_dir),
        }

    def _get_kokoro(self) -> Any:
        with self._lock:
            if self._kokoro is not None:
                return self._kokoro
            try:
                import espeakng_loader

                lib_path = espeakng_loader.get_library_path()
                if lib_path:
                    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", lib_path)
            except Exception:
                pass
            from kokoro_onnx import Kokoro

            model_path = resolve_kokoro_model_path(self.model_dir)
            voices_path = self.model_dir / KOKORO_VOICES_FILE
            if model_path is None or not _is_nonempty_file(voices_path):
                model_path, voices_path = download_kokoro_models(self.model_dir)
            self._kokoro = Kokoro(str(model_path), str(voices_path))
            return self._kokoro

    def _get_whisper(self) -> Any:
        with self._lock:
            if self._whisper is not None:
                return self._whisper
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            from faster_whisper import WhisperModel

            whisper_dir = self.model_dir.parent / "whisper-tiny"
            whisper_dir.mkdir(parents=True, exist_ok=True)
            self._whisper = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8",
                download_root=str(whisper_dir),
            )
            return self._whisper

    def synthesize(self, text: str, voice: str = "auto", speed: float = 1.0, lang: str = "auto") -> bytes:
        chunks = split_speech_sentences(text)
        if not chunks:
            raise ValueError("Nothing speakable in input text.")
        kokoro = self._get_kokoro()
        safe_speed = max(0.65, min(1.6, float(speed or 1.0)))

        import numpy as np

        audio_segments = []
        sr = 24000
        for chunk in chunks:
            resolved_voice, resolved_lang = infer_voice_and_lang(voice, chunk, lang)
            try:
                samples, cur_sr = kokoro.create(
                    chunk,
                    voice=resolved_voice,
                    speed=safe_speed,
                    lang=resolved_lang,
                )
            except Exception:
                # Safe fallback if requested language phonemizer isn't matched
                samples, cur_sr = kokoro.create(
                    chunk,
                    voice="af_heart",
                    speed=safe_speed,
                    lang="en-us",
                )
            sr = int(cur_sr or 24000)
            audio_segments.append(np.asarray(samples, dtype=np.float32).reshape(-1))
        combined = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]
        return encode_pcm16_wav(combined, sr)

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> dict[str, Any]:
        if not wav_bytes or len(wav_bytes) < 64:
            return {"text": "", "language": language or "", "duration_sec": 0.0}
        if _wav_rms(wav_bytes) < 0.0035:
            return {"text": "", "language": language or "", "duration_sec": 0.0}

        whisper = self._get_whisper()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tf.write(wav_bytes)
                tmp_path = tf.name
            lang_arg = language.strip() if isinstance(language, str) and language.strip() and language != "auto" else None
            segments, info = whisper.transcribe(
                tmp_path,
                language=lang_arg,
                vad_filter=True,
                beam_size=1,
                initial_prompt="Sytra Studio, MCP, mcp_config.json, SKILL.md, skill, skills, Lil Sytra, Kokoro, Ollama, LM Studio, GPU, VRAM, API, JSON.",
            )
            text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
            return {
                "text": text,
                "language": getattr(info, "language", "") or "",
                "duration_sec": float(getattr(info, "duration", 0.0) or 0.0),
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def _make_handler(runtime: VoiceRuntime, server_ref: dict[str, ThreadingHTTPServer]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0 or length > 25 * 1024 * 1024:
                raise ValueError("Invalid request body length")
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("JSON object expected")
            return data

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, runtime.status())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/shutdown":
                    self._send_json(200, {"ok": True})
                    srv = server_ref.get("srv")
                    if srv is not None:
                        threading.Thread(target=srv.shutdown, daemon=True).start()
                    runtime.shutdown_event.set()
                    return

                data = self._read_json()
                if self.path == "/tts":
                    wav_bytes = runtime.synthesize(
                        text=str(data.get("text", "")),
                        voice=str(data.get("voice", "auto")),
                        speed=float(data.get("speed", 1.0) or 1.0),
                        lang=str(data.get("lang", "auto")),
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(wav_bytes)))
                    self.end_headers()
                    self.wfile.write(wav_bytes)
                    return

                if self.path == "/stt":
                    b64 = str(data.get("audio_base64", ""))
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    wav_bytes = base64.b64decode(b64)
                    res = runtime.transcribe(wav_bytes, data.get("language"))
                    self._send_json(200, res)
                    return

                self._send_json(404, {"error": "not found"})
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sytra Studio Kokoro-82M & Whisper voice server")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args(argv)

    if args.download_only:
        download_kokoro_models(args.model_dir)
        print(json.dumps({"ok": True, "model_dir": str(args.model_dir)}))
        return 0

    runtime = VoiceRuntime(args.model_dir)
    if args.warmup:
        runtime._get_kokoro()

    server_ref: dict[str, ThreadingHTTPServer] = {}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(runtime, server_ref))
    server_ref["srv"] = server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
