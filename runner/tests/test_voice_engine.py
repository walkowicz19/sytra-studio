import io
import wave

from pathlib import Path
import tempfile

from sytra_runner.voice_engine import (
    clean_speech_text,
    encode_pcm16_wav,
    infer_voice_and_lang,
    resolve_kokoro_model_path,
    split_speech_sentences,
)


def test_clean_speech_text_strips_code_and_urls() -> None:
    raw = (
        "### Report\n"
        "Check [Docs](https://example.com) and run:\n"
        "```python\nprint('hi')\n```\n"
        "Also visit https://sytra.dev/guide."
    )
    cleaned = clean_speech_text(raw)
    assert "```" not in cleaned
    assert "https://" not in cleaned
    assert "Code block shown on screen." in cleaned
    assert "Docs" in cleaned


def test_split_speech_sentences_batches_compactly() -> None:
    text = "First check passed. Second item has 99% score. Third note finished!"
    chunks = split_speech_sentences(text)
    assert len(chunks) >= 1
    assert "First check passed." in " ".join(chunks)
    assert "Third note finished!" in " ".join(chunks)


def test_infer_voice_and_lang_switches_portuguese_and_english_automatically() -> None:
    v_pt, lang_pt = infer_voice_and_lang("auto", "Olá! Tudo bem com você hoje?")
    assert v_pt == "pf_dora"
    assert lang_pt == "pt-br"

    v_en, lang_en = infer_voice_and_lang("auto", "Hello! All GPUs are cool and idle.")
    assert v_en == "af_heart"
    assert lang_en == "en-us"

    v_explicit, lang_explicit = infer_voice_and_lang("pm_alex", "Hello", "auto")
    assert v_explicit == "pm_alex"
    assert lang_explicit == "pt-br"


def test_encode_pcm16_wav_produces_valid_wav_header() -> None:
    wav_bytes = encode_pcm16_wav([0.0, 0.5, -0.5, 0.25], sample_rate=24000)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 4


def test_resolve_kokoro_model_path_prefers_int8_and_ignores_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert resolve_kokoro_model_path(root) is None
        (root / "kokoro-v1.0.int8.onnx").write_bytes(b"")
        assert resolve_kokoro_model_path(root) is None
        (root / "kokoro-v1.0.onnx").write_bytes(b"fp32")
        assert resolve_kokoro_model_path(root).name == "kokoro-v1.0.onnx"
        (root / "kokoro-v1.0.int8.onnx").write_bytes(b"int8")
        assert resolve_kokoro_model_path(root).name == "kokoro-v1.0.int8.onnx"
