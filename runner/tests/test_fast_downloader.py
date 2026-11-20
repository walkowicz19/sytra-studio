"""Tests for verified file selection, shard safety, and expert pager plumbing."""
import os
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sytra_runner.fast_downloader import FastHFDownloader, RemoteFile, _DownloadProgress, _DownloadStalled, _partial_bytes
from sytra_runner.shard_manager import ShardManager
from sytra_runner.backends.expert_pager import GPUExpertPager
from sytra_runner.xet_safety import XET_SAFETY_ENV, apply_xet_safety


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_verifies_lfs_sha256(tmp_path, monkeypatch, corrupt):
    data = b"model weights"
    expected = hashlib.sha256(data).hexdigest()
    sibling = SimpleNamespace(rfilename="model.safetensors", size=len(data),
                              blob_id="git-blob", lfs={"sha256": expected})
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    monkeypatch.setattr(downloader, "_model_info", lambda revision: SimpleNamespace(sha="commit", siblings=[sibling]))

    def fetch(filename, revision):
        path = downloader.repo_dir / filename
        path.write_bytes(b"X" * len(data) if corrupt else data)
        return path

    monkeypatch.setattr(downloader, "download_file", fetch)
    if corrupt:
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            downloader.download_all(tmp_path / "model")
        assert not (tmp_path / "model" / ".sytra-model.json").exists()
    else:
        downloader.download_all(tmp_path / "model")
        manifest = json.loads((tmp_path / "model" / ".sytra-model.json").read_text())
        assert manifest["files"][0]["sha256"] == expected


def test_stalled_xet_retries_once_with_http(tmp_path, monkeypatch):
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    model = tmp_path / "weights"
    model.write_bytes(b"data")
    calls = []

    def attempt(filename, revision, *, disable_xet):
        calls.append((filename, revision, disable_xet))
        if not disable_xet:
            raise _DownloadStalled("stalled")
        return model

    monkeypatch.setattr(downloader, "_download_attempt", attempt)
    assert downloader.download_file("weights", "commit") == model
    assert calls == [("weights", "commit", False), ("weights", "commit", True)]


def test_non_stall_error_does_not_trigger_fallback(tmp_path, monkeypatch):
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    calls = []

    def attempt(filename, revision, *, disable_xet):
        calls.append(disable_xet)
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(downloader, "_download_attempt", attempt)
    with pytest.raises(RuntimeError, match="unauthorized"):
        downloader.download_file("weights")
    assert calls == [False]


def test_stalled_lfs_shard_uses_verified_parallel_fallback(tmp_path, monkeypatch):
    import huggingface_hub
    import sytra_runner.http_ranges as ranges

    downloader = FastHFDownloader('org/model', cache_dir=tmp_path)
    downloader.repo_dir.mkdir(parents=True)
    data = b'weights'
    digest = hashlib.sha256(data).hexdigest()
    downloader._file_metadata = {'weights.gguf': RemoteFile('weights.gguf', len(data), sha256=digest)}
    monkeypatch.setenv('HF_TOKEN', 'local-test-token')
    monkeypatch.setattr(downloader, '_download_attempt', lambda *a, **k: (_ for _ in ()).throw(_DownloadStalled()))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_url', lambda repo, name, revision: f'https://example.test/{revision}/{name}')
    calls = []
    def transfer(url, destination, size, sha256, token):
        calls.append((url, size, sha256, token))
        destination.write_bytes(data)
        return destination
    monkeypatch.setattr(ranges, 'download_ranges', transfer)
    assert downloader.download_file('weights.gguf', 'pinned').read_bytes() == data
    assert calls == [('https://example.test/pinned/weights.gguf', len(data), digest, 'local-test-token')]


def test_download_shards_reports_failure_without_masking_it(tmp_path, monkeypatch):
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    def fail(filename, revision):
        raise ValueError("network failure")
    monkeypatch.setattr(downloader, "download_file", fail)
    with pytest.raises(RuntimeError, match="network failure"):
        downloader.download_shards(["weights"], op_id="test")


def test_download_attempt_kills_stalled_child(tmp_path, monkeypatch):
    import sytra_runner.fast_downloader as module

    class Process:
        returncode = None
        killed = False
        def poll(self):
            return -9 if self.killed else None
        def kill(self):
            self.killed = True
        def wait(self):
            return -9

    process = Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: process)
    ticks = iter([0, 16])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    with pytest.raises(_DownloadStalled):
        downloader._download_attempt("weights", "commit", disable_xet=False)
    assert process.killed


def test_partial_progress_counts_nested_hub_incomplete_files(tmp_path):
    cache = tmp_path / ".cache" / "huggingface" / "download" / "subdir"
    cache.mkdir(parents=True)
    (cache / "file.hash.incomplete").write_bytes(b"12345")
    (cache / "file.lock").write_bytes(b"ignored")
    assert _partial_bytes(tmp_path) == 5


def test_http_fallback_progress_does_not_double_count_xet_partials(tmp_path):
    cache = tmp_path / '.cache' / 'huggingface' / 'download'
    cache.mkdir(parents=True)
    (cache / 'file.incomplete').write_bytes(b'x' * 20)
    (tmp_path / 'weights.gguf.sytra-download.json').write_text('{}')
    (tmp_path / 'weights.gguf.sytra-download.part0').write_bytes(b'x' * 7)
    monitor = _DownloadProgress('org/model', tmp_path, [RemoteFile('weights.gguf', 100)], tmp_path / 'status.json', None)
    monitor.mark_active('weights.gguf')
    assert monitor._downloaded_bytes() == 7


def test_worker_uses_environment_before_import_and_keeps_token_out_of_arguments(tmp_path):
    package = tmp_path / "huggingface_hub"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import os, json\n"
        "SETTING = os.environ.get('HF_HUB_DISABLE_XET')\n"
        "def hf_hub_download(**kwargs):\n"
        "    return json.dumps({'setting': SETTING, 'token': kwargs['token'], 'revision': kwargs['revision']})\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update({"PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1])]),
                "HF_HUB_DISABLE_XET": "1", "HF_TOKEN": "secret-for-test"})
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    result = subprocess.run([sys.executable, "-m", "sytra_runner.hub_download_worker",
                             "org/model", "weights", "commit", str(tmp_path), "1"],
                            env=env, capture_output=True, text=True, check=True)
    payload = json.loads(json.loads(result.stdout.strip().split("=", 1)[1]))
    assert payload == {"setting": "1", "token": "secret-for-test", "revision": "commit"}


def test_fast_downloader_init():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("Qwen/Qwen2.5-Coder-7B", cache_dir=tmpdir, tokenless=True)
        assert downloader.tokenless is True
        assert downloader.repo_id == "Qwen/Qwen2.5-Coder-7B"


def test_xet_safety_overwrites_high_ram_defaults():
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "8gb"
    os.environ["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] = "1gb"
    apply_xet_safety()
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert os.environ["HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY"] == "false"
    assert os.environ["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "1"
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"
    assert os.environ["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] == "16mb"
    for key, expected in XET_SAFETY_ENV.items():
        assert os.environ[key] == expected


def test_child_process_env_caps_xet_for_mergekit_grandchildren():
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "8gb"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from sytra_runner.xet_safety import child_process_env

    env = child_process_env({"OMP_NUM_THREADS": "2"})
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"
    assert env["HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER"] == "16mb"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["OMP_NUM_THREADS"] == "2"


def test_downloader_is_sequential_on_every_os():
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = FastHFDownloader("org/tiny", cache_dir=tmpdir, max_workers=8)
    assert downloader.max_workers == 1


def test_downloader_caps_xet_buffers_even_when_cache_is_custom():
    os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] = "2gb"
    with tempfile.TemporaryDirectory() as tmpdir:
        FastHFDownloader("org/tiny", cache_dir=tmpdir)
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] == "64mb"
    assert os.environ["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "128mb"


def test_selects_every_split_file_for_one_gguf_quantization():
    files = [
        "README.md",
        "config.json",
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q8_0-00001-of-00002.gguf",
        "model-Q8_0-00002-of-00002.gguf",
        "mmproj-model-f16.gguf",
    ]
    selected = FastHFDownloader.select_files(files, purpose="inference", quant="Q4_K_M")
    assert "model-Q4_K_M-00001-of-00002.gguf" in selected
    assert "model-Q4_K_M-00002-of-00002.gguf" in selected
    assert "mmproj-model-f16.gguf" in selected
    assert not any("Q8_0" in name for name in selected)


def test_safetensors_selection_keeps_all_shards():
    files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    selected = FastHFDownloader.select_files(files, purpose="inference", quant="auto")
    assert {name for name in selected if name.endswith(".safetensors")} == set(files[2:])


def test_shard_manager_mock_manifest():
    class MockDownloader:
        def fetch_manifest(self, revision="main"):
            return {
                "config": {
                    "num_local_experts": 8,
                    "num_experts_per_tok": 2,
                },
                "index": {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.0.w1.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.1.w1.weight": "model-00002-of-00002.safetensors",
                        "model.layers.0.block_sparse_moe.experts.7.w1.weight": "model-00003-of-00003.safetensors",
                    }
                }
            }

    sm = ShardManager("mock/moe-model", downloader=MockDownloader())
    shards = sm.select_active_shards(auto_top_k=True)
    assert shards == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model-00003-of-00003.safetensors",
    ]


def test_python_expert_pager_cannot_bypass_the_native_runtime():
    with pytest.raises(RuntimeError, match="native Rust runtime"):
        GPUExpertPager(num_experts=8, vram_limit_mb=2048, quant_bits=4)


def test_cas_xet_transport_error_triggers_http_fallback_while_auth_401_fails_fast(tmp_path, monkeypatch):
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    out = tmp_path / "model.safetensors"
    out.write_bytes(b"ok-weights")
    calls = []

    def xet_cas_fail(filename, revision, *, disable_xet):
        calls.append(("cas", disable_xet))
        if not disable_xet:
            raise RuntimeError("hf_xet cas_client transport error: reset by peer")
        return out

    monkeypatch.setattr(downloader, "_download_attempt", xet_cas_fail)
    assert downloader.download_file("model.safetensors", "commit") == out
    assert calls == [("cas", False), ("cas", True)]

    calls.clear()
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_token_999")

    def auth_401(filename, revision, *, disable_xet):
        calls.append(("auth", disable_xet))
        raise RuntimeError("401 Unauthorized for token hf_super_secret_token_999")

    monkeypatch.setattr(downloader, "_download_attempt", auth_401)
    with pytest.raises(RuntimeError, match="401 Unauthorized") as exc_info:
        downloader.download_file("model.safetensors", "commit")
    assert calls == [("auth", False)]
    assert "hf_super_secret_token_999" not in str(exc_info.value)


def test_download_all_skips_already_verified_shards_and_deletes_corrupt_files(tmp_path, monkeypatch):
    shard1 = b"verified-shard-one"
    shard2 = b"fresh-shard-two"
    dest = tmp_path / "model"
    dest.mkdir()
    (dest / "model-00001-of-00002.safetensors").write_bytes(shard1)

    siblings = [
        SimpleNamespace(
            rfilename="model-00001-of-00002.safetensors",
            size=len(shard1),
            blob_id="b1",
            lfs={"sha256": hashlib.sha256(shard1).hexdigest()},
        ),
        SimpleNamespace(
            rfilename="model-00002-of-00002.safetensors",
            size=len(shard2),
            blob_id="b2",
            lfs={"sha256": hashlib.sha256(shard2).hexdigest()},
        ),
    ]
    downloader = FastHFDownloader("org/model", cache_dir=tmp_path)
    monkeypatch.setattr(downloader, "_model_info", lambda rev: SimpleNamespace(sha="rev-pin", siblings=siblings))

    fetched = []

    def mock_download(filename, revision):
        fetched.append(filename)
        path = downloader.repo_dir / filename
        path.write_bytes(shard2)
        return path

    monkeypatch.setattr(downloader, "download_file", mock_download)
    result = downloader.download_all(dest)
    assert fetched == ["model-00002-of-00002.safetensors"]
    assert len(result) == 2
    assert (dest / ".sytra-model.json").is_file()

    # Now remove shard2 and deliver corrupt bytes for shard2 -> corrupt file must be deleted cleanly
    (dest / ".sytra-model.json").unlink()
    (dest / "model-00002-of-00002.safetensors").unlink()

    def corrupt_download(fn, revision):
        p = downloader.repo_dir / fn
        p.write_bytes(b"X" * len(shard2))
        return p

    monkeypatch.setattr(downloader, "download_file", corrupt_download)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        downloader.download_all(dest)
    assert not (dest / "model-00002-of-00002.safetensors").exists()
    assert not (dest / ".sytra-model.json").exists()


def test_downloader_redacts_tokens_and_proxy_passwords_in_status_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_top_secret_value_777")
    downloader = FastHFDownloader("org/private", cache_dir=tmp_path)

    def fail_preflight(_rev):
        raise RuntimeError(
            "failed via https://alice:proxyPass99@proxy.local using Bearer hf_top_secret_value_777"
        )

    monkeypatch.setattr(downloader, "_model_info", fail_preflight)
    dest = tmp_path / "dest"
    with pytest.raises(RuntimeError) as exc_info:
        downloader.download_all(dest)

    err_msg = str(exc_info.value)
    assert "hf_top_secret_value_777" not in err_msg
    assert "proxyPass99" not in err_msg
    status = json.loads((dest / ".download_status.json").read_text(encoding="utf-8"))
    assert "hf_top_secret_value_777" not in status["error"]
    assert "proxyPass99" not in status["error"]
