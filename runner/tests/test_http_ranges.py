import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request

import pytest
import sytra_runner.http_ranges as ranges
from sytra_runner.http_ranges import download_ranges


def test_redirect_drops_credentials_on_origin_change():
    request = Request('https://huggingface.co/file', headers={'Authorization': 'Bearer secret', 'Range': 'bytes=0-1'})
    redirect = ranges._SafeRedirect()
    external = redirect.redirect_request(request, None, 302, '', {}, 'https://cdn.example/file')
    assert external.get_header('Authorization') is None
    assert external.get_header('Range') == 'bytes=0-1'
    same_origin = redirect.redirect_request(request, None, 302, '', {}, 'https://huggingface.co/other')
    assert same_origin.get_header('Authorization') == 'Bearer secret'


class Response:
    def __init__(self, data, start, end, *, bad_range=False, interrupted=False):
        self.data, self.start, self.end, self.interrupted = data, start, end, interrupted
        self.status = 206
        self.offset = 0
        self.headers = {'Content-Range': 'wrong' if bad_range else f'bytes {start}-{end}/{len(data)}'}

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self, chunk_size):
        body = self.data[self.start:self.end + 1]
        if self.interrupted:
            if not self.offset:
                self.offset = 2
                return body[:2]
            raise ConnectionError('interrupted')
        result = body[self.offset:self.offset + chunk_size]
        self.offset += len(result)
        return result


def test_real_http_range_responses(tmp_path, monkeypatch):
    data = b'local range transfer' * 100
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers['Range'][6:].split('-'))
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.send_header('Content-Length', str(end - start + 1))
            self.end_headers()
            self.wfile.write(data[start:end + 1])
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    try:
        target = tmp_path / 'model.gguf'
        download_ranges(f'http://127.0.0.1:{server.server_port}/file', target,
                        len(data), hashlib.sha256(data).hexdigest())
        assert target.read_bytes() == data
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_parallel_ranges_resume_and_publish_only_after_verification(tmp_path, monkeypatch):
    data = bytes(range(100))
    digest = hashlib.sha256(data).hexdigest()
    calls = []
    fail_once = {25}

    def get(url, headers):
        start, end = map(int, headers['Range'][6:].split('-'))
        calls.append((start, end, headers.get('Authorization')))
        fail = start in fail_once
        fail_once.discard(start)
        return Response(data, start, end, interrupted=fail)

    monkeypatch.setattr(ranges, '_open_range', get)
    target = tmp_path / 'model.gguf'
    with pytest.raises(ConnectionError):
        download_ranges('https://example.test/file', target, len(data), digest, 'secret')
    assert not target.exists()
    calls.clear()
    assert download_ranges('https://example.test/file', target, len(data), digest, 'secret') == target
    assert target.read_bytes() == data
    assert (27, 49, 'Bearer secret') in calls
    assert all(call in {(27, 49, 'Bearer secret'), (50, 74, 'Bearer secret'),
                        (75, 99, 'Bearer secret')} for call in calls)
    assert list(tmp_path.iterdir()) == [target]


def test_bad_content_range_is_never_published(tmp_path, monkeypatch):
    data = b'abcdefgh'
    monkeypatch.setattr(ranges, '_open_range', lambda *a, **k: Response(data, 0, 1, bad_range=True))
    target = tmp_path / 'model.safetensors'
    with pytest.raises(RuntimeError, match='requested byte range'):
        download_ranges('https://example.test', target, len(data), hashlib.sha256(data).hexdigest())
    assert not target.exists()


def test_corrupt_complete_ranges_are_discarded(tmp_path, monkeypatch):
    actual = b'abcdefgh'
    monkeypatch.setattr(ranges, '_open_range', lambda url, headers: Response(
        actual, *map(int, headers['Range'][6:].split('-'))))
    target = tmp_path / 'model.gguf'
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        download_ranges('https://example.test', target, len(actual), hashlib.sha256(b'12345678').hexdigest())
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def _start_server(handler_cls):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_non_range_200_ok_fallback_real_server(tmp_path, monkeypatch):
    data = (b'no-range-payload-' * 1500)[:24576]
    seen_ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_ranges.append(self.headers.get('Range'))
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server, thread = _start_server(Handler)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    try:
        target = tmp_path / 'model.safetensors'
        out = download_ranges(
            f'http://127.0.0.1:{server.server_port}/weights',
            target,
            len(data),
            hashlib.sha256(data).hexdigest(),
            connections=4,
        )
        assert out == target
        assert target.read_bytes() == data
        # Clean staging/part cleanup leaves only final target
        assert list(tmp_path.iterdir()) == [target]
        assert any(r is None for r in seen_ranges)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_remote_etag_change_resets_resume_and_rejects_corrupt_mutation(tmp_path, monkeypatch):
    v1 = b'version-one-' * 40
    v2 = b'version-two-' * 40
    state = {'etag': '"v1"', 'payload': v1, 'drop_once': True, 'ranges': []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers['Range']
            start, end = map(int, rng[6:].split('-'))
            state['ranges'].append((start, end, state['etag']))
            chunk = state['payload'][start : end + 1]
            self.send_response(206)
            self.send_header('ETag', state['etag'])
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(state["payload"])}')
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            if state['drop_once'] and start > 0:
                state['drop_once'] = False
                self.wfile.write(chunk[:5])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(chunk)

        def log_message(self, *args):
            pass

    server, thread = _start_server(Handler)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    try:
        target = tmp_path / 'weights.safetensors'
        # Interrupt mid-range under ETag "v1" (retains partial chunks)
        with pytest.raises(Exception):
            download_ranges(
                f'http://127.0.0.1:{server.server_port}/file',
                target,
                len(v1),
                hashlib.sha256(v1).hexdigest(),
                connections=2,
            )
        assert not target.exists()

        # Server rotates ETag to "v2" cleanly while payload matches v1 SHA -> restarts cleanly from byte 0
        state['etag'] = '"v2"'
        state['ranges'].clear()
        download_ranges(
            f'http://127.0.0.1:{server.server_port}/file',
            target,
            len(v1),
            hashlib.sha256(v1).hexdigest(),
            connections=2,
        )
        assert target.read_bytes() == v1
        assert (0, len(v1) // 2 - 1, '"v2"') in state['ranges']
        assert list(tmp_path.iterdir()) == [target]

        # Remote mutates content AND ETag -> old hash fails without publishing target
        target2 = tmp_path / 'mutated.safetensors'
        state['payload'] = v2
        state['etag'] = '"v3"'
        with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
            download_ranges(
                f'http://127.0.0.1:{server.server_port}/file',
                target2,
                len(v1),
                hashlib.sha256(v1).hexdigest(),
                connections=2,
            )
        assert not target2.exists()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_proxy_forwarding_and_auth_error_cleanup_with_secret_redaction(tmp_path, monkeypatch):
    data = b'proxied-shard-bytes' * 50
    proxied_requests = []

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            proxied_requests.append((self.path, self.headers.get('Authorization')))
            if 'deny' in self.path:
                self.send_response(403)
                self.end_headers()
                return
            rng = self.headers['Range']
            start, end = map(int, rng[6:].split('-'))
            body = data[start : end + 1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(data)}')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    proxy, thread = _start_server(ProxyHandler)
    monkeypatch.delenv('NO_PROXY', raising=False)
    monkeypatch.delenv('no_proxy', raising=False)
    monkeypatch.setenv('HTTP_PROXY', f'http://proxyuser:supersecretpass@127.0.0.1:{proxy.server_port}')
    try:
        target = tmp_path / 'proxied.gguf'
        download_ranges(
            'http://upstream.example.test/ok/model.gguf',
            target,
            len(data),
            hashlib.sha256(data).hexdigest(),
            token='hf_secret_proxy_token_12345',
            connections=2,
        )
        assert target.read_bytes() == data
        assert any(auth == 'Bearer hf_secret_proxy_token_12345' for _, auth in proxied_requests)

        denied = tmp_path / 'denied.gguf'
        with pytest.raises(RuntimeError) as exc_info:
            download_ranges(
                'http://upstream.example.test/deny/model.gguf',
                denied,
                len(data),
                hashlib.sha256(data).hexdigest(),
                token='hf_secret_proxy_token_12345',
                connections=2,
            )
        assert 'supersecretpass' not in str(exc_info.value)
        assert 'hf_secret_proxy_token_12345' not in str(exc_info.value)
        assert not denied.exists()
        assert not any(p.name.startswith('denied.gguf.sytra-download') for p in tmp_path.iterdir())
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()


def test_large_multichunk_range_streaming_and_cleanup(tmp_path, monkeypatch):
    # ~3.1 MiB (>1 MiB streaming chunk boundary) across 4 parallel connections
    payload = (b'0123456789ABCDEF' * 204800)
    digest = hashlib.sha256(payload).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers['Range'][6:].split('-'))
            chunk = payload[start : end + 1]
            self.send_response(206)
            self.send_header('ETag', '"large-v1"')
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, *args):
            pass

    server, thread = _start_server(Handler)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    try:
        target = tmp_path / 'large-shard.safetensors'
        download_ranges(
            f'http://127.0.0.1:{server.server_port}/large',
            target,
            len(payload),
            digest,
            connections=4,
        )
        assert target.stat().st_size == len(payload)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        assert list(tmp_path.iterdir()) == [target]
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
