import http.client
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sytra_runner.idle_proxy import IdleState, make_proxy, serve_with_idle_timeout


def test_proxy_streaming_keeps_model_active_until_response_finishes():
    release = threading.Event()
    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args): pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')
        def do_POST(self):
            assert self.rfile.read(int(self.headers['Content-Length'])) == b'prompt'
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'a')
            self.wfile.flush()
            release.wait(3)
            self.wfile.write(b'b')

    backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    state = IdleState(.1)
    proxy = make_proxy('127.0.0.1', 0, backend.server_port, state)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (backend, proxy)]
    for thread in threads: thread.start()
    try:
        connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
        connection.request('GET', '/health')
        assert connection.getresponse().read() == b'ok'
        connection.close()
        connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
        connection.request('POST', '/v1/chat/completions', body=b'prompt')
        response = connection.getresponse()
        assert response.read(1) == b'a'
        time.sleep(.2)
        assert not state.expire()
        release.set()
        assert response.read() == b'b'
        connection.close()
        deadline = time.monotonic() + 2
        while state.active and time.monotonic() < deadline:
            time.sleep(.01)
        assert not state.active
        assert not state.expire()
        time.sleep(.15)
        assert state.expire()
        assert not state.begin()
    finally:
        release.set()
        for server in (proxy, backend): server.shutdown()
        for thread in threads: thread.join()
        for server in (proxy, backend): server.server_close()


def test_health_checks_do_not_keep_idle_model_alive():
    state = IdleState(.01)
    state.mark_ready()
    time.sleep(.02)
    state.mark_ready()
    assert state.expire()


def test_proxy_exposes_configured_chat_temperature():
    state = IdleState(0)
    proxy = make_proxy('127.0.0.1', 0, 1, state, temperature=0.25)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
        conn.request('GET', '/sytra/preferences', headers={'Origin': 'http://tauri.localhost'})
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {'temperature': 0.25}
        conn.close()
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()


def test_proxy_exposes_context_window_and_model():
    state = IdleState(0)
    proxy = make_proxy('127.0.0.1', 0, 1, state, temperature=0.3, context_window=8192, model_name='qwen2.5-7b')
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
        conn.request('GET', '/sytra/preferences', headers={'Origin': 'http://tauri.localhost'})
        response = conn.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {
            'temperature': 0.3,
            'context_window': 8192,
            'model': 'qwen2.5-7b',
        }
        conn.close()
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()


def test_proxy_allows_only_the_desktop_and_local_dev_origins():
    state = IdleState(0)
    proxy = make_proxy('127.0.0.1', 0, 1, state)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
        conn.request('OPTIONS', '/v1/chat/completions', headers={'Origin': 'http://tauri.localhost',
                     'Access-Control-Request-Method': 'POST'})
        response = conn.getresponse()
        assert response.status == 204
        assert response.getheader('Access-Control-Allow-Origin') == 'http://tauri.localhost'
        response.read()
        conn.close()
        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
        conn.request('POST', '/v1/chat/completions', body='{}', headers={'Origin': 'https://evil.example'})
        try:
            response = conn.getresponse()
            assert response.status == 403
            assert response.getheader('Access-Control-Allow-Origin') is None
            response.read()
        except (ConnectionResetError, ConnectionAbortedError):
            # Windows can reset the socket when rejecting an unread POST body.
            pass
        conn.close()
        state.mark_ready()
        assert not state.expire()
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()


def test_mcp_proxy_requires_session_token_and_trusted_origin():
    class Broker:
        def execute(self, operation, payload): return {'operation': operation, 'query': payload['query']}
    proxy = make_proxy('127.0.0.1', 0, 1, IdleState(0), broker=Broker(), token='test-secret')
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    def request(headers):
        connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=2)
        try:
            connection.request('POST', '/sytra/mcp/search', body='{"query":"weather"}', headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        except (ConnectionResetError, ConnectionAbortedError):
            return None, b''
        finally:
            connection.close()
    try:
        assert request({'Origin': 'http://tauri.localhost'})[0] in (403, None)
        assert request({'X-Sytra-Session': 'test-secret', 'Origin': 'https://evil.example'})[0] in (403, None)
        assert request({'X-Sytra-Session': 'test-secret'})[0] in (403, None)
        status, body = request({'X-Sytra-Session': 'test-secret', 'Origin': 'http://tauri.localhost'})
        assert status == 200
        assert json.loads(body) == {'operation': 'search', 'query': 'weather'}
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()


def test_idle_timeout_stops_actual_backend_process():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    children = []
    result = []
    code = """import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')
 def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    def launch(backend_port):
        child = subprocess.Popen([sys.executable, '-u', '-c', code, str(backend_port)])
        children.append(child)
        return child
    thread = threading.Thread(target=lambda: result.append(serve_with_idle_timeout('127.0.0.1', port, 1, launch)), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                connection.request('GET', '/health')
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status == 200: break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise AssertionError('backend did not become ready')
            time.sleep(.05)
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert result and children[0].poll() is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()


def test_proxy_concurrent_requests_and_external_temperature_injection():
    seen_payloads = []
    release = threading.Event()
    both_inside = threading.Event()
    entered = 0
    lock = threading.Lock()

    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')

        def do_POST(self):
            nonlocal entered
            length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(length))
            with lock:
                seen_payloads.append(body)
                entered += 1
                if entered == 2:
                    both_inside.set()
            release.wait(3)
            out = json.dumps({'temperature_seen': body.get('temperature')}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    state = IdleState(0.1)
    proxy = make_proxy('127.0.0.1', 0, backend.server_port, state, temperature=0.35)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (backend, proxy)]
    for t in threads:
        t.start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
        conn.request('GET', '/health')
        assert conn.getresponse().read() == b'ok'
        conn.close()

        results = {}
        def send_req(name, payload):
            c = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
            c.request(
                'POST',
                '/v1/chat/completions',
                body=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
            )
            r = c.getresponse()
            results[name] = json.loads(r.read())
            c.close()

        t1 = threading.Thread(target=send_req, args=('implicit', {'model': 'm', 'messages': []}))
        t2 = threading.Thread(target=send_req, args=('explicit', {'model': 'm', 'messages': [], 'temperature': 1.2}))
        t1.start()
        t2.start()
        assert both_inside.wait(2)
        time.sleep(0.15)
        assert not state.expire()
        release.set()
        t1.join(timeout=3)
        t2.join(timeout=3)

        assert results['implicit'] == {'temperature_seen': 0.35}
        assert results['explicit'] == {'temperature_seen': 1.2}
        deadline = time.monotonic() + 2
        while state.active and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not state.active
        time.sleep(0.15)
        assert state.expire()
    finally:
        release.set()
        for s in (proxy, backend):
            s.shutdown()
        for t in threads:
            t.join()
        for s in (proxy, backend):
            s.server_close()


def test_proxy_aborted_client_decrements_active_and_restarts_cleanly():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]

    code = """import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')
 def do_POST(self):
  length = int(self.headers.get('Content-Length', '0'))
  self.rfile.read(length)
  self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')
 def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    children = []
    def launch(backend_port):
        child = subprocess.Popen([sys.executable, '-u', '-c', code, str(backend_port)])
        children.append(child)
        return child

    for _cycle in range(2):
        result = []
        thread = threading.Thread(
            target=lambda: result.append(serve_with_idle_timeout('127.0.0.1', port, 1, launch, temperature=0.4)),
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while True:
                try:
                    c = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                    c.request('GET', '/health')
                    r = c.getresponse()
                    r.read()
                    c.close()
                    if r.status == 200:
                        break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise AssertionError('backend did not start')
                time.sleep(0.05)

            # Simulate an abrupt mid-POST disconnect before body is sent, then verify clean TTL shutdown & restart.
            sock = socket.create_connection(('127.0.0.1', port), timeout=2)
            sock.sendall(b'POST /v1/chat/completions HTTP/1.0\r\nContent-Length: 500\r\n\r\npartial')
            sock.close()

            thread.join(timeout=10)
            assert not thread.is_alive()
            assert result == [0]
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
