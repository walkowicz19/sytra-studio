"""Loopback inference proxy which unloads its backend after actual inactivity."""
from __future__ import annotations

import http.client
import hmac
import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .subprocess_utils import silent_subprocess_kwargs


_HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
                'te', 'trailer', 'transfer-encoding', 'upgrade', 'host'}
_UI_ORIGINS = {'http://tauri.localhost', 'https://tauri.localhost', 'tauri://localhost',
               'http://localhost:5173', 'http://127.0.0.1:5173'}
_COMPLETION_PATHS = {'/v1/chat/completions', '/v1/completions', '/chat/completions', '/completions', '/v1/messages'}
_MAX_INJECT_BODY_BYTES = 8 * 1024 * 1024


def _maybe_inject_temperature(method: str, path: str, headers, body: bytes, temperature: float | None) -> bytes:
    """Inject default temperature into JSON completion requests when omitted by external API clients."""
    if temperature is None or method != 'POST' or path not in _COMPLETION_PATHS:
        return body
    ctype = (headers.get('Content-Type') or '').lower()
    if 'json' not in ctype and not (not ctype and body.lstrip().startswith(b'{')):
        return body
    try:
        payload = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body
    if not isinstance(payload, dict) or 'temperature' in payload:
        return body
    payload['temperature'] = temperature
    return json.dumps(payload, ensure_ascii=False).encode('utf-8')


class IdleState:
    def __init__(self, ttl: float):
        self.ttl = ttl
        self.lock = threading.Lock()
        self.active = 0
        self.ready = False
        self.stopping = False
        self.last_activity = time.monotonic()

    def begin(self) -> bool:
        with self.lock:
            if self.stopping:
                return False
            self.active += 1
            return True

    def end(self) -> None:
        with self.lock:
            self.active -= 1
            self.last_activity = time.monotonic()

    def mark_ready(self) -> None:
        with self.lock:
            if not self.ready:
                self.ready = True
                self.last_activity = time.monotonic()

    def expire(self) -> bool:
        with self.lock:
            if self.ttl > 0 and self.ready and not self.active and time.monotonic() - self.last_activity >= self.ttl:
                self.stopping = True
                return True
            return False


def make_proxy(host: str, port: int, backend_port: int, state: IdleState, *, broker=None, token=None,
               temperature: float | None = None, context_window: int | None = None,
               model_name: str | None = None) -> ThreadingHTTPServer:
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('Idle proxy only supports loopback')
    if temperature is not None and (type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError('Chat temperature must be between 0 and 2')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_args):
            pass

        def end_headers(self):
            origin = self.headers.get('Origin')
            if origin in _UI_ORIGINS:
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
            super().end_headers()

        def do_OPTIONS(self):
            if self.headers.get('Origin') not in _UI_ORIGINS:
                self.send_error(403)
                return
            self.send_response(204)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Sytra-Session')
            self.end_headers()

        def handle_request(self):
            if self.headers.get('Origin') and self.headers['Origin'] not in _UI_ORIGINS:
                self.send_error(403, 'Origin not permitted')
                return
            health = self.command == 'GET' and urlsplit(self.path).path == '/health'
            if not health and not state.begin():
                self.send_error(503, 'Model server is stopping')
                return
            connection = None
            sent_headers = False
            try:
                path = urlsplit(self.path).path
                if path == '/sytra/preferences' and self.command == 'GET':
                    prefs: dict[str, object] = {'temperature': temperature}
                    if isinstance(context_window, int) and context_window > 0:
                        prefs['context_window'] = context_window
                    if isinstance(model_name, str) and model_name:
                        prefs['model'] = model_name
                    body = json.dumps(prefs).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path.startswith('/sytra/mcp/'):
                    self.handle_mcp(path)
                    return
                if self.headers.get('Transfer-Encoding'):
                    self.send_error(411, 'Content-Length required')
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except ValueError:
                    self.send_error(400, 'Invalid Content-Length')
                    return
                if length < 0:
                    self.send_error(400, 'Invalid Content-Length')
                    return
                connection = http.client.HTTPConnection('127.0.0.1', backend_port)
                headers = {key: value for key, value in self.headers.items() if key.lower() not in _HOP_HEADERS}
                if (
                    temperature is not None
                    and self.command == 'POST'
                    and path in _COMPLETION_PATHS
                    and 0 < length <= _MAX_INJECT_BODY_BYTES
                ):
                    raw_chunks = []
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 65536))
                        if not chunk:
                            raise ConnectionError('Client request ended prematurely')
                        raw_chunks.append(chunk)
                        remaining -= len(chunk)
                    body = _maybe_inject_temperature(self.command, path, self.headers, b''.join(raw_chunks), temperature)
                    headers = {key: value for key, value in headers.items() if key.lower() != 'content-length'}
                    headers['Content-Length'] = str(len(body))
                    connection.putrequest(self.command, self.path, skip_accept_encoding=True)
                    for key, value in headers.items():
                        connection.putheader(key, value)
                    connection.endheaders()
                    if body:
                        connection.send(body)
                else:
                    connection.putrequest(self.command, self.path, skip_accept_encoding=True)
                    for key, value in headers.items():
                        connection.putheader(key, value)
                    connection.endheaders()
                    while length:
                        chunk = self.rfile.read(min(length, 65536))
                        if not chunk:
                            raise ConnectionError('Client request ended prematurely')
                        connection.send(chunk)
                        length -= len(chunk)
                response = connection.getresponse()
                if health and 200 <= response.status < 300:
                    state.mark_ready()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() not in _HOP_HEADERS and not key.lower().startswith('access-control-'):
                        self.send_header(key, value)
                self.end_headers()
                sent_headers = True
                if self.command != 'HEAD':
                    while True:
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (OSError, http.client.HTTPException):
                if not sent_headers:
                    try:
                        self.send_error(502, 'Model server unavailable')
                    except OSError:
                        pass
            finally:
                if connection is not None:
                    connection.close()
                if not health:
                    state.end()

        def handle_mcp(self, path):
            supplied = self.headers.get('X-Sytra-Session', '')
            if (self.command != 'POST' or broker is None or not token or
                    self.headers.get('Origin') not in _UI_ORIGINS or
                    not hmac.compare_digest(supplied, token)):
                self.send_error(403, 'MCP access denied')
                return
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Content-Length required')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Expected an object')
                operation = path.removeprefix('/sytra/mcp/')
                if operation == 'cancel':
                    result = {'cancelled': broker.cancel(payload.get('request_id', ''))}
                else:
                    result = broker.execute(operation, payload)
                body = json.dumps(result, ensure_ascii=False).encode('utf-8')
                if len(body) > 2 * 1024 * 1024:
                    raise ValueError('MCP result exceeds 2 MiB')
                status = 200
            except Exception as exc:
                body = json.dumps({'error': str(exc)}, ensure_ascii=False).encode('utf-8')
                status = 400
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = handle_request

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve_with_idle_timeout(host: str, port: int, ttl: int, launch, *, temperature: float | None = None,
                            context_window: int | None = None, model_name: str | None = None) -> int:
    """Run an engine on an internal port and stop it when no request is active."""
    if ttl < 0:
        raise ValueError('Idle timeout must be non-negative')
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1', 0))
        backend_port = reserved.getsockname()[1]
    state = IdleState(ttl)
    token = os.environ.pop('SYTRA_MCP_SESSION_TOKEN', None)
    broker = None
    if token:
        from .mcp_chat import McpBroker
        broker = McpBroker()
    try:
        server = make_proxy(host, port, backend_port, state, broker=broker, token=token,
                            temperature=temperature, context_window=context_window, model_name=model_name)
    except BaseException:
        if broker:
            broker.close()
        raise
    try:
        process = launch(backend_port)
    except BaseException:
        server.server_close()
        if broker:
            broker.close()
        raise

    expired = threading.Event()
    def monitor():
        while process.poll() is None:
            if state.expire():
                expired.set()
                _stop_tree(process)
                break
            time.sleep(min(1, ttl / 4) if ttl else 1)
        server.shutdown()

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        server.serve_forever(poll_interval=0.25)
        watcher.join()
        status = process.wait()
        return 0 if expired.is_set() else status
    finally:
        server.server_close()
        if process.poll() is None:
            _stop_tree(process)
        if broker:
            broker.close()


def _stop_tree(process) -> None:
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                           **silent_subprocess_kwargs())
        except OSError:
            pass
    else:
        # Descendants share the host's process group, so killing that group here
        # would also kill the proxy. Enumerate descendants instead.
        try:
            result = subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
        except (OSError, subprocess.CalledProcessError):
            lines = []
        parents = {}
        for line in lines:
            try:
                pid, ppid = map(int, line.split())
                parents[pid] = ppid
            except ValueError:
                continue
        descendants = {process.pid}
        while True:
            expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        for pid in descendants - {process.pid}:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
