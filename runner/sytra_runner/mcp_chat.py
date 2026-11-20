"""Local MCP stdio sessions and a bounded in-memory tool search index."""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import re
import sqlite3
import threading
import time
from typing import Any

from .subprocess_utils import silent_subprocess_kwargs

_MAX_MESSAGE = 4 * 1024 * 1024
_MAX_TOOLS = 1024
_CANCELLATION_TTL = 90.0
_MAX_CANCELLATIONS = 4096


def _valid_request_id(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9-]{1,64}', value) is not None


class McpError(RuntimeError):
    pass


class ToolIndex:
    """SQLite FTS5/BM25 index over currently connected servers only."""

    def __init__(self):
        self.db = sqlite3.connect(':memory:')
        self.db.execute('CREATE TABLE tools (server TEXT, name TEXT, definition TEXT, PRIMARY KEY(server,name))')
        self.db.execute('CREATE VIRTUAL TABLE tool_fts USING fts5(server UNINDEXED, name, description, tokenize="unicode61")')

    def replace(self, server: str, tools: list[dict[str, Any]]) -> None:
        if not server or len(tools) > _MAX_TOOLS:
            raise McpError('Invalid server or too many tools')
        rows = []
        seen = set()
        for tool in tools:
            if not isinstance(tool, dict):
                raise McpError('Invalid MCP tool definition')
            name, schema = tool.get('name'), tool.get('inputSchema')
            description = tool.get('description', '')
            if not isinstance(name, str) or not name or name in seen or not isinstance(schema, dict) or not isinstance(description, str):
                raise McpError('Invalid or duplicate MCP tool definition')
            if len(name) > 256 or len(description) > 32_768 or len(json.dumps(tool)) > 256_000:
                raise McpError('MCP tool definition exceeds the allowed size')
            seen.add(name)
            rows.append((server, name, description, json.dumps(tool, ensure_ascii=False)))
        with self.db:
            self.db.execute('DELETE FROM tools WHERE server=?', (server,))
            self.db.execute('DELETE FROM tool_fts WHERE server=?', (server,))
            self.db.executemany('INSERT INTO tools VALUES (?, ?, ?)', [(s, n, d) for s, n, _, d in rows])
            self.db.executemany('INSERT INTO tool_fts(server,name,description) VALUES (?, ?, ?)', [(s, n, d) for s, n, d, _ in rows])

    def remove(self, server: str) -> None:
        self.replace(server, [])

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if not 1 <= limit <= 20:
            raise ValueError('Tool search limit must be between 1 and 20')
        terms = re.findall(r'[^\W_]+', query.lower(), flags=re.UNICODE)[:16]
        if not terms:
            return []
        expression = ' OR '.join('"' + term + '"' for term in terms)
        rows = self.db.execute('''SELECT f.server, t.definition FROM tool_fts AS f
            JOIN tools AS t ON t.server=f.server AND t.name=f.name
            WHERE tool_fts MATCH ? ORDER BY bm25(tool_fts, 0.0, 4.0, 1.0), f.server, f.name LIMIT ?''',
            (expression, limit)).fetchall()
        return [{'server': server, 'tool': json.loads(definition)} for server, definition in rows]


class StdioSession:
    """One JSON-RPC stdio session, supporting concurrent calls and cancellation."""

    def __init__(self, process: asyncio.subprocess.Process):
        self.process = process
        self.pending: dict[int, asyncio.Future] = {}
        self.next_id = 0
        self.write_lock = asyncio.Lock()
        self.changed = asyncio.Event()
        self.reader = asyncio.create_task(self._read_loop())

    @classmethod
    async def start(
        cls,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> 'StdioSession':
        if not isinstance(argv, list) or not argv or len(argv) > 128 or any(not isinstance(arg, str) or not arg or '\x00' in arg or len(arg) > 32768 for arg in argv):
            raise McpError('MCP command must be a non-empty argument array')
        merged_env = None
        if env is not None:
            if not isinstance(env, dict) or len(env) > 64 or any(
                not isinstance(k, str) or not isinstance(v, str) or not k or '=' in k or '\x00' in k or '\x00' in v
                for k, v in env.items()
            ):
                raise McpError('MCP env must be a string map of at most 64 entries')
            import os
            merged_env = {**os.environ, **env}
        if cwd is not None and (not isinstance(cwd, str) or not cwd or '\x00' in cwd):
            raise McpError('MCP cwd must be a valid path string')
        process = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=merged_env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, limit=_MAX_MESSAGE,
            **silent_subprocess_kwargs())
        session = cls(process)
        try:
            result = await session.request('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                'clientInfo': {'name': 'sytra-studio', 'version': '1.3.1'}}, timeout=20, cancellable=False)
            if not isinstance(result.get('protocolVersion'), str) or 'tools' not in result.get('capabilities', {}):
                raise McpError('Server did not advertise the MCP tools capability')
            await session.notify('notifications/initialized', {})
            return session
        except BaseException:
            await session.close()
            raise

    async def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
        if len(data) > _MAX_MESSAGE:
            raise McpError('MCP message is too large')
        async with self.write_lock:
            if self.process.stdin is None or self.process.stdin.is_closing():
                raise McpError('MCP server is disconnected')
            self.process.stdin.write(data)
            await self.process.stdin.drain()

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({'jsonrpc': '2.0', 'method': method, 'params': params})

    async def request(self, method: str, params: dict[str, Any], *, timeout: float = 60,
                      cancellable: bool = True) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._send({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params})
            result = await asyncio.wait_for(future, timeout)
            if not isinstance(result, dict):
                raise McpError('Invalid MCP response')
            return result
        except (asyncio.CancelledError, asyncio.TimeoutError):
            if cancellable and not future.done():
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.notify('notifications/cancelled', {'requestId': request_id, 'reason': 'Cancelled by client'}))
            # wait_for may cancel the future before the exception is raised;
            # the request itself can still be running at the server.
            elif cancellable and future.cancelled():
                with contextlib.suppress(Exception):
                    await asyncio.shield(self.notify('notifications/cancelled', {'requestId': request_id, 'reason': 'Cancelled by client'}))
            raise
        finally:
            self.pending.pop(request_id, None)

    async def _read_loop(self) -> None:
        error = McpError('MCP server closed its output')
        try:
            assert self.process.stdout is not None
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
                    raise McpError('Invalid MCP JSON-RPC message')
                if 'method' in message:
                    if message['method'] == 'notifications/tools/list_changed':
                        self.changed.set()
                    if 'id' in message:
                        await self._send({'jsonrpc': '2.0', 'id': message['id'],
                            'error': {'code': -32601, 'message': 'Client method not supported'}})
                    continue
                future = self.pending.get(message.get('id'))
                if future is None or future.done():
                    continue
                if 'error' in message:
                    future.set_exception(McpError(str(message['error'])))
                else:
                    future.set_result(message.get('result'))
        except (OSError, ValueError, TypeError, McpError) as exc:
            error = McpError(f'MCP transport failed: {exc}')
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursors: set[str] = set()
        cursor = None
        while True:
            page = await self.request('tools/list', {'cursor': cursor} if cursor else {})
            if not isinstance(page.get('tools'), list):
                raise McpError('Invalid MCP tools page')
            tools.extend(page['tools'])
            if len(tools) > _MAX_TOOLS:
                raise McpError('MCP server returned too many tools')
            cursor = page.get('nextCursor')
            if cursor is None:
                return tools
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise McpError('Invalid MCP pagination cursor')
            cursors.add(cursor)

    async def call(self, name: str, arguments: dict[str, Any], *, timeout: float = 60) -> dict[str, Any]:
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError('Tool name and object arguments are required')
        return await self.request('tools/call', {'name': name, 'arguments': arguments}, timeout=timeout)

    async def close(self) -> None:
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.reader


class McpRegistry:
    """Connect user-configured stdio servers and maintain a live search index."""

    def __init__(self):
        self.index = ToolIndex()
        self.sessions: dict[str, StdioSession] = {}
        self.watchers: dict[str, asyncio.Task] = {}
        self.refresh_errors: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.sandbox = None

    async def connect(
        self,
        server: str,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        async with self.lock:
            await self._connect(server, argv, cwd=cwd, env=env)

    async def _connect(
        self,
        server: str,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', server) or server == 'sandbox' or server in self.sessions or len(self.sessions) >= 16:
            raise McpError('Invalid, duplicate, or too many MCP servers')
        session = await StdioSession.start(argv, cwd=cwd, env=env)
        try:
            tools = await session.list_tools()
            self.index.replace(server, tools)
        except BaseException:
            await session.close()
            raise
        self.sessions[server] = session
        self.watchers[server] = asyncio.create_task(self._watch(server, session))

    async def _watch(self, server: str, session: StdioSession) -> None:
        exit_task = asyncio.create_task(session.process.wait())
        changed_task = asyncio.create_task(session.changed.wait())
        try:
            while True:
                done, _ = await asyncio.wait((exit_task, changed_task, session.reader), return_when=asyncio.FIRST_COMPLETED)
                if exit_task in done or session.reader in done:
                    if self.sessions.get(server) is session:
                        self.index.remove(server)
                        self.sessions.pop(server, None)
                        self.watchers.pop(server, None)
                    await session.close()
                    return
                session.changed.clear()
                try:
                    self.index.replace(server, await session.list_tools())
                    self.refresh_errors.pop(server, None)
                except Exception as exc:
                    self.refresh_errors[server] = str(exc)
                changed_task = asyncio.create_task(session.changed.wait())
        finally:
            for task in (exit_task, changed_task):
                task.cancel()
            await asyncio.gather(exit_task, changed_task, return_exceptions=True)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.index.search(query, limit)

    async def call(self, server: str, name: str, arguments: dict[str, Any], *, timeout: float = 60) -> dict[str, Any]:
        if server == 'sandbox':
            if self.sandbox is None or name != 'execute_javascript' or not isinstance(arguments, dict) or set(arguments) != {'code'}:
                raise McpError('JavaScript sandbox is unavailable or arguments are invalid')
            result = await self.sandbox.run(arguments['code'])
            return {'content': [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False)}],
                    'isError': result['exit_code'] != 0}
        session = self.sessions.get(server)
        if session is None or not self.index.db.execute('SELECT 1 FROM tools WHERE server=? AND name=?', (server, name)).fetchone():
            raise McpError('Tool is not available on the connected server')
        return await session.call(name, arguments, timeout=timeout)

    async def enable_sandbox(self) -> None:
        from .deno_sandbox import DenoSandbox
        sandbox = await DenoSandbox.create()
        self.index.replace('sandbox', [{'name': 'execute_javascript',
            'description': 'Run execute JavaScript code to calculate math, compute values, analyze or transform data in an isolated sandbox.',
            'inputSchema': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code'], 'additionalProperties': False}}])
        self.sandbox = sandbox

    def disable_sandbox(self) -> None:
        self.index.remove('sandbox')
        self.sandbox = None

    async def disconnect(self, server: str) -> None:
        async with self.lock:
            await self._disconnect(server)

    async def _disconnect(self, server: str) -> None:
        session = self.sessions.pop(server, None)
        watcher = self.watchers.pop(server, None)
        self.index.remove(server)
        self.refresh_errors.pop(server, None)
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if session:
            await session.close()

    async def close(self) -> None:
        for server in list(self.sessions):
            await self.disconnect(server)
        self.disable_sandbox()
        self.index.db.close()


class McpBroker:
    """Bridge threaded HTTP handlers to one asyncio-owned MCP registry."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True, name='sytra-mcp')
        self.thread.start()
        self.pending: dict[str, tuple[concurrent.futures.Future, str]] = {}
        self.cancelled: dict[str, float] = {}
        self.lock = threading.Lock()
        async def create():
            return McpRegistry()
        self.registry = asyncio.run_coroutine_threadsafe(create(), self.loop).result(timeout=5)

    def execute(self, operation: str, payload: dict[str, Any]) -> Any:
        async def run():
            if operation == 'connect':
                await self.registry.connect(
                    payload.get('server', ''),
                    payload.get('argv', []),
                    cwd=payload.get('cwd'),
                    env=payload.get('env'),
                )
                return {'connected': True}
            if operation == 'disconnect':
                await self.registry.disconnect(payload.get('server', ''))
                return {'connected': False}
            if operation == 'status':
                return {'servers': sorted(self.registry.sessions), 'errors': self.registry.refresh_errors.copy(), 'sandbox': self.registry.sandbox is not None}
            if operation == 'enable-sandbox':
                await self.registry.enable_sandbox()
                return {'enabled': True}
            if operation == 'disable-sandbox':
                self.registry.disable_sandbox()
                return {'enabled': False}
            if operation == 'search':
                query = payload.get('query')
                if not isinstance(query, str) or len(query) > 8192:
                    raise ValueError('Search query must be a string of at most 8192 characters')
                return {'tools': self.registry.search(query)}
            if operation == 'web-search':
                query = payload.get('query')
                if not isinstance(query, str) or not query.strip() or len(query) > 512:
                    raise ValueError('Web search query must be 1 to 512 characters')
                from .web_search import search_web
                return await asyncio.to_thread(search_web, query)
            if operation == 'call':
                server = payload.get('server', '')
                try:
                    return await self.registry.call(server, payload.get('name', ''), payload.get('arguments', {}))
                except asyncio.TimeoutError:
                    await self.registry.disconnect(server)
                    raise McpError('Tool timed out; the MCP server was disconnected')
            raise ValueError('Unknown MCP operation')

        request_id = payload.get('request_id') if operation == 'call' else None
        if request_id is not None and not _valid_request_id(request_id):
            raise ValueError('Invalid request identifier')
        with self.lock:
            self._prune_cancelled()
            if request_id and request_id in self.cancelled:
                raise McpError('Request was cancelled before dispatch')
            if request_id and request_id in self.pending:
                raise ValueError('Request identifier is already in use')
            future = asyncio.run_coroutine_threadsafe(run(), self.loop)
            if request_id:
                self.pending[request_id] = (future, payload.get('server', ''))
        try:
            return future.result(timeout=75)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise McpError('MCP operation timed out')
        finally:
            if request_id:
                with self.lock:
                    self.pending.pop(request_id, None)

    def _prune_cancelled(self) -> None:
        now = time.monotonic()
        for request_id, deadline in list(self.cancelled.items()):
            if deadline <= now:
                del self.cancelled[request_id]

    def cancel(self, request_id: str) -> bool:
        if not _valid_request_id(request_id):
            raise ValueError('Invalid request identifier')
        with self.lock:
            self._prune_cancelled()
            item = self.pending.get(request_id)
            # Active work must remain cancellable even when the bounded
            # allowance for unknown, early identifiers is exhausted.
            if item is None and request_id not in self.cancelled and len(self.cancelled) >= _MAX_CANCELLATIONS:
                raise McpError('Too many outstanding cancellation identifiers')
            # Keep the marker even when the call has not reached the broker.
            # Reusing an identifier is blocked for the TTL as well.
            self.cancelled[request_id] = time.monotonic() + _CANCELLATION_TTL
        if not item:
            return True
        future, server = item
        future.cancel()
        # An MCP server may ignore cancellation. Close the session to prevent
        # subsequent work continuing in that server process.
        if server != 'sandbox':
            asyncio.run_coroutine_threadsafe(self.registry.disconnect(server), self.loop).result(timeout=10)
        return True

    def close(self) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self.registry.close(), self.loop).result(timeout=15)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            if not self.thread.is_alive():
                self.loop.close()
