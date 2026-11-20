"""Opt-in JavaScript execution with a locally verified Deno permission boundary."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote

from .subprocess_utils import silent_subprocess_kwargs

MAX_CODE = 64 * 1024
MAX_OUTPUT = 64 * 1024
_FLAGS = ('run', '--no-prompt', '--no-config', '--no-lock', '--no-npm', '--no-remote',
          '--cached-only', '--deny-read', '--deny-write', '--deny-net', '--deny-env',
          '--deny-run', '--deny-sys', '--deny-ffi', '--deny-import',
          '--v8-flags=--max-old-space-size=128', '-')


class SandboxError(RuntimeError):
    pass


class DenoSandbox:
    def __init__(self, executable: str):
        self.executable = executable
        self.verified = False
        self.fingerprint = None

    def _fingerprint(self):
        stat = Path(self.executable).stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @classmethod
    async def create(cls) -> 'DenoSandbox':
        executable = shutil.which('deno') or shutil.which('deno.exe')
        if not executable:
            raise SandboxError('Deno was not found; JavaScript sandbox is unavailable')
        sandbox = cls(str(Path(executable).resolve()))
        try:
            before = sandbox._fingerprint()
        except OSError as exc:
            raise SandboxError(f'Cannot inspect Deno: {exc}') from exc
        await sandbox.verify()
        if before != sandbox._fingerprint():
            raise SandboxError('Deno changed during its isolation self-test')
        sandbox.fingerprint = before
        sandbox.verified = True
        return sandbox

    async def verify(self) -> None:
        with tempfile.TemporaryDirectory(prefix='sytra-deno-probe-') as directory:
            root = Path(directory)
            secret = root / 'secret.mjs'
            secret.write_text('export const secret = "private";', encoding='utf-8')
            target = (root / 'must-not-exist').as_posix()
            uri = secret.as_uri()
            nested = 'data:text/javascript,' + quote(
                f'import {{ secret }} from {json.dumps(uri)}; export default secret;')
            code = f'''const denied = async (label, op) => {{
              try {{ await op(); throw new Error("allowed:" + label); }}
              catch (e) {{ if (String(e.message).startsWith("allowed:")) throw e;
                if (!['NotCapable', 'PermissionDenied'].includes(e.name)) throw new Error("unexpected:" + label + ":" + e.name); }}
            }};
            await denied('read', () => Deno.readTextFile({json.dumps(str(secret))}));
            await denied('write', () => Deno.writeTextFile({json.dumps(target)}, 'x'));
            await denied('env', () => Deno.env.get('PATH'));
            await denied('run', () => new Deno.Command('not-a-real-command').output());
            await denied('net', () => fetch('http://127.0.0.1:9/'));
            await denied('dynamic-import', () => import({json.dumps(uri)}));
            await denied('nested-import', () => import({json.dumps(nested)}));
            for (const name of ['read','write','net','env','run','sys','ffi','import']) {{
              const result = await Deno.permissions.query({{ name }});
              if (result.state !== 'denied') throw new Error('permission:' + name + ':' + result.state);
            }}
            console.log('SYTRA_SANDBOX_OK');'''
            result = await self._execute(code, timeout=15)
            if result['exit_code'] != 0 or result['stdout'].strip() != 'SYTRA_SANDBOX_OK' or (root / 'must-not-exist').exists():
                raise SandboxError('Deno isolation self-test failed; sandbox remains disabled')

    async def run(self, code: str) -> dict:
        if not self.verified:
            raise SandboxError('Sandbox must pass its isolation self-test before use')
        try:
            if self._fingerprint() != self.fingerprint:
                raise SandboxError('Deno changed after its isolation self-test; enable the sandbox again')
        except OSError as exc:
            raise SandboxError(f'Cannot inspect Deno: {exc}') from exc
        if not isinstance(code, str) or not code or len(code.encode('utf-8')) > MAX_CODE:
            raise SandboxError('JavaScript code must be non-empty and at most 64 KiB')
        return await self._execute(code, timeout=10)

    async def _execute(self, code: str, *, timeout: int) -> dict:
        # The untrusted program is data inside a fixed entrypoint, not part of
        # Deno's initial static import graph (which bypasses read permissions).
        source = ('const AsyncFunction = (async()=>{}).constructor;\n'
                  f'await new AsyncFunction({json.dumps(code, ensure_ascii=True)})();\n').encode('utf-8')
        with tempfile.TemporaryDirectory(prefix='sytra-deno-') as directory:
            env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP') if key in os.environ}
            env['DENO_DIR'] = str(Path(directory) / 'cache')
            env['NO_COLOR'] = '1'
            try:
                process = await asyncio.create_subprocess_exec(self.executable, *_FLAGS, cwd=directory, env=env,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    **silent_subprocess_kwargs())
            except OSError as exc:
                raise SandboxError(f'Cannot start Deno: {exc}') from exc

            async def collect(stream):
                data = bytearray()
                while chunk := await stream.read(8192):
                    data.extend(chunk)
                    if len(data) > MAX_OUTPUT:
                        raise SandboxError('Sandbox output exceeds 64 KiB')
                return bytes(data).decode('utf-8', errors='replace')

            async def exchange():
                process.stdin.write(source)
                await process.stdin.drain()
                process.stdin.close()
                stdout, stderr = await asyncio.gather(collect(process.stdout), collect(process.stderr))
                await process.wait()
                return {'exit_code': process.returncode, 'stdout': stdout, 'stderr': stderr}

            try:
                return await asyncio.wait_for(exchange(), timeout)
            except asyncio.TimeoutError as exc:
                raise SandboxError('Sandbox exceeded its time limit') from exc
            finally:
                if process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
