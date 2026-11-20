import asyncio
import shutil

import pytest

from sytra_runner.deno_sandbox import DenoSandbox, SandboxError
from sytra_runner.mcp_chat import McpRegistry


def test_sandbox_fails_closed_when_deno_is_absent(monkeypatch):
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    async def check():
        with pytest.raises(SandboxError, match='not found'):
            await DenoSandbox.create()
        with pytest.raises(SandboxError, match='self-test'):
            await DenoSandbox('untrusted').run('console.log(1)')
    asyncio.run(check())


def test_registry_only_exposes_verified_sandbox(monkeypatch):
    class FakeSandbox:
        async def run(self, code): return {'stdout': code, 'stderr': '', 'exit_code': 0}
    async def create(): return FakeSandbox()
    monkeypatch.setattr(DenoSandbox, 'create', create)
    async def check():
        registry = McpRegistry()
        try:
            assert not registry.search('javascript')
            await registry.enable_sandbox()
            assert registry.search('javascript')[0]['server'] == 'sandbox'
            result = await registry.call('sandbox', 'execute_javascript', {'code': 'console.log(1)'})
            assert 'console.log(1)' in result['content'][0]['text']
            registry.disable_sandbox()
            assert not registry.search('javascript')
        finally:
            await registry.close()
    asyncio.run(check())


@pytest.mark.skipif(not (shutil.which('deno') or shutil.which('deno.exe')), reason='Deno is not installed')
def test_real_deno_denies_host_access_and_runs_basic_javascript():
    async def check():
        sandbox = await DenoSandbox.create()
        result = await sandbox.run('console.log(1 + 1)')
        assert result['exit_code'] == 0 and result['stdout'].strip() == '2'
        result = await sandbox.run("await Deno.readTextFile('secret.txt')")
        assert result['exit_code'] != 0
    asyncio.run(check())
