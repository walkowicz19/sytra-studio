import asyncio
import concurrent.futures
import http.client
import json
import sys
import threading

import pytest

from sytra_runner.mcp_chat import McpBroker, McpError, McpRegistry, ToolIndex
from sytra_runner.idle_proxy import IdleState, make_proxy


def test_fts_ranks_names_and_descriptions_and_replaces_atomically():
    index = ToolIndex()
    tools = [
        {'name': 'find_file', 'description': 'Find a file by name', 'inputSchema': {'type': 'object'}},
        {'name': 'weather', 'description': 'Weather forecast by city', 'inputSchema': {'type': 'object'}},
    ]
    index.replace('alpha', tools)
    assert index.search('find file')[0]['tool']['name'] == 'find_file'
    assert index.search('weather OR "*') [0]['tool']['name'] == 'weather'
    with pytest.raises(McpError):
        index.replace('alpha', [tools[0], tools[0]])
    assert index.search('weather')[0]['server'] == 'alpha'
    index.remove('alpha')
    assert index.search('weather') == []


def test_real_stdio_server_pagination_refresh_and_cancellation(tmp_path):
    server = tmp_path / 'server.py'
    marker = tmp_path / 'cancelled.json'
    server.write_text('''import json, sys, threading, time
marker = sys.argv[1]
changed = False
def send(value):
 print(json.dumps(value), flush=True)
def reply(id, result):
 send({'jsonrpc':'2.0','id':id,'result':result})
for line in sys.stdin:
 msg = json.loads(line); method = msg.get('method'); id = msg.get('id'); params = msg.get('params', {})
 if method == 'initialize': reply(id, {'protocolVersion':'2025-06-18','capabilities':{'tools':{}}})
 elif method == 'tools/list':
  if params.get('cursor') == 'next': reply(id, {'tools':[{'name':'weather','description':'Weather forecast','inputSchema':{'type':'object'}}]})
  elif changed: reply(id, {'tools':[{'name':'updated','description':'Updated search','inputSchema':{'type':'object'}}]})
  else: reply(id, {'tools':[{'name':'refresh','description':'Refresh tools','inputSchema':{'type':'object'}}, {'name':'break','description':'Break transport','inputSchema':{'type':'object'}}], 'nextCursor':'next'})
 elif method == 'tools/call' and params['name'] == 'refresh':
  changed = True; reply(id, {'content':[{'type':'text','text':'done'}]}); send({'jsonrpc':'2.0','method':'notifications/tools/list_changed'})
 elif method == 'tools/call' and params['name'] == 'break': print('invalid-json', flush=True)
 elif method == 'tools/call':
  threading.Thread(target=lambda: (time.sleep(2), reply(id, {'content':[]})), daemon=True).start()
 elif method == 'notifications/cancelled':
  open(marker, 'w').write(json.dumps(params))
''', encoding='utf-8')
    async def exercise():
        registry = McpRegistry()
        try:
            await registry.connect('demo', [sys.executable, '-u', str(server), str(marker)])
            assert registry.search('weather')[0]['tool']['name'] == 'weather'
            with pytest.raises(asyncio.TimeoutError):
                await registry.call('demo', 'weather', {}, timeout=.15)
            for _ in range(30):
                if marker.exists(): break
                await asyncio.sleep(.02)
            assert json.loads(marker.read_text())['requestId'] > 0
            result = await registry.call('demo', 'refresh', {})
            assert result['content'][0]['text'] == 'done'
            for _ in range(50):
                if registry.search('updated'): break
                await asyncio.sleep(.02)
            assert registry.search('updated')[0]['tool']['name'] == 'updated'
            assert not registry.search('weather')
            await registry.disconnect('demo')
            assert registry.search('updated') == []
            await registry.connect('demo', [sys.executable, '-u', str(server), str(marker)])
            with pytest.raises(McpError, match='transport failed'):
                await registry.call('demo', 'break', {})
            for _ in range(50):
                if 'demo' not in registry.sessions: break
                await asyncio.sleep(.02)
            assert 'demo' not in registry.sessions
            assert not registry.search('weather')
        finally:
            await registry.close()
    asyncio.run(exercise())


def test_http_broker_connects_searches_and_calls_a_real_mcp_server(tmp_path):
    server = tmp_path / 'mcp.py'
    server.write_text('''import json, sys, time
for line in sys.stdin:
 m=json.loads(line); method=m.get('method'); id=m.get('id')
 if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{'tools':{}}}
 elif method=='tools/list': result={'tools':[{'name':'lookup','description':'Lookup documentation','inputSchema':{'type':'object'}}, {'name':'wait','description':'Wait slowly','inputSchema':{'type':'object'}}]}
 elif method=='tools/call' and m['params']['name']=='wait':
  open(sys.argv[1], 'w').write('started'); time.sleep(20); result={'content':[]}
 elif method=='tools/call': result={'content':[{'type':'text','text':m['params']['arguments']['query']}]}
 else: continue
 print(json.dumps({'jsonrpc':'2.0','id':id,'result':result}),flush=True)
''', encoding='utf-8')
    broker = McpBroker()
    proxy = make_proxy('127.0.0.1', 0, 1, IdleState(0), broker=broker, token='secret')
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    def post(operation, payload, expected=200):
        connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=5)
        connection.request('POST', f'/sytra/mcp/{operation}', body=json.dumps(payload),
            headers={'Origin': 'http://tauri.localhost', 'X-Sytra-Session': 'secret', 'Content-Type': 'application/json'})
        response = connection.getresponse()
        data = json.loads(response.read())
        connection.close()
        assert response.status == expected, data
        return data
    try:
        marker = tmp_path / 'started'
        post('connect', {'server': 'demo', 'argv': [sys.executable, '-u', str(server), str(marker)]})
        assert post('search', {'query': 'documentation'})['tools'][0]['tool']['name'] == 'lookup'
        assert post('cancel', {'request_id': 'early-cancel'})['cancelled']
        rejected = post('call', {'server': 'demo', 'name': 'lookup', 'arguments': {'query': 'must not run'},
                                 'request_id': 'early-cancel'}, 400)
        assert 'cancelled before dispatch' in rejected['error']
        assert post('status', {})['servers'] == ['demo']
        post('cancel', {'request_id': []}, 400)
        result = post('call', {'server': 'demo', 'name': 'lookup', 'arguments': {'query': 'hello'}, 'request_id': 'test-1'})
        assert result['content'][0]['text'] == 'hello'
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            blocked = pool.submit(post, 'call', {'server': 'demo', 'name': 'wait', 'arguments': {}, 'request_id': 'test-wait'}, 400)
            import time
            deadline = time.monotonic() + 5
            while not marker.exists() and time.monotonic() < deadline: time.sleep(.02)
            assert marker.exists()
            assert post('cancel', {'request_id': 'test-wait'})['cancelled']
            blocked.result(timeout=5)
        assert post('status', {})['servers'] == []
        assert post('search', {'query': 'documentation'})['tools'] == []
    finally:
        proxy.shutdown()
        thread.join()
        proxy.server_close()
        broker.close()


def test_preemptive_cancellation_expires_and_never_dispatches_early():
    broker = McpBroker()
    calls = []
    async def call(server, name, arguments):
        calls.append((server, name))
        return {'content': []}
    broker.registry.call = call
    payload = {'server': 'example', 'name': 'tool', 'arguments': {}, 'request_id': 'early'}
    try:
        assert broker.cancel('early')
        with pytest.raises(McpError, match='cancelled before dispatch'):
            broker.execute('call', payload)
        assert not calls
        with broker.lock:
            broker.cancelled['early'] = 0
        assert broker.execute('call', payload) == {'content': []}
        assert calls == [('example', 'tool')]
    finally:
        broker.close()


def test_preemptive_cancellation_storage_is_bounded(monkeypatch):
    monkeypatch.setattr('sytra_runner.mcp_chat._MAX_CANCELLATIONS', 2)
    broker = McpBroker()
    try:
        assert broker.cancel('first')
        assert broker.cancel('second')
        with pytest.raises(McpError, match='Too many'):
            broker.cancel('third')
        active = concurrent.futures.Future()
        with broker.lock:
            broker.pending['active'] = (active, 'sandbox')
        assert broker.cancel('active')
        assert active.cancelled()
        with broker.lock:
            broker.cancelled['first'] = 0
            broker.cancelled['active'] = 0
            broker.pending.pop('active')
        assert broker.cancel('third')
        assert len(broker.cancelled) == 2
    finally:
        broker.close()
