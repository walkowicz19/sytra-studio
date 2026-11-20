import assert from 'node:assert/strict'
import test from 'node:test'
import { readChatStream } from '../src/features/chat/stream.ts'
import { buildSkillPrompt } from '../src/features/chat/skills.ts'
import { parseMcpConfigJson } from '../src/features/chat/mcpConfig.ts'
import { parseToolArguments } from '../src/features/chat/toolArguments.ts'
import { chatTemperature } from '../src/features/chat/sampling.ts'

function chunks(text) {
  const bytes = new TextEncoder().encode(text)
  let index = 0
  return new ReadableStream({ pull(controller) {
    if (index === bytes.length) controller.close()
    else controller.enqueue(bytes.slice(index, ++index))
  } })
}

test('uses the managed model temperature and falls back for external servers', async () => {
  const signal = new AbortController().signal
  assert.equal(await chatTemperature(signal, async () => Response.json({ temperature: 0 })), 0)
  assert.equal(await chatTemperature(signal, async () => Response.json({ temperature: 0.35 })), 0.35)
  assert.equal(await chatTemperature(signal, async () => new Response('', { status: 404 })), undefined)
  assert.equal(await chatTemperature(signal, async () => Response.json({ temperature: 3 })), undefined)
  const controller = new AbortController()
  controller.abort()
  await assert.rejects(chatTemperature(controller.signal, async () => { throw new DOMException('Aborted', 'AbortError') }), /Aborted/)
})

test('decodes split SSE frames and multibyte text', async () => {
  let received = ''
  await readChatStream(chunks('data: {"choices":[{"delta":{"content":"Olá"}}]}\r\n\r\ndata: [DONE]\r\n\r\n'), text => { received += text })
  assert.equal(received, 'Olá')
})

test('rejects incomplete streams and incomplete tool calls', async () => {
  await assert.rejects(readChatStream(chunks('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'), () => {}), /before completion/)
  await assert.rejects(readChatStream(chunks('data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n'), () => {}), /incomplete tool call/)
})

test('assembles fragmented tool calls and repairs only bounded JSON defects', async () => {
  const stream = 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":{"name":"mcp_","arguments":"{\\"n\\":"}}]}}]}\n\n' +
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"0","arguments":"2,}"}}]},"finish_reason":"tool_calls"}]}\n\n'
  const calls = await readChatStream(chunks(stream), () => {})
  assert.equal(calls[0].function.name, 'mcp_0')
  assert.deepEqual(parseToolArguments(calls[0].function.arguments), { value: { n: 2 }, repaired: true })
  assert.throws(() => parseToolArguments('{"x":"unfinished'), /safely/)
})

test('skill injection preserves the selected instructions and rejects unsafe identifiers', () => {
  const contents = 'Use `<T>` and A & B.\n'
  assert.equal(buildSkillPrompt(['workspace:code-review'], [contents]),
    `<skill name="workspace:code-review">\n${contents}\n</skill>`)
  assert.throws(() => buildSkillPrompt(['workspace:../other'], ['secret']), /identifier/)
  assert.throws(() => buildSkillPrompt(['user:x'], []), /selection/)
})

test('parses mcp_config.json with mcpServers, env, and rejects empty or URL-only configs', () => {
  const parsed = parseMcpConfigJson(JSON.stringify({
    mcpServers: {
      'my.tools': { command: 'python', args: ['-m', 'srv'], env: { API_KEY: 'secret' }, cwd: 'D:\\ws' },
    },
  }))
  assert.deepEqual(parsed, [{
    server: 'my-tools',
    argv: ['python', '-m', 'srv'],
    env: { API_KEY: 'secret' },
    cwd: 'D:\\ws',
  }])
  assert.throws(() => parseMcpConfigJson('{}'), /Expected/)
  assert.throws(() => parseMcpConfigJson('{"mcpServers":{"bad":{"url":"http://127.0.0.1"}}}'), /missing a string "command"/)
})


test('readChatStream extracts think tags, Harmony analysis, reasoning_content, and metrics', async () => {
  const { readChatStream } = await import('../src/features/chat/stream.ts')
  const makeSse = (events) => {
    const encoder = new TextEncoder()
    return new ReadableStream({
      start(controller) {
        for (const ev of events) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(ev)}\n\n`))
        }
        controller.enqueue(encoder.encode('data: [DONE]\n\n'))
        controller.close()
      },
    })
  }

  let visible = ''
  let reasoning = ''
  let latestMetrics = {}
  await readChatStream(
    makeSse([
      { choices: [{ delta: { reasoning_content: 'First check ' } }] },
      { choices: [{ delta: { content: '<think>step 2' } }] },
      { choices: [{ delta: { content: '</think>Answer ready.' } }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }], timings: { predicted_per_second: 54.32, predicted_n: 42, prompt_n: 120 } },
    ]),
    {
      onText: (chunk) => { visible += chunk },
      onReasoning: (chunk) => { reasoning += chunk },
      onMetrics: (m) => { latestMetrics = m },
    },
  )
  assert.equal(visible, 'Answer ready.')
  assert.equal(reasoning, 'First check step 2')
  assert.equal(Math.round(latestMetrics.tkPerSec * 10) / 10, 54.3)
  assert.equal(latestMetrics.completionTokens, 42)
  assert.equal(latestMetrics.promptTokens, 120)

  let harmOut = ''
  let harmReason = ''
  await readChatStream(
    makeSse([
      { choices: [{ delta: { content: '<|channel|>analysis<|message|>Plan A<|end|><|start|>assistant<|channel|>final<|message|>Done!' } }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]),
    {
      onText: (c) => { harmOut += c },
      onReasoning: (c) => { harmReason += c },
    },
  )
  assert.equal(harmOut, 'Done!')
  assert.equal(harmReason, 'Plan A')
})

test('promptBuilder keeps prompt lightweight and extracts inline tool_call fallback', async () => {
  const { buildLightweightSystemPrompt, extractInlineToolCalls } = await import('../src/features/chat/promptBuilder.ts')
  const tools = [
    {
      exposedName: 'web_search',
      item: {
        server: 'web',
        tool: {
          name: 'web_search',
          description: 'Search live web results.',
          inputSchema: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
        },
      },
    },
  ]
  const promptText = buildLightweightSystemPrompt({
    tools,
    skillsPrompt: `<skill name="reviewer">${'x'.repeat(5000)}</skill>`,
  })
  assert.match(promptText, /web_search\(query:string\): Search live web results\./)
  assert.match(promptText, /<tool_call>/)
  assert.match(promptText, /\[skill truncated for context budget\]/)

  const parsed = extractInlineToolCalls(
    'Searching now.\n<tool_call>{"name":"web_search","arguments":{"query":"sytra studio"}}</tool_call>',
    tools,
  )
  assert.equal(parsed.calls.length, 1)
  assert.equal(parsed.calls[0].function.name, 'web_search')
  assert.deepEqual(JSON.parse(parsed.calls[0].function.arguments), { query: 'sytra studio' })
  assert.equal(parsed.cleanText, 'Searching now.')
})

test('chatServerPreferences reads managed metadata and falls back to /props on external servers', async () => {
  const { chatServerPreferences } = await import('../src/features/chat/sampling.ts')
  const managed = await chatServerPreferences(new AbortController().signal, async (url) => {
    assert.match(String(url), /\/sytra\/preferences$/)
    return new Response(JSON.stringify({ temperature: 0.2, context_window: 16384, model: 'qwen-test' }), { status: 200 })
  })
  assert.deepEqual(managed, { temperature: 0.2, contextWindow: 16384, model: 'qwen-test' })

  const external = await chatServerPreferences(new AbortController().signal, async (url) => {
    if (String(url).endsWith('/sytra/preferences')) return new Response('missing', { status: 404 })
    if (String(url).endsWith('/props')) {
      return new Response(JSON.stringify({ default_generation_settings: { n_ctx: 4096, params: { temp: 0.65 } } }), { status: 200 })
    }
    return new Response(JSON.stringify({ data: [{ id: 'ext-model' }] }), { status: 200 })
  })
  assert.equal(external.temperature, 0.65)
  assert.equal(external.contextWindow, 4096)
  assert.equal(external.model, 'ext-model')
})

test('parseRichBlocks parses chartable tables, charts, kpis, unclosed fences, and safe links', async () => {
  const { parseRichBlocks } = await import('../src/features/chat/markdown.ts')
  const blocks = parseRichBlocks([
    '# Summary',
    '',
    '| Quarter | Revenue | Profit |',
    '| --- | ---: | ---: |',
    '| Q1 | 120 | 30 |',
    '| Q2 | 165 | 48 |',
    '',
    '```kpi',
    '{"title":"Highlights","items":[{"label":"ARR","value":"$2.4M","delta":"+18%"}]}',
    '```',
    '',
    '```chart',
    '{"type":"line","title":"Growth","labels":["Jan","Feb"],"values":[10,24],"unit":"%"}',
    '```',
    '',
    '[Bad](javascript:alert(1)) and [Good](https://example.com)',
    '',
    '```python',
    'print("still streaming"',
  ].join('\n'))

  assert.equal(blocks[0].type, 'heading')
  assert.equal(blocks[1].type, 'table')
  assert.equal(blocks[1].chartable?.labels.length, 2)
  assert.equal(blocks[1].chartable?.series?.length, 2)
  assert.equal(blocks[2].type, 'kpi')
  assert.equal(blocks[2].spec.items[0].tone, 'positive')
  assert.equal(blocks[3].type, 'chart')
  assert.equal(blocks[3].spec.kind, 'line')
  assert.equal(blocks[4].type, 'paragraph')
  assert.equal(blocks[4].spans.filter((s) => s.type === 'link').length, 1)
  assert.equal(blocks[5].type, 'code')
  assert.equal(blocks[5].streaming, true)

  // Models frequently output ```json or Chart.js shapes when asked for a chart; auto-promote to interactive chart.
  const jsonCharts = parseRichBlocks([
    '```json',
    '{',
    '  "type": "line",',
    '  "title": "NVIDIA Asset Values",',
    '  "labels": ["2023-01-01", "2023-02-01", "2023-03-01"],',
    '  "values": [100, 105, 115],',
    '  "unit": ""',
    '}',
    '```',
    '',
    '```json',
    '{"name":"not-a-chart","ok":true}',
    '```',
  ].join('\n'))
  assert.equal(jsonCharts[0].type, 'chart')
  assert.equal(jsonCharts[0].spec.kind, 'line')
  assert.deepEqual(jsonCharts[0].spec.values, [100, 105, 115])
  assert.equal(jsonCharts[1].type, 'code')
  assert.equal(jsonCharts[1].lang, 'json')
})


test('buildCompactionDigest and buildLightweightSystemPrompt include agent persona and recalled memories', async () => {
  const { buildCompactionDigest, buildLightweightSystemPrompt } = await import('../src/features/chat/promptBuilder.ts')
  const sys = buildLightweightSystemPrompt({
    agentPrompt: 'You are Pochi, a concise research companion.',
    computerMode: 'sandbox',
    skillsCatalog: ['code-review', 'bug-finder'],
    mcpServers: ['klayer'],
    mcpPermission: 'ask',
    recalledMemories: [
      { kind: 'compaction', session_title: 'Prior Audit', summary: 'Selected Qwen2.5 7B on 12 GB VRAM.' },
    ],
    tools: [],
  })
  assert.match(sys, /Skills \(SKILL\.md/)
  assert.match(sys, /code-review, bug-finder/)
  assert.match(sys, /MCP.*1 connected \(klayer\)/)
  assert.match(sys, /Agent persona:/)
  assert.match(sys, /Isolated 24\/7 agent workspace/)
  assert.match(sys, /Recalled session memory:/)
  assert.match(sys, /Prior Audit/)

  const digest = buildCompactionDigest([
    { role: 'user', content: 'Benchmark Q1 throughput' },
    { role: 'assistant', content: 'Q1 measured 48.2 tk/s on CUDA.' },
  ], 'Initial context note')
  assert.match(digest, /Prior summary: Initial context note/)
  assert.match(digest, /User: Benchmark Q1 throughput/)
  assert.match(digest, /Assistant: Q1 measured 48\.2 tk\/s on CUDA\./)
})


test('promptBuilder enforces matching user language and modelRouter routes deterministically by task and agent objective', async () => {
  const { buildLightweightSystemPrompt } = await import('../src/features/chat/promptBuilder.ts')
  const { inferLanguageScope, routeBestModel } = await import('../src/features/chat/modelRouter.ts')

  const sys = buildLightweightSystemPrompt({ tools: [] })
  assert.match(sys, /same language as the user/i)

  assert.equal(inferLanguageScope('qwen2.5:7b').scope, 'multilingual')
  assert.equal(inferLanguageScope('tinyllama-1.1b').scope, 'english')
  assert.equal(inferLanguageScope('unknown-custom-v1').scope, 'unknown')

  const targets = {
    providers: {
      sytra_running: true,
      sytra_models: ['tinyllama-1.1b'],
      ollama_running: true,
      ollama_installed: true,
      ollama_models: [
        { name: 'qwen2.5-coder:7b', size_gb: 4.7, parameter_size: '7B', quantization: 'Q4', language_scope: 'multilingual' },
        { name: 'llava:7b', size_gb: 4.9, parameter_size: '7B', quantization: 'Q4', language_scope: 'multilingual' },
        { name: 'lfm3-1.7b:latest', size_gb: 1.0, parameter_size: '1.7B', quantization: 'Q4', language_scope: 'multilingual' },
      ],
      lmstudio_running: false,
      lmstudio_models: [],
    },
    local_models: [],
  }

  const codingRoute = routeBestModel({
    prompt: 'Fix this Rust function bug and SQL query',
    agentRole: 'Code Maintainer',
    targets,
    vramMb: 12288,
  })
  assert.equal(codingRoute?.endpoint.model, 'qwen2.5-coder:7b')
  assert.match(codingRoute?.reason ?? '', /coding/i)

  const visionRoute = routeBestModel({
    prompt: 'Analyze screenshot',
    hasImages: true,
    targets,
    vramMb: 12288,
  })
  assert.equal(visionRoute?.endpoint.model, 'llava:7b')

  const ptQuickRoute = routeBestModel({
    prompt: 'Olá, tudo bem?',
    targets,
    vramMb: 12288,
  })
  assert.equal(ptQuickRoute?.endpoint.model, 'lfm3-1.7b:latest')
  assert.equal(ptQuickRoute?.language.scope, 'multilingual')
})

test('readChatStream distinguishes first-token load grace from steady idle timeout', async () => {
  const encoder = new TextEncoder()
  const delayedFirstChunk = new ReadableStream({
    start(controller) {
      setTimeout(() => {
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"Ready"}}]}\n\n'))
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'))
        controller.close()
      }, 45)
    },
  })
  let out = ''
  await readChatStream(delayedFirstChunk, {
    firstTokenTimeoutMs: 400,
    idleTimeoutMs: 20,
    onText: (c) => { out += c },
  })
  assert.equal(out, 'Ready')
})

test('checkpoint system records lightweight snapshots and rolls back turns and workspace deltas', async () => {
  const { pushCheckpoint, recordWorkspaceDelta, rollbackLastCheckpoint, lastCheckpointTargetIndex } = await import('../src/features/chat/checkpoint.ts')
  const session = {
    messages: [],
    lastCompactedSummary: 'Before turn 1',
    compactionCount: 1,
  }
  pushCheckpoint(session, 'Write file')
  session.messages.push({ role: 'user', content: 'Write file' })
  recordWorkspaceDelta(session, { agentId: 'lil-sytra', path: 'notes.md', existed: true, previousContent: 'old' })
  recordWorkspaceDelta(session, { agentId: 'lil-sytra', path: 'notes.md', existed: true, previousContent: 'mid' })
  session.messages.push({ role: 'assistant', content: 'Updated notes.md' })

  assert.equal(lastCheckpointTargetIndex(session), 0)
  const undone = rollbackLastCheckpoint(session)
  assert.equal(undone?.promptText, 'Write file')
  assert.equal(undone?.workspaceDeltas.length, 1)
  assert.equal(undone?.workspaceDeltas[0].previousContent, 'old')
  assert.equal(session.messages.length, 0)
  assert.equal(session.lastCompactedSummary, 'Before turn 1')
})
