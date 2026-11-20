import { expect, test } from '@playwright/test'

// Fonts are optional; keep UI checks offline without hiding app errors.
test.beforeEach(async ({ page }) => {
  await page.route('https://fonts.googleapis.com/**', route => route.fulfill({ contentType: 'text/css', body: '' }))
})

test('renders v1 UI and shared resource controls without browser errors', async ({ page }) => {
  const errors: string[] = []
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text())
  })
  page.on('pageerror', error => errors.push(error.message))

  await page.goto('/')

  await expect(page.getByText('STUDIO', { exact: true })).toBeVisible()
  await expect(page.getByText('STUDIO — 01', { exact: true })).toHaveCount(0)
  await expect(page.getByText('v1.3.0', { exact: true })).toBeVisible()
  await expect(page.locator('.pagination-bar')).toContainText('Showing 1–3 of 3 models')
  await expect(page.getByLabel('Models per page')).toBeVisible()
  await expect(page.locator('.pagination-bar button')).toHaveCount(0)
  await expect(page.getByText('org/knowledge-ft')).toHaveCount(0)
  const qwen9bCard = page.locator('.model-card', { hasText: 'Qwen3.5 9B Base' })
  await expect(qwen9bCard).toContainText('Fits GPU')
  const kimiCard = page.locator('.model-card', { hasText: 'Kimi K2.7 Coder' })
  await expect(kimiCard).toContainText('Too large for this PC')

  // Guided & simple mode on Models: simple UI + MoE capacity in B + server runtime card
  await expect(page.locator('#moe-capacity-card')).toContainText('Up to ~112B')
  await expect(page.locator('#moe-capacity-card')).toContainText('Up to 500B–700B+ MoE')
  await expect(page.getByText('Native MoE index')).toHaveCount(0)
  await expect(page.getByText('Download from catalog ID')).toHaveCount(0)
  await page.locator('#serve-model-path').fill('D:\\models\\qwen-moe')
  await page.locator('#start-model-server').click()
  await expect(page.locator('#active-runtime-summary')).toBeVisible()
  await expect(page.locator('#active-runtime-summary')).toContainText('MoE')

  await page.getByRole('button', { name: /Settings/ }).click()
  await expect(page.getByText('Choose where models are downloaded')).toBeVisible()
  await expect(page.locator('#btn-pick-cache-dir')).toBeVisible()
  await expect(page.locator('#select-memory-limit option', { hasText: '75%' })).toHaveCount(1)

  const memory = page.getByLabel('Main memory')
  await expect(memory).toBeVisible()
  await expect(memory).toHaveValue('auto')
  const seventyFiveValue = await memory.locator('option').filter({ hasText: '75%' }).getAttribute('value')
  expect(seventyFiveValue).not.toBeNull()
  await memory.selectOption(seventyFiveValue!)
  await expect(page.getByText('Main memory limit saved')).toBeVisible()

  await expect(page.getByText('MCP Integration', { exact: true })).toBeVisible()
  await expect(page.getByText('Direct native MCP binary')).toBeVisible()
  await page.locator('#btn-install-mcp-auto').click()
  await expect(page.getByText(/MCP config updated/)).toBeVisible()
  const mcpBadgeBox = await page.locator('.mcp-status-badge').first().boundingBox()
  const mcpRemoveBox = await page.locator('.mcp-action-btn', { hasText: 'Remove' }).first().boundingBox()
  const mcpConfigureBox = await page.locator('.mcp-action-btn', { hasText: 'Configure' }).first().boundingBox()
  expect(mcpBadgeBox).not.toBeNull()
  expect(mcpRemoveBox).not.toBeNull()
  expect(mcpConfigureBox).not.toBeNull()
  expect(Math.abs(mcpBadgeBox!.height - mcpRemoveBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(mcpRemoveBox!.width - mcpConfigureBox!.width)).toBeLessThanOrEqual(1)
  expect(Math.abs(mcpRemoveBox!.x - mcpConfigureBox!.x)).toBeLessThanOrEqual(1)

  // Guided & simple mode on Combine
  await page.locator('#tab-merge').click()
  await expect(page.getByRole('heading', { name: 'Combine models' })).toBeVisible()
  await expect(page.locator('.step-title', { hasText: 'Performance' })).toBeVisible()
  const safePerfBtn = page.getByRole('button', { name: /Safe — low memory/ })
  await safePerfBtn.click()
  await expect(safePerfBtn).toHaveAttribute('aria-pressed', 'true')

  // Guided & simple mode on Datasets
  await page.getByRole('button', { name: /Datasets/ }).click()
  await expect(page.getByRole('heading', { name: 'Datasets' })).toBeVisible()
  await expect(page.locator('.step-title', { hasText: 'What would you like to do?' })).toBeVisible()
  await expect(page.getByText('Rows per shard')).toHaveCount(0)
  await expect(page.getByText('Use weighted sample counts')).toHaveCount(0)
  const checkCardBox = await page.locator('.guided-check-card').first().boundingBox()
  const checkInputBox = await page.locator('.guided-check-card input[type="checkbox"]').first().boundingBox()
  expect(checkCardBox).not.toBeNull()
  expect(checkInputBox).not.toBeNull()
  const cardMidY = checkCardBox!.y + checkCardBox!.height / 2
  const inputMidY = checkInputBox!.y + checkInputBox!.height / 2
  expect(Math.abs(cardMidY - inputMidY)).toBeLessThanOrEqual(2)
  await page.getByRole('button', { name: /Split a file for testing/ }).click()
  await expect(page.locator('.step-title', { hasText: 'How much to save for testing?' })).toBeVisible()
  await expect(page.getByRole('button', { name: /10% — Balanced/ })).toBeVisible()

  // Advanced mode on Datasets + Merge
  await page.getByRole('button', { name: /Switch to Advanced mode/ }).click()
  await page.getByRole('button', { name: 'Combine', exact: true }).click()
  await expect(page.getByLabel('Rows per shard')).toBeVisible()
  const addDatasetBox = await page.getByRole('button', { name: 'Add dataset', exact: true }).boundingBox()
  const weightedCheckBox = await page.getByText('Use weighted sample counts').boundingBox()
  expect(addDatasetBox).not.toBeNull()
  expect(weightedCheckBox).not.toBeNull()
  expect(weightedCheckBox!.x - (addDatasetBox!.x + addDatasetBox!.width)).toBeGreaterThanOrEqual(8)
  expect(weightedCheckBox!.x - (addDatasetBox!.x + addDatasetBox!.width)).toBeLessThanOrEqual(40)
  await page.getByRole('button', { name: 'Split', exact: true }).click()
  await expect(page.getByLabel('Input JSONL or Parquet')).toBeVisible()

  await page.locator('#tab-merge').click()
  await expect(page.locator('#section-merge-performance')).toBeVisible()
  await page.locator('#tab-models').click()
  await expect(page.getByText('Native MoE index')).toBeVisible()
  await expect(page.getByText('Download from catalog ID')).toBeVisible()
  await expect(page.locator('#moe-capacity-card')).toBeVisible()
  await page.getByRole('button', { name: /Runs/ }).click()
  await expect(page.getByRole('heading', { name: /history/i })).toBeVisible()

  expect(errors).toEqual([])
})

test('Datasets and Chat follow the saved interface language and appearance', async ({ page }) => {
  await page.goto('/')
  const settings = page.locator('#tab-settings')
  const datasets = page.locator('#tab-datasets')
  const chat = page.locator('#tab-chat')

  await settings.click()
  await page.locator('.hex-input').first().fill('#007e75')
  await page.locator('#vs-tab-layout').click()
  await page.locator('.radius-grid').getByRole('button', { name: 'Rounded' }).click()
  await page.locator('.shadow-grid').getByRole('button', { name: 'None' }).click()
  await page.locator('.anim-grid').getByRole('button', { name: 'Off' }).click()
  await page.locator('#font-jetbrains').click()
  await page.locator('#select-language').selectOption('pt')

  await expect(page.locator('html')).toHaveAttribute('lang', 'pt')
  await expect(datasets).toContainText('Conjuntos de dados')
  await datasets.click()
  await expect(page.getByRole('heading', { name: 'Conjuntos de dados' })).toBeVisible()
  await expect(page.getByLabel('Arquivo de exemplos 1')).toBeVisible()
  await expect(page.locator('.choice.card').first()).toHaveCSS('border-radius', '12px')
  await expect(page.locator('.guided-check-card').first()).toHaveCSS('box-shadow', 'none')
  await expect(page.locator('.choice.card').first()).toHaveCSS('font-family', /JetBrains Mono/)
  await expect(page.locator('.pane')).toHaveCSS('animation-duration', '0s')
  await page.getByRole('button', { name: /mude para o modo avançado/ }).click()
  await expect(page.getByLabel('Linhas por fragmento')).toBeVisible()

  await chat.click()
  await expect(page.getByRole('heading', { name: 'Chat local' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Nova conversa', exact: true })).toBeVisible()
  await expect(page.getByLabel('Selecionar modelo ou runtime do chat')).toBeVisible()
  await expect(page.locator('.chat-layout')).toHaveCSS('border-radius', '12px')
  await expect(page.locator('.composer-shell')).toHaveCSS('box-shadow', 'none')
  await page.getByRole('button', { name: 'Personalizar', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Personalização do chat' })).toBeVisible()
  await expect(page.getByText('Cantos (12 px)')).toBeVisible()
  await expect(page.locator('.conversation')).toHaveCSS('--chat-accent', '#007e75')
  await page.getByRole('button', { name: 'Cor de destaque #2563eb' }).click()
  await expect(page.locator('.conversation')).toHaveCSS('--chat-accent', '#2563eb')
  await page.getByRole('button', { name: 'Usar configurações gerais' }).click()
  await expect(page.locator('.conversation')).toHaveCSS('--chat-accent', '#007e75')

  for (const [locale, datasetsTitle, chatTitle] of [
    ['es', 'Conjuntos de datos', 'Chat local'],
    ['zh', '数据集', '本地聊天'],
    ['pl', 'Zbiory danych', 'Czat lokalny'],
  ]) {
    await settings.click()
    await page.locator('#select-language').selectOption(locale)
    await datasets.click()
    await expect(page.getByRole('heading', { name: datasetsTitle })).toBeVisible()
    await chat.click()
    await expect(page.getByRole('heading', { name: chatTitle })).toBeVisible()
  }

  await page.reload()
  await chat.click()
  await expect(page.getByRole('heading', { name: 'Czat lokalny' })).toBeVisible()
  await expect(page.locator('.chat-layout')).toHaveCSS('border-radius', '12px')
})

test('post-training keeps continuation distinct from full resumption', async ({ page }) => {
  await page.goto('/')
  await page.locator('#tab-train').click()
  await page.getByRole('button', { name: 'Post-train from an existing adapter' }).click()
  await expect(page.getByLabel('Starting point')).toHaveValue('continue')
  await expect(page.getByLabel('Source adapter directory')).toBeVisible()
  await expect(page.getByText('Source adapter', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: /Apply continued SFT preset/ }).click()
  await expect(page.getByLabel('Output Path')).toHaveValue(/runs\/continued-\d+/)
  await page.getByLabel('Performance', { exact: true }).selectOption('safe')
  await expect(page.getByLabel('Performance', { exact: true })).toHaveValue('safe')
  await expect(page.getByLabel('Sequence packing', { exact: true })).toHaveValue('auto')
  await page.getByLabel('Starting point').selectOption('resume')
  await expect(page.getByLabel('Trainer checkpoint directory')).toBeVisible()
  await expect(page.getByText(/Restores weights, optimizer, scheduler, and step counter/)).toBeVisible()
  await page.getByLabel('Starting point').scrollIntoViewIfNeeded()
  await page.screenshot({ path: 'test-results/continuation-130.png', fullPage: true })
})

test('chat supports streaming, branching, @skill autocomplete, MCP validation, error states, and reload persistence', async ({ page }) => {
  await page.route('http://127.0.0.1:8080/v1/models', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: [{ id: 'local-test-model' }] }) })
  )
  await page.route('http://127.0.0.1:8080/sytra/preferences', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ temperature: 0.4 }) })
  )
  await page.route('http://127.0.0.1:8080/v1/chat/completions', route => {
    const sse = [
      'data: {"choices":[{"delta":{"content":"Hello from "}}]}',
      '',
      'data: {"choices":[{"delta":{"content":"Sytra 1.3.0"}}]}',
      '',
      'data: {"choices":[{"finish_reason":"stop","delta":{}}]}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    return route.fulfill({ contentType: 'text/event-stream', body: sse })
  })

  await page.goto('/')
  await page.locator('#tab-chat').click()
  await expect(page.getByRole('region', { name: 'Local model chat' }).or(page.locator('section.conversation'))).toBeVisible()

  await page.locator('details.mcp-settings summary').click()
  await expect(page.locator('#mcp-json-input')).toBeVisible()
  await expect(page.locator('#btn-apply-mcp-json')).toBeDisabled()
  await expect(page.locator('#mcp-name')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Connect', exact: true })).toBeDisabled()
  await expect(page.getByRole('button', { name: /Enable JavaScript sandbox/ })).toBeVisible()

  await page.locator('details.skills-panel summary').click()
  await page.locator('#skill-install-name').fill('reviewer')
  await page.locator('#skill-install-content').fill('# Reviewer\nCheck diffs cleanly.')
  await page.locator('#btn-install-skill').click()
  await expect(page.locator('.skill-tags')).toContainText('workspace:reviewer')
  await expect(page.locator('.skills-panel')).toContainText('✓ Active · @reviewer')

  const prompt = page.locator('#chat-prompt')
  await prompt.fill('@test')
  await prompt.dispatchEvent('input')
  await expect(page.getByRole('listbox', { name: 'Available skills' })).toBeVisible()
  await prompt.press('Escape')

  await prompt.fill('Hi local model')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByText('Hello from Sytra 1.3.0')).toBeVisible()

  await page.reload()
  await page.locator('#tab-chat').click()
  await expect(page.locator('.message-body', { hasText: 'Hi local model' })).toBeVisible()

  // User message is spacious, free of "Branch here" clutter, and carries a single Undo changes icon that rolls back cleanly.
  await expect(page.locator('.branch')).toHaveCount(0)
  const undoBtn = page.getByRole('button', { name: 'Undo changes', exact: true })
  await expect(undoBtn).toHaveCount(1)
  await undoBtn.click()
  await expect(page.locator('.message')).toHaveCount(0)
  await expect(prompt).toHaveValue('Hi local model')

  await page.route('http://127.0.0.1:8080/v1/chat/completions', route =>
    route.fulfill({ status: 500, body: 'backend error' })
  )
  await prompt.fill('Trigger error')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('Server returned 500')
})


test('chat supports thinking toggle, inline tool fallback, web search, permissions, rich charts/KPIs, and telemetry', async ({ page }) => {
  await page.addInitScript(() => {
    const installed = [
      { meta: { id: 'workspace:analyst', name: 'analyst', scope: 'workspace' }, content: '# Analyst\nAlways compare numbers clearly.' },
    ]
    ;(window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
      transformCallback: () => 1,
      invoke: async (cmd: string, args: Record<string, unknown> = {}) => {
        switch (cmd) {
          case 'get_hardware_info':
            return { backend: 'cuda', vram_mb: 12288, ram_mb: 16384 }
          case 'get_settings':
            return { hf_cache_dir: 'D:\\models', is_custom: false, main_memory_limit_mb: null, effective_main_memory_mb: 16384, detected_ram_mb: 16384 }
          case 'list_runs':
          case 'list_local_models':
          case 'list_catalog':
            return []
          case 'chat_mcp_token':
            return 'test-mcp-token'
          case 'list_chat_skills':
            return installed.map(s => s.meta)
          case 'read_chat_skill':
            return installed.find(s => s.meta.id === args.id)?.content ?? ''
          case 'install_chat_skill': {
            const name = String(args.name || 'custom')
            const entry = { meta: { id: `workspace:${name}`, name, scope: 'workspace' }, content: String(args.content || '') }
            installed.push(entry)
            return entry.meta
          }
          default:
            return null
        }
      },
    }
  })

  await page.route('http://127.0.0.1:8080/v1/models', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: [{ id: 'qwen-legacy-3b' }] }) }),
  )
  await page.route('http://127.0.0.1:8080/sytra/preferences', route =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ temperature: 0.3, context_window: 8192, model: 'qwen-legacy-3b' }),
    }),
  )
  await page.route('http://127.0.0.1:8080/sytra/mcp/servers', route =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ servers: [], sandbox: { available: false, enabled: false } }),
    }),
  )

  let webSearchCalled = false
  await page.route('http://127.0.0.1:8080/sytra/mcp/web-search', async route => {
    webSearchCalled = true
    const body = route.request().postDataJSON() as { query?: string }
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        query: body.query ?? '',
        results: [{ title: 'Latency Report', url: 'https://example.com/report', snippet: 'Q2 latency dropped 25%.' }],
      }),
    })
  })

  let completionTurn = 0
  await page.route('http://127.0.0.1:8080/v1/chat/completions', async route => {
    const reqBody = route.request().postDataJSON() as { tools?: unknown[]; messages?: { role: string; content: string }[] }
    completionTurn += 1
    // Simulate a legacy model server that rejects native OpenAI `tools` with HTTP 400 on first try.
    if (completionTurn === 1 && Array.isArray(reqBody.tools) && reqBody.tools.length > 0) {
      return route.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ error: 'tools unsupported in template' }) })
    }
    if (completionTurn === 2) {
      const sse = [
        'data: {"choices":[{"delta":{"content":"<think>Planning web search query...</think>Searching now.\\n"}}]}',
        '',
        'data: {"choices":[{"delta":{"content":"<tool_call>{\\"name\\":\\"web_search\\",\\"arguments\\":{\\"query\\":\\"sytra latency\\"}}</tool_call>"}}]}',
        '',
        'data: {"choices":[{"finish_reason":"stop","delta":{}}],"timings":{"predicted_per_second":48.6,"predicted_n":24,"prompt_n":110}}',
        '',
        'data: [DONE]',
        '',
      ].join('\n')
      return route.fulfill({ contentType: 'text/event-stream', body: sse })
    }
    const lastMsg = reqBody.messages?.[reqBody.messages.length - 1]
    const systemMsg = reqBody.messages?.find(m => m.role === 'system')?.content || ''
    const hasRuntimeContext = systemMsg.includes('SKILL.md') && systemMsg.includes('MCP')
    const hasToolFeedback = lastMsg?.role === 'user' && lastMsg.content.includes('[Tool result for web_search]')
    const markdownReply = [
      hasToolFeedback && hasRuntimeContext ? '### Latency Comparison' : 'Missing tool result',
      '',
      '| Quarter | Latency | Throughput |',
      '| --- | ---: | ---: |',
      '| Q1 | 40 | 120 |',
      '| Q2 | 30 | 160 |',
      '',
      '```json',
      '{"type":"line","title":"NVIDIA Asset Values","labels":["Jan","Feb","Mar","Apr"],"values":[100,105,110,115],"unit":"$"}',
      '```',
      '',
      '```kpi',
      '{"title":"Benchmark","items":[{"label":"Peak Speed","value":"52.4 tk/s","delta":"+12%"}]}',
      '```',
    ].join('\n')
    const sse = [
      `data: ${JSON.stringify({ choices: [{ delta: { content: markdownReply } }] })}`,
      '',
      'data: {"choices":[{"finish_reason":"stop","delta":{}}],"timings":{"predicted_per_second":52.4,"predicted_n":64,"prompt_n":280}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    return route.fulfill({ contentType: 'text/event-stream', body: sse })
  })

  await page.goto('/')
  await page.locator('#tab-chat').click()

  await expect(page.locator('[data-testid="chat-context-pill"]')).toContainText('8,192 tok')
  await page.locator('#btn-toggle-web-search').click()
  await expect(page.locator('#btn-toggle-web-search')).toHaveAttribute('aria-pressed', 'true')

  // Attach @analyst skill and verify Skills permission pill stays enabled.
  await page.locator('details.skills-panel summary').click()
  await page.getByRole('button', { name: /Use · @analyst/ }).click()
  await expect(page.locator('#btn-toggle-skills')).toHaveAttribute('aria-pressed', 'true')

  // Cycle MCP permission pill to Auto and back to Ask.
  await page.getByRole('group', { name: 'MCP tools' }).getByRole('button', { name: 'Auto', exact: true }).click()
  await page.getByRole('group', { name: 'MCP tools' }).getByRole('button', { name: 'Ask', exact: true }).click()

  await page.locator('#chat-prompt').fill('Compare Q1 and Q2 latency with web search')
  await page.getByRole('button', { name: 'Send', exact: true }).click()

  await expect.poll(() => webSearchCalled).toBe(true)
  await expect(page.getByText('Latency Comparison')).toBeVisible()
  await expect(page.locator('[data-testid="chat-speed-pill"]')).toContainText('52.4 tk/s')
  await expect(page.locator('[data-testid="chat-context-pill"]')).toContainText('280 / 8,192 tok')

  // Reasoning, hammer tool icon, and list skill icon appear in clickable trace drawer.
  const trace = page.locator('details.activity-trace').last()
  await expect(trace.locator('.bi-lightbulb').first()).toBeVisible()
  await expect(trace.locator('.bi-hammer').first()).toBeVisible()
  await expect(trace.locator('.bi-list-ul').first()).toBeVisible()
  await trace.locator('summary.activity-summary').click()
  await expect(trace).toContainText('Planning web search query...')
  await expect(trace).toContainText('Web/web_search')
  await expect(trace).toContainText('@analyst')

  // Structured table + interactive ```json auto-promoted chart + KPI card render cleanly.
  await expect(page.locator('.rich-table')).toContainText('Q1')
  await page.locator('.rich-table-card').getByRole('button', { name: 'Bar', exact: true }).click()
  await expect(page.locator('.rich-table-card [data-chart-kind="bar"] svg')).toBeVisible()

  const nvdaChart = page.locator('.rich-chart[data-chart-kind="line"]').last()
  await expect(nvdaChart.locator('svg')).toBeVisible()
  await expect(nvdaChart).toContainText('NVIDIA Asset Values')
  await expect(nvdaChart).toContainText('+15%')
  await nvdaChart.getByRole('button', { name: 'Area', exact: true }).click()
  await expect(page.locator('.rich-chart[data-chart-kind="area"] svg')).toBeVisible()
  await expect(page.locator('.rich-kpi-wrap')).toContainText('Peak Speed')
  await expect(page.locator('.rich-kpi-wrap')).toContainText('+12%')
})


test('chat customizer, auto-route without emojis, Lil Sytra compact pet, model discovery, GIF attachments, and 75%/85% auto-compaction', async ({ page }) => {
  await page.addInitScript(() => {
    ;(window as unknown as Record<string, unknown>).__mockMemories = []
    ;(window as unknown as Record<string, unknown>).__mockSkills = []
    ;(window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {
      transformCallback: () => 1,
      invoke: async (cmd: string, args: Record<string, unknown> = {}) => {
        const w = window as unknown as Record<string, any>
        switch (cmd) {
          case 'get_hardware_info':
            return { backend: 'cuda', vram_mb: 12288, ram_mb: 16384 }
          case 'get_settings':
            return { hf_cache_dir: 'D:\\local-models\\hf-cache', is_custom: true, main_memory_limit_mb: null, effective_main_memory_mb: 16384, detected_ram_mb: 16384 }
          case 'list_runs':
          case 'list_local_models':
          case 'list_catalog':
            return []
          case 'discover_chat_targets':
            return {
              providers: {
                sytra_running: true,
                sytra_models: ['sytra-local-q4'],
                ollama_running: true,
                ollama_installed: true,
                ollama_models: [{ name: 'qwen2.5:7b-ollama', size_gb: 4.7, parameter_size: '7B', quantization: 'Q4_K_M', language_scope: 'multilingual' }],
                lmstudio_running: true,
                lmstudio_models: ['lmstudio-community/Qwen2.5-3B-GGUF'],
              },
              local_models: [
                { id: 'Qwen2.5-0.5B', name: 'Qwen2.5-0.5B.gguf', category: 'hf-cache', path: 'D:\\local-models\\hf-cache\\Qwen2.5-0.5B.gguf', size_gb: 0.49, format: 'gguf' },
              ],
            }
          case 'unload_chat_model':
            return 'Unloaded active GPU model.'
          case 'get_ollama_context':
            return 1000
          case 'list_chat_agents':
            return w.__mockAgents
          case 'save_chat_agent': {
            const ag = args.agent
            w.__mockAgents = [...w.__mockAgents.filter((x: any) => x.id !== ag.id), ag]
            return ag
          }
          case 'delete_chat_agent':
            w.__mockAgents = w.__mockAgents.filter((x: any) => x.id !== args.id)
            return null
          case 'list_agent_memories':
            return w.__mockMemories
          case 'save_agent_memory': {
            const entry = {
              id: `mem-${w.__mockMemories.length + 1}`,
              agent_id: args.agentId || 'default',
              session_id: args.sessionId || 's1',
              session_title: args.sessionTitle || 'Session',
              kind: args.kind || 'fact',
              summary: args.summary || '',
              tags: args.tags || [],
              created_at: Math.floor(Date.now() / 1000),
            }
            w.__mockMemories = [entry, ...w.__mockMemories]
            return entry
          }
          case 'recall_agent_memories':
            return w.__mockMemories.slice(0, 5)
          case 'list_chat_skills':
            return w.__mockSkills.map((s: any) => s.meta)
          case 'read_chat_skill':
            return w.__mockSkills.find((s: any) => s.meta.id === args.id)?.content ?? ''
          case 'install_chat_skill': {
            const name = String(args.name || 'skill').toLowerCase().replace(/\s+/g, '-')
            const item = { meta: { id: `workspace:${name}`, name, scope: 'workspace' }, content: String(args.content || '') }
            w.__mockSkills = [...w.__mockSkills.filter((s: any) => s.meta.id !== item.meta.id), item]
            return item.meta
          }
          default:
            return null
        }
      },
    }
  })

  await page.route('http://127.0.0.1:8080/v1/models', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: [{ id: 'sytra-local-q4' }] }) }),
  )
  await page.route('http://127.0.0.1:8080/sytra/preferences', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ temperature: 0.3, context_window: 1000, model: 'sytra-local-q4' }) }),
  )
  await page.route('http://127.0.0.1:11434/v1/chat/completions', route => {
    const sse = [
      `data: ${JSON.stringify({ choices: [{ delta: { content: 'Ollama custom reply ok.' } }] })}`,
      '',
      'data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":780,"completion_tokens":30}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    return route.fulfill({ contentType: 'text/event-stream', body: sse })
  })
  await page.route('http://127.0.0.1:11434/props', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ default_generation_settings: { n_ctx: 1000 } }) }),
  )
  await page.route('http://127.0.0.1:8080/v1/chat/completions', route => {
    const sse = [
      `data: ${JSON.stringify({ choices: [{ delta: { content: 'Reply' } }] })}`,
      '',
      'data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":900,"completion_tokens":20}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    return route.fulfill({ contentType: 'text/event-stream', body: sse })
  })

  await page.goto('/')
  await page.locator('#tab-chat').click()

  // Agents and Customize pills are aligned on the right in .telemetry-bar matching telemetry height
  const agentsPill = page.locator('.telemetry-bar #btn-open-agents')
  const customizerPill = page.locator('.telemetry-bar #btn-open-customizer')
  await expect(agentsPill).toBeVisible()
  await expect(customizerPill).toBeVisible()
  const agBox = await agentsPill.boundingBox()
  const custBox = await customizerPill.boundingBox()
  const ctxBox = await page.locator('[data-testid="chat-context-pill"]').boundingBox()
  const speedBox = await page.locator('[data-testid="chat-speed-pill"]').boundingBox()
  expect(agBox).not.toBeNull()
  expect(custBox).not.toBeNull()
  expect(ctxBox).not.toBeNull()
  expect(speedBox).not.toBeNull()
  expect(Math.abs(agBox!.height - custBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(custBox!.height - ctxBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(custBox!.height - speedBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(agBox!.y - custBox!.y)).toBeLessThanOrEqual(2)
  expect(custBox!.x).toBeGreaterThan(agBox!.x)

  // Compact centered Lil Sytra companion pet
  const petDock = page.locator('.floating-pet-dock')
  await expect(petDock).toContainText('Lil Sytra')
  const dockBox = await petDock.boundingBox()
  const stagePetBox = await petDock.locator('.pet-stage').boundingBox()
  expect(dockBox).not.toBeNull()
  expect(stagePetBox).not.toBeNull()
  expect(dockBox!.height).toBeLessThanOrEqual(72)
  const dockCenterX = dockBox!.x + dockBox!.width / 2
  const petCenterX = stagePetBox!.x + stagePetBox!.width / 2
  expect(Math.abs(dockCenterX - petCenterX)).toBeLessThanOrEqual(2)

  // 1. Model picker discovers Sytra, Ollama, LM Studio, HF-cache, Auto-route without emojis, & Turn Off Model
  const select = page.locator('#chat-model-select')
  const selectText = await select.innerText()
  expect(selectText).toContain('Auto-route')
  expect(selectText).not.toMatch(/[\u{1F300}-\u{1FAFF}]/u)
  await expect(select).toContainText('qwen2.5:7b-ollama')
  await expect(select).toContainText('Qwen2.5-3B-GGUF')
  await expect(select).toContainText('[HF Cache] Qwen2.5-0.5B.gguf')

  // Auto-route selects model and displays language badge + Turn Off Model sleep state
  await select.selectOption('auto-route')
  await page.locator('#chat-prompt').fill('Ola, tudo bem?')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.locator('[data-testid="chat-route-pill"]')).toContainText('Auto-route ->')
  await expect(page.locator('[data-testid="chat-lang-pill"]')).toContainText('Multilingual')

  await page.locator('#btn-unload-model').click()
  await expect(page.locator('#btn-unload-model')).toContainText('Turn Off Model')
  await expect(page.locator('[data-testid="chat-sleep-pill"]')).toBeVisible()

  const unloadBox = await page.locator('#btn-unload-model').boundingBox()
  const idleCheckBox = await page.locator('.power-check').boundingBox()
  const idleWrapBox = await page.locator('.idle-select-wrap').boundingBox()
  expect(unloadBox).not.toBeNull()
  expect(idleCheckBox).not.toBeNull()
  expect(idleWrapBox).not.toBeNull()
  expect(Math.abs(unloadBox!.height - idleCheckBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(idleCheckBox!.height - idleWrapBox!.height)).toBeLessThanOrEqual(1)

  await select.selectOption('ollama::http://127.0.0.1:11434::qwen2.5:7b-ollama')

  // 2. Chat appearance customizer (Split Bubbles + Top composer + Lil Sytra pet look)
  await page.locator('#btn-open-customizer').click()
  await page.getByRole('button', { name: /Split Bubbles/ }).click()
  await page.getByRole('button', { name: 'Top', exact: true }).click()
  await page.getByRole('button', { name: 'Kitsune', exact: true }).click()
  await expect(page.locator('.chat-stage.composer-top')).toBeVisible()
  await page.getByRole('button', { name: 'Done', exact: true }).click()

  // 3. Attach a valid GIF image to message & send multiple turns to hit 75% alert and 85% auto-compaction
  const gifBytes = Buffer.from('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7', 'base64')
  await page.locator('input[accept*="image/*"]').last().setInputFiles({
    name: 'reaction.gif',
    mimeType: 'image/gif',
    buffer: gifBytes,
  })
  await expect(page.locator('.attach-chip')).toContainText('reaction.gif (GIF)')

  for (const text of [
    'Turn 1',
    'Turn 2',
    'Turn 3 with a much longer user message spanning multiple clauses so wrapping and right edge alignment with the composer border are verified cleanly.',
  ]) {
    await page.locator('#chat-prompt').fill(text)
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.locator('.message')).not.toHaveCount(0)
  }

  // Short & long user messages align flush with composer right edge, separated cleanly from left-aligned model output
  const userShortBox = await page.locator('.message.user[data-align="right"]').first().boundingBox()
  const userLongBox = await page.locator('.message.user[data-align="right"]').last().boundingBox()
  const composerBox = await page.locator('.composer-shell').boundingBox()
  const assistantMsg = page.locator('.message:not(.user)[data-align="left"]').first()
  const assistantBox = await assistantMsg.boundingBox()
  expect(userShortBox).not.toBeNull()
  expect(userLongBox).not.toBeNull()
  expect(composerBox).not.toBeNull()
  expect(assistantBox).not.toBeNull()
  expect(Math.abs((userShortBox!.x + userShortBox!.width) - (composerBox!.x + composerBox!.width))).toBeLessThanOrEqual(4)
  expect(Math.abs((userLongBox!.x + userLongBox!.width) - (composerBox!.x + composerBox!.width))).toBeLessThanOrEqual(4)
  expect(Math.abs(assistantBox!.x - composerBox!.x)).toBeLessThanOrEqual(4)
  expect(userLongBox!.y - (assistantBox!.y + assistantBox!.height)).toBeGreaterThanOrEqual(24)

  await expect(assistantMsg).toContainText('Lil Sytra')
  const assistantStyle = await assistantMsg.evaluate((el) => {
    const cs = getComputedStyle(el)
    return { bg: cs.backgroundColor, borderW: cs.borderTopWidth, bodyFontPx: parseFloat(getComputedStyle(el.querySelector('.rich-message') || el).fontSize) }
  })
  expect(assistantStyle.bg).toBe('rgba(0, 0, 0, 0)')
  expect(assistantStyle.borderW).toBe('0px')
  expect(assistantStyle.bodyFontPx).toBeGreaterThanOrEqual(15)
  await expect(page.locator('.msg-media-card img').first()).toBeVisible()

  // Turn 3 hit 780 / 1000 = 78% -> Compaction Alert & Approve/Continue bar shows up!
  await expect(page.locator('.compaction-alert')).toBeVisible()
  await expect(page.locator('#btn-dismiss-compact')).toBeVisible()
  await expect(page.locator('#btn-approve-compact')).toBeVisible()

  // Dismiss at 78%, then hit 85%+ with sytra endpoint returning 900/1000 -> auto-compacts into compacted message & saves memory!
  await page.locator('#btn-dismiss-compact').click()
  await expect(page.locator('.compaction-alert')).toBeHidden()
  await select.selectOption('sytra::http://127.0.0.1:8080::sytra-local-q4')
  await page.locator('#chat-prompt').fill('Turn 4 trigger 85 percent')
  await page.getByRole('button', { name: 'Send', exact: true }).click()

  await expect(page.locator('.compacted-msg')).toBeVisible()
  await expect(page.getByRole('status')).toContainText('85%')

  // Verify compacted digest is saved in cross-session memory backend
  const storedMemories = await page.evaluate(() => (window as unknown as { __mockMemories: { kind: string; summary: string }[] }).__mockMemories)
  expect(storedMemories.length).toBeGreaterThan(0)
  expect(storedMemories[0].kind).toBe('compaction')
  expect(storedMemories[0].summary).toContain('Turn 1')
})


test('Lil Sytra custom agents drawer, voice dictation, Kokoro-82M setup/live chat, and proactive auto-watch without prompting', async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as Record<string, any>
    w.__mockAgents = []
    w.__mockMemories = []
    w.__kokoroSetupDone = false
    w.__kokoroSynthCalls = []

    // Deterministic mock SpeechRecognition & Audio playback for headless E2E
    class MockSpeechRecognition {
      continuous = false
      interimResults = false
      lang = 'en-US'
      onresult: ((ev: any) => void) | null = null
      onerror: ((ev: any) => void) | null = null
      onend: (() => void) | null = null
      start() {
        w.__activeMockRec = this
      }
      stop() {
        this.onend?.()
      }
      abort() {
        /* stopped */
      }
      __emit(transcript: string) {
        this.onresult?.({
          resultIndex: 0,
          results: [{ isFinal: true, 0: { transcript } }],
        })
      }
    }
    w.SpeechRecognition = MockSpeechRecognition
    w.webkitSpeechRecognition = MockSpeechRecognition
    w.Audio = class MockAudio {
      src = ''
      onended: (() => void) | null = null
      onerror: (() => void) | null = null
      play() {
        setTimeout(() => this.onended?.(), 30)
        return Promise.resolve()
      }
      pause() { /* stopped */ }
    }

    w.__TAURI_INTERNALS__ = {
      transformCallback: () => 1,
      invoke: async (cmd: string, args: Record<string, unknown> = {}) => {
        switch (cmd) {
          case 'get_hardware_info':
            return { backend: 'cuda', vram_mb: 12288, ram_mb: 16384 }
          case 'get_settings':
            return { hf_cache_dir: 'D:\\local-models\\hf-cache', is_custom: true, main_memory_limit_mb: null, effective_main_memory_mb: 16384, detected_ram_mb: 16384 }
          case 'list_runs':
          case 'list_local_models':
          case 'list_catalog':
            return []
          case 'discover_chat_targets':
            return {
              providers: {
                sytra_running: true,
                sytra_models: ['sytra-qwen-q4'],
                ollama_running: false,
                ollama_installed: false,
                ollama_models: [],
                lmstudio_running: false,
                lmstudio_models: [],
              },
              local_models: [],
            }
          case 'list_chat_agents':
            return w.__mockAgents
          case 'save_chat_agent': {
            const ag = args.agent
            w.__mockAgents = [...w.__mockAgents.filter((x: any) => x.id !== ag.id), ag]
            return ag
          }
          case 'list_agent_memories':
            return w.__mockMemories
          case 'save_agent_memory': {
            const entry = {
              id: `mem-${w.__mockMemories.length + 1}`,
              agent_id: args.agentId || 'default',
              session_id: args.sessionId || 's1',
              session_title: args.sessionTitle || 'Session',
              kind: args.kind || 'routine',
              summary: args.summary || '',
              tags: args.tags || [],
              created_at: Math.floor(Date.now() / 1000),
            }
            w.__mockMemories = [entry, ...w.__mockMemories]
            return entry
          }
          case 'recall_agent_memories':
            return []
          case 'list_chat_skills':
            return []
          case 'voice_status':
            return {
              env_ready: w.__kokoroSetupDone,
              kokoro_model_ready: w.__kokoroSetupDone,
              server_running: w.__kokoroSetupDone,
              installing: false,
              stage: w.__kokoroSetupDone ? 'ready' : 'idle',
              message: w.__kokoroSetupDone ? 'Kokoro-82M ready' : '',
              active_port: w.__kokoroSetupDone ? 8089 : null,
              model_dir: 'D:\\.sytra\\voice-models\\kokoro-82m',
              voices: [
                { id: 'auto', label: 'Auto (EN / PT-BR)', lang: 'auto' },
                { id: 'af_heart', label: 'Kokoro Heart (US F)', lang: 'en-us' },
                { id: 'pf_dora', label: 'Kokoro Dora (PT-BR F)', lang: 'pt-br' },
              ],
            }
          case 'setup_kokoro_voice':
            w.__kokoroSetupDone = true
            return {
              env_ready: true,
              kokoro_model_ready: true,
              server_running: true,
              installing: false,
              stage: 'ready',
              message: 'Kokoro-82M ready',
              active_port: 8089,
              model_dir: 'D:\\.sytra\\voice-models\\kokoro-82m',
              voices: [
                { id: 'auto', label: 'Auto (EN / PT-BR)', lang: 'auto' },
                { id: 'af_heart', label: 'Kokoro Heart (US F)', lang: 'en-us' },
                { id: 'pf_dora', label: 'Kokoro Dora (PT-BR F)', lang: 'pt-br' },
              ],
            }
          case 'voice_synthesize':
            w.__kokoroSynthCalls.push({ text: args.text, voice: args.voice })
            return {
              audio_base64: 'UklGRjIAAABXQVZFZm10IBAAAAABAAEAQBYAAIA+AAACABAAZGF0YQ4AAAAAAAAAAAAAAAAAAAAAAA==',
              mime: 'audio/wav',
            }
          default:
            return null
        }
      },
    }
  })

  await page.route('http://127.0.0.1:8080/v1/models', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ data: [{ id: 'sytra-qwen-q4' }] }) }),
  )
  await page.route('http://127.0.0.1:8080/sytra/preferences', route =>
    route.fulfill({ contentType: 'application/json', body: JSON.stringify({ temperature: 0.3, context_window: 4096, model: 'sytra-qwen-q4' }) }),
  )
  await page.route('http://127.0.0.1:8080/v1/chat/completions', async route => {
    const body = route.request().postDataJSON() as { messages: { role: string; content: string }[] }
    const lastUser = [...(body.messages ?? [])].reverse().find(m => m.role === 'user')?.content || ''
    const reply = lastUser.includes('Proactive Autonomous Update')
      ? '### Autonomous Health Briefing\n\n| Metric | Value |\n| --- | --- |\n| GPU Backend | CUDA 12GB |\n\nAll local systems healthy.'
      : 'Hello! Voice pipeline with Kokoro 82M is working smoothly.'
    const sse = [
      `data: ${JSON.stringify({ choices: [{ delta: { content: reply } }] })}`,
      '',
      'data: {"choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":90,"completion_tokens":24}}',
      '',
      'data: [DONE]',
      '',
    ].join('\n')
    await route.fulfill({ contentType: 'text/event-stream', body: sse })
  })

  await page.goto('/')
  await page.locator('#tab-chat').click()

  // 1. Clicking floating Lil Sytra mascot opens Custom Agent Config Studio (NOT Customize Chat!)
  await page.locator('.floating-pet-dock .pet-stage').click()
  await expect(page.getByRole('region', { name: 'Agent & Memory Studio' }).or(page.locator('.studio-drawer'))).toBeVisible()
  await expect(page.locator('.customizer-drawer')).toHaveCount(0)

  // Configure a custom agent with the quick System Monitor preset
  await page.getByRole('button', { name: 'System Monitor', exact: true }).click()
  await expect(page.locator('#agent-name-input')).toHaveValue('Lil Sytra')
  await page.locator('#btn-save-agent').click()
  await expect(page.locator('.studio-status')).toContainText('Saved agent "Lil Sytra"')
  await page.getByRole('button', { name: 'Close', exact: true }).click()

  // Verify aligned compact composer controls (26px) & embedded voice controls (28px beside Send)
  const webBox = await page.locator('#btn-toggle-web-search').boundingBox()
  const mcpBox = await page.locator('.perm-group').boundingBox()
  const skillsBox = await page.locator('#btn-toggle-skills').boundingBox()
  const micBox = await page.locator('#btn-voice-mic').boundingBox()
  const liveBox = await page.locator('#btn-live-voice').boundingBox()
  const sendBox = await page.getByRole('button', { name: 'Send', exact: true }).boundingBox()
  expect(webBox).not.toBeNull()
  expect(mcpBox).not.toBeNull()
  expect(skillsBox).not.toBeNull()
  expect(micBox).not.toBeNull()
  expect(liveBox).not.toBeNull()
  expect(sendBox).not.toBeNull()
  expect(Math.abs(webBox!.height - mcpBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(skillsBox!.height - mcpBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(micBox!.height - liveBox!.height)).toBeLessThanOrEqual(1)
  expect(Math.abs(micBox!.height - sendBox!.height)).toBeLessThanOrEqual(1)
  const sendSpanBox = await page.getByRole('button', { name: 'Send', exact: true }).locator('span').boundingBox()
  expect(sendSpanBox).not.toBeNull()
  const sendCenterY = sendBox!.y + sendBox!.height / 2
  const spanCenterY = sendSpanBox!.y + sendSpanBox!.height / 2
  expect(Math.abs(sendCenterY - spanCenterY)).toBeLessThanOrEqual(1.5)

  // 2. Open Voice / Kokoro-82M config panel, run Setup Kokoro 82M, pick pf_dora voice
  await page.locator('#btn-voice-panel').click()
  await expect(page.locator('#voice-kokoro-panel')).toBeVisible()
  await page.locator('#btn-setup-kokoro').click()
  await expect(page.locator('[data-testid="kokoro-ready-badge"]')).toBeVisible()
  await page.locator('#select-kokoro-voice').selectOption('pf_dora')

  // 3. Single Mic dictation populates composer input
  await page.locator('#btn-voice-mic').click()
  await expect(page.locator('#btn-voice-mic')).toHaveAttribute('aria-pressed', 'true')
  await page.evaluate(() => (window as any).__activeMockRec.__emit('Check status via voice'))
  await expect(page.locator('#chat-prompt')).toHaveValue('Check status via voice')

  // 4. Start Real-Time Live Voice chat -> speaking triggers automatic turn + Kokoro-82M synthesis
  await page.locator('#btn-live-voice').click()
  await expect(page.locator('#btn-live-voice')).toHaveAttribute('aria-pressed', 'true')
  await page.evaluate(() => (window as any).__activeMockRec.__emit('Oi Lil Sytra, tudo bem?'))
  await expect(page.locator('.message')).toHaveCount(2)
  await expect(page.locator('.message').last()).toContainText('Kokoro 82M is working smoothly')
  await expect.poll(async () => page.evaluate(() => (window as any).__kokoroSynthCalls.length)).toBeGreaterThan(0)
  await page.locator('#btn-live-voice').click()
  await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeVisible()

  // 5. Proactive Auto-Watch: Lil Sytra acts and surfaces an autonomous briefing card without prompting!
  await page.locator('#btn-proactive-watch').click()
  await expect(page.locator('.proactive-badge').first()).toBeVisible()
  await expect(page.locator('.message').last()).toContainText('Autonomous Health Briefing')
  await expect(page.locator('.rich-table').last()).toContainText('CUDA 12GB')
})

test('scheduled agent runs on the Chat timer and saves its result', async ({ page }) => {
  const started = new Date('2026-11-20T18:00:00Z')
  await page.clock.install({ time: started })
  await page.addInitScript((lastRun) => {
    localStorage.setItem('sytra-locale', 'en')
    localStorage.setItem('sytra-mock-agents-v1', JSON.stringify([{
      id: 'fixture-agent', name: 'Fixture Agent', role: 'Tester', system_prompt: 'Report the result.',
      avatar_kind: 'pochi', computer_mode: 'sandbox', skill_ids: [], routine_enabled: true,
      routine_interval_mins: 1, routine_prompt: 'Report the fixture status.', last_routine_at: lastRun,
    }]))
  }, started.getTime() / 1000)
  await page.route('http://127.0.0.1:8080/props', route =>
    route.fulfill({ json: { model: 'routine-fixture', context_window: 8192 } }),
  )
  await page.route('http://127.0.0.1:8080/v1/models', route =>
    route.fulfill({ json: { data: [{ id: 'routine-fixture' }] } }),
  )
  let runs = 0
  await page.route('http://127.0.0.1:8080/v1/chat/completions', async route => {
    const body = route.request().postDataJSON() as { messages: { role: string; content: string }[] }
    expect(body.messages.at(-1)?.content).toContain('Report the fixture status.')
    runs += 1
    return route.fulfill({
      contentType: 'text/event-stream',
      body: `data: {"choices":[{"delta":{"content":"Routine result ${runs}"}}]}\n\ndata: [DONE]\n\n`,
    })
  })

  await page.goto('/')
  await page.locator('#tab-chat').click()
  await expect(page.getByRole('heading', { name: 'Chat', exact: true })).toBeVisible()
  await page.clock.runFor(30_000)
  expect(runs).toBe(0)
  await page.clock.runFor(30_000)
  await expect(page.getByText('Routine result 1', { exact: true })).toBeVisible()
  await expect.poll(() => page.evaluate(() =>
    JSON.parse(localStorage.getItem('sytra-mock-memories-v1') || '[]')
      .filter((entry: { agent_id: string; kind: string }) => entry.agent_id === 'fixture-agent' && entry.kind === 'routine').length,
  )).toBe(1)
  await page.clock.runFor(30_000)
  expect(runs).toBe(1)
  await page.clock.runFor(30_000)
  await expect(page.getByText('Routine result 2', { exact: true })).toBeVisible()
  await page.locator('#tab-datasets').click()
  await page.clock.runFor(90_000)
  expect(runs).toBe(2)
})

test('models hub configures Sytra-Helix 30B+ MoE indexing, oracle generation, and unverified server launch', async ({ page }) => {
  await page.addInitScript(() => {
    const tauriCalls: { cmd: string; args: Record<string, unknown> }[] = []
    ;(window as any).__helixTauriCalls = tauriCalls
    ;(window as any).__TAURI_INTERNALS__ = {
      invoke: async (cmd: string, args: Record<string, unknown> = {}) => {
        tauriCalls.push({ cmd, args })
        switch (cmd) {
          case 'get_hardware_info':
            return { backend: 'cuda', gpu_name: 'RTX 3060', vram_mb: 12288, ram_mb: 16384 }
          case 'get_settings':
            return {
              hf_cache_dir: 'D:\\local-models\\hf-cache',
              is_custom: true,
              main_memory_limit_mb: 12000,
              effective_main_memory_mb: 12000,
              detected_ram_mb: 16384,
              default_context_window: 4096,
              default_temperature: 0.7,
              enable_flash_attention: true,
              kv_cache_quant: 'q8_0',
              vram_limit_mb: 8192,
              cpu_kv_cache: false,
            }
          case 'list_catalog':
            return []
          case 'list_local_models':
            return [{
              id: 'RedHatAI/Qwen3-30B-A3B-quantized.w4a16',
              name: 'RedHatAI--Qwen3-30B-A3B-quantized.w4a16',
              category: 'hf-cache',
              path: 'D:\\local-models\\RedHatAI--Qwen3-30B-A3B-quantized.w4a16',
              size_gb: 15.57,
              format: 'safetensors',
            }]
          case 'get_download_status':
            return null
          case 'chat_server_running':
            return false
          case 'get_model_settings':
            return {
              context_length: null, temperature: null, vram_limit_mb: null,
              vram_guardrail_mb: null, kv_cache_quant: null, cpu_kv_cache: null,
              flash_attention: null, idle_ttl_minutes: null, draft_model_path: null,
              allow_unverified: false,
            }
          case 'build_moe_index':
            return { runtime_manifest: '.sytra-runtime.json', experts_indexed: 6144, dense_bytes: 1737391104, forward_verified: false }
          case 'generate_moe_oracle':
            return { oracle: 'D:\\local-models\\RedHatAI--Qwen3-30B-A3B-quantized.w4a16\\.sytra-oracle.json' }
          case 'plan_inference':
            return {
              compatible: true,
              backend: 'sytra_moe',
              reasons: ['Qwen3 MoE native runtime ready'],
              warnings: [],
              estimates: {
                architecture: 'Qwen3MoeForCausalLM',
                quantization: 'INT4-G128',
                is_moe: true,
                total_params: 30519328768,
                active_params_per_token: 3340000000,
                peak_vram_mb: 8192,
                peak_ram_mb: 12000,
                helix_resource_plan: {
                  profile: 'vram_8gb_hybrid',
                  execution_strategy: 'hybrid-gpu-attention-cpu-experts-double-buffer',
                  tier0_vram_attention_mb: 3440,
                  tier0_vram_hot_expert_cache_mb: 3441,
                  tier1_pinned_ram_pool_mb: 9600,
                  tier2_nvme_weight_bytes: 0,
                  effective_estimated_tps: 3.7,
                  meets_minimum_envelope: true,
                },
              },
              command: ['sytra-engine', 'serve', '--allow-unverified-checkpoint'],
            }
          case 'start_chat_server':
            return true
          default:
            return null
        }
      },
    }
  })

  await page.goto('/')
  await page.getByRole('button', { name: /Switch to Advanced mode/ }).click()

  await page.locator('#serve-model-path').fill('D:\\local-models\\RedHatAI--Qwen3-30B-A3B-quantized.w4a16')
  await page.getByRole('button', { name: /Build expert index/ }).click()
  await expect(page.locator('.server-status')).toContainText('6144')

  await page.getByRole('button', { name: /Generate Oracle/ }).click()
  await expect(page.locator('.server-status')).toContainText('.sytra-oracle.json')

  const allowBox = page.locator('#native-allow-unverified')
  await expect(allowBox).toBeVisible()
  await allowBox.check()
  await expect(allowBox).toBeChecked()

  await page.getByRole('button', { name: /Inspect plan/ }).click()
  await expect(page.locator('#active-runtime-summary')).toContainText('30.5B')
  await expect(page.locator('#active-runtime-summary')).toContainText('Within host MoE envelope')

  await page.locator('#start-model-server').click()
  await expect(page.locator('.server-status')).toContainText('OpenAI-compatible model server is running.')

  const calls = await page.evaluate(() => (window as any).__helixTauriCalls as { cmd: string; args: Record<string, unknown> }[])
  const startCall = calls.find(c => c.cmd === 'start_chat_server')
  expect(startCall).toBeTruthy()
  expect(startCall!.args.allowUnverified).toBe(true)
})
