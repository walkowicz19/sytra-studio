import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildProactiveBriefingPrompt,
  cleanSpeechText,
  computeRmsLevel,
  detectSpeechLangTag,
  downsampleFloat32,
  encodePcm16WavBase64,
  extractReadySentences,
} from '../src/features/chat/voiceHelpers.ts'

test('cleanSpeechText normalizes markdown, URLs, and code fences cleanly', () => {
  const input = '### Update\nVisit [Guide](https://sytra.dev) or run:\n```ts\nconsole.log(1)\n```\nDone!'
  const out = cleanSpeechText(input)
  assert.doesNotMatch(out, /```|https:\/\//)
  assert.match(out, /Code block shown on screen\./)
  assert.match(out, /Guide/)
  assert.match(out, /Done!/)
})

test('extractReadySentences streams finished sentences while holding incomplete tail or open code blocks', () => {
  const step1 = extractReadySentences('Checking GPU now. Memory usage looks ', false)
  assert.deepEqual(step1.sentences, ['Checking GPU now.'])
  assert.equal(step1.remainder, 'Memory usage looks ')

  const step2 = extractReadySentences(step1.remainder + 'healthy and ```py\ncode', false)
  assert.deepEqual(step2.sentences, [])

  const flushed = extractReadySentences('healthy and cool.', true)
  assert.deepEqual(flushed.sentences, ['healthy and cool.'])
  assert.equal(flushed.remainder, '')
})

test('detectSpeechLangTag identifies Brazilian Portuguese vs English', () => {
  assert.equal(detectSpeechLangTag('Olá, você pode verificar o relatório agora?'), 'pt-BR')
  assert.equal(detectSpeechLangTag('Checking GPU latency and model status now.'), 'en-US')
})

test('encodePcm16WavBase64 creates valid RIFF/WAVE header after downsampling', () => {
  const samples48k = new Float32Array(480)
  for (let i = 0; i < samples48k.length; i += 1) {
    samples48k[i] = Math.sin(i * 0.1) * 0.4
  }
  const down = downsampleFloat32(samples48k, 48000, 16000)
  assert.equal(down.length, 160)
  assert.ok(computeRmsLevel(down) > 0.1)

  const b64 = encodePcm16WavBase64(down, 16000)
  const raw = Buffer.from(b64, 'base64')
  assert.equal(raw.subarray(0, 4).toString('ascii'), 'RIFF')
  assert.equal(raw.subarray(8, 12).toString('ascii'), 'WAVE')
  assert.equal(raw.readUInt32LE(24), 16000)
  assert.equal(raw.readUInt16LE(34), 16)
})

test('buildProactiveBriefingPrompt embeds hardware, running models, agent objective, and recent memory', () => {
  const prompt = buildProactiveBriefingPrompt({
    agent: {
      id: 'lil-sytra',
      name: 'Lil Sytra',
      role: 'Watch system & workspace',
      system_prompt: '',
      avatar_kind: 'pochi',
      skill_ids: [],
      computer_mode: 'sandbox',
      routine_enabled: true,
      routine_interval_mins: 5,
      routine_prompt: 'Summarize active hardware and workspace changes',
    },
    hw: { backend: 'cuda', vram_mb: 12288, ram_mb: 16384 },
    contextPct: 24,
    webSearchEnabled: true,
  })
  assert.match(prompt, /Lil Sytra/)
  assert.match(prompt, /VRAM=12288MB/)
  assert.match(prompt, /Summarize active hardware/)
  assert.match(prompt, /Web Search is ON/)
})
