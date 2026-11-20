import { api } from '../../api'
import { t } from '../../i18n.svelte'
import type { VoiceEngineMode, VoiceRuntimeStatus } from '../../types'
import {
  computeRmsLevel,
  detectSpeechLangTag,
  downsampleFloat32,
  encodePcm16WavBase64,
  extractReadySentences,
} from './voiceHelpers'

const VOICE_PREFS_KEY = 'sytra-chat-voice-prefs-v1'

export interface VoicePrefs {
  engine: VoiceEngineMode
  voiceId: string
  speed: number
  autoSpeakReplies: boolean
  proactiveEnabled: boolean
  proactiveIntervalMins: number
}

const DEFAULT_PREFS: VoicePrefs = {
  engine: 'browser',
  voiceId: 'auto',
  speed: 1.0,
  autoSpeakReplies: false,
  proactiveEnabled: false,
  proactiveIntervalMins: 5,
}

function loadPrefs(): VoicePrefs {
  try {
    const raw = localStorage.getItem(VOICE_PREFS_KEY)
    if (!raw) return { ...DEFAULT_PREFS }
    const parsed = JSON.parse(raw) as Partial<VoicePrefs>
    return {
      engine: parsed.engine === 'kokoro' ? 'kokoro' : 'browser',
      voiceId: typeof parsed.voiceId === 'string' && parsed.voiceId.trim() ? parsed.voiceId : 'auto',
      speed: typeof parsed.speed === 'number' && parsed.speed >= 0.65 && parsed.speed <= 1.6 ? parsed.speed : 1.0,
      autoSpeakReplies: Boolean(parsed.autoSpeakReplies),
      proactiveEnabled: Boolean(parsed.proactiveEnabled),
      proactiveIntervalMins:
        typeof parsed.proactiveIntervalMins === 'number' && parsed.proactiveIntervalMins >= 1
          ? Math.min(120, Math.round(parsed.proactiveIntervalMins))
          : 5,
    }
  } catch {
    return { ...DEFAULT_PREFS }
  }
}

export const voicePrefs = $state<VoicePrefs>(loadPrefs())

export function saveVoicePrefs(): void {
  try {
    localStorage.setItem(VOICE_PREFS_KEY, JSON.stringify(voicePrefs))
  } catch { /* ignore storage quota errors */ }
}

export const voiceRuntime = $state({
  micState: 'idle' as 'idle' | 'listening' | 'recording' | 'transcribing' | 'error',
  micLevel: 0,
  liveMode: false,
  speaking: false,
  interimText: '',
  statusNote: '',
  kokoroStatus: null as VoiceRuntimeStatus | null,
})

interface SpeechRecognitionAlternativeLike {
  transcript: string
}
interface SpeechRecognitionResultLike {
  isFinal: boolean
  0: SpeechRecognitionAlternativeLike
}
interface SpeechRecognitionEventLike {
  resultIndex: number
  results: ArrayLike<SpeechRecognitionResultLike>
}
interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null
  onerror: ((ev: { error?: string }) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
  abort: () => void
}

let recognition: SpeechRecognitionLike | null = null
let audioCtx: AudioContext | null = null
let mediaStream: MediaStream | null = null
let scriptProcessor: ScriptProcessorNode | null = null
let capturedChunks: Float32Array[] = []
let recordedSampleRate = 16000
let speechDetected = false
let lastSpeechAt = 0
let transcriptHandler: ((text: string, autoSend: boolean) => void) | null = null
let isStoppingCapture = false
let setupPollTimer: ReturnType<typeof setInterval> | null = null

let speechQueue: string[] = []
let speechDrainActive = false
let activeAudio: HTMLAudioElement | null = null
let streamTextBuffer = ''
let streamQueuedLen = 0
let activeSpeechEpoch = 0

export async function refreshKokoroStatus(): Promise<VoiceRuntimeStatus | null> {
  try {
    const st = await api.voiceStatus()
    voiceRuntime.kokoroStatus = st
    return st
  } catch {
    return null
  }
}

export async function startKokoroSetup(): Promise<void> {
  try {
    voiceRuntime.statusNote = t('chat.installingVoice')
    const st = await api.setupKokoroVoice()
    voiceRuntime.kokoroStatus = st
    if (setupPollTimer) { clearInterval(setupPollTimer); setupPollTimer = null }
    if (st.error) {
      voiceRuntime.statusNote = st.error
      return
    }
    if (!st.installing && (st.stage === 'ready' || st.kokoro_model_ready)) {
      voiceRuntime.statusNote = t('chat.voiceReady')
      voicePrefs.engine = 'kokoro'
      saveVoicePrefs()
      return
    }
    setupPollTimer = setInterval(async () => {
      const latest = await refreshKokoroStatus()
      if (!latest) return
      if (latest.error) {
        voiceRuntime.statusNote = latest.error
        if (setupPollTimer) { clearInterval(setupPollTimer); setupPollTimer = null }
      } else if (!latest.installing && (latest.stage === 'ready' || latest.kokoro_model_ready)) {
        voiceRuntime.statusNote = t('chat.voiceReady')
        voicePrefs.engine = 'kokoro'
        saveVoicePrefs()
        if (setupPollTimer) { clearInterval(setupPollTimer); setupPollTimer = null }
      } else if (latest.message) {
        voiceRuntime.statusNote = latest.message
      }
    }, 500)
  } catch (err) {
    voiceRuntime.statusNote = String(err)
  }
}

function getSpeechRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  if (typeof window === 'undefined') return null
  const w = window as unknown as Record<string, unknown>
  return (w.SpeechRecognition || w.webkitSpeechRecognition || null) as (new () => SpeechRecognitionLike) | null
}

function releaseMediaNodes(): void {
  if (scriptProcessor) {
    try { scriptProcessor.disconnect() } catch { /* ignore */ }
    scriptProcessor.onaudioprocess = null
    scriptProcessor = null
  }
  if (audioCtx) {
    try { void audioCtx.close() } catch { /* ignore */ }
    audioCtx = null
  }
  if (mediaStream) {
    for (const tr of mediaStream.getTracks()) {
      try { tr.stop() } catch { /* ignore */ }
    }
    mediaStream = null
  }
  voiceRuntime.micLevel = 0
}

async function flushWavToWhisper(autoSend: boolean): Promise<void> {
  const chunks = capturedChunks
  capturedChunks = []
  if (!chunks.length) {
    voiceRuntime.micState = voiceRuntime.liveMode && !voiceRuntime.speaking ? 'listening' : 'idle'
    return
  }

  let total = 0
  for (const c of chunks) total += c.length
  const merged = new Float32Array(total)
  let offset = 0
  for (const c of chunks) {
    merged.set(c, offset)
    offset += c.length
  }

  const downsampled = downsampleFloat32(merged, recordedSampleRate, 16000)
  if (computeRmsLevel(downsampled) < 0.004) {
    voiceRuntime.micState = voiceRuntime.liveMode && !voiceRuntime.speaking ? 'listening' : 'idle'
    return
  }

  voiceRuntime.micState = 'transcribing'
  voiceRuntime.statusNote = t('chat.transcribing')
  try {
    const b64 = encodePcm16WavBase64(downsampled, 16000)
    const res = await api.voiceTranscribe(b64, 'auto')
    const txt = (res?.text || '').trim()
    voiceRuntime.statusNote = txt ? t('chat.heard', { language: res.language || t('chat.local'), text: txt }) : t('chat.noSpeech')
    if (txt && transcriptHandler) {
      transcriptHandler(txt, autoSend)
    }
  } catch (err) {
    voiceRuntime.micState = 'error'
    voiceRuntime.statusNote = t('chat.whisperError', { error: String(err) })
    return
  }

  if (voiceRuntime.liveMode && !voiceRuntime.speaking) {
    await startWavRecording(true)
  } else {
    voiceRuntime.micState = 'idle'
  }
}

async function startWavRecording(autoSend: boolean): Promise<void> {
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    voiceRuntime.micState = 'error'
    voiceRuntime.liveMode = false
    voiceRuntime.statusNote = t('chat.micUnavailable')
    return
  }
  releaseMediaNodes()
  capturedChunks = []
  speechDetected = false
  lastSpeechAt = performance.now()

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    audioCtx = new AudioCtx()
    recordedSampleRate = audioCtx.sampleRate || 16000
    const src = audioCtx.createMediaStreamSource(mediaStream)
    scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1)
    voiceRuntime.micState = autoSend ? 'listening' : 'recording'
    voiceRuntime.statusNote = autoSend ? t('chat.listeningWhisper') : t('chat.recordingMic')

    scriptProcessor.onaudioprocess = (ev) => {
      if (isStoppingCapture || (voiceRuntime.liveMode && voiceRuntime.speaking)) return
      const input = ev.inputBuffer.getChannelData(0)
      const copy = new Float32Array(input.length)
      copy.set(input)
      capturedChunks.push(copy)

      const rms = computeRmsLevel(copy)
      voiceRuntime.micLevel = Math.min(100, Math.round(rms * 420))
      const now = performance.now()
      if (rms > 0.015) {
        speechDetected = true
        lastSpeechAt = now
        if (voiceRuntime.micState === 'listening') voiceRuntime.micState = 'recording'
      } else if (autoSend && speechDetected && now - lastSpeechAt > 1150) {
        releaseMediaNodes()
        void flushWavToWhisper(true)
      }
    }

    src.connect(scriptProcessor)
    scriptProcessor.connect(audioCtx.destination)
  } catch (err) {
    releaseMediaNodes()
    voiceRuntime.micState = 'error'
    voiceRuntime.liveMode = false
    voiceRuntime.statusNote = t('chat.micPermission', { error: String(err) })
  }
}

function startWebSpeech(autoSend: boolean, Ctor: new () => SpeechRecognitionLike): void {
  try {
    const rec = new Ctor()
    recognition = rec
    rec.continuous = autoSend
    rec.interimResults = true
    rec.lang = typeof navigator !== 'undefined' ? (navigator.language || 'en-US') : 'en-US'
    voiceRuntime.micState = 'listening'
    voiceRuntime.interimText = ''
    voiceRuntime.statusNote = autoSend ? t('chat.listeningHandsFree') : t('chat.speakNow')

    rec.onresult = (ev) => {
      if (voiceRuntime.liveMode && voiceRuntime.speaking) return
      let interim = ''
      let finals = ''
      for (let i = ev.resultIndex; i < ev.results.length; i += 1) {
        const res = ev.results[i]
        const text = res?.[0]?.transcript || ''
        if (res?.isFinal) finals += `${text} `
        else interim += text
      }
      voiceRuntime.interimText = interim.trim()
      if (finals.trim() && transcriptHandler) {
        voiceRuntime.interimText = ''
        transcriptHandler(finals.trim(), autoSend)
        if (!autoSend) {
          stopMicCapture(false)
        }
      }
    }

    rec.onerror = (ev) => {
      const code = ev?.error || ''
      recognition = null
      if (code === 'aborted') return
      // Desktop WebView2 offline often emits 'network' or 'service-not-allowed' for cloud Web Speech API.
      // Automatically switch cleanly to offline Web Audio + Whisper local transcription.
      if (['network', 'service-not-allowed', 'audio-capture'].includes(code)) {
        void startWavRecording(autoSend)
      } else {
        voiceRuntime.micState = 'error'
        voiceRuntime.liveMode = false
        voiceRuntime.statusNote = t('chat.recognitionError', { error: code || t('chat.unknown') })
      }
    }

    rec.onend = () => {
      recognition = null
      if (voiceRuntime.liveMode && !voiceRuntime.speaking && !isStoppingCapture) {
        startWebSpeech(true, Ctor)
      } else if (voiceRuntime.micState === 'listening') {
        voiceRuntime.micState = 'idle'
      }
    }

    rec.start()
  } catch {
    recognition = null
    void startWavRecording(autoSend)
  }
}

export function stopMicCapture(finalizeWav = true): void {
  isStoppingCapture = true
  if (recognition) {
    try { recognition.abort() } catch { /* ignore */ }
    recognition = null
  }
  const hadWav = capturedChunks.length > 0 && scriptProcessor !== null
  releaseMediaNodes()
  voiceRuntime.interimText = ''
  if (hadWav && finalizeWav) {
    void flushWavToWhisper(false)
  } else if (voiceRuntime.micState !== 'transcribing') {
    voiceRuntime.micState = 'idle'
  }
  isStoppingCapture = false
}

export function toggleDictation(onTranscript: (text: string, autoSend: boolean) => void): void {
  transcriptHandler = onTranscript
  if (voiceRuntime.liveMode) {
    stopLiveVoiceChat()
    return
  }
  if (voiceRuntime.micState === 'listening' || voiceRuntime.micState === 'recording') {
    stopMicCapture(true)
    return
  }
  const Ctor = getSpeechRecognitionCtor()
  if (Ctor) {
    startWebSpeech(false, Ctor)
  } else {
    void startWavRecording(false)
  }
}

export function startLiveVoiceChat(onTranscript: (text: string, autoSend: boolean) => void): void {
  transcriptHandler = onTranscript
  voiceRuntime.liveMode = true
  voicePrefs.autoSpeakReplies = true
  saveVoicePrefs()
  if (voiceRuntime.speaking) return
  const Ctor = getSpeechRecognitionCtor()
  if (Ctor) {
    startWebSpeech(true, Ctor)
  } else {
    void startWavRecording(true)
  }
}

export function stopLiveVoiceChat(): void {
  voiceRuntime.liveMode = false
  stopMicCapture(false)
  interruptSpeech()
  voiceRuntime.statusNote = t('chat.liveStopped')
}

export function resetStreamingSpeechTracker(): void {
  streamTextBuffer = ''
  streamQueuedLen = 0
}

export function pushStreamingSpeechText(fullContent: string): void {
  if (!voicePrefs.autoSpeakReplies && !voiceRuntime.liveMode) return
  const delta = fullContent.slice(streamQueuedLen)
  if (!delta) return
  streamQueuedLen = fullContent.length
  streamTextBuffer += delta
  const { sentences, remainder } = extractReadySentences(streamTextBuffer, false)
  streamTextBuffer = remainder
  for (const s of sentences) {
    enqueueSpeechSentence(s)
  }
}

export function flushStreamingSpeechText(): void {
  if (!voicePrefs.autoSpeakReplies && !voiceRuntime.liveMode) {
    streamTextBuffer = ''
    streamQueuedLen = 0
    return
  }
  const { sentences } = extractReadySentences(streamTextBuffer, true)
  streamTextBuffer = ''
  streamQueuedLen = 0
  for (const s of sentences) {
    enqueueSpeechSentence(s)
  }
}

export function enqueueSpeechSentence(text: string): void {
  const trimmed = text.trim()
  if (!trimmed) return
  speechQueue.push(trimmed)
  void drainSpeechQueue()
}

export function speakFullText(text: string): void {
  interruptSpeech()
  const { sentences } = extractReadySentences(text, true)
  for (const s of sentences) {
    speechQueue.push(s)
  }
  void drainSpeechQueue()
}

export function interruptSpeech(): void {
  activeSpeechEpoch += 1
  speechQueue = []
  if (activeAudio) {
    try {
      activeAudio.pause()
      activeAudio.src = ''
    } catch { /* ignore */ }
    activeAudio = null
  }
  if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
    try { window.speechSynthesis.cancel() } catch { /* ignore */ }
  }
  voiceRuntime.speaking = false
}

function playWavBase64(b64: string, mime: string, epoch: number): Promise<void> {
  return new Promise((resolve, reject) => {
    if (epoch !== activeSpeechEpoch || typeof Audio === 'undefined') {
      resolve()
      return
    }
    const audio = new Audio(`data:${mime};base64,${b64}`)
    activeAudio = audio
    audio.onended = () => {
      if (activeAudio === audio) activeAudio = null
      resolve()
    }
    audio.onerror = () => {
      if (activeAudio === audio) activeAudio = null
      reject(new Error('Audio element playback failed'))
    }
    const p = audio.play()
    if (p && typeof p.catch === 'function') {
      p.catch((err) => {
        if (activeAudio === audio) activeAudio = null
        reject(err)
      })
    }
  })
}

function speakWithBrowser(text: string, epoch: number): Promise<void> {
  return new Promise((resolve) => {
    if (epoch !== activeSpeechEpoch || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      resolve()
      return
    }
    try {
      const utt = new SpeechSynthesisUtterance(text)
      utt.rate = voicePrefs.speed
      utt.lang = detectSpeechLangTag(text)
      utt.onend = () => resolve()
      utt.onerror = () => resolve()
      window.speechSynthesis.speak(utt)
    } catch {
      resolve()
    }
  })
}

async function drainSpeechQueue(): Promise<void> {
  if (speechDrainActive) return
  speechDrainActive = true
  const epoch = activeSpeechEpoch

  // Pause active listening during playback so mic doesn't echo Lil Sytra
  if (voiceRuntime.liveMode) {
    stopMicCapture(false)
  }
  voiceRuntime.speaking = true

  try {
    while (speechQueue.length && epoch === activeSpeechEpoch) {
      const sentence = speechQueue.shift()!
      if (voicePrefs.engine === 'kokoro') {
        try {
          const audioRes = await api.voiceSynthesize(sentence, voicePrefs.voiceId, voicePrefs.speed, 'auto')
          if (epoch !== activeSpeechEpoch) break
          await playWavBase64(audioRes.audio_base64, audioRes.mime, epoch)
          continue
        } catch (err) {
          voiceRuntime.statusNote = t('chat.kokoroError', { error: String(err).slice(0, 60) })
        }
      }
      await speakWithBrowser(sentence, epoch)
    }
  } finally {
    speechDrainActive = false
    if (epoch === activeSpeechEpoch) {
      voiceRuntime.speaking = false
      if (voiceRuntime.liveMode && transcriptHandler) {
        const Ctor = getSpeechRecognitionCtor()
        setTimeout(() => {
          if (voiceRuntime.liveMode && !voiceRuntime.speaking && transcriptHandler) {
            if (Ctor) startWebSpeech(true, Ctor)
            else void startWavRecording(true)
          }
        }, 250)
      }
    }
  }
}
