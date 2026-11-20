//! Managed local voice engine runtime (Kokoro-82M ONNX TTS + Faster-Whisper STT).

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::env_provisioner::EnvProvisioner;
use crate::process::{apply_desktop_priority, apply_hidden_window, kill_process_tree};
use crate::settings::AppSettings;
use crate::xet_safety::apply_xet_safety;

const KOKORO_MODEL_INT8_FILE: &str = "kokoro-v1.0.int8.onnx";
const KOKORO_MODEL_FP32_FILE: &str = "kokoro-v1.0.onnx";
const KOKORO_VOICES_FILE: &str = "voices-v1.0.bin";

const KOKORO_MODEL_MIRRORS: &[(&str, &str)] = &[
    (
        KOKORO_MODEL_INT8_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    ),
    (
        KOKORO_MODEL_FP32_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    ),
    (
        KOKORO_MODEL_FP32_FILE,
        "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/kokoro-v1.0.onnx",
    ),
];

const KOKORO_VOICES_MIRRORS: &[(&str, &str)] = &[
    (
        KOKORO_VOICES_FILE,
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ),
    (
        KOKORO_VOICES_FILE,
        "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/voices-v1.0.bin",
    ),
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceOption {
    pub id: String,
    pub label: String,
    pub lang: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceRuntimeStatus {
    pub env_ready: bool,
    pub kokoro_model_ready: bool,
    pub server_running: bool,
    pub installing: bool,
    pub stage: String,
    pub message: String,
    pub error: Option<String>,
    pub active_port: Option<u16>,
    pub model_dir: String,
    pub voices: Vec<VoiceOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceAudioResponse {
    pub audio_base64: String,
    pub mime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscriptResponse {
    pub text: String,
    pub language: String,
    pub duration_sec: f64,
}

#[derive(Debug, Clone, Default)]
struct SetupState {
    installing: bool,
    stage: String,
    message: String,
    error: Option<String>,
}

#[derive(Clone)]
pub struct VoiceService {
    workspace: PathBuf,
    active_pid: Arc<Mutex<Option<u32>>>,
    active_port: Arc<Mutex<Option<u16>>>,
    worker_lock: Arc<Mutex<()>>,
    setup_state: Arc<Mutex<SetupState>>,
}

impl Drop for VoiceService {
    fn drop(&mut self) {
        // Only the last surviving strong reference kills the child when AppState drops.
        if Arc::strong_count(&self.active_pid) == 1 {
            let _ = self.stop();
        }
    }
}

pub fn default_voice_options() -> Vec<VoiceOption> {
    vec![
        VoiceOption { id: "auto".into(), label: "Auto (EN / PT-BR)".into(), lang: "auto".into() },
        VoiceOption { id: "af_heart".into(), label: "Kokoro Heart (US F)".into(), lang: "en-us".into() },
        VoiceOption { id: "af_bella".into(), label: "Kokoro Bella (US F)".into(), lang: "en-us".into() },
        VoiceOption { id: "af_sky".into(), label: "Kokoro Sky (US F)".into(), lang: "en-us".into() },
        VoiceOption { id: "am_adam".into(), label: "Kokoro Adam (US M)".into(), lang: "en-us".into() },
        VoiceOption { id: "bf_emma".into(), label: "Kokoro Emma (UK F)".into(), lang: "en-gb".into() },
        VoiceOption { id: "bm_george".into(), label: "Kokoro George (UK M)".into(), lang: "en-gb".into() },
        VoiceOption { id: "pf_dora".into(), label: "Kokoro Dora (PT-BR F)".into(), lang: "pt-br".into() },
        VoiceOption { id: "pm_alex".into(), label: "Kokoro Alex (PT-BR M)".into(), lang: "pt-br".into() },
    ]
}

pub fn kokoro_model_dir(workspace: &Path) -> PathBuf {
    workspace.join("voice-models").join("kokoro-82m")
}

fn is_nonempty_file(path: &Path) -> bool {
    path.is_file() && fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
}

pub fn kokoro_model_path(model_dir: &Path) -> Option<PathBuf> {
    [KOKORO_MODEL_INT8_FILE, KOKORO_MODEL_FP32_FILE]
        .iter()
        .map(|name| model_dir.join(name))
        .find(|p| is_nonempty_file(p))
}

pub fn kokoro_files_ready(model_dir: &Path) -> bool {
    kokoro_model_path(model_dir).is_some() && is_nonempty_file(&model_dir.join(KOKORO_VOICES_FILE))
}

fn cleanup_stale_kokoro_cache(model_dir: &Path) {
    let cache_dir = model_dir.join(".cache");
    if cache_dir.exists() {
        let _ = fs::remove_dir_all(&cache_dir);
    }
    if let Ok(entries) = fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with(".tmp."))
                .unwrap_or(false)
            {
                let _ = fs::remove_file(&p);
            }
        }
    }
}

fn stream_url_to_file(
    client: &Client,
    url: &str,
    dest_path: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<(), String> {
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| format!("HTTP request failed for {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} from {url}", resp.status()));
    }
    let total = resp.content_length().filter(|&n| n > 0);
    let tmp_name = format!(
        ".tmp.{}",
        dest_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("asset")
    );
    let tmp_path = dest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(tmp_name);

    let result = (|| -> Result<(), String> {
        let mut file = File::create(&tmp_path)
            .map_err(|e| format!("Cannot create temporary download file: {e}"))?;
        let mut buf = [0u8; 65_536];
        let mut downloaded: u64 = 0;
        let mut last_emit = Instant::now();
        on_progress(0, total);
        loop {
            let n = resp
                .read(&mut buf)
                .map_err(|e| format!("Download read interrupted for {url}: {e}"))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| format!("Failed writing downloaded bytes: {e}"))?;
            downloaded += n as u64;
            if last_emit.elapsed() >= Duration::from_millis(120) {
                on_progress(downloaded, total);
                last_emit = Instant::now();
            }
        }
        file.flush()
            .map_err(|e| format!("Failed flushing downloaded file: {e}"))?;
        drop(file);
        if downloaded == 0 {
            return Err(format!("Empty response body from {url}"));
        }
        if let Some(expected) = total {
            if downloaded != expected {
                return Err(format!(
                    "Incomplete download from {url}: got {downloaded} of {expected} bytes"
                ));
            }
        }
        on_progress(downloaded, total);
        if dest_path.exists() {
            let _ = fs::remove_file(dest_path);
        }
        fs::rename(&tmp_path, dest_path)
            .map_err(|e| format!("Failed moving downloaded file into place: {e}"))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }
    result
}

fn download_with_mirrors(
    client: &Client,
    dest_dir: &Path,
    candidates: &[(&str, &str)],
    on_progress: &mut dyn FnMut(&str, u64, Option<u64>),
) -> Result<PathBuf, String> {
    fs::create_dir_all(dest_dir)
        .map_err(|e| format!("Cannot create Kokoro model directory: {e}"))?;
    let mut errors = Vec::new();
    for &(filename, url) in candidates {
        let dest_path = dest_dir.join(filename);
        let mut progress_cb = |done: u64, total: Option<u64>| on_progress(filename, done, total);
        match stream_url_to_file(client, url, &dest_path, &mut progress_cb) {
            Ok(()) => return Ok(dest_path),
            Err(err) => errors.push(err),
        }
    }
    Err(format!("All mirrors failed: {}", errors.join(" | ")))
}

fn format_download_progress(label: &str, done: u64, total: Option<u64>) -> String {
    let done_mb = done as f64 / 1_000_000.0;
    if let Some(tot) = total.filter(|&t| t > 0) {
        let tot_mb = (tot as f64 / 1_000_000.0).max(1.0);
        let pct = ((done.saturating_mul(100)) / tot).min(100);
        format!("Downloading {label} ({done_mb:.0} / {tot_mb:.0} MB · {pct}%)...")
    } else {
        format!("Downloading {label} ({done_mb:.1} MB)...")
    }
}

fn download_kokoro_weights(
    model_dir: &Path,
    mut on_status: impl FnMut(String),
) -> Result<(), String> {
    fs::create_dir_all(model_dir)
        .map_err(|e| format!("Cannot create Kokoro model directory: {e}"))?;
    cleanup_stale_kokoro_cache(model_dir);

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(300))
        .user_agent("sytra-studio/1.3")
        .build()
        .map_err(|e| format!("Cannot initialize HTTP download client: {e}"))?;

    if !is_nonempty_file(&model_dir.join(KOKORO_VOICES_FILE)) {
        let mut cb = |_file: &str, done: u64, total: Option<u64>| {
            on_status(format_download_progress("Kokoro-82M voice bank", done, total));
        };
        download_with_mirrors(&client, model_dir, KOKORO_VOICES_MIRRORS, &mut cb)?;
    }

    if kokoro_model_path(model_dir).is_none() {
        let mut cb = |_file: &str, done: u64, total: Option<u64>| {
            on_status(format_download_progress("Kokoro-82M model", done, total));
        };
        download_with_mirrors(&client, model_dir, KOKORO_MODEL_MIRRORS, &mut cb)?;
    }

    if !kokoro_files_ready(model_dir) {
        return Err("Kokoro-82M model files were missing after download.".into());
    }
    Ok(())
}

pub fn sanitize_voice_id(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.len() > 40
        || !trimmed.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        "auto".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn encode_base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(((data.len() + 2) / 3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = if chunk.len() > 1 { chunk[1] as usize } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as usize } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) & 63] as char);
        out.push(TABLE[(n >> 12) & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n & 63] as char } else { '=' });
    }
    out
}

fn free_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Cannot allocate local voice server port: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Cannot inspect local voice server port: {e}"))?
        .port();
    drop(listener);
    Ok(port)
}

impl VoiceService {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            active_pid: Arc::new(Mutex::new(None)),
            active_port: Arc::new(Mutex::new(None)),
            worker_lock: Arc::new(Mutex::new(())),
            setup_state: Arc::new(Mutex::new(SetupState {
                installing: false,
                stage: "idle".into(),
                message: String::new(),
                error: None,
            })),
        }
    }

    pub fn status(&self) -> VoiceRuntimeStatus {
        let prov = EnvProvisioner::new(&self.workspace);
        let env_ready = prov.is_voice_provisioned();
        let model_dir = kokoro_model_dir(&self.workspace);
        let kokoro_model_ready = kokoro_files_ready(&model_dir);
        let port = *self.active_port.lock().unwrap_or_else(|p| p.into_inner());
        let pid_active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner()).is_some();
        let server_running = pid_active
            && port
                .map(|p| TcpStream::connect(("127.0.0.1", p)).is_ok())
                .unwrap_or(false);
        let setup = self.setup_state.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let stage = if setup.installing {
            setup.stage
        } else if server_running && kokoro_model_ready {
            "ready".into()
        } else if env_ready && kokoro_model_ready {
            "installed".into()
        } else if setup.error.is_some() {
            "error".into()
        } else {
            "idle".into()
        };

        VoiceRuntimeStatus {
            env_ready,
            kokoro_model_ready,
            server_running,
            installing: setup.installing,
            stage,
            message: setup.message,
            error: setup.error,
            active_port: if server_running { port } else { None },
            model_dir: model_dir.display().to_string(),
            voices: default_voice_options(),
        }
    }

    pub fn setup_kokoro_async(&self) -> VoiceRuntimeStatus {
        {
            let mut setup = self.setup_state.lock().unwrap_or_else(|p| p.into_inner());
            if setup.installing {
                drop(setup);
                return self.status();
            }
            setup.installing = true;
            setup.stage = "installing_env".into();
            setup.message = "Provisioning Kokoro-82M & Whisper voice runtime...".into();
            setup.error = None;
        }

        let svc = self.clone();
        thread::spawn(move || {
            let res = (|| -> Result<(), String> {
                let prov = EnvProvisioner::new(&svc.workspace);
                let _python = prov
                    .ensure_voice()
                    .map_err(|e| format!("Failed to provision voice environment: {e}"))?;

                let model_dir = kokoro_model_dir(&svc.workspace);
                if !kokoro_files_ready(&model_dir) {
                    {
                        let mut setup = svc.setup_state.lock().unwrap_or_else(|p| p.into_inner());
                        setup.stage = "downloading_model".into();
                        setup.message = "Downloading Kokoro-82M (~115 MB)...".into();
                    }
                    download_kokoro_weights(&model_dir, |msg| {
                        let mut setup = svc.setup_state.lock().unwrap_or_else(|p| p.into_inner());
                        setup.stage = "downloading_model".into();
                        setup.message = msg;
                    })?;
                }

                {
                    let mut setup = svc.setup_state.lock().unwrap_or_else(|p| p.into_inner());
                    setup.stage = "starting".into();
                    setup.message = "Starting Kokoro-82M local audio engine...".into();
                }

                svc.ensure_running_inner()?;
                Ok(())
            })();

            let mut setup = svc.setup_state.lock().unwrap_or_else(|p| p.into_inner());
            setup.installing = false;
            match res {
                Ok(()) => {
                    setup.stage = "ready".into();
                    setup.message = "Kokoro-82M ready".into();
                    setup.error = None;
                }
                Err(err) => {
                    setup.stage = "error".into();
                    setup.message = String::new();
                    setup.error = Some(err);
                }
            }
        });

        self.status()
    }

    pub fn synthesize(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
        lang: Option<&str>,
    ) -> Result<VoiceAudioResponse, String> {
        if text.trim().is_empty() {
            return Err("Text is empty.".into());
        }
        let port = self.ensure_running_inner()?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(45))
            .build()
            .map_err(|e| format!("Voice HTTP client error: {e}"))?;

        let safe_voice = sanitize_voice_id(voice.unwrap_or("auto"));
        let safe_speed = speed.unwrap_or(1.0).clamp(0.65, 1.6);
        let safe_lang = lang.unwrap_or("auto").trim();

        let resp = client
            .post(format!("http://127.0.0.1:{port}/tts"))
            .json(&json!({
                "text": text,
                "voice": safe_voice,
                "speed": safe_speed,
                "lang": safe_lang,
            }))
            .send()
            .map_err(|e| format!("Kokoro TTS request failed: {e}"))?;

        if !resp.status().is_success() {
            let err_text = resp.text().unwrap_or_default();
            return Err(format!("Kokoro TTS error: {err_text}"));
        }
        let wav = resp
            .bytes()
            .map_err(|e| format!("Failed to read Kokoro WAV response: {e}"))?;
        Ok(VoiceAudioResponse {
            audio_base64: encode_base64(&wav),
            mime: "audio/wav".into(),
        })
    }

    pub fn transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<VoiceTranscriptResponse, String> {
        if audio_base64.trim().is_empty() {
            return Ok(VoiceTranscriptResponse {
                text: String::new(),
                language: String::new(),
                duration_sec: 0.0,
            });
        }
        let port = self.ensure_running_inner()?;
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| format!("Voice HTTP client error: {e}"))?;

        let resp = client
            .post(format!("http://127.0.0.1:{port}/stt"))
            .json(&json!({
                "audio_base64": audio_base64,
                "language": language.unwrap_or("auto"),
            }))
            .send()
            .map_err(|e| format!("Local speech recognition request failed: {e}"))?;

        if !resp.status().is_success() {
            let err_text = resp.text().unwrap_or_default();
            return Err(format!("Speech recognition error: {err_text}"));
        }
        resp.json::<VoiceTranscriptResponse>()
            .map_err(|e| format!("Invalid speech transcript response: {e}"))
    }

    pub fn stop(&self) -> Result<(), String> {
        if let Some(port) = self.active_port.lock().unwrap_or_else(|p| p.into_inner()).take() {
            if let Ok(client) = reqwest::blocking::Client::builder()
                .no_proxy()
                .timeout(Duration::from_millis(600))
                .build()
            {
                let _ = client.post(format!("http://127.0.0.1:{port}/shutdown")).send();
            }
        }
        if let Some(pid) = self.active_pid.lock().unwrap_or_else(|p| p.into_inner()).take() {
            kill_process_tree(pid);
        }
        Ok(())
    }

    fn configure_python_cmd(&self, cmd: &mut Command) {
        let runner_dir = self.workspace.join("runner");
        apply_xet_safety(cmd);
        apply_desktop_priority(cmd);
        apply_hidden_window(cmd);
        cmd.current_dir(&self.workspace)
            .env("PYTHONPATH", &runner_dir)
            .env("HF_HUB_DISABLE_XET", "1")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .stdin(Stdio::null());
        let settings = AppSettings::load(&self.workspace);
        if let Some(hf_dir) = settings.hf_cache_dir.as_deref() {
            if !hf_dir.as_os_str().is_empty() {
                cmd.env("HF_HOME", hf_dir);
            }
        }
    }

    fn ensure_running_inner(&self) -> Result<u16, String> {
        let _guard = self
            .worker_lock
            .lock()
            .map_err(|_| "Voice runtime lock was poisoned".to_string())?;

        if let Some(port) = *self.active_port.lock().unwrap_or_else(|p| p.into_inner()) {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Ok(port);
            }
        }

        let prov = EnvProvisioner::new(&self.workspace);
        let python = prov
            .ensure_voice()
            .map_err(|e| format!("Voice environment unavailable: {e}"))?;
        let port = free_loopback_port()?;
        let model_dir = kokoro_model_dir(&self.workspace);

        let mut cmd = Command::new(&python);
        self.configure_python_cmd(&mut cmd);
        cmd.stdout(Stdio::null())
            .stderr(Stdio::null())
            .args([
                "-m",
                "sytra_runner.voice_engine",
                "--port",
                &port.to_string(),
                "--model-dir",
            ])
            .arg(&model_dir);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to start Kokoro/Whisper voice worker: {e}"))?;
        let pid = child.id();
        *self.active_pid.lock().unwrap_or_else(|p| p.into_inner()) = Some(pid);
        *self.active_port.lock().unwrap_or_else(|p| p.into_inner()) = Some(port);

        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_millis(750))
            .build()
            .map_err(|e| format!("Cannot create voice health client: {e}"))?;

        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if let Ok(Some(status)) = child.try_wait() {
                *self.active_pid.lock().unwrap_or_else(|p| p.into_inner()) = None;
                *self.active_port.lock().unwrap_or_else(|p| p.into_inner()) = None;
                return Err(format!("Voice worker exited before becoming ready: {status}"));
            }
            if client
                .get(format!("http://127.0.0.1:{port}/health"))
                .send()
                .map(|r| r.status().is_success())
                .unwrap_or(false)
            {
                let pid_ref = self.active_pid.clone();
                let port_ref = self.active_port.clone();
                thread::spawn(move || {
                    let _ = child.wait();
                    let mut p = pid_ref.lock().unwrap_or_else(|e| e.into_inner());
                    if *p == Some(pid) {
                        *p = None;
                        *port_ref.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    }
                });
                return Ok(port);
            }
            thread::sleep(Duration::from_millis(150));
        }

        let _ = self.stop();
        Err("Voice engine did not become ready within 30 seconds.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_status_starts_idle_when_unprovisioned() {
        let tmp = std::env::temp_dir().join(format!("sytra-voice-status-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let svc = VoiceService::new(&tmp);
        let st = svc.status();
        assert!(!st.env_ready);
        assert!(!st.kokoro_model_ready);
        assert!(!st.server_running);
        assert_eq!(st.stage, "idle");
        assert!(st.voices.iter().any(|v| v.id == "af_heart"));
        assert!(st.voices.iter().any(|v| v.id == "pf_dora"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn kokoro_files_ready_accepts_int8_or_fp32_with_voices() {
        let tmp = std::env::temp_dir().join(format!("sytra-voice-files-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        assert!(!kokoro_files_ready(&tmp));

        // Empty 0-byte incomplete files must not count as ready.
        fs::write(tmp.join(KOKORO_MODEL_INT8_FILE), b"").unwrap();
        fs::write(tmp.join(KOKORO_VOICES_FILE), b"voices").unwrap();
        assert!(!kokoro_files_ready(&tmp));

        fs::write(tmp.join(KOKORO_MODEL_INT8_FILE), b"int8-onnx").unwrap();
        assert!(kokoro_files_ready(&tmp));
        assert_eq!(
            kokoro_model_path(&tmp).unwrap().file_name().unwrap(),
            KOKORO_MODEL_INT8_FILE
        );

        let _ = fs::remove_file(tmp.join(KOKORO_MODEL_INT8_FILE));
        fs::write(tmp.join(KOKORO_MODEL_FP32_FILE), b"fp32-onnx").unwrap();
        assert!(kokoro_files_ready(&tmp));
        assert_eq!(
            kokoro_model_path(&tmp).unwrap().file_name().unwrap(),
            KOKORO_MODEL_FP32_FILE
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn download_with_mirrors_falls_back_and_streams_atomically() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server_handle = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut req_buf = [0u8; 1024];
                let n = stream.read(&mut req_buf).unwrap();
                let req = String::from_utf8_lossy(&req_buf[..n]);
                if req.contains("GET /fail ") {
                    let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                } else {
                    let body = b"SYTRA-KOKORO-ONNX-STREAM";
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(body);
                }
            }
        });

        let tmp = std::env::temp_dir().join(format!("sytra-voice-dl-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let client = Client::builder().no_proxy().build().unwrap();
        let fail_url = format!("http://127.0.0.1:{port}/fail");
        let ok_url = format!("http://127.0.0.1:{port}/kokoro-v1.0.int8.onnx");
        let mirrors = [
            (KOKORO_MODEL_INT8_FILE, fail_url.as_str()),
            (KOKORO_MODEL_INT8_FILE, ok_url.as_str()),
        ];

        let mut seen_bytes = 0u64;
        let path = download_with_mirrors(&client, &tmp, &mirrors, &mut |_name, done, _tot| {
            seen_bytes = seen_bytes.max(done);
        })
        .unwrap();

        server_handle.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"SYTRA-KOKORO-ONNX-STREAM");
        assert_eq!(seen_bytes, b"SYTRA-KOKORO-ONNX-STREAM".len() as u64);
        assert!(!tmp.join(".tmp.kokoro-v1.0.int8.onnx").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn sanitize_voice_id_rejects_injection_attempts() {
        assert_eq!(sanitize_voice_id("af_bella"), "af_bella");
        assert_eq!(sanitize_voice_id("../bad"), "auto");
        assert_eq!(sanitize_voice_id("bad voice!"), "auto");
    }

    #[test]
    fn encode_base64_matches_rfc4648() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn stop_is_ok_when_idle() {
        let svc = VoiceService::new(Path::new("."));
        assert!(svc.stop().is_ok());
    }
}
