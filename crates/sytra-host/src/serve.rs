use std::ffi::OsString;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::backend_resolver::BackendResolver;
use crate::process::kill_process_tree;
use crate::settings::AppSettings;
use crate::workspace::python_executable;

#[derive(Clone)]
pub struct ChatServer {
    workspace: PathBuf,
    active_pid: Arc<Mutex<Option<u32>>>,
    start_lock: Arc<Mutex<()>>,
    generation: Arc<AtomicU64>,
    mcp_token: Arc<String>,
}

impl ChatServer {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            active_pid: Arc::new(Mutex::new(None)),
            start_lock: Arc::new(Mutex::new(())),
            generation: Arc::new(AtomicU64::new(0)),
            mcp_token: Arc::new(format!("{}{}", uuid::Uuid::new_v4().simple(), uuid::Uuid::new_v4().simple())),
        }
    }

    pub fn stop(&self) -> Result<(), String> {
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(pid) = self.active_pid.lock().unwrap_or_else(|p| p.into_inner()).take() {
            kill_process_tree(pid);
        }
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.active_pid.lock().unwrap_or_else(|p| p.into_inner()).is_some()
    }

    pub fn mcp_token(&self) -> String { self.mcp_token.as_ref().clone() }

    pub fn start(
        &self,
        model_path: &str,
        context: Option<usize>,
        vram_limit: Option<usize>,
        cpu_kv_cache: Option<bool>,
    ) -> Result<(), String> {
        self.start_with_options(model_path, context, vram_limit, cpu_kv_cache, None)
    }

    pub fn start_with_options(
        &self,
        model_path: &str,
        context: Option<usize>,
        vram_limit: Option<usize>,
        cpu_kv_cache: Option<bool>,
        allow_unverified: Option<bool>,
    ) -> Result<(), String> {
        let _start_guard = self.start_lock.try_lock().map_err(|_| "A chat server is already starting".to_string())?;
        if model_path.trim().starts_with('-') {
            return Err("Invalid model_path parameter: cannot start with '-'".into());
        }
        self.stop()?;
        let generation = self.generation.load(Ordering::SeqCst);
        if TcpStream::connect(("127.0.0.1", 8080)).is_ok() {
            return Err(
                "Port 127.0.0.1:8080 is still occupied after stop. Close the stale llama.cpp, Ollama, or LM Studio process and retry."
                    .into(),
            );
        }

        let settings = AppSettings::load(&self.workspace);
        let model_settings = crate::model_settings::load(&self.workspace, Path::new(model_path))?;
        let detected_ram = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM; refusing to start a chat server without a memory envelope"
                .to_string()
        })?;
        let ctx_val = context.or(model_settings.context_length).unwrap_or(settings.default_context_window);
        let temperature = model_settings.temperature.unwrap_or(settings.default_temperature);
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err("Temperature must be between 0 and 2".into());
        }
        let detected_vram = BackendResolver::detect_system_vram_mb();
        let mut vram_val = vram_limit
            .or(model_settings.vram_limit_mb)
            .or_else(|| settings.vram_limit_mb.map(|v| v as usize))
            .or_else(|| detected_vram.map(|v| v as usize));
        if let (Some(val), Some(det)) = (vram_val, detected_vram) {
            vram_val = Some(val.min(det as usize));
        }
        let ram_val = settings.effective_main_memory_mb(detected_ram);
        let use_cpu_kv = cpu_kv_cache.or(model_settings.cpu_kv_cache).unwrap_or(settings.cpu_kv_cache);
        let kv_cache_quant = model_settings.kv_cache_quant.as_deref().unwrap_or(&settings.kv_cache_quant);
        let flash_attention = model_settings.flash_attention.unwrap_or(settings.enable_flash_attention);
        let idle_ttl = model_settings.idle_ttl_minutes.unwrap_or(30);
        let script = self
            .workspace
            .join("runner")
            .join("scripts")
            .join("serve_model.py");
        let python = python_executable(&self.workspace);

        let mut server_args: Vec<OsString> = vec![
            script.as_os_str().to_owned(),
            "--model".into(),
            model_path.into(),
            "--context".into(),
            ctx_val.to_string().into(),
            "--temperature".into(),
            temperature.to_string().into(),
            "--ram-limit".into(),
            ram_val.to_string().into(),
            "--kv-cache-quant".into(),
            kv_cache_quant.into(),
            "--project-root".into(),
            self.workspace.as_os_str().to_owned(),
            "--port".into(),
            "8080".into(),
            "--idle-ttl-seconds".into(),
            (idle_ttl * 60).to_string().into(),
            "--managed-proxy".into(),
        ];
        if let Some(vram) = vram_val {
            server_args.push("--vram-limit".into());
            server_args.push(vram.to_string().into());
        }
        if use_cpu_kv {
            server_args.push("--cpu-kv-cache".into());
        }
        if let Some(guardrail) = model_settings.vram_guardrail_mb {
            server_args.push("--vram-guardrail-mb".into());
            server_args.push(guardrail.to_string().into());
        }
        if let Some(draft) = &model_settings.draft_model_path {
            server_args.push("--draft-model-path".into());
            server_args.push(draft.into());
        }
        if !flash_attention {
            server_args.push("--no-flash-attention".into());
        }
        if allow_unverified
            .or(model_settings.allow_unverified)
            .unwrap_or(false)
        {
            server_args.push("--allow-unverified".into());
        }

        let mut preflight = std::process::Command::new(&python);
        crate::apply_xet_safety(&mut preflight);
        crate::apply_desktop_priority(&mut preflight);
        preflight.args(&server_args);
        preflight
            .args(["--dry-run", "--verify-engine"])
            .current_dir(&self.workspace);
        let preflight_output = preflight
            .output()
            .map_err(|e| format!("Failed to run model preflight: {e}"))?;
        if !preflight_output.status.success() {
            let stderr = String::from_utf8_lossy(&preflight_output.stderr);
            let stdout = String::from_utf8_lossy(&preflight_output.stdout);
            let detail = if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };
            return Err(format!("Model is not ready to serve: {detail}"));
        }

        let mut cmd = std::process::Command::new(&python);
        let client = reqwest::blocking::Client::builder().no_proxy().timeout(Duration::from_secs(1))
            .build().map_err(|e| format!("Cannot create health client: {e}"))?;
        crate::apply_xet_safety(&mut cmd);
        crate::apply_desktop_priority(&mut cmd);
        cmd.args(&server_args).arg("--launch").env("SYTRA_MCP_SESSION_TOKEN", self.mcp_token.as_str()).current_dir(&self.workspace);
        let mut active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err("Chat server startup was cancelled".into());
        }
        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn chat server: {e}"))?;
        let pid = child.id();
        *active = Some(pid);
        drop(active);
        let deadline = Instant::now() + Duration::from_secs(360);
        while Instant::now() < deadline {
            if self.generation.load(Ordering::SeqCst) != generation {
                let _ = child.wait();
                return Err("Chat server startup was cancelled".into());
            }
            let exit_status = match child.try_wait() {
                Ok(status) => status,
                Err(error) => {
                    kill_process_tree(pid);
                    let _ = child.wait();
                    let mut active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
                    if *active == Some(pid) { *active = None; }
                    return Err(format!("Cannot inspect chat server: {error}"));
                }
            };
            if let Some(status) = exit_status {
                let mut active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
                if *active == Some(pid) { *active = None; }
                return Err(format!("Chat server exited before becoming ready: {status}"));
            }
            if health_ready(&client, "http://127.0.0.1:8080/health") {
                let pid_ref = self.active_pid.clone();
                thread::spawn(move || {
                    let _ = child.wait();
                    let mut active = pid_ref.lock().unwrap_or_else(|p| p.into_inner());
                    if *active == Some(pid) { *active = None; }
                });
                return Ok(());
            }
            thread::sleep(Duration::from_millis(250));
        }
        kill_process_tree(pid);
        let _ = child.wait();
        let mut active = self.active_pid.lock().unwrap_or_else(|p| p.into_inner());
        if *active == Some(pid) { *active = None; }
        Err("Chat server did not become ready within 360 seconds".into())
    }

    pub fn shutdown(&self) {
        let _ = self.stop();
    }
}

fn health_ready(client: &reqwest::blocking::Client, url: &str) -> bool {
    client.get(url).send().map(|response| response.status().is_success()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_is_ok_when_nothing_is_running() {
        let server = ChatServer::new(Path::new("."));
        assert!(server.stop().is_ok());
    }

    #[test]
    fn health_requires_success_response() {
        use std::io::{Read, Write};
        let Ok(listener) = std::net::TcpListener::bind("127.0.0.1:0") else { return; };
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            for status in ["503 Service Unavailable", "200 OK"] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request);
                write!(stream, "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            }
        });
        let client = reqwest::blocking::Client::builder().no_proxy().timeout(Duration::from_secs(1)).build().unwrap();
        let url = format!("http://{addr}/health");
        assert!(!health_ready(&client, &url));
        assert!(health_ready(&client, &url));
        worker.join().unwrap();
    }

    #[test]
    fn start_fails_when_port_8080_is_occupied() {
        let listener = std::net::TcpListener::bind("127.0.0.1:8080");
        let Ok(_listener) = listener else {
            return;
        };
        let server = ChatServer::new(Path::new("."));
        let err = server
            .start("missing.gguf", None, None, None)
            .expect_err("occupied port must fail closed");
        assert!(err.contains("8080"), "{err}");
    }
}
