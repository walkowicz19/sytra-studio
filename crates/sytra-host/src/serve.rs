use std::ffi::OsString;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::backend_resolver::BackendResolver;
use crate::process::kill_process_tree;
use crate::settings::AppSettings;
use crate::workspace::python_executable;

pub struct ChatServer {
    workspace: PathBuf,
    active_pid: Arc<Mutex<Option<u32>>>,
}

impl ChatServer {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            active_pid: Arc::new(Mutex::new(None)),
        }
    }

    pub fn stop(&self) -> Result<(), String> {
        if let Some(pid) = self.active_pid.lock().map_err(|_| "lock")?.take() {
            kill_process_tree(pid);
        }
        Ok(())
    }

    pub fn start(
        &self,
        model_path: &str,
        context: Option<usize>,
        vram_limit: Option<usize>,
        cpu_kv_cache: Option<bool>,
    ) -> Result<(), String> {
        self.stop()?;
        if TcpStream::connect(("127.0.0.1", 8080)).is_ok() {
            return Err(
                "Port 127.0.0.1:8080 is still occupied after stop. Close the stale llama.cpp, Ollama, or LM Studio process and retry."
                    .into(),
            );
        }

        let settings = AppSettings::load(&self.workspace);
        let detected_ram = BackendResolver::detect_system_ram_mb().ok_or_else(|| {
            "Could not detect system RAM; refusing to start a chat server without a memory envelope"
                .to_string()
        })?;
        let ctx_val = context.unwrap_or(settings.default_context_window);
        let vram_val = vram_limit.unwrap_or(settings.vram_limit_mb.unwrap_or(8192) as usize);
        let ram_val = settings.effective_main_memory_mb(detected_ram);
        let use_cpu_kv = cpu_kv_cache.unwrap_or(settings.cpu_kv_cache);
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
            "--vram-limit".into(),
            vram_val.to_string().into(),
            "--ram-limit".into(),
            ram_val.to_string().into(),
            "--kv-cache-quant".into(),
            settings.kv_cache_quant.clone().into(),
            "--project-root".into(),
            self.workspace.as_os_str().to_owned(),
            "--port".into(),
            "8080".into(),
        ];
        if use_cpu_kv {
            server_args.push("--cpu-kv-cache".into());
        }
        if !settings.enable_flash_attention {
            server_args.push("--no-flash-attention".into());
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

        let workspace = self.workspace.clone();
        let pid_ref = self.active_pid.clone();
        let launch_args = server_args;
        let python_launch = python;
        let started = Arc::new(Mutex::new(Err("chat server did not start".to_string())));
        let started_flag = started.clone();

        thread::spawn(move || {
            let mut cmd = std::process::Command::new(&python_launch);
            crate::apply_xet_safety(&mut cmd);
            crate::apply_desktop_priority(&mut cmd);
            cmd.args(&launch_args);
            cmd.current_dir(&workspace);
            match cmd.spawn() {
                Ok(mut child) => {
                    let pid = child.id();
                    if let Ok(mut g) = pid_ref.lock() {
                        *g = Some(pid);
                    }
                    if let Ok(mut g) = started_flag.lock() {
                        *g = Ok(());
                    }
                    let _ = child.wait();
                    if let Ok(mut g) = pid_ref.lock() {
                        if *g == Some(pid) {
                            *g = None;
                        }
                    }
                }
                Err(err) => {
                    if let Ok(mut g) = started_flag.lock() {
                        *g = Err(format!("Failed to spawn chat server: {err}"));
                    }
                }
            }
        });

        for _ in 0..50 {
            thread::sleep(Duration::from_millis(20));
            if let Ok(g) = started.lock() {
                match &*g {
                    Ok(()) => return Ok(()),
                    Err(msg) if msg != "chat server did not start" => return Err(msg.clone()),
                    _ => {}
                }
            }
        }
        Err("chat server did not confirm startup".into())
    }

    pub fn shutdown(&self) {
        let _ = self.stop();
    }
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
