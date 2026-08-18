use std::path::Path;
use std::process::Command;

use serde_json::Value;

use crate::workspace::python_executable;

pub fn plan_inference(
    workspace: &Path,
    model: &str,
    vram_limit_mb: u64,
    ram_limit_mb: Option<u64>,
    context: Option<usize>,
    export_runtimes: bool,
) -> Result<Value, String> {
    let script = workspace
        .join("runner")
        .join("scripts")
        .join("plan_inference.py");
    let python = python_executable(workspace);
    let runner = workspace.join("runner");
    let mut cmd = Command::new(python);
    cmd.env("PYTHONPATH", &runner);
    crate::apply_xet_safety(&mut cmd);
    crate::apply_desktop_priority(&mut cmd);
    cmd.arg(&script)
        .arg("--model")
        .arg(model)
        .arg("--vram-limit")
        .arg(vram_limit_mb.to_string())
        .arg("--context")
        .arg(context.unwrap_or(4096).to_string())
        .arg("--project-root")
        .arg(workspace);
    if let Some(ram) = ram_limit_mb {
        cmd.arg("--ram-limit").arg(ram.to_string());
    }
    if export_runtimes {
        cmd.arg("--export-runtimes");
    }
    let output = cmd
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("Failed to start plan_inference.py: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed: Value = serde_json::from_str(stdout.trim()).map_err(|e| {
        let detail = if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            stderr.trim().to_string()
        };
        format!("plan_inference.py returned invalid JSON: {e}; {detail}")
    })?;
    Ok(parsed)
}
