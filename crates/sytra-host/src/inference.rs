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
    plan_inference_with_options(
        workspace,
        model,
        vram_limit_mb,
        ram_limit_mb,
        context,
        export_runtimes,
        None,
    )
}

pub fn plan_inference_with_options(
    workspace: &Path,
    model: &str,
    vram_limit_mb: u64,
    ram_limit_mb: Option<u64>,
    context: Option<usize>,
    export_runtimes: bool,
    allow_unverified: Option<bool>,
) -> Result<Value, String> {
    if model.trim().starts_with('-') {
        return Err("Invalid model parameter: cannot start with '-'".into());
    }
    let script = workspace
        .join("runner")
        .join("scripts")
        .join("plan_inference.py");
    let python = python_executable(workspace);
    let per_model = crate::model_settings::load(workspace, Path::new(model))?;
    let settings = crate::settings::AppSettings::load(workspace);
    let runner = workspace.join("runner");
    let mut cmd = Command::new(python);
    cmd.env("PYTHONPATH", &runner);
    crate::apply_xet_safety(&mut cmd);
    crate::apply_desktop_priority(&mut cmd);
    let effective_vram = per_model
        .vram_limit_mb
        .map(|limit| (limit as u64).min(vram_limit_mb))
        .unwrap_or_else(|| {
            settings
                .vram_limit_mb
                .map(|limit| limit.min(vram_limit_mb))
                .unwrap_or(vram_limit_mb)
        });
    cmd.arg(&script)
        .arg("--model")
        .arg(model)
        .arg("--vram-limit")
        .arg(effective_vram.to_string())
        .arg("--context")
        .arg(context.or(per_model.context_length).unwrap_or(settings.default_context_window).to_string())
        .arg("--kv-cache-quant")
        .arg(per_model.kv_cache_quant.as_deref().unwrap_or(&settings.kv_cache_quant))
        .arg("--project-root")
        .arg(workspace);
    if per_model.cpu_kv_cache.unwrap_or(settings.cpu_kv_cache) {
        cmd.arg("--cpu-kv-cache");
    }
    if let Some(guardrail) = per_model.vram_guardrail_mb {
        cmd.arg("--vram-guardrail-mb").arg(guardrail.to_string());
    }
    if let Some(draft) = &per_model.draft_model_path {
        cmd.arg("--draft-model-path").arg(draft);
    }
    if !per_model.flash_attention.unwrap_or(settings.enable_flash_attention) {
        cmd.arg("--no-flash-attention");
    }
    if let Some(ram) = ram_limit_mb {
        cmd.arg("--ram-limit").arg(ram.to_string());
    }
    if export_runtimes {
        cmd.arg("--export-runtimes");
    }
    if allow_unverified.or(per_model.allow_unverified).unwrap_or(false) {
        cmd.arg("--allow-unverified");
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
