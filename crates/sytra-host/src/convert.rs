use std::path::Path;
use std::process::Command;

use serde_json::Value;
use uuid::Uuid;

use crate::workspace::python_executable;

const ADAPTERS: &[&str] = &[
    "auto",
    "sytra-glm52",
    "sytra-kimi-k2.7-code",
    "sytra-kimi-k3",
    "sytra-inkling",
    "sytra-deepseek-v3",
    "sytra-qwen3-moe",
    "sytra-qwen2-moe",
    "sytra-mixtral",
    "sytra-olmoe",
    "sytra-dbrx",
    "sytra-granite-moe",
    "sytra-arctic",
    "sytra-minimax-moe",
    "sytra-generic-moe",
];
const FORMATS: &[&str] = &[
    "auto",
    "f32",
    "f16",
    "bf16",
    "int8",
    "int4_group",
    "packed_int4_group32",
    "fp8_e4m3",
    "nvfp4",
    "mxfp4",
    "gguf",
    "custom",
];

fn run_script(workspace: &Path, script_name: &str, args: &[&str]) -> Result<std::process::Output, String> {
    let script = workspace.join("runner").join("scripts").join(script_name);
    let python = python_executable(workspace);
    let mut cmd = Command::new(python);
    crate::apply_xet_safety(&mut cmd);
    crate::apply_desktop_priority(&mut cmd);
    let output = cmd
        .arg(script)
        .args(args)
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("Failed to start {script_name}: {e}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let msg = if detail.trim().is_empty() {
            stdout.trim()
        } else {
            detail.trim()
        };
        return Err(format!("{script_name} failed: {msg}"));
    }
    Ok(output)
}

pub fn convert_model(
    workspace: &Path,
    model: &str,
    outtype: Option<&str>,
    outfile: Option<&str>,
) -> Result<String, String> {
    let mut args = vec![
        model.to_string(),
        "--outtype".into(),
        outtype.unwrap_or("auto").to_string(),
    ];
    if let Some(out) = outfile {
        args.push("--outfile".into());
        args.push(out.to_string());
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_script(workspace, "convert_hf_to_gguf.py", &arg_refs)?;
    Ok(Uuid::new_v4().to_string())
}

pub fn export_model(
    workspace: &Path,
    model: &str,
    name: Option<&str>,
    context: usize,
) -> Result<String, String> {
    let mut args = vec![
        "--model".to_string(),
        model.to_string(),
        "--context".into(),
        context.to_string(),
    ];
    if let Some(n) = name {
        args.push("--name".into());
        args.push(n.to_string());
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_script(workspace, "export_model.py", &arg_refs)?;
    Ok(Uuid::new_v4().to_string())
}

pub fn build_moe_index(
    workspace: &Path,
    model_path: &str,
    adapter: &str,
    expert_format: &str,
    expert_regex: Option<&str>,
) -> Result<Value, String> {
    if !ADAPTERS.contains(&adapter) {
        return Err(format!("Unknown compiled Sytra adapter: {adapter}"));
    }
    if !FORMATS.contains(&expert_format) {
        return Err(format!("Unsupported expert weight format: {expert_format}"));
    }
    let root = std::path::PathBuf::from(model_path);
    if !root.is_dir() {
        return Err("Native MoE indexing requires a complete model directory".into());
    }
    let mut args = vec![
        "--model".to_string(),
        model_path.to_string(),
        "--adapter".into(),
        adapter.to_string(),
        "--expert-format".into(),
        expert_format.to_string(),
    ];
    if let Some(pattern) = expert_regex {
        if !pattern.trim().is_empty() {
            args.push("--expert-regex".into());
            args.push(pattern.trim().to_string());
        }
    }
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let output = run_script(workspace, "build_moe_index.py", &arg_refs)?;
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("Native MoE indexer returned invalid JSON: {error}"))
}
