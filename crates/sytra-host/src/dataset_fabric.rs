//! Dataset preparation shared by MCP and desktop commands.
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::env_provisioner::EnvProvisioner;

/// Validate the Dataset Fabric index and fingerprint its complete shard set.
pub fn validate_manifest(path: &Path) -> Result<(Vec<PathBuf>, String), String> {
    let manifest_path = path.canonicalize().map_err(|e| format!("Invalid dataset manifest: {e}"))?;
    let parent = manifest_path.parent().ok_or("Dataset manifest has no parent")?;
    let metadata = std::fs::metadata(&manifest_path).map_err(|e| e.to_string())?;
    if metadata.len() > 4 * 1024 * 1024 {
        return Err("Dataset manifest is too large".into());
    }
    let contents = std::fs::read(&manifest_path).map_err(|e| e.to_string())?;
    let manifest: Value = serde_json::from_slice(&contents).map_err(|e| format!("Invalid dataset manifest: {e}"))?;
    if manifest.get("version").and_then(Value::as_u64) != Some(1) {
        return Err("Unsupported dataset manifest version".into());
    }
    let entries = manifest.get("shards").and_then(Value::as_array).ok_or("Manifest requires shards")?;
    if entries.is_empty() || entries.len() > 10_000 { return Err("Manifest must contain 1 to 10000 shards".into()); }
    let mut paths = Vec::new();
    let mut rows = 0u64;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    for entry in entries {
        let name = entry.get("path").and_then(Value::as_str).ok_or("Shard requires a path")?;
        let relative = Path::new(name);
        if relative.components().count() != 1 || relative.file_name().and_then(|s| s.to_str()) != Some(name)
            || !name.to_ascii_lowercase().ends_with(".jsonl") {
            return Err("Shard paths must be JSONL filenames in the manifest directory".into());
        }
        let shard = parent.join(relative).canonicalize().map_err(|e| format!("Invalid shard {name}: {e}"))?;
        if shard.parent() != Some(parent) || !shard.is_file() || paths.contains(&shard) {
            return Err(format!("Invalid or duplicate shard: {name}"));
        }
        let count = entry.get("rows").and_then(Value::as_u64).filter(|n| *n > 0).ok_or("Shard rows must be positive")?;
        rows = rows.checked_add(count).ok_or("Shard row count overflow")?;
        let mut file = std::fs::File::open(&shard).map_err(|e| e.to_string())?;
        let mut buf = [0u8; 64 * 1024];
        let mut actual_rows = 0u64;
        let mut last_byte = None;
        loop {
            let n = file.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 { break; }
            hasher.update(&buf[..n]);
            actual_rows += buf[..n].iter().filter(|byte| **byte == b'\n').count() as u64;
            last_byte = Some(buf[n - 1]);
        }
        if last_byte.is_some_and(|byte| byte != b'\n') { actual_rows += 1; }
        if actual_rows != count { return Err(format!("Shard row count mismatch for {name}")); }
        paths.push(shard);
    }
    if manifest.get("rows").and_then(Value::as_u64) != Some(rows) {
        return Err("Manifest total does not match shard counts".into());
    }
    Ok((paths, format!("sha256:{:x}", hasher.finalize())))
}

fn input_path(workspace: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = Path::new(raw);
    let path = if path.is_absolute() { path.to_path_buf() } else { workspace.join(path) };
    let resolved = path.canonicalize().map_err(|e| format!("Invalid input path: {e}"))?;
    let root = workspace.canonicalize().map_err(|e| format!("Invalid workspace: {e}"))?;
    if !resolved.starts_with(root) || !resolved.is_file() {
        return Err("Input must be a file inside the workspace".into());
    }
    Ok(resolved)
}

fn model_directory(workspace: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = Path::new(raw);
    let path = if path.is_absolute() { path.to_path_buf() } else { workspace.join(path) };
    let resolved = path.canonicalize().map_err(|e| format!("Invalid local model directory: {e}"))?;
    let root = workspace.canonicalize().map_err(|e| format!("Invalid workspace: {e}"))?;
    if !resolved.starts_with(&root) || !resolved.is_dir() || !resolved.join("config.json").is_file() {
        return Err("Model or tokenizer must be a Transformers directory inside the workspace".into());
    }
    for entry in std::fs::read_dir(&resolved).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let target = entry.path().canonicalize().map_err(|e| format!("Invalid model file: {e}"))?;
        if !target.starts_with(&root) {
            return Err("Model or tokenizer files must remain inside the workspace".into());
        }
    }
    Ok(resolved)
}

fn output_path(workspace: &Path, raw: &str) -> Result<PathBuf, String> {
    let path = Path::new(raw);
    let path = if path.is_absolute() { path.to_path_buf() } else { workspace.join(path) };
    let name = path.file_name().ok_or("Output must name a new directory")?;
    let parent = path.parent().ok_or("Output must have an existing parent directory")?
        .canonicalize().map_err(|e| format!("Invalid output parent: {e}"))?;
    let root = workspace.canonicalize().map_err(|e| format!("Invalid workspace: {e}"))?;
    if !parent.starts_with(root) {
        return Err("Output must be inside the workspace".into());
    }
    let output = parent.join(name);
    if std::fs::symlink_metadata(&output).is_ok() {
        return Err("Output already exists".into());
    }
    Ok(output)
}

/// Validate paths before invoking the runner. Existing files are never overwritten.
pub fn prepare_request(workspace: &Path, operation: &str, args: &Value) -> Result<Value, String> {
    let output = output_path(workspace, args.get("output_dir").and_then(Value::as_str).ok_or("missing output_dir")?)?;
    let mut request = args.as_object().ok_or("Arguments must be an object")?.clone();
    request.insert("operation".into(), json!(operation));
    request.insert("output_dir".into(), json!(output));
    match operation {
        "merge" => {
            let sources = args.get("inputs").and_then(Value::as_array).ok_or("inputs must be an array")?;
            if sources.is_empty() || sources.len() > 150 {
                return Err("Provide between 1 and 150 input files".into());
            }
            let inputs: Result<Vec<_>, _> = sources.iter().map(|v| {
                input_path(workspace, v.as_str().ok_or("inputs must contain paths")?)
            }).collect();
            request.insert("inputs".into(), json!(inputs?));
            if let Some(curation) = request.get_mut("curation") {
                let object = curation.as_object_mut().ok_or("curation must be an object")?;
                if let Some(benchmarks) = object.get("benchmark_files") {
                    let paths = benchmarks.as_array().ok_or("benchmark_files must be an array")?;
                    if paths.len() > 50 { return Err("At most 50 benchmark files are supported".into()); }
                    let resolved: Result<Vec<_>, _> = paths.iter().map(|v| {
                        input_path(workspace, v.as_str().ok_or("benchmark_files must contain paths")?)
                    }).collect();
                    object.insert("benchmark_files".into(), json!(resolved?));
                }
                if let Some(semantic) = object.get_mut("semantic") {
                    let config = semantic.as_object_mut().ok_or("semantic must be an object")?;
                    let path = config.get("model_path").and_then(Value::as_str).ok_or("semantic model_path is required")?;
                    let resolved = model_directory(workspace, path)?;
                    config.insert("model_path".into(), json!(resolved));
                }
                if let Some(context) = object.get_mut("context") {
                    let config = context.as_object_mut().ok_or("context must be an object")?;
                    let path = config.get("tokenizer_path").and_then(Value::as_str).ok_or("context tokenizer_path is required")?;
                    let resolved = model_directory(workspace, path)?;
                    config.insert("tokenizer_path".into(), json!(resolved));
                }
            }
        }
        "split" => {
            let input = input_path(workspace, args.get("input").and_then(Value::as_str).ok_or("missing input")?)?;
            request.insert("input".into(), json!(input));
        }
        _ => return Err("Unknown dataset operation".into()),
    }
    Ok(Value::Object(request))
}

pub fn run(workspace: &Path, operation: &str, args: &Value) -> Result<Value, String> {
    let request = prepare_request(workspace, operation, args)?;
    let has_parquet = match operation {
        "merge" => request["inputs"].as_array().unwrap().iter().any(|v| v.as_str().unwrap_or("").to_ascii_lowercase().ends_with(".parquet")),
        _ => request["input"].as_str().unwrap_or("").to_ascii_lowercase().ends_with(".parquet"),
    } || request.get("curation").and_then(|v| v.get("benchmark_files")).and_then(Value::as_array)
        .is_some_and(|files| files.iter().any(|v| v.as_str().unwrap_or("").to_ascii_lowercase().ends_with(".parquet")));
    let provisioner = EnvProvisioner::new(workspace);
    let model_runtime = request.get("curation").is_some_and(|v| v.get("semantic").is_some() || v.get("context").is_some());
    let python = if has_parquet || model_runtime || (!provisioner.is_download_provisioned() && provisioner.is_train_provisioned()) {
        provisioner.ensure_train()
    } else {
        provisioner.ensure_download()
    }
        .map_err(|e| format!("Dataset runtime unavailable: {e}"))?;
    let mut cmd = Command::new(python);
    crate::apply_desktop_priority(&mut cmd);
    let mut child = cmd
        .arg(workspace.join("runner/scripts/dataset_fabric.py"))
        .current_dir(workspace)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().map_err(|e| format!("Cannot start dataset preparation: {e}"))?;
    let payload = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    child.stdin.take().ok_or("Cannot open dataset request stream")?.write_all(&payload).map_err(|e| e.to_string())?;
    let output = child.wait_with_output().map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let mut result: Value = serde_json::from_slice(&output.stdout).map_err(|e| format!("Invalid dataset result: {e}"))?;
    result["output_dir"] = request["output_dir"].clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_workspace_paths_and_refuses_existing_outputs() {
        let root = std::env::temp_dir().join(format!("sytra-fabric-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("input.jsonl"), "{}\n").unwrap();
        let args = json!({"inputs": ["input.jsonl"], "output_dir": "prepared"});
        let request = prepare_request(&root, "merge", &args).unwrap();
        assert!(Path::new(request["inputs"][0].as_str().unwrap()).is_absolute());
        let benchmark_args = json!({"inputs": ["input.jsonl"], "output_dir": "prepared", "curation": {"benchmark_files": ["input.jsonl"]}});
        let prepared = prepare_request(&root, "merge", &benchmark_args).unwrap();
        assert!(Path::new(prepared["curation"]["benchmark_files"][0].as_str().unwrap()).is_absolute());
        let escape = json!({"inputs": ["input.jsonl"], "output_dir": "prepared", "curation": {"benchmark_files": ["../outside.jsonl"]}});
        assert!(prepare_request(&root, "merge", &escape).is_err());
        std::fs::create_dir(root.join("reviewer")).unwrap();
        std::fs::write(root.join("reviewer/config.json"), "{}").unwrap();
        let semantic = json!({"inputs": ["input.jsonl"], "output_dir": "prepared", "curation": {"semantic": {"model_path": "reviewer"}}});
        let prepared = prepare_request(&root, "merge", &semantic).unwrap();
        assert!(Path::new(prepared["curation"]["semantic"]["model_path"].as_str().unwrap()).is_absolute());
        let semantic_escape = json!({"inputs": ["input.jsonl"], "output_dir": "prepared", "curation": {"semantic": {"model_path": ".."}}});
        assert!(prepare_request(&root, "merge", &semantic_escape).is_err());
        std::fs::create_dir(root.join("prepared")).unwrap();
        assert!(prepare_request(&root, "merge", &args).unwrap_err().contains("already exists"));
        let outside = json!({"input": "input.jsonl", "output_dir": "../escape"});
        assert!(prepare_request(&root, "split", &outside).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
