//! Resolve local adapters before dataset work or spawning an ML process.
use serde_json::Value;
use std::path::{Path, PathBuf};
use sytra_contracts::run_config::{AdapterType, RunConfig};

pub fn normalize_aliases(raw: &mut Value) -> Result<(), String> {
    if raw.get("data").is_none() {
        if let Some(path) = raw.get("train_dataset").and_then(Value::as_str) {
            let format = match Path::new(path).extension().and_then(|v| v.to_str()).map(str::to_ascii_lowercase).as_deref() {
                Some("jsonl") => "jsonl",
                Some("parquet") => "parquet",
                Some("json") => "manifest",
                _ => return Err("train_dataset must be a JSONL, Parquet, or dataset manifest file".into()),
            };
            raw["data"] = serde_json::json!({"source": "local", "local": {
                "path": path, "format": format, "mapping": {"prompt": "prompt", "completion": "completion"}
            }});
        }
    }
    let mut values = Vec::new();
    for (block, key) in [
        ("optim", "grad_accumulation_steps"),
        ("optim", "gradient_accumulation_steps"),
        ("train", "gradient_accumulation_steps"),
        ("train", "grad_accumulation_steps"),
    ] {
        if let Some(value) = raw.get(block).and_then(|v| v.get(key)) {
            values.push(value.clone());
        }
    }
    if let Some(value) = values.first() {
        if values.iter().any(|v| v != value) || value.as_u64().unwrap_or(0) == 0 {
            return Err("Conflicting or invalid gradient accumulation aliases".into());
        }
        if raw.get("optim").is_none() {
            raw["optim"] = serde_json::json!({});
        }
        raw["optim"]["grad_accumulation_steps"] = value.clone();
    }
    for (block, key) in [
        ("optim", "gradient_accumulation_steps"),
        ("train", "gradient_accumulation_steps"),
        ("train", "grad_accumulation_steps"),
    ] {
        if let Some(object) = raw.get_mut(block).and_then(Value::as_object_mut) {
            object.remove(key);
        }
    }
    if let Some(output) = raw.get_mut("output").and_then(Value::as_object_mut) {
        if let Some(legacy) = output.remove("resume_from").filter(|v| !v.is_null()) {
            if output
                .get("resume_from_checkpoint")
                .is_some_and(|v| !v.is_null() && v != &legacy)
            {
                return Err("Conflicting checkpoint resume aliases".into());
            }
            output.insert("resume_from_checkpoint".into(), legacy);
        }
    }
    Ok(())
}

pub fn resolve(config: &mut RunConfig) -> Result<String, String> {
    if config.init_adapter.is_some() && config.output.resume_from.is_some() {
        return Err("Use init_adapter OR output.resume_from_checkpoint, not both".into());
    }
    if config.train.batch_size == 0
        || config.train.max_seq_len == 0
        || config.optim.grad_accumulation_steps == 0
    {
        return Err(
            "Batch size, sequence length, and gradient accumulation must be positive".into(),
        );
    }
    let model_path = Path::new(&config.model);
    let local_adapter = model_path.join("adapter_config.json").is_file();
    let source = config
        .output
        .resume_from
        .clone()
        .or_else(|| config.init_adapter.clone())
        .or_else(|| local_adapter.then(|| model_path.to_path_buf()));
    let Some(source) = source else {
        return Ok(config.model.clone());
    };
    let source = source
        .canonicalize()
        .map_err(|e| format!("Adapter path: {e}"))?;
    if local_adapter
        && config.init_adapter.is_some()
        && model_path.canonicalize().ok().as_ref() != Some(&source)
    {
        return Err("model and init_adapter refer to different adapters".into());
    }
    let metadata: Value = serde_json::from_str(
        &std::fs::read_to_string(source.join("adapter_config.json"))
            .map_err(|e| format!("Cannot read adapter_config.json: {e}"))?,
    )
    .map_err(|e| e.to_string())?;
    if metadata["peft_type"] != "LORA"
        || (!metadata["task_type"].is_null() && metadata["task_type"] != "CAUSAL_LM")
    {
        return Err("Continuation requires a causal-LM LoRA/DoRA adapter".into());
    }
    if std::fs::metadata(source.join("adapter_model.safetensors"))
        .map(|m| !m.is_file() || m.len() == 0)
        .unwrap_or(true)
    {
        return Err("Adapter requires non-empty adapter_model.safetensors".into());
    }
    let base = metadata["base_model_name_or_path"]
        .as_str()
        .filter(|v| !v.is_empty())
        .ok_or("Adapter metadata has no base_model_name_or_path")?
        .to_owned();
    if !local_adapter && config.model != base {
        let same_local = Path::new(&base)
            .canonicalize()
            .ok()
            .zip(model_path.canonicalize().ok())
            .is_some_and(|(a, b)| a == b);
        if !same_local {
            return Err(format!(
                "Adapter base '{base}' does not match model '{}'",
                config.model
            ));
        }
    }
    if matches!(config.adapter.kind, AdapterType::Full | AdapterType::Qat) {
        return Err("Existing adapters cannot initialize full/QAT training".into());
    }
    config.adapter.rank = metadata["r"]
        .as_u64()
        .filter(|r| *r > 0 && *r <= u32::MAX as u64)
        .ok_or("Invalid adapter rank")? as u32;
    config.adapter.alpha = metadata["lora_alpha"]
        .as_u64()
        .ok_or("Invalid adapter alpha")? as u32;
    config.adapter.dropout = metadata["lora_dropout"].as_f64().unwrap_or(0.0);
    config.adapter.use_rslora = metadata["use_rslora"].as_bool().unwrap_or(false);
    config.adapter.target_modules = serde_json::from_value(metadata["target_modules"].clone())
        .map_err(|_| "Adapter target_modules must be a list of module names")?;
    if config.adapter.target_modules.is_empty() {
        return Err("Adapter target_modules is empty".into());
    }
    if config.output.resume_from.is_some() {
        for name in [
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ] {
            if !source.join(name).is_file() {
                return Err(format!(
                    "Full resume is missing {name}; use init_adapter for weights-only continuation"
                ));
            }
        }
        let state: Value = serde_json::from_str(
            &std::fs::read_to_string(source.join("trainer_state.json"))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let step = state["global_step"]
            .as_u64()
            .ok_or("Checkpoint has no valid global_step")?;
        if config.train.max_steps.is_some_and(|max| max <= step) {
            return Err("Resume max_steps must exceed checkpoint global_step".into());
        }
        let provenance = source.parent().map(|parent| parent.join("sytra-provenance.json"));
        let (saved_ratio, saved_aux) = provenance.as_ref().filter(|path| path.is_file()).map(|path| {
            let raw = std::fs::read_to_string(path).map_err(|e| format!("Cannot read source provenance: {e}"))?;
            let value: Value = serde_json::from_str(&raw).map_err(|e| format!("Invalid source provenance: {e}"))?;
            let optim = match value.get("optim") {
                None => None,
                Some(Value::Object(map)) => Some(map),
                _ => return Err("Invalid optimizer section in source provenance".to_string()),
            };
            let numeric = |key: &str| match optim.and_then(|optim| optim.get(key)) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => value.as_f64().map(Some).ok_or_else(|| format!("Invalid saved {key}")),
            };
            Ok::<_, String>((numeric("router_lr_ratio")?, numeric("router_aux_coef")?))
        }).transpose()?.unwrap_or((None, None));
        if config.optim.router_lr_ratio.is_some() && config.optim.router_lr_ratio != saved_ratio {
            return Err("Resume must keep the saved router LR ratio and optimizer groups".into());
        }
        config.optim.router_lr_ratio = saved_ratio;
        if config.optim.router_aux_coef.is_some() && config.optim.router_aux_coef != saved_aux {
            return Err("Resume must keep the saved router auxiliary coefficient".into());
        }
        config.optim.router_aux_coef = saved_aux;
        config.output.resume_from = Some(source);
    } else {
        let output = absolute_output(&config.output.adapter_path)?;
        if output.starts_with(&source) || source.starts_with(&output) {
            return Err(
                "Continuation requires a separate output directory outside the source run".into(),
            );
        }
        config.init_adapter = Some(source);
    }
    config.model = base.clone();
    Ok(base)
}

fn absolute_output(path: &Path) -> Result<PathBuf, String> {
    let absolute = std::path::absolute(path).map_err(|e| e.to_string())?;
    // Resolve existing parents so junctions and .. cannot hide an overwrite.
    let mut ancestor = absolute.as_path();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(
            ancestor
                .file_name()
                .ok_or("Invalid output path")?
                .to_owned(),
        );
        ancestor = ancestor.parent().ok_or("Invalid output parent")?;
    }
    let mut resolved = ancestor.canonicalize().map_err(|e| e.to_string())?;
    for part in suffix.into_iter().rev() {
        resolved.push(part);
    }
    Ok(resolved)
}

pub fn latest_checkpoint(output: &Path) -> Result<PathBuf, String> {
    std::fs::read_dir(output)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let step = entry
                .file_name()
                .to_str()?
                .strip_prefix("checkpoint-")?
                .parse::<u64>()
                .ok()?;
            entry
                .path()
                .join("trainer_state.json")
                .is_file()
                .then(|| (step, entry.path()))
        })
        .max_by_key(|(step, _)| *step)
        .map(|(_, path)| path)
        .ok_or_else(|| "No numbered Trainer checkpoint found in the output directory".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn aliases_normalize_before_defaults() {
        let mut raw = json!({"train": {"gradient_accumulation_steps": 32}, "output": {"resume_from": "checkpoint-400"}});
        normalize_aliases(&mut raw).unwrap();
        assert_eq!(raw["optim"]["grad_accumulation_steps"], 32);
        assert_eq!(raw["output"]["resume_from_checkpoint"], "checkpoint-400");
        raw["train"]["gradient_accumulation_steps"] = json!(8);
        assert!(normalize_aliases(&mut raw).is_err());
    }

    #[test]
    fn local_adapter_initialization_and_full_resume_are_distinct() {
        let root = std::env::temp_dir().join(format!("sytra-source-{}", uuid::Uuid::new_v4()));
        let source = root.join("source/checkpoint-400");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("adapter_config.json"), json!({
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 32, "lora_alpha": 64, "target_modules": ["q_proj"]
        }).to_string()).unwrap();
        std::fs::write(source.join("adapter_model.safetensors"), b"weights").unwrap();
        let mut config =
            RunConfig::from_yaml_str(include_str!("../../../fixtures/run.golden.yaml")).unwrap();
        config.model = source.display().to_string();
        config.output.adapter_path = root.join("new-run");
        resolve(&mut config).unwrap();
        assert_eq!(config.model, "mistralai/Mistral-7B-v0.1");
        assert_eq!(config.adapter.rank, 32);
        assert!(config.init_adapter.is_some());
        config.init_adapter = None;
        config.output.resume_from = Some(source.clone());
        assert!(resolve(&mut config)
            .unwrap_err()
            .contains("missing trainer_state"));
        for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth"] {
            std::fs::write(source.join(name), b"state").unwrap();
        }
        std::fs::write(source.join("trainer_state.json"), r#"{"global_step":400}"#).unwrap();
        config.train.max_steps = Some(500);
        resolve(&mut config).unwrap();
        assert!(config.init_adapter.is_none());
        assert_eq!(latest_checkpoint(&root.join("source")).unwrap(), source);
        std::fs::write(source.parent().unwrap().join("sytra-provenance.json"),
                       r#"{"optim":{"router_lr_ratio":0.1,"router_aux_coef":0.02}}"#).unwrap();
        resolve(&mut config).unwrap();
        assert_eq!(config.optim.router_lr_ratio, Some(0.1));
        assert_eq!(config.optim.router_aux_coef, Some(0.02));
        config.optim.router_lr_ratio = Some(0.2);
        assert!(resolve(&mut config).unwrap_err().contains("saved router LR ratio"));
        config.optim.router_lr_ratio = Some(0.1);
        config.optim.router_aux_coef = Some(0.03);
        assert!(resolve(&mut config).unwrap_err().contains("saved router auxiliary coefficient"));
    }
}
