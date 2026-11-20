//! Dataset materialization shared by every front door (Tauri GUI, MCP
//! server, future CLI): resolves the `data:` block of a run config into a
//! canonical JSONL file + fingerprint before the runner is spawned.

use std::path::Path;

use sytra_contracts::run_config::{DataSpec, TrainMode};

use crate::{get_datasource, DatasetSpec, SourceKind};

/// Resolve one training source and an optional independent validation file.
pub async fn prepare_training_datasets(
    config: &mut sytra_contracts::run_config::RunConfig,
    out_dir: &Path,
) -> Result<(), String> {
    if matches!(config.data, DataSpec::Multi { .. }) {
        return Err("Training accepts one dataset; combine sources with Dataset Fabric first".into());
    }
    let original_train = match &config.data {
        DataSpec::Local { local, .. } => local.path.canonicalize().ok(),
        _ => None,
    };
    if let Some(explicit) = &config.train_dataset {
        let explicit = crate::datasource::local::LocalDataSource::ensure_safe_dataset_path(explicit)
            .map_err(|e| e.to_string())?;
        let previous_output = config.data.jsonl_path().and_then(|path| path.canonicalize().ok());
        if original_train.as_ref() != Some(&explicit) && previous_output.as_ref() != Some(&explicit) {
            return Err("train_dataset conflicts with data source".into());
        }
    }
    if let Some(path) = &config.val_dataset {
        let resolved = crate::datasource::local::LocalDataSource::ensure_safe_dataset_path(path)
            .map_err(|e| e.to_string())?;
        let ext = resolved.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if !resolved.is_file() || !matches!(ext.as_str(), "jsonl" | "parquet" | "json") {
            return Err("val_dataset must be a JSONL, Parquet, or dataset manifest file".into());
        }
        let val_shards = if ext == "json" {
            let (shards, _) = crate::dataset_fabric::validate_manifest(&resolved)?;
            shards
        } else {
            vec![resolved.clone()]
        };
        if original_train.as_ref().is_some_and(|train| train == &resolved || val_shards.contains(train)) {
            return Err("Training and validation must use different files".into());
        }
        let output_target = out_dir.join("data.jsonl").canonicalize().ok();
        if output_target.as_ref().is_some_and(|target| target == &resolved || val_shards.contains(target)) {
            return Err("Validation cannot be the materialized training output".into());
        }
        config.val_dataset = Some(resolved);
    }
    if let DataSpec::Local { local, jsonl_path, fingerprint } = &mut config.data {
        if matches!(local.format, sytra_contracts::run_config::LocalFormat::Manifest) {
            let path = crate::datasource::local::LocalDataSource::ensure_safe_dataset_path(&local.path)
                .map_err(|e| e.to_string())?;
            let (shards, digest) = crate::dataset_fabric::validate_manifest(&path)?;
            if let Some(val) = &config.val_dataset {
                let val_ext = val.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                if val == &path || shards.contains(val) {
                    return Err("Validation cannot be a training shard".into());
                }
                if val_ext == "json" {
                    let (val_shards, _) = crate::dataset_fabric::validate_manifest(val)?;
                    if val_shards.iter().any(|item| shards.contains(item)) {
                        return Err("Training and validation manifests share overlapping shards".into());
                    }
                }
            }
            *jsonl_path = Some(path.clone());
            *fingerprint = Some(digest);
            config.train_dataset = Some(path);
            return Ok(());
        }
    }
    materialize_dataset_for_config(&mut config.data, config.train_mode, out_dir).await?;
    config.train_dataset = config.data.jsonl_path().cloned();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn separates_training_and_validation_files() {
        let root = std::env::temp_dir().join(format!("sytra-datasets-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let train = root.join("train.jsonl");
        let val = root.join("validation.jsonl");
        let row = "{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\"}]}\n";
        std::fs::write(&train, row).unwrap();
        std::fs::write(&val, row).unwrap();
        let fixture = sytra_contracts::run_config::RunConfig::from_yaml_str(include_str!("../../../fixtures/run.golden.yaml")).unwrap();
        let mut raw = serde_json::to_value(fixture).unwrap();
        raw["train_dataset"] = serde_json::json!(train);
        raw["val_dataset"] = serde_json::json!(val);
        raw.as_object_mut().unwrap().remove("data");
        crate::training_source::normalize_aliases(&mut raw).unwrap();
        let mut config: sytra_contracts::run_config::RunConfig = serde_json::from_value(raw).unwrap();
        let mut overlap = config.clone();
        overlap.val_dataset = Some(train.clone());
        assert!(prepare_training_datasets(&mut overlap, &root.join("invalid")).await.unwrap_err().contains("different files"));
        prepare_training_datasets(&mut config, &root.join("output")).await.unwrap();
        assert_eq!(config.val_dataset.unwrap(), val.canonicalize().unwrap());
        let actual: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(config.train_dataset.unwrap()).unwrap()).unwrap();
        assert_eq!(actual, serde_json::from_str::<serde_json::Value>(row).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn indexed_shards_are_preserved_and_validation_cannot_overlap() {
        let root = std::env::temp_dir().join(format!("sytra-shards-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let one = root.join("one.jsonl");
        let two = root.join("two.jsonl");
        std::fs::write(&one, "{}\n").unwrap();
        std::fs::write(&two, "{}\n").unwrap();
        let manifest = root.join("manifest.json");
        std::fs::write(&manifest, serde_json::json!({"version":1,"rows":2,"shards":[
            {"path":"one.jsonl","rows":1},{"path":"two.jsonl","rows":1}
        ]}).to_string()).unwrap();
        let fixture = sytra_contracts::run_config::RunConfig::from_yaml_str(include_str!("../../../fixtures/run.golden.yaml")).unwrap();
        let mut raw = serde_json::to_value(fixture).unwrap();
        raw["train_dataset"] = serde_json::json!(manifest);
        raw["val_dataset"] = serde_json::json!(one);
        raw.as_object_mut().unwrap().remove("data");
        crate::training_source::normalize_aliases(&mut raw).unwrap();
        let mut config: sytra_contracts::run_config::RunConfig = serde_json::from_value(raw).unwrap();
        assert!(prepare_training_datasets(&mut config, &root.join("out")).await.unwrap_err().contains("training shard"));
        config.val_dataset = None;
        prepare_training_datasets(&mut config, &root.join("out")).await.unwrap();
        assert_eq!(config.train_dataset.as_ref().unwrap(), &manifest.canonicalize().unwrap());
        assert!(config.data.fingerprint().unwrap().starts_with("sha256:"));
        assert!(!root.join("out").exists());

        let val_manifest = root.join("val_manifest.json");
        let three = root.join("three.jsonl");
        std::fs::write(&three, "{}\n").unwrap();
        std::fs::write(&val_manifest, serde_json::json!({"version":1,"rows":1,"shards":[{"path":"three.jsonl","rows":1}]}).to_string()).unwrap();
        config.val_dataset = Some(val_manifest.clone());
        prepare_training_datasets(&mut config, &root.join("out")).await.unwrap();
        assert_eq!(config.val_dataset.as_ref().unwrap(), &val_manifest.canonicalize().unwrap());

        let overlap_manifest = root.join("overlap_manifest.json");
        std::fs::write(&overlap_manifest, serde_json::json!({"version":1,"rows":1,"shards":[{"path":"two.jsonl","rows":1}]}).to_string()).unwrap();
        config.val_dataset = Some(overlap_manifest);
        assert!(prepare_training_datasets(&mut config, &root.join("out")).await.unwrap_err().contains("overlapping shards"));

        std::fs::remove_dir_all(root).unwrap();
    }
}

pub async fn materialize_dataset_for_config(
    data: &mut DataSpec,
    train_mode: TrainMode,
    out_dir: &Path,
) -> Result<(), String> {
    match data {
        DataSpec::Multi {
            jsonl_path,
            fingerprint,
            datasets,
        } => {
            if datasets.is_empty() {
                return Err("No datasets provided".to_string());
            }
            if datasets.len() > 150 {
                return Err("Too many datasets (maximum limit is 150)".to_string());
            }

            let mut materialized_paths = Vec::new();
            let mut fingerprints = Vec::new();

            for (idx, ds) in datasets.iter_mut().enumerate() {
                let sub_out_dir = out_dir.join(format!("sub_{}", idx));
                std::fs::create_dir_all(&sub_out_dir).map_err(|e| e.to_string())?;

                // Box::pin allows recursive async call in Rust
                Box::pin(materialize_dataset_for_config(ds, train_mode, &sub_out_dir)).await?;

                if let Some(p) = ds.jsonl_path() {
                    materialized_paths.push(p.clone());
                }
                if let Some(f) = ds.fingerprint() {
                    fingerprints.push(f.clone());
                }
            }

            let consolidated_path = out_dir.join("consolidated.jsonl");
            let mut writer = std::fs::File::create(&consolidated_path)
                .map_err(|e| format!("failed to create consolidated file: {e}"))?;

            use std::io::Write;
            for path in materialized_paths {
                let file = std::fs::File::open(&path)
                    .map_err(|e| format!("failed to open sub-dataset file {:?}: {e}", path))?;
                let reader = std::io::BufReader::new(file);
                for line in std::io::BufRead::lines(reader) {
                    let line = line.map_err(|e| e.to_string())?;
                    writeln!(writer, "{}", line).map_err(|e| e.to_string())?;
                }
            }

            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            for f in &fingerprints {
                f.hash(&mut hasher);
            }
            let combined_fingerprint = format!("{:016x}", hasher.finish());

            *jsonl_path = Some(consolidated_path);
            *fingerprint = Some(combined_fingerprint);
            Ok(())
        }
        _ => {
            let (kind, params) = match data {
                DataSpec::Hf { hf, .. } => (SourceKind::Hf, serde_json::to_value(hf).unwrap()),
                DataSpec::Local { local, .. } => {
                    (SourceKind::Local, serde_json::to_value(local).unwrap())
                }
                DataSpec::Synthetic { synthetic, .. } => (
                    SourceKind::Synthetic,
                    serde_json::to_value(synthetic).unwrap(),
                ),
                DataSpec::Klayer { klayer, .. } => {
                    (SourceKind::Klayer, serde_json::to_value(klayer).unwrap())
                }
                DataSpec::Multi { .. } => unreachable!(),
            };

            let spec = DatasetSpec {
                source: kind,
                train_mode,
                params,
            };

            let provider = get_datasource(kind);
            provider
                .validate(&spec)
                .map_err(|e| format!("validation error: {e}"))?;

            let materialized = provider
                .materialize(&spec, out_dir)
                .await
                .map_err(|e| format!("materialize error: {e}"))?;

            match data {
                DataSpec::Hf {
                    jsonl_path,
                    fingerprint,
                    ..
                }
                | DataSpec::Local {
                    jsonl_path,
                    fingerprint,
                    ..
                }
                | DataSpec::Synthetic {
                    jsonl_path,
                    fingerprint,
                    ..
                }
                | DataSpec::Klayer {
                    jsonl_path,
                    fingerprint,
                    ..
                } => {
                    *jsonl_path = Some(materialized.jsonl_path);
                    *fingerprint = Some(materialized.fingerprint);
                }
                DataSpec::Multi { .. } => unreachable!(),
            }

            Ok(())
        }
    }
}
