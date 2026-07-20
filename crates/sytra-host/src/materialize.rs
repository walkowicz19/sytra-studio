//! Dataset materialization shared by every front door (Tauri GUI, MCP
//! server, future CLI): resolves the `data:` block of a run config into a
//! canonical JSONL file + fingerprint before the runner is spawned.

use std::path::Path;

use sytra_contracts::run_config::{DataSpec, TrainMode};

use crate::{get_datasource, DatasetSpec, SourceKind};

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
