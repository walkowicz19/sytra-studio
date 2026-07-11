use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct SyntheticSourceParams {
    generator_model: String,
    judge_model: String,
    mode: String, // "prompts" | "sft" | "dpo"
    count: u32,
    topic: String,
}

pub struct SyntheticDataSource;

impl SyntheticDataSource {
    fn parse_params(spec: &DatasetSpec) -> Result<SyntheticSourceParams, DataSourceError> {
        serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
    }
}

#[async_trait]
impl DataSource for SyntheticDataSource {
    fn id(&self) -> &'static str {
        "synthetic"
    }

    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError> {
        let _params = Self::parse_params(spec)?;
        Ok(())
    }

    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let temp_dir =
            std::env::temp_dir().join(format!("sytra-synth-preview-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir)?;

        let mut spec_copy = spec.clone();
        let mut synth_params = params.clone();
        synth_params.count = n as u32;
        spec_copy.params = serde_json::to_value(&synth_params).unwrap();

        let materialized = self.materialize(&spec_copy, &temp_dir).await?;
        let content = std::fs::read_to_string(&materialized.jsonl_path)?;
        let mut rows = Vec::new();
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line)
                .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
            rows.push(row);
        }

        let _ = std::fs::remove_dir_all(&temp_dir);

        Ok(PreviewRows {
            rows,
            total_estimate: Some(params.count as usize),
        })
    }

    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let workspace = std::env::var("SYTRA_WORKSPACE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        // Locate python executable in train-env if provisioned
        let env_dir = workspace.join(".sytra-envs").join("train-env");
        let python_path = if cfg!(target_os = "windows") {
            env_dir.join("Scripts").join("python.exe")
        } else {
            env_dir.join("bin").join("python")
        };
        let python_exec = if python_path.exists() {
            python_path
        } else {
            PathBuf::from("python")
        };

        let out_file = out_dir.join("data.jsonl");

        let mut cmd = std::process::Command::new(python_exec);
        cmd.args(&[
            "-m",
            "sytra_runner.synth",
            "--generator",
            &params.generator_model,
            "--judge",
            &params.judge_model,
            "--mode",
            &params.mode.to_lowercase(),
            "--count",
            &params.count.to_string(),
            "--topic",
            &params.topic,
            "--output",
            &out_file.display().to_string(),
        ])
        .env("PYTHONPATH", workspace.join("runner"));

        let status = cmd.status()?;
        if !status.success() {
            return Err(DataSourceError::InvalidSpec(format!(
                "Synthetic generation script failed with status {}",
                status
            )));
        }

        Ok(Materialized {
            jsonl_path: out_file,
            fingerprint: self.fingerprint(spec)?,
            row_count: params.count as usize,
            provenance: None,
        })
    }

    fn fingerprint(&self, spec: &DatasetSpec) -> Result<String, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let mut hasher = Sha256::new();
        hasher.update(params.generator_model.as_bytes());
        hasher.update(params.judge_model.as_bytes());
        hasher.update(params.mode.as_bytes());
        hasher.update(params.count.to_string().as_bytes());
        hasher.update(params.topic.as_bytes());
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}
