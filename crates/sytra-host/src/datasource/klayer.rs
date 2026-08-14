use std::path::Path;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows, Provenance};

#[derive(Debug, Clone, serde::Deserialize)]
struct KlayerSourceParams {
    query: String,
    min_trust_tier: String,
    snapshot: String,
}

pub struct KlayerDataSource;

impl KlayerDataSource {
    fn parse_params(spec: &DatasetSpec) -> Result<KlayerSourceParams, DataSourceError> {
        serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
    }
}

#[async_trait]
impl DataSource for KlayerDataSource {
    fn id(&self) -> &'static str {
        "klayer"
    }

    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError> {
        let _params = Self::parse_params(spec)?;
        Ok(())
    }

    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError> {
        let temp = std::env::temp_dir().join(format!("sytra-klayer-preview-{}", uuid::Uuid::new_v4()));
        let materialized = self.materialize(spec, &temp).await?;
        let content = std::fs::read_to_string(&materialized.jsonl_path)?;
        let rows: Vec<_> = content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(n)
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let _ = std::fs::remove_dir_all(&temp);
        Ok(PreviewRows {
            rows,
            total_estimate: Some(materialized.row_count),
        })
    }

    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError> {
        let params = Self::parse_params(spec)?;
        std::fs::create_dir_all(out_dir)?;
        let jsonl_path = out_dir.join("data.jsonl");

        let status = std::process::Command::new("kl-train")
            .arg("--help")
            .status()
            .map_err(|_| {
                DataSourceError::InvalidSpec(
                    "kl-train is not installed or not on PATH; refusing to fabricate dataset rows"
                        .into(),
                )
            })?;
        if !status.success() {
            return Err(DataSourceError::InvalidSpec(
                "kl-train is not usable; refusing to fabricate dataset rows".into(),
            ));
        }

        let status = std::process::Command::new("kl-train")
            .args([
                "materialize",
                "--query",
                &params.query,
                "--min-trust-tier",
                &params.min_trust_tier,
                "--snapshot",
                &params.snapshot,
                "--output",
                &jsonl_path.display().to_string(),
            ])
            .status()?;
        if !status.success() {
            return Err(DataSourceError::InvalidSpec(
                "kl-train execution failed".into(),
            ));
        }
        let content = std::fs::read_to_string(&jsonl_path)?;
        let row_count = content.lines().filter(|l| !l.trim().is_empty()).count();

        Ok(Materialized {
            jsonl_path,
            fingerprint: self.fingerprint(spec)?,
            row_count,
            provenance: Some(Provenance {
                query: params.query.clone(),
                min_trust_tier: params.min_trust_tier.clone(),
                snapshot: params.snapshot.clone(),
            }),
        })
    }

    fn fingerprint(&self, spec: &DatasetSpec) -> Result<String, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let mut hasher = Sha256::new();
        hasher.update(params.query.as_bytes());
        hasher.update(params.min_trust_tier.as_bytes());
        hasher.update(params.snapshot.as_bytes());
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}
