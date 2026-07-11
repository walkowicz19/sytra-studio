use std::path::Path;

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows};

#[derive(Debug, Clone, serde::Deserialize)]
struct HfSourceParams {
    repo_id: String,
    split: String,
    revision: Option<String>,
    config: Option<String>,
}

pub struct HfDataSource;

impl HfDataSource {
    fn parse_params(spec: &DatasetSpec) -> Result<HfSourceParams, DataSourceError> {
        serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
    }

    async fn resolve_config_name(
        client: &reqwest::Client,
        repo_id: &str,
        split: &str,
    ) -> Option<String> {
        let url = format!(
            "https://datasets-server.huggingface.co/splits?dataset={}",
            repo_id
        );
        let resp = client
            .get(&url)
            .header("User-Agent", "SytraStudio/0.1.0")
            .send()
            .await
            .ok()?;

        if resp.status().is_success() {
            if let Ok(body) = resp.json::<Value>().await {
                if let Some(splits_arr) = body.get("splits").and_then(|s| s.as_array()) {
                    for item in splits_arr {
                        let item_split = item.get("split").and_then(|s| s.as_str()).unwrap_or("");
                        if item_split == split {
                            if let Some(config_name) = item.get("config").and_then(|c| c.as_str()) {
                                return Some(config_name.to_string());
                            }
                        }
                    }
                    if let Some(first_item) = splits_arr.first() {
                        if let Some(config_name) = first_item.get("config").and_then(|c| c.as_str())
                        {
                            return Some(config_name.to_string());
                        }
                    }
                }
            }
        }
        None
    }

    async fn fetch_rows_from_hub(
        repo_id: &str,
        split: &str,
        config: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Value>, DataSourceError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| {
                DataSourceError::InvalidSpec(format!("failed to build http client: {e}"))
            })?;

        let mut resolved_config = config.map(|s| s.to_string());
        if resolved_config.as_deref().unwrap_or("").trim().is_empty() {
            if let Some(auto_cfg) = Self::resolve_config_name(&client, repo_id, split).await {
                resolved_config = Some(auto_cfg);
            }
        }

        let mut url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&split={}&offset=0&limit={}",
            repo_id, split, limit
        );
        if let Some(ref cfg) = resolved_config {
            if !cfg.trim().is_empty() {
                url = format!("{}&config={}", url, cfg);
            }
        }
        println!("HF Query URL: {}", url);

        let resp = client
            .get(&url)
            .header("User-Agent", "SytraStudio/0.1.0")
            .send()
            .await
            .map_err(|e| {
                DataSourceError::InvalidSpec(format!("failed to fetch from hf hub: {e}"))
            })?;

        if !resp.status().is_success() {
            return Err(DataSourceError::InvalidSpec(format!(
                "HF datasets server returned status {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }

        let body: Value = resp.json().await.map_err(|e| {
            DataSourceError::InvalidSpec(format!("failed to parse hf json response: {e}"))
        })?;

        // Response contains "rows": [{"row_idx": 0, "row": {...}}]
        let rows_val = body.get("rows").and_then(|r| r.as_array()).ok_or_else(|| {
            DataSourceError::InvalidSpec(
                "invalid response structure: 'rows' array not found".into(),
            )
        })?;

        let mut canonical_rows = Vec::new();
        for item in rows_val {
            let row = item
                .get("row")
                .ok_or_else(|| DataSourceError::InvalidSpec("invalid row structure".into()))?;

            // Map common instruction keys to prompt/completion
            let prompt = row
                .get("prompt")
                .or_else(|| row.get("instruction"))
                .or_else(|| row.get("input"))
                .or_else(|| row.get("text"))
                .cloned()
                .unwrap_or(Value::Null);

            let completion = row
                .get("completion")
                .or_else(|| row.get("response"))
                .or_else(|| row.get("output"))
                .cloned()
                .unwrap_or(Value::Null);

            canonical_rows.push(serde_json::json!({
                "prompt": prompt,
                "completion": completion
            }));
        }

        Ok(canonical_rows)
    }
}

#[async_trait]
impl DataSource for HfDataSource {
    fn id(&self) -> &'static str {
        "hf"
    }

    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError> {
        let _params = Self::parse_params(spec)?;
        Ok(())
    }

    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let rows =
            Self::fetch_rows_from_hub(&params.repo_id, &params.split, params.config.as_deref(), n)
                .await?;
        let total_estimate = Some(rows.len());
        Ok(PreviewRows {
            rows: rows.into_iter().take(n).collect(),
            total_estimate,
        })
    }

    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError> {
        let params = Self::parse_params(spec)?;
        // Fetch up to 1000 rows to materialize for fine-tuning split
        let rows = Self::fetch_rows_from_hub(
            &params.repo_id,
            &params.split,
            params.config.as_deref(),
            1000,
        )
        .await?;
        std::fs::create_dir_all(out_dir)?;
        let jsonl_path = out_dir.join("data.jsonl");
        let body = rows
            .iter()
            .map(|r| r.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&jsonl_path, body)?;
        Ok(Materialized {
            jsonl_path,
            fingerprint: self.fingerprint(spec)?,
            row_count: rows.len(),
            provenance: None,
        })
    }

    fn fingerprint(&self, spec: &DatasetSpec) -> Result<String, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let mut hasher = Sha256::new();
        hasher.update(params.repo_id.as_bytes());
        hasher.update(params.split.as_bytes());
        if let Some(ref rev) = params.revision {
            hasher.update(rev.as_bytes());
        }
        if let Some(ref cfg) = params.config {
            hasher.update(cfg.as_bytes());
        }
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}
