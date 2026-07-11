use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use polars::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows};

#[derive(Debug, Clone, serde::Deserialize)]
struct LocalSourceParams {
    path: PathBuf,
    format: String,
    mapping: BTreeMap<String, String>,
}

pub struct LocalDataSource;

fn any_value_to_json(val: polars::prelude::AnyValue) -> serde_json::Value {
    match val {
        polars::prelude::AnyValue::Null => serde_json::Value::Null,
        polars::prelude::AnyValue::Boolean(b) => serde_json::Value::Bool(b),
        polars::prelude::AnyValue::String(s) => serde_json::Value::String(s.to_string()),
        polars::prelude::AnyValue::StringOwned(s) => serde_json::Value::String(s.to_string()),
        polars::prelude::AnyValue::Int64(i) => serde_json::Value::Number(i.into()),
        polars::prelude::AnyValue::Int32(i) => serde_json::Value::Number(i.into()),
        polars::prelude::AnyValue::UInt64(u) => serde_json::Value::Number(u.into()),
        polars::prelude::AnyValue::UInt32(u) => serde_json::Value::Number(u.into()),
        polars::prelude::AnyValue::Float64(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        polars::prelude::AnyValue::Float32(f) => serde_json::Number::from_f64(f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        other => serde_json::Value::String(other.to_string()),
    }
}

impl LocalDataSource {
    fn parse_params(spec: &DatasetSpec) -> Result<LocalSourceParams, DataSourceError> {
        serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))
    }

    /// Reads the source file and maps each row to the canonical
    /// `{prompt, completion}` (sft) row schema.
    fn read_canonical_rows(params: &LocalSourceParams) -> Result<Vec<Value>, DataSourceError> {
        match params.format.as_str() {
            "jsonl" => {
                let content = std::fs::read_to_string(&params.path)?;
                let prompt_col = params.mapping.get("prompt").ok_or_else(|| {
                    DataSourceError::InvalidSpec("mapping.prompt is required".into())
                })?;
                let completion_col = params.mapping.get("completion").ok_or_else(|| {
                    DataSourceError::InvalidSpec("mapping.completion is required".into())
                })?;

                let mut rows = Vec::new();
                for line in content.lines().filter(|l| !l.trim().is_empty()) {
                    let raw: Value = serde_json::from_str(line)
                        .map_err(|e| DataSourceError::InvalidSpec(format!("bad jsonl row: {e}")))?;
                    let prompt = raw.get(prompt_col).cloned().unwrap_or(Value::Null);
                    let completion = raw.get(completion_col).cloned().unwrap_or(Value::Null);
                    rows.push(serde_json::json!({ "prompt": prompt, "completion": completion }));
                }
                Ok(rows)
            }
            "csv" | "parquet" => {
                let df = if params.format == "csv" {
                    CsvReader::from_path(&params.path)
                        .map_err(|e| {
                            DataSourceError::InvalidSpec(format!("failed to read csv: {e}"))
                        })?
                        .has_header(true)
                        .finish()
                        .map_err(|e| {
                            DataSourceError::InvalidSpec(format!("failed to parse csv: {e}"))
                        })?
                } else {
                    let file =
                        std::fs::File::open(&params.path).map_err(|e| DataSourceError::Io(e))?;
                    ParquetReader::new(file).finish().map_err(|e| {
                        DataSourceError::InvalidSpec(format!("failed to parse parquet: {e}"))
                    })?
                };

                let prompt_col = params.mapping.get("prompt").ok_or_else(|| {
                    DataSourceError::InvalidSpec("mapping.prompt is required".into())
                })?;
                let completion_col = params.mapping.get("completion").ok_or_else(|| {
                    DataSourceError::InvalidSpec("mapping.completion is required".into())
                })?;

                let prompt_series = df.column(prompt_col).map_err(|e| {
                    DataSourceError::InvalidSpec(format!("column not found: {prompt_col} ({e})"))
                })?;
                let completion_series = df.column(completion_col).map_err(|e| {
                    DataSourceError::InvalidSpec(format!(
                        "column not found: {completion_col} ({e})"
                    ))
                })?;

                let mut rows = Vec::new();
                for i in 0..df.height() {
                    let prompt_val = prompt_series.get(i).unwrap();
                    let completion_val = completion_series.get(i).unwrap();
                    rows.push(serde_json::json!({
                        "prompt": any_value_to_json(prompt_val),
                        "completion": any_value_to_json(completion_val)
                    }));
                }
                Ok(rows)
            }
            _ => Err(DataSourceError::NotImplemented("local:unknown_format")),
        }
    }
}

#[async_trait]
impl DataSource for LocalDataSource {
    fn id(&self) -> &'static str {
        "local"
    }

    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError> {
        let params = Self::parse_params(spec)?;
        if !params.path.exists() {
            return Err(DataSourceError::InvalidSpec(format!(
                "path does not exist: {}",
                params.path.display()
            )));
        }
        Ok(())
    }

    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let rows = Self::read_canonical_rows(&params)?;
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
        let rows = Self::read_canonical_rows(&params)?;
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
        let content = std::fs::read(&params.path)?;
        let mapping_repr = serde_json::to_string(&params.mapping).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&content);
        hasher.update(mapping_repr.as_bytes());
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}
