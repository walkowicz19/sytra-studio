use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{DataSource, DataSourceError, DatasetSpec, Materialized, PreviewRows};

#[cfg(feature = "local-datasets")]
use polars::prelude::*;

#[derive(Debug, Clone, serde::Deserialize)]
struct LocalSourceParams {
    path: PathBuf,
    format: String,
    mapping: BTreeMap<String, String>,
}

pub struct LocalDataSource;

#[cfg(feature = "local-datasets")]
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

    fn map_jsonl_row(raw: &Value, params: &LocalSourceParams) -> Result<Value, DataSourceError> {
        let prompt_col = params.mapping.get("prompt").ok_or_else(|| {
            DataSourceError::InvalidSpec("mapping.prompt is required".into())
        })?;
        let completion_col = params.mapping.get("completion").ok_or_else(|| {
            DataSourceError::InvalidSpec("mapping.completion is required".into())
        })?;
        let prompt = raw.get(prompt_col).cloned().unwrap_or(Value::Null);
        let completion = raw.get(completion_col).cloned().unwrap_or(Value::Null);
        let mut canonical = serde_json::json!({ "prompt": prompt, "completion": completion });
        if let Some(messages) = raw.get("messages") {
            if !messages.is_array() {
                return Err(DataSourceError::InvalidSpec(
                    "messages must be an array when present".into(),
                ));
            }
            canonical["messages"] = messages.clone();
        }
        Ok(canonical)
    }

    fn read_jsonl_rows(
        params: &LocalSourceParams,
        limit: Option<usize>,
    ) -> Result<(Vec<Value>, usize), DataSourceError> {
        let file = std::fs::File::open(&params.path)?;
        let reader = BufReader::new(file);
        let mut rows = Vec::new();
        let mut total = 0usize;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            total += 1;
            if let Some(n) = limit {
                if rows.len() >= n {
                    continue;
                }
            }
            let raw: Value = serde_json::from_str(&line)
                .map_err(|e| DataSourceError::InvalidSpec(format!("bad jsonl row: {e}")))?;
            rows.push(Self::map_jsonl_row(&raw, params)?);
        }
        Ok((rows, total))
    }

    #[cfg(feature = "local-datasets")]
    fn read_tabular_rows(params: &LocalSourceParams) -> Result<Vec<Value>, DataSourceError> {
        let lf = if params.format == "csv" {
            LazyCsvReader::new(&params.path)
                .has_header(true)
                .finish()
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to read csv: {e}")))?
        } else {
            LazyFrame::scan_parquet(&params.path, ScanArgsParquet::default())
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to parse parquet: {e}")))?
        };
        let df = lf.collect().map_err(|e| {
            DataSourceError::InvalidSpec(format!("failed to collect tabular dataset: {e}"))
        })?;

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
            DataSourceError::InvalidSpec(format!("column not found: {completion_col} ({e})"))
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

    fn read_canonical_rows(
        params: &LocalSourceParams,
        limit: Option<usize>,
    ) -> Result<(Vec<Value>, usize), DataSourceError> {
        match params.format.as_str() {
            "jsonl" => Self::read_jsonl_rows(params, limit),
            "csv" | "parquet" => {
                #[cfg(feature = "local-datasets")]
                {
                    let rows = Self::read_tabular_rows(params)?;
                    let total = rows.len();
                    let rows = match limit {
                        Some(n) => rows.into_iter().take(n).collect(),
                        None => rows,
                    };
                    Ok((rows, total))
                }
                #[cfg(not(feature = "local-datasets"))]
                {
                    let _ = limit;
                    Err(DataSourceError::NotImplemented(
                        "local csv/parquet requires the local-datasets feature",
                    ))
                }
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
        let (rows, total) = Self::read_canonical_rows(&params, Some(n))?;
        Ok(PreviewRows {
            rows,
            total_estimate: Some(total),
        })
    }

    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError> {
        let params = Self::parse_params(spec)?;
        let (rows, _) = Self::read_canonical_rows(&params, None)?;
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
        let mut file = std::fs::File::open(&params.path)?;
        let mapping_repr = serde_json::to_string(&params.mapping).unwrap_or_default();
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hasher.update(mapping_repr.as_bytes());
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }
}
