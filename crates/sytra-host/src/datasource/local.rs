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
    #[serde(default)]
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
    pub(crate) fn ensure_safe_dataset_path(path: &Path) -> Result<PathBuf, DataSourceError> {
        let canonical_target = path
            .canonicalize()
            .map_err(|e| DataSourceError::InvalidSpec(format!("Invalid dataset path: {e}")))?;
        let workspace = crate::workspace::resolve_workspace();
        let canonical_ws = workspace
            .canonicalize()
            .map_err(|e| DataSourceError::InvalidSpec(format!("Invalid workspace: {e}")))?;
        let temp_dir = std::env::temp_dir().canonicalize().ok();

        let in_ws = canonical_target.starts_with(&canonical_ws);
        let in_temp = temp_dir
            .as_ref()
            .map(|t| canonical_target.starts_with(t))
            .unwrap_or(false);

        if !in_ws && !in_temp {
            return Err(DataSourceError::InvalidSpec(
                "Access denied: Dataset path must reside inside the active workspace".into(),
            ));
        }
        Ok(canonical_target)
    }

    fn parse_params(spec: &DatasetSpec) -> Result<LocalSourceParams, DataSourceError> {
        let mut params: LocalSourceParams = serde_json::from_value(spec.params.clone())
            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
        if !params.path.exists() {
            return Err(DataSourceError::InvalidSpec(format!(
                "path does not exist: {}",
                params.path.display()
            )));
        }
        params.path = Self::ensure_safe_dataset_path(&params.path)?;
        Ok(params)
    }

    fn map_jsonl_row(raw: &Value, params: &LocalSourceParams) -> Result<Value, DataSourceError> {
        if let Some(messages) = raw.get("messages") {
            if !messages.is_array() {
                return Err(DataSourceError::InvalidSpec("messages must be an array when present".into()));
            }
            let has_mapped_pair = params.mapping.get("prompt").and_then(|key| raw.get(key)).is_some()
                && params.mapping.get("completion").and_then(|key| raw.get(key)).is_some();
            if !has_mapped_pair {
                return Ok(serde_json::json!({"messages": messages}));
            }
        }
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
    fn lazy_tabular_frame(params: &LocalSourceParams) -> Result<LazyFrame, DataSourceError> {
        if params.format == "csv" {
            LazyCsvReader::new(&params.path)
                .has_header(true)
                .finish()
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to read csv: {e}")))
        } else {
            LazyFrame::scan_parquet(&params.path, ScanArgsParquet::default())
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to parse parquet: {e}")))
        }
    }

    #[cfg(feature = "local-datasets")]
    fn tabular_df_rows(df: &DataFrame, params: &LocalSourceParams) -> Result<Vec<Value>, DataSourceError> {
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
        let mut rows = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            let prompt_val = prompt_series.get(i).map_err(|e| {
                DataSourceError::InvalidSpec(format!("Error reading row {i} from prompt column: {e}"))
            })?;
            let completion_val = completion_series.get(i).map_err(|e| {
                DataSourceError::InvalidSpec(format!("Error reading row {i} from completion column: {e}"))
            })?;
            rows.push(serde_json::json!({
                "prompt": any_value_to_json(prompt_val),
                "completion": any_value_to_json(completion_val)
            }));
        }
        Ok(rows)
    }

    #[cfg(feature = "local-datasets")]
    fn read_tabular_rows(params: &LocalSourceParams, limit: Option<usize>) -> Result<(Vec<Value>, usize), DataSourceError> {
        const CHUNK_ROWS: usize = 2048;
        let lf = Self::lazy_tabular_frame(params)?;
        if let Some(n) = limit {
            let sliced = if n == 0 {
                lf.slice(0, 0).collect()
            } else {
                lf.slice(0, n as u32).collect()
            }
            .map_err(|e| DataSourceError::InvalidSpec(format!("failed to collect tabular preview: {e}")))?;
            let rows = Self::tabular_df_rows(&sliced, params)?;
            let count_df = Self::lazy_tabular_frame(params)?
                .select([polars::lazy::dsl::count().alias("rows")])
                .collect()
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to count tabular rows: {e}")))?;
            let total = count_df
                .column("rows")
                .ok()
                .and_then(|col| col.get(0).ok())
                .and_then(|v| match v {
                    polars::prelude::AnyValue::UInt32(c) => Some(c as usize),
                    polars::prelude::AnyValue::UInt64(c) => Some(c as usize),
                    polars::prelude::AnyValue::Int64(c) if c >= 0 => Some(c as usize),
                    _ => None,
                })
                .unwrap_or(rows.len());
            return Ok((rows, total));
        }
        let mut all = Vec::new();
        let mut offset: i64 = 0;
        loop {
            let chunk = Self::lazy_tabular_frame(params)?
                .slice(offset, CHUNK_ROWS as u32)
                .collect()
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to collect tabular chunk: {e}")))?;
            let height = chunk.height();
            if height == 0 {
                break;
            }
            all.extend(Self::tabular_df_rows(&chunk, params)?);
            offset += height as i64;
        }
        let total = all.len();
        Ok((all, total))
    }

    #[cfg(feature = "local-datasets")]
    fn stream_tabular_to_writer<W: std::io::Write>(
        params: &LocalSourceParams,
        writer: &mut W,
    ) -> Result<usize, DataSourceError> {
        const CHUNK_ROWS: usize = 2048;
        let mut offset: i64 = 0;
        let mut total = 0usize;
        loop {
            let chunk = Self::lazy_tabular_frame(params)?
                .slice(offset, CHUNK_ROWS as u32)
                .collect()
                .map_err(|e| DataSourceError::InvalidSpec(format!("failed to stream tabular dataset: {e}")))?;
            let height = chunk.height();
            if height == 0 {
                break;
            }
            for row in Self::tabular_df_rows(&chunk, params)? {
                serde_json::to_writer(&mut *writer, &row)
                    .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
                writer.write_all(b"\n")?;
                total += 1;
            }
            offset += height as i64;
        }
        Ok(total)
    }

    fn read_canonical_rows(
        params: &LocalSourceParams,
        limit: Option<usize>,
    ) -> Result<(Vec<Value>, usize), DataSourceError> {
        match params.format.as_str() {
            "jsonl" => Self::read_jsonl_rows(params, limit),
            "manifest" => {
                let (shards, _) = crate::dataset_fabric::validate_manifest(&params.path)
                    .map_err(DataSourceError::InvalidSpec)?;
                let mut all = Vec::new();
                let mut total = 0;
                for path in shards {
                    let mut shard_params = params.clone();
                    shard_params.path = path;
                    let remaining = limit.map(|n| n.saturating_sub(all.len()));
                    let (rows, count) = Self::read_jsonl_rows(&shard_params, remaining)?;
                    all.extend(rows);
                    total += count;
                }
                Ok((all, total))
            }
            "csv" | "parquet" => {
                #[cfg(feature = "local-datasets")]
                {
                    Self::read_tabular_rows(params, limit)
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
        std::fs::create_dir_all(out_dir)?;
        let jsonl_path = out_dir.join("data.jsonl");

        if jsonl_path.exists() && jsonl_path.canonicalize()? == params.path {
            return Err(DataSourceError::InvalidSpec("Dataset input and materialized output must differ".into()));
        }

        let tmp_path = out_dir.join(format!(".data-{}.jsonl.tmp", uuid::Uuid::new_v4()));
        use std::io::Write;
        let write_result: Result<usize, DataSourceError> = (|| {
            let file = std::fs::File::create(&tmp_path)?;
            let mut writer = std::io::BufWriter::new(file);
            let mut row_count = 0usize;
            match params.format.as_str() {
                "jsonl" => {
                    for line in BufReader::new(std::fs::File::open(&params.path)?).lines() {
                        let line = line?;
                        if line.trim().is_empty() { continue; }
                        let raw: Value = serde_json::from_str(&line)
                            .map_err(|e| DataSourceError::InvalidSpec(format!("bad jsonl row: {e}")))?;
                        let row = Self::map_jsonl_row(&raw, &params)?;
                        serde_json::to_writer(&mut writer, &row)
                            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
                        writer.write_all(b"\n")?;
                        row_count += 1;
                    }
                }
                "csv" | "parquet" => {
                    #[cfg(feature = "local-datasets")]
                    {
                        row_count = Self::stream_tabular_to_writer(&params, &mut writer)?;
                    }
                    #[cfg(not(feature = "local-datasets"))]
                    {
                        return Err(DataSourceError::NotImplemented(
                            "local csv/parquet requires the local-datasets feature",
                        ));
                    }
                }
                _ => {
                    let (rows, _) = Self::read_canonical_rows(&params, None)?;
                    for row in &rows {
                        serde_json::to_writer(&mut writer, row)
                            .map_err(|e| DataSourceError::InvalidSpec(e.to_string()))?;
                        writer.write_all(b"\n")?;
                    }
                    row_count = rows.len();
                }
            }
            writer.flush()?;
            Ok(row_count)
        })();
        let row_count = match write_result {
            Ok(count) => count,
            Err(err) => {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(err);
            }
        };
        if let Err(err) = std::fs::rename(&tmp_path, &jsonl_path) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(err.into());
        }

        Ok(Materialized {
            jsonl_path,
            fingerprint: self.fingerprint(spec)?,
            row_count,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_conversations_do_not_require_prompt_mapping() {
        let params = LocalSourceParams { path: PathBuf::new(), format: "jsonl".into(), mapping: BTreeMap::new() };
        let raw = serde_json::json!({"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "final"}
        ]});
        assert_eq!(LocalDataSource::map_jsonl_row(&raw, &params).unwrap(), raw);
    }

    #[tokio::test]
    async fn csv_preview_and_materialize_stream_atomically_and_clean_up_failures() {
        let root = std::env::temp_dir().join(format!("sytra-local-csv-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let csv_path = root.join("sample.csv");
        let mut csv = String::from("prompt,completion\n");
        for i in 0..2500 {
            csv.push_str(&format!("q{i},a{i}\n"));
        }
        std::fs::write(&csv_path, csv).unwrap();

        let source = LocalDataSource;
        let spec = DatasetSpec {
            source: crate::SourceKind::Local,
            train_mode: sytra_contracts::run_config::TrainMode::Sft,
            params: serde_json::json!({
                "path": csv_path,
                "format": "csv",
                "mapping": {"prompt": "prompt", "completion": "completion"}
            }),
        };
        let preview = source.preview(&spec, 3).await.unwrap();
        assert_eq!(preview.rows.len(), 3);
        assert_eq!(preview.total_estimate, Some(2500));

        let out_dir = root.join("out");
        let materialized = source.materialize(&spec, &out_dir).await.unwrap();
        assert_eq!(materialized.row_count, 2500);
        assert!(materialized.jsonl_path.is_file());

        let broken_jsonl = root.join("broken.jsonl");
        std::fs::write(&broken_jsonl, "{\"prompt\":\"ok\",\"completion\":\"yes\"}\n{bad-json\n").unwrap();
        let fail_dir = root.join("fail_out");
        let bad_spec = DatasetSpec {
            source: crate::SourceKind::Local,
            train_mode: sytra_contracts::run_config::TrainMode::Sft,
            params: serde_json::json!({
                "path": broken_jsonl,
                "format": "jsonl",
                "mapping": {"prompt": "prompt", "completion": "completion"}
            }),
        };
        assert!(source.materialize(&bad_spec, &fail_dir).await.is_err());
        assert!(!fail_dir.join("data.jsonl").exists());
        assert_eq!(std::fs::read_dir(&fail_dir).unwrap().count(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }
}
