use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sytra_contracts::run_config::TrainMode;

pub mod hf;
pub mod klayer;
pub mod local;
pub mod synthetic;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Hf,
    Local,
    Synthetic,
    Klayer,
}

/// Deserialized from the `data:` block of run.yaml.
#[derive(Debug, Clone)]
pub struct DatasetSpec {
    pub source: SourceKind,
    pub train_mode: TrainMode,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct PreviewRows {
    pub rows: Vec<serde_json::Value>,
    pub total_estimate: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub query: String,
    pub min_trust_tier: String,
    pub snapshot: String,
}

#[derive(Debug, Clone)]
pub struct Materialized {
    pub jsonl_path: PathBuf,
    pub fingerprint: String,
    pub row_count: usize,
    pub provenance: Option<Provenance>,
}

#[derive(Debug, thiserror::Error)]
pub enum DataSourceError {
    #[error("invalid dataset spec: {0}")]
    InvalidSpec(String),
    #[error("{0} provider is not implemented yet")]
    NotImplemented(&'static str),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[async_trait]
pub trait DataSource: Send + Sync {
    fn id(&self) -> &'static str;
    fn validate(&self, spec: &DatasetSpec) -> Result<(), DataSourceError>;
    async fn preview(&self, spec: &DatasetSpec, n: usize) -> Result<PreviewRows, DataSourceError>;
    async fn materialize(
        &self,
        spec: &DatasetSpec,
        out_dir: &Path,
    ) -> Result<Materialized, DataSourceError>;
    fn fingerprint(&self, spec: &DatasetSpec) -> Result<String, DataSourceError>;
}

pub fn get_datasource(kind: SourceKind) -> Box<dyn DataSource> {
    match kind {
        SourceKind::Hf => Box::new(hf::HfDataSource),
        SourceKind::Local => Box::new(local::LocalDataSource),
        SourceKind::Synthetic => Box::new(synthetic::SyntheticDataSource),
        SourceKind::Klayer => Box::new(klayer::KlayerDataSource),
    }
}
