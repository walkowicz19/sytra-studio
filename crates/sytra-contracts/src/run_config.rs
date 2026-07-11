use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainMode {
    Sft,
    Dpo,
    Cpo,
    Orpo,
    Grpo,
    OnlineDpo,
    Xpo,
    RlhfReinforce,
    Ppo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Auto,
    Cuda,
    Rocm,
    Mps,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub kind: BackendKind,
    pub judge_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfParams {
    pub repo_id: String,
    #[serde(default = "default_split")]
    pub split: String,
    pub revision: Option<String>,
    pub config: Option<String>,
}

fn default_split() -> String {
    "train".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LocalFormat {
    Jsonl,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalParams {
    pub path: PathBuf,
    pub format: LocalFormat,
    pub mapping: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticMode {
    Prompts,
    Sft,
    Dpo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticParams {
    pub generator_model: String,
    pub judge_model: String,
    pub mode: SyntheticMode,
    pub count: u32,
    pub topic: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlayerParams {
    pub query: String,
    pub min_trust_tier: String,
    pub snapshot: String,
}

/// Discriminated union on `source`. The `source` value is also the key
/// holding the source-specific block, matching the run.yaml layout in
/// Contract 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum DataSpec {
    Hf {
        jsonl_path: Option<PathBuf>,
        fingerprint: Option<String>,
        hf: HfParams,
    },
    Local {
        jsonl_path: Option<PathBuf>,
        fingerprint: Option<String>,
        local: LocalParams,
    },
    Synthetic {
        jsonl_path: Option<PathBuf>,
        fingerprint: Option<String>,
        synthetic: SyntheticParams,
    },
    Klayer {
        jsonl_path: Option<PathBuf>,
        fingerprint: Option<String>,
        klayer: KlayerParams,
    },
    Multi {
        jsonl_path: Option<PathBuf>,
        fingerprint: Option<String>,
        datasets: Vec<DataSpec>,
    },
}

impl DataSpec {
    pub fn jsonl_path(&self) -> Option<&PathBuf> {
        match self {
            DataSpec::Hf { jsonl_path, .. }
            | DataSpec::Local { jsonl_path, .. }
            | DataSpec::Synthetic { jsonl_path, .. }
            | DataSpec::Klayer { jsonl_path, .. }
            | DataSpec::Multi { jsonl_path, .. } => jsonl_path.as_ref(),
        }
    }

    pub fn fingerprint(&self) -> Option<&String> {
        match self {
            DataSpec::Hf { fingerprint, .. }
            | DataSpec::Local { fingerprint, .. }
            | DataSpec::Synthetic { fingerprint, .. }
            | DataSpec::Klayer { fingerprint, .. }
            | DataSpec::Multi { fingerprint, .. } => fingerprint.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdapterType {
    Lora,
    Dora,
    Qlora,
    Full,
    Qat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    #[serde(rename = "type")]
    pub kind: AdapterType,
    pub rank: u32,
    pub alpha: u32,
    pub dropout: f64,
    pub target_modules: Vec<String>,
    pub quant_bits: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Schedule {
    Cosine,
    Linear,
    Constant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimConfig {
    pub learning_rate: f64,
    pub schedule: Schedule,
    pub warmup_steps: u32,
    pub weight_decay: f64,
    pub grad_accumulation_steps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainParams {
    pub max_steps: Option<u64>,
    pub epochs: Option<u32>,
    pub batch_size: u32,
    pub max_seq_len: u32,
    pub save_every: u64,
    pub packing: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgoConfig {
    pub beta: Option<f64>,
    pub label_smoothing: Option<f64>,
    pub group_size: Option<u32>,
    pub kl_coef: Option<f64>,
    #[serde(default)]
    pub completion_only_loss: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutput {
    pub adapter_path: PathBuf,
    pub resume_from: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub version: u32,
    pub run_id: Option<Uuid>,
    pub train_mode: TrainMode,
    pub model: String,
    pub backend: BackendConfig,
    pub data: DataSpec,
    pub adapter: AdapterConfig,
    pub optim: OptimConfig,
    pub train: TrainParams,
    #[serde(default)]
    pub algo: AlgoConfig,
    pub output: RunOutput,
}

impl RunConfig {
    pub const SUPPORTED_VERSION: u32 = 1;

    pub fn from_yaml_str(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }

    pub fn to_yaml_string(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }
}
