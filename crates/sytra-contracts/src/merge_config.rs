use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeMethod {
    Linear,
    Slerp,
    Ties,
    DareTies,
    TaskArithmetic,
    Passthrough,
    Moe,
}

impl MergeMethod {
    pub fn is_task_vector(&self) -> bool {
        matches!(
            self,
            MergeMethod::Ties | MergeMethod::DareTies | MergeMethod::TaskArithmetic
        )
    }

    pub fn allows_cross_architecture(&self) -> bool {
        matches!(self, MergeMethod::Passthrough | MergeMethod::Moe)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelParameters {
    pub weight: Option<f64>,
    pub density: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub model: String,
    #[serde(default)]
    pub parameters: Option<ModelParameters>,
    /// Only present for `passthrough`.
    #[serde(default)]
    pub layer_range: Option<[u32; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenizerSource {
    Named(TokenizerNamed),
    Index(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerNamed {
    Base,
    Union,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub source: TokenizerSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Green,
    Amber,
    Red,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatConfig {
    pub verdict: Verdict,
    pub fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeOutput {
    pub model_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    pub version: u32,
    pub op_id: Option<Uuid>,
    pub merge_method: MergeMethod,
    pub base_model: Option<String>,
    pub dtype: String,
    pub models: Vec<ModelEntry>,
    /// Method-global parameters forwarded verbatim to mergekit (e.g.
    /// slerp's `t`). Without this field the typed round-trip silently
    /// dropped them and slerp merges failed with "Missing required
    /// parameter t".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    pub tokenizer: TokenizerConfig,
    pub compat: CompatConfig,
    pub output: MergeOutput,
}

impl MergeConfig {
    pub const SUPPORTED_VERSION: u32 = 1;

    pub fn from_yaml_str(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }

    pub fn to_yaml_string(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }
}
