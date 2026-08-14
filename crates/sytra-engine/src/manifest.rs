use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::store::ExpertKey;

pub const RUNTIME_MANIFEST: &str = ".sytra-runtime.json";
pub const CURRENT_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelFamily {
    KimiK27,
    DeepseekV3,
    Qwen3Moe,
    Qwen2Moe,
    Mixtral,
    Olmoe,
    Dbrx,
    GraniteMoe,
    Arctic,
    MinimaxMoe,
    GlmMoe,
    KimiK3,
    Inkling,
    /// Storage/index support for an unrecognized discrete-expert model. It
    /// can be inspected and benchmarked but never unlocks token serving.
    #[default]
    Generic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExpertTensorLayout {
    /// Each expert owns separately named projection tensors.
    #[default]
    Discrete,
    /// Tensor axis zero is the expert dimension.
    StackedAxis0,
    /// Expert rows are concatenated along tensor axis zero (DBRX-style).
    MergedRows,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ActivationKind {
    #[default]
    Silu,
    Gelu,
    GeluTanh,
    Relu,
    Relu2,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RouterScoreKind {
    #[default]
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterContract {
    #[serde(default)]
    pub score: RouterScoreKind,
    #[serde(default)]
    pub normalize_selected: bool,
    #[serde(default = "one_f32")]
    pub scaling_factor: f32,
    #[serde(default)]
    pub correction_bias: bool,
    #[serde(default = "one_u32")]
    pub groups: u32,
    #[serde(default = "one_u32")]
    pub selected_groups: u32,
}

impl Default for RouterContract {
    fn default() -> Self {
        Self {
            score: RouterScoreKind::Softmax,
            normalize_selected: false,
            scaling_factor: 1.0,
            correction_bias: false,
            groups: 1,
            selected_groups: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct AttentionContract {
    #[serde(default)]
    pub heads: u32,
    #[serde(default)]
    pub kv_heads: u32,
    #[serde(default)]
    pub head_dim: u32,
    #[serde(default)]
    pub q_lora_rank: u32,
    #[serde(default)]
    pub kv_lora_rank: u32,
    #[serde(default)]
    pub qk_nope_head_dim: u32,
    #[serde(default)]
    pub qk_rope_head_dim: u32,
    #[serde(default)]
    pub value_head_dim: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct QuantizationContract {
    #[serde(default)]
    pub bits: u8,
    #[serde(default)]
    pub group_size: u32,
    #[serde(default)]
    pub symmetric: bool,
    #[serde(default)]
    pub scale_dtype: Option<String>,
}

const fn one_u32() -> u32 {
    1
}

const fn one_f32() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    Standard,
    Mla,
    SlidingWindow,
    Hybrid,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RouterSemantics {
    TopKWeighted,
    TopKNormalized,
    GroupLimitedTopK,
    TopKSoftmax,
    TopKSigmoid,
    NoAuxTc,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    F32,
    F16,
    Bf16,
    Int8,
    Int4Group,
    /// compressed-tensors `pack-quantized`, symmetric INT4, group size 32.
    PackedInt4Group32,
    Fp8E4m3,
    Nvfp4,
    Mxfp4,
    Gguf,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArchitectureContract {
    /// Stable Sytra adapter identifier, not an executable command.
    pub adapter: String,
    pub model_type: String,
    pub attention: AttentionKind,
    pub router: RouterSemantics,
    pub expert_format: WeightFormat,
    #[serde(default)]
    pub family: ModelFamily,
    #[serde(default)]
    pub expert_layout: ExpertTensorLayout,
    #[serde(default)]
    pub activation: ActivationKind,
    #[serde(default)]
    pub router_config: RouterContract,
    #[serde(default)]
    pub attention_config: AttentionContract,
    #[serde(default)]
    pub quantization: QuantizationContract,
    #[serde(default)]
    pub hidden_size: u32,
    #[serde(default)]
    pub expert_intermediate_size: u32,
    /// Exact decoder layers that contain routed experts. This supports dense
    /// prefixes and hybrid dense/MoE architectures without guessing a cadence.
    #[serde(default)]
    pub moe_layers: Vec<u32>,
    pub num_layers: u32,
    pub experts_per_layer: u32,
    pub experts_per_token: u32,
    /// Must be true only after reference-logit/token tests pass.
    #[serde(default)]
    pub forward_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorSegment {
    /// Original tensor name used by the architecture adapter.
    pub tensor: String,
    /// SafeTensors dtype (for example `I32` or `BF16`).
    #[serde(default)]
    pub dtype: Option<String>,
    /// Logical tensor shape before any packed representation.
    #[serde(default)]
    pub shape: Vec<u64>,
    /// Relative path under the model root.
    pub shard: PathBuf,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertLocation {
    pub layer: u32,
    pub expert: u32,
    pub segments: Vec<TensorSegment>,
}

impl ExpertLocation {
    pub fn key(&self) -> ExpertKey {
        ExpertKey::new(self.layer, self.expert)
    }

    pub fn byte_len(&self) -> u64 {
        self.segments.iter().map(|segment| segment.length).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpertStorageManifest {
    /// True when every expert's matrices occupy one contiguous byte range.
    pub contiguous_experts: bool,
    pub experts: Vec<ExpertLocation>,
    /// Non-routed tensors remain individually addressable so architectures
    /// whose dense backbone exceeds RAM can stream layer bundles as well.
    #[serde(default)]
    pub dense_tensors: Vec<TensorSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeManifest {
    pub schema_version: u32,
    pub architecture: ArchitectureContract,
    /// Total non-routed bytes. Individual dense tensors remain streamable.
    pub dense_bytes: u64,
    pub storage: ExpertStorageManifest,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("could not read runtime manifest {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("runtime manifest {path} is invalid JSON: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("unsupported runtime manifest schema {0}")]
    Schema(u32),
    #[error("invalid architecture contract: {0}")]
    Contract(String),
    #[error("invalid expert location: {0}")]
    Location(String),
    #[error("expert shard is missing or too small: {0}")]
    Shard(String),
}

impl RuntimeManifest {
    pub fn load(model_root: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let path = model_root.as_ref().join(RUNTIME_MANIFEST);
        let bytes = fs::read(&path).map_err(|source| ManifestError::Read {
            path: path.clone(),
            source,
        })?;
        let manifest = serde_json::from_slice(&bytes).map_err(|source| ManifestError::Json {
            path: path.clone(),
            source,
        })?;
        Ok(manifest)
    }

    pub fn validate(&self, model_root: impl AsRef<Path>) -> Result<(), ManifestError> {
        if self.schema_version != CURRENT_SCHEMA {
            return Err(ManifestError::Schema(self.schema_version));
        }
        let contract = &self.architecture;
        if contract.adapter.trim().is_empty() || contract.model_type.trim().is_empty() {
            return Err(ManifestError::Contract(
                "adapter and model_type must be non-empty".into(),
            ));
        }
        if contract.num_layers == 0
            || contract.experts_per_layer == 0
            || contract.experts_per_token == 0
            || contract.experts_per_token > contract.experts_per_layer
        {
            return Err(ManifestError::Contract(
                "layer/expert counts and top-k must be positive and consistent".into(),
            ));
        }
        if contract.hidden_size > 0 && contract.expert_intermediate_size == 0 {
            return Err(ManifestError::Contract(
                "expert_intermediate_size is required when the extended contract is present".into(),
            ));
        }
        if contract.router_config.groups == 0
            || contract.router_config.selected_groups == 0
            || contract.router_config.selected_groups > contract.router_config.groups
            || contract.experts_per_layer % contract.router_config.groups != 0
            || !contract.router_config.scaling_factor.is_finite()
            || contract.router_config.scaling_factor <= 0.0
        {
            return Err(ManifestError::Contract(
                "router groups and scaling factor are invalid".into(),
            ));
        }
        if self.storage.experts.is_empty() {
            return Err(ManifestError::Location(
                "the expert index cannot be empty".into(),
            ));
        }

        let root = model_root.as_ref();
        let mut seen = HashSet::new();
        let declared_moe_layers: HashSet<u32> = contract.moe_layers.iter().copied().collect();
        if declared_moe_layers.len() != contract.moe_layers.len()
            || declared_moe_layers
                .iter()
                .any(|layer| *layer >= contract.num_layers)
        {
            return Err(ManifestError::Contract(
                "moe_layers contains duplicate or out-of-range layers".into(),
            ));
        }
        let mut indexed_layers = HashSet::new();
        for location in &self.storage.experts {
            if location.layer >= contract.num_layers
                || location.expert >= contract.experts_per_layer
                || location.segments.is_empty()
            {
                return Err(ManifestError::Location(format!(
                    "layer {} expert {} is outside the contract or empty",
                    location.layer, location.expert
                )));
            }
            if !seen.insert(location.key()) {
                return Err(ManifestError::Location(format!(
                    "duplicate layer {} expert {}",
                    location.layer, location.expert
                )));
            }
            indexed_layers.insert(location.layer);
            let mut tensors = HashSet::new();
            for segment in &location.segments {
                if segment.tensor.trim().is_empty()
                    || segment.length == 0
                    || !tensors.insert(segment.tensor.as_str())
                {
                    return Err(ManifestError::Location(format!(
                        "layer {} expert {} has an empty/duplicate tensor segment",
                        location.layer, location.expert
                    )));
                }
                if !safe_relative_path(&segment.shard) {
                    return Err(ManifestError::Location(format!(
                        "unsafe shard path {}",
                        segment.shard.display()
                    )));
                }
                let shard = root.join(&segment.shard);
                let required = segment.offset.checked_add(segment.length).ok_or_else(|| {
                    ManifestError::Location(format!(
                        "byte range overflow for {}",
                        segment.shard.display()
                    ))
                })?;
                let actual = fs::metadata(&shard)
                    .map_err(|_| ManifestError::Shard(shard.display().to_string()))?
                    .len();
                if required > actual {
                    return Err(ManifestError::Shard(format!(
                        "{} needs {} bytes but has {}",
                        shard.display(),
                        required,
                        actual
                    )));
                }
            }
        }
        if !declared_moe_layers.is_empty() && indexed_layers != declared_moe_layers {
            return Err(ManifestError::Location(
                "indexed expert layers do not exactly match architecture.moe_layers".into(),
            ));
        }
        for layer in &indexed_layers {
            let count = (0..contract.experts_per_layer)
                .filter(|expert| seen.contains(&ExpertKey::new(*layer, *expert)))
                .count();
            if count != contract.experts_per_layer as usize {
                return Err(ManifestError::Location(format!(
                    "layer {layer} indexes {count} of {} routed experts",
                    contract.experts_per_layer
                )));
            }
            let reference = self
                .storage
                .experts
                .iter()
                .find(|location| location.layer == *layer && location.expert == 0)
                .map(tensor_signature)
                .expect("complete layer contains expert zero");
            for expert in 1..contract.experts_per_layer {
                let signature = self
                    .storage
                    .experts
                    .iter()
                    .find(|location| location.layer == *layer && location.expert == expert)
                    .map(tensor_signature)
                    .expect("complete layer contains every expert");
                if signature != reference {
                    return Err(ManifestError::Location(format!(
                        "layer {layer} expert {expert} has a different tensor signature than expert 0"
                    )));
                }
            }
        }
        let mut dense_names = HashSet::new();
        let mut dense_total = 0_u64;
        for segment in &self.storage.dense_tensors {
            if segment.tensor.trim().is_empty()
                || segment.length == 0
                || !dense_names.insert(segment.tensor.as_str())
                || !safe_relative_path(&segment.shard)
            {
                return Err(ManifestError::Location(
                    "dense tensor index contains an invalid or duplicate entry".into(),
                ));
            }
            let required = segment
                .offset
                .checked_add(segment.length)
                .ok_or_else(|| ManifestError::Location("dense byte range overflow".into()))?;
            let shard = root.join(&segment.shard);
            let actual = fs::metadata(&shard)
                .map_err(|_| ManifestError::Shard(shard.display().to_string()))?
                .len();
            if required > actual {
                return Err(ManifestError::Shard(format!(
                    "{} needs {} bytes but has {}",
                    shard.display(),
                    required,
                    actual
                )));
            }
            dense_total = dense_total
                .checked_add(segment.length)
                .ok_or_else(|| ManifestError::Location("dense byte count overflow".into()))?;
        }
        if !self.storage.dense_tensors.is_empty() && dense_total != self.dense_bytes {
            return Err(ManifestError::Contract(format!(
                "dense tensor index contains {dense_total} bytes but dense_bytes is {}",
                self.dense_bytes
            )));
        }
        Ok(())
    }

    pub fn expert_bytes(&self) -> u64 {
        self.storage
            .experts
            .iter()
            .map(ExpertLocation::byte_len)
            .sum()
    }
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn tensor_signature(location: &ExpertLocation) -> Vec<(Option<String>, Vec<u64>, u64)> {
    let mut signature: Vec<_> = location
        .segments
        .iter()
        .map(|segment| (segment.dtype.clone(), segment.shape.clone(), segment.length))
        .collect();
    signature.sort();
    signature
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest() -> RuntimeManifest {
        RuntimeManifest {
            schema_version: CURRENT_SCHEMA,
            architecture: ArchitectureContract {
                adapter: "sytra-test".into(),
                model_type: "test_moe".into(),
                attention: AttentionKind::Standard,
                router: RouterSemantics::TopKWeighted,
                expert_format: WeightFormat::Int4Group,
                family: ModelFamily::Generic,
                expert_layout: ExpertTensorLayout::Discrete,
                activation: ActivationKind::Silu,
                router_config: RouterContract::default(),
                attention_config: AttentionContract::default(),
                quantization: QuantizationContract::default(),
                hidden_size: 0,
                expert_intermediate_size: 0,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 2,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: 16,
            storage: ExpertStorageManifest {
                contiguous_experts: true,
                experts: vec![
                    ExpertLocation {
                        layer: 0,
                        expert: 0,
                        segments: vec![TensorSegment {
                            tensor: "down_proj".into(),
                            dtype: None,
                            shape: vec![],
                            shard: "experts.bin".into(),
                            offset: 0,
                            length: 4,
                        }],
                    },
                    ExpertLocation {
                        layer: 0,
                        expert: 1,
                        segments: vec![TensorSegment {
                            tensor: "down_proj_1".into(),
                            dtype: None,
                            shape: vec![],
                            shard: "experts.bin".into(),
                            offset: 4,
                            length: 4,
                        }],
                    },
                ],
                dense_tensors: vec![],
            },
        }
    }

    #[test]
    fn path_validation_rejects_escape() {
        assert!(safe_relative_path(Path::new("weights/expert.bin")));
        assert!(!safe_relative_path(Path::new("../expert.bin")));
        assert!(!safe_relative_path(Path::new("C:\\expert.bin")));
    }

    #[test]
    fn contract_requires_real_top_k() {
        let mut manifest = valid_manifest();
        manifest.architecture.experts_per_token = 3;
        let error = manifest
            .validate(std::env::temp_dir())
            .expect_err("invalid top-k must fail before file checks");
        assert!(matches!(error, ManifestError::Contract(_)));
    }
}
