use crate::{
    manifest::{
        ActivationKind, AttentionKind, ExpertTensorLayout, ModelFamily, RouterSemantics,
        RuntimeManifest, WeightFormat,
    },
    scheduler::{PreparedLayer, RoutingBatch},
};
use serde::Serialize;
use thiserror::Error;

/// Static capabilities compiled into this Sytra engine build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AdapterDescriptor {
    pub id: &'static str,
    pub model_type_prefixes: &'static [&'static str],
    pub families: &'static [ModelFamily],
    pub attention: &'static [AttentionKind],
    pub routers: &'static [RouterSemantics],
    pub expert_formats: &'static [WeightFormat],
    pub expert_layouts: &'static [ExpertTensorLayout],
    pub activations: &'static [ActivationKind],
    /// A compiled CPU reference exists for the family semantics. This does
    /// not imply that a complete layer or token-serving kernel is ready.
    pub reference_math: bool,
    /// Storage-only profiles can index/stream unknown models but can never
    /// become serving-capable without promotion to an exact compiled family.
    pub storage_only: bool,
    /// True only when a complete forward kernel is linked. Serving still
    /// requires a checkpoint-bound oracle suite to pass at startup.
    pub forward_kernel: bool,
}

/// Shared input passed from the streaming scheduler to a model adapter.
///
/// The adapter receives the original route vectors plus unique prepared
/// experts. It is responsible for the model's exact activation, quantization,
/// accumulation, and router weighting semantics.
#[derive(Debug)]
pub struct KernelInput<'a> {
    pub hidden_size: usize,
    pub positions: usize,
    pub hidden: &'a [f32],
    pub routes: &'a RoutingBatch,
    pub prepared: &'a PreparedLayer,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KernelOutput {
    pub hidden: Vec<f32>,
}

pub trait ArchitectureKernel: Send + Sync {
    fn descriptor(&self) -> &'static AdapterDescriptor;
    fn forward_layer(&self, input: KernelInput<'_>) -> Result<KernelOutput, AdapterError>;
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("adapter {0} is not compiled into this Sytra engine")]
    Unknown(String),
    #[error("adapter contract mismatch: {0}")]
    Contract(String),
    #[error("adapter {0} has no token-verified forward kernel in this build")]
    KernelUnavailable(String),
    #[error("adapter kernel failed: {0}")]
    Kernel(String),
}

static GLM52: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-glm52",
    model_type_prefixes: &["glm"],
    families: &[ModelFamily::GlmMoe],
    attention: &[AttentionKind::Mla, AttentionKind::Custom],
    routers: &[
        RouterSemantics::TopKWeighted,
        RouterSemantics::GroupLimitedTopK,
    ],
    expert_formats: &[
        WeightFormat::Int4Group,
        WeightFormat::Int8,
        WeightFormat::Custom,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: false,
    storage_only: false,
    forward_kernel: false,
};

static KIMI_K3: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-kimi-k3",
    model_type_prefixes: &["kimi_k3", "kimi-k3"],
    families: &[ModelFamily::KimiK3],
    attention: &[AttentionKind::Mla, AttentionKind::Custom],
    routers: &[
        RouterSemantics::TopKWeighted,
        RouterSemantics::GroupLimitedTopK,
    ],
    expert_formats: &[WeightFormat::Mxfp4],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: false,
    storage_only: false,
    forward_kernel: false,
};

static KIMI_K27_CODE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-kimi-k2.7-code",
    // The public checkpoint is a KimiK25 multimodal wrapper around a kimi_k2
    // DeepSeek-V3-compatible text tower. Both types are validated from
    // config.json by `KimiK27Config`; the manifest records the outer type.
    model_type_prefixes: &["kimi_k25"],
    families: &[ModelFamily::KimiK27],
    attention: &[AttentionKind::Mla],
    routers: &[RouterSemantics::GroupLimitedTopK, RouterSemantics::NoAuxTc],
    expert_formats: &[WeightFormat::PackedInt4Group32],
    expert_layouts: &[ExpertTensorLayout::Discrete],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    // The complete text forward path is linked. A checkpoint-specific
    // reference-logit and teacher-forced oracle is still mandatory at runtime.
    forward_kernel: true,
};

static INKLING: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-inkling",
    model_type_prefixes: &["inkling"],
    families: &[ModelFamily::Inkling],
    attention: &[AttentionKind::Standard, AttentionKind::Custom],
    routers: &[
        RouterSemantics::TopKWeighted,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Int4Group,
        WeightFormat::Bf16,
        WeightFormat::Custom,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu, ActivationKind::Gelu],
    reference_math: false,
    storage_only: false,
    forward_kernel: false,
};

static DEEPSEEK_V3: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-deepseek-v3",
    model_type_prefixes: &["deepseek_v2", "deepseek_v3"],
    families: &[ModelFamily::DeepseekV3],
    attention: &[AttentionKind::Mla],
    routers: &[RouterSemantics::NoAuxTc, RouterSemantics::GroupLimitedTopK],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Fp8E4m3,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    forward_kernel: false,
};

static QWEN3_MOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-qwen3-moe",
    model_type_prefixes: &["qwen3_moe", "qwen3_next"],
    families: &[ModelFamily::Qwen3Moe],
    attention: &[
        AttentionKind::Standard,
        AttentionKind::SlidingWindow,
        AttentionKind::Hybrid,
    ],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Fp8E4m3,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    // The exact BF16/packed-INT4 qwen3_moe (not qwen3_next) subset is checked dynamically.
    forward_kernel: true,
};

static QWEN2_MOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-qwen2-moe",
    model_type_prefixes: &["qwen2_moe"],
    families: &[ModelFamily::Qwen2Moe],
    attention: &[AttentionKind::Standard, AttentionKind::SlidingWindow],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    // The exact BF16/packed-INT4 discrete qwen2_moe subset is checked dynamically.
    forward_kernel: true,
};

static MIXTRAL: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-mixtral",
    model_type_prefixes: &["mixtral"],
    families: &[ModelFamily::Mixtral],
    attention: &[AttentionKind::Standard, AttentionKind::SlidingWindow],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    // The exact BF16/packed-INT4 discrete or stacked-axis0 subset is checked dynamically by
    // `MixtralConfig::validate_manifest`; broader storage contracts remain
    // rejected by runtime construction.
    forward_kernel: true,
};

static OLMOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-olmoe",
    model_type_prefixes: &["olmoe"],
    families: &[ModelFamily::Olmoe],
    attention: &[AttentionKind::Standard],
    routers: &[RouterSemantics::TopKSoftmax, RouterSemantics::TopKWeighted],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    // The exact BF16/packed-INT4 standard-attention subset is checked dynamically.
    forward_kernel: true,
};

static DBRX: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-dbrx",
    model_type_prefixes: &["dbrx"],
    families: &[ModelFamily::Dbrx],
    attention: &[AttentionKind::Standard],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
    ],
    expert_layouts: &[
        ExpertTensorLayout::MergedRows,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu, ActivationKind::Gelu],
    reference_math: true,
    storage_only: false,
    forward_kernel: false,
};

static GRANITE_MOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-granite-moe",
    model_type_prefixes: &["granitemoe", "granite_moe"],
    families: &[ModelFamily::GraniteMoe],
    attention: &[AttentionKind::Standard],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::F32,
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu, ActivationKind::Gelu],
    reference_math: true,
    storage_only: false,
    // Exact GraniteMoE multipliers, selected-top-k softmax, stacked experts,
    // standard GQA, and logits scaling are handled by the standard executor.
    forward_kernel: true,
};

static ARCTIC: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-arctic",
    model_type_prefixes: &["arctic"],
    families: &[ModelFamily::Arctic],
    attention: &[AttentionKind::Standard],
    routers: &[
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKNormalized,
    ],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Int4Group,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu, ActivationKind::Gelu],
    reference_math: true,
    storage_only: false,
    forward_kernel: false,
};

static MINIMAX_MOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-minimax-moe",
    model_type_prefixes: &["minimax"],
    families: &[ModelFamily::MinimaxMoe],
    attention: &[
        AttentionKind::Standard,
        AttentionKind::Mla,
        AttentionKind::Hybrid,
    ],
    routers: &[RouterSemantics::TopKSoftmax, RouterSemantics::TopKSigmoid],
    expert_formats: &[
        WeightFormat::Bf16,
        WeightFormat::F16,
        WeightFormat::Fp8E4m3,
        WeightFormat::Int4Group,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
    ],
    activations: &[ActivationKind::Silu],
    reference_math: true,
    storage_only: false,
    forward_kernel: false,
};

static GENERIC_MOE: AdapterDescriptor = AdapterDescriptor {
    id: "sytra-generic-moe",
    model_type_prefixes: &[],
    families: &[ModelFamily::Generic],
    attention: &[
        AttentionKind::Standard,
        AttentionKind::Mla,
        AttentionKind::SlidingWindow,
        AttentionKind::Hybrid,
        AttentionKind::Custom,
    ],
    routers: &[
        RouterSemantics::TopKWeighted,
        RouterSemantics::TopKNormalized,
        RouterSemantics::TopKSoftmax,
        RouterSemantics::TopKSigmoid,
        RouterSemantics::GroupLimitedTopK,
        RouterSemantics::NoAuxTc,
        RouterSemantics::Custom,
    ],
    expert_formats: &[
        WeightFormat::F32,
        WeightFormat::F16,
        WeightFormat::Bf16,
        WeightFormat::Int8,
        WeightFormat::Int4Group,
        WeightFormat::PackedInt4Group32,
        WeightFormat::Fp8E4m3,
        WeightFormat::Nvfp4,
        WeightFormat::Mxfp4,
        WeightFormat::Custom,
    ],
    expert_layouts: &[
        ExpertTensorLayout::Discrete,
        ExpertTensorLayout::StackedAxis0,
        ExpertTensorLayout::MergedRows,
    ],
    activations: &[
        ActivationKind::Silu,
        ActivationKind::Gelu,
        ActivationKind::GeluTanh,
        ActivationKind::Relu,
        ActivationKind::Relu2,
    ],
    reference_math: false,
    storage_only: true,
    forward_kernel: false,
};

pub fn compiled_adapter(id: &str) -> Option<&'static AdapterDescriptor> {
    compiled_adapters()
        .into_iter()
        .find(|descriptor| descriptor.id == id)
}

pub fn compiled_adapters() -> Vec<&'static AdapterDescriptor> {
    vec![
        &GLM52,
        &KIMI_K27_CODE,
        &KIMI_K3,
        &INKLING,
        &DEEPSEEK_V3,
        &QWEN3_MOE,
        &QWEN2_MOE,
        &MIXTRAL,
        &OLMOE,
        &DBRX,
        &GRANITE_MOE,
        &ARCTIC,
        &MINIMAX_MOE,
        &GENERIC_MOE,
    ]
}

pub fn validate_compiled_contract(
    manifest: &RuntimeManifest,
) -> Result<&'static AdapterDescriptor, AdapterError> {
    let contract = &manifest.architecture;
    let descriptor = compiled_adapter(&contract.adapter)
        .ok_or_else(|| AdapterError::Unknown(contract.adapter.clone()))?;
    let model_type = contract.model_type.to_ascii_lowercase();
    if !descriptor.storage_only
        && !descriptor
            .model_type_prefixes
            .iter()
            .any(|prefix| model_type.starts_with(prefix))
    {
        return Err(AdapterError::Contract(format!(
            "{} does not accept model_type {}",
            descriptor.id, contract.model_type
        )));
    }
    if !descriptor.families.contains(&contract.family) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} family contracts",
            descriptor.id, contract.family
        )));
    }
    if !descriptor.attention.contains(&contract.attention) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} attention",
            descriptor.id, contract.attention
        )));
    }
    if !descriptor.routers.contains(&contract.router) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} routing",
            descriptor.id, contract.router
        )));
    }
    if !descriptor.expert_formats.contains(&contract.expert_format) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} expert weights",
            descriptor.id, contract.expert_format
        )));
    }
    if !descriptor.expert_layouts.contains(&contract.expert_layout) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} expert tensor layout",
            descriptor.id, contract.expert_layout
        )));
    }
    if !descriptor.activations.contains(&contract.activation) {
        return Err(AdapterError::Contract(format!(
            "{} does not accept {:?} expert activation",
            descriptor.id, contract.activation
        )));
    }
    Ok(descriptor)
}

/// Validate the narrower contract handled by a complete forward executor.
/// Descriptors may accept additional formats for storage/indexing, so callers
/// must use this predicate rather than treating `forward_kernel` as applying
/// to every format listed by the descriptor.
pub fn validate_forward_contract(
    manifest: &RuntimeManifest,
) -> Result<&'static AdapterDescriptor, AdapterError> {
    let descriptor = validate_compiled_contract(manifest)?;
    if !descriptor.forward_kernel {
        return Err(AdapterError::KernelUnavailable(descriptor.id.into()));
    }
    if matches!(
        descriptor.id,
        "sytra-mixtral"
            | "sytra-qwen3-moe"
            | "sytra-qwen2-moe"
            | "sytra-olmoe"
            | "sytra-granite-moe"
    ) {
        let contract = &manifest.architecture;
        let exact_expert_format = contract.expert_format == WeightFormat::Bf16
            || (descriptor.id == "sytra-granite-moe"
                && contract.expert_format == WeightFormat::F32)
            || (contract.expert_format == WeightFormat::PackedInt4Group32
                && contract.quantization.bits == 4
                && contract.quantization.group_size == 32
                && contract.quantization.symmetric
                && contract
                    .quantization
                    .scale_dtype
                    .as_deref()
                    .is_some_and(|dtype| dtype.eq_ignore_ascii_case("BF16")));
        if !exact_expert_format
            || (descriptor.id == "sytra-mixtral"
                && !matches!(
                    contract.expert_layout,
                    ExpertTensorLayout::Discrete | ExpertTensorLayout::StackedAxis0
                ))
            || (descriptor.id == "sytra-qwen3-moe"
                && !matches!(
                    contract.expert_layout,
                    ExpertTensorLayout::Discrete | ExpertTensorLayout::StackedAxis0
                ))
            || (descriptor.id == "sytra-qwen2-moe"
                && contract.expert_layout != ExpertTensorLayout::Discrete)
            || (descriptor.id == "sytra-olmoe"
                && !matches!(
                    contract.expert_layout,
                    ExpertTensorLayout::Discrete | ExpertTensorLayout::StackedAxis0
                ))
            || (descriptor.id == "sytra-granite-moe"
                && contract.expert_layout != ExpertTensorLayout::StackedAxis0)
            || !matches!(
                contract.attention,
                AttentionKind::Standard | AttentionKind::SlidingWindow
            )
            || (descriptor.id == "sytra-qwen3-moe" && contract.model_type != "qwen3_moe")
            || (descriptor.id == "sytra-qwen2-moe" && contract.model_type != "qwen2_moe")
            || (descriptor.id == "sytra-olmoe" && contract.model_type != "olmoe")
            || (descriptor.id == "sytra-granite-moe" && contract.model_type != "granitemoe")
            || contract.activation != ActivationKind::Silu
            || !matches!(
                contract.router,
                RouterSemantics::TopKSoftmax | RouterSemantics::TopKNormalized
            ) && !(descriptor.id == "sytra-olmoe"
                && contract.router == RouterSemantics::TopKWeighted)
        {
            return Err(AdapterError::KernelUnavailable(format!(
                "{} requires its exact BF16 or packed symmetric INT4/BF16 group-32 standard-GQA forward subset",
                descriptor.id
            )));
        }
    }
    Ok(descriptor)
}

#[cfg(test)]
mod tests {
    use crate::manifest::{
        ArchitectureContract, ExpertStorageManifest, RuntimeManifest, CURRENT_SCHEMA,
    };

    use super::*;

    fn kimi_manifest() -> RuntimeManifest {
        RuntimeManifest {
            schema_version: CURRENT_SCHEMA,
            architecture: ArchitectureContract {
                adapter: "sytra-kimi-k3".into(),
                model_type: "kimi_k3".into(),
                attention: AttentionKind::Mla,
                router: RouterSemantics::TopKWeighted,
                expert_format: WeightFormat::Mxfp4,
                family: crate::manifest::ModelFamily::KimiK3,
                expert_layout: crate::manifest::ExpertTensorLayout::Discrete,
                activation: crate::manifest::ActivationKind::Silu,
                router_config: crate::manifest::RouterContract::default(),
                attention_config: crate::manifest::AttentionContract::default(),
                quantization: crate::manifest::QuantizationContract::default(),
                hidden_size: 0,
                expert_intermediate_size: 0,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 8,
                experts_per_token: 2,
                forward_verified: true,
            },
            dense_bytes: 0,
            storage: ExpertStorageManifest {
                contiguous_experts: true,
                experts: vec![],
                dense_tensors: vec![],
            },
        }
    }

    #[test]
    fn model_metadata_cannot_turn_on_an_uncompiled_kernel() {
        let descriptor = validate_compiled_contract(&kimi_manifest()).unwrap();
        assert!(!descriptor.forward_kernel);
    }

    #[test]
    fn kimi_k2_cannot_claim_kimi_k3_contract() {
        let mut manifest = kimi_manifest();
        manifest.architecture.model_type = "kimi_k2".into();
        assert!(matches!(
            validate_compiled_contract(&manifest),
            Err(AdapterError::Contract(_))
        ));
    }

    #[test]
    fn kimi_k27_requires_its_exact_packed_int4_contract() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-kimi-k2.7-code".into();
        manifest.architecture.model_type = "kimi_k25".into();
        manifest.architecture.family = ModelFamily::KimiK27;
        manifest.architecture.router = RouterSemantics::GroupLimitedTopK;
        manifest.architecture.expert_format = WeightFormat::PackedInt4Group32;
        let descriptor = validate_compiled_contract(&manifest).unwrap();
        assert_eq!(descriptor.id, "sytra-kimi-k2.7-code");
        assert!(descriptor.forward_kernel);

        manifest.architecture.expert_format = WeightFormat::Int4Group;
        assert!(matches!(
            validate_compiled_contract(&manifest),
            Err(AdapterError::Contract(_))
        ));
    }

    #[test]
    fn general_profiles_reject_cross_family_contracts() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-qwen3-moe".into();
        manifest.architecture.model_type = "qwen3_moe".into();
        manifest.architecture.family = ModelFamily::Qwen3Moe;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKSoftmax;
        manifest.architecture.expert_format = WeightFormat::Bf16;
        assert_eq!(
            validate_compiled_contract(&manifest).unwrap().id,
            "sytra-qwen3-moe"
        );
        manifest.architecture.family = ModelFamily::Mixtral;
        assert!(matches!(
            validate_compiled_contract(&manifest),
            Err(AdapterError::Contract(_))
        ));
    }

    #[test]
    fn mixtral_forward_subset_accepts_only_exact_bf16_or_packed_int4_experts() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-mixtral".into();
        manifest.architecture.model_type = "mixtral".into();
        manifest.architecture.family = ModelFamily::Mixtral;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKSoftmax;
        manifest.architecture.router_config.normalize_selected = true;
        manifest.architecture.expert_format = WeightFormat::Bf16;
        assert!(validate_forward_contract(&manifest).is_ok());
        manifest.architecture.expert_format = WeightFormat::F16;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.expert_format = WeightFormat::PackedInt4Group32;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.quantization = crate::manifest::QuantizationContract {
            bits: 4,
            group_size: 32,
            symmetric: true,
            scale_dtype: Some("bf16".into()),
        };
        assert!(validate_forward_contract(&manifest).is_ok());
    }

    #[test]
    fn qwen3_forward_subset_rejects_next_hybrid_and_non_exact_quantization() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-qwen3-moe".into();
        manifest.architecture.model_type = "qwen3_moe".into();
        manifest.architecture.family = ModelFamily::Qwen3Moe;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKSoftmax;
        manifest.architecture.expert_format = WeightFormat::Bf16;
        manifest.architecture.expert_layout = ExpertTensorLayout::StackedAxis0;
        assert!(validate_forward_contract(&manifest).is_ok());

        manifest.architecture.model_type = "qwen3_next".into();
        manifest.architecture.attention = AttentionKind::Hybrid;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.model_type = "qwen3_moe".into();
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.expert_format = WeightFormat::Fp8E4m3;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.expert_format = WeightFormat::PackedInt4Group32;
        manifest.architecture.quantization = crate::manifest::QuantizationContract {
            bits: 4,
            group_size: 32,
            symmetric: true,
            scale_dtype: Some("BF16".into()),
        };
        assert!(validate_forward_contract(&manifest).is_ok());
    }

    #[test]
    fn qwen2_forward_subset_requires_discrete_exact_experts() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-qwen2-moe".into();
        manifest.architecture.model_type = "qwen2_moe".into();
        manifest.architecture.family = ModelFamily::Qwen2Moe;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKSoftmax;
        manifest.architecture.expert_format = WeightFormat::Bf16;
        manifest.architecture.expert_layout = ExpertTensorLayout::Discrete;
        assert!(validate_forward_contract(&manifest).is_ok());
        manifest.architecture.expert_layout = ExpertTensorLayout::StackedAxis0;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.expert_layout = ExpertTensorLayout::Discrete;
        manifest.architecture.expert_format = WeightFormat::PackedInt4Group32;
        manifest.architecture.quantization = crate::manifest::QuantizationContract {
            bits: 4,
            group_size: 32,
            symmetric: true,
            scale_dtype: Some("BF16".into()),
        };
        assert!(validate_forward_contract(&manifest).is_ok());
    }

    #[test]
    fn olmoe_forward_subset_accepts_weighted_stacked_exact_experts() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-olmoe".into();
        manifest.architecture.model_type = "olmoe".into();
        manifest.architecture.family = ModelFamily::Olmoe;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKWeighted;
        manifest.architecture.expert_format = WeightFormat::Bf16;
        manifest.architecture.expert_layout = ExpertTensorLayout::StackedAxis0;
        assert!(validate_forward_contract(&manifest).is_ok());
        manifest.architecture.expert_format = WeightFormat::F16;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
        manifest.architecture.expert_format = WeightFormat::PackedInt4Group32;
        manifest.architecture.quantization = crate::manifest::QuantizationContract {
            bits: 4,
            group_size: 32,
            symmetric: true,
            scale_dtype: Some("BF16".into()),
        };
        assert!(validate_forward_contract(&manifest).is_ok());
    }

    #[test]
    fn granite_forward_subset_accepts_exact_stacked_f32_and_bf16_experts() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-granite-moe".into();
        manifest.architecture.model_type = "granitemoe".into();
        manifest.architecture.family = ModelFamily::GraniteMoe;
        manifest.architecture.attention = AttentionKind::Standard;
        manifest.architecture.router = RouterSemantics::TopKSoftmax;
        manifest.architecture.expert_layout = ExpertTensorLayout::StackedAxis0;
        manifest.architecture.expert_format = WeightFormat::F32;
        assert!(validate_forward_contract(&manifest).is_ok());
        manifest.architecture.expert_format = WeightFormat::Bf16;
        assert!(validate_forward_contract(&manifest).is_ok());
        manifest.architecture.expert_layout = ExpertTensorLayout::Discrete;
        assert!(matches!(
            validate_forward_contract(&manifest),
            Err(AdapterError::KernelUnavailable(_))
        ));
    }

    #[test]
    fn generic_profile_is_storage_only_and_never_has_a_forward_kernel() {
        let mut manifest = kimi_manifest();
        manifest.architecture.adapter = "sytra-generic-moe".into();
        manifest.architecture.model_type = "future_moe".into();
        manifest.architecture.family = ModelFamily::Generic;
        let descriptor = validate_compiled_contract(&manifest).unwrap();
        assert!(descriptor.storage_only);
        assert!(!descriptor.forward_kernel);
    }
}
