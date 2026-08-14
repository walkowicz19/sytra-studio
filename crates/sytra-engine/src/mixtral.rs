//! Exact storage-backed Mixtral text decoder built on the standard MoE core.

use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    apply_standard_rope, bf16_tile_matmul_cpu, decode_float_values, rms_norm, route_topk_logits,
    standard_attention_decode_window, standard_gated_expert_batch_resident, ActivationKind,
    CudaAccelerator, DenseTensorStore, KimiSpeculativeOutput, KimiStepMetrics, ModelFamily, Route,
    RouterSemantics, RoutingBatch, RuntimeManifest, StandardKvEntry, StandardMoeKvState,
    StreamingScheduler, TiledMatmulOutput, WeightFormat,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtralConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(alias = "num_experts")]
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub clip_qkv: Option<f32>,
    #[serde(default = "default_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub mlp_only_layers: Option<Vec<usize>>,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub embedding_multiplier: Option<f32>,
    #[serde(default)]
    pub logits_scaling: Option<f32>,
    #[serde(default)]
    pub residual_multiplier: Option<f32>,
    #[serde(default)]
    pub attention_multiplier: Option<f32>,
}

const fn default_sparse_step() -> usize {
    1
}

const fn default_true() -> bool {
    true
}

impl MixtralConfig {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, MixtralError> {
        let path = root.as_ref().join("config.json");
        let bytes = fs::read(&path).map_err(|error| MixtralError::Contract(error.to_string()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| MixtralError::Contract(error.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dimension(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn expert_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    pub fn is_qwen3(&self) -> bool {
        self.model_type == "qwen3_moe"
    }

    pub fn is_qwen2(&self) -> bool {
        self.model_type == "qwen2_moe"
    }

    pub fn is_qwen(&self) -> bool {
        self.is_qwen2() || self.is_qwen3()
    }

    pub fn is_olmoe(&self) -> bool {
        self.model_type == "olmoe"
    }

    pub fn is_granite(&self) -> bool {
        self.model_type == "granitemoe"
    }

    fn embedding_multiplier_value(&self) -> f32 {
        self.embedding_multiplier.unwrap_or(1.0)
    }

    fn logits_scaling_value(&self) -> f32 {
        self.logits_scaling.unwrap_or(1.0)
    }

    fn residual_multiplier_value(&self) -> f32 {
        self.residual_multiplier.unwrap_or(1.0)
    }

    fn attention_multiplier_value(&self) -> f32 {
        self.attention_multiplier
            .unwrap_or_else(|| 1.0 / (self.head_dimension() as f32).sqrt())
    }

    pub fn is_moe_layer(&self, layer: usize) -> bool {
        !self.is_qwen()
            || (!self
                .mlp_only_layers
                .as_deref()
                .unwrap_or_default()
                .contains(&layer)
                && (layer + 1).is_multiple_of(self.decoder_sparse_step))
    }

    pub fn rope_theta_value(&self) -> f32 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(serde_json::Value::as_object)
                    .and_then(|parameters| parameters.get("rope_theta"))
                    .and_then(serde_json::Value::as_f64)
                    .map(|value| value as f32)
            })
            .unwrap_or(10_000.0)
    }

    pub fn attention_window(&self) -> Option<usize> {
        if self.is_qwen() && !self.use_sliding_window {
            None
        } else {
            self.sliding_window
        }
    }

    fn has_supported_rope(&self) -> bool {
        self.rope_scaling.is_none()
            && self
                .rope_parameters
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .and_then(|parameters| {
                    parameters
                        .get("rope_type")
                        .or_else(|| parameters.get("type"))
                })
                .and_then(serde_json::Value::as_str)
                .is_none_or(|rope_type| rope_type == "default")
    }

    fn moe_prefix_candidates(&self, layer: usize) -> Vec<String> {
        if self.is_qwen() || self.is_olmoe() {
            vec![format!("model.layers.{layer}.mlp")]
        } else {
            vec![
                format!("model.layers.{layer}.block_sparse_moe"),
                format!("model.layers.{layer}.mlp"),
            ]
        }
    }

    fn resolve_moe_prefix(
        &self,
        layer: usize,
        contains: impl Fn(&str) -> bool,
    ) -> Result<String, MixtralError> {
        let matches: Vec<_> = self
            .moe_prefix_candidates(layer)
            .into_iter()
            .filter(|prefix| {
                self.router_weight_candidates(prefix)
                    .into_iter()
                    .any(|router| {
                        contains(&router) || contains(&router.replace(".weight", ".weight_packed"))
                    })
            })
            .collect();
        match matches.as_slice() {
            [prefix] => Ok(prefix.clone()),
            [] => Err(MixtralError::Contract(format!(
                "layer {layer} has no supported MoE router tensor prefix"
            ))),
            _ => Err(MixtralError::Contract(format!(
                "layer {layer} has ambiguous MoE router tensor prefixes"
            ))),
        }
    }

    fn router_weight_candidates(&self, prefix: &str) -> Vec<String> {
        if self.is_granite() {
            vec![
                format!("{prefix}.router.weight"),
                format!("{prefix}.router.layer.weight"),
            ]
        } else {
            vec![format!("{prefix}.gate.weight")]
        }
    }

    fn resolve_router_weight(
        &self,
        prefix: &str,
        contains: impl Fn(&str) -> bool,
    ) -> Result<String, MixtralError> {
        let matches: Vec<_> = self
            .router_weight_candidates(prefix)
            .into_iter()
            .filter(|name| contains(name) || contains(&name.replace(".weight", ".weight_packed")))
            .collect();
        match matches.as_slice() {
            [name] => Ok(name.clone()),
            [] => Err(MixtralError::Contract(format!(
                "MoE block {prefix} has no exact router tensor"
            ))),
            _ => Err(MixtralError::Contract(format!(
                "MoE block {prefix} has ambiguous router tensors"
            ))),
        }
    }

    fn attention_projection_has_bias(&self, projection: &str) -> bool {
        if self.is_qwen2() {
            matches!(projection, "q_proj" | "k_proj" | "v_proj")
        } else {
            self.attention_bias
        }
    }

    pub fn validate(&self) -> Result<(), MixtralError> {
        if !matches!(
            self.model_type.as_str(),
            "mixtral" | "qwen3_moe" | "qwen2_moe" | "olmoe" | "granitemoe"
        ) || self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.expert_intermediate_size() == 0
            || (self.is_qwen2() && self.shared_expert_intermediate_size.unwrap_or(0) == 0)
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.head_dimension() == 0
            || !self.head_dimension().is_multiple_of(2)
            || (self.is_olmoe()
                && self.num_attention_heads * self.head_dimension() != self.hidden_size)
            || self.num_local_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_local_experts
            || self.max_position_embeddings == 0
            || self.decoder_sparse_step == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || self
                .clip_qkv
                .is_some_and(|clip| !clip.is_finite() || clip <= 0.0)
            || [
                self.embedding_multiplier_value(),
                self.logits_scaling_value(),
                self.residual_multiplier_value(),
                self.attention_multiplier_value(),
            ]
            .into_iter()
            .any(|value| !value.is_finite() || value <= 0.0)
            || !self.rope_theta_value().is_finite()
            || self.rope_theta_value() <= 0.0
            || !self.has_supported_rope()
        {
            return Err(MixtralError::Contract(
                "standard MoE dimensions, RoPE, or normalization are invalid".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_manifest(&self, manifest: &RuntimeManifest) -> Result<(), MixtralError> {
        let architecture = &manifest.architecture;
        let expected_family = if self.is_qwen3() {
            ModelFamily::Qwen3Moe
        } else if self.is_qwen2() {
            ModelFamily::Qwen2Moe
        } else if self.is_olmoe() {
            ModelFamily::Olmoe
        } else if self.is_granite() {
            ModelFamily::GraniteMoe
        } else {
            ModelFamily::Mixtral
        };
        let expected_adapter = if self.is_qwen3() {
            "sytra-qwen3-moe"
        } else if self.is_qwen2() {
            "sytra-qwen2-moe"
        } else if self.is_olmoe() {
            "sytra-olmoe"
        } else if self.is_granite() {
            "sytra-granite-moe"
        } else {
            "sytra-mixtral"
        };
        let exact_packed_contract = architecture.quantization.bits == 4
            && architecture.quantization.group_size == 32
            && architecture.quantization.symmetric
            && architecture
                .quantization
                .scale_dtype
                .as_deref()
                .is_some_and(|dtype| dtype.eq_ignore_ascii_case("BF16"));
        let exact_expert_format = architecture.expert_format == WeightFormat::Bf16
            || (self.is_granite() && architecture.expert_format == WeightFormat::F32)
            || (architecture.expert_format == WeightFormat::PackedInt4Group32
                && exact_packed_contract);
        let has_packed_dense = manifest
            .storage
            .dense_tensors
            .iter()
            .any(|tensor| tensor.tensor.ends_with(".weight_packed"));
        if architecture.adapter != expected_adapter
            || architecture.family != expected_family
            || !exact_expert_format
            || (has_packed_dense && !exact_packed_contract)
            || architecture.hidden_size as usize != self.hidden_size
            || architecture.expert_intermediate_size as usize != self.expert_intermediate_size()
            || architecture.num_layers as usize != self.num_hidden_layers
            || architecture.experts_per_layer as usize != self.num_local_experts
            || architecture.experts_per_token as usize != self.num_experts_per_tok
            || architecture.attention_config.heads as usize != self.num_attention_heads
            || architecture.attention_config.kv_heads as usize != self.num_key_value_heads
            || architecture.attention_config.head_dim as usize != self.head_dimension()
            || !matches!(
                architecture.router,
                RouterSemantics::TopKNormalized
                    | RouterSemantics::TopKSoftmax
                    | RouterSemantics::TopKWeighted
            )
            || architecture.router_config.normalize_selected != self.norm_topk_prob
            || architecture.router_config.scaling_factor != 1.0
        {
            return Err(MixtralError::Contract(
                "runtime manifest does not match the compiled BF16/packed-INT4 standard-MoE path"
                    .into(),
            ));
        }
        let expected_moe_layers: Vec<u32> = (0..self.num_hidden_layers)
            .filter(|layer| self.is_moe_layer(*layer))
            .map(|layer| layer as u32)
            .collect();
        if architecture.moe_layers != expected_moe_layers {
            return Err(MixtralError::Contract(
                "manifest routed layers do not match the decoder sparse cadence".into(),
            ));
        }
        let expected_experts = expected_moe_layers.len() * self.num_local_experts;
        if manifest.storage.experts.len() != expected_experts {
            return Err(MixtralError::Contract(format!(
                "manifest contains {} expert payloads, expected {expected_experts}",
                manifest.storage.experts.len()
            )));
        }
        for expert in &manifest.storage.experts {
            validate_expert_tensors(
                expert,
                self.hidden_size,
                self.expert_intermediate_size(),
                &architecture.expert_format,
            )?;
        }
        let dense: HashMap<_, _> = manifest
            .storage
            .dense_tensors
            .iter()
            .map(|tensor| (tensor.tensor.as_str(), tensor))
            .collect();
        let allow_f32 = self.is_granite() && architecture.expert_format == WeightFormat::F32;
        expect_float(
            &dense,
            "model.embed_tokens.weight",
            &[self.vocab_size, self.hidden_size],
            allow_f32,
        )?;
        expect_float(&dense, "model.norm.weight", &[self.hidden_size], allow_f32)?;
        if !self.tie_word_embeddings {
            expect_matrix(
                &dense,
                "lm_head.weight",
                &[self.vocab_size, self.hidden_size],
                allow_f32,
            )?;
        }
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            expect_float(
                &dense,
                &format!("{prefix}.input_layernorm.weight"),
                &[self.hidden_size],
                allow_f32,
            )?;
            expect_float(
                &dense,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[self.hidden_size],
                allow_f32,
            )?;
            expect_matrix(
                &dense,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &[
                    self.num_attention_heads * self.head_dimension(),
                    self.hidden_size,
                ],
                allow_f32,
            )?;
            for projection in ["k_proj", "v_proj"] {
                expect_matrix(
                    &dense,
                    &format!("{prefix}.self_attn.{projection}.weight"),
                    &[
                        self.num_key_value_heads * self.head_dimension(),
                        self.hidden_size,
                    ],
                    allow_f32,
                )?;
            }
            expect_matrix(
                &dense,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &[
                    self.hidden_size,
                    self.num_attention_heads * self.head_dimension(),
                ],
                allow_f32,
            )?;
            if self.is_qwen3() {
                for projection in ["q_norm", "k_norm"] {
                    expect_float(
                        &dense,
                        &format!("{prefix}.self_attn.{projection}.weight"),
                        &[self.head_dimension()],
                        allow_f32,
                    )?;
                }
            }
            if self.is_olmoe() {
                expect_float(
                    &dense,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[self.num_attention_heads * self.head_dimension()],
                    allow_f32,
                )?;
                expect_float(
                    &dense,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[self.num_key_value_heads * self.head_dimension()],
                    allow_f32,
                )?;
            }
            for (projection, width) in [
                ("q_proj", self.num_attention_heads * self.head_dimension()),
                ("k_proj", self.num_key_value_heads * self.head_dimension()),
                ("v_proj", self.num_key_value_heads * self.head_dimension()),
                ("o_proj", self.hidden_size),
            ] {
                if self.attention_projection_has_bias(projection) {
                    expect_float(
                        &dense,
                        &format!("{prefix}.self_attn.{projection}.bias"),
                        &[width],
                        allow_f32,
                    )?;
                }
            }
            if self.is_moe_layer(layer) {
                let moe_prefix = self.resolve_moe_prefix(layer, |name| dense.contains_key(name))?;
                let router =
                    self.resolve_router_weight(&moe_prefix, |name| dense.contains_key(name))?;
                expect_matrix(
                    &dense,
                    &router,
                    &[self.num_local_experts, self.hidden_size],
                    allow_f32,
                )?;
                if self.is_qwen2() {
                    let shared = self.shared_expert_intermediate_size.unwrap_or(0);
                    expect_matrix(
                        &dense,
                        &format!("{prefix}.mlp.shared_expert_gate.weight"),
                        &[1, self.hidden_size],
                        allow_f32,
                    )?;
                    for projection in ["gate_proj", "up_proj"] {
                        expect_matrix(
                            &dense,
                            &format!("{prefix}.mlp.shared_expert.{projection}.weight"),
                            &[shared, self.hidden_size],
                            allow_f32,
                        )?;
                    }
                    expect_matrix(
                        &dense,
                        &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                        &[self.hidden_size, shared],
                        allow_f32,
                    )?;
                }
            } else {
                for projection in ["gate_proj", "up_proj"] {
                    expect_matrix(
                        &dense,
                        &format!("{prefix}.mlp.{projection}.weight"),
                        &[self.intermediate_size, self.hidden_size],
                        allow_f32,
                    )?;
                }
                expect_matrix(
                    &dense,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[self.hidden_size, self.intermediate_size],
                    allow_f32,
                )?;
            }
        }
        Ok(())
    }
}

fn validate_expert_tensors(
    expert: &crate::ExpertLocation,
    hidden: usize,
    intermediate: usize,
    format: &WeightFormat,
) -> Result<(), MixtralError> {
    if *format == WeightFormat::PackedInt4Group32 {
        return validate_packed_expert_tensors(expert, hidden, intermediate);
    }
    let expected_dtype = match format {
        WeightFormat::Bf16 => "BF16",
        WeightFormat::F32 => "F32",
        _ => {
            return Err(MixtralError::Contract(format!(
                "floating expert format {format:?} has no exact standard-MoE kernel"
            )))
        }
    };
    let matrix = |predicate: fn(&str) -> bool, shape: &[u64]| {
        expert.segments.iter().any(|tensor| {
            tensor.dtype.as_deref() == Some(expected_dtype)
                && tensor.shape == shape
                && predicate(&tensor.tensor.to_ascii_lowercase())
        })
    };
    let down_shape = [hidden as u64, intermediate as u64];
    let has_down = matrix(
        |name| {
            name.contains("down_proj")
                || name.contains("output_linear")
                || name.ends_with(".w2")
                || name.ends_with(".w2.weight")
        },
        &down_shape,
    );
    let fused_shape = [(2 * intermediate) as u64, hidden as u64];
    let has_fused = matrix(
        |name| {
            name.contains("gate_up_proj")
                || name.contains("gate_up.weight")
                || name.contains("input_linear")
        },
        &fused_shape,
    );
    let separate_shape = [intermediate as u64, hidden as u64];
    let has_gate = matrix(
        |name| name.contains("gate_proj") || name.ends_with(".w1") || name.ends_with(".w1.weight"),
        &separate_shape,
    );
    let has_up = matrix(
        |name| {
            name.contains("up_proj")
                || name.ends_with(".w3")
                || name.ends_with(".w3.weight")
                || name.ends_with(".v1")
                || name.ends_with(".v1.weight")
        },
        &separate_shape,
    );
    if !has_down || !(has_fused || (has_gate && has_up)) || (has_fused && (has_gate || has_up)) {
        return Err(MixtralError::Contract(format!(
            "layer {} expert {} does not have one exact {expected_dtype} gated projection contract",
            expert.layer, expert.expert,
        )));
    }
    Ok(())
}

fn validate_packed_expert_tensors(
    expert: &crate::ExpertLocation,
    hidden: usize,
    intermediate: usize,
) -> Result<(), MixtralError> {
    let projection = |predicate: fn(&str) -> bool, rows: usize, cols: usize| {
        let find = |suffix: &str| {
            expert.segments.iter().find(|tensor| {
                let name = tensor.tensor.to_ascii_lowercase();
                name.ends_with(suffix) && predicate(&name)
            })
        };
        let packed = find("weight_packed");
        let scales = find("weight_scale");
        let logical_shape = find("weight_shape");
        packed.is_some_and(|tensor| {
            tensor.dtype.as_deref() == Some("I32")
                && tensor.shape == [rows as u64, cols.div_ceil(8) as u64]
                && tensor.length == (rows * cols.div_ceil(8) * 4) as u64
        }) && scales.is_some_and(|tensor| {
            tensor.dtype.as_deref() == Some("BF16")
                && tensor.shape == [rows as u64, (cols / 32) as u64]
                && tensor.length == (rows * (cols / 32) * 2) as u64
        }) && logical_shape.is_some_and(|tensor| {
            tensor.dtype.as_deref() == Some("I32") && tensor.shape == [2] && tensor.length == 8
        })
    };
    if !hidden.is_multiple_of(32) || !intermediate.is_multiple_of(32) {
        return Err(MixtralError::Contract(
            "packed INT4 standard-MoE widths must be divisible by group size 32".into(),
        ));
    }
    let down = projection(
        |name| name.contains("down_proj") || name.contains(".w2.weight_"),
        hidden,
        intermediate,
    );
    let fused = projection(
        |name| name.contains("gate_up_proj") || name.contains("gate_up.weight_"),
        2 * intermediate,
        hidden,
    );
    let gate = projection(
        |name| {
            !name.contains("gate_up_proj")
                && (name.contains("gate_proj") || name.contains(".w1.weight_"))
        },
        intermediate,
        hidden,
    );
    let up = projection(
        |name| {
            !name.contains("gate_up_proj")
                && (name.contains("up_proj")
                    || name.contains(".w3.weight_")
                    || name.contains(".v1.weight_"))
        },
        intermediate,
        hidden,
    );
    if !down || !(fused || (gate && up)) || (fused && (gate || up)) {
        return Err(MixtralError::Contract(format!(
            "layer {} expert {} does not have one exact packed INT4/BF16 group-32 gated projection contract",
            expert.layer, expert.expert
        )));
    }
    Ok(())
}

fn expect_float(
    tensors: &HashMap<&str, &crate::TensorSegment>,
    name: &str,
    shape: &[usize],
    allow_f32: bool,
) -> Result<(), MixtralError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| MixtralError::Contract(format!("dense tensor {name} is missing")))?;
    let expected: Vec<u64> = shape.iter().map(|value| *value as u64).collect();
    let dtype = tensor.dtype.as_deref();
    if !(dtype == Some("BF16") || (allow_f32 && dtype == Some("F32"))) || tensor.shape != expected {
        return Err(MixtralError::Contract(format!(
            "dense tensor {name} must be exact BF16{} with shape {expected:?}",
            if allow_f32 { " or F32" } else { "" }
        )));
    }
    Ok(())
}

fn expect_matrix(
    tensors: &HashMap<&str, &crate::TensorSegment>,
    name: &str,
    shape: &[usize],
    allow_f32: bool,
) -> Result<(), MixtralError> {
    if tensors.contains_key(name) {
        return expect_float(tensors, name, shape, allow_f32);
    }
    if shape.len() != 2 || !shape[1].is_multiple_of(32) {
        return Err(MixtralError::Contract(format!(
            "dense matrix {name} is missing and cannot use packed group-32 dimensions {shape:?}"
        )));
    }
    let prefix = name.strip_suffix(".weight").ok_or_else(|| {
        MixtralError::Contract(format!(
            "dense matrix {name} has no supported logical suffix"
        ))
    })?;
    let packed_name = format!("{prefix}.weight_packed");
    let scale_name = format!("{prefix}.weight_scale");
    let shape_name = format!("{prefix}.weight_shape");
    let find = |tensor_name: &str| {
        tensors
            .get(tensor_name)
            .copied()
            .ok_or_else(|| MixtralError::Contract(format!("dense tensor {tensor_name} is missing")))
    };
    let packed = find(&packed_name)?;
    let scales = find(&scale_name)?;
    let logical = find(&shape_name)?;
    let rows = shape[0];
    let cols = shape[1];
    let words_per_row = cols.div_ceil(8);
    let groups_per_row = cols / 32;
    if packed.dtype.as_deref() != Some("I32")
        || packed.shape != [rows as u64, words_per_row as u64]
        || packed.length != (rows * words_per_row * 4) as u64
        || scales.dtype.as_deref() != Some("BF16")
        || scales.shape != [rows as u64, groups_per_row as u64]
        || scales.length != (rows * groups_per_row * 2) as u64
        || logical.dtype.as_deref() != Some("I32")
        || logical.shape != [2]
        || logical.length != 8
    {
        return Err(MixtralError::Contract(format!(
            "dense matrix {name} does not have an exact packed INT4/BF16 group-32 triplet"
        )));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum MixtralError {
    #[error("Mixtral contract is invalid: {0}")]
    Contract(String),
    #[error("Mixtral execution failed: {0}")]
    Execution(String),
}

pub struct MixtralExecutor<'a> {
    config: &'a MixtralConfig,
    dense: &'a DenseTensorStore,
    scheduler: &'a StreamingScheduler,
    cuda: Option<&'a CudaAccelerator>,
    dense_tile_bytes: u64,
}

impl<'a> MixtralExecutor<'a> {
    pub fn new(
        config: &'a MixtralConfig,
        dense: &'a DenseTensorStore,
        scheduler: &'a StreamingScheduler,
        cuda: Option<&'a CudaAccelerator>,
        dense_tile_bytes: u64,
    ) -> Result<Self, MixtralError> {
        config.validate()?;
        if dense_tile_bytes
            < (config
                .hidden_size
                .max(config.intermediate_size)
                .max(config.expert_intermediate_size())
                * 2) as u64
        {
            return Err(MixtralError::Contract(
                "dense tile cannot hold the largest matrix row".into(),
            ));
        }
        Ok(Self {
            config,
            dense,
            scheduler,
            cuda,
            dense_tile_bytes,
        })
    }

    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        state: &mut StandardMoeKvState,
    ) -> Result<(Vec<f32>, KimiStepMetrics), MixtralError> {
        if tokens.is_empty()
            || state.layers.len() != self.config.num_hidden_layers
            || state
                .layers
                .iter()
                .any(|cache| cache.len() != state.position())
            || state.position().saturating_add(tokens.len()) > self.config.max_position_embeddings
        {
            return Err(MixtralError::Contract(
                "token batch and KV state are inconsistent".into(),
            ));
        }
        let base_position = state.position();
        let mut hidden = Vec::with_capacity(tokens.len() * self.config.hidden_size);
        let embedding_dtype = self
            .dense
            .metadata("model.embed_tokens.weight")
            .and_then(|tensor| tensor.dtype.as_deref())
            .ok_or_else(|| MixtralError::Contract("embedding dtype is missing".into()))?;
        let embedding_scalar_bytes = match embedding_dtype {
            "BF16" => 2,
            "F32" if self.config.is_granite() => 4,
            dtype => {
                return Err(MixtralError::Contract(format!(
                    "embedding dtype {dtype} has no exact standard-MoE kernel"
                )))
            }
        };
        let row_bytes = self.config.hidden_size * embedding_scalar_bytes;
        for token in tokens {
            if *token as usize >= self.config.vocab_size {
                return Err(MixtralError::Contract("token is outside vocabulary".into()));
            }
            let bytes = self
                .dense
                .read_window(
                    "model.embed_tokens.weight",
                    u64::from(*token) * row_bytes as u64,
                    row_bytes as u64,
                )
                .map_err(exec)?;
            hidden.extend(
                decode_float_values(embedding_dtype, &bytes)
                    .map_err(exec)?
                    .into_iter()
                    .map(|value| value * self.config.embedding_multiplier_value()),
            );
        }
        let mut metrics = KimiStepMetrics::default();
        for layer in 0..self.config.num_hidden_layers {
            hidden = self.forward_layer_batch(
                layer,
                base_position,
                tokens.len(),
                &hidden,
                &mut state.layers[layer],
                &mut metrics,
            )?;
        }
        Ok((hidden, metrics))
    }

    pub fn logits(
        &self,
        hidden: &[f32],
        positions: usize,
    ) -> Result<(Vec<f32>, KimiStepMetrics), MixtralError> {
        let norm = self.read_vector("model.norm.weight", self.config.hidden_size)?;
        let normalized = norm_batch(hidden, positions, &norm, self.config.rms_norm_eps)?;
        let tensor = if self.config.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let mut output = self.dense_matmul(tensor, positions, &normalized)?;
        if self.config.logits_scaling_value() != 1.0 {
            for value in &mut output.values {
                *value /= self.config.logits_scaling_value();
            }
        }
        Ok((output.values, dense_metrics(output.metrics)))
    }

    pub fn greedy_tokens(
        &self,
        hidden: &[f32],
        positions: usize,
    ) -> Result<(Vec<u32>, KimiStepMetrics), MixtralError> {
        let (logits, metrics) = self.logits(hidden, positions)?;
        let tokens = logits
            .chunks_exact(self.config.vocab_size)
            .map(|position| {
                position
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(token, _)| token as u32)
                    .ok_or_else(|| {
                        MixtralError::Execution("language head returned no logits".into())
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((tokens, metrics))
    }

    pub fn verify_greedy_draft(
        &self,
        current_token: u32,
        draft_tokens: &[u32],
        state: &mut StandardMoeKvState,
    ) -> Result<KimiSpeculativeOutput, MixtralError> {
        if draft_tokens.is_empty() {
            return Err(MixtralError::Contract("draft batch is empty".into()));
        }
        let base = state.position();
        let mut inputs = Vec::with_capacity(draft_tokens.len() + 1);
        inputs.push(current_token);
        inputs.extend_from_slice(draft_tokens);
        let (hidden, mut metrics) = match self.forward_tokens(&inputs, state) {
            Ok(output) => output,
            Err(error) => {
                state.truncate(base);
                return Err(error);
            }
        };
        let (target_predictions, head) = match self.greedy_tokens(&hidden, inputs.len()) {
            Ok(output) => output,
            Err(error) => {
                state.truncate(base);
                return Err(error);
            }
        };
        metrics.merge(head);
        let verification = crate::verify_greedy(draft_tokens, &target_predictions)
            .map_err(|error| MixtralError::Execution(error.to_string()))?;
        state.truncate(base + 1 + verification.accepted_draft_tokens);
        Ok(KimiSpeculativeOutput {
            verification,
            target_predictions,
            metrics,
        })
    }

    fn forward_layer_batch(
        &self,
        layer: usize,
        base_position: usize,
        positions: usize,
        hidden: &[f32],
        cache: &mut crate::StandardKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, MixtralError> {
        let prefix = format!("model.layers.{layer}");
        let norm = self.read_vector(
            &format!("{prefix}.input_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = norm_batch(hidden, positions, &norm, self.config.rms_norm_eps)?;
        let attention = self.forward_attention_batch(
            &format!("{prefix}.self_attn"),
            base_position,
            positions,
            &normalized,
            cache,
            metrics,
        )?;
        let residual = add_scaled(hidden, &attention, self.config.residual_multiplier_value())?;
        let norm = self.read_vector(
            &format!("{prefix}.post_attention_layernorm.weight"),
            self.config.hidden_size,
        )?;
        let normalized = norm_batch(&residual, positions, &norm, self.config.rms_norm_eps)?;
        let moe = if self.config.is_moe_layer(layer) {
            let moe_prefix = self
                .config
                .resolve_moe_prefix(layer, |name| self.dense.contains(name))?;
            self.forward_moe_batch(layer, &moe_prefix, positions, &normalized, metrics)?
        } else {
            self.forward_dense_mlp_batch(&format!("{prefix}.mlp"), positions, &normalized, metrics)?
        };
        add_scaled(&residual, &moe, self.config.residual_multiplier_value())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_batch(
        &self,
        prefix: &str,
        base_position: usize,
        positions: usize,
        hidden: &[f32],
        cache: &mut crate::StandardKvCache,
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, MixtralError> {
        let mut q = self.dense_matmul(&format!("{prefix}.q_proj.weight"), positions, hidden)?;
        merge_dense(metrics, q.metrics);
        let mut k = self.dense_matmul(&format!("{prefix}.k_proj.weight"), positions, hidden)?;
        merge_dense(metrics, k.metrics);
        let mut v = self.dense_matmul(&format!("{prefix}.v_proj.weight"), positions, hidden)?;
        merge_dense(metrics, v.metrics);
        if self.config.attention_projection_has_bias("q_proj") {
            self.add_bias(&mut q.values, positions, &format!("{prefix}.q_proj.bias"))?;
        }
        if self.config.attention_projection_has_bias("k_proj") {
            self.add_bias(&mut k.values, positions, &format!("{prefix}.k_proj.bias"))?;
        }
        if self.config.attention_projection_has_bias("v_proj") {
            self.add_bias(&mut v.values, positions, &format!("{prefix}.v_proj.bias"))?;
        }
        let head_dim = self.config.head_dimension();
        if self.config.is_qwen3() {
            let q_norm = self.read_vector(&format!("{prefix}.q_norm.weight"), head_dim)?;
            let k_norm = self.read_vector(&format!("{prefix}.k_norm.weight"), head_dim)?;
            norm_heads(&mut q.values, head_dim, &q_norm, self.config.rms_norm_eps)?;
            norm_heads(&mut k.values, head_dim, &k_norm, self.config.rms_norm_eps)?;
        }
        let query_width = self.config.num_attention_heads * head_dim;
        let kv_width = self.config.num_key_value_heads * head_dim;
        if self.config.is_olmoe() {
            let q_norm = self.read_vector(&format!("{prefix}.q_norm.weight"), query_width)?;
            let k_norm = self.read_vector(&format!("{prefix}.k_norm.weight"), kv_width)?;
            q.values = norm_batch(&q.values, positions, &q_norm, self.config.rms_norm_eps)?;
            k.values = norm_batch(&k.values, positions, &k_norm, self.config.rms_norm_eps)?;
        }
        if let Some(clip) = self.config.clip_qkv {
            for value in q
                .values
                .iter_mut()
                .chain(&mut k.values)
                .chain(&mut v.values)
            {
                *value = value.clamp(-clip, clip);
            }
        }
        let mut attended = Vec::with_capacity(positions * query_width);
        for position in 0..positions {
            let absolute = base_position + position;
            let q_start = position * query_width;
            let kv_start = position * kv_width;
            let query = apply_standard_rope(
                &q.values[q_start..q_start + query_width],
                absolute,
                self.config.num_attention_heads,
                head_dim,
                self.config.rope_theta_value(),
            )
            .map_err(exec)?;
            let key = apply_standard_rope(
                &k.values[kv_start..kv_start + kv_width],
                absolute,
                self.config.num_key_value_heads,
                head_dim,
                self.config.rope_theta_value(),
            )
            .map_err(exec)?;
            cache
                .push(
                    StandardKvEntry {
                        key,
                        value: v.values[kv_start..kv_start + kv_width].to_vec(),
                    },
                    self.config.num_key_value_heads,
                    head_dim,
                    head_dim,
                )
                .map_err(exec)?;
            attended.extend(
                standard_attention_decode_window(
                    &query,
                    cache,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    head_dim,
                    head_dim,
                    self.config.attention_multiplier_value(),
                    self.config.attention_window(),
                )
                .map_err(exec)?,
            );
        }
        let mut output =
            self.dense_matmul(&format!("{prefix}.o_proj.weight"), positions, &attended)?;
        merge_dense(metrics, output.metrics);
        if self.config.attention_projection_has_bias("o_proj") {
            self.add_bias(
                &mut output.values,
                positions,
                &format!("{prefix}.o_proj.bias"),
            )?;
        }
        Ok(output.values)
    }

    fn forward_moe_batch(
        &self,
        layer: usize,
        prefix: &str,
        positions: usize,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, MixtralError> {
        let router = self
            .config
            .resolve_router_weight(prefix, |name| self.dense.contains(name))?;
        let gate = self.dense_matmul(&router, positions, hidden)?;
        merge_dense(metrics, gate.metrics);
        let contract = crate::RouterContract {
            score: crate::RouterScoreKind::Softmax,
            normalize_selected: self.config.norm_topk_prob,
            ..crate::RouterContract::default()
        };
        let semantics = if self.config.norm_topk_prob {
            RouterSemantics::TopKNormalized
        } else {
            RouterSemantics::TopKSoftmax
        };
        let routing = RoutingBatch {
            positions: gate
                .values
                .chunks_exact(self.config.num_local_experts)
                .map(|logits| {
                    route_topk_logits(
                        logits,
                        None,
                        self.config.num_experts_per_tok,
                        semantics.clone(),
                        &contract,
                    )
                    .map(|routes| {
                        routes
                            .into_iter()
                            .map(|route| Route {
                                expert: route.expert as u32,
                                weight: route.weight,
                            })
                            .collect()
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(exec)?,
        };
        let (mut routed, wave_count) = self
            .scheduler
            .for_each_prepared_wave_fold(
                layer as u32,
                routing,
                vec![0.0; positions * self.config.hidden_size],
                |routed, prepared| {
                    for resident in &prepared.experts {
                        let active: Vec<_> = prepared
                            .routes
                            .positions
                            .iter()
                            .enumerate()
                            .filter_map(|(position, routes)| {
                                routes
                                    .iter()
                                    .find(|route| route.expert == resident.key.expert)
                                    .map(|route| (position, route.weight))
                            })
                            .collect();
                        let mut gathered =
                            Vec::with_capacity(active.len() * self.config.hidden_size);
                        for (position, _) in &active {
                            gathered.extend_from_slice(
                                &hidden[*position * self.config.hidden_size
                                    ..(*position + 1) * self.config.hidden_size],
                            );
                        }
                        let output = standard_gated_expert_batch_resident(
                            resident,
                            self.config.hidden_size,
                            self.config.expert_intermediate_size(),
                            active.len(),
                            &gathered,
                            ActivationKind::Silu,
                            self.cuda,
                        )
                        .map_err(|error| error.to_string())?;
                        for (active_index, (position, weight)) in active.into_iter().enumerate() {
                            for index in 0..self.config.hidden_size {
                                routed[position * self.config.hidden_size + index] +=
                                    weight * output[active_index * self.config.hidden_size + index];
                            }
                        }
                    }
                    Ok(())
                },
            )
            .map_err(exec)?;
        metrics.expert_waves += wave_count as u64;
        if self.config.is_qwen2() {
            let shared = self.forward_dense_mlp_batch(
                &format!("{prefix}.shared_expert"),
                positions,
                hidden,
                metrics,
            )?;
            let gate = self.dense_matmul(
                &format!("{prefix}.shared_expert_gate.weight"),
                positions,
                hidden,
            )?;
            merge_dense(metrics, gate.metrics);
            if gate.values.len() != positions {
                return Err(MixtralError::Contract(
                    "Qwen2 shared-expert gate must return one value per position".into(),
                ));
            }
            for position in 0..positions {
                let weight = crate::sigmoid(gate.values[position]);
                for index in 0..self.config.hidden_size {
                    routed[position * self.config.hidden_size + index] +=
                        weight * shared[position * self.config.hidden_size + index];
                }
            }
        }
        Ok(routed)
    }

    fn forward_dense_mlp_batch(
        &self,
        prefix: &str,
        positions: usize,
        hidden: &[f32],
        metrics: &mut KimiStepMetrics,
    ) -> Result<Vec<f32>, MixtralError> {
        let gate = self.dense_matmul(&format!("{prefix}.gate_proj.weight"), positions, hidden)?;
        merge_dense(metrics, gate.metrics);
        let up = self.dense_matmul(&format!("{prefix}.up_proj.weight"), positions, hidden)?;
        merge_dense(metrics, up.metrics);
        let mut activated = gate.values;
        for (gate, up) in activated.iter_mut().zip(up.values) {
            *gate = crate::apply_activation(ActivationKind::Silu, *gate) * up;
        }
        let down =
            self.dense_matmul(&format!("{prefix}.down_proj.weight"), positions, &activated)?;
        merge_dense(metrics, down.metrics);
        Ok(down.values)
    }

    fn add_bias(
        &self,
        values: &mut [f32],
        positions: usize,
        tensor: &str,
    ) -> Result<(), MixtralError> {
        if positions == 0 || !values.len().is_multiple_of(positions) {
            return Err(MixtralError::Contract(
                "biased projection batch has inconsistent dimensions".into(),
            ));
        }
        let width = values.len() / positions;
        let bias = self.read_vector(tensor, width)?;
        for row in values.chunks_exact_mut(width) {
            for (value, bias) in row.iter_mut().zip(&bias) {
                *value += bias;
            }
        }
        Ok(())
    }

    fn dense_matmul(
        &self,
        tensor: &str,
        positions: usize,
        input: &[f32],
    ) -> Result<TiledMatmulOutput, MixtralError> {
        let cuda = self.cuda;
        if self.dense.contains(tensor) {
            match self
                .dense
                .metadata(tensor)
                .and_then(|metadata| metadata.dtype.as_deref())
            {
                Some("BF16") => crate::tiled_bf16_matmul(
                    self.dense,
                    tensor,
                    self.dense_tile_bytes,
                    positions,
                    input,
                    move |weights, rows, cols, positions, input| match cuda {
                        Some(cuda) => cuda.bf16_matmul_bytes(weights, rows, cols, positions, input),
                        None => bf16_tile_matmul_cpu(weights, rows, cols, positions, input),
                    },
                )
                .map_err(exec),
                Some("F32") if self.config.is_granite() => crate::tiled_f32_matmul(
                    self.dense,
                    tensor,
                    self.dense_tile_bytes,
                    positions,
                    input,
                )
                .map_err(exec),
                dtype => Err(MixtralError::Contract(format!(
                    "dense tensor {tensor} has unsupported dtype {dtype:?}"
                ))),
            }
        } else {
            crate::tiled_packed_int4_group32_bf16_matmul(
                self.dense,
                tensor,
                self.dense_tile_bytes,
                positions,
                input,
                move |packed, scales, rows, cols, positions, input| match cuda {
                    Some(cuda) => cuda.int4_group32_bf16_bytes_matmul(
                        packed, scales, rows, cols, positions, input,
                    ),
                    None => crate::int4_group32_bf16_matmul_cpu(
                        packed, scales, rows, cols, positions, input,
                    )
                    .map_err(|error| error.to_string()),
                },
            )
            .map_err(exec)
        }
    }

    fn read_vector(&self, tensor: &str, length: usize) -> Result<Vec<f32>, MixtralError> {
        let metadata = self
            .dense
            .metadata(tensor)
            .ok_or_else(|| MixtralError::Contract(format!("tensor {tensor} is missing")))?;
        let dtype = metadata.dtype.as_deref().unwrap_or("");
        if !matches!(dtype, "BF16" | "F32")
            || (dtype == "F32" && !self.config.is_granite())
            || metadata.shape != [length as u64]
        {
            return Err(MixtralError::Contract(format!(
                "tensor {tensor} is not an exact supported floating vector"
            )));
        }
        decode_float_values(dtype, &self.dense.read(tensor).map_err(exec)?).map_err(exec)
    }
}

fn norm_batch(
    values: &[f32],
    positions: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, MixtralError> {
    if values.len() != positions * weight.len() {
        return Err(MixtralError::Contract(
            "normalization batch is invalid".into(),
        ));
    }
    let mut output = Vec::with_capacity(values.len());
    for position in values.chunks_exact(weight.len()) {
        output.extend(rms_norm(position, weight, epsilon).map_err(exec)?);
    }
    Ok(output)
}

fn norm_heads(
    values: &mut [f32],
    head_dim: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<(), MixtralError> {
    if head_dim == 0 || weight.len() != head_dim || !values.len().is_multiple_of(head_dim) {
        return Err(MixtralError::Contract(
            "Q/K head normalization dimensions are inconsistent".into(),
        ));
    }
    for head in values.chunks_exact_mut(head_dim) {
        let normalized = rms_norm(head, weight, epsilon).map_err(exec)?;
        head.copy_from_slice(&normalized);
    }
    Ok(())
}

fn add_scaled(left: &[f32], right: &[f32], scale: f32) -> Result<Vec<f32>, MixtralError> {
    if left.len() != right.len() {
        return Err(MixtralError::Contract("residual dimensions differ".into()));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right * scale)
        .collect())
}

fn merge_dense(target: &mut KimiStepMetrics, metrics: crate::DenseTileMetrics) {
    target.dense_tiles += metrics.tiles;
    target.dense_storage_bytes += metrics.storage_bytes;
    target.peak_dense_tile_bytes = target.peak_dense_tile_bytes.max(metrics.peak_tile_bytes);
}

fn dense_metrics(metrics: crate::DenseTileMetrics) -> KimiStepMetrics {
    let mut output = KimiStepMetrics::default();
    merge_dense(&mut output, metrics);
    output
}

fn exec(error: impl std::fmt::Display) -> MixtralError {
    MixtralError::Execution(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{ExpertLocation, NoAccelerator, ResidencyManager, TensorSegment, WeightedMirror};

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn identity(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                if index / cols == index % cols {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn packed_standard_expert_manifest_requires_exact_group32_tensor_triplets() {
        let tensor = |name: &str, dtype: &str, shape: Vec<u64>, length: u64| TensorSegment {
            tensor: name.into(),
            dtype: Some(dtype.into()),
            shape,
            shard: "weights.safetensors".into(),
            offset: 0,
            length,
        };
        let mut expert = ExpertLocation {
            layer: 0,
            expert: 0,
            segments: vec![
                tensor(
                    "expert.gate_up_proj.weight_packed",
                    "I32",
                    vec![64, 4],
                    1024,
                ),
                tensor("expert.gate_up_proj.weight_scale", "BF16", vec![64, 1], 128),
                tensor("expert.gate_up_proj.weight_shape", "I32", vec![2], 8),
                tensor("expert.down_proj.weight_packed", "I32", vec![32, 4], 512),
                tensor("expert.down_proj.weight_scale", "BF16", vec![32, 1], 64),
                tensor("expert.down_proj.weight_shape", "I32", vec![2], 8),
            ],
        };
        validate_expert_tensors(&expert, 32, 32, &WeightFormat::PackedInt4Group32).unwrap();
        expert.segments[4].length = 62;
        assert!(
            validate_expert_tensors(&expert, 32, 32, &WeightFormat::PackedInt4Group32).is_err()
        );
    }

    #[test]
    fn packed_dense_manifest_accepts_only_exact_group32_tensor_triplets() {
        let segments = vec![
            TensorSegment {
                tensor: "linear.weight_packed".into(),
                dtype: Some("I32".into()),
                shape: vec![64, 4],
                shard: "weights.safetensors".into(),
                offset: 0,
                length: 1024,
            },
            TensorSegment {
                tensor: "linear.weight_scale".into(),
                dtype: Some("BF16".into()),
                shape: vec![64, 1],
                shard: "weights.safetensors".into(),
                offset: 1024,
                length: 128,
            },
            TensorSegment {
                tensor: "linear.weight_shape".into(),
                dtype: Some("I32".into()),
                shape: vec![2],
                shard: "weights.safetensors".into(),
                offset: 1152,
                length: 8,
            },
        ];
        let dense: HashMap<_, _> = segments
            .iter()
            .map(|tensor| (tensor.tensor.as_str(), tensor))
            .collect();
        expect_matrix(&dense, "linear.weight", &[64, 32], false).unwrap();
        assert!(expect_matrix(&dense, "linear.weight", &[64, 64], false).is_err());
    }

    #[test]
    fn standard_executor_dispatches_packed_dense_projection_without_weight_expansion() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-packed-projection-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let rows = 32_usize;
        let cols = 32_usize;
        let packed = 0x9999_9999_u32.to_le_bytes().repeat(rows * (cols / 8));
        let scale = ((0.5_f32.to_bits() >> 16) as u16).to_le_bytes();
        let scales = scale.repeat(rows);
        let mut payload = packed.clone();
        let scale_offset = payload.len() as u64;
        payload.extend_from_slice(&scales);
        let shape_offset = payload.len() as u64;
        payload.extend_from_slice(&(rows as i32).to_le_bytes());
        payload.extend_from_slice(&(cols as i32).to_le_bytes());
        fs::write(root.join("weights.bin"), payload).unwrap();
        let dense_store = DenseTensorStore::new(
            &root,
            vec![],
            [
                TensorSegment {
                    tensor: "linear.weight_packed".into(),
                    dtype: Some("I32".into()),
                    shape: vec![rows as u64, (cols / 8) as u64],
                    shard: "weights.bin".into(),
                    offset: 0,
                    length: packed.len() as u64,
                },
                TensorSegment {
                    tensor: "linear.weight_scale".into(),
                    dtype: Some("BF16".into()),
                    shape: vec![rows as u64, 1],
                    shard: "weights.bin".into(),
                    offset: scale_offset,
                    length: scales.len() as u64,
                },
                TensorSegment {
                    tensor: "linear.weight_shape".into(),
                    dtype: Some("I32".into()),
                    shape: vec![2],
                    shard: "weights.bin".into(),
                    offset: shape_offset,
                    length: 8,
                },
            ],
        );
        let expert_store = Arc::new(crate::ExpertStore::new(
            &root,
            Vec::<WeightedMirror>::new(),
            Vec::<ExpertLocation>::new(),
        ));
        let residency = Arc::new(ResidencyManager::new(
            expert_store,
            Arc::new(NoAccelerator),
            0,
            0,
        ));
        let scheduler = StreamingScheduler::with_inflight_budget(residency, 0, 1).unwrap();
        let config = MixtralConfig {
            model_type: "mixtral".into(),
            vocab_size: 32,
            hidden_size: 32,
            intermediate_size: 32,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            num_local_experts: 1,
            num_experts_per_tok: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            head_dim: Some(8),
            rope_theta: Some(10_000.0),
            rope_parameters: None,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            attention_bias: false,
            clip_qkv: None,
            decoder_sparse_step: 1,
            mlp_only_layers: Some(Vec::new()),
            norm_topk_prob: true,
            tie_word_embeddings: false,
            embedding_multiplier: None,
            logits_scaling: None,
            residual_multiplier: None,
            attention_multiplier: None,
        };
        let executor = MixtralExecutor::new(&config, &dense_store, &scheduler, None, 64).unwrap();
        let output = executor
            .dense_matmul("linear.weight", 2, &[1.0; 64])
            .unwrap();
        assert_eq!(output.values, vec![16.0; 64]);
        assert_eq!(output.metrics.peak_tile_bytes, 54);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_mixtral_batch_matches_sequential_and_keeps_bf16_gqa_kv() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-mixtral-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let config = MixtralConfig {
            model_type: "mixtral".into(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 4,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            head_dim: None,
            rope_theta: Some(10_000.0),
            rope_parameters: None,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            attention_bias: false,
            clip_qkv: None,
            decoder_sparse_step: 1,
            mlp_only_layers: Some(Vec::new()),
            norm_topk_prob: true,
            tie_word_embeddings: false,
            embedding_multiplier: None,
            logits_scaling: None,
            residual_multiplier: None,
            attention_multiplier: None,
        };
        let mut payload = Vec::new();
        let mut dense = Vec::new();
        let mut append_dense = |name: &str, shape: &[u64], values: &[f32]| {
            let offset = payload.len() as u64;
            let bytes = bf16(values);
            payload.extend(&bytes);
            dense.push(TensorSegment {
                tensor: name.into(),
                dtype: Some("BF16".into()),
                shape: shape.to_vec(),
                shard: "weights.bin".into(),
                offset,
                length: bytes.len() as u64,
            });
        };
        let embeddings: Vec<f32> = (0..8)
            .flat_map(|token| (0..4).map(move |column| if token % 4 == column { 1.0 } else { 0.0 }))
            .collect();
        append_dense("model.embed_tokens.weight", &[8, 4], &embeddings);
        append_dense("model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]);
        append_dense(
            "model.layers.0.self_attn.q_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.self_attn.k_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.v_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.o_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.post_attention_layernorm.weight",
            &[4],
            &[1.0; 4],
        );
        append_dense(
            "model.layers.0.block_sparse_moe.gate.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        append_dense("model.norm.weight", &[4], &[1.0; 4]);
        append_dense("lm_head.weight", &[8, 4], &embeddings);
        drop(append_dense);

        let mut experts = Vec::new();
        for expert in 0..2_u32 {
            let fused_offset = payload.len() as u64;
            let mut fused = identity(4, 4);
            fused.extend(identity(4, 4));
            let fused = bf16(&fused);
            payload.extend(&fused);
            let down_offset = payload.len() as u64;
            let down = bf16(&identity(4, 4));
            payload.extend(&down);
            experts.push(ExpertLocation {
                layer: 0,
                expert,
                segments: vec![
                    TensorSegment {
                        tensor: format!(
                            "model.layers.0.block_sparse_moe.experts.{expert}.gate_up_proj"
                        ),
                        dtype: Some("BF16".into()),
                        shape: vec![8, 4],
                        shard: "weights.bin".into(),
                        offset: fused_offset,
                        length: fused.len() as u64,
                    },
                    TensorSegment {
                        tensor: format!(
                            "model.layers.0.block_sparse_moe.experts.{expert}.down_proj"
                        ),
                        dtype: Some("BF16".into()),
                        shape: vec![4, 4],
                        shard: "weights.bin".into(),
                        offset: down_offset,
                        length: down.len() as u64,
                    },
                ],
            });
        }
        fs::write(root.join("weights.bin"), payload).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manifest = RuntimeManifest {
            schema_version: crate::manifest::CURRENT_SCHEMA,
            architecture: crate::ArchitectureContract {
                adapter: "sytra-mixtral".into(),
                model_type: "mixtral".into(),
                attention: crate::AttentionKind::Standard,
                router: RouterSemantics::TopKSoftmax,
                expert_format: WeightFormat::Bf16,
                family: ModelFamily::Mixtral,
                expert_layout: crate::ExpertTensorLayout::Discrete,
                activation: ActivationKind::Silu,
                router_config: crate::RouterContract {
                    normalize_selected: true,
                    ..crate::RouterContract::default()
                },
                attention_config: crate::AttentionContract {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 2,
                    value_head_dim: 2,
                    ..crate::AttentionContract::default()
                },
                quantization: crate::QuantizationContract::default(),
                hidden_size: 4,
                expert_intermediate_size: 4,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 2,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: dense.iter().map(|tensor| tensor.length).sum(),
            storage: crate::ExpertStorageManifest {
                contiguous_experts: false,
                experts: experts.clone(),
                dense_tensors: dense.clone(),
            },
        };
        config.validate_manifest(&manifest).unwrap();
        let dense_store = DenseTensorStore::new(&root, Vec::<WeightedMirror>::new(), dense);
        let expert_store = Arc::new(crate::ExpertStore::new(
            &root,
            Vec::<WeightedMirror>::new(),
            experts,
        ));
        let residency = Arc::new(ResidencyManager::new(
            expert_store,
            Arc::new(NoAccelerator),
            192,
            0,
        ));
        let scheduler = StreamingScheduler::with_inflight_budget(residency, 0, 96).unwrap();
        let executor = MixtralExecutor::new(&config, &dense_store, &scheduler, None, 8).unwrap();

        let mut sequential_state = StandardMoeKvState::new(1);
        let (first, first_metrics) = executor
            .forward_tokens(&[0], &mut sequential_state)
            .unwrap();
        let (second, second_metrics) = executor
            .forward_tokens(&[1], &mut sequential_state)
            .unwrap();
        let mut sequential = first;
        sequential.extend(second);
        let mut batch_state = StandardMoeKvState::new(1);
        let (batch, batch_metrics) = executor.forward_tokens(&[0, 1], &mut batch_state).unwrap();
        for (actual, expected) in batch.iter().zip(&sequential) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        assert_eq!(batch_state, sequential_state);
        assert_eq!(batch_state.position(), 2);
        assert_eq!(batch_state.bytes(), 16);
        assert!(
            batch_metrics.dense_storage_bytes
                < first_metrics.dense_storage_bytes + second_metrics.dense_storage_bytes
        );
        let (logits, _) = executor.logits(&batch, 2).unwrap();
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|value| value.is_finite()));
        drop(executor);
        drop(scheduler);
        let memory =
            crate::plan_memory_envelope(&manifest, 32, 512 * 1024 * 1024, 0, 2, 8, 2).unwrap();
        let runtime = crate::MixtralRuntime::new(
            &root,
            manifest,
            memory,
            vec![],
            crate::MixtralRuntimeOptions {
                dense_tile_bytes: 8,
                ..crate::MixtralRuntimeOptions::default()
            },
        )
        .unwrap();
        let oracle = runtime.oracle_outputs(&[0, 1]).unwrap();
        assert_eq!(oracle.teacher_forced_predictions.len(), 2);
        assert_eq!(oracle.final_logits.len(), 8);
        assert!(runtime.metrics().unwrap().experts.ram_peak_bytes <= 96);
        drop(runtime);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_qwen3_stacked_experts_apply_qk_norm_and_use_moe_width() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-qwen3-moe-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let config = MixtralConfig {
            model_type: "qwen3_moe".into(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 6,
            moe_intermediate_size: Some(4),
            shared_expert_intermediate_size: None,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            head_dim: Some(2),
            rope_theta: None,
            rope_parameters: Some(serde_json::json!({"rope_theta": 1_000_000.0})),
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            attention_bias: false,
            clip_qkv: None,
            decoder_sparse_step: 1,
            mlp_only_layers: Some(Vec::new()),
            norm_topk_prob: false,
            tie_word_embeddings: false,
            embedding_multiplier: None,
            logits_scaling: None,
            residual_multiplier: None,
            attention_multiplier: None,
        };
        let mut payload = Vec::new();
        let mut dense = Vec::new();
        let mut append_dense = |name: &str, shape: &[u64], values: &[f32]| {
            let offset = payload.len() as u64;
            let bytes = bf16(values);
            payload.extend(&bytes);
            dense.push(TensorSegment {
                tensor: name.into(),
                dtype: Some("BF16".into()),
                shape: shape.to_vec(),
                shard: "weights.bin".into(),
                offset,
                length: bytes.len() as u64,
            });
        };
        let embeddings: Vec<f32> = (0..8)
            .flat_map(|token| (0..4).map(move |column| if token % 4 == column { 1.0 } else { 0.0 }))
            .collect();
        append_dense("model.embed_tokens.weight", &[8, 4], &embeddings);
        append_dense("model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]);
        append_dense(
            "model.layers.0.self_attn.q_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.self_attn.k_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.v_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.o_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense("model.layers.0.self_attn.q_norm.weight", &[2], &[1.0; 2]);
        append_dense("model.layers.0.self_attn.k_norm.weight", &[2], &[1.0; 2]);
        append_dense(
            "model.layers.0.post_attention_layernorm.weight",
            &[4],
            &[1.0; 4],
        );
        append_dense(
            "model.layers.0.mlp.gate.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        append_dense("model.norm.weight", &[4], &[1.0; 4]);
        append_dense("lm_head.weight", &[8, 4], &embeddings);
        drop(append_dense);

        let mut experts = Vec::new();
        for expert in 0..2_u32 {
            let fused_offset = payload.len() as u64;
            let mut fused = identity(4, 4);
            fused.extend(identity(4, 4));
            let fused = bf16(&fused);
            payload.extend(&fused);
            let down_offset = payload.len() as u64;
            let down = bf16(&identity(4, 4));
            payload.extend(&down);
            experts.push(ExpertLocation {
                layer: 0,
                expert,
                segments: vec![
                    TensorSegment {
                        tensor: "model.layers.0.mlp.experts.gate_up_proj".into(),
                        dtype: Some("BF16".into()),
                        shape: vec![8, 4],
                        shard: "weights.bin".into(),
                        offset: fused_offset,
                        length: fused.len() as u64,
                    },
                    TensorSegment {
                        tensor: "model.layers.0.mlp.experts.down_proj".into(),
                        dtype: Some("BF16".into()),
                        shape: vec![4, 4],
                        shard: "weights.bin".into(),
                        offset: down_offset,
                        length: down.len() as u64,
                    },
                ],
            });
        }
        fs::write(root.join("weights.bin"), payload).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manifest = RuntimeManifest {
            schema_version: crate::manifest::CURRENT_SCHEMA,
            architecture: crate::ArchitectureContract {
                adapter: "sytra-qwen3-moe".into(),
                model_type: "qwen3_moe".into(),
                attention: crate::AttentionKind::Standard,
                router: RouterSemantics::TopKSoftmax,
                expert_format: WeightFormat::Bf16,
                family: ModelFamily::Qwen3Moe,
                expert_layout: crate::ExpertTensorLayout::StackedAxis0,
                activation: ActivationKind::Silu,
                router_config: crate::RouterContract {
                    normalize_selected: false,
                    ..crate::RouterContract::default()
                },
                attention_config: crate::AttentionContract {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 2,
                    value_head_dim: 2,
                    ..crate::AttentionContract::default()
                },
                quantization: crate::QuantizationContract::default(),
                hidden_size: 4,
                expert_intermediate_size: 4,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 2,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: dense.iter().map(|tensor| tensor.length).sum(),
            storage: crate::ExpertStorageManifest {
                contiguous_experts: false,
                experts,
                dense_tensors: dense,
            },
        };
        config.validate_manifest(&manifest).unwrap();
        crate::validate_forward_contract(&manifest).unwrap();
        let memory =
            crate::plan_memory_envelope(&manifest, 32, 512 * 1024 * 1024, 0, 2, 12, 2).unwrap();
        let runtime = crate::MixtralRuntime::new(
            &root,
            manifest,
            memory,
            vec![],
            crate::MixtralRuntimeOptions {
                dense_tile_bytes: 12,
                ..crate::MixtralRuntimeOptions::default()
            },
        )
        .unwrap();
        let oracle = runtime.oracle_outputs(&[0, 1]).unwrap();
        assert_eq!(oracle.teacher_forced_predictions.len(), 2);
        assert_eq!(oracle.final_logits.len(), 8);
        assert!(oracle.final_logits.iter().all(|value| value.is_finite()));
        assert_eq!(runtime.new_state().bytes(), 0);
        drop(runtime);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_qwen2_applies_qkv_bias_and_sigmoid_gated_shared_expert() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-qwen2-moe-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let config = MixtralConfig {
            model_type: "qwen2_moe".into(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 6,
            moe_intermediate_size: Some(4),
            shared_expert_intermediate_size: Some(4),
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            head_dim: Some(2),
            rope_theta: Some(1_000_000.0),
            rope_parameters: None,
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            attention_bias: true,
            clip_qkv: None,
            decoder_sparse_step: 1,
            mlp_only_layers: Some(Vec::new()),
            norm_topk_prob: false,
            tie_word_embeddings: false,
            embedding_multiplier: None,
            logits_scaling: None,
            residual_multiplier: None,
            attention_multiplier: None,
        };
        let mut payload = Vec::new();
        let mut dense = Vec::new();
        let mut append_dense = |name: &str, shape: &[u64], values: &[f32]| {
            let offset = payload.len() as u64;
            let bytes = bf16(values);
            payload.extend(&bytes);
            dense.push(TensorSegment {
                tensor: name.into(),
                dtype: Some("BF16".into()),
                shape: shape.to_vec(),
                shard: "weights.bin".into(),
                offset,
                length: bytes.len() as u64,
            });
        };
        let embeddings: Vec<f32> = (0..8)
            .flat_map(|token| (0..4).map(move |column| if token % 4 == column { 1.0 } else { 0.0 }))
            .collect();
        append_dense("model.embed_tokens.weight", &[8, 4], &embeddings);
        append_dense("model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]);
        append_dense(
            "model.layers.0.self_attn.q_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.self_attn.k_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.v_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.o_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense("model.layers.0.self_attn.q_proj.bias", &[4], &[0.1; 4]);
        append_dense("model.layers.0.self_attn.k_proj.bias", &[2], &[0.1; 2]);
        append_dense("model.layers.0.self_attn.v_proj.bias", &[2], &[0.1; 2]);
        append_dense(
            "model.layers.0.post_attention_layernorm.weight",
            &[4],
            &[1.0; 4],
        );
        append_dense(
            "model.layers.0.mlp.gate.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        for projection in ["gate_proj", "up_proj"] {
            append_dense(
                &format!("model.layers.0.mlp.shared_expert.{projection}.weight"),
                &[4, 4],
                &identity(4, 4),
            );
        }
        append_dense(
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.mlp.shared_expert_gate.weight",
            &[1, 4],
            &[0.0; 4],
        );
        append_dense("model.norm.weight", &[4], &[1.0; 4]);
        append_dense("lm_head.weight", &[8, 4], &embeddings);
        drop(append_dense);

        let mut experts = Vec::new();
        for expert in 0..2_u32 {
            let mut segments = Vec::new();
            for (projection, shape) in [
                ("gate_proj", vec![4, 4]),
                ("up_proj", vec![4, 4]),
                ("down_proj", vec![4, 4]),
            ] {
                let offset = payload.len() as u64;
                let bytes = bf16(&identity(4, 4));
                payload.extend(&bytes);
                segments.push(TensorSegment {
                    tensor: format!("model.layers.0.mlp.experts.{expert}.{projection}.weight"),
                    dtype: Some("BF16".into()),
                    shape,
                    shard: "weights.bin".into(),
                    offset,
                    length: bytes.len() as u64,
                });
            }
            experts.push(ExpertLocation {
                layer: 0,
                expert,
                segments,
            });
        }
        fs::write(root.join("weights.bin"), payload).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manifest = RuntimeManifest {
            schema_version: crate::manifest::CURRENT_SCHEMA,
            architecture: crate::ArchitectureContract {
                adapter: "sytra-qwen2-moe".into(),
                model_type: "qwen2_moe".into(),
                attention: crate::AttentionKind::Standard,
                router: RouterSemantics::TopKSoftmax,
                expert_format: WeightFormat::Bf16,
                family: ModelFamily::Qwen2Moe,
                expert_layout: crate::ExpertTensorLayout::Discrete,
                activation: ActivationKind::Silu,
                router_config: crate::RouterContract {
                    normalize_selected: false,
                    ..crate::RouterContract::default()
                },
                attention_config: crate::AttentionContract {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 2,
                    value_head_dim: 2,
                    ..crate::AttentionContract::default()
                },
                quantization: crate::QuantizationContract::default(),
                hidden_size: 4,
                expert_intermediate_size: 4,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 2,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: dense.iter().map(|tensor| tensor.length).sum(),
            storage: crate::ExpertStorageManifest {
                contiguous_experts: false,
                experts,
                dense_tensors: dense,
            },
        };
        config.validate_manifest(&manifest).unwrap();
        crate::validate_forward_contract(&manifest).unwrap();
        let memory =
            crate::plan_memory_envelope(&manifest, 32, 512 * 1024 * 1024, 0, 2, 12, 2).unwrap();
        let runtime = crate::MixtralRuntime::new(
            &root,
            manifest,
            memory,
            vec![],
            crate::MixtralRuntimeOptions {
                dense_tile_bytes: 12,
                ..crate::MixtralRuntimeOptions::default()
            },
        )
        .unwrap();
        let oracle = runtime.oracle_outputs(&[0, 1]).unwrap();
        assert_eq!(oracle.teacher_forced_predictions.len(), 2);
        assert_eq!(oracle.final_logits.len(), 8);
        assert!(oracle.final_logits.iter().all(|value| value.is_finite()));
        drop(runtime);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bounded_olmoe_applies_full_qk_norm_clip_and_weighted_routing() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-olmoe-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let config = MixtralConfig {
            model_type: "olmoe".into(),
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 4,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            head_dim: Some(2),
            rope_theta: None,
            rope_parameters: Some(
                serde_json::json!({"rope_type": "default", "rope_theta": 10_000.0}),
            ),
            rope_scaling: None,
            sliding_window: None,
            use_sliding_window: false,
            attention_bias: false,
            clip_qkv: Some(0.5),
            decoder_sparse_step: 1,
            mlp_only_layers: None,
            norm_topk_prob: false,
            tie_word_embeddings: false,
            embedding_multiplier: None,
            logits_scaling: None,
            residual_multiplier: None,
            attention_multiplier: None,
        };
        let mut payload = Vec::new();
        let mut dense = Vec::new();
        let mut append_dense = |name: &str, shape: &[u64], values: &[f32]| {
            let offset = payload.len() as u64;
            let bytes = bf16(values);
            payload.extend(&bytes);
            dense.push(TensorSegment {
                tensor: name.into(),
                dtype: Some("BF16".into()),
                shape: shape.to_vec(),
                shard: "weights.bin".into(),
                offset,
                length: bytes.len() as u64,
            });
        };
        let embeddings: Vec<f32> = (0..8)
            .flat_map(|token| (0..4).map(move |column| if token % 4 == column { 2.0 } else { 0.0 }))
            .collect();
        append_dense("model.embed_tokens.weight", &[8, 4], &embeddings);
        append_dense("model.layers.0.input_layernorm.weight", &[4], &[1.0; 4]);
        append_dense(
            "model.layers.0.self_attn.q_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense(
            "model.layers.0.self_attn.k_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.v_proj.weight",
            &[2, 4],
            &identity(2, 4),
        );
        append_dense(
            "model.layers.0.self_attn.o_proj.weight",
            &[4, 4],
            &identity(4, 4),
        );
        append_dense("model.layers.0.self_attn.q_norm.weight", &[4], &[1.0; 4]);
        append_dense("model.layers.0.self_attn.k_norm.weight", &[2], &[1.0; 2]);
        append_dense(
            "model.layers.0.post_attention_layernorm.weight",
            &[4],
            &[1.0; 4],
        );
        append_dense(
            "model.layers.0.mlp.gate.weight",
            &[2, 4],
            &[1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        );
        append_dense("model.norm.weight", &[4], &[1.0; 4]);
        append_dense("lm_head.weight", &[8, 4], &embeddings);
        drop(append_dense);
        let mut experts = Vec::new();
        for expert in 0..2_u32 {
            let fused_offset = payload.len() as u64;
            let mut fused = identity(4, 4);
            fused.extend(identity(4, 4));
            let fused = bf16(&fused);
            payload.extend(&fused);
            let down_offset = payload.len() as u64;
            let down = bf16(&identity(4, 4));
            payload.extend(&down);
            experts.push(ExpertLocation {
                layer: 0,
                expert,
                segments: vec![
                    TensorSegment {
                        tensor: "model.layers.0.mlp.experts.gate_up_proj".into(),
                        dtype: Some("BF16".into()),
                        shape: vec![8, 4],
                        shard: "weights.bin".into(),
                        offset: fused_offset,
                        length: fused.len() as u64,
                    },
                    TensorSegment {
                        tensor: "model.layers.0.mlp.experts.down_proj".into(),
                        dtype: Some("BF16".into()),
                        shape: vec![4, 4],
                        shard: "weights.bin".into(),
                        offset: down_offset,
                        length: down.len() as u64,
                    },
                ],
            });
        }
        fs::write(root.join("weights.bin"), payload).unwrap();
        fs::write(
            root.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        let manifest = RuntimeManifest {
            schema_version: crate::manifest::CURRENT_SCHEMA,
            architecture: crate::ArchitectureContract {
                adapter: "sytra-olmoe".into(),
                model_type: "olmoe".into(),
                attention: crate::AttentionKind::Standard,
                router: RouterSemantics::TopKWeighted,
                expert_format: WeightFormat::Bf16,
                family: ModelFamily::Olmoe,
                expert_layout: crate::ExpertTensorLayout::StackedAxis0,
                activation: ActivationKind::Silu,
                router_config: crate::RouterContract {
                    normalize_selected: false,
                    ..crate::RouterContract::default()
                },
                attention_config: crate::AttentionContract {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 2,
                    value_head_dim: 2,
                    ..crate::AttentionContract::default()
                },
                quantization: crate::QuantizationContract::default(),
                hidden_size: 4,
                expert_intermediate_size: 4,
                moe_layers: vec![0],
                num_layers: 1,
                experts_per_layer: 2,
                experts_per_token: 1,
                forward_verified: false,
            },
            dense_bytes: dense.iter().map(|tensor| tensor.length).sum(),
            storage: crate::ExpertStorageManifest {
                contiguous_experts: false,
                experts,
                dense_tensors: dense,
            },
        };
        config.validate_manifest(&manifest).unwrap();
        crate::validate_forward_contract(&manifest).unwrap();
        let memory =
            crate::plan_memory_envelope(&manifest, 32, 512 * 1024 * 1024, 0, 2, 8, 2).unwrap();
        let runtime = crate::MixtralRuntime::new(
            &root,
            manifest,
            memory,
            vec![],
            crate::MixtralRuntimeOptions {
                dense_tile_bytes: 8,
                ..crate::MixtralRuntimeOptions::default()
            },
        )
        .unwrap();
        let oracle = runtime.oracle_outputs(&[0, 1]).unwrap();
        assert_eq!(oracle.teacher_forced_predictions.len(), 2);
        assert_eq!(oracle.final_logits.len(), 8);
        assert!(oracle.final_logits.iter().all(|value| value.is_finite()));
        drop(runtime);
        fs::remove_dir_all(root).unwrap();
    }
}
