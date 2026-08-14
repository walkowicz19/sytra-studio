use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::{Map, Value};
use thiserror::Error;

use crate::{
    adapter::AdapterDescriptor,
    manifest::{ActivationKind, RouterScoreKind, RuntimeManifest},
};

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("could not read model config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("model config {path} is invalid JSON: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("model config does not match the runtime contract: {0}")]
    Mismatch(String),
}

/// Re-check the downloaded config independently from the Python indexer.
/// Downloaded metadata may describe tensors, but cannot promote itself into a
/// different compiled family or silently alter dimensions/router semantics.
pub fn validate_model_config(
    model_root: impl AsRef<Path>,
    manifest: &RuntimeManifest,
    descriptor: &AdapterDescriptor,
) -> Result<(), ProfileError> {
    let path = model_root.as_ref().join("config.json");
    let bytes = fs::read(&path).map_err(|source| ProfileError::Read {
        path: path.clone(),
        source,
    })?;
    let root: Value = serde_json::from_slice(&bytes).map_err(|source| ProfileError::Json {
        path: path.clone(),
        source,
    })?;
    let outer = root
        .as_object()
        .ok_or_else(|| ProfileError::Mismatch("config.json must contain an object".into()))?;
    let language = language_config(outer);
    let contract = &manifest.architecture;

    if let Some(model_type) = string(outer, &["model_type"]) {
        if contract.model_type != model_type {
            return Err(ProfileError::Mismatch(format!(
                "model_type is {model_type:?}, manifest declares {:?}",
                contract.model_type
            )));
        }
    }
    compare_u32(
        "num_layers",
        contract.num_layers,
        number(
            language,
            outer,
            &["num_hidden_layers", "n_layer", "n_layers", "num_layers"],
        ),
    )?;
    compare_u32(
        "hidden_size",
        contract.hidden_size,
        number(language, outer, &["hidden_size", "d_model", "model_dim"]),
    )?;
    compare_u32(
        "experts_per_layer",
        contract.experts_per_layer,
        number(
            language,
            outer,
            &[
                "num_local_experts",
                "n_routed_experts",
                "num_experts",
                "moe_num_experts",
            ],
        ),
    )?;
    compare_u32(
        "experts_per_token",
        contract.experts_per_token,
        number(
            language,
            outer,
            &[
                "num_experts_per_tok",
                "num_experts_per_token",
                "num_selected_experts",
                "top_k",
                "moe_top_k",
            ],
        ),
    )?;
    compare_u32(
        "expert_intermediate_size",
        contract.expert_intermediate_size,
        number(
            language,
            outer,
            &[
                "moe_intermediate_size",
                "expert_intermediate_size",
                "ffn_hidden_size",
                "intermediate_size",
            ],
        ),
    )?;

    if let Some(raw) = string(language, &["hidden_act", "activation_function"])
        .or_else(|| string(outer, &["hidden_act", "activation_function"]))
    {
        let activation = match raw {
            "silu" | "swiglu" => ActivationKind::Silu,
            "gelu" => ActivationKind::Gelu,
            "gelu_pytorch_tanh" | "gelu_tanh" => ActivationKind::GeluTanh,
            "relu" => ActivationKind::Relu,
            "relu2" | "relu_squared" => ActivationKind::Relu2,
            other => {
                return Err(ProfileError::Mismatch(format!(
                    "unsupported activation {other:?}"
                )))
            }
        };
        compare("activation", &contract.activation, &activation)?;
    }

    if let Some(score) =
        string(language, &["scoring_func"]).or_else(|| string(outer, &["scoring_func"]))
    {
        let score = match score {
            "softmax" => RouterScoreKind::Softmax,
            "sigmoid" => RouterScoreKind::Sigmoid,
            other => {
                return Err(ProfileError::Mismatch(format!(
                    "unsupported router score {other:?}"
                )))
            }
        };
        compare("router score", &contract.router_config.score, &score)?;
    }
    if let Some(normalize) = boolean(language, outer, "norm_topk_prob") {
        compare(
            "router normalization",
            &contract.router_config.normalize_selected,
            &normalize,
        )?;
    }
    if let Some(groups) = number(language, outer, &["n_group", "num_expert_groups"]) {
        compare("router groups", &contract.router_config.groups, &groups)?;
    }
    if let Some(groups) = number(language, outer, &["topk_group", "num_limited_groups"]) {
        compare(
            "selected router groups",
            &contract.router_config.selected_groups,
            &groups,
        )?;
    }
    compare_optional_u32(
        "attention heads",
        contract.attention_config.heads,
        number(language, outer, &["num_attention_heads", "n_heads"]),
    )?;
    compare_optional_u32(
        "KV heads",
        contract.attention_config.kv_heads,
        number(language, outer, &["num_key_value_heads", "n_kv_heads"]),
    )?;
    compare_optional_u32(
        "Q LoRA rank",
        contract.attention_config.q_lora_rank,
        number(language, outer, &["q_lora_rank"]),
    )?;
    compare_optional_u32(
        "KV LoRA rank",
        contract.attention_config.kv_lora_rank,
        number(language, outer, &["kv_lora_rank"]),
    )?;

    if descriptor.storage_only && descriptor.forward_kernel {
        return Err(ProfileError::Mismatch(
            "a storage-only adapter cannot expose a forward kernel".into(),
        ));
    }
    Ok(())
}

fn language_config(root: &Map<String, Value>) -> &Map<String, Value> {
    for key in ["text_config", "language_config", "llm_config"] {
        if let Some(value) = root.get(key).and_then(Value::as_object) {
            return value;
        }
    }
    root
}

fn number(language: &Map<String, Value>, root: &Map<String, Value>, keys: &[&str]) -> Option<u32> {
    for source in [
        Some(language),
        language.get("ffn_config").and_then(Value::as_object),
        Some(root),
        root.get("ffn_config").and_then(Value::as_object),
    ]
    .into_iter()
    .flatten()
    {
        for key in keys {
            if let Some(value) = source
                .get(*key)
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
            {
                if value > 0 {
                    return Some(value);
                }
            }
        }
    }
    None
}

fn string<'a>(source: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| source.get(*key).and_then(Value::as_str))
}

fn boolean(language: &Map<String, Value>, root: &Map<String, Value>, key: &str) -> Option<bool> {
    language
        .get(key)
        .or_else(|| root.get(key))
        .and_then(Value::as_bool)
}

fn compare<T: PartialEq + std::fmt::Debug>(
    name: &str,
    expected: &T,
    actual: &T,
) -> Result<(), ProfileError> {
    if expected != actual {
        return Err(ProfileError::Mismatch(format!(
            "{name} is {actual:?}, manifest declares {expected:?}"
        )));
    }
    Ok(())
}

fn compare_u32(name: &str, expected: u32, actual: Option<u32>) -> Result<(), ProfileError> {
    let actual =
        actual.ok_or_else(|| ProfileError::Mismatch(format!("config is missing {name}")))?;
    compare(name, &expected, &actual)
}

fn compare_optional_u32(
    name: &str,
    expected: u32,
    actual: Option<u32>,
) -> Result<(), ProfileError> {
    if expected == 0 && actual.is_none() {
        return Ok(());
    }
    if let Some(actual) = actual {
        compare(name, &expected, &actual)
    } else {
        Ok(())
    }
}
