use sytra_contracts::merge_config::Verdict;
use sytra_contracts::Operation;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("merge compat verdict is red; this operation must not reach the runner")]
    RedVerdict,
    #[error("slerp accepts at most 2 models, got {0}")]
    SlerpTooManyModels(usize),
    #[error("merge method requires at least 2 models, got {0}")]
    TooFewModels(usize),
    #[error("model count {0} exceeds the limit for this merge method")]
    ModelCountExceedsCap(usize),
    #[error("base_model is required for task-vector merge methods")]
    MissingBaseModel,
    #[error("invalid MoE configuration: {0}")]
    InvalidMoe(String),
    #[error("invalid training configuration: {0}")]
    InvalidTraining(String),
}

/// Enforces the hard rules from Contract 2 in code, right before the host
/// spawns the runner subprocess. This is the trust-boundary gate: every
/// field checked here was filled in by host-side resolution (catalog
/// lookups, `Guider.merge_check`) and must be re-validated, not assumed
/// safe just because it round-tripped through serde.
pub fn validate_before_spawn(op: &Operation) -> Result<(), ValidationError> {
    if let Operation::Train(spec) = op {
        use sytra_contracts::run_config::{AdapterType, Packing, TrainMode};
        let config = &spec.config;
        if config.performance.get("training_engine").and_then(serde_json::Value::as_str) == Some("helix") {
            let has_router = config.adapter.moe_topology != "attention" || config.adapter.target_modules.iter()
                .any(|target| target == "router_projection" || target.ends_with(".router_projection"));
            let valid_adapter = (config.adapter.kind == AdapterType::Lora && matches!(config.adapter.quant_bits, None | Some(16))) ||
                (config.adapter.kind == AdapterType::Qlora && matches!(config.adapter.quant_bits, Some(4) | Some(8)));
            let packed = matches!(config.train.packing, Packing::Enabled(true)) ||
                config.performance.get("packing").and_then(serde_json::Value::as_bool) == Some(true);
            let has_horizon = config.train.epochs.is_some() || config.train.max_steps.is_some();
            if config.train_mode != TrainMode::Sft || !valid_adapter ||
                ((config.optim.router_lr_ratio.is_some() || config.optim.router_aux_coef.is_some_and(|v| v > 0.0)) && !has_router) ||
                (packed && (config.train.batch_size != 1 || config.backend.kind == sytra_contracts::run_config::BackendKind::Cpu)) ||
                !has_horizon {
                return Err(ValidationError::InvalidTraining("Helix requires SFT, unquantized LoRA or 4-bit/8-bit QLoRA, max_steps or epochs, and CUDA with batch size 1 for packing".into()));
            }
        }
        if config.optim.router_lr_ratio.is_some_and(|ratio| !ratio.is_finite() || ratio <= 0.0 || ratio > 1.0) {
            return Err(ValidationError::InvalidTraining("router LR ratio must be greater than 0 and at most 1".into()));
        }
        if config.optim.router_aux_coef.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value) || (value > 0.0 && config.train_mode != TrainMode::Sft)) {
            return Err(ValidationError::InvalidTraining("router auxiliary coefficient requires SFT and a value between 0 and 1".into()));
        }
        if !matches!(config.adapter.moe_topology.as_str(), "attention" | "moe_hybrid" | "moe_full") {
            return Err(ValidationError::InvalidTraining("unknown MoE LoRA topology".into()));
        }
        if config.adapter.use_rslora && matches!(config.adapter.kind, AdapterType::Full | AdapterType::Qat) {
            return Err(ValidationError::InvalidTraining("rsLoRA requires a LoRA adapter".into()));
        }
        if let Some(noise) = config.train.neftune_noise_alpha {
            if !noise.is_finite() || !(0.0..=100.0).contains(&noise) || (noise > 0.0 && config.train_mode != TrainMode::Sft) {
                return Err(ValidationError::InvalidTraining("NEFTune requires SFT and an alpha between 0 and 100".into()));
            }
        }
        return Ok(());
    }
    let Operation::Merge(spec) = op else {
        return Ok(());
    };
    let config = &spec.config;

    if config.compat.verdict == Verdict::Red {
        return Err(ValidationError::RedVerdict);
    }

    let cap = if config.merge_method == sytra_contracts::merge_config::MergeMethod::Moe { 16 } else { 3 };
    if config.models.len() > cap {
        return Err(ValidationError::ModelCountExceedsCap(config.models.len()));
    }

    if config.merge_method == sytra_contracts::merge_config::MergeMethod::Moe {
        let options = config.moe.as_ref().ok_or_else(|| ValidationError::InvalidMoe("missing moe options".into()))?;
        if config.base_model.as_deref().unwrap_or("").trim().is_empty() || config.models.len() < 2 {
            return Err(ValidationError::InvalidMoe("a base and at least two experts are required".into()));
        }
        if !matches!(options.architecture.as_str(), "mixtral" | "deepseek" | "qwen" | "qwen3")
            || options.experts_per_token == 0 || options.experts_per_token as usize > config.models.len()
            || options.shared_experts.len() > 16 {
            return Err(ValidationError::InvalidMoe("invalid architecture or expert count".into()));
        }
        if config.models.iter().any(|expert| expert.model.trim().is_empty() || expert.positive_prompts.is_empty()
            || expert.positive_prompts.iter().any(|p| p.trim().is_empty()) || expert.negative_prompts.iter().any(|p| p.trim().is_empty())) {
            return Err(ValidationError::InvalidMoe("each expert needs a model and non-empty positive prompts".into()));
        }
        if options.shared_experts.iter().any(|expert| expert.source_model.trim().is_empty()) {
            return Err(ValidationError::InvalidMoe("shared experts require a model".into()));
        }
        if options.architecture == "mixtral" && !options.shared_experts.is_empty() {
            return Err(ValidationError::InvalidMoe("Mixtral does not support shared experts".into()));
        }
        if options.architecture == "deepseek" && (options.shared_experts.len() > 1 || options.shared_experts.iter().any(|e| !e.positive_prompts.is_empty() || !e.negative_prompts.is_empty())) {
            return Err(ValidationError::InvalidMoe("DeepSeek supports at most one shared expert without routing prompts".into()));
        }
        if options.architecture == "qwen" && (options.shared_experts.len() != 1 || options.shared_experts[0].positive_prompts.is_empty()) {
            return Err(ValidationError::InvalidMoe("Qwen MoE requires one shared expert with positive routing prompts".into()));
        }
        let prompts: std::collections::HashSet<_> = config.models.iter().map(|m| (&m.positive_prompts, &m.negative_prompts)).collect();
        let models: std::collections::HashSet<_> = config.models.iter().map(|m| m.model.trim()).collect();
        if prompts.len() < 2 || models.len() < 2 { return Err(ValidationError::InvalidMoe("experts need distinct models and routing prompts".into())); }
        if let Some(first) = options.shared_experts.first() {
            if options.shared_experts.iter().any(|e| e.positive_prompts.is_empty() != first.positive_prompts.is_empty()
                || e.positive_prompts.iter().chain(&e.negative_prompts).any(|p| p.trim().is_empty())) {
                return Err(ValidationError::InvalidMoe("shared expert prompts must be consistent and non-empty".into()));
            }
        }
    }

    if config.merge_method == sytra_contracts::merge_config::MergeMethod::Slerp
        && config.models.len() > 2
    {
        return Err(ValidationError::SlerpTooManyModels(config.models.len()));
    }

    if matches!(
        config.merge_method,
        sytra_contracts::merge_config::MergeMethod::Slerp
            | sytra_contracts::merge_config::MergeMethod::MultiSlerp
            | sytra_contracts::merge_config::MergeMethod::Nuslerp
    ) && config.models.len() < 2
    {
        return Err(ValidationError::TooFewModels(config.models.len()));
    }

    if config.merge_method.is_task_vector() && config.base_model.is_none() {
        return Err(ValidationError::MissingBaseModel);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sytra_contracts::merge_config::{CompatConfig, MergeConfig, MergeMethod, ModelEntry};
    use sytra_contracts::merge_config::{
        MergeOutput, TokenizerConfig, TokenizerNamed, TokenizerSource,
    };
    use sytra_contracts::operation::MergeSpec;

    fn base_config(method: MergeMethod, n_models: usize, verdict: Verdict) -> MergeConfig {
        MergeConfig {
            moe: None,
            performance: Default::default(),
            version: 1,
            op_id: None,
            merge_method: method,
            base_model: Some("mistralai/Mistral-7B-v0.1".to_string()),
            dtype: "bfloat16".to_string(),
            models: (0..n_models)
                .map(|i| ModelEntry {
                    model: format!("org/model-{i}"),
                    parameters: None,
                    layer_range: None,
                    positive_prompts: vec![],
                    negative_prompts: vec![],
                })
                .collect(),
            parameters: None,
            tokenizer: TokenizerConfig {
                source: TokenizerSource::Named(TokenizerNamed::Base),
            },
            compat: CompatConfig {
                verdict,
                fingerprint: None,
            },
            output: MergeOutput {
                model_path: "out/merged".into(),
            },
        }
    }

    fn op(config: MergeConfig) -> Operation {
        Operation::Merge(MergeSpec {
            config,
            config_path: "merge.yaml".into(),
        })
    }

    #[test]
    fn rejects_red_verdict() {
        let result =
            validate_before_spawn(&op(base_config(MergeMethod::DareTies, 2, Verdict::Red)));
        assert_eq!(result, Err(ValidationError::RedVerdict));
    }

    #[test]
    fn rejects_slerp_with_more_than_two_models() {
        let result = validate_before_spawn(&op(base_config(MergeMethod::Slerp, 3, Verdict::Green)));
        assert_eq!(result, Err(ValidationError::SlerpTooManyModels(3)));
    }

    #[test]
    fn rejects_more_than_three_models() {
        let result =
            validate_before_spawn(&op(base_config(MergeMethod::Linear, 4, Verdict::Green)));
        assert_eq!(result, Err(ValidationError::ModelCountExceedsCap(4)));
    }

    #[test]
    fn rejects_task_vector_method_without_base_model() {
        let mut config = base_config(MergeMethod::Ties, 2, Verdict::Green);
        config.base_model = None;
        let result = validate_before_spawn(&op(config));
        assert_eq!(result, Err(ValidationError::MissingBaseModel));
    }

    #[test]
    fn accepts_valid_dare_ties_config() {
        let result =
            validate_before_spawn(&op(base_config(MergeMethod::DareTies, 2, Verdict::Green)));
        assert!(result.is_ok());
    }

    #[test]
    fn accepts_amber_verdict() {
        let result =
            validate_before_spawn(&op(base_config(MergeMethod::DareTies, 2, Verdict::Amber)));
        assert!(result.is_ok());
    }

    #[test]
    fn moe_accepts_up_to_sixteen_experts_with_distinct_prompts() {
        use sytra_contracts::merge_config::MoeOptions;
        let mut config = base_config(MergeMethod::Moe, 4, Verdict::Amber);
        config.moe = Some(MoeOptions { architecture: "mixtral".into(), experts_per_token: 2, shared_experts: vec![] });
        for (index, expert) in config.models.iter_mut().enumerate() {
            expert.positive_prompts = vec![format!("domain {index}")];
        }
        assert_eq!(validate_before_spawn(&op(config.clone())), Ok(()));
        config.models[0].positive_prompts.clear();
        assert!(matches!(validate_before_spawn(&op(config)), Err(ValidationError::InvalidMoe(_))));
    }

    #[test]
    fn moe_checks_architecture_specific_shared_experts() {
        use sytra_contracts::merge_config::{MoeExpert, MoeOptions};
        let mut config = base_config(MergeMethod::Moe, 2, Verdict::Amber);
        for (index, expert) in config.models.iter_mut().enumerate() {
            expert.positive_prompts = vec![format!("domain {index}")];
        }
        let shared = MoeExpert { source_model: "shared".into(), positive_prompts: vec![], negative_prompts: vec![] };
        config.moe = Some(MoeOptions { architecture: "mixtral".into(), experts_per_token: 2, shared_experts: vec![shared] });
        assert!(matches!(validate_before_spawn(&op(config.clone())), Err(ValidationError::InvalidMoe(_))));
        config.moe.as_mut().unwrap().architecture = "qwen".into();
        assert!(matches!(validate_before_spawn(&op(config.clone())), Err(ValidationError::InvalidMoe(_))));
        config.moe.as_mut().unwrap().shared_experts[0].positive_prompts.push("general".into());
        assert_eq!(validate_before_spawn(&op(config.clone())), Ok(()));
        config.moe.as_mut().unwrap().architecture = "deepseek".into();
        assert!(matches!(validate_before_spawn(&op(config)), Err(ValidationError::InvalidMoe(_))));
    }
}
