use sytra_contracts::merge_config::Verdict;
use sytra_contracts::Operation;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("merge compat verdict is red; this operation must not reach the runner")]
    RedVerdict,
    #[error("slerp accepts at most 2 models, got {0}")]
    SlerpTooManyModels(usize),
    #[error("model count {0} exceeds the product cap of 3")]
    ModelCountExceedsCap(usize),
    #[error("base_model is required for task-vector merge methods")]
    MissingBaseModel,
}

/// Enforces the hard rules from Contract 2 in code, right before the host
/// spawns the runner subprocess. This is the trust-boundary gate: every
/// field checked here was filled in by host-side resolution (catalog
/// lookups, `Guider.merge_check`) and must be re-validated, not assumed
/// safe just because it round-tripped through serde.
pub fn validate_before_spawn(op: &Operation) -> Result<(), ValidationError> {
    let Operation::Merge(spec) = op else {
        return Ok(());
    };
    let config = &spec.config;

    if config.compat.verdict == Verdict::Red {
        return Err(ValidationError::RedVerdict);
    }

    if config.models.len() > 3 {
        return Err(ValidationError::ModelCountExceedsCap(config.models.len()));
    }

    if config.merge_method == sytra_contracts::merge_config::MergeMethod::Slerp
        && config.models.len() > 2
    {
        return Err(ValidationError::SlerpTooManyModels(config.models.len()));
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
}
