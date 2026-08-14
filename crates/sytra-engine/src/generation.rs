//! Deterministic bounded text generation over a verified Kimi runtime.

use std::{collections::HashSet, time::Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    DraftModel, KimiDecodeState, KimiRuntime, KimiSpeculativeOutput, KimiStepMetrics,
    MemoryEnvelope, MixtralRuntime, ModelTokenizer, RuntimeError, SpeculativeController,
    StandardMoeKvState, TokenizerError,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stop: Vec<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: 0,
            stop: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub finish_reason: String,
    pub elapsed_seconds: f64,
    pub tokens_per_second: f64,
    pub metrics: KimiStepMetrics,
}

#[derive(Debug, Error)]
pub enum GenerationError {
    #[error("generation request is invalid: {0}")]
    Invalid(String),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("generation was cancelled: {0}")]
    Cancelled(String),
}

pub trait GenerationRuntime {
    type State;
    fn generation_memory(&self) -> &MemoryEnvelope;
    fn generation_max_positions(&self) -> usize;
    fn generation_vocab_size(&self) -> usize;
    fn generation_new_state(&self) -> Self::State;
    fn generation_logits_last(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<f32>, KimiStepMetrics), RuntimeError>;
    fn generation_prefill_next(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(u32, KimiStepMetrics), RuntimeError>;
    fn generation_verify_draft(
        &self,
        current: u32,
        draft: &[u32],
        state: &mut Self::State,
    ) -> Result<KimiSpeculativeOutput, RuntimeError>;
    fn generation_health(&self) -> Result<serde_json::Value, RuntimeError>;
}

impl GenerationRuntime for KimiRuntime {
    type State = KimiDecodeState;
    fn generation_memory(&self) -> &MemoryEnvelope {
        self.memory()
    }
    fn generation_max_positions(&self) -> usize {
        self.config().max_position_embeddings
    }
    fn generation_vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn generation_new_state(&self) -> Self::State {
        self.new_state()
    }
    fn generation_logits_last(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<f32>, KimiStepMetrics), RuntimeError> {
        self.logits_last(tokens, state)
    }
    fn generation_prefill_next(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(u32, KimiStepMetrics), RuntimeError> {
        self.prefill_next(tokens, state)
    }
    fn generation_verify_draft(
        &self,
        current: u32,
        draft: &[u32],
        state: &mut Self::State,
    ) -> Result<KimiSpeculativeOutput, RuntimeError> {
        self.verify_greedy_draft(current, draft, state)
    }
    fn generation_health(&self) -> Result<serde_json::Value, RuntimeError> {
        Ok(serde_json::json!({
            "memory": self.memory(),
            "placement": self.placement(),
            "metrics": self.metrics()?,
        }))
    }
}

impl GenerationRuntime for MixtralRuntime {
    type State = StandardMoeKvState;
    fn generation_memory(&self) -> &MemoryEnvelope {
        self.memory()
    }
    fn generation_max_positions(&self) -> usize {
        self.config().max_position_embeddings
    }
    fn generation_vocab_size(&self) -> usize {
        self.config().vocab_size
    }
    fn generation_new_state(&self) -> Self::State {
        self.new_state()
    }
    fn generation_logits_last(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(Vec<f32>, KimiStepMetrics), RuntimeError> {
        self.logits_last(tokens, state)
    }
    fn generation_prefill_next(
        &self,
        tokens: &[u32],
        state: &mut Self::State,
    ) -> Result<(u32, KimiStepMetrics), RuntimeError> {
        self.prefill_next(tokens, state)
    }
    fn generation_verify_draft(
        &self,
        current: u32,
        draft: &[u32],
        state: &mut Self::State,
    ) -> Result<KimiSpeculativeOutput, RuntimeError> {
        self.verify_greedy_draft(current, draft, state)
    }
    fn generation_health(&self) -> Result<serde_json::Value, RuntimeError> {
        Ok(serde_json::json!({
            "memory": self.memory(),
            "placement": self.placement(),
            "metrics": self.metrics()?,
        }))
    }
}

pub struct ModelGenerator<'a, R: GenerationRuntime> {
    runtime: &'a R,
    tokenizer: &'a ModelTokenizer,
}

pub type KimiGenerator<'a> = ModelGenerator<'a, KimiRuntime>;
pub type MixtralGenerator<'a> = ModelGenerator<'a, MixtralRuntime>;

impl<'a, R: GenerationRuntime> ModelGenerator<'a, R> {
    pub fn new(runtime: &'a R, tokenizer: &'a ModelTokenizer) -> Self {
        Self { runtime, tokenizer }
    }

    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        mut on_delta: impl FnMut(&str) -> Result<(), String>,
    ) -> Result<GenerationOutput, GenerationError> {
        validate_config(config)?;
        if prompt_tokens.is_empty() {
            return Err(GenerationError::Invalid(
                "prompt must contain at least one token".into(),
            ));
        }
        let context = usize::try_from(self.runtime.generation_memory().effective_context_tokens)
            .unwrap_or(usize::MAX)
            .min(self.runtime.generation_max_positions());
        let available = context.saturating_sub(prompt_tokens.len());
        if available == 0 {
            return Err(GenerationError::Invalid(format!(
                "prompt consumes the complete bounded context of {context} tokens"
            )));
        }
        let maximum = config.max_tokens.min(available);
        let started = Instant::now();
        let mut state = self.runtime.generation_new_state();
        let mut generated = Vec::with_capacity(maximum);
        let mut metrics = KimiStepMetrics::default();
        let (mut logits, step) = self
            .runtime
            .generation_logits_last(prompt_tokens, &mut state)?;
        metrics.merge(step);
        let mut random = XorShift64::new(config.seed);
        let mut decoded = String::new();
        let mut emitted = String::new();
        let mut finish_reason = "length".to_owned();
        let mut seen: HashSet<u32> = prompt_tokens.iter().copied().collect();

        for index in 0..maximum {
            apply_repetition_penalty(&mut logits, &seen, config.repetition_penalty);
            let token = sample_token(
                &logits,
                config.temperature,
                config.top_p,
                config.top_k,
                &mut random,
            )?;
            if self.tokenizer.is_eos(token) {
                finish_reason = "stop".into();
                break;
            }
            generated.push(token);
            seen.insert(token);
            decoded = self.tokenizer.decode(&generated, true)?;
            let stop_at = config
                .stop
                .iter()
                .filter_map(|stop| (!stop.is_empty()).then(|| decoded.find(stop)).flatten())
                .min();
            let visible = stop_at
                .map(|position| &decoded[..position])
                .unwrap_or(&decoded);
            let common = common_prefix_bytes(&emitted, visible);
            if common < emitted.len() {
                // Byte-level tokenizers can revise an incomplete trailing
                // character. Hold it until the decoded prefix is stable.
                emitted.truncate(common);
            }
            let delta = &visible[emitted.len()..];
            if !delta.is_empty() {
                on_delta(delta).map_err(GenerationError::Cancelled)?;
                emitted.push_str(delta);
            }
            if stop_at.is_some() {
                finish_reason = "stop".into();
                decoded = visible.to_owned();
                break;
            }
            if index + 1 < maximum {
                let output = self.runtime.generation_logits_last(&[token], &mut state)?;
                logits = output.0;
                metrics.merge(output.1);
            }
        }
        if finish_reason == "length" {
            decoded = emitted;
        }
        let elapsed = started.elapsed().as_secs_f64();
        let completion_tokens = generated.len();
        Ok(GenerationOutput {
            text: decoded,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            generated_tokens: generated,
            finish_reason,
            elapsed_seconds: elapsed,
            tokens_per_second: if elapsed > 0.0 {
                completion_tokens as f64 / elapsed
            } else {
                0.0
            },
            metrics,
        })
    }

    /// Exact greedy speculative decode. Draft failures fall back to one target
    /// step; target outputs remain authoritative at every accepted position.
    pub fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        draft: &dyn DraftModel,
        target_tokens_per_second: f32,
        mut on_delta: impl FnMut(&str) -> Result<(), String>,
    ) -> Result<GenerationOutput, GenerationError> {
        if config.temperature != 0.0
            || config.top_p != 1.0
            || config.top_k != 0
            || config.repetition_penalty != 1.0
        {
            return self.generate(prompt_tokens, config, on_delta);
        }
        validate_config(config)?;
        if prompt_tokens.is_empty() {
            return Err(GenerationError::Invalid(
                "prompt must contain at least one token".into(),
            ));
        }
        let context_limit =
            usize::try_from(self.runtime.generation_memory().effective_context_tokens)
                .unwrap_or(usize::MAX)
                .min(self.runtime.generation_max_positions());
        let maximum = config
            .max_tokens
            .min(context_limit.saturating_sub(prompt_tokens.len()));
        if maximum == 0 {
            return Err(GenerationError::Invalid(
                "prompt consumes the complete bounded context".into(),
            ));
        }
        let position_cap =
            usize::try_from(self.runtime.generation_memory().max_verification_positions)
                .unwrap_or(usize::MAX);
        if position_cap < 2 {
            return self.generate(prompt_tokens, config, on_delta);
        }
        let mut controller =
            SpeculativeController::new(1, position_cap - 1, target_tokens_per_second)
                .map_err(|error| GenerationError::Invalid(error.to_string()))?;
        let started = Instant::now();
        let mut state = self.runtime.generation_new_state();
        let (mut current, first_metrics) = self
            .runtime
            .generation_prefill_next(prompt_tokens, &mut state)?;
        let mut metrics = first_metrics;
        let mut generated = Vec::with_capacity(maximum);
        let mut decoded = String::new();
        let mut emitted = String::new();
        let mut finish_reason = "length".to_owned();

        while generated.len() < maximum {
            if self.tokenizer.is_eos(current) {
                finish_reason = "stop".into();
                break;
            }
            if emit_token(
                self.tokenizer,
                current,
                &mut generated,
                &mut decoded,
                &mut emitted,
                &config.stop,
                &mut on_delta,
            )? {
                finish_reason = "stop".into();
                break;
            }
            if generated.len() >= maximum {
                break;
            }
            let remaining = maximum - generated.len();
            let draft_count = controller
                .draft_tokens()
                .min(remaining)
                .min(position_cap - 1);
            let mut full_context = Vec::with_capacity(prompt_tokens.len() + generated.len());
            full_context.extend_from_slice(prompt_tokens);
            full_context.extend_from_slice(&generated);
            let proposals = draft.propose(&full_context, draft_count);
            let Ok(proposals) = proposals else {
                let output = self
                    .runtime
                    .generation_prefill_next(&[current], &mut state)?;
                current = output.0;
                metrics.merge(output.1);
                continue;
            };
            if proposals.is_empty() {
                let output = self
                    .runtime
                    .generation_prefill_next(&[current], &mut state)?;
                current = output.0;
                metrics.merge(output.1);
                continue;
            }
            let verification_started = Instant::now();
            let output = self
                .runtime
                .generation_verify_draft(current, &proposals, &mut state)?;
            metrics.merge(output.metrics);
            controller.observe(
                output.verification.drafted_tokens,
                output.verification.accepted_draft_tokens,
                output.verification.emitted_tokens.len(),
                verification_started
                    .elapsed()
                    .as_secs_f32()
                    .max(f32::EPSILON),
                position_cap - 1,
            );
            for token in output.verification.emitted_tokens {
                current = token;
                if generated.len() >= maximum || self.tokenizer.is_eos(current) {
                    if self.tokenizer.is_eos(current) {
                        finish_reason = "stop".into();
                    }
                    break;
                }
                if emit_token(
                    self.tokenizer,
                    current,
                    &mut generated,
                    &mut decoded,
                    &mut emitted,
                    &config.stop,
                    &mut on_delta,
                )? {
                    finish_reason = "stop".into();
                    break;
                }
            }
            if finish_reason == "stop" {
                break;
            }
        }
        if finish_reason == "length" {
            decoded = emitted;
        }
        let elapsed = started.elapsed().as_secs_f64();
        let completion_tokens = generated.len();
        Ok(GenerationOutput {
            text: decoded,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            generated_tokens: generated,
            finish_reason,
            elapsed_seconds: elapsed,
            tokens_per_second: if elapsed > 0.0 {
                completion_tokens as f64 / elapsed
            } else {
                0.0
            },
            metrics,
        })
    }
}

fn emit_token(
    tokenizer: &ModelTokenizer,
    token: u32,
    generated: &mut Vec<u32>,
    decoded: &mut String,
    emitted: &mut String,
    stops: &[String],
    on_delta: &mut impl FnMut(&str) -> Result<(), String>,
) -> Result<bool, GenerationError> {
    generated.push(token);
    *decoded = tokenizer.decode(generated, true)?;
    let stop_at = stops
        .iter()
        .filter_map(|stop| (!stop.is_empty()).then(|| decoded.find(stop)).flatten())
        .min();
    let visible = stop_at
        .map(|position| &decoded[..position])
        .unwrap_or(decoded);
    let common = common_prefix_bytes(emitted, visible);
    if common < emitted.len() {
        emitted.truncate(common);
    }
    let delta = &visible[emitted.len()..];
    if !delta.is_empty() {
        on_delta(delta).map_err(GenerationError::Cancelled)?;
        emitted.push_str(delta);
    }
    if stop_at.is_some() {
        *decoded = visible.to_owned();
        return Ok(true);
    }
    Ok(false)
}

fn validate_config(config: &GenerationConfig) -> Result<(), GenerationError> {
    if config.max_tokens == 0
        || !config.temperature.is_finite()
        || config.temperature < 0.0
        || !config.top_p.is_finite()
        || !(0.0 < config.top_p && config.top_p <= 1.0)
        || !config.repetition_penalty.is_finite()
        || config.repetition_penalty < 1.0
    {
        return Err(GenerationError::Invalid(
            "max_tokens, temperature, top_p, and repetition_penalty are outside supported bounds"
                .into(),
        ));
    }
    Ok(())
}

fn apply_repetition_penalty(logits: &mut [f32], seen: &HashSet<u32>, penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    for token in seen {
        if let Some(logit) = logits.get_mut(*token as usize) {
            *logit = if *logit >= 0.0 {
                *logit / penalty
            } else {
                *logit * penalty
            };
        }
    }
}

fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    random: &mut XorShift64,
) -> Result<u32, GenerationError> {
    if logits.is_empty() {
        return Err(GenerationError::Invalid("model returned no logits".into()));
    }
    if temperature == 0.0 {
        return logits
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| GenerationError::Invalid("all model logits are non-finite".into()));
    }
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(token, value)| (token as u32, *value / temperature))
        .collect();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if top_k > 0 {
        candidates.truncate(top_k.min(candidates.len()));
    }
    let maximum = candidates
        .first()
        .map(|(_, value)| *value)
        .ok_or_else(|| GenerationError::Invalid("all model logits are non-finite".into()))?;
    let mut total = 0.0_f64;
    for (_, value) in &mut candidates {
        *value = (*value - maximum).exp();
        total += f64::from(*value);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(GenerationError::Invalid(
            "sampling probabilities are non-finite".into(),
        ));
    }
    if top_p < 1.0 {
        let mut cumulative = 0.0_f64;
        let keep = candidates
            .iter()
            .position(|(_, probability)| {
                cumulative += f64::from(*probability) / total;
                cumulative >= f64::from(top_p)
            })
            .map(|index| index + 1)
            .unwrap_or(candidates.len());
        candidates.truncate(keep.max(1));
        total = candidates
            .iter()
            .map(|(_, probability)| f64::from(*probability))
            .sum();
    }
    let mut target = random.next_f64() * total;
    for (token, probability) in &candidates {
        target -= f64::from(*probability);
        if target <= 0.0 {
            return Ok(*token);
        }
    }
    Ok(candidates.last().expect("non-empty candidates").0)
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.char_indices()
        .zip(right.char_indices())
        .take_while(|((_, left), (_, right))| left == right)
        .map(|((index, character), _)| index + character.len_utf8())
        .last()
        .unwrap_or(0)
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        (value as f64) / (u64::MAX as f64 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded_sampling_are_deterministic() {
        let logits = [0.0, 2.0, 1.0, f32::NAN];
        let mut random = XorShift64::new(7);
        assert_eq!(sample_token(&logits, 0.0, 1.0, 0, &mut random).unwrap(), 1);
        let first = sample_token(&logits, 0.8, 0.9, 2, &mut XorShift64::new(9)).unwrap();
        let second = sample_token(&logits, 0.8, 0.9, 2, &mut XorShift64::new(9)).unwrap();
        assert_eq!(first, second);
        assert!(matches!(first, 1 | 2));
    }
}
