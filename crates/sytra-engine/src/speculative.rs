//! Exact speculative-decoding coordination.
//!
//! A small draft model proposes several tokens. The large streamed MoE then
//! verifies them as one batch, amortizing dense-weight I/O across accepted
//! tokens without changing the target model's greedy result.

use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GreedyVerification {
    pub emitted_tokens: Vec<u32>,
    pub accepted_draft_tokens: usize,
    pub drafted_tokens: usize,
    pub used_bonus_token: bool,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SpeculativeError {
    #[error("draft batch must contain at least one token")]
    EmptyDraft,
    #[error("target verification requires one prediction per draft token plus one bonus token")]
    MissingTargetPrediction,
}

/// Exact greedy verification. `target_predictions[i]` is the target model's
/// next token at draft position `i`; the final element is the bonus prediction
/// after a fully accepted draft.
pub fn verify_greedy(
    draft_tokens: &[u32],
    target_predictions: &[u32],
) -> Result<GreedyVerification, SpeculativeError> {
    if draft_tokens.is_empty() {
        return Err(SpeculativeError::EmptyDraft);
    }
    if target_predictions.len() != draft_tokens.len() + 1 {
        return Err(SpeculativeError::MissingTargetPrediction);
    }
    let mut emitted = Vec::with_capacity(draft_tokens.len() + 1);
    for (index, draft) in draft_tokens.iter().enumerate() {
        let target = target_predictions[index];
        if *draft != target {
            emitted.push(target);
            return Ok(GreedyVerification {
                emitted_tokens: emitted,
                accepted_draft_tokens: index,
                drafted_tokens: draft_tokens.len(),
                used_bonus_token: false,
            });
        }
        emitted.push(*draft);
    }
    emitted.push(target_predictions[draft_tokens.len()]);
    Ok(GreedyVerification {
        emitted_tokens: emitted,
        accepted_draft_tokens: draft_tokens.len(),
        drafted_tokens: draft_tokens.len(),
        used_bonus_token: true,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SpeculativeController {
    min_draft_tokens: usize,
    max_draft_tokens: usize,
    current_draft_tokens: usize,
    acceptance_ema: f32,
    target_tokens_per_second: f32,
}

impl SpeculativeController {
    pub fn new(
        min_draft_tokens: usize,
        max_draft_tokens: usize,
        target_tokens_per_second: f32,
    ) -> Result<Self, SpeculativeError> {
        if min_draft_tokens == 0
            || max_draft_tokens < min_draft_tokens
            || !target_tokens_per_second.is_finite()
            || target_tokens_per_second <= 0.0
        {
            return Err(SpeculativeError::EmptyDraft);
        }
        Ok(Self {
            min_draft_tokens,
            max_draft_tokens,
            current_draft_tokens: min_draft_tokens,
            acceptance_ema: 1.0,
            target_tokens_per_second,
        })
    }

    pub fn draft_tokens(&self) -> usize {
        self.current_draft_tokens
    }

    pub fn acceptance_ema(&self) -> f32 {
        self.acceptance_ema
    }

    /// Adapt lookahead while respecting the caller's current memory-derived
    /// position cap. Low acceptance shrinks immediately; high acceptance grows
    /// only when measured throughput remains below the target.
    pub fn observe(
        &mut self,
        drafted: usize,
        accepted: usize,
        emitted: usize,
        verification_seconds: f32,
        memory_position_cap: usize,
    ) {
        if drafted == 0 || !verification_seconds.is_finite() || verification_seconds <= 0.0 {
            return;
        }
        let acceptance = accepted.min(drafted) as f32 / drafted as f32;
        self.acceptance_ema = self.acceptance_ema * 0.8 + acceptance * 0.2;
        let throughput = emitted as f32 / verification_seconds;
        let hard_max = self.max_draft_tokens.min(memory_position_cap.max(1));
        if self.acceptance_ema < 0.55 {
            self.current_draft_tokens = (self.current_draft_tokens / 2).max(self.min_draft_tokens);
        } else if self.acceptance_ema > 0.8 && throughput < self.target_tokens_per_second {
            self.current_draft_tokens = (self.current_draft_tokens + 1).min(hard_max);
        }
        self.current_draft_tokens = self
            .current_draft_tokens
            .clamp(self.min_draft_tokens.min(hard_max), hard_max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_verification_is_target_exact_at_first_mismatch() {
        let result = verify_greedy(&[10, 11, 99, 13], &[10, 11, 12, 13, 14]).unwrap();
        assert_eq!(result.emitted_tokens, [10, 11, 12]);
        assert_eq!(result.accepted_draft_tokens, 2);
        assert!(!result.used_bonus_token);
    }

    #[test]
    fn fully_accepted_batch_emits_the_target_bonus_token() {
        let result = verify_greedy(&[10, 11, 12], &[10, 11, 12, 13]).unwrap();
        assert_eq!(result.emitted_tokens, [10, 11, 12, 13]);
        assert!(result.used_bonus_token);
    }

    #[test]
    fn controller_never_exceeds_the_memory_position_cap() {
        let mut controller = SpeculativeController::new(1, 32, 5.0).unwrap();
        for _ in 0..20 {
            controller.observe(4, 4, 5, 2.0, 6);
        }
        assert_eq!(controller.draft_tokens(), 6);
        controller.observe(6, 0, 1, 2.0, 6);
        controller.observe(3, 0, 1, 2.0, 6);
        controller.observe(1, 0, 1, 2.0, 6);
        assert!(controller.draft_tokens() < 6);
    }
}
