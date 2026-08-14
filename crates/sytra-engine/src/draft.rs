//! Bounded tokenizer-compatible draft-model client for exact speculative decode.

use std::{io::Read, sync::Arc, time::Duration};

use reqwest::blocking::Client;
use serde_json::json;
use thiserror::Error;

use crate::ModelTokenizer;

const MAX_DRAFT_RESPONSE_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Error)]
pub enum DraftError {
    #[error("draft configuration is invalid: {0}")]
    Invalid(String),
    #[error("draft request failed: {0}")]
    Request(String),
    #[error("draft response is invalid: {0}")]
    Response(String),
}

pub trait DraftModel: Send + Sync + 'static {
    fn propose(&self, context: &[u32], max_tokens: usize) -> Result<Vec<u32>, DraftError>;
}

pub struct OpenAiDraftModel {
    endpoint: String,
    model: String,
    tokenizer: Arc<ModelTokenizer>,
    client: Client,
}

impl OpenAiDraftModel {
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        tokenizer: Arc<ModelTokenizer>,
        timeout: Duration,
    ) -> Result<Self, DraftError> {
        let endpoint = endpoint.into().trim_end_matches('/').to_owned();
        let model = model.into();
        if !(endpoint.starts_with("http://127.0.0.1:") || endpoint.starts_with("http://localhost:"))
        {
            return Err(DraftError::Invalid(
                "draft endpoint must be an explicit loopback HTTP URL".into(),
            ));
        }
        if model.trim().is_empty() || timeout.is_zero() {
            return Err(DraftError::Invalid(
                "draft model and positive timeout are required".into(),
            ));
        }
        let client = Client::builder()
            .connect_timeout(timeout.min(Duration::from_secs(5)))
            .timeout(timeout)
            .build()
            .map_err(|error| DraftError::Invalid(error.to_string()))?;
        Ok(Self {
            endpoint,
            model,
            tokenizer,
            client,
        })
    }
}

impl DraftModel for OpenAiDraftModel {
    fn propose(&self, context: &[u32], max_tokens: usize) -> Result<Vec<u32>, DraftError> {
        if context.is_empty() || max_tokens == 0 {
            return Err(DraftError::Invalid(
                "draft context and token budget must be non-empty".into(),
            ));
        }
        let prompt = self
            .tokenizer
            .decode(context, false)
            .map_err(|error| DraftError::Invalid(error.to_string()))?;
        let response = self
            .client
            .post(format!("{}/v1/completions", self.endpoint))
            .json(&json!({
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "stream": false,
            }))
            .send()
            .map_err(|error| DraftError::Request(error.to_string()))?;
        if !response.status().is_success() {
            return Err(DraftError::Response(format!(
                "HTTP {} from draft server",
                response.status()
            )));
        }
        if response
            .content_length()
            .is_some_and(|size| size > MAX_DRAFT_RESPONSE_BYTES)
        {
            return Err(DraftError::Response("response exceeds 1 MiB".into()));
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_DRAFT_RESPONSE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| DraftError::Response(error.to_string()))?;
        if bytes.len() as u64 > MAX_DRAFT_RESPONSE_BYTES {
            return Err(DraftError::Response("response exceeds 1 MiB".into()));
        }
        let payload: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| DraftError::Response(error.to_string()))?;
        let text = payload
            .pointer("/choices/0/text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| DraftError::Response("choices[0].text is missing".into()))?;
        let mut tokens = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| DraftError::Response(error.to_string()))?;
        tokens.truncate(max_tokens);
        if tokens.is_empty() {
            return Err(DraftError::Response("draft returned no tokens".into()));
        }
        Ok(tokens)
    }
}
