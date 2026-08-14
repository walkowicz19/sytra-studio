//! Hugging Face tokenizer and chat-template integration.

use std::{
    fs,
    path::{Path, PathBuf},
};

use minijinja::{Environment, Error as TemplateError, ErrorKind, UndefinedBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer file {path} is unavailable: {reason}")]
    Load { path: PathBuf, reason: String },
    #[error("tokenizer operation failed: {0}")]
    Operation(String),
    #[error("chat template is unavailable or invalid: {0}")]
    ChatTemplate(String),
}

pub struct ModelTokenizer {
    tokenizer: Tokenizer,
    config: Value,
    chat_template: Option<String>,
    eos_token_ids: Vec<u32>,
}

impl std::fmt::Debug for ModelTokenizer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelTokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("has_chat_template", &self.chat_template.is_some())
            .field("eos_token_ids", &self.eos_token_ids)
            .finish()
    }
}

impl ModelTokenizer {
    pub fn load(model_root: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let root = model_root.as_ref();
        let tokenizer_path = root.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|error| TokenizerError::Load {
                path: tokenizer_path,
                reason: error.to_string(),
            })?;
        let config_path = root.join("tokenizer_config.json");
        let config = if config_path.is_file() {
            let bytes = fs::read(&config_path).map_err(|error| TokenizerError::Load {
                path: config_path.clone(),
                reason: error.to_string(),
            })?;
            serde_json::from_slice(&bytes).map_err(|error| TokenizerError::Load {
                path: config_path,
                reason: error.to_string(),
            })?
        } else {
            Value::Object(Map::new())
        };
        let chat_template = select_chat_template(config.get("chat_template"));
        let mut eos_token_ids = Vec::new();
        collect_token_strings(config.get("eos_token"), &mut |token| {
            if let Some(id) = tokenizer.token_to_id(token) {
                eos_token_ids.push(id);
            }
        });
        let generation_path = root.join("generation_config.json");
        if generation_path.is_file() {
            if let Ok(bytes) = fs::read(&generation_path) {
                if let Ok(generation) = serde_json::from_slice::<Value>(&bytes) {
                    collect_token_ids(generation.get("eos_token_id"), &mut eos_token_ids);
                }
            }
        }
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();
        Ok(Self {
            tokenizer,
            config,
            chat_template,
            eos_token_ids,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    pub fn is_eos(&self, token: u32) -> bool {
        self.eos_token_ids.binary_search(&token).is_ok()
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        self.tokenizer
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| TokenizerError::Operation(error.to_string()))
    }

    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|error| TokenizerError::Operation(error.to_string()))
    }

    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        tools: Option<&Value>,
        add_generation_prompt: bool,
    ) -> Result<String, TokenizerError> {
        let template = self.chat_template.as_deref().ok_or_else(|| {
            TokenizerError::ChatTemplate("tokenizer_config.json has no chat_template".into())
        })?;
        if messages.is_empty() {
            return Err(TokenizerError::ChatTemplate(
                "at least one chat message is required".into(),
            ));
        }
        let mut context = Map::new();
        context.insert(
            "messages".into(),
            serde_json::to_value(messages)
                .map_err(|error| TokenizerError::ChatTemplate(error.to_string()))?,
        );
        context.insert("add_generation_prompt".into(), add_generation_prompt.into());
        context.insert("tools".into(), tools.cloned().unwrap_or(Value::Null));
        if let Some(config) = self.config.as_object() {
            for (key, value) in config {
                if key.ends_with("_token") {
                    if let Some(content) = token_content(value) {
                        context.insert(key.clone(), Value::String(content.to_owned()));
                    }
                }
            }
        }
        let mut environment = Environment::new();
        environment.set_undefined_behavior(UndefinedBehavior::Strict);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, TemplateError> {
                Err(TemplateError::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment
            .render_str(template, Value::Object(context))
            .map_err(|error| TokenizerError::ChatTemplate(error.to_string()))
    }

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
        tools: Option<&Value>,
    ) -> Result<Vec<u32>, TokenizerError> {
        let prompt = self.apply_chat_template(messages, tools, true)?;
        self.encode(&prompt, false)
    }

    /// Decode an accumulated generation and return only its stable suffix.
    /// Decoding the whole short generated sequence handles byte-level BPE
    /// tokens whose individual token text is not valid UTF-8.
    pub fn decode_delta(
        &self,
        generated: &[u32],
        previous: &str,
    ) -> Result<(String, String), TokenizerError> {
        let decoded = self.decode(generated, true)?;
        let common = decoded
            .char_indices()
            .zip(previous.char_indices())
            .take_while(|((_, left), (_, right))| left == right)
            .map(|((index, character), _)| index + character.len_utf8())
            .last()
            .unwrap_or(0);
        let delta = decoded[common..].to_owned();
        Ok((decoded, delta))
    }
}

fn select_chat_template(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(template) => Some(template.clone()),
        Value::Array(templates) => templates
            .iter()
            .find(|entry| entry.get("name").and_then(Value::as_str) == Some("default"))
            .or_else(|| templates.first())
            .and_then(|entry| entry.get("template"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        Value::Object(templates) => templates
            .get("default")
            .or_else(|| templates.values().next())
            .and_then(Value::as_str)
            .map(str::to_owned),
        _ => None,
    }
}

fn token_content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}

fn collect_token_strings(value: Option<&Value>, visit: &mut impl FnMut(&str)) {
    match value {
        Some(Value::Array(values)) => {
            for value in values {
                collect_token_strings(Some(value), visit);
            }
        }
        Some(value) => {
            if let Some(content) = token_content(value) {
                visit(content);
            }
        }
        None => {}
    }
}

fn collect_token_ids(value: Option<&Value>, output: &mut Vec<u32>) {
    match value {
        Some(Value::Array(values)) => {
            for value in values {
                collect_token_ids(Some(value), output);
            }
        }
        Some(value) => {
            if let Some(id) = value.as_u64().and_then(|id| u32::try_from(id).ok()) {
                output.push(id);
            }
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

    use super::*;

    #[test]
    fn tokenizer_renders_chat_and_detects_eos() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sytra-tokenizer-{nonce}"));
        fs::create_dir_all(&root).unwrap();
        let vocab_path = root.join("vocab.json");
        fs::write(
            &vocab_path,
            br#"{"[UNK]":0,"user":1,"hello":2,"assistant":3,"[EOS]":4}"#,
        )
        .unwrap();
        let model = WordLevel::from_file(vocab_path.to_str().unwrap(), "[UNK]".into()).unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.save(root.join("tokenizer.json"), false).unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            serde_json::to_vec(&serde_json::json!({
                "eos_token": "[EOS]",
                "chat_template": "{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
            }))
            .unwrap(),
        )
        .unwrap();
        let tokenizer = ModelTokenizer::load(&root).unwrap();
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hello".into(),
        }];
        let rendered = tokenizer
            .apply_chat_template(&messages, None, true)
            .unwrap();
        assert_eq!(rendered, "user hello assistant");
        assert_eq!(
            tokenizer.encode_chat(&messages, None).unwrap(),
            vec![1, 2, 3]
        );
        assert!(tokenizer.is_eos(4));
        fs::remove_dir_all(root).unwrap();
    }
}
