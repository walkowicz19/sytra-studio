//! Bounded OpenAI-compatible HTTP serving.

use std::{
    io::{Cursor, Read},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver},
        Arc,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;
use serde_json::{json, Value};
use thiserror::Error;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::{
    ChatMessage, DraftModel, GenerationConfig, GenerationError, GenerationOutput,
    GenerationRuntime, KimiRuntime, MixtralRuntime, ModelGenerator, ModelTokenizer,
};

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct CompletionBody {
    model: Option<String>,
    prompt: Value,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopSequences>,
    n: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatCompletionBody {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    tools: Option<Value>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopSequences>,
    n: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum InferencePrompt {
    Text(String),
    Chat {
        messages: Vec<ChatMessage>,
        tools: Option<Value>,
    },
}

#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub prompt: InferencePrompt,
    pub generation: GenerationConfig,
}

pub trait CompletionBackend: Send + Sync + 'static {
    fn model_id(&self) -> &str;
    fn health(&self) -> Result<Value, ServerError>;
    fn complete(
        &self,
        request: InferenceRequest,
        on_delta: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<GenerationOutput, ServerError>;
}

pub struct RuntimeCompletionBackend<R: GenerationRuntime + Send + Sync + 'static> {
    model_id: String,
    runtime: Arc<R>,
    tokenizer: Arc<ModelTokenizer>,
    draft: Option<Arc<dyn DraftModel>>,
    target_tokens_per_second: f32,
}

pub type KimiCompletionBackend = RuntimeCompletionBackend<KimiRuntime>;
pub type MixtralCompletionBackend = RuntimeCompletionBackend<MixtralRuntime>;

impl<R: GenerationRuntime + Send + Sync + 'static> RuntimeCompletionBackend<R> {
    pub fn new(
        model_id: impl Into<String>,
        runtime: Arc<R>,
        tokenizer: Arc<ModelTokenizer>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            runtime,
            tokenizer,
            draft: None,
            target_tokens_per_second: 5.0,
        }
    }

    pub fn with_draft(mut self, draft: Arc<dyn DraftModel>, target_tokens_per_second: f32) -> Self {
        self.draft = Some(draft);
        if target_tokens_per_second.is_finite() && target_tokens_per_second > 0.0 {
            self.target_tokens_per_second = target_tokens_per_second;
        }
        self
    }
}

impl<R: GenerationRuntime + Send + Sync + 'static> CompletionBackend
    for RuntimeCompletionBackend<R>
{
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn health(&self) -> Result<Value, ServerError> {
        let runtime = self
            .runtime
            .generation_health()
            .map_err(|error| ServerError::Backend(error.to_string()))?;
        Ok(json!({
            "ready": true,
            "model": self.model_id,
            "runtime": runtime,
        }))
    }

    fn complete(
        &self,
        request: InferenceRequest,
        on_delta: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<GenerationOutput, ServerError> {
        let tokens = match request.prompt {
            InferencePrompt::Text(prompt) => self
                .tokenizer
                .encode(&prompt, true)
                .map_err(|error| ServerError::Invalid(error.to_string()))?,
            InferencePrompt::Chat { messages, tools } => self
                .tokenizer
                .encode_chat(&messages, tools.as_ref())
                .map_err(|error| ServerError::Invalid(error.to_string()))?,
        };
        let generator = ModelGenerator::new(self.runtime.as_ref(), &self.tokenizer);
        match &self.draft {
            Some(draft) => generator.generate_speculative(
                &tokens,
                &request.generation,
                draft.as_ref(),
                self.target_tokens_per_second,
                on_delta,
            ),
            None => generator.generate(&tokens, &request.generation, on_delta),
        }
        .map_err(ServerError::Generation)
    }
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("inference backend failed: {0}")]
    Backend(String),
    #[error(transparent)]
    Generation(#[from] GenerationError),
    #[error("HTTP server failed: {0}")]
    Http(String),
}

pub struct OpenAiServer {
    bind: String,
    backend: Arc<dyn CompletionBackend>,
    max_concurrent_requests: usize,
}

impl OpenAiServer {
    pub fn new(
        bind: impl Into<String>,
        backend: Arc<dyn CompletionBackend>,
        max_concurrent_requests: usize,
    ) -> Result<Self, ServerError> {
        if max_concurrent_requests == 0 {
            return Err(ServerError::Invalid(
                "max concurrent requests must be positive".into(),
            ));
        }
        Ok(Self {
            bind: bind.into(),
            backend,
            max_concurrent_requests,
        })
    }

    pub fn run(self) -> Result<(), ServerError> {
        let server =
            Server::http(&self.bind).map_err(|error| ServerError::Http(error.to_string()))?;
        let limiter = Arc::new(RequestLimiter::new(self.max_concurrent_requests));
        for request in server.incoming_requests() {
            let Some(permit) = limiter.try_acquire() else {
                let _ = respond_error(request, StatusCode(429), "server request queue is full");
                continue;
            };
            let backend = self.backend.clone();
            thread::Builder::new()
                .name("sytra-http-request".into())
                .spawn(move || {
                    let _permit = permit;
                    if let Err(error) = handle_request(request, backend) {
                        eprintln!("sytra-engine request failed: {error}");
                    }
                })
                .map_err(|error| ServerError::Http(error.to_string()))?;
        }
        Ok(())
    }
}

fn handle_request(
    mut request: Request,
    backend: Arc<dyn CompletionBackend>,
) -> Result<(), ServerError> {
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or(request.url())
        .to_owned();
    match (request.method(), path.as_str()) {
        (&Method::Get, "/health") => match backend.health() {
            Ok(health) => respond_json(request, StatusCode(200), &health),
            Err(error) => respond_error(request, StatusCode(500), &error.to_string()),
        },
        (&Method::Get, "/v1/models") => respond_json(
            request,
            StatusCode(200),
            &json!({
                "object": "list",
                "data": [{
                    "id": backend.model_id(),
                    "object": "model",
                    "owned_by": "sytra"
                }]
            }),
        ),
        (&Method::Post, "/v1/completions") => {
            let body: CompletionBody = match read_json_body(&mut request) {
                Ok(body) => body,
                Err(error) => return respond_error(request, StatusCode(400), &error.to_string()),
            };
            let Some(prompt) = body.prompt.as_str().map(str::to_owned) else {
                return respond_error(
                    request,
                    StatusCode(400),
                    "only one string prompt is supported",
                );
            };
            if let Err(error) =
                validate_requested_model(body.model.as_deref(), backend.model_id(), body.n)
            {
                return respond_error(request, StatusCode(400), &error.to_string());
            }
            let generation = match generation_config(
                body.max_tokens,
                body.temperature,
                body.top_p,
                body.seed,
                body.stop,
            ) {
                Ok(config) => config,
                Err(error) => return respond_error(request, StatusCode(400), &error.to_string()),
            };
            complete_request(
                request,
                backend,
                InferenceRequest {
                    prompt: InferencePrompt::Text(prompt),
                    generation,
                },
                body.stream,
                false,
            )
        }
        (&Method::Post, "/v1/chat/completions") => {
            let body: ChatCompletionBody = match read_json_body(&mut request) {
                Ok(body) => body,
                Err(error) => return respond_error(request, StatusCode(400), &error.to_string()),
            };
            if let Err(error) =
                validate_requested_model(body.model.as_deref(), backend.model_id(), body.n)
            {
                return respond_error(request, StatusCode(400), &error.to_string());
            }
            let generation = match generation_config(
                body.max_tokens,
                body.temperature,
                body.top_p,
                body.seed,
                body.stop,
            ) {
                Ok(config) => config,
                Err(error) => return respond_error(request, StatusCode(400), &error.to_string()),
            };
            complete_request(
                request,
                backend,
                InferenceRequest {
                    prompt: InferencePrompt::Chat {
                        messages: body.messages,
                        tools: body.tools,
                    },
                    generation,
                },
                body.stream,
                true,
            )
        }
        _ => respond_error(request, StatusCode(404), "endpoint not found"),
    }
}

fn complete_request(
    request: Request,
    backend: Arc<dyn CompletionBackend>,
    inference: InferenceRequest,
    stream: bool,
    chat: bool,
) -> Result<(), ServerError> {
    let id = format!(
        "sytra-{}-{}",
        unix_seconds(),
        REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    );
    let model = backend.model_id().to_owned();
    if stream {
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(8);
        let stream_id = id.clone();
        let stream_model = model.clone();
        thread::Builder::new()
            .name("sytra-generation".into())
            .spawn(move || {
                if chat {
                    let initial = sse_chunk(&json!({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": unix_seconds(),
                        "model": stream_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
                    }));
                    if sender.send(initial).is_err() {
                        return;
                    }
                }
                let delta_id = stream_id.clone();
                let delta_model = stream_model.clone();
                let mut on_delta = |delta: &str| {
                    let event = if chat {
                        json!({
                            "id": delta_id,
                            "object": "chat.completion.chunk",
                            "created": unix_seconds(),
                            "model": delta_model,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
                        })
                    } else {
                        json!({
                            "id": delta_id,
                            "object": "text_completion",
                            "created": unix_seconds(),
                            "model": delta_model,
                            "choices": [{"index": 0, "text": delta, "finish_reason": null}]
                        })
                    };
                    sender.send(sse_chunk(&event)).map_err(|_| "client disconnected".to_owned())
                };
                match backend.complete(inference, &mut on_delta) {
                    Ok(output) => {
                        let final_event = if chat {
                            json!({
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": unix_seconds(),
                                "model": stream_model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": output.finish_reason}],
                                "usage": usage(&output)
                            })
                        } else {
                            json!({
                                "id": stream_id,
                                "object": "text_completion",
                                "created": unix_seconds(),
                                "model": stream_model,
                                "choices": [{"index": 0, "text": "", "finish_reason": output.finish_reason}],
                                "usage": usage(&output)
                            })
                        };
                        let _ = sender.send(sse_chunk(&final_event));
                    }
                    Err(error) => {
                        let _ = sender.send(sse_chunk(&json!({
                            "error": {"message": error.to_string(), "type": "server_error"}
                        })));
                    }
                }
                let _ = sender.send(b"data: [DONE]\n\n".to_vec());
            })
            .map_err(|error| ServerError::Http(error.to_string()))?;
        let response = Response::new(
            StatusCode(200),
            vec![content_type("text/event-stream"), no_cache()],
            ChannelReader::new(receiver),
            None,
            None,
        );
        request
            .respond(response)
            .map_err(|error| ServerError::Http(error.to_string()))
    } else {
        let mut discard = |_delta: &str| Ok(());
        match backend.complete(inference, &mut discard) {
            Ok(output) => {
                let response = if chat {
                    json!({
                        "id": id,
                        "object": "chat.completion",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": output.text},
                            "finish_reason": output.finish_reason
                        }],
                        "usage": usage(&output),
                        "sytra": {"tokens_per_second": output.tokens_per_second}
                    })
                } else {
                    json!({
                        "id": id,
                        "object": "text_completion",
                        "created": unix_seconds(),
                        "model": model,
                        "choices": [{"index": 0, "text": output.text, "finish_reason": output.finish_reason}],
                        "usage": usage(&output),
                        "sytra": {"tokens_per_second": output.tokens_per_second}
                    })
                };
                respond_json(request, StatusCode(200), &response)
            }
            Err(error) => respond_error(request, StatusCode(500), &error.to_string()),
        }
    }
}

fn generation_config(
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopSequences>,
) -> Result<GenerationConfig, ServerError> {
    let config = GenerationConfig {
        max_tokens: max_tokens.unwrap_or(256),
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        seed: seed.unwrap_or(0),
        stop: stop.map(StopSequences::into_vec).unwrap_or_default(),
        ..GenerationConfig::default()
    };
    if config.stop.len() > 16 || config.stop.iter().any(|stop| stop.len() > 1024) {
        return Err(ServerError::Invalid(
            "at most 16 stop strings of 1024 bytes are supported".into(),
        ));
    }
    Ok(config)
}

fn validate_requested_model(
    requested: Option<&str>,
    actual: &str,
    n: Option<usize>,
) -> Result<(), ServerError> {
    if requested.is_some_and(|requested| requested != actual) {
        return Err(ServerError::Invalid(format!(
            "requested model does not match served model {actual}"
        )));
    }
    if n.unwrap_or(1) != 1 {
        return Err(ServerError::Invalid(
            "only one bounded completion per request is supported".into(),
        ));
    }
    Ok(())
}

fn read_json_body<T: for<'de> Deserialize<'de>>(request: &mut Request) -> Result<T, ServerError> {
    let mut bytes = Vec::new();
    request
        .as_reader()
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| ServerError::Invalid(error.to_string()))?;
    if bytes.len() as u64 > MAX_REQUEST_BYTES {
        return Err(ServerError::Invalid("request body exceeds 1 MiB".into()));
    }
    serde_json::from_slice(&bytes).map_err(|error| ServerError::Invalid(error.to_string()))
}

fn respond_json(request: Request, status: StatusCode, value: &Value) -> Result<(), ServerError> {
    let bytes = serde_json::to_vec(value).map_err(|error| ServerError::Http(error.to_string()))?;
    request
        .respond(Response::new(
            status,
            vec![content_type("application/json")],
            Cursor::new(bytes.clone()),
            Some(bytes.len()),
            None,
        ))
        .map_err(|error| ServerError::Http(error.to_string()))
}

fn respond_error(request: Request, status: StatusCode, message: &str) -> Result<(), ServerError> {
    respond_json(
        request,
        status,
        &json!({"error": {"message": message, "type": "invalid_request_error"}}),
    )
}

fn usage(output: &GenerationOutput) -> Value {
    json!({
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens
    })
}

fn sse_chunk(value: &Value) -> Vec<u8> {
    format!("data: {}\n\n", value).into_bytes()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn content_type(value: &str) -> Header {
    Header::from_bytes("Content-Type", value).expect("valid content-type header")
}

fn no_cache() -> Header {
    Header::from_bytes("Cache-Control", "no-cache").expect("valid cache-control header")
}

struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    current: Cursor<Vec<u8>>,
    closed: bool,
}

impl ChannelReader {
    fn new(receiver: Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            current: Cursor::new(Vec::new()),
            closed: false,
        }
    }
}

impl Read for ChannelReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let read = self.current.read(output)?;
            if read > 0 || self.closed {
                return Ok(read);
            }
            match self.receiver.recv() {
                Ok(bytes) => self.current = Cursor::new(bytes),
                Err(_) => self.closed = true,
            }
        }
    }
}

struct RequestLimiter {
    active: AtomicUsize,
    maximum: usize,
}

impl RequestLimiter {
    fn new(maximum: usize) -> Self {
        Self {
            active: AtomicUsize::new(0),
            maximum,
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<RequestPermit> {
        let mut current = self.active.load(Ordering::Acquire);
        loop {
            if current >= self.maximum {
                return None;
            }
            match self.active.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(RequestPermit(self.clone())),
                Err(actual) => current = actual,
            }
        }
    }
}

struct RequestPermit(Arc<RequestLimiter>);

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::TcpStream;

    use super::*;

    struct MockBackend;

    impl CompletionBackend for MockBackend {
        fn model_id(&self) -> &str {
            "mock-moe"
        }

        fn health(&self) -> Result<Value, ServerError> {
            Ok(json!({"ready": true}))
        }

        fn complete(
            &self,
            _request: InferenceRequest,
            on_delta: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<GenerationOutput, ServerError> {
            on_delta("hello").map_err(ServerError::Backend)?;
            Ok(GenerationOutput {
                text: "hello".into(),
                prompt_tokens: 2,
                completion_tokens: 1,
                generated_tokens: vec![1],
                finish_reason: "stop".into(),
                elapsed_seconds: 0.1,
                tokens_per_second: 10.0,
                metrics: Default::default(),
            })
        }
    }

    #[test]
    fn request_limiter_never_exceeds_its_bound() {
        let limiter = Arc::new(RequestLimiter::new(2));
        let first = limiter.try_acquire().unwrap();
        let second = limiter.try_acquire().unwrap();
        assert!(limiter.try_acquire().is_none());
        drop(first);
        assert!(limiter.try_acquire().is_some());
        drop(second);
    }

    #[test]
    fn generation_request_rejects_unbounded_fanout_and_stop_payloads() {
        assert!(validate_requested_model(None, "model", Some(2)).is_err());
        let stop = StopSequences::Many(vec!["x".repeat(1025)]);
        assert!(generation_config(None, None, None, None, Some(stop)).is_err());
    }

    #[test]
    fn openai_completion_stream_uses_bounded_sse_channel() {
        let server = Server::http("127.0.0.1:0").unwrap();
        let address = server.server_addr().to_ip().unwrap();
        let handler = thread::spawn(move || {
            let request = server.recv().unwrap();
            handle_request(request, Arc::new(MockBackend)).unwrap();
        });
        let body = r#"{"model":"mock-moe","prompt":"hi","stream":true,"max_tokens":1}"#;
        let mut socket = TcpStream::connect(address).unwrap();
        write!(
            socket,
            "POST /v1/completions HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
        socket.shutdown(std::net::Shutdown::Write).unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).unwrap();
        handler.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200"));
        assert!(response.contains("data: [DONE]"));
        assert!(response.contains("hello"));
    }
}
