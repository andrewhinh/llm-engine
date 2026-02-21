use std::convert::Infallible;
use std::env;
use std::fs;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};

use crate::StreamOutput;
use crate::api::LLM;
use crate::ipc::{
    FrontendToTokenizerBatch, FrontendToTokenizerMsg, PromptPayload, TokenizeRequest,
    TokenizerToFrontendBatch, TokenizerToFrontendMsg,
};
use crate::server::streaming::{chunk_event, done_event, finish_chunk, token_chunk};
use crate::tokenizer::{frontend_reply_path, frontend_to_tokenizer_path};
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::tokenizer::{encode_prompt, load_tokenizer};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
}

impl ServerConfig {
    pub fn from_env() -> Result<Self> {
        let model = env::var("LLM_MODEL")
            .or_else(|_| env::var("MODEL"))
            .map_err(|_| anyhow!("set LLM_MODEL (or MODEL) to model directory path"))?;
        let host = env::var("LLM_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port = env::var("LLM_PORT")
            .ok()
            .and_then(|raw| raw.parse::<u16>().ok())
            .unwrap_or(8000);
        Ok(Self { model, host, port })
    }
}

#[derive(Clone)]
struct ServerState {
    backend: Arc<ServerBackend>,
    model_id: String,
}

enum ServerBackend {
    Local {
        llm: Box<Mutex<LLM>>,
    },
    Worker {
        client: Box<Mutex<WorkerFrontendClient>>,
    },
}

struct WorkerFrontendClient {
    namespace: String,
    tokenizer_workers: usize,
    next_worker: usize,
    request_seq: AtomicU64,
    send_socket: UnixDatagram,
    recv_socket: UnixDatagram,
    recv_path: PathBuf,
    tokenizer: tokenizers::Tokenizer,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    prompt: Option<String>,
    messages: Option<Vec<ChatMessage>>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<isize>,
    top_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    ignore_eos: Option<bool>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatResponseMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct ChatResponseMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
struct ModelCard {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

pub async fn run_server(config: ServerConfig) -> Result<()> {
    let backend = if should_use_worker_backend() {
        Arc::new(ServerBackend::Worker {
            client: Box::new(Mutex::new(WorkerFrontendClient::from_env(&config.model)?)),
        })
    } else {
        let engine_config = EngineConfig {
            model: config.model.clone(),
            ..EngineConfig::default()
        };
        Arc::new(ServerBackend::Local {
            llm: Box::new(Mutex::new(LLM::new(engine_config)?)),
        })
    };

    let state = ServerState {
        backend,
        model_id: config.model,
    };

    let app = Router::new()
        .route("/v1", get(v1_health))
        .route("/v1/models", get(v1_models))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn v1_health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn v1_models(State(state): State<ServerState>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelCard {
            id: state.model_id,
            object: "model",
            created: now_unix_secs(),
            owned_by: "llm-engine",
        }],
    })
}

async fn v1_chat_completions(
    State(state): State<ServerState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let model = request
        .model
        .clone()
        .unwrap_or_else(|| state.model_id.clone());
    let prompt = match normalize_prompt(request.prompt.clone(), request.messages.as_deref()) {
        Ok(prompt) => prompt,
        Err(err) => return (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
    };
    let sampling = sampling_from_request(&request);

    if request.stream {
        return stream_chat_completion(state, model, prompt, sampling).into_response();
    }

    match tokio::task::spawn_blocking(move || -> Result<(usize, String, Vec<u32>)> {
        match &*state.backend {
            ServerBackend::Local { llm } => {
                let mut llm = llm
                    .lock()
                    .map_err(|_| anyhow!("failed to acquire llm lock"))?;
                let prompt_tokens = llm.count_prompt_tokens(&prompt)?;
                let prompts = vec![prompt];
                let params = vec![sampling];
                let mut outputs = llm.generate(&prompts, &params)?;
                ensure!(!outputs.is_empty(), "engine returned no outputs");
                let output = outputs.remove(0);
                Ok((prompt_tokens, output.text, output.token_ids))
            }
            ServerBackend::Worker { client } => {
                let mut client = client
                    .lock()
                    .map_err(|_| anyhow!("failed to acquire worker client lock"))?;
                let prompt_tokens = client.count_prompt_tokens(&prompt)?;
                let (text, chunk_count) = client.generate_sync(prompt, sampling)?;
                Ok((prompt_tokens, text, vec![0u32; chunk_count]))
            }
        }
    })
    .await
    {
        Ok(Ok((prompt_tokens, text, token_ids))) => {
            let created = now_unix_secs();
            let completion_tokens = token_ids.len();
            let response = ChatCompletionResponse {
                id: completion_id(created),
                object: "chat.completion",
                created,
                model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant",
                        content: text,
                    },
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                },
            };
            Json(response).into_response()
        }
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation failed: {err}"),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("generation task failed: {err}"),
        )
            .into_response(),
    }
}

fn stream_chat_completion(
    state: ServerState,
    model: String,
    prompt: String,
    sampling: SamplingParams,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(128);
    let created = now_unix_secs();
    let completion_id = completion_id(created);

    tokio::task::spawn_blocking(move || {
        let mut sent_terminal = false;
        let mut sent_first_role = false;

        let result = (|| -> Result<()> {
            match &*state.backend {
                ServerBackend::Local { llm } => {
                    let mut llm = llm
                        .lock()
                        .map_err(|_| anyhow!("failed to acquire llm lock"))?;
                    let prompts = vec![prompt];
                    let params = vec![sampling];
                    let stream = llm.generate_stream(&prompts, &params)?;

                    for item in stream {
                        match item? {
                            StreamOutput::Token { text, .. } => {
                                let chunk = token_chunk(
                                    &completion_id,
                                    created,
                                    &model,
                                    text,
                                    !sent_first_role,
                                );
                                sent_first_role = true;
                                if tx.blocking_send(Ok(chunk_event(&chunk))).is_err() {
                                    return Ok(());
                                }
                            }
                            StreamOutput::Done(_) => {
                                let chunk = finish_chunk(&completion_id, created, &model);
                                if tx.blocking_send(Ok(chunk_event(&chunk))).is_err() {
                                    return Ok(());
                                }
                                if tx.blocking_send(Ok(done_event())).is_err() {
                                    return Ok(());
                                }
                                sent_terminal = true;
                                return Ok(());
                            }
                            StreamOutput::Cancelled(_) => {
                                if tx
                                    .blocking_send(Ok(
                                        Event::default().data("{\"error\":\"request cancelled\"}")
                                    ))
                                    .is_err()
                                {
                                    return Ok(());
                                }
                                if tx.blocking_send(Ok(done_event())).is_err() {
                                    return Ok(());
                                }
                                sent_terminal = true;
                                return Ok(());
                            }
                        }
                    }
                }
                ServerBackend::Worker { client } => {
                    let mut client = client
                        .lock()
                        .map_err(|_| anyhow!("failed to acquire worker client lock"))?;
                    client.stream(prompt, sampling, |delta| {
                        if delta.is_empty() {
                            return Ok(());
                        }
                        let chunk =
                            token_chunk(&completion_id, created, &model, delta, !sent_first_role);
                        sent_first_role = true;
                        if tx.blocking_send(Ok(chunk_event(&chunk))).is_err() {
                            return Ok(());
                        }
                        Ok(())
                    })?;
                    let chunk = finish_chunk(&completion_id, created, &model);
                    if tx.blocking_send(Ok(chunk_event(&chunk))).is_err() {
                        return Ok(());
                    }
                    if tx.blocking_send(Ok(done_event())).is_err() {
                        return Ok(());
                    }
                    sent_terminal = true;
                    return Ok(());
                }
            }
            Ok(())
        })();

        if let Err(err) = result {
            let _ = tx.blocking_send(Ok(Event::default().data(format!("{{\"error\":\"{err}\"}}"))));
        }
        if !sent_terminal {
            let _ = tx.blocking_send(Ok(done_event()));
        }
    });

    Sse::new(ReceiverStream::new(rx))
}

fn normalize_prompt(prompt: Option<String>, messages: Option<&[ChatMessage]>) -> Result<String> {
    if let Some(messages) = messages
        && !messages.is_empty()
    {
        let merged = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        ensure!(
            !merged.trim().is_empty(),
            "messages content must not be empty"
        );
        return Ok(merged);
    }

    let prompt = prompt.unwrap_or_default();
    ensure!(
        !prompt.trim().is_empty(),
        "either prompt or messages is required"
    );
    Ok(prompt)
}

fn sampling_from_request(request: &ChatCompletionRequest) -> SamplingParams {
    let mut params = SamplingParams::default();
    if let Some(max_tokens) = request.max_tokens {
        params.max_tokens = max_tokens;
    }
    if let Some(temperature) = request.temperature {
        params.temperature = temperature;
    }
    if let Some(top_k) = request.top_k {
        params.top_k = top_k;
    }
    if let Some(top_p) = request.top_p {
        params.top_p = top_p;
    }
    if let Some(frequency_penalty) = request.frequency_penalty {
        params.frequency_penalty = frequency_penalty;
    }
    if let Some(presence_penalty) = request.presence_penalty {
        params.presence_penalty = presence_penalty;
    }
    if let Some(ignore_eos) = request.ignore_eos {
        params.ignore_eos = ignore_eos;
    }
    params
}

fn completion_id(created: u64) -> String {
    format!("chatcmpl-{created}")
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn should_use_worker_backend() -> bool {
    env::var("LLM_SCHED_NAMESPACE").is_ok()
        && env::var("LLM_PROCESS_ROLE")
            .ok()
            .as_deref()
            .map(|role| role == "frontend")
            .unwrap_or(false)
}

impl WorkerFrontendClient {
    fn from_env(model: &str) -> Result<Self> {
        let namespace = env::var("LLM_SCHED_NAMESPACE")
            .map_err(|_| anyhow!("missing LLM_SCHED_NAMESPACE for worker frontend mode"))?;
        let tokenizer_workers = env::var("LLM_TOKENIZER_WORKERS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(0)
            .max(1);
        let recv_path = frontend_reply_path(&namespace);
        if recv_path.exists() {
            fs::remove_file(&recv_path).with_context(|| {
                format!(
                    "failed removing stale frontend reply socket {}",
                    recv_path.display()
                )
            })?;
        }
        let recv_socket = UnixDatagram::bind(&recv_path).with_context(|| {
            format!(
                "failed binding frontend reply socket {}",
                recv_path.display()
            )
        })?;
        let send_socket =
            UnixDatagram::unbound().context("failed creating frontend send socket")?;
        let tokenizer = load_tokenizer(Path::new(model))?;
        Ok(Self {
            namespace,
            tokenizer_workers,
            next_worker: 0,
            request_seq: AtomicU64::new(1),
            send_socket,
            recv_socket,
            recv_path,
            tokenizer,
        })
    }

    fn count_prompt_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(encode_prompt(&self.tokenizer, prompt)?.len())
    }

    fn generate_sync(
        &mut self,
        prompt: String,
        sampling: SamplingParams,
    ) -> Result<(String, usize)> {
        let request_id = self.submit(prompt, sampling)?;
        let mut output = String::new();
        let mut chunks = 0usize;
        loop {
            let reply = self.recv_reply_for(request_id)?;
            if !reply.incremental_output.is_empty() {
                output.push_str(&reply.incremental_output);
                chunks += 1;
            }
            if reply.finished {
                break;
            }
        }
        Ok((output, chunks))
    }

    fn stream<F>(&mut self, prompt: String, sampling: SamplingParams, mut on_delta: F) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        let request_id = self.submit(prompt, sampling)?;
        loop {
            let reply = self.recv_reply_for(request_id)?;
            on_delta(reply.incremental_output)?;
            if reply.finished {
                return Ok(());
            }
        }
    }

    fn submit(&mut self, prompt: String, sampling: SamplingParams) -> Result<u64> {
        let request_id = self.request_seq.fetch_add(1, Ordering::Relaxed);
        let req = TokenizeRequest {
            request_id,
            prompt: PromptPayload::Text(prompt),
            sampling_params: sampling,
        };
        let batch = FrontendToTokenizerBatch {
            data: vec![FrontendToTokenizerMsg::Tokenize(req)],
        };
        let idx = self.next_worker;
        self.next_worker = (self.next_worker + 1) % self.tokenizer_workers;
        let target = frontend_to_tokenizer_path(&self.namespace, idx);
        let bytes = serde_json::to_vec(&batch).context("failed serializing frontend request")?;
        self.send_socket
            .send_to(&bytes, &target)
            .with_context(|| format!("failed sending frontend request to {}", target.display()))?;
        Ok(request_id)
    }

    fn recv_reply_for(&self, request_id: u64) -> Result<crate::ipc::UserReply> {
        let mut buffer = vec![0u8; 2 * 1024 * 1024];
        loop {
            let size = self
                .recv_socket
                .recv(&mut buffer)
                .context("failed receiving frontend reply datagram")?;
            if size == 0 {
                continue;
            }
            let batch = serde_json::from_slice::<TokenizerToFrontendBatch>(&buffer[..size])
                .context("failed deserializing frontend reply datagram")?;
            for msg in batch.data {
                let TokenizerToFrontendMsg::Reply(reply) = msg;
                if reply.request_id == request_id {
                    return Ok(reply);
                }
            }
        }
    }
}

impl Drop for WorkerFrontendClient {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.recv_path);
    }
}
