use std::collections::{HashSet, VecDeque};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::ipc::{
    AbortRequest, DetokenizeRequest, FrontendToTokenizerBatch, FrontendToTokenizerMsg, RequestId,
    SchedulerToTokenizerBatch, SchedulerToTokenizerMsg, TokenizeRequest, TokenizerToFrontendBatch,
    TokenizerToFrontendMsg, TokenizerToSchedulerBatch, TokenizerToSchedulerMsg,
};
use crate::scheduler::SchedulerIoConfig;
use crate::scheduler::io::LLM_SCHED_NAMESPACE_ENV;
use crate::utils::tokenizer::{load_tokenizer, load_tokenizer_config, resolve_eos_id};

use super::{DetokenizeManager, TokenizeManager};

const FRONTEND_TO_TOKENIZER_PREFIX: &str = "frontend-tokenizer";
const FRONTEND_REPLY_SUFFIX: &str = "frontend-replies.sock";
const MAX_DATAGRAM_BYTES: usize = 2 * 1024 * 1024;
const IDLE_SLEEP: Duration = Duration::from_millis(10);

#[derive(Debug, Default, Clone, PartialEq)]
pub struct BridgeOutputBatches {
    pub to_scheduler: Vec<TokenizerToSchedulerBatch>,
    pub to_frontend: Vec<TokenizerToFrontendBatch>,
}

#[derive(Debug, Clone)]
pub struct TokenizerWorkerBridge {
    local_bs: usize,
    tokenize_manager: TokenizeManager,
    detokenize_manager: DetokenizeManager,
    pending: VecDeque<PendingMsg>,
    aborted: HashSet<RequestId>,
}

#[derive(Debug, Clone)]
enum PendingMsg {
    Tokenize(TokenizeRequest),
    Detokenize(DetokenizeRequest),
    Abort(AbortRequest),
}

impl TokenizerWorkerBridge {
    pub fn new(
        local_bs: usize,
        tokenize_manager: TokenizeManager,
        detokenize_manager: DetokenizeManager,
    ) -> Result<Self> {
        ensure!(local_bs > 0, "local_bs must be positive");
        Ok(Self {
            local_bs,
            tokenize_manager,
            detokenize_manager,
            pending: VecDeque::new(),
            aborted: HashSet::new(),
        })
    }

    pub fn enqueue_frontend_batch(&mut self, batch: FrontendToTokenizerBatch) {
        self.pending
            .extend(batch.data.into_iter().map(|msg| match msg {
                FrontendToTokenizerMsg::Tokenize(request) => PendingMsg::Tokenize(request),
                FrontendToTokenizerMsg::Abort(request) => PendingMsg::Abort(request),
            }));
    }

    pub fn enqueue_scheduler_batch(&mut self, batch: SchedulerToTokenizerBatch) {
        self.pending
            .extend(batch.data.into_iter().map(|msg| match msg {
                SchedulerToTokenizerMsg::Detokenize(request) => PendingMsg::Detokenize(request),
            }));
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn process_ready_batches(&mut self) -> Result<BridgeOutputBatches> {
        self.process(false)
    }

    pub fn flush(&mut self) -> Result<BridgeOutputBatches> {
        self.process(true)
    }

    fn process(&mut self, allow_partial: bool) -> Result<BridgeOutputBatches> {
        let mut outputs = BridgeOutputBatches::default();
        while self.pending.len() >= self.local_bs || (allow_partial && !self.pending.is_empty()) {
            let take = if self.pending.len() >= self.local_bs {
                self.local_bs
            } else {
                self.pending.len()
            };
            let mut tokenize_requests = Vec::new();
            let mut detokenize_requests = Vec::new();
            for _ in 0..take {
                match self.pending.pop_front() {
                    Some(PendingMsg::Tokenize(request))
                        if !self.aborted.contains(&request.request_id) =>
                    {
                        tokenize_requests.push(request);
                    }
                    Some(PendingMsg::Detokenize(request))
                        if !self.aborted.contains(&request.request_id) =>
                    {
                        detokenize_requests.push(request);
                    }
                    Some(PendingMsg::Abort(request)) => {
                        self.register_abort(request.request_id);
                    }
                    Some(_) | None => {}
                }
            }

            if !tokenize_requests.is_empty() {
                let tokenized = self.tokenize_manager.tokenize_batch(&tokenize_requests)?;
                let messages = tokenized
                    .into_iter()
                    .map(TokenizerToSchedulerMsg::User)
                    .collect::<Vec<_>>();
                outputs
                    .to_scheduler
                    .push(TokenizerToSchedulerBatch { data: messages });
            }
            if !detokenize_requests.is_empty() {
                let replies = self
                    .detokenize_manager
                    .detokenize_batch(&detokenize_requests)?;
                let messages = replies
                    .into_iter()
                    .map(TokenizerToFrontendMsg::Reply)
                    .collect::<Vec<_>>();
                outputs
                    .to_frontend
                    .push(TokenizerToFrontendBatch { data: messages });
            }
        }
        Ok(outputs)
    }

    fn register_abort(&mut self, request_id: RequestId) {
        self.aborted.insert(request_id);
        self.detokenize_manager.abort(request_id);
    }
}

pub fn frontend_to_tokenizer_path(namespace: &str, index: usize) -> PathBuf {
    PathBuf::from("/tmp").join(format!(
        "{namespace}-{FRONTEND_TO_TOKENIZER_PREFIX}-{index}.sock"
    ))
}

pub fn frontend_reply_path(namespace: &str) -> PathBuf {
    PathBuf::from("/tmp").join(format!("{namespace}-{FRONTEND_REPLY_SUFFIX}"))
}

pub fn run_tokenizer_worker(index: usize) -> Result<()> {
    let namespace = env::var(LLM_SCHED_NAMESPACE_ENV)
        .unwrap_or_else(|_| format!("llm-engine-sched-{}", std::process::id()));
    let model = env::var("LLM_MODEL")
        .or_else(|_| env::var("MODEL"))
        .map_err(|_| anyhow!("set LLM_MODEL (or MODEL) for tokenizer worker"))?;
    let model_path = Path::new(&model);
    let tokenizer = load_tokenizer(model_path)?;
    let tokenizer_cfg = load_tokenizer_config(model_path)?;
    let eos_token_id = resolve_eos_id(&tokenizer, tokenizer_cfg.as_ref(), -1)?;
    let local_bs = env::var("LLM_TOKENIZER_LOCAL_BS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1);

    let tokenize_manager = TokenizeManager::new(tokenizer.clone());
    let detokenize_manager = DetokenizeManager::new(tokenizer, eos_token_id as u32);
    let mut bridge = TokenizerWorkerBridge::new(local_bs, tokenize_manager, detokenize_manager)?;

    let recv_path = frontend_to_tokenizer_path(&namespace, index);
    remove_stale_socket(&recv_path)?;
    let recv_socket = UnixDatagram::bind(&recv_path)
        .with_context(|| format!("failed binding tokenizer socket {}", recv_path.display()))?;
    recv_socket
        .set_nonblocking(true)
        .context("failed setting tokenizer socket nonblocking")?;
    let send_socket = UnixDatagram::unbound().context("failed creating tokenizer send socket")?;
    let scheduler_path = SchedulerIoConfig::tokenizer_path_for_namespace(&namespace);

    loop {
        let mut progressed = false;
        if let Some(batch) = try_recv_datagram::<FrontendToTokenizerBatch>(&recv_socket)? {
            bridge.enqueue_frontend_batch(batch);
            progressed = true;
        }
        let out = bridge.process_ready_batches()?;
        if !out.to_scheduler.is_empty() {
            progressed = true;
        }
        for batch in out.to_scheduler {
            send_datagram(&send_socket, &scheduler_path, &batch, true)?;
        }
        if !progressed {
            thread::sleep(IDLE_SLEEP);
        }
    }
}

pub fn run_detokenizer_worker() -> Result<()> {
    let namespace = env::var(LLM_SCHED_NAMESPACE_ENV)
        .unwrap_or_else(|_| format!("llm-engine-sched-{}", std::process::id()));
    let model = env::var("LLM_MODEL")
        .or_else(|_| env::var("MODEL"))
        .map_err(|_| anyhow!("set LLM_MODEL (or MODEL) for detokenizer worker"))?;
    let model_path = Path::new(&model);
    let tokenizer = load_tokenizer(model_path)?;
    let tokenizer_cfg = load_tokenizer_config(model_path)?;
    let eos_token_id = resolve_eos_id(&tokenizer, tokenizer_cfg.as_ref(), -1)?;
    let local_bs = env::var("LLM_TOKENIZER_LOCAL_BS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1);

    let tokenize_manager = TokenizeManager::new(tokenizer.clone());
    let detokenize_manager = DetokenizeManager::new(tokenizer, eos_token_id as u32);
    let mut bridge = TokenizerWorkerBridge::new(local_bs, tokenize_manager, detokenize_manager)?;

    let recv_path = SchedulerIoConfig::detokenizer_path_for_namespace(&namespace);
    remove_stale_socket(&recv_path)?;
    let recv_socket = UnixDatagram::bind(&recv_path)
        .with_context(|| format!("failed binding detokenizer socket {}", recv_path.display()))?;
    recv_socket
        .set_nonblocking(true)
        .context("failed setting detokenizer socket nonblocking")?;
    let send_socket = UnixDatagram::unbound().context("failed creating detokenizer send socket")?;
    let frontend_path = frontend_reply_path(&namespace);

    loop {
        let mut progressed = false;
        if let Some(batch) = try_recv_datagram::<SchedulerToTokenizerBatch>(&recv_socket)? {
            bridge.enqueue_scheduler_batch(batch);
            progressed = true;
        }
        let out = bridge.process_ready_batches()?;
        if !out.to_frontend.is_empty() {
            progressed = true;
        }
        for batch in out.to_frontend {
            send_datagram(&send_socket, &frontend_path, &batch, true)?;
        }
        if !progressed {
            thread::sleep(IDLE_SLEEP);
        }
    }
}

fn send_datagram<T: Serialize>(
    socket: &UnixDatagram,
    path: &Path,
    payload: &T,
    ignore_not_found: bool,
) -> Result<()> {
    let bytes = serde_json::to_vec(payload).context("failed serializing tokenizer datagram")?;
    match socket.send_to(&bytes, path) {
        Ok(_) => Ok(()),
        Err(err)
            if ignore_not_found
                && matches!(
                    err.kind(),
                    ErrorKind::NotFound
                        | ErrorKind::ConnectionRefused
                        | ErrorKind::AddrNotAvailable
                ) =>
        {
            Ok(())
        }
        Err(err) => Err(err)
            .with_context(|| format!("failed sending tokenizer datagram to {}", path.display())),
    }
}

fn try_recv_datagram<T: DeserializeOwned>(socket: &UnixDatagram) -> Result<Option<T>> {
    let mut buffer = vec![0u8; MAX_DATAGRAM_BYTES];
    match socket.recv(&mut buffer) {
        Ok(size) => {
            if size == 0 {
                return Ok(None);
            }
            let parsed = serde_json::from_slice::<T>(&buffer[..size])
                .context("failed deserializing tokenizer datagram")?;
            Ok(Some(parsed))
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
        Err(err) => Err(err).context("failed receiving tokenizer datagram"),
    }
}

fn remove_stale_socket(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("failed removing stale socket {}", path.display()))?;
    }
    Ok(())
}
