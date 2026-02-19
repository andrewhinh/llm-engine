use std::collections::{HashSet, VecDeque};

use anyhow::{Result, ensure};

use crate::ipc::{
    AbortRequest, DetokenizeRequest, FrontendToTokenizerBatch, FrontendToTokenizerMsg, RequestId,
    SchedulerToTokenizerBatch, SchedulerToTokenizerMsg, TokenizeRequest, TokenizerToFrontendBatch,
    TokenizerToFrontendMsg, TokenizerToSchedulerBatch, TokenizerToSchedulerMsg,
};

use super::{DetokenizeManager, TokenizeManager};

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
