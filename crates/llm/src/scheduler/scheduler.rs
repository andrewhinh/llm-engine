use std::collections::HashMap;
use std::env;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::core::{Engine, StepOutput};
use crate::ipc::{DetokenizeRequest, RequestId, TokenizerToSchedulerMsg};
use crate::scheduler::SchedulerIo;
use crate::scheduler::io::SchedulerIoConfig;
use crate::utils::EngineConfig;

const TP_SIZE_ENV: &str = "LLM_TP_SIZE";
const MODEL_ENV: &str = "LLM_MODEL";
const MODEL_FALLBACK_ENV: &str = "MODEL";
const DISABLE_OVERLAP_ENV: &str = "LLM_DISABLE_OVERLAP_SCHEDULING";
const SCHED_LOOP_INTERVAL_MS: u64 = 20;

#[derive(Debug)]
pub struct SchedulerWorker {
    io: SchedulerIo,
    engine: Engine,
    request_map: HashMap<usize, RequestId>,
    pending_step: Option<StepOutput>,
    overlap_enabled: bool,
    idle_steps: u64,
    processed_messages: u64,
}

impl SchedulerWorker {
    pub fn from_env(rank: usize) -> Result<Self> {
        let world_size = env::var(TP_SIZE_ENV)
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(1);
        let config = SchedulerIoConfig::from_env(rank, world_size)?;
        let io = SchedulerIo::new(config)?;
        let overlap_enabled = !read_env_flag(DISABLE_OVERLAP_ENV);
        let engine = engine_from_env()?;
        Ok(Self::from_parts(io, engine, overlap_enabled))
    }

    pub fn from_config(config: SchedulerIoConfig) -> Result<Self> {
        let io = SchedulerIo::new(config)?;
        let overlap_enabled = !read_env_flag(DISABLE_OVERLAP_ENV);
        let engine = engine_from_env()?;
        Ok(Self::from_parts(io, engine, overlap_enabled))
    }

    pub fn from_parts(io: SchedulerIo, engine: Engine, overlap_enabled: bool) -> Self {
        Self {
            io,
            engine,
            request_map: HashMap::new(),
            pending_step: None,
            overlap_enabled,
            idle_steps: 0,
            processed_messages: 0,
        }
    }

    pub fn tp_rank(&self) -> usize {
        self.io.tp_info().rank
    }

    pub fn tick(&mut self) -> Result<bool> {
        let blocking = self.pending_step.is_none() && self.engine.is_finished();
        let messages = self.io.receive_msg(blocking, || {
            self.idle_steps += 1;
        })?;
        if messages.is_empty() {
            self.idle_steps += 1;
        }
        if self.handle_messages(messages)? {
            self.flush_pending_step()?;
            return Ok(true);
        }

        let ongoing_step = if self.engine.is_finished() {
            None
        } else {
            Some(self.engine.step()?)
        };
        let (ready, next_pending) =
            plan_step_processing(self.overlap_enabled, self.pending_step.take(), ongoing_step);
        for step in ready {
            self.process_step(step)?;
        }
        self.pending_step = next_pending;

        Ok(false)
    }

    fn handle_messages(&mut self, messages: Vec<TokenizerToSchedulerMsg>) -> Result<bool> {
        for msg in messages {
            self.processed_messages += 1;
            match msg {
                TokenizerToSchedulerMsg::User(request) => {
                    let seq_id = self
                        .engine
                        .add_tokenized_request(request.input_ids, request.sampling_params)?;
                    self.request_map.insert(seq_id, request.request_id);
                }
                TokenizerToSchedulerMsg::Exit => {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn process_pending_step(&mut self) -> Result<()> {
        let (ready, next_pending) = plan_step_processing(false, self.pending_step.take(), None);
        for step in ready {
            self.process_step(step)?;
        }
        self.pending_step = next_pending;
        Ok(())
    }

    fn flush_pending_step(&mut self) -> Result<()> {
        self.process_pending_step()
    }

    fn process_step(&mut self, step: StepOutput) -> Result<()> {
        let reply = build_detokenize_replies(&step, &mut self.request_map);
        self.io.send_result(reply)
    }

    pub fn idle_steps(&self) -> u64 {
        self.idle_steps
    }

    pub fn processed_messages(&self) -> u64 {
        self.processed_messages
    }

    pub async fn run_until_shutdown(mut self) -> Result<()> {
        let mut interval = tokio::time::interval(Duration::from_millis(SCHED_LOOP_INTERVAL_MS));
        let shutdown_signal = wait_for_shutdown_signal();
        tokio::pin!(shutdown_signal);

        loop {
            tokio::select! {
                signal = &mut shutdown_signal => {
                    signal?;
                    self.flush_pending_step()?;
                    return Ok(());
                }
                _ = interval.tick() => {
                    if self.tick()? {
                        return Ok(());
                    }
                }
            }
        }
    }
}

pub async fn run_scheduler_worker(rank: usize) -> Result<()> {
    let worker = SchedulerWorker::from_env(rank)?;
    worker.run_until_shutdown().await
}

fn engine_from_env() -> Result<Engine> {
    let model = env::var(MODEL_ENV)
        .or_else(|_| env::var(MODEL_FALLBACK_ENV))
        .context("set LLM_MODEL (or MODEL) for scheduler worker")?;
    let config = EngineConfig {
        model,
        ..EngineConfig::default()
    };
    Engine::new(config)
}

fn read_env_flag(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn build_detokenize_replies(
    step: &StepOutput,
    request_map: &mut HashMap<usize, RequestId>,
) -> Vec<DetokenizeRequest> {
    let mut reply = Vec::new();
    for token in &step.tokens {
        if let Some(&request_id) = request_map.get(&token.seq_id) {
            reply.push(DetokenizeRequest {
                request_id,
                next_token: token.token_id,
                finished: token.finished,
            });
        }
    }
    for finished in &step.finished {
        request_map.remove(&finished.seq_id);
    }
    for cancelled in &step.cancelled {
        request_map.remove(&cancelled.seq_id);
    }
    reply
}

fn plan_step_processing(
    overlap_enabled: bool,
    pending_step: Option<StepOutput>,
    ongoing_step: Option<StepOutput>,
) -> (Vec<StepOutput>, Option<StepOutput>) {
    let mut ready = Vec::new();
    if let Some(step) = pending_step {
        ready.push(step);
    }
    if overlap_enabled {
        (ready, ongoing_step)
    } else {
        if let Some(step) = ongoing_step {
            ready.push(step);
        }
        (ready, None)
    }
}

async fn wait_for_shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term =
            signal(SignalKind::terminate()).context("failed installing SIGTERM handler")?;
        tokio::select! {
            signal = tokio::signal::ctrl_c() => {
                signal.context("failed waiting for ctrl-c")?;
            }
            _ = term.recv() => {}
        }
        Ok(())
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().context("failed waiting for ctrl-c")?;
        Ok(())
    }
}
