use std::env;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::ipc::TokenizerToSchedulerMsg;
use crate::scheduler::SchedulerIo;
use crate::scheduler::io::SchedulerIoConfig;

const TP_SIZE_ENV: &str = "LLM_TP_SIZE";
const SCHED_LOOP_INTERVAL_MS: u64 = 20;

#[derive(Debug)]
pub struct SchedulerWorker {
    io: SchedulerIo,
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
        Self::from_config(config)
    }

    pub fn from_config(config: SchedulerIoConfig) -> Result<Self> {
        let io = SchedulerIo::new(config)?;
        Ok(Self {
            io,
            idle_steps: 0,
            processed_messages: 0,
        })
    }

    pub fn tp_rank(&self) -> usize {
        self.io.tp_info().rank
    }

    pub fn tick(&mut self) -> Result<bool> {
        let messages = self.io.receive_msg(false, || {})?;
        if messages.is_empty() {
            self.idle_steps += 1;
            return Ok(false);
        }
        for msg in messages {
            self.processed_messages += 1;
            if matches!(msg, TokenizerToSchedulerMsg::Exit) {
                return Ok(true);
            }
        }
        Ok(false)
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
