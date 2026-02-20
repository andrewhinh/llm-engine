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
    DetokenizeRequest, SchedulerRankControlMsg, SchedulerRankMsg, SchedulerToTokenizerBatch,
    SchedulerToTokenizerMsg, TokenizerToSchedulerBatch, TokenizerToSchedulerMsg,
};
use crate::models::TpInfo;

pub const LLM_SCHED_NAMESPACE_ENV: &str = "LLM_SCHED_NAMESPACE";

const IO_POLL_INTERVAL_MS: u64 = 20;
const MAX_DATAGRAM_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerIoConfig {
    pub tp: TpInfo,
    pub namespace: String,
    pub poll_interval: Duration,
}

impl SchedulerIoConfig {
    pub fn from_env(rank: usize, world_size: usize) -> Result<Self> {
        let tp = TpInfo::new(rank, world_size)?;
        let namespace = env::var(LLM_SCHED_NAMESPACE_ENV)
            .unwrap_or_else(|_| format!("llm-engine-sched-{}", std::process::id()));
        Ok(Self {
            tp,
            namespace,
            poll_interval: Duration::from_millis(IO_POLL_INTERVAL_MS),
        })
    }

    pub fn rank_path(&self, rank: usize) -> PathBuf {
        Self::rank_path_for_namespace(&self.namespace, rank)
    }

    pub fn tokenizer_path(&self) -> PathBuf {
        Self::tokenizer_path_for_namespace(&self.namespace)
    }

    pub fn detokenizer_path(&self) -> PathBuf {
        Self::detokenizer_path_for_namespace(&self.namespace)
    }

    pub fn rank_path_for_namespace(namespace: &str, rank: usize) -> PathBuf {
        PathBuf::from("/tmp").join(format!("{namespace}-scheduler-rank-{rank}.sock"))
    }

    pub fn tokenizer_path_for_namespace(namespace: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("{namespace}-scheduler-tokenizer.sock"))
    }

    pub fn detokenizer_path_for_namespace(namespace: &str) -> PathBuf {
        PathBuf::from("/tmp").join(format!("{namespace}-scheduler-detokenizer.sock"))
    }
}

#[derive(Debug)]
pub struct SchedulerIo {
    config: SchedulerIoConfig,
    rank_socket: UnixDatagram,
    tokenizer_socket: Option<UnixDatagram>,
    owned_paths: Vec<PathBuf>,
}

impl SchedulerIo {
    pub fn new(config: SchedulerIoConfig) -> Result<Self> {
        let rank_path = config.rank_path(config.tp.rank);
        remove_stale_socket(&rank_path)?;
        let rank_socket = UnixDatagram::bind(&rank_path)
            .with_context(|| format!("failed to bind rank socket {}", rank_path.display()))?;
        rank_socket
            .set_nonblocking(true)
            .context("failed to set rank socket nonblocking")?;

        let mut owned_paths = vec![rank_path];
        let tokenizer_socket = if config.tp.is_primary() {
            let tokenizer_path = config.tokenizer_path();
            remove_stale_socket(&tokenizer_path)?;
            let socket = UnixDatagram::bind(&tokenizer_path).with_context(|| {
                format!(
                    "failed to bind tokenizer ingress socket {}",
                    tokenizer_path.display()
                )
            })?;
            socket
                .set_nonblocking(true)
                .context("failed to set tokenizer socket nonblocking")?;
            owned_paths.push(tokenizer_path);
            Some(socket)
        } else {
            None
        };

        Ok(Self {
            config,
            rank_socket,
            tokenizer_socket,
            owned_paths,
        })
    }

    pub fn tp_info(&self) -> TpInfo {
        self.config.tp
    }

    pub fn run_when_idle(&self) {
        thread::sleep(self.config.poll_interval);
    }

    pub fn receive_msg<F>(
        &self,
        blocking: bool,
        mut on_idle: F,
    ) -> Result<Vec<TokenizerToSchedulerMsg>>
    where
        F: FnMut(),
    {
        if self.config.tp.is_primary() {
            self.receive_rank0_msg(blocking, &mut on_idle)
        } else {
            self.receive_non_primary_msg(blocking, &mut on_idle)
        }
    }

    pub fn broadcast_forward_batch(&self, batch: &TokenizerToSchedulerBatch) -> Result<()> {
        self.broadcast_rank_msg(&SchedulerRankMsg::ForwardBatch(batch.clone()))
    }

    pub fn broadcast_control(&self, control: SchedulerRankControlMsg) -> Result<()> {
        self.broadcast_rank_msg(&SchedulerRankMsg::Control(control))
    }

    pub fn send_result(&self, reply: Vec<DetokenizeRequest>) -> Result<()> {
        if !self.config.tp.is_primary() || reply.is_empty() {
            return Ok(());
        }
        let batch = SchedulerToTokenizerBatch {
            data: reply
                .into_iter()
                .map(SchedulerToTokenizerMsg::Detokenize)
                .collect(),
        };
        let detok_path = self.config.detokenizer_path();
        send_datagram(&self.rank_socket, &detok_path, &batch, true)
    }

    fn receive_rank0_msg<F>(
        &self,
        blocking: bool,
        on_idle: &mut F,
    ) -> Result<Vec<TokenizerToSchedulerMsg>>
    where
        F: FnMut(),
    {
        let tokenizer_socket = self
            .tokenizer_socket
            .as_ref()
            .ok_or_else(|| anyhow!("rank0 scheduler io missing tokenizer socket"))?;
        loop {
            if let Some(batch) = try_recv_datagram::<TokenizerToSchedulerBatch>(tokenizer_socket)? {
                self.broadcast_forward_batch(&batch)?;
                return Ok(batch.data);
            }
            if !blocking {
                return Ok(Vec::new());
            }
            on_idle();
            self.run_when_idle();
        }
    }

    fn receive_non_primary_msg<F>(
        &self,
        blocking: bool,
        on_idle: &mut F,
    ) -> Result<Vec<TokenizerToSchedulerMsg>>
    where
        F: FnMut(),
    {
        loop {
            if let Some(msg) = try_recv_datagram::<SchedulerRankMsg>(&self.rank_socket)? {
                let data = match msg {
                    SchedulerRankMsg::Forward(item) => vec![item],
                    SchedulerRankMsg::ForwardBatch(batch) => batch.data,
                    SchedulerRankMsg::Control(SchedulerRankControlMsg::Shutdown) => {
                        vec![TokenizerToSchedulerMsg::Exit]
                    }
                    SchedulerRankMsg::Control(_) => Vec::new(),
                };
                return Ok(data);
            }
            if !blocking {
                return Ok(Vec::new());
            }
            on_idle();
            self.run_when_idle();
        }
    }

    fn broadcast_rank_msg(&self, msg: &SchedulerRankMsg) -> Result<()> {
        ensure!(
            self.config.tp.is_primary(),
            "only rank0 can broadcast scheduler rank messages"
        );
        if self.config.tp.world_size <= 1 {
            return Ok(());
        }
        for rank in 1..self.config.tp.world_size {
            let path = self.config.rank_path(rank);
            send_datagram(&self.rank_socket, &path, msg, false)?;
        }
        Ok(())
    }
}

impl Drop for SchedulerIo {
    fn drop(&mut self) {
        for path in &self.owned_paths {
            let _ = fs::remove_file(path);
        }
    }
}

fn send_datagram<T: Serialize>(
    socket: &UnixDatagram,
    path: &Path,
    msg: &T,
    ignore_not_found: bool,
) -> Result<()> {
    let payload = serde_json::to_vec(msg).context("failed serializing datagram payload")?;
    match socket.send_to(&payload, path) {
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
        Err(err)
            if matches!(
                err.kind(),
                ErrorKind::NotFound | ErrorKind::ConnectionRefused | ErrorKind::AddrNotAvailable
            ) =>
        {
            Ok(())
        }
        Err(err) => {
            Err(err).with_context(|| format!("failed sending datagram to {}", path.display()))
        }
    }
}

fn try_recv_datagram<T: DeserializeOwned>(socket: &UnixDatagram) -> Result<Option<T>> {
    let mut buffer = vec![0u8; MAX_DATAGRAM_BYTES];
    match socket.recv(&mut buffer) {
        Ok(size) => {
            if size == 0 {
                return Ok(None);
            }
            let msg = serde_json::from_slice::<T>(&buffer[..size])
                .context("failed deserializing datagram payload")?;
            Ok(Some(msg))
        }
        Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
        Err(err) => Err(err).context("failed receiving datagram"),
    }
}

fn remove_stale_socket(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("failed removing stale socket {}", path.display()))?;
    }
    Ok(())
}
