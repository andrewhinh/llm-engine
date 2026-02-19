use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, ensure};

use crate::server::api::{ServerConfig, run_server};

const ROLE_ENV: &str = "LLM_PROCESS_ROLE";
const READY_FILE_ENV: &str = "LLM_READY_FILE";
const MODEL_ENV: &str = "LLM_MODEL";
const HOST_ENV: &str = "LLM_HOST";
const PORT_ENV: &str = "LLM_PORT";
const TP_SIZE_ENV: &str = "LLM_TP_SIZE";
const TOKENIZER_WORKERS_ENV: &str = "LLM_TOKENIZER_WORKERS";
const STARTUP_TIMEOUT_ENV: &str = "LLM_STARTUP_TIMEOUT_SECS";
const SHUTDOWN_TIMEOUT_ENV: &str = "LLM_SHUTDOWN_TIMEOUT_SECS";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessRole {
    Launcher,
    Frontend,
    Tokenizer { index: usize },
    Detokenizer,
    Scheduler { rank: usize },
}

impl ProcessRole {
    pub fn parse(raw: &str) -> Result<Self> {
        if raw == "launcher" {
            return Ok(Self::Launcher);
        }
        if raw == "frontend" {
            return Ok(Self::Frontend);
        }
        if raw == "detokenizer" {
            return Ok(Self::Detokenizer);
        }
        if let Some(raw_index) = raw.strip_prefix("tokenizer:") {
            let index = raw_index
                .parse::<usize>()
                .with_context(|| format!("invalid tokenizer index in role: {raw}"))?;
            return Ok(Self::Tokenizer { index });
        }
        if let Some(raw_rank) = raw.strip_prefix("scheduler:") {
            let rank = raw_rank
                .parse::<usize>()
                .with_context(|| format!("invalid scheduler rank in role: {raw}"))?;
            return Ok(Self::Scheduler { rank });
        }
        Err(anyhow!("unsupported process role: {raw}"))
    }

    pub fn as_env_value(&self) -> String {
        match self {
            Self::Launcher => "launcher".to_string(),
            Self::Frontend => "frontend".to_string(),
            Self::Tokenizer { index } => format!("tokenizer:{index}"),
            Self::Detokenizer => "detokenizer".to_string(),
            Self::Scheduler { rank } => format!("scheduler:{rank}"),
        }
    }

    pub fn ready_file_name(&self) -> String {
        match self {
            Self::Launcher => "launcher.ready".to_string(),
            Self::Frontend => "frontend.ready".to_string(),
            Self::Tokenizer { index } => format!("tokenizer-{index}.ready"),
            Self::Detokenizer => "detokenizer.ready".to_string(),
            Self::Scheduler { rank } => format!("scheduler-{rank}.ready"),
        }
    }

    pub fn is_worker_stub(&self) -> bool {
        matches!(
            self,
            Self::Tokenizer { .. } | Self::Detokenizer | Self::Scheduler { .. }
        )
    }
}

#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub server: ServerConfig,
    pub tp_size: usize,
    pub tokenizer_workers: usize,
    pub startup_timeout: Duration,
    pub shutdown_timeout: Duration,
}

impl LaunchConfig {
    pub fn from_env() -> Result<Self> {
        let server = ServerConfig::from_env()?;
        let tp_size = parse_env_or_default(TP_SIZE_ENV, 1usize)?;
        let tokenizer_workers = parse_env_or_default(TOKENIZER_WORKERS_ENV, 0usize)?;
        let startup_timeout_secs = parse_env_or_default(STARTUP_TIMEOUT_ENV, 20u64)?;
        let shutdown_timeout_secs = parse_env_or_default(SHUTDOWN_TIMEOUT_ENV, 5u64)?;
        ensure!(tp_size > 0, "{TP_SIZE_ENV} must be positive");
        Ok(Self {
            server,
            tp_size,
            tokenizer_workers,
            startup_timeout: Duration::from_secs(startup_timeout_secs),
            shutdown_timeout: Duration::from_secs(shutdown_timeout_secs),
        })
    }

    pub fn planned_roles(&self) -> Vec<ProcessRole> {
        let mut roles = Vec::with_capacity(2 + self.tokenizer_workers + self.tp_size);
        roles.push(ProcessRole::Frontend);
        roles.push(ProcessRole::Detokenizer);
        for index in 0..self.tokenizer_workers {
            roles.push(ProcessRole::Tokenizer { index });
        }
        for rank in 0..self.tp_size {
            roles.push(ProcessRole::Scheduler { rank });
        }
        roles
    }
}

#[derive(Debug)]
struct ChildProcess {
    role: ProcessRole,
    ready_file: PathBuf,
    child: Child,
}

pub fn process_role_from_env() -> Result<Option<ProcessRole>> {
    match env::var(ROLE_ENV) {
        Ok(value) => Ok(Some(ProcessRole::parse(&value)?)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(anyhow!("failed reading {ROLE_ENV}: {err}")),
    }
}

pub async fn run_launcher(config: LaunchConfig) -> Result<()> {
    let ack_dir = create_ack_dir()?;
    let mut children = spawn_children(&config, &ack_dir)?;
    let ready_files = children
        .iter()
        .map(|child| child.ready_file.clone())
        .collect::<Vec<_>>();
    let wait_result = wait_for_ready_files(&ready_files, config.startup_timeout);
    if let Err(err) = wait_result {
        shutdown_children(&mut children, config.shutdown_timeout)?;
        return Err(err);
    }
    let run_result = wait_for_shutdown_or_child_exit(&mut children).await;
    let shutdown_result = shutdown_children(&mut children, config.shutdown_timeout);
    let _ = fs::remove_dir_all(&ack_dir);
    run_result?;
    shutdown_result?;
    Ok(())
}

pub async fn run_frontend_role(server_config: ServerConfig) -> Result<()> {
    signal_ready(&ProcessRole::Frontend)?;
    run_server(server_config).await
}

pub async fn run_worker_role(role: ProcessRole) -> Result<()> {
    ensure!(
        role.is_worker_stub(),
        "worker role runner received unsupported role: {}",
        role.as_env_value()
    );
    signal_ready(&role)?;
    wait_for_shutdown_signal().await
}

fn parse_env_or_default<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Ok(raw) => raw
            .parse::<T>()
            .map_err(|err| anyhow!("invalid {name} value '{raw}': {err}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(anyhow!("failed reading {name}: {err}")),
    }
}

fn create_ack_dir() -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = env::temp_dir().join(format!(
        "llm-engine-launch-{}-{}",
        std::process::id(),
        nonce
    ));
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed creating ack dir {}", dir.display()))?;
    Ok(dir)
}

fn spawn_children(config: &LaunchConfig, ack_dir: &Path) -> Result<Vec<ChildProcess>> {
    let exe = env::current_exe().context("failed to resolve current executable path")?;
    let mut children = Vec::new();
    for role in config.planned_roles() {
        let ready_file = ack_dir.join(role.ready_file_name());
        let child = Command::new(&exe)
            .env(ROLE_ENV, role.as_env_value())
            .env(READY_FILE_ENV, ready_file.as_os_str())
            .env(MODEL_ENV, config.server.model.as_str())
            .env(HOST_ENV, config.server.host.as_str())
            .env(PORT_ENV, config.server.port.to_string())
            .env(TP_SIZE_ENV, config.tp_size.to_string())
            .env(TOKENIZER_WORKERS_ENV, config.tokenizer_workers.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed spawning child role {}", role.as_env_value()))?;
        children.push(ChildProcess {
            role,
            ready_file,
            child,
        });
    }
    Ok(children)
}

fn wait_for_ready_files(ready_files: &[PathBuf], timeout: Duration) -> Result<()> {
    let start = Instant::now();
    loop {
        if ready_files_present(ready_files) {
            return Ok(());
        }
        if start.elapsed() > timeout {
            let missing = ready_files
                .iter()
                .filter(|path| !path.exists())
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>();
            return Err(anyhow!(
                "timed out waiting for child startup ack: {missing:?}"
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn ready_files_present(ready_files: &[PathBuf]) -> bool {
    ready_files.iter().all(|path| path.exists())
}

async fn wait_for_shutdown_or_child_exit(children: &mut [ChildProcess]) -> Result<()> {
    let signal_fut = wait_for_shutdown_signal();
    tokio::pin!(signal_fut);
    loop {
        for child in children.iter_mut() {
            if let Some(status) = child.child.try_wait()? {
                return Err(anyhow!(
                    "child {} exited early with status {status}",
                    child.role.as_env_value()
                ));
            }
        }
        tokio::select! {
            signal = &mut signal_fut => {
                signal?;
                return Ok(());
            }
            _ = tokio::time::sleep(Duration::from_millis(200)) => {}
        }
    }
}

fn shutdown_children(children: &mut [ChildProcess], timeout: Duration) -> Result<()> {
    for child in children.iter_mut() {
        if child.child.try_wait()?.is_some() {
            continue;
        }
        send_terminate_signal(&mut child.child);
    }

    let start = Instant::now();
    while start.elapsed() < timeout {
        let mut all_done = true;
        for child in children.iter_mut() {
            if child.child.try_wait()?.is_none() {
                all_done = false;
            }
        }
        if all_done {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    for child in children.iter_mut() {
        if child.child.try_wait()?.is_none() {
            let _ = child.child.kill();
            let _ = child.child.wait();
        }
    }
    Ok(())
}

fn send_terminate_signal(child: &mut Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .status();
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
}

fn signal_ready(role: &ProcessRole) -> Result<()> {
    let ready_file = match env::var(READY_FILE_ENV) {
        Ok(path) => PathBuf::from(path),
        Err(env::VarError::NotPresent) => return Ok(()),
        Err(err) => return Err(anyhow!("failed reading {READY_FILE_ENV}: {err}")),
    };
    if let Some(parent) = ready_file.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed creating ready dir {}", parent.display()))?;
    }
    fs::write(&ready_file, role.as_env_value())
        .with_context(|| format!("failed writing ready file {}", ready_file.display()))?;
    Ok(())
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
