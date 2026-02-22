use std::env;
use std::io::{self, Write};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::task;

use crate::server::api::{ServerConfig, build_backend};
use crate::utils::config::SamplingParams;

const SHELL_MAX_TOKENS_ENV: &str = "LLM_SHELL_MAX_TOKENS";
const SHELL_TOP_K_ENV: &str = "LLM_SHELL_TOP_K";
const SHELL_TOP_P_ENV: &str = "LLM_SHELL_TOP_P";
const SHELL_TEMPERATURE_ENV: &str = "LLM_SHELL_TEMPERATURE";

#[derive(Debug, Clone)]
pub struct ShellConfig {
    pub max_tokens: usize,
    pub top_k: isize,
    pub top_p: f32,
    pub temperature: f32,
}

impl ShellConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            max_tokens: parse_env_or_default(SHELL_MAX_TOKENS_ENV, 2048usize)?,
            top_k: parse_env_or_default(SHELL_TOP_K_ENV, -1isize)?,
            top_p: parse_env_or_default(SHELL_TOP_P_ENV, 1.0f32)?,
            temperature: parse_env_or_default(SHELL_TEMPERATURE_ENV, 0.6f32)?,
        })
    }

    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            max_tokens: self.max_tokens,
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
            ..SamplingParams::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ShellCommand {
    Exit,
    Reset,
    Prompt(String),
}

fn parse_shell_command(raw: &str) -> Result<Option<ShellCommand>> {
    let input = raw.trim();
    if input.is_empty() {
        return Ok(None);
    }
    if input.starts_with('/') {
        return match input {
            "/exit" => Ok(Some(ShellCommand::Exit)),
            "/reset" => Ok(Some(ShellCommand::Reset)),
            _ => Err(anyhow!("unknown command: {input}")),
        };
    }
    Ok(Some(ShellCommand::Prompt(input.to_string())))
}

pub async fn run_shell(server_config: ServerConfig) -> Result<()> {
    let backend = build_backend(&server_config)?;
    let shell = ShellConfig::from_env()?;
    let mut history: Vec<(String, String)> = Vec::new();

    loop {
        print!("$ ");
        io::stdout().flush()?;
        let line = task::spawn_blocking(read_stdin_line)
            .await
            .map_err(|err| anyhow!("shell input task failed: {err}"))??;
        let Some(line) = line else {
            break;
        };

        let command = match parse_shell_command(&line) {
            Ok(Some(command)) => command,
            Ok(None) => continue,
            Err(err) => {
                eprintln!("{err}");
                continue;
            }
        };

        match command {
            ShellCommand::Exit => break,
            ShellCommand::Reset => {
                history.clear();
            }
            ShellCommand::Prompt(user_prompt) => {
                let prompt = render_prompt_with_history(&history, &user_prompt);
                let sampling = shell.sampling_params();
                let backend = Arc::clone(&backend);
                let assistant = task::spawn_blocking(move || {
                    let mut response = String::new();
                    backend.stream(prompt, sampling, |delta| {
                        if delta.is_empty() {
                            return Ok(());
                        }
                        print!("{delta}");
                        io::stdout().flush()?;
                        response.push_str(&delta);
                        Ok(())
                    })?;
                    println!();
                    Ok::<String, anyhow::Error>(response)
                })
                .await
                .map_err(|err| anyhow!("shell generation task failed: {err}"))??;
                history.push((user_prompt, assistant));
            }
        }
    }

    Ok(())
}

fn read_stdin_line() -> Result<Option<String>> {
    let mut line = String::new();
    let size = io::stdin().read_line(&mut line)?;
    if size == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}

fn render_prompt_with_history(history: &[(String, String)], user_prompt: &str) -> String {
    let mut lines = Vec::with_capacity(history.len() * 2 + 1);
    for (user, assistant) in history {
        lines.push(format!("user: {user}"));
        lines.push(format!("assistant: {assistant}"));
    }
    lines.push(format!("user: {user_prompt}"));
    lines.join("\n")
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
