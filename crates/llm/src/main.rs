use anyhow::Result;
use llm::server::{
    LaunchConfig, ProcessRole, ServerConfig, process_role_from_env, run_frontend_role,
    run_launcher, run_server, run_shell, run_worker_role,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli_shell_mode = shell_mode_from_args();
    match process_role_from_env()? {
        Some(ProcessRole::Launcher) => {
            let mut config = LaunchConfig::from_env()?;
            if cli_shell_mode {
                config.shell_mode = true;
            }
            run_launcher(config).await?;
        }
        Some(ProcessRole::Frontend) => {
            let config = ServerConfig::from_env()?;
            run_frontend_role(config).await?;
        }
        Some(role) => {
            run_worker_role(role).await?;
        }
        None => {
            let config = ServerConfig::from_env()?;
            if cli_shell_mode {
                run_shell(config).await?;
            } else {
                run_server(config).await?;
            }
        }
    }
    Ok(())
}

fn shell_mode_from_args() -> bool {
    std::env::args().skip(1).any(|arg| arg == "--shell")
}
