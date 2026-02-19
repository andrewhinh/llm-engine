use anyhow::Result;
use llm::server::{
    LaunchConfig, ProcessRole, ServerConfig, process_role_from_env, run_frontend_role,
    run_launcher, run_server, run_worker_role,
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    match process_role_from_env()? {
        Some(ProcessRole::Launcher) => {
            let config = LaunchConfig::from_env()?;
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
            run_server(config).await?;
        }
    }
    Ok(())
}
