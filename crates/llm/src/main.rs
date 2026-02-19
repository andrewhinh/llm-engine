use anyhow::Result;
use llm::server::{ServerConfig, run_server};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let config = ServerConfig::from_env()?;
    run_server(config).await?;
    Ok(())
}
