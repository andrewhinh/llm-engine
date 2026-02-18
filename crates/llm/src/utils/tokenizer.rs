use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub hidden_size: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub architectures: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub eos_token: Option<TokenizerToken>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TokenizerToken {
    Text(String),
    Meta { content: String },
}

impl TokenizerToken {
    fn content(&self) -> &str {
        match self {
            Self::Text(text) => text.as_str(),
            Self::Meta { content } => content.as_str(),
        }
    }
}

pub fn load_model_config(model_dir: &Path) -> Result<ModelConfig> {
    let path = model_dir.join("config.json");
    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed reading model config at {}", path.display()))?;
    let config: ModelConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed parsing model config at {}", path.display()))?;
    Ok(config)
}

pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    let path_str = path
        .to_str()
        .with_context(|| format!("tokenizer path is not valid UTF-8: {}", path.display()))?;
    let tokenizer = Tokenizer::from_file(path_str)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed loading tokenizer at {}", path.display()))?;
    Ok(tokenizer)
}

pub fn load_tokenizer_config(model_dir: &Path) -> Result<Option<TokenizerConfig>> {
    let path = model_dir.join("tokenizer_config.json");
    if !path.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed reading tokenizer config at {}", path.display()))?;
    let config: TokenizerConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed parsing tokenizer config at {}", path.display()))?;
    Ok(Some(config))
}

pub fn resolve_eos_id(
    tokenizer: &Tokenizer,
    tokenizer_config: Option<&TokenizerConfig>,
    fallback_eos: i64,
) -> Result<i64> {
    if let Some(config) = tokenizer_config
        && let Some(token) = &config.eos_token
        && let Some(token_id) = tokenizer.token_to_id(token.content())
    {
        return Ok(i64::from(token_id));
    }

    ensure!(
        fallback_eos >= 0,
        "unable to resolve eos token from tokenizer config and fallback eos is invalid"
    );
    Ok(fallback_eos)
}

pub fn encode_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(anyhow::Error::msg)
        .context("failed encoding prompt")?;
    Ok(encoding.get_ids().to_vec())
}

pub fn decode_tokens(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    let decoded = tokenizer
        .decode(token_ids, false)
        .map_err(anyhow::Error::msg)
        .context("failed decoding token ids")?;
    Ok(decoded)
}
