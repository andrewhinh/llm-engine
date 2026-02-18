use anyhow::{Result, ensure};
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::utils::config::EngineConfig;
use crate::utils::kvcache_allocator::{KVCacheAllocator, KVCachePlan};
use crate::utils::tokenizer::{
    ModelConfig, decode_tokens, encode_prompt, load_model_config, load_tokenizer,
    load_tokenizer_config, resolve_eos_id,
};

#[derive(Debug)]
pub struct Engine {
    pub config: EngineConfig,
    pub model_config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub kv_cache_plan: KVCachePlan,
}

impl Engine {
    pub fn new(mut config: EngineConfig) -> Result<Self> {
        config.validate()?;

        let model_dir = PathBuf::from(&config.model);
        ensure!(
            model_dir.is_dir(),
            "model must be an existing directory: {}",
            model_dir.display()
        );

        let model_config = load_model_config(&model_dir)?;
        config.max_model_len = config
            .max_model_len
            .min(model_config.max_position_embeddings);
        ensure!(
            config.max_num_batched_tokens >= config.max_model_len,
            "max_num_batched_tokens must be >= clamped max_model_len"
        );
        let allocator = KVCacheAllocator::new(&config, &model_config)?;
        let kv_cache_plan = allocator.plan_auto()?;
        KVCacheAllocator::apply_plan(&mut config, &kv_cache_plan);
        ensure!(
            config.max_num_batched_tokens >= config.max_model_len,
            "max_num_batched_tokens must be >= max_model_len after kv cache planning"
        );

        let tokenizer = load_tokenizer(&model_dir)?;
        let tokenizer_config = load_tokenizer_config(&model_dir)?;
        config.eos = resolve_eos_id(&tokenizer, tokenizer_config.as_ref(), config.eos)?;

        Ok(Self {
            config,
            model_config,
            tokenizer,
            kv_cache_plan,
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        encode_prompt(&self.tokenizer, prompt)
    }

    pub fn decode_tokens(&self, token_ids: &[u32]) -> Result<String> {
        decode_tokens(&self.tokenizer, token_ids)
    }
}
