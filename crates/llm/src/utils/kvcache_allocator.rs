use anyhow::{Result, ensure};
use sysinfo::System;

use crate::utils::config::EngineConfig;
use crate::utils::tokenizer::ModelConfig;

const DEFAULT_CACHE_DTYPE_BYTES: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KVCachePlan {
    pub num_kvcache_blocks: usize,
    pub per_block_bytes: usize,
    pub kvcache_memory_bytes: usize,
    pub max_kvcache_tokens: usize,
    pub max_num_batched_tokens_cap: usize,
    pub max_num_seqs_cap: usize,
}

#[derive(Debug, Clone)]
pub struct KVCacheAllocator {
    num_hidden_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    tensor_parallel_size: usize,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    configured_num_blocks: isize,
}

impl KVCacheAllocator {
    pub fn new(config: &EngineConfig, model_config: &ModelConfig) -> Result<Self> {
        ensure!(
            model_config.num_attention_heads > 0,
            "model num_attention_heads must be positive"
        );
        ensure!(
            model_config.num_hidden_layers > 0,
            "model num_hidden_layers must be positive"
        );
        let num_kv_heads = model_config
            .num_key_value_heads
            .unwrap_or(model_config.num_attention_heads);
        ensure!(
            num_kv_heads > 0,
            "model num_key_value_heads must be positive"
        );
        let head_dim = match model_config.head_dim {
            Some(dim) => dim,
            None => {
                ensure!(
                    model_config
                        .hidden_size
                        .is_multiple_of(model_config.num_attention_heads),
                    "hidden_size must be divisible by num_attention_heads when head_dim is absent"
                );
                model_config.hidden_size / model_config.num_attention_heads
            }
        };
        ensure!(head_dim > 0, "model head_dim must be positive");

        Ok(Self {
            num_hidden_layers: model_config.num_hidden_layers,
            num_kv_heads,
            head_dim,
            block_size: config.kvcache_block_size,
            tensor_parallel_size: config.tensor_parallel_size,
            max_model_len: config.max_model_len,
            gpu_memory_utilization: config.gpu_memory_utilization,
            configured_num_blocks: config.num_kvcache_blocks,
        })
    }

    pub fn query_available_memory_bytes() -> Result<u64> {
        let mut system = System::new();
        system.refresh_memory();
        let available = system.available_memory();
        ensure!(available > 0, "available memory must be positive");
        Ok(available)
    }

    pub fn kv_heads_per_shard(&self) -> usize {
        if self.num_kv_heads >= self.tensor_parallel_size {
            self.num_kv_heads / self.tensor_parallel_size
        } else {
            1
        }
    }

    pub fn per_block_bytes(&self) -> Result<usize> {
        let mut bytes = self.block_size;
        bytes = checked_mul(bytes, self.kv_heads_per_shard())?;
        bytes = checked_mul(bytes, self.head_dim)?;
        bytes = checked_mul(bytes, DEFAULT_CACHE_DTYPE_BYTES)?;
        bytes = checked_mul(bytes, 2)?;
        bytes = checked_mul(bytes, self.num_hidden_layers)?;
        Ok(bytes)
    }

    pub fn plan_with_available_memory(&self, available_bytes: u64) -> Result<KVCachePlan> {
        ensure!(available_bytes > 0, "available memory must be positive");
        let per_block_bytes = self.per_block_bytes()?;
        ensure!(per_block_bytes > 0, "per-block bytes must be positive");

        let usable_bytes = (available_bytes as f64 * self.gpu_memory_utilization as f64) as u64;
        ensure!(
            usable_bytes > 0,
            "usable memory after gpu_memory_utilization must be positive"
        );

        let num_blocks = if self.configured_num_blocks > 0 {
            let configured = self.configured_num_blocks as usize;
            let required = (configured as u128) * (per_block_bytes as u128);
            ensure!(
                required <= usable_bytes as u128,
                "configured num_kvcache_blocks={} exceeds memory budget",
                configured
            );
            configured
        } else {
            (usable_bytes as usize) / per_block_bytes
        };

        ensure!(
            num_blocks > 0,
            "computed num_kvcache_blocks must be positive"
        );
        let kvcache_memory_bytes = checked_mul(num_blocks, per_block_bytes)?;
        let max_kvcache_tokens = checked_mul(num_blocks, self.block_size)?;
        ensure!(
            max_kvcache_tokens >= self.max_model_len,
            "kv cache capacity {} tokens is smaller than max_model_len {}",
            max_kvcache_tokens,
            self.max_model_len
        );

        let max_num_batched_tokens_cap = max_kvcache_tokens;
        let max_num_seqs_cap = usize::max(1, max_kvcache_tokens / self.max_model_len);

        Ok(KVCachePlan {
            num_kvcache_blocks: num_blocks,
            per_block_bytes,
            kvcache_memory_bytes,
            max_kvcache_tokens,
            max_num_batched_tokens_cap,
            max_num_seqs_cap,
        })
    }

    pub fn plan_auto(&self) -> Result<KVCachePlan> {
        let available_bytes = Self::query_available_memory_bytes()?;
        self.plan_with_available_memory(available_bytes)
    }

    pub fn apply_plan(config: &mut EngineConfig, plan: &KVCachePlan) {
        config.num_kvcache_blocks = plan.num_kvcache_blocks as isize;
        config.max_num_batched_tokens = config
            .max_num_batched_tokens
            .min(plan.max_num_batched_tokens_cap);
        config.max_num_seqs = config.max_num_seqs.min(plan.max_num_seqs_cap);
    }
}

fn checked_mul(lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| anyhow::anyhow!("kv cache size overflow"))
}
