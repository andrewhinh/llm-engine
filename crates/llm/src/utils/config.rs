use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

const MIN_TEMPERATURE: f32 = 1e-10;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum PrefixCacheBackend {
    #[default]
    Hash,
    Radix,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    pub model: String,
    #[serde(default = "default_max_num_batched_tokens")]
    pub max_num_batched_tokens: usize,
    #[serde(default = "default_max_num_seqs")]
    pub max_num_seqs: usize,
    #[serde(default = "default_max_model_len")]
    pub max_model_len: usize,
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f32,
    #[serde(default = "default_tensor_parallel_size")]
    pub tensor_parallel_size: usize,
    #[serde(default)]
    pub enforce_eager: bool,
    #[serde(default = "default_eos")]
    pub eos: i64,
    #[serde(default = "default_kvcache_block_size")]
    pub kvcache_block_size: usize,
    #[serde(default = "default_num_kvcache_blocks")]
    pub num_kvcache_blocks: isize,
    #[serde(default)]
    pub prefix_cache_enabled: bool,
    #[serde(default)]
    pub prefix_cache_max_cached_blocks: usize,
    #[serde(default)]
    pub prefix_cache_backend: PrefixCacheBackend,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_num_batched_tokens: default_max_num_batched_tokens(),
            max_num_seqs: default_max_num_seqs(),
            max_model_len: default_max_model_len(),
            gpu_memory_utilization: default_gpu_memory_utilization(),
            tensor_parallel_size: default_tensor_parallel_size(),
            enforce_eager: false,
            eos: default_eos(),
            kvcache_block_size: default_kvcache_block_size(),
            num_kvcache_blocks: default_num_kvcache_blocks(),
            prefix_cache_enabled: false,
            prefix_cache_max_cached_blocks: 0,
            prefix_cache_backend: PrefixCacheBackend::Hash,
        }
    }
}

impl EngineConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.model.trim().is_empty(), "model must not be empty");
        ensure!(
            self.max_num_batched_tokens > 0,
            "max_num_batched_tokens must be positive"
        );
        ensure!(self.max_num_seqs > 0, "max_num_seqs must be positive");
        ensure!(self.max_model_len > 0, "max_model_len must be positive");
        ensure!(
            self.max_num_batched_tokens >= self.max_model_len,
            "max_num_batched_tokens must be >= max_model_len"
        );
        ensure!(
            self.gpu_memory_utilization > 0.0 && self.gpu_memory_utilization <= 1.0,
            "gpu_memory_utilization must be in (0, 1]"
        );
        ensure!(
            (1..=8).contains(&self.tensor_parallel_size),
            "tensor_parallel_size must be in [1, 8]"
        );
        ensure!(
            self.kvcache_block_size.is_multiple_of(256),
            "kvcache_block_size must be divisible by 256"
        );
        ensure!(
            self.num_kvcache_blocks == -1 || self.num_kvcache_blocks > 0,
            "num_kvcache_blocks must be -1 or positive"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SamplingParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default = "default_top_k")]
    pub top_k: isize,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            ignore_eos: false,
            top_k: default_top_k(),
            top_p: default_top_p(),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature > MIN_TEMPERATURE,
            "temperature must be > 1e-10"
        );
        ensure!(self.max_tokens > 0, "max_tokens must be positive");
        ensure!(
            self.top_k == -1 || self.top_k >= 1,
            "top_k must be -1 or >= 1"
        );
        ensure!(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "top_p must be in (0, 1]"
        );
        ensure!(
            self.frequency_penalty.is_finite(),
            "frequency_penalty must be finite"
        );
        ensure!(
            self.presence_penalty.is_finite(),
            "presence_penalty must be finite"
        );
        Ok(())
    }
}

const fn default_max_num_batched_tokens() -> usize {
    16_384
}

const fn default_max_num_seqs() -> usize {
    512
}

const fn default_max_model_len() -> usize {
    4_096
}

const fn default_gpu_memory_utilization() -> f32 {
    0.9
}

const fn default_tensor_parallel_size() -> usize {
    1
}

const fn default_eos() -> i64 {
    -1
}

const fn default_kvcache_block_size() -> usize {
    256
}

const fn default_num_kvcache_blocks() -> isize {
    -1
}

const fn default_temperature() -> f32 {
    1.0
}

const fn default_max_tokens() -> usize {
    64
}

const fn default_top_k() -> isize {
    -1
}

const fn default_top_p() -> f32 {
    1.0
}
