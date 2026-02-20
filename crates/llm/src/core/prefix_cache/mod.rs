use crate::utils::PrefixCacheBackend;

mod hash;
mod radix;

use hash::PrefixCache as HashPrefixCache;
use radix::PrefixCache as RadixPrefixCache;

#[derive(Clone, Debug, Default)]
pub struct PrefixCacheConfig {
    pub enabled: bool,
    pub max_cached_blocks: usize,
    pub backend: PrefixCacheBackend,
}

#[derive(Clone, Debug)]
pub struct PrefixMatch {
    pub matched_blocks: usize,
    pub last_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheUpdate {
    pub inserted: Vec<usize>,
    pub evicted: Vec<usize>,
}

#[derive(Debug)]
pub enum PrefixCache {
    Hash(HashPrefixCache),
    Radix(RadixPrefixCache),
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        match config.backend {
            PrefixCacheBackend::Hash => Self::Hash(HashPrefixCache::new(block_size, config)),
            PrefixCacheBackend::Radix => Self::Radix(RadixPrefixCache::new(block_size, config)),
        }
    }

    pub fn enabled(&self) -> bool {
        match self {
            Self::Hash(cache) => cache.enabled(),
            Self::Radix(cache) => cache.enabled(),
        }
    }

    pub fn cached_blocks(&self) -> usize {
        match self {
            Self::Hash(cache) => cache.cached_blocks(),
            Self::Radix(cache) => cache.cached_blocks(),
        }
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        match self {
            Self::Hash(cache) => cache.match_prefix(tokens),
            Self::Radix(cache) => cache.match_prefix(tokens),
        }
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<usize> {
        match self {
            Self::Hash(cache) => cache.blocks_for_match(last_hash),
            Self::Radix(cache) => cache.blocks_for_match(last_hash),
        }
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) -> PrefixCacheUpdate {
        match self {
            Self::Hash(cache) => cache.insert_prefix(tokens, blocks),
            Self::Radix(cache) => cache.insert_prefix(tokens, blocks),
        }
    }

    pub fn clear(&mut self) -> Vec<usize> {
        match self {
            Self::Hash(cache) => cache.clear(),
            Self::Radix(cache) => cache.clear(),
        }
    }
}
