use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow, ensure};

use crate::core::Sequence;
use crate::core::prefix_cache::{PrefixCache, PrefixCacheConfig, PrefixMatch};

#[derive(Debug, Clone)]
pub struct Block {
    pub id: usize,
    pub ref_count: usize,
    pub hash: Option<u64>,
    pub token_ids: Vec<u32>,
}

impl Block {
    fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids.clear();
    }

    fn update(&mut self, hash: u64, token_ids: &[u32]) {
        self.hash = Some(hash);
        self.token_ids.clear();
        self.token_ids.extend_from_slice(token_ids);
    }
}

#[derive(Debug)]
pub struct BlockManager {
    blocks: Vec<Block>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
    block_size: usize,
    prefix_cache: Option<PrefixCache>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize, prefix_cfg: PrefixCacheConfig) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_block_ids = VecDeque::with_capacity(num_blocks);
        for id in 0..num_blocks {
            blocks.push(Block {
                id,
                ref_count: 0,
                hash: None,
                token_ids: Vec::new(),
            });
            free_block_ids.push_back(id);
        }

        let prefix_cache = if prefix_cfg.enabled && prefix_cfg.max_cached_blocks > 0 {
            Some(PrefixCache::new(block_size, prefix_cfg))
        } else {
            None
        };

        Self {
            blocks,
            free_block_ids,
            used_block_ids: HashSet::new(),
            block_size,
            prefix_cache,
        }
    }

    pub fn can_allocate(&mut self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= self.required_blocks(seq)
    }

    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        ensure!(
            seq.block_table.is_empty(),
            "sequence block_table must be empty"
        );
        if self.prefix_cache.is_some() {
            self.allocate_with_prefix(seq)
        } else {
            self.allocate_fresh(seq)
        }
    }

    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            self.decrement_block_ref(block_id as usize);
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        if seq.len() % self.block_size == 1 {
            !self.free_block_ids.is_empty()
        } else {
            true
        }
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<()> {
        ensure!(!seq.block_table.is_empty(), "sequence has no block_table");
        let len_mod = seq.len() % self.block_size;
        let last_block_id =
            seq.block_table
                .last()
                .copied()
                .ok_or_else(|| anyhow!("sequence has empty block_table"))? as usize;

        if len_mod == 1 {
            ensure!(
                self.blocks[last_block_id].hash.is_some(),
                "last block must be complete before appending a new block"
            );
            let new_block_id = self
                .free_block_ids
                .front()
                .copied()
                .ok_or_else(|| anyhow!("no free blocks available"))?;
            self.allocate_block(new_block_id)?;
            seq.block_table.push(new_block_id as u32);
            return Ok(());
        }

        if len_mod == 0 {
            ensure!(
                self.blocks[last_block_id].hash.is_none(),
                "last block must be partial before finalizing hash"
            );
            let token_ids = seq.block(seq.num_blocks() - 1).to_vec();
            let prefix_hash = if seq.block_table.len() > 1 {
                let prev_block_id = seq.block_table[seq.block_table.len() - 2] as usize;
                self.blocks[prev_block_id].hash
            } else {
                None
            };
            let hash = Self::compute_hash(&token_ids, prefix_hash);
            self.blocks[last_block_id].update(hash, &token_ids);
            return Ok(());
        }

        ensure!(
            self.blocks[last_block_id].hash.is_none(),
            "incomplete block must not have hash assigned yet"
        );
        Ok(())
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache.is_some()
    }

    pub fn prefix_cache_blocks(&self) -> usize {
        self.prefix_cache
            .as_ref()
            .map_or(0, PrefixCache::cached_blocks)
    }

    pub fn clear_prefix_cache(&mut self) -> usize {
        self.prefix_cache
            .as_mut()
            .map_or(0, |cache| cache.clear().len())
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    pub fn num_total_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    fn required_blocks(&mut self, seq: &Sequence) -> usize {
        let Some(cache) = self.prefix_cache.as_mut() else {
            return seq.num_blocks();
        };
        let prefix_match = cache.match_prefix(&seq.token_ids);
        let matched_blocks =
            Self::adjusted_matched_blocks(seq.token_ids.len(), self.block_size, &prefix_match);
        seq.num_blocks().saturating_sub(matched_blocks)
    }

    fn allocate_with_prefix(&mut self, seq: &mut Sequence) -> Result<()> {
        let (matched_blocks, matched_block_ids) = if let Some(cache) = self.prefix_cache.as_mut() {
            let prefix_match = cache.match_prefix(&seq.token_ids);
            let mut matched =
                Self::adjusted_matched_blocks(seq.token_ids.len(), self.block_size, &prefix_match);

            let mut ids = if let Some(last_hash) = prefix_match.last_hash {
                let mut ids = cache.blocks_for_match(last_hash);
                ids.truncate(matched);
                ids
            } else {
                Vec::new()
            };

            matched = self.validate_cached_prefix(seq, &ids);
            ids.truncate(matched);
            (matched, ids)
        } else {
            (0, Vec::new())
        };

        for block_id in matched_block_ids {
            self.increment_block_ref(block_id);
            seq.block_table.push(block_id as u32);
        }
        seq.num_cached_tokens = matched_blocks * self.block_size;

        while seq.block_table.len() < seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .front()
                .copied()
                .ok_or_else(|| anyhow!("no free blocks available"))?;
            self.allocate_block(block_id)?;
            seq.block_table.push(block_id as u32);
        }

        self.rehash_sequence_blocks(seq)?;

        if let Some(cache) = self.prefix_cache.as_mut() {
            let full_blocks = seq.token_ids.len() / self.block_size;
            if full_blocks > 0 {
                let blocks: Vec<usize> = seq
                    .block_table
                    .iter()
                    .take(full_blocks)
                    .map(|id| *id as usize)
                    .collect();
                let _ = cache.insert_prefix(&seq.token_ids, &blocks);
            }
        }

        Ok(())
    }

    fn allocate_fresh(&mut self, seq: &mut Sequence) -> Result<()> {
        seq.num_cached_tokens = 0;
        for _ in 0..seq.num_blocks() {
            let block_id = self
                .free_block_ids
                .front()
                .copied()
                .ok_or_else(|| anyhow!("no free blocks available"))?;
            self.allocate_block(block_id)?;
            seq.block_table.push(block_id as u32);
        }
        self.rehash_sequence_blocks(seq)
    }

    fn allocate_block(&mut self, block_id: usize) -> Result<()> {
        ensure!(
            self.blocks
                .get(block_id)
                .is_some_and(|block| block.ref_count == 0),
            "block {} must be free before allocation",
            block_id
        );
        let pos = self
            .free_block_ids
            .iter()
            .position(|id| *id == block_id)
            .ok_or_else(|| anyhow!("block {} is not in free list", block_id))?;
        let _ = self.free_block_ids.remove(pos);
        self.used_block_ids.insert(block_id);
        self.blocks[block_id].reset();
        Ok(())
    }

    fn deallocate_block(&mut self, block_id: usize) {
        if self.used_block_ids.remove(&block_id) {
            self.free_block_ids.push_back(block_id);
        }
    }

    fn increment_block_ref(&mut self, block_id: usize) {
        if self.blocks[block_id].ref_count == 0 {
            if let Some(pos) = self.free_block_ids.iter().position(|id| *id == block_id) {
                let _ = self.free_block_ids.remove(pos);
            }
            self.used_block_ids.insert(block_id);
        }
        self.blocks[block_id].ref_count += 1;
    }

    fn decrement_block_ref(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        if block.ref_count > 0 {
            block.ref_count -= 1;
        }
        if block.ref_count == 0 {
            self.deallocate_block(block_id);
        }
    }

    fn rehash_sequence_blocks(&mut self, seq: &Sequence) -> Result<()> {
        let full_blocks = seq.token_ids.len() / self.block_size;
        if full_blocks == 0 {
            return Ok(());
        }

        let mut parent_hash = None;
        for block_index in 0..full_blocks {
            let block_id = *seq
                .block_table
                .get(block_index)
                .ok_or_else(|| anyhow!("sequence missing block entry {}", block_index))?
                as usize;
            let token_ids = seq.block(block_index);
            let hash = Self::compute_hash(token_ids, parent_hash);
            self.blocks[block_id].update(hash, token_ids);
            parent_hash = Some(hash);
        }
        Ok(())
    }

    fn validate_cached_prefix(&self, seq: &Sequence, cached_block_ids: &[usize]) -> usize {
        let max_full_blocks = seq.token_ids.len() / self.block_size;
        let mut matched = 0usize;
        for (idx, block_id) in cached_block_ids.iter().copied().enumerate() {
            if idx >= max_full_blocks {
                break;
            }
            let Some(block) = self.blocks.get(block_id) else {
                break;
            };
            if block.token_ids != seq.block(idx) {
                break;
            }
            matched += 1;
        }
        matched
    }

    fn adjusted_matched_blocks(
        tokens_len: usize,
        block_size: usize,
        prefix_match: &PrefixMatch,
    ) -> usize {
        let full_blocks = tokens_len / block_size;
        if prefix_match.matched_blocks == full_blocks
            && tokens_len.is_multiple_of(block_size)
            && prefix_match.matched_blocks > 0
        {
            prefix_match.matched_blocks - 1
        } else {
            prefix_match.matched_blocks
        }
    }

    fn compute_hash(token_ids: &[u32], prefix_hash: Option<u64>) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        prefix_hash.unwrap_or(0).hash(&mut hasher);
        token_ids.hash(&mut hasher);
        hasher.finish()
    }
}
