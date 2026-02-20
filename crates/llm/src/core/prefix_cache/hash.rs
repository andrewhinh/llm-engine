use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use super::{PrefixCacheConfig, PrefixCacheUpdate, PrefixMatch};

#[derive(Clone, Debug)]
struct PrefixEntry {
    parent: Option<u64>,
    block_id: usize,
    children: usize,
    access_id: u64,
}

#[derive(Debug)]
pub struct PrefixCache {
    block_size: usize,
    config: PrefixCacheConfig,
    entries: HashMap<u64, PrefixEntry>,
    leaf_set: HashSet<u64>,
    leaf_lru: VecDeque<(u64, u64)>,
    access_counter: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        Self {
            block_size,
            config,
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.max_cached_blocks > 0
    }

    pub fn cached_blocks(&self) -> usize {
        self.entries.len()
    }

    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        if full_blocks == 0 {
            return PrefixMatch {
                matched_blocks: 0,
                last_hash: None,
            };
        }

        let mut matched = 0usize;
        let mut parent_hash = None;
        let mut last_hash = None;

        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            let hash = Self::hash_block(parent_hash.unwrap_or(0), block_tokens);
            if self.entries.contains_key(&hash) {
                matched += 1;
                parent_hash = Some(hash);
                last_hash = Some(hash);
                self.touch(hash);
            } else {
                break;
            }
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
        }
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<usize> {
        let mut blocks = Vec::new();
        let mut current = Some(last_hash);
        while let Some(hash) = current {
            let Some(entry) = self.entries.get(&hash) else {
                break;
            };
            blocks.push(entry.block_id);
            current = entry.parent;
        }
        blocks.reverse();
        blocks
    }

    pub fn insert_prefix(&mut self, tokens: &[u32], blocks: &[usize]) -> PrefixCacheUpdate {
        if !self.enabled() {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
            };
        }

        let full_blocks = tokens.len() / self.block_size;
        let max_blocks = usize::min(full_blocks, blocks.len());
        if max_blocks == 0 {
            return PrefixCacheUpdate {
                inserted: Vec::new(),
                evicted: Vec::new(),
            };
        }

        let mut inserted = Vec::new();
        let mut parent_hash = None;

        for (block_id, block_tokens) in blocks
            .iter()
            .zip(tokens.chunks(self.block_size))
            .take(max_blocks)
        {
            let hash = Self::hash_block(parent_hash.unwrap_or(0), block_tokens);
            if self.entries.contains_key(&hash) {
                let access_id = self.next_access_id();
                if let Some(entry) = self.entries.get_mut(&hash) {
                    entry.access_id = access_id;
                }
                self.touch_leaf(hash);
            } else {
                if let Some(parent) = parent_hash
                    && let Some(parent_entry) = self.entries.get_mut(&parent)
                {
                    if parent_entry.children == 0 {
                        self.leaf_set.remove(&parent);
                    }
                    parent_entry.children += 1;
                }

                let access_id = self.next_access_id();
                self.entries.insert(
                    hash,
                    PrefixEntry {
                        parent: parent_hash,
                        block_id: *block_id,
                        children: 0,
                        access_id,
                    },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access_id));
                inserted.push(*block_id);
            }
            parent_hash = Some(hash);
        }

        let evicted = self.evict_if_needed();
        PrefixCacheUpdate { inserted, evicted }
    }

    pub fn clear(&mut self) -> Vec<usize> {
        let blocks: Vec<usize> = self.entries.values().map(|entry| entry.block_id).collect();
        self.entries.clear();
        self.leaf_set.clear();
        self.leaf_lru.clear();
        blocks
    }

    fn touch(&mut self, hash: u64) {
        if self.entries.contains_key(&hash) {
            let access_id = self.next_access_id();
            if let Some(entry) = self.entries.get_mut(&hash) {
                entry.access_id = access_id;
            }
            self.touch_leaf(hash);
        }
    }

    fn touch_leaf(&mut self, hash: u64) {
        if self.leaf_set.contains(&hash)
            && let Some(entry) = self.entries.get(&hash)
        {
            self.leaf_lru.push_back((hash, entry.access_id));
        }
    }

    fn evict_if_needed(&mut self) -> Vec<usize> {
        let mut evicted = Vec::new();
        while self.entries.len() > self.config.max_cached_blocks {
            let Some((hash, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            if !self.leaf_set.contains(&hash) {
                continue;
            }
            let Some(entry) = self.entries.get(&hash) else {
                continue;
            };
            if entry.access_id != access_id || entry.children > 0 {
                continue;
            }

            let entry = self.entries.remove(&hash).expect("entry must exist");
            self.leaf_set.remove(&hash);
            evicted.push(entry.block_id);

            if let Some(parent_hash) = entry.parent
                && let Some(parent_entry) = self.entries.get_mut(&parent_hash)
            {
                if parent_entry.children > 0 {
                    parent_entry.children -= 1;
                }
                if parent_entry.children == 0 {
                    self.leaf_set.insert(parent_hash);
                    self.leaf_lru
                        .push_back((parent_hash, parent_entry.access_id));
                }
            }
        }
        evicted
    }

    fn next_access_id(&mut self) -> u64 {
        self.access_counter = self.access_counter.wrapping_add(1);
        self.access_counter
    }

    fn hash_block(parent_hash: u64, tokens: &[u32]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        parent_hash.hash(&mut hasher);
        tokens.hash(&mut hasher);
        hasher.finish()
    }
}
