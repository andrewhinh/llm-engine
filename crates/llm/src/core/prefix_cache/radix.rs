use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use super::{PrefixCacheConfig, PrefixCacheUpdate, PrefixMatch};

#[derive(Clone, Debug)]
struct RadixNode {
    parent: Option<usize>,
    hash: u64,
    block_id: usize,
    block_tokens: Vec<u32>,
    children: Vec<usize>,
    access_id: u64,
    alive: bool,
}

#[derive(Debug)]
pub struct PrefixCache {
    block_size: usize,
    config: PrefixCacheConfig,
    nodes: Vec<RadixNode>,
    hash_to_node: HashMap<u64, usize>,
    leaf_set: HashSet<usize>,
    leaf_lru: VecDeque<(usize, u64)>,
    access_counter: u64,
    num_entries: usize,
}

impl PrefixCache {
    pub fn new(block_size: usize, config: PrefixCacheConfig) -> Self {
        let root = RadixNode {
            parent: None,
            hash: 0,
            block_id: usize::MAX,
            block_tokens: Vec::new(),
            children: Vec::new(),
            access_id: 0,
            alive: true,
        };
        Self {
            block_size,
            config,
            nodes: vec![root],
            hash_to_node: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access_counter: 0,
            num_entries: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled && self.config.max_cached_blocks > 0
    }

    pub fn cached_blocks(&self) -> usize {
        self.num_entries
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
        let mut current = 0usize;
        let mut parent_hash = 0u64;
        let mut last_hash = None;

        for block_tokens in tokens.chunks(self.block_size).take(full_blocks) {
            let Some(child) = self.find_child(current, block_tokens) else {
                break;
            };
            self.touch(child);
            let hash = Self::hash_block(parent_hash, block_tokens);
            last_hash = Some(hash);
            parent_hash = hash;
            current = child;
            matched += 1;
        }

        PrefixMatch {
            matched_blocks: matched,
            last_hash,
        }
    }

    pub fn blocks_for_match(&self, last_hash: u64) -> Vec<usize> {
        let Some(&node_idx) = self.hash_to_node.get(&last_hash) else {
            return Vec::new();
        };
        if !self.nodes[node_idx].alive {
            return Vec::new();
        }

        let mut blocks = Vec::new();
        let mut current = Some(node_idx);
        while let Some(node_idx) = current {
            if node_idx == 0 {
                break;
            }
            let node = &self.nodes[node_idx];
            if !node.alive {
                break;
            }
            blocks.push(node.block_id);
            current = node.parent;
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
        let mut current = 0usize;
        let mut parent_hash = 0u64;

        for (block_id, block_tokens) in blocks
            .iter()
            .zip(tokens.chunks(self.block_size))
            .take(max_blocks)
        {
            if let Some(child) = self.find_child(current, block_tokens) {
                self.touch(child);
                current = child;
                parent_hash = Self::hash_block(parent_hash, block_tokens);
                continue;
            }

            if current != 0 && self.is_leaf(current) {
                self.leaf_set.remove(&current);
            }

            let access_id = self.next_access_id();
            let hash = Self::hash_block(parent_hash, block_tokens);
            let node_idx = self.nodes.len();
            self.nodes.push(RadixNode {
                parent: Some(current),
                hash,
                block_id: *block_id,
                block_tokens: block_tokens.to_vec(),
                children: Vec::new(),
                access_id,
                alive: true,
            });
            self.nodes[current].children.push(node_idx);
            self.hash_to_node.insert(hash, node_idx);
            self.leaf_set.insert(node_idx);
            self.leaf_lru.push_back((node_idx, access_id));
            self.num_entries += 1;

            inserted.push(*block_id);
            current = node_idx;
            parent_hash = hash;
        }

        let evicted = self.evict_if_needed();
        PrefixCacheUpdate { inserted, evicted }
    }

    pub fn clear(&mut self) -> Vec<usize> {
        let blocks = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                if idx == 0 || !node.alive {
                    None
                } else {
                    Some(node.block_id)
                }
            })
            .collect::<Vec<_>>();

        self.nodes.truncate(1);
        self.nodes[0].children.clear();
        self.hash_to_node.clear();
        self.leaf_set.clear();
        self.leaf_lru.clear();
        self.access_counter = 0;
        self.num_entries = 0;
        blocks
    }

    fn find_child(&self, parent_idx: usize, block_tokens: &[u32]) -> Option<usize> {
        self.nodes[parent_idx]
            .children
            .iter()
            .copied()
            .find(|child_idx| {
                let child = &self.nodes[*child_idx];
                child.alive && child.block_tokens == block_tokens
            })
    }

    fn is_leaf(&self, node_idx: usize) -> bool {
        self.nodes[node_idx]
            .children
            .iter()
            .all(|child_idx| !self.nodes[*child_idx].alive)
    }

    fn touch(&mut self, node_idx: usize) {
        if !self.nodes[node_idx].alive {
            return;
        }
        let access_id = self.next_access_id();
        self.nodes[node_idx].access_id = access_id;
        if self.leaf_set.contains(&node_idx) {
            self.leaf_lru.push_back((node_idx, access_id));
        }
    }

    fn evict_if_needed(&mut self) -> Vec<usize> {
        let mut evicted = Vec::new();
        while self.num_entries > self.config.max_cached_blocks {
            let Some((node_idx, access_id)) = self.leaf_lru.pop_front() else {
                break;
            };
            if !self.leaf_set.contains(&node_idx) {
                continue;
            }
            if !self.nodes[node_idx].alive {
                continue;
            }
            if self.nodes[node_idx].access_id != access_id {
                continue;
            }
            if !self.is_leaf(node_idx) {
                continue;
            }

            let block_id = self.nodes[node_idx].block_id;
            let parent = self.nodes[node_idx].parent;
            let hash = self.nodes[node_idx].hash;
            self.nodes[node_idx].alive = false;
            self.leaf_set.remove(&node_idx);
            self.hash_to_node.remove(&hash);
            self.num_entries = self.num_entries.saturating_sub(1);
            evicted.push(block_id);

            if let Some(parent_idx) = parent
                && parent_idx != 0
                && self.nodes[parent_idx].alive
                && self.is_leaf(parent_idx)
            {
                self.leaf_set.insert(parent_idx);
                self.leaf_lru
                    .push_back((parent_idx, self.nodes[parent_idx].access_id));
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
