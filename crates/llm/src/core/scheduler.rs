use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow, ensure};

use crate::core::block_manager::BlockManager;
use crate::core::prefix_cache_hash::PrefixCacheConfig;
use crate::core::sequence::{Sequence, SequenceStatus};
use crate::utils::config::EngineConfig;

#[derive(Debug)]
pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    eos_token_id: u32,
    pub block_manager: BlockManager,
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
    next_seq_id: usize,
}

impl Scheduler {
    pub fn new(config: &EngineConfig, prefix_cfg: PrefixCacheConfig) -> Result<Self> {
        let num_blocks = usize::try_from(config.num_kvcache_blocks)
            .map_err(|_| anyhow!("num_kvcache_blocks must be positive before scheduler init"))?;
        ensure!(num_blocks > 0, "num_kvcache_blocks must be positive");
        let eos_token_id = u32::try_from(config.eos)
            .map_err(|_| anyhow!("eos must be non-negative before scheduler init"))?;

        Ok(Self {
            max_num_seqs: config.max_num_seqs,
            max_num_batched_tokens: config.max_num_batched_tokens,
            eos_token_id,
            block_manager: BlockManager::new(num_blocks, config.kvcache_block_size, prefix_cfg),
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            next_seq_id: 0,
        })
    }

    pub fn add(&mut self, mut seq: Sequence) -> usize {
        seq.id = self.next_seq_id;
        let id = seq.id;
        self.next_seq_id += 1;
        self.waiting.push_back(seq);
        id
    }

    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn get_running_sequence(&self, seq_id: usize) -> Option<&Sequence> {
        self.running.iter().find(|seq| seq.id == seq_id)
    }

    pub fn get_running_sequences<'a>(&'a self, seq_ids: &[usize]) -> Result<Vec<&'a Sequence>> {
        let mut out = Vec::with_capacity(seq_ids.len());
        for &seq_id in seq_ids {
            let seq = self
                .get_running_sequence(seq_id)
                .ok_or_else(|| anyhow!("running sequence {} not found", seq_id))?;
            out.push(seq);
        }
        Ok(out)
    }

    pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;

        while !self.waiting.is_empty() && num_seqs < self.max_num_seqs {
            let should_break = {
                let seq = self.waiting.front().expect("waiting is not empty");
                num_batched_tokens + seq.len() > self.max_num_batched_tokens
                    || !self.block_manager.can_allocate(seq)
            };
            if should_break {
                break;
            }

            let mut seq = self.waiting.pop_front().expect("waiting is not empty");
            self.block_manager.allocate(&mut seq)?;
            num_batched_tokens += seq.len().saturating_sub(seq.num_cached_tokens);
            seq.status = SequenceStatus::Running;
            scheduled_ids.push(seq.id);
            self.running.push_back(seq);
            num_seqs += 1;
        }

        if !scheduled_ids.is_empty() {
            return Ok((scheduled_ids, true));
        }

        let mut scheduled_decode = Vec::new();
        while !self.running.is_empty() && num_seqs < self.max_num_seqs {
            let mut current = Some(self.running.pop_front().expect("running is not empty"));
            let mut preempted_current = false;
            while !self
                .block_manager
                .can_append(current.as_ref().expect("current is set"))
            {
                if let Some(preempt_seq) = self.running.pop_back() {
                    self.preempt(preempt_seq);
                } else {
                    self.preempt(current.take().expect("current is set"));
                    preempted_current = true;
                    break;
                }
            }

            if preempted_current {
                continue;
            }

            let mut seq = current.expect("current is set");
            self.block_manager.may_append(&mut seq)?;
            scheduled_ids.push(seq.id);
            scheduled_decode.push(seq);
            num_seqs += 1;
        }

        ensure!(
            !scheduled_decode.is_empty(),
            "no sequences available to schedule"
        );
        for seq in scheduled_decode.into_iter().rev() {
            self.running.push_front(seq);
        }
        Ok((scheduled_ids, false))
    }

    pub fn postprocess(&mut self, seq_ids: &[usize], output_ids: &[u32]) -> Result<Vec<usize>> {
        ensure!(
            seq_ids.len() == output_ids.len(),
            "postprocess input size mismatch"
        );

        let mut finished_ids = Vec::new();
        for (&seq_id, &token_id) in seq_ids.iter().zip(output_ids.iter()) {
            let seq = self
                .running
                .iter_mut()
                .find(|seq| seq.id == seq_id)
                .ok_or_else(|| anyhow!("running sequence {} not found", seq_id))?;

            seq.append_token(token_id);
            let reached_eos = !seq.sampling_params.ignore_eos && token_id == self.eos_token_id;
            let reached_max = seq.num_completion_tokens() == seq.sampling_params.max_tokens;
            if reached_eos || reached_max {
                seq.status = SequenceStatus::Finished;
                finished_ids.push(seq.id);
            }
        }

        if finished_ids.is_empty() {
            return Ok(finished_ids);
        }

        let finished_set: HashSet<usize> = finished_ids.iter().copied().collect();
        let mut kept = VecDeque::with_capacity(self.running.len());
        while let Some(mut seq) = self.running.pop_front() {
            if finished_set.contains(&seq.id) {
                self.block_manager.deallocate(&mut seq);
            } else {
                kept.push_back(seq);
            }
        }
        self.running = kept;
        Ok(finished_ids)
    }

    fn preempt(&mut self, mut seq: Sequence) {
        seq.status = SequenceStatus::Waiting;
        self.block_manager.deallocate(&mut seq);
        self.waiting.push_front(seq);
    }
}
