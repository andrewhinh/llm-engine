use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow, ensure};

use crate::core::block_manager::BlockManager;
use crate::core::prefix_cache::PrefixCacheConfig;
use crate::core::sequence::{Sequence, SequenceStatus};
use crate::scheduler::{PrefillBudget, PrefillPolicy};
use crate::utils::config::EngineConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostprocessToken {
    pub seq_id: usize,
    pub token_id: u32,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostprocessOutput {
    pub tokens: Vec<PostprocessToken>,
    pub finished: Vec<(usize, Vec<u32>)>,
}

#[derive(Debug)]
pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    max_prefill_tokens: usize,
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
            max_prefill_tokens: config.max_prefill_tokens,
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

    pub fn contains(&self, seq_id: usize) -> bool {
        self.waiting.iter().any(|seq| seq.id == seq_id)
            || self.running.iter().any(|seq| seq.id == seq_id)
    }

    pub fn cancel(&mut self, seq_id: usize) -> bool {
        let removed = Self::remove_sequence(&mut self.waiting, seq_id)
            .or_else(|| Self::remove_sequence(&mut self.running, seq_id));
        let Some(mut seq) = removed else {
            return false;
        };
        seq.status = SequenceStatus::Finished;
        self.block_manager.deallocate(&mut seq);
        true
    }

    pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {
        let mut scheduled_ids = Vec::new();
        let mut num_seqs = 0usize;
        let mut num_batched_tokens = 0usize;
        let mut prefill_budget = PrefillBudget::new(PrefillPolicy {
            max_prefill_tokens: self.max_prefill_tokens,
            max_num_batched_tokens: self.max_num_batched_tokens,
        });

        while !self.waiting.is_empty()
            && num_seqs < self.max_num_seqs
            && !prefill_budget.is_exhausted()
        {
            let should_break = {
                let seq = self.waiting.front().expect("waiting is not empty");
                let max_batch_tokens_left = self
                    .max_num_batched_tokens
                    .saturating_sub(num_batched_tokens);
                let chunk_tokens = prefill_budget
                    .plan_chunk(seq.remaining_prefill_tokens(), max_batch_tokens_left);
                chunk_tokens == 0
                    || (seq.block_table.is_empty() && !self.block_manager.can_allocate(seq))
            };
            if should_break {
                break;
            }

            let mut seq = self.waiting.pop_front().expect("waiting is not empty");
            if seq.block_table.is_empty() {
                self.block_manager.allocate(&mut seq)?;
            }
            let max_batch_tokens_left = self
                .max_num_batched_tokens
                .saturating_sub(num_batched_tokens);
            let chunk_tokens =
                prefill_budget.plan_chunk(seq.remaining_prefill_tokens(), max_batch_tokens_left);
            ensure!(chunk_tokens > 0, "prefill chunk size must be positive");
            seq.prefill_chunk_tokens = chunk_tokens;
            num_batched_tokens += chunk_tokens;
            prefill_budget.consume(chunk_tokens);
            seq.status = SequenceStatus::Running;
            scheduled_ids.push(seq.id);
            self.running.push_back(seq);
            num_seqs += 1;
        }

        if !scheduled_ids.is_empty() {
            return Ok((scheduled_ids, true));
        }

        let mut scheduled_decode = Vec::new();
        'decode: while !self.running.is_empty() && num_seqs < self.max_num_seqs {
            let Some(mut seq) = self.running.pop_front() else {
                break;
            };
            while !self.block_manager.can_append(&seq) {
                if let Some(preempt_seq) = self.running.pop_back() {
                    self.preempt(preempt_seq);
                    continue;
                }
                self.preempt(seq);
                continue 'decode;
            }
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

    pub fn postprocess(
        &mut self,
        seq_ids: &[usize],
        output_ids: &[u32],
        is_prefill: bool,
    ) -> Result<PostprocessOutput> {
        ensure!(
            seq_ids.len() == output_ids.len(),
            "postprocess input size mismatch"
        );

        let mut emitted_tokens = Vec::with_capacity(seq_ids.len());
        let mut finished_ids = Vec::new();
        let mut requeue_ids = Vec::new();
        for (&seq_id, &token_id) in seq_ids.iter().zip(output_ids.iter()) {
            let seq = self
                .running
                .iter_mut()
                .find(|seq| seq.id == seq_id)
                .ok_or_else(|| anyhow!("running sequence {} not found", seq_id))?;

            if is_prefill {
                if Self::update_prefill_progress(seq)? {
                    seq.status = SequenceStatus::Waiting;
                    requeue_ids.push(seq.id);
                    continue;
                }
            } else {
                seq.clear_prefill_chunk_tokens();
            }

            seq.append_token(token_id);
            let reached_eos = !seq.sampling_params.ignore_eos && token_id == self.eos_token_id;
            let reached_max = seq.num_completion_tokens() == seq.sampling_params.max_tokens;
            let finished = reached_eos || reached_max;
            if finished {
                seq.status = SequenceStatus::Finished;
                finished_ids.push(seq.id);
            }
            emitted_tokens.push(Self::postprocess_token(seq.id, token_id, finished));
        }

        if finished_ids.is_empty() && requeue_ids.is_empty() {
            return Ok(PostprocessOutput {
                tokens: emitted_tokens,
                finished: Vec::new(),
            });
        }

        let finished = self.drain_finished_and_requeue(&finished_ids, &requeue_ids);
        Ok(PostprocessOutput {
            tokens: emitted_tokens,
            finished,
        })
    }

    fn remove_sequence(queue: &mut VecDeque<Sequence>, seq_id: usize) -> Option<Sequence> {
        let pos = queue.iter().position(|seq| seq.id == seq_id)?;
        queue.remove(pos)
    }

    fn update_prefill_progress(seq: &mut Sequence) -> Result<bool> {
        let consumed_prefill_tokens = if seq.prefill_chunk_tokens > 0 {
            seq.prefill_chunk_tokens.min(seq.remaining_prefill_tokens())
        } else {
            seq.remaining_prefill_tokens()
        };
        ensure!(
            consumed_prefill_tokens > 0,
            "prefill postprocess requires positive consumed tokens"
        );
        seq.num_cached_tokens = seq
            .num_cached_tokens
            .saturating_add(consumed_prefill_tokens)
            .min(seq.len());
        seq.clear_prefill_chunk_tokens();
        Ok(seq.num_cached_tokens < seq.len())
    }

    fn postprocess_token(seq_id: usize, token_id: u32, finished: bool) -> PostprocessToken {
        PostprocessToken {
            seq_id,
            token_id,
            finished,
        }
    }

    fn drain_finished_and_requeue(
        &mut self,
        finished_ids: &[usize],
        requeue_ids: &[usize],
    ) -> Vec<(usize, Vec<u32>)> {
        let finished_set: HashSet<usize> = finished_ids.iter().copied().collect();
        let requeue_set: HashSet<usize> = requeue_ids.iter().copied().collect();
        let mut finished = Vec::new();
        let mut kept = VecDeque::with_capacity(self.running.len());
        while let Some(mut seq) = self.running.pop_front() {
            if finished_set.contains(&seq.id) {
                finished.push((seq.id, seq.completion_token_ids().to_vec()));
                self.block_manager.deallocate(&mut seq);
            } else if requeue_set.contains(&seq.id) {
                seq.status = SequenceStatus::Waiting;
                self.waiting.push_back(seq);
            } else {
                kept.push_back(seq);
            }
        }
        self.running = kept;
        finished
    }

    fn preempt(&mut self, mut seq: Sequence) {
        seq.clear_prefill_chunk_tokens();
        seq.status = SequenceStatus::Waiting;
        self.block_manager.deallocate(&mut seq);
        self.waiting.push_front(seq);
    }
}
