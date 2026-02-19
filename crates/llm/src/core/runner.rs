use anyhow::{Result, anyhow, ensure};
use candle_core::Tensor;

use crate::core::Sequence;
use crate::models::{Comm, KvCache, Qwen3ForCausalLM, RuntimeContext, reset_context, set_context};
use crate::runner::Sampler;
use crate::utils::{DecodeExecutionPlan, DecodeGraphRuntime};

#[derive(Debug)]
pub struct ModelRunner {
    model: Qwen3ForCausalLM,
    sampler: Sampler,
    kv_cache: KvCache,
    block_size: usize,
    comm: Comm,
    graph_runtime: DecodeGraphRuntime,
}

#[derive(Debug)]
struct PreparedBatch {
    input_ids: Tensor,
    positions: Tensor,
    slot_mapping: Option<Vec<i32>>,
}

impl ModelRunner {
    pub fn new(
        model: Qwen3ForCausalLM,
        sampler: Sampler,
        num_kvcache_blocks: usize,
        block_size: usize,
        comm: Comm,
        max_num_seqs: usize,
        enforce_eager: bool,
    ) -> Result<Self> {
        ensure!(
            num_kvcache_blocks > 0,
            "num_kvcache_blocks must be positive"
        );
        ensure!(block_size > 0, "block_size must be positive");
        let num_slots = num_kvcache_blocks
            .checked_mul(block_size)
            .ok_or_else(|| anyhow!("num_kvcache_blocks * block_size overflow"))?;
        let cfg = model.config();
        let kv_cache = KvCache::new(num_slots, model.local_num_kv_heads(), cfg.head_dim)?;
        let graph_runtime = DecodeGraphRuntime::new(max_num_seqs, enforce_eager, false);
        Ok(Self {
            model,
            sampler,
            kv_cache,
            block_size,
            comm,
            graph_runtime,
        })
    }

    pub fn run(&mut self, seqs: &[&Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let result = (|| -> Result<Vec<u32>> {
            let prepared = if is_prefill {
                self.prepare_prefill(seqs)?
            } else {
                self.prepare_decode(seqs)?
            };
            let decode_batch = prepared.input_ids.dims1()?;
            let exec_plan = self.graph_runtime.plan(is_prefill, decode_batch);
            let logits = self.run_model_with_plan(&prepared, exec_plan)?;
            self.sample(&logits, seqs)
        })();
        reset_context();
        result
    }

    pub fn model(&self) -> &Qwen3ForCausalLM {
        &self.model
    }

    pub fn comm(&self) -> &Comm {
        &self.comm
    }

    pub fn kv_cache_size(&self) -> usize {
        self.kv_cache.len()
    }

    pub fn graph_runtime(&self) -> &DecodeGraphRuntime {
        &self.graph_runtime
    }

    fn run_model_with_plan(
        &mut self,
        prepared: &PreparedBatch,
        exec_plan: DecodeExecutionPlan,
    ) -> Result<Tensor> {
        match exec_plan {
            DecodeExecutionPlan::Eager => self.model.forward_logits(
                &prepared.input_ids,
                &prepared.positions,
                prepared.slot_mapping.as_deref(),
                Some(&mut self.kv_cache),
            ),
            DecodeExecutionPlan::GraphReplay { .. } => self.model.forward_logits(
                &prepared.input_ids,
                &prepared.positions,
                prepared.slot_mapping.as_deref(),
                Some(&mut self.kv_cache),
            ),
        }
    }

    fn prepare_prefill(&self, seqs: &[&Sequence]) -> Result<PreparedBatch> {
        ensure!(!seqs.is_empty(), "prefill requires at least one sequence");

        let mut input_ids = Vec::new();
        let mut positions = Vec::new();
        let mut cu_seqlens_q = vec![0u32];
        let mut cu_seqlens_k = vec![0u32];
        let mut max_seqlen_q = 0usize;
        let mut max_seqlen_k = 0usize;
        let mut slot_mapping = Vec::new();

        for seq in seqs {
            ensure!(
                seq.len() >= seq.num_cached_tokens,
                "sequence cached tokens exceed sequence length"
            );
            let seq_len = seq.len();
            let new_tokens = seq_len - seq.num_cached_tokens;
            input_ids.extend_from_slice(&seq.token_ids[seq.num_cached_tokens..]);
            positions.extend((seq.num_cached_tokens as u32)..(seq_len as u32));

            cu_seqlens_q.push(
                cu_seqlens_q
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(new_tokens as u32),
            );
            let k_len = if seq.num_cached_tokens > 0 {
                seq_len
            } else {
                new_tokens
            };
            cu_seqlens_k.push(
                cu_seqlens_k
                    .last()
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(k_len as u32),
            );
            max_seqlen_q = max_seqlen_q.max(new_tokens);
            max_seqlen_k = max_seqlen_k.max(k_len);

            if seq.block_table.is_empty() {
                continue;
            }

            let start_block = seq.num_cached_blocks();
            let total_blocks = seq.num_blocks();
            let mut mapped = 0usize;
            for block_idx in start_block..total_blocks {
                if mapped >= new_tokens {
                    break;
                }
                let block_id = seq.block_table[block_idx] as usize;
                let offset = if block_idx == start_block {
                    seq.num_cached_tokens % self.block_size
                } else {
                    0
                };
                let base = block_id
                    .checked_mul(self.block_size)
                    .ok_or_else(|| anyhow!("slot mapping overflow"))?;
                let start = base + offset;
                let remaining = new_tokens - mapped;
                let available = self.block_size - offset;
                let take = remaining.min(available);
                for slot in start..start + take {
                    slot_mapping
                        .push(i32::try_from(slot).map_err(|_| anyhow!("slot id exceeds i32"))?);
                }
                mapped += take;
            }

            ensure!(
                mapped == new_tokens,
                "slot mapping length mismatch for prefill sequence"
            );
        }

        ensure!(
            !input_ids.is_empty(),
            "prefill produced no input ids; nothing to run"
        );

        let use_slot_mapping = !slot_mapping.is_empty();
        if use_slot_mapping {
            ensure!(
                slot_mapping.len() == input_ids.len(),
                "prefill slot mapping must match input length"
            );
        }

        let device = self.model.device();
        let input_len = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (input_len,), device)?;
        let positions = Tensor::from_vec(positions, (input_len,), device)?;
        let q_len = cu_seqlens_q.len();
        let cu_seqlens_q = Tensor::from_vec(cu_seqlens_q, (q_len,), device)?;
        let k_len = cu_seqlens_k.len();
        let cu_seqlens_k = Tensor::from_vec(cu_seqlens_k, (k_len,), device)?;

        let block_tables = if max_seqlen_k > max_seqlen_q {
            Some(self.prepare_block_tables(seqs)?)
        } else {
            None
        };
        let context_lens = if block_tables.is_some() {
            let lens = seqs.iter().map(|seq| seq.len() as u32).collect::<Vec<_>>();
            let len = lens.len();
            Some(Tensor::from_vec(lens, (len,), device)?)
        } else {
            None
        };
        let slot_mapping_tensor = if use_slot_mapping {
            let s_len = slot_mapping.len();
            Some(Tensor::from_vec(slot_mapping.clone(), (s_len,), device)?)
        } else {
            None
        };
        set_context(RuntimeContext {
            is_prefill: true,
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping: slot_mapping_tensor,
            context_lens,
            block_tables,
            block_size: self.block_size,
        });

        Ok(PreparedBatch {
            input_ids,
            positions,
            slot_mapping: use_slot_mapping.then_some(slot_mapping),
        })
    }

    fn prepare_decode(&self, seqs: &[&Sequence]) -> Result<PreparedBatch> {
        ensure!(!seqs.is_empty(), "decode requires at least one sequence");

        let mut input_ids = Vec::with_capacity(seqs.len());
        let mut positions = Vec::with_capacity(seqs.len());
        let mut slot_mapping = Vec::with_capacity(seqs.len());
        let mut context_lens = Vec::with_capacity(seqs.len());

        for seq in seqs {
            ensure!(
                !seq.block_table.is_empty(),
                "decode sequence must have block_table"
            );
            ensure!(!seq.is_empty(), "decode sequence length must be positive");
            input_ids.push(seq.last_token);
            positions.push((seq.len() - 1) as u32);
            context_lens.push(seq.len() as u32);
            let block_id = *seq
                .block_table
                .last()
                .ok_or_else(|| anyhow!("decode sequence missing block_table tail"))?
                as usize;
            let slot = block_id
                .checked_mul(self.block_size)
                .and_then(|base| base.checked_add(seq.last_block_num_tokens().saturating_sub(1)))
                .ok_or_else(|| anyhow!("decode slot mapping overflow"))?;
            slot_mapping.push(i32::try_from(slot).map_err(|_| anyhow!("slot id exceeds i32"))?);
        }

        let device = self.model.device();
        let batch = input_ids.len();
        let input_ids = Tensor::from_vec(input_ids, (batch,), device)?;
        let positions = Tensor::from_vec(positions, (batch,), device)?;
        let slot_mapping_len = slot_mapping.len();
        let slot_mapping_tensor =
            Tensor::from_vec(slot_mapping.clone(), (slot_mapping_len,), device)?;
        let context_lens_len = context_lens.len();
        let context_lens_tensor = Tensor::from_vec(context_lens, (context_lens_len,), device)?;
        let block_tables = self.prepare_block_tables(seqs)?;
        set_context(RuntimeContext {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: Some(slot_mapping_tensor),
            context_lens: Some(context_lens_tensor),
            block_tables: Some(block_tables),
            block_size: self.block_size,
        });

        Ok(PreparedBatch {
            input_ids,
            positions,
            slot_mapping: Some(slot_mapping),
        })
    }

    fn prepare_block_tables(&self, seqs: &[&Sequence]) -> Result<Tensor> {
        ensure!(!seqs.is_empty(), "block table prep requires sequences");
        let max_len = seqs
            .iter()
            .map(|seq| seq.block_table.len())
            .max()
            .unwrap_or(0);
        ensure!(max_len > 0, "all block tables are empty");

        let mut flat = Vec::with_capacity(seqs.len() * max_len);
        for seq in seqs {
            flat.extend_from_slice(&seq.block_table);
            flat.extend(std::iter::repeat_n(0u32, max_len - seq.block_table.len()));
        }
        Ok(Tensor::from_vec(
            flat,
            (seqs.len(), max_len),
            self.model.device(),
        )?)
    }

    fn sample(&self, logits: &Tensor, seqs: &[&Sequence]) -> Result<Vec<u32>> {
        ensure!(!seqs.is_empty(), "sampling requires at least one sequence");
        let (rows, _) = logits.dims2()?;
        ensure!(
            rows == seqs.len(),
            "logit rows {} must match sequence count {}",
            rows,
            seqs.len()
        );

        let first_params = seqs[0].sampling_params.clone();
        let all_same_params = seqs.iter().all(|seq| seq.sampling_params == first_params);

        if all_same_params {
            let histories = seqs
                .iter()
                .map(|seq| seq.completion_token_ids().to_vec())
                .collect::<Vec<_>>();
            let history_opt = histories
                .iter()
                .any(|h| !h.is_empty())
                .then_some(histories.as_slice());
            return self
                .sampler
                .sample_from_params(logits, &first_params, history_opt);
        }

        let mut output = Vec::with_capacity(seqs.len());
        for (row_idx, seq) in seqs.iter().enumerate() {
            let row = logits.narrow(0, row_idx, 1)?;
            let row_history = (!seq.completion_token_ids().is_empty())
                .then_some(vec![seq.completion_token_ids().to_vec()]);
            let sampled = self.sampler.sample_from_params(
                &row,
                &seq.sampling_params,
                row_history.as_deref(),
            )?;
            let token = sampled
                .first()
                .copied()
                .ok_or_else(|| anyhow!("sampler returned no token"))?;
            output.push(token);
        }
        Ok(output)
    }
}
