use std::collections::HashMap;

use anyhow::{Result, anyhow, ensure};
use candle_core::{DType, Tensor};

use crate::core::Sequence;
use crate::models::layers::context::FlashInferRuntimeMetadata;
use crate::models::{Comm, KvCache, Qwen3ForCausalLM, RuntimeContext, reset_context, set_context};
use crate::runner::Sampler;
#[cfg(feature = "cuda-graph")]
use crate::utils::DecodeCudaGraph;
use crate::utils::{DecodeExecutionPlan, DecodeGraphCaptures, DecodeGraphRuntime};

#[derive(Debug)]
pub struct ModelRunner {
    model: Qwen3ForCausalLM,
    sampler: Sampler,
    kv_cache: KvCache,
    block_size: usize,
    comm: Comm,
    graph_runtime: DecodeGraphRuntime,
    graph_captures: DecodeGraphCaptures,
    graph_buffers: HashMap<usize, DecodeGraphCaptureState>,
}

#[derive(Debug)]
struct PreparedBatch {
    input_ids: Tensor,
    positions: Tensor,
    slot_mapping: Option<Vec<i32>>,
    decode_meta: Option<DecodeBatchMeta>,
}

#[derive(Debug)]
struct DecodeBatchMeta {
    slot_mapping: Vec<i32>,
    slot_mapping_tensor: Tensor,
    context_lens_tensor: Tensor,
    block_tables_tensor: Tensor,
    flashinfer: Option<FlashInferRuntimeMetadata>,
}

#[derive(Debug)]
struct DecodeGraphCaptureState {
    capture_batch: usize,
    input_ids: Tensor,
    positions: Tensor,
    slot_mapping: Vec<i32>,
    slot_mapping_tensor: Tensor,
    context_lens_tensor: Tensor,
    block_tables_tensor: Tensor,
    output: Tensor,
    #[cfg(feature = "cuda-graph")]
    graph: DecodeCudaGraph,
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
        let cfg = model.config();
        let kv_cache = KvCache::new(
            num_kvcache_blocks,
            block_size,
            model.local_num_kv_heads(),
            cfg.head_dim,
            DType::F32,
            model.device(),
        )?;
        let cuda_supported =
            cfg!(feature = "cuda-graph") && matches!(model.device(), candle_core::Device::Cuda(_));
        let graph_runtime = DecodeGraphRuntime::new(max_num_seqs, enforce_eager, cuda_supported);
        Ok(Self {
            model,
            sampler,
            kv_cache,
            block_size,
            comm,
            graph_runtime,
            graph_captures: DecodeGraphCaptures::new(),
            graph_buffers: HashMap::new(),
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

    pub fn graph_captures(&self) -> Vec<usize> {
        self.graph_captures.captured_batches()
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
            DecodeExecutionPlan::GraphReplay { capture_batch } => {
                self.run_decode_graph(capture_batch, prepared)
            }
        }
    }

    fn run_decode_graph(
        &mut self,
        capture_batch: usize,
        prepared: &PreparedBatch,
    ) -> Result<Tensor> {
        if !self.can_use_decode_graph(prepared) {
            return self.model.forward_logits(
                &prepared.input_ids,
                &prepared.positions,
                prepared.slot_mapping.as_deref(),
                Some(&mut self.kv_cache),
            );
        }
        let decode_batch = prepared.input_ids.dims1()?;
        ensure!(
            decode_batch > 0,
            "decode graph replay requires non-empty batch"
        );
        ensure!(
            decode_batch <= capture_batch,
            "decode batch {} exceeds capture batch {}",
            decode_batch,
            capture_batch
        );
        if !self.graph_captures.is_captured(capture_batch) {
            self.capture_decode_graph(capture_batch, prepared)?;
        }
        self.replay_decode_graph(capture_batch, prepared)
    }

    fn capture_decode_graph(
        &mut self,
        capture_batch: usize,
        prepared: &PreparedBatch,
    ) -> Result<()> {
        let decode_batch = prepared.input_ids.dims1()?;
        ensure!(
            decode_batch <= capture_batch,
            "cannot capture graph for batch {} with decode batch {}",
            capture_batch,
            decode_batch
        );
        let meta = prepared
            .decode_meta
            .as_ref()
            .ok_or_else(|| anyhow!("decode graph capture requires decode metadata"))?;
        let max_blocks = meta.block_tables_tensor.dims2()?.1;

        let input_ids_cpu = prepared.input_ids.to_device(&candle_core::Device::Cpu)?;
        let positions_cpu = prepared.positions.to_device(&candle_core::Device::Cpu)?;
        let slot_cpu = meta
            .slot_mapping_tensor
            .to_device(&candle_core::Device::Cpu)?;
        let context_cpu = meta
            .context_lens_tensor
            .to_device(&candle_core::Device::Cpu)?;
        let block_cpu = meta
            .block_tables_tensor
            .to_device(&candle_core::Device::Cpu)?;

        let mut capture_input_ids = input_ids_cpu.to_vec1::<u32>()?;
        capture_input_ids.resize(capture_batch, 0);
        let mut capture_positions = positions_cpu.to_vec1::<u32>()?;
        capture_positions.resize(capture_batch, 0);
        let mut capture_slot_mapping = slot_cpu.to_vec1::<i32>()?;
        capture_slot_mapping.resize(capture_batch, 0);
        let mut capture_context_lens = context_cpu.to_vec1::<u32>()?;
        capture_context_lens.resize(capture_batch, 0);
        let mut capture_block_tables = block_cpu.to_vec2::<u32>()?;
        capture_block_tables.resize(capture_batch, vec![0u32; max_blocks]);
        for row in &mut capture_block_tables {
            row.resize(max_blocks, 0);
        }
        let capture_block_tables_flat = capture_block_tables
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();

        let device = prepared.input_ids.device();
        let input_ids = Tensor::from_vec(capture_input_ids, (capture_batch,), device)?;
        let positions = Tensor::from_vec(capture_positions, (capture_batch,), device)?;
        let slot_mapping_tensor =
            Tensor::from_vec(capture_slot_mapping.clone(), (capture_batch,), device)?;
        let context_lens_tensor = Tensor::from_vec(capture_context_lens, (capture_batch,), device)?;
        let block_tables_tensor = Tensor::from_vec(
            capture_block_tables_flat,
            (capture_batch, max_blocks),
            device,
        )?;

        set_context(RuntimeContext {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: Some(slot_mapping_tensor.clone()),
            context_lens: Some(context_lens_tensor.clone()),
            block_tables: Some(block_tables_tensor.clone()),
            block_size: self.block_size,
            flashinfer: None,
        });
        #[cfg(feature = "cuda-graph")]
        DecodeCudaGraph::begin_capture(self.model.device())?;
        let output = self.model.forward_logits(
            &input_ids,
            &positions,
            Some(capture_slot_mapping.as_slice()),
            Some(&mut self.kv_cache),
        )?;
        #[cfg(feature = "cuda-graph")]
        let graph = DecodeCudaGraph::end_capture(self.model.device())?;

        self.graph_buffers.insert(
            capture_batch,
            DecodeGraphCaptureState {
                capture_batch,
                input_ids,
                positions,
                slot_mapping: capture_slot_mapping,
                slot_mapping_tensor,
                context_lens_tensor,
                block_tables_tensor,
                output,
                #[cfg(feature = "cuda-graph")]
                graph,
            },
        );
        self.graph_captures.mark_captured(capture_batch);
        Ok(())
    }

    fn replay_decode_graph(
        &mut self,
        capture_batch: usize,
        prepared: &PreparedBatch,
    ) -> Result<Tensor> {
        let state = self
            .graph_buffers
            .get_mut(&capture_batch)
            .ok_or_else(|| anyhow!("missing decode graph buffer for batch {}", capture_batch))?;
        let expected = state.capture_batch;
        ensure!(
            state.input_ids.dims1()? == expected && state.positions.dims1()? == expected,
            "decode graph buffer shape mismatch for capture batch {}",
            capture_batch
        );
        update_decode_graph_inputs(state, prepared)?;
        set_context(RuntimeContext {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: Some(state.slot_mapping_tensor.clone()),
            context_lens: Some(state.context_lens_tensor.clone()),
            block_tables: Some(state.block_tables_tensor.clone()),
            block_size: self.block_size,
            flashinfer: None,
        });
        #[cfg(feature = "cuda-graph")]
        state.graph.launch()?;
        let decode_batch = prepared.input_ids.dims1()?;
        Ok(state.output.narrow(0, 0, decode_batch)?)
    }

    fn can_use_decode_graph(&self, prepared: &PreparedBatch) -> bool {
        if !cfg!(feature = "cuda-graph") {
            return false;
        }
        if !self.graph_runtime.is_enabled() {
            return false;
        }
        if !matches!(self.model.device(), candle_core::Device::Cuda(_)) {
            return false;
        }
        let Some(meta) = prepared.decode_meta.as_ref() else {
            return false;
        };
        meta.flashinfer.is_none()
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
        let mut prefill_tokens = Vec::with_capacity(seqs.len());
        let mut batch_indices = Vec::new();

        for (seq_idx, seq) in seqs.iter().enumerate() {
            ensure!(
                seq.len() >= seq.num_cached_tokens,
                "sequence cached tokens exceed sequence length"
            );
            let seq_len = seq.len();
            let remaining_tokens = seq_len - seq.num_cached_tokens;
            let new_tokens = if seq.prefill_chunk_tokens > 0 {
                seq.prefill_chunk_tokens.min(remaining_tokens)
            } else {
                remaining_tokens
            };
            ensure!(
                new_tokens > 0,
                "prefill sequence must contribute at least one token"
            );
            let prefill_start = seq.num_cached_tokens;
            let prefill_end = prefill_start + new_tokens;
            input_ids.extend_from_slice(&seq.token_ids[prefill_start..prefill_end]);
            positions.extend((prefill_start as u32)..(prefill_end as u32));
            batch_indices.extend(std::iter::repeat_n(seq_idx as u32, new_tokens));
            prefill_tokens.push(new_tokens);

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
        let positions_i64 = positions.iter().map(|&p| p as i64).collect::<Vec<_>>();
        let input_ids = Tensor::from_vec(input_ids, (input_len,), device)?;
        let positions = Tensor::from_vec(positions, (input_len,), device)?;
        let cu_seqlens_q_host = cu_seqlens_q.clone();
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
        let flashinfer = self.build_flashinfer_prefill_metadata(
            seqs,
            &prefill_tokens,
            &batch_indices,
            &positions_i64,
            &cu_seqlens_q_host,
        )?;
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
            flashinfer,
        });

        Ok(PreparedBatch {
            input_ids,
            positions,
            slot_mapping: use_slot_mapping.then_some(slot_mapping),
            decode_meta: None,
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
        let flashinfer = self.build_flashinfer_decode_metadata(seqs)?;
        set_context(RuntimeContext {
            is_prefill: false,
            cu_seqlens_q: None,
            cu_seqlens_k: None,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            slot_mapping: Some(slot_mapping_tensor.clone()),
            context_lens: Some(context_lens_tensor.clone()),
            block_tables: Some(block_tables.clone()),
            block_size: self.block_size,
            flashinfer: flashinfer.clone(),
        });

        Ok(PreparedBatch {
            input_ids,
            positions,
            slot_mapping: Some(slot_mapping.clone()),
            decode_meta: Some(DecodeBatchMeta {
                slot_mapping,
                slot_mapping_tensor,
                context_lens_tensor,
                block_tables_tensor: block_tables,
                flashinfer,
            }),
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

    fn build_flashinfer_prefill_metadata(
        &self,
        seqs: &[&Sequence],
        prefill_tokens: &[usize],
        batch_indices: &[u32],
        positions: &[i64],
        cu_seqlens_q_host: &[u32],
    ) -> Result<Option<FlashInferRuntimeMetadata>> {
        #[cfg(feature = "flashinfer")]
        {
            ensure!(
                seqs.len() == prefill_tokens.len(),
                "prefill flashinfer metadata expects one token count per sequence"
            );
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();

            for (seq, &new_tokens) in seqs.iter().zip(prefill_tokens.iter()) {
                let effective_len = seq.num_cached_tokens + new_tokens;
                let max_blocks = seq.block_table.len();
                let num_blocks = if effective_len == 0 {
                    0
                } else {
                    (effective_len + self.block_size - 1) / self.block_size
                };
                let num_blocks = num_blocks.min(max_blocks);
                let table = &seq.block_table[..num_blocks];
                indices.extend(table.iter().copied());
                indptr.push(indices.len() as u32);
                let last = if effective_len == 0 {
                    0
                } else {
                    ((effective_len - 1) % self.block_size + 1) as u32
                };
                last_len.push(last);
            }

            if let Some((pos, bad_idx)) = indices
                .iter()
                .copied()
                .enumerate()
                .find(|(_, idx)| *idx as usize >= self.kv_cache.num_blocks())
            {
                anyhow::bail!(
                    "flashinfer prefill block index out of range: indices[{pos}]={bad_idx} >= num_blocks ({})",
                    self.kv_cache.num_blocks()
                );
            }

            let kv_len_arr_host = build_kv_len_arr_host(&indptr, &last_len, self.block_size)?;
            let device = self.model.device();
            let indptr_tensor = Tensor::from_vec(indptr.clone(), (indptr.len(),), device)?;
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), device)?;
            let last_len_tensor = Tensor::from_vec(last_len.clone(), (last_len.len(),), device)?;
            let batch_indices_tensor =
                Tensor::from_vec(batch_indices.to_vec(), (batch_indices.len(),), device)?;
            let positions_tensor =
                Tensor::from_vec(positions.to_vec(), (positions.len(),), device)?;

            return Ok(Some(FlashInferRuntimeMetadata {
                indptr: indptr_tensor,
                indptr_host: indptr,
                indices: indices_tensor,
                last_len: last_len_tensor,
                last_len_host: Some(last_len),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: Some(cu_seqlens_q_host.to_vec()),
                total_num_rows: cu_seqlens_q_host.last().copied(),
                batch_indices: Some(batch_indices_tensor),
                positions: Some(positions_tensor),
                use_cuda_graph: false,
            }));
        }
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (
                seqs,
                prefill_tokens,
                batch_indices,
                positions,
                cu_seqlens_q_host,
            );
            Ok(None)
        }
    }

    fn build_flashinfer_decode_metadata(
        &self,
        seqs: &[&Sequence],
    ) -> Result<Option<FlashInferRuntimeMetadata>> {
        #[cfg(feature = "flashinfer")]
        {
            let mut indptr = vec![0u32];
            let mut indices = Vec::new();
            let mut last_len = Vec::new();
            for seq in seqs {
                indices.extend(seq.block_table.iter().copied());
                indptr.push(indices.len() as u32);
                let len = seq.len();
                let last = if len == 0 {
                    0
                } else {
                    ((len - 1) % self.block_size + 1) as u32
                };
                last_len.push(last);
            }

            if let Some((pos, bad_idx)) = indices
                .iter()
                .copied()
                .enumerate()
                .find(|(_, idx)| *idx as usize >= self.kv_cache.num_blocks())
            {
                anyhow::bail!(
                    "flashinfer decode block index out of range: indices[{pos}]={bad_idx} >= num_blocks ({})",
                    self.kv_cache.num_blocks()
                );
            }

            let kv_len_arr_host = build_kv_len_arr_host(&indptr, &last_len, self.block_size)?;
            let device = self.model.device();
            let indptr_tensor = Tensor::from_vec(indptr.clone(), (indptr.len(),), device)?;
            let indices_tensor = Tensor::from_vec(indices.clone(), (indices.len(),), device)?;
            let last_len_tensor = Tensor::from_vec(last_len.clone(), (last_len.len(),), device)?;

            return Ok(Some(FlashInferRuntimeMetadata {
                indptr: indptr_tensor,
                indptr_host: indptr,
                indices: indices_tensor,
                last_len: last_len_tensor,
                last_len_host: Some(last_len),
                kv_len_arr_host: Some(kv_len_arr_host),
                cu_seqlens_q_host: None,
                total_num_rows: None,
                batch_indices: None,
                positions: None,
                use_cuda_graph: false,
            }));
        }
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = seqs;
            Ok(None)
        }
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

fn update_decode_graph_inputs(
    state: &mut DecodeGraphCaptureState,
    prepared: &PreparedBatch,
) -> Result<()> {
    let meta = prepared
        .decode_meta
        .as_ref()
        .ok_or_else(|| anyhow!("decode metadata missing for graph replay"))?;
    let capture_batch = state.capture_batch;
    let decode_batch = prepared.input_ids.dims1()?;
    ensure!(
        decode_batch <= capture_batch,
        "decode batch {} exceeds capture batch {}",
        decode_batch,
        capture_batch
    );

    let input_ids_cpu = prepared.input_ids.to_device(&candle_core::Device::Cpu)?;
    let positions_cpu = prepared.positions.to_device(&candle_core::Device::Cpu)?;
    let context_cpu = meta
        .context_lens_tensor
        .to_device(&candle_core::Device::Cpu)?;
    let block_cpu = meta
        .block_tables_tensor
        .to_device(&candle_core::Device::Cpu)?;

    let mut input_ids = input_ids_cpu.to_vec1::<u32>()?;
    input_ids.resize(capture_batch, 0);
    let mut positions = positions_cpu.to_vec1::<u32>()?;
    positions.resize(capture_batch, 0);
    let mut slot_mapping = meta.slot_mapping.clone();
    slot_mapping.resize(capture_batch, 0);
    let mut context_lens = context_cpu.to_vec1::<u32>()?;
    context_lens.resize(capture_batch, 0);

    let max_blocks = state.block_tables_tensor.dims2()?.1;
    let mut block_rows = block_cpu.to_vec2::<u32>()?;
    block_rows.resize(capture_batch, vec![0u32; max_blocks]);
    for row in &mut block_rows {
        row.resize(max_blocks, 0);
    }
    let block_flat = block_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();

    let device = state.input_ids.device();
    let input_ids_t = Tensor::from_vec(input_ids, (capture_batch,), device)?;
    let positions_t = Tensor::from_vec(positions, (capture_batch,), device)?;
    let slot_t = Tensor::from_vec(slot_mapping.clone(), (capture_batch,), device)?;
    let context_t = Tensor::from_vec(context_lens, (capture_batch,), device)?;
    let block_t = Tensor::from_vec(block_flat, (capture_batch, max_blocks), device)?;

    copy_tensor_data(&state.input_ids, &input_ids_t)?;
    copy_tensor_data(&state.positions, &positions_t)?;
    copy_tensor_data(&state.slot_mapping_tensor, &slot_t)?;
    copy_tensor_data(&state.context_lens_tensor, &context_t)?;
    copy_tensor_data(&state.block_tables_tensor, &block_t)?;
    state.slot_mapping = slot_mapping;
    Ok(())
}

fn copy_tensor_data(dst: &Tensor, src: &Tensor) -> Result<()> {
    ensure!(
        dst.shape().elem_count() == src.shape().elem_count(),
        "tensor copy element count mismatch"
    );
    let elem_count = dst.shape().elem_count();
    if elem_count == 0 {
        return Ok(());
    }
    let idx = Tensor::arange(0u32, elem_count as u32, dst.device())?;
    let src_flat = src.flatten_all()?;
    let dst_flat = dst.flatten_all()?;
    Ok(dst_flat.scatter_set(&idx, &src_flat, 0usize)?)
}

#[cfg(feature = "flashinfer")]
fn build_kv_len_arr_host(indptr: &[u32], last_len: &[u32], block_size: usize) -> Result<Vec<u32>> {
    ensure!(
        indptr.len() == last_len.len() + 1,
        "flashinfer indptr size must be last_len size + 1"
    );
    let mut out = Vec::with_capacity(last_len.len());
    for i in 0..last_len.len() {
        let num_pages = indptr[i + 1] - indptr[i];
        if num_pages == 0 {
            out.push(0);
            continue;
        }
        let full = (num_pages - 1)
            .checked_mul(block_size as u32)
            .ok_or_else(|| anyhow!("flashinfer kv len overflow"))?;
        out.push(full + last_len[i]);
    }
    Ok(out)
}
