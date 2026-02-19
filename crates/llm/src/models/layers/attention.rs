use std::collections::HashMap;

use anyhow::{Result, ensure};
use candle_core::{IndexOp, Tensor};
use candle_nn::ops::softmax_last_dim;

#[derive(Debug, Clone)]
pub struct KvSlot {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Debug, Clone)]
pub struct KvCache {
    num_slots: usize,
    num_kv_heads: usize,
    head_dim: usize,
    slots: HashMap<usize, KvSlot>,
}

impl KvCache {
    pub fn new(num_slots: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        ensure!(num_slots > 0, "num_slots must be positive");
        ensure!(num_kv_heads > 0, "num_kv_heads must be positive");
        ensure!(head_dim > 0, "head_dim must be positive");
        Ok(Self {
            num_slots,
            num_kv_heads,
            head_dim,
            slots: HashMap::new(),
        })
    }

    pub fn write_from_mapping(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        slot_mapping: &[i32],
    ) -> Result<()> {
        ensure!(key.dims() == value.dims(), "key/value shapes must match");
        let (tokens, kv_heads, head_dim) = key.dims3()?;
        ensure!(
            kv_heads == self.num_kv_heads && head_dim == self.head_dim,
            "key/value shape must match cache dimensions"
        );
        ensure!(
            slot_mapping.len() == tokens,
            "slot_mapping must have one entry per token"
        );

        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            ensure!(slot < self.num_slots, "slot index out of range");
            self.slots.insert(
                slot,
                KvSlot {
                    key: key.i(token_idx)?.contiguous()?,
                    value: value.i(token_idx)?.contiguous()?,
                },
            );
        }
        Ok(())
    }

    pub fn get(&self, slot: usize) -> Option<&KvSlot> {
        self.slots.get(&slot)
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct Attention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        ensure!(num_heads > 0, "num_heads must be positive");
        ensure!(num_kv_heads > 0, "num_kv_heads must be positive");
        ensure!(head_dim > 0, "head_dim must be positive");
        ensure!(
            num_heads.is_multiple_of(num_kv_heads),
            "num_heads must be divisible by num_kv_heads"
        );
        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(),
        })
    }

    pub fn write_kv_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        slot_mapping: &[i32],
        cache: &mut KvCache,
    ) -> Result<()> {
        cache.write_from_mapping(key, value, slot_mapping)
    }

    pub fn forward_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        let (q_tokens, q_heads, q_dim) = query.dims3()?;
        let (kv_tokens, kv_heads, kv_dim) = key.dims3()?;
        ensure!(key.dims() == value.dims(), "key/value shapes must match");
        ensure!(q_tokens == kv_tokens, "query and key token dims must match");
        ensure!(q_heads == self.num_heads, "query head count mismatch");
        ensure!(q_dim == self.head_dim, "query head_dim mismatch");
        ensure!(
            kv_heads == self.num_kv_heads,
            "key/value head count mismatch"
        );
        ensure!(kv_dim == self.head_dim, "key/value head_dim mismatch");

        let q = query.transpose(0, 1)?.contiguous()?;
        let k = expand_kv_heads(key, self.num_heads)?
            .transpose(0, 1)?
            .contiguous()?;
        let v = expand_kv_heads(value, self.num_heads)?
            .transpose(0, 1)?
            .contiguous()?;

        let mut scores = q
            .matmul(&k.transpose(1, 2)?)?
            .affine(1.0 / self.scale, 0.0)?;
        if causal {
            scores = apply_causal_mask(scores)?;
        }
        let probs = softmax_last_dim(&scores)?;
        Ok(probs.matmul(&v)?.transpose(0, 1)?.contiguous()?)
    }

    pub fn forward_decode(
        &self,
        query: &Tensor,
        cache: &KvCache,
        block_tables: &[Vec<u32>],
        context_lens: &[usize],
        block_size: usize,
    ) -> Result<Tensor> {
        let (batch_size, q_heads, q_dim) = query.dims3()?;
        ensure!(batch_size > 0, "decode batch must not be empty");
        ensure!(q_heads == self.num_heads, "query head count mismatch");
        ensure!(q_dim == self.head_dim, "query head_dim mismatch");
        ensure!(
            block_tables.len() == batch_size,
            "block_tables rows must match decode batch size"
        );
        ensure!(
            context_lens.len() == batch_size,
            "context_lens rows must match decode batch size"
        );
        ensure!(block_size > 0, "block_size must be positive");

        let mut outputs = Vec::with_capacity(batch_size);
        for seq_idx in 0..batch_size {
            let context_len = context_lens[seq_idx];
            ensure!(context_len > 0, "decode context length must be positive");
            let table = &block_tables[seq_idx];
            ensure!(!table.is_empty(), "decode block table must not be empty");

            let mut key_tokens = Vec::with_capacity(context_len);
            let mut value_tokens = Vec::with_capacity(context_len);
            for pos in 0..context_len {
                let slot = slot_for_position(table, pos, block_size)?;
                let slot_data = cache
                    .get(slot)
                    .ok_or_else(|| anyhow::anyhow!("missing kv slot {} in decode cache", slot))?;
                key_tokens.push(slot_data.key.clone());
                value_tokens.push(slot_data.value.clone());
            }

            let key_refs: Vec<&Tensor> = key_tokens.iter().collect();
            let value_refs: Vec<&Tensor> = value_tokens.iter().collect();
            let key = Tensor::stack(&key_refs, 0)?;
            let value = Tensor::stack(&value_refs, 0)?;
            let query_seq = query.narrow(0, seq_idx, 1)?;
            let out = self.forward_prefill(&query_seq, &key, &value, false)?;
            outputs.push(out);
        }

        let output_refs: Vec<&Tensor> = outputs.iter().collect();
        Ok(Tensor::cat(&output_refs, 0)?)
    }
}

fn slot_for_position(block_table: &[u32], position: usize, block_size: usize) -> Result<usize> {
    let block_idx = position / block_size;
    ensure!(
        block_idx < block_table.len(),
        "decode block index {} out of range for table length {}",
        block_idx,
        block_table.len()
    );
    let offset = position % block_size;
    let base = (block_table[block_idx] as usize)
        .checked_mul(block_size)
        .ok_or_else(|| anyhow::anyhow!("slot id overflow"))?;
    base.checked_add(offset)
        .ok_or_else(|| anyhow::anyhow!("slot id overflow"))
}

fn apply_causal_mask(scores: Tensor) -> Result<Tensor> {
    let (heads, q_tokens, k_tokens) = scores.dims3()?;
    let mut mask = vec![f32::NEG_INFINITY; q_tokens * k_tokens];
    for q in 0..q_tokens {
        for k in 0..k_tokens {
            if k <= q {
                mask[q * k_tokens + k] = 0.0;
            }
        }
    }
    let bias = Tensor::from_vec(mask, (1, q_tokens, k_tokens), scores.device())?
        .broadcast_as((heads, q_tokens, k_tokens))?;
    Ok(scores.broadcast_add(&bias)?)
}

fn expand_kv_heads(kv: &Tensor, target_heads: usize) -> Result<Tensor> {
    let (_, kv_heads, _) = kv.dims3()?;
    if kv_heads == target_heads {
        return Ok(kv.clone());
    }
    ensure!(kv_heads > 0, "kv heads must be positive");
    ensure!(
        target_heads.is_multiple_of(kv_heads),
        "target_heads must be divisible by kv_heads"
    );

    let repeat_factor = target_heads / kv_heads;
    let mut repeated_heads = Vec::with_capacity(target_heads);
    for head_idx in 0..kv_heads {
        let head = kv.narrow(1, head_idx, 1)?;
        for _ in 0..repeat_factor {
            repeated_heads.push(head.clone());
        }
    }
    let refs: Vec<&Tensor> = repeated_heads.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}
