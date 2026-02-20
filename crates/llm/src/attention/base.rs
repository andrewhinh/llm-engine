use anyhow::{Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};

use crate::models::layers::context::RuntimeContext;

#[derive(Debug, Clone)]
pub struct KvSlot {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Debug, Clone)]
pub struct KvCache {
    num_blocks: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_used: Vec<bool>,
    used_slots: usize,
}

impl KvCache {
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        ensure!(num_blocks > 0, "num_blocks must be positive");
        ensure!(block_size > 0, "block_size must be positive");
        ensure!(num_kv_heads > 0, "num_kv_heads must be positive");
        ensure!(head_dim > 0, "head_dim must be positive");
        let num_slots = num_blocks
            .checked_mul(block_size)
            .ok_or_else(|| anyhow::anyhow!("num_blocks * block_size overflow"))?;
        let key_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype,
            device,
        )?;
        let value_cache = Tensor::zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype,
            device,
        )?;
        Ok(Self {
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            key_cache,
            value_cache,
            slot_used: vec![false; num_slots],
            used_slots: 0,
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
            ensure!(slot < self.slot_used.len(), "slot index out of range");
            let block = slot / self.block_size;
            let offset = slot % self.block_size;
            let token_key =
                key.narrow(0, token_idx, 1)?
                    .reshape((1, 1, self.num_kv_heads, self.head_dim))?;
            let token_value =
                value
                    .narrow(0, token_idx, 1)?
                    .reshape((1, 1, self.num_kv_heads, self.head_dim))?;
            self.key_cache = self.key_cache.slice_assign(
                &[
                    block..block + 1,
                    offset..offset + 1,
                    0..self.num_kv_heads,
                    0..self.head_dim,
                ],
                &token_key,
            )?;
            self.value_cache = self.value_cache.slice_assign(
                &[
                    block..block + 1,
                    offset..offset + 1,
                    0..self.num_kv_heads,
                    0..self.head_dim,
                ],
                &token_value,
            )?;
            if !self.slot_used[slot] {
                self.slot_used[slot] = true;
                self.used_slots += 1;
            }
        }
        Ok(())
    }

    pub fn read_slot(&self, slot: usize) -> Result<Option<KvSlot>> {
        if slot >= self.slot_used.len() || !self.slot_used[slot] {
            return Ok(None);
        }
        let block = slot / self.block_size;
        let offset = slot % self.block_size;
        Ok(Some(KvSlot {
            key: self.key_cache.i((block, offset))?.contiguous()?,
            value: self.value_cache.i((block, offset))?.contiguous()?,
        }))
    }

    pub fn key_cache(&self) -> &Tensor {
        &self.key_cache
    }

    pub fn value_cache(&self) -> &Tensor {
        &self.value_cache
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn len(&self) -> usize {
        self.used_slots
    }

    pub fn is_empty(&self) -> bool {
        self.used_slots == 0
    }
}

pub trait AttentionBackend: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &'static str;

    fn write_kv_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        slot_mapping: &[i32],
        cache: &mut KvCache,
    ) -> Result<()>;

    fn forward_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cache: Option<&KvCache>,
        ctx: &RuntimeContext,
    ) -> Result<Tensor>;

    fn forward_decode(
        &self,
        query: &Tensor,
        cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<Tensor>;
}
