use anyhow::Result;
#[cfg(feature = "flash-attn")]
use anyhow::ensure;
#[cfg(feature = "flash-attn")]
use attention_rs::InputMetadata;
#[cfg(feature = "flash-attn")]
use candle_core::DType;
use candle_core::Tensor;
#[cfg(feature = "flash-attn")]
use tracing::warn;

use crate::attention::{AttentionBackend, EagerAttentionBackend, KvCache};
use crate::models::layers::context::RuntimeContext;

#[derive(Debug, Clone)]
pub struct FlashAttentionBackend {
    fallback: EagerAttentionBackend,
    #[cfg(feature = "flash-attn")]
    num_heads: usize,
    #[cfg(feature = "flash-attn")]
    num_kv_heads: usize,
    #[cfg(feature = "flash-attn")]
    head_dim: usize,
    #[cfg(feature = "flash-attn")]
    paged: attention_rs::PagedAttention,
}

impl FlashAttentionBackend {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let fallback = EagerAttentionBackend::new(num_heads, num_kv_heads, head_dim)?;
        #[cfg(not(feature = "flash-attn"))]
        let _ = device;
        #[cfg(feature = "flash-attn")]
        {
            let paged = attention_rs::PagedAttention::new(
                num_heads,
                head_dim,
                (head_dim as f32).powf(-0.5),
                Some(num_kv_heads),
                None,
                device.clone(),
                None,
                false,
            )?;
            return Ok(Self {
                fallback,
                num_heads,
                num_kv_heads,
                head_dim,
                paged,
            });
        }
        #[cfg(not(feature = "flash-attn"))]
        Ok(Self { fallback })
    }

    #[cfg(feature = "flash-attn")]
    fn build_metadata(&self, query: &Tensor, ctx: &RuntimeContext) -> Result<InputMetadata> {
        let seq_len = query.dims3()?.0;
        let slot_mapping = if let Some(slot_mapping) = ctx.slot_mapping.as_ref() {
            slot_mapping.clone()
        } else {
            Tensor::zeros((seq_len,), DType::I64, query.device())?
        };
        let max_context_len = if let Some(context_lens) = ctx.context_lens.as_ref() {
            context_lens
                .to_vec1::<u32>()?
                .into_iter()
                .map(|v| v as usize)
                .max()
                .unwrap_or(0)
        } else {
            ctx.max_seqlen_k
        };
        Ok(InputMetadata {
            is_prefill: ctx.is_prefill,
            sequence_ids: None,
            mamba_slot_mapping: None,
            slot_mapping,
            block_tables: ctx.block_tables.clone(),
            context_lens: ctx.context_lens.clone(),
            cu_seqlens_q: ctx.cu_seqlens_q.clone(),
            cu_seqlens_k: ctx.cu_seqlens_k.clone(),
            max_seqlen_q: ctx.max_seqlen_q,
            max_seqlen_k: ctx.max_seqlen_k,
            max_context_len,
            disable_flash_attn: Some(false),
            seqlens: None,
            flashinfer_metadata: None,
        })
    }

    #[cfg(feature = "flash-attn")]
    fn run_paged_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<Tensor> {
        let metadata = self.build_metadata(query, ctx)?;
        let (q, k, v) = pack_qkv(
            query,
            key,
            value,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )?;
        let out = self.paged.forward(
            &q,
            &k,
            &v,
            None,
            Some(cache.key_cache().clone()),
            Some(cache.value_cache().clone()),
            &metadata,
            None,
        )?;
        unpack_output(out, query.dims3()?.0, self.num_heads, self.head_dim)
    }
}

impl AttentionBackend for FlashAttentionBackend {
    fn name(&self) -> &'static str {
        "fa"
    }

    fn write_kv_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        slot_mapping: &[i32],
        cache: &mut KvCache,
    ) -> Result<()> {
        self.fallback
            .write_kv_cache(key, value, slot_mapping, cache)
    }

    fn forward_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cache: Option<&KvCache>,
        ctx: &RuntimeContext,
    ) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            if let Some(cache) = cache {
                match self.run_paged_attention(query, key, value, cache, ctx) {
                    Ok(out) => return Ok(out),
                    Err(err) => warn!("flash-attn prefill failed, using eager fallback: {err}"),
                }
            }
        }
        self.fallback.forward_prefill(query, key, value, cache, ctx)
    }

    fn forward_decode(
        &self,
        query: &Tensor,
        cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            ensure!(
                query.dims3()?.2 == self.head_dim,
                "flash-attn decode head_dim mismatch"
            );
            let empty = Tensor::zeros(
                (query.dims3()?.0, self.num_kv_heads, self.head_dim),
                query.dtype(),
                query.device(),
            )?;
            match self.run_paged_attention(query, &empty, &empty, cache, ctx) {
                Ok(out) => return Ok(out),
                Err(err) => warn!("flash-attn decode failed, using eager fallback: {err}"),
            }
        }
        self.fallback.forward_decode(query, cache, ctx)
    }
}

#[cfg(feature = "flash-attn")]
fn pack_qkv(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let tokens = query.dims3()?.0;
    let q = query
        .reshape((1, tokens, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = key
        .reshape((1, tokens, num_kv_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = value
        .reshape((1, tokens, num_kv_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    Ok((q, k, v))
}

#[cfg(feature = "flash-attn")]
fn unpack_output(out: Tensor, tokens: usize, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    match out.rank() {
        4 => {
            let (_, _, out_tokens, out_dim) = out.dims4()?;
            ensure!(
                out_tokens == tokens,
                "flash-attn output token shape mismatch"
            );
            ensure!(out_dim == head_dim, "flash-attn output head_dim mismatch");
            Ok(out
                .transpose(1, 2)?
                .reshape((tokens, num_heads, head_dim))?
                .contiguous()?)
        }
        3 => Ok(out),
        2 => Ok(out.reshape((tokens, num_heads, head_dim))?),
        other => anyhow::bail!("unexpected flash-attn output rank {other}"),
    }
}
