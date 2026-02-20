use anyhow::Result;
#[cfg(feature = "flashinfer")]
use anyhow::ensure;
#[cfg(feature = "flashinfer")]
use attention_rs::{FlashInferMetadata, InputMetadata};
#[cfg(feature = "flashinfer")]
use candle_core::DType;
use candle_core::Tensor;
#[cfg(feature = "flashinfer")]
use tracing::warn;

use crate::attention::{AttentionBackend, EagerAttentionBackend, KvCache};
use crate::models::layers::context::RuntimeContext;

#[derive(Debug, Clone)]
pub struct FlashInferBackend {
    fallback: EagerAttentionBackend,
    #[cfg(feature = "flashinfer")]
    num_heads: usize,
    #[cfg(feature = "flashinfer")]
    num_kv_heads: usize,
    #[cfg(feature = "flashinfer")]
    head_dim: usize,
    #[cfg(feature = "flashinfer")]
    paged: attention_rs::PagedAttention,
}

impl FlashInferBackend {
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let fallback = EagerAttentionBackend::new(num_heads, num_kv_heads, head_dim)?;
        #[cfg(not(feature = "flashinfer"))]
        let _ = device;
        #[cfg(feature = "flashinfer")]
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
        #[cfg(not(feature = "flashinfer"))]
        Ok(Self { fallback })
    }

    #[cfg(feature = "flashinfer")]
    fn build_metadata(
        &self,
        query: &Tensor,
        cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<InputMetadata> {
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
        let mut flashinfer_metadata = self.build_flashinfer_metadata(query, cache, ctx)?;
        if !ctx.is_prefill {
            let plan = attention_rs::flashinfer::decode_plan(
                query.device(),
                cache.key_cache().dtype(),
                query.dtype(),
                &flashinfer_metadata.indptr_host,
                flashinfer_metadata.last_len_host.as_deref(),
                flashinfer_metadata.kv_len_arr_host.as_deref(),
                seq_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                cache.block_size(),
                flashinfer_metadata.use_cuda_graph,
            )?;
            flashinfer_metadata.decode_plan_info = Some(plan);
        }
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
            disable_flash_attn: Some(true),
            seqlens: None,
            flashinfer_metadata: Some(flashinfer_metadata),
        })
    }

    #[cfg(feature = "flashinfer")]
    fn build_flashinfer_metadata(
        &self,
        _query: &Tensor,
        _cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<FlashInferMetadata> {
        let metadata = ctx
            .flashinfer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("flashinfer runtime metadata missing"))?;
        Ok(FlashInferMetadata {
            indptr: metadata.indptr.clone(),
            indptr_host: metadata.indptr_host.clone(),
            indices: metadata.indices.clone(),
            last_len: metadata.last_len.clone(),
            last_len_host: metadata.last_len_host.clone(),
            kv_len_arr_host: metadata.kv_len_arr_host.clone(),
            cu_seqlens_q_host: metadata.cu_seqlens_q_host.clone(),
            total_num_rows: metadata.total_num_rows,
            batch_indices: metadata.batch_indices.clone(),
            positions: metadata.positions.clone(),
            use_cuda_graph: metadata.use_cuda_graph,
            decode_plan_info: None,
        })
    }

    #[cfg(feature = "flashinfer")]
    fn run_paged_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        cache: &KvCache,
        ctx: &RuntimeContext,
    ) -> Result<Tensor> {
        let metadata = self.build_metadata(query, cache, ctx)?;
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

impl AttentionBackend for FlashInferBackend {
    fn name(&self) -> &'static str {
        "fi"
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
        #[cfg(feature = "flashinfer")]
        {
            if let Some(cache) = cache {
                match self.run_paged_attention(query, key, value, cache, ctx) {
                    Ok(out) => return Ok(out),
                    Err(err) => warn!("flashinfer prefill failed, using eager fallback: {err}"),
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
        #[cfg(feature = "flashinfer")]
        {
            ensure!(
                query.dims3()?.2 == self.head_dim,
                "flashinfer decode head_dim mismatch"
            );
            let empty = Tensor::zeros(
                (query.dims3()?.0, self.num_kv_heads, self.head_dim),
                query.dtype(),
                query.device(),
            )?;
            match self.run_paged_attention(query, &empty, &empty, cache, ctx) {
                Ok(out) => return Ok(out),
                Err(err) => warn!("flashinfer decode failed, using eager fallback: {err}"),
            }
        }
        self.fallback.forward_decode(query, cache, ctx)
    }
}

#[cfg(feature = "flashinfer")]
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

#[cfg(feature = "flashinfer")]
fn unpack_output(out: Tensor, tokens: usize, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    match out.rank() {
        4 => {
            let (_, _, out_tokens, out_dim) = out.dims4()?;
            ensure!(
                out_tokens == tokens,
                "flashinfer output token shape mismatch"
            );
            ensure!(out_dim == head_dim, "flashinfer output head_dim mismatch");
            Ok(out
                .transpose(1, 2)?
                .reshape((tokens, num_heads, head_dim))?
                .contiguous()?)
        }
        3 => Ok(out),
        2 => Ok(out.reshape((tokens, num_heads, head_dim))?),
        other => anyhow::bail!("unexpected flashinfer output rank {other}"),
    }
}
