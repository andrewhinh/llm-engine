use std::cell::RefCell;

use anyhow::{Result, ensure};
use candle_core::{DType, Tensor};

#[derive(Debug, Clone)]
pub struct FlashInferRuntimeMetadata {
    pub indptr: Tensor,
    pub indptr_host: Vec<u32>,
    pub indices: Tensor,
    pub last_len: Tensor,
    pub last_len_host: Option<Vec<u32>>,
    pub kv_len_arr_host: Option<Vec<u32>>,
    pub cu_seqlens_q_host: Option<Vec<u32>>,
    pub total_num_rows: Option<u32>,
    pub batch_indices: Option<Tensor>,
    pub positions: Option<Tensor>,
    pub use_cuda_graph: bool,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeContext {
    pub is_prefill: bool,
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_k: Option<Tensor>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub slot_mapping: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub block_tables: Option<Tensor>,
    pub block_size: usize,
    pub flashinfer: Option<FlashInferRuntimeMetadata>,
}

impl RuntimeContext {
    pub fn prefill_last_indices(&self) -> Result<Vec<u32>> {
        ensure!(
            self.is_prefill,
            "prefill_last_indices is only valid in prefill mode"
        );
        let cu = self
            .cu_seqlens_q
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing cu_seqlens_q for prefill"))?;
        let cu = cu.to_dtype(DType::U32)?;
        let cu = cu.to_vec1::<u32>()?;
        ensure!(
            cu.len() >= 2,
            "cu_seqlens_q must include at least one sequence"
        );
        ensure!(cu[0] == 0, "cu_seqlens_q must begin with 0, got {}", cu[0]);
        let mut out = Vec::with_capacity(cu.len() - 1);
        for value in cu.into_iter().skip(1) {
            ensure!(value > 0, "prefill sequence length must be positive");
            out.push(value - 1);
        }
        Ok(out)
    }
}

thread_local! {
    static RUNTIME_CONTEXT: RefCell<RuntimeContext> = RefCell::new(RuntimeContext::default());
}

pub fn get_context() -> RuntimeContext {
    RUNTIME_CONTEXT.with(|ctx| ctx.borrow().clone())
}

pub fn set_context(context: RuntimeContext) {
    RUNTIME_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = context;
    });
}

pub fn reset_context() {
    set_context(RuntimeContext::default());
}
