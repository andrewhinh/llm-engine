use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use super::context::get_context;
use super::linear::Linear;

#[derive(Debug, Clone)]
pub struct VocabEmbedding {
    weight: Tensor,
}

impl VocabEmbedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn zeros(
        vocab_size: usize,
        embedding_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        ensure!(vocab_size > 0, "vocab_size must be positive");
        ensure!(embedding_dim > 0, "embedding_dim must be positive");
        let weight = Tensor::zeros((vocab_size, embedding_dim), dtype, device)?;
        Ok(Self::new(weight))
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let tokens = input_ids.dims1()?;
        ensure!(tokens > 0, "input_ids must not be empty");
        let ids = input_ids.to_dtype(DType::U32)?;
        Ok(self.weight.index_select(&ids, 0)?)
    }
}

#[derive(Debug, Clone)]
pub struct LmHead {
    proj: Linear,
}

impl LmHead {
    pub fn new(weight: Tensor) -> Self {
        Self {
            proj: Linear::new(weight, None),
        }
    }

    pub fn zeros(
        hidden_size: usize,
        vocab_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        ensure!(hidden_size > 0, "hidden_size must be positive");
        ensure!(vocab_size > 0, "vocab_size must be positive");
        let weight = Tensor::zeros((vocab_size, hidden_size), dtype, device)?;
        Ok(Self::new(weight))
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let context = get_context();
        let hidden_states = if context.is_prefill {
            let indices = context.prefill_last_indices()?;
            let index_len = indices.len();
            ensure!(index_len > 0, "prefill must include at least one sequence");
            let indices = Tensor::from_vec(indices, (index_len,), hidden_states.device())?;
            hidden_states.index_select(&indices, 0)?
        } else {
            hidden_states.clone()
        };
        Ok(self.proj.forward(&hidden_states)?)
    }
}
