use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm as CandleRmsNorm};

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: CandleRmsNorm,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            inner: CandleRmsNorm::new(weight, eps),
        }
    }

    pub fn ones(hidden_size: usize, eps: f64, dtype: DType, device: &Device) -> Result<Self> {
        let weight = Tensor::ones((hidden_size,), dtype, device)?;
        Ok(Self::new(weight, eps))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let merged = (x + residual)?;
        let normalized = self.inner.forward(&merged)?;
        Ok((normalized, merged))
    }
}
