use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, RmsNorm as CandleRmsNorm};

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
    inner: CandleRmsNorm,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        let inner = CandleRmsNorm::new(weight.clone(), eps);
        Self { weight, eps, inner }
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

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.dims() != self.weight.dims() {
            candle_core::bail!(
                "rmsnorm weight shape mismatch: expected {:?}, got {:?}",
                self.weight.dims(),
                weight.dims()
            );
        }
        let mut weight = weight.to_device(self.weight.device())?;
        if weight.dtype() != self.weight.dtype() {
            weight = weight.to_dtype(self.weight.dtype())?;
        }
        self.inner = CandleRmsNorm::new(weight.clone(), self.eps);
        self.weight = weight;
        Ok(())
    }
}
