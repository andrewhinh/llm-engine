use candle_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn zeros(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::zeros((out_dim, in_dim), dtype, device)?;
        let bias = if bias {
            Some(Tensor::zeros((out_dim,), dtype, device)?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }
}
