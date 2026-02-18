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

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.dims() != self.weight.dims() {
            candle_core::bail!(
                "weight shape mismatch: expected {:?}, got {:?}",
                self.weight.dims(),
                weight.dims()
            );
        }
        let mut weight = weight.to_device(self.weight.device())?;
        if weight.dtype() != self.weight.dtype() {
            weight = weight.to_dtype(self.weight.dtype())?;
        }
        self.weight = weight;
        Ok(())
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        let current = match self.bias.as_ref() {
            Some(current) => current,
            None => candle_core::bail!("cannot load bias into linear layer without bias"),
        };
        let expected = current.dims().to_vec();
        if bias.dims() != expected.as_slice() {
            candle_core::bail!(
                "bias shape mismatch: expected {:?}, got {:?}",
                expected,
                bias.dims()
            );
        }
        let mut bias = bias.to_device(current.device())?;
        if bias.dtype() != current.dtype() {
            bias = bias.to_dtype(current.dtype())?;
        }
        self.bias = Some(bias);
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul(&self.weight.t()?)?;
        match &self.bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }
}
