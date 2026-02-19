use candle_core::{DType, Device, Result, Tensor};

use super::distributed::{Comm, shard_range, shard_size};

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

fn shard_tensor_dim(full: &Tensor, dim: usize, comm: &Comm) -> Result<Tensor> {
    if comm.world_size() == 1 {
        return Ok(full.clone());
    }
    let total = full.dim(dim)?;
    let range = shard_range(total, comm.rank(), comm.world_size())
        .map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let len = range.end - range.start;
    full.narrow(dim, range.start, len)
}

#[derive(Debug, Clone)]
pub struct TensorParallelColumnLinear {
    linear: Linear,
    comm: Comm,
}

impl TensorParallelColumnLinear {
    pub fn zeros(
        input_size: usize,
        output_size: usize,
        bias: bool,
        dtype: DType,
        device: &Device,
        comm: Comm,
    ) -> Result<Self> {
        let local_output = shard_size(output_size, comm.world_size())
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let linear = Linear::zeros(input_size, local_output, bias, dtype, device)?;
        Ok(Self { linear, comm })
    }

    pub fn set_weight_from_full(&mut self, full_weight: Tensor) -> Result<()> {
        let sharded = shard_tensor_dim(&full_weight, 0, &self.comm)?;
        self.linear.set_weight(sharded)
    }

    pub fn set_bias_from_full(&mut self, full_bias: Tensor) -> Result<()> {
        let sharded = shard_tensor_dim(&full_bias, 0, &self.comm)?;
        self.linear.set_bias(sharded)
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.linear.set_weight(weight)
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.linear.set_bias(bias)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }

    pub fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
}

#[derive(Debug, Clone)]
pub struct TensorParallelRowLinear {
    linear: Linear,
    comm: Comm,
}

impl TensorParallelRowLinear {
    pub fn zeros(
        input_size: usize,
        output_size: usize,
        bias: bool,
        dtype: DType,
        device: &Device,
        comm: Comm,
    ) -> Result<Self> {
        let local_input = shard_size(input_size, comm.world_size())
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let linear = Linear::zeros(local_input, output_size, bias, dtype, device)?;
        Ok(Self { linear, comm })
    }

    pub fn set_weight_from_full(&mut self, full_weight: Tensor) -> Result<()> {
        let sharded = shard_tensor_dim(&full_weight, 1, &self.comm)?;
        self.linear.set_weight(sharded)
    }

    pub fn set_bias_from_full(&mut self, full_bias: Tensor) -> Result<()> {
        self.linear.set_bias(full_bias)
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.linear.set_weight(weight)
    }

    pub fn set_bias(&mut self, bias: Tensor) -> Result<()> {
        self.linear.set_bias(bias)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.linear.forward(x)?;
        self.comm
            .all_reduce_sum(&y)
            .map_err(|e| candle_core::Error::msg(e.to_string()))
    }

    pub fn weight(&self) -> &Tensor {
        self.linear.weight()
    }
}
