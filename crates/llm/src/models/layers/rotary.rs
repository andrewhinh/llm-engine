use anyhow::Result;
use anyhow::ensure;
use candle_core::{D, DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        base: f64,
        device: &Device,
    ) -> Result<Self> {
        ensure!(
            head_dim.is_multiple_of(2),
            "head_dim must be even for rotary embedding"
        );
        ensure!(
            max_position_embeddings > 0,
            "max_position_embeddings must be positive"
        );

        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1f32 / base.powf((2 * i) as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            head_dim,
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn forward(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        ensure!(
            query.rank() == 3,
            "query must be rank-3 [tokens, heads, dim]"
        );
        ensure!(key.rank() == 3, "key must be rank-3 [tokens, heads, dim]");
        ensure!(
            query.dim(0)? == key.dim(0)?,
            "query and key token dims must match for rotary embedding"
        );
        ensure!(
            query.dim(D::Minus1)? == self.head_dim,
            "query last dim must match rotary head_dim"
        );
        ensure!(
            key.dim(D::Minus1)? == self.head_dim,
            "key last dim must match rotary head_dim"
        );
        let cos = self.cos.index_select(positions, 0)?.unsqueeze(1)?;
        let sin = self.sin.index_select(positions, 0)?.unsqueeze(1)?;
        let query = apply_rotary_emb(query, &cos, &sin)?;
        let key = apply_rotary_emb(key, &cos, &sin)?;
        Ok((query, key))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let half = dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let y1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let y2 = (x2.broadcast_mul(cos)? + x1.broadcast_mul(sin)?)?;
    Ok(Tensor::cat(&[&y1, &y2], D::Minus1)?)
}
