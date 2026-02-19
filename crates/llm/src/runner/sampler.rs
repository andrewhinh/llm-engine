use anyhow::{Result, anyhow, ensure};
use candle_core::{DType, Tensor};
use parking_lot::Mutex;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::{SeedableRng, rngs::StdRng};

use crate::utils::SamplingParams;

const MIN_TEMPERATURE: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub enum SamplingStrategy {
    ArgMax,
    All { temperature: f32 },
    TopK { k: usize, temperature: f32 },
    TopP { p: f32, temperature: f32 },
    TopKThenTopP { k: usize, p: f32, temperature: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenaltyConfig {
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl PenaltyConfig {
    pub fn from_sampling_params(params: &SamplingParams) -> Option<Self> {
        if params.frequency_penalty == 0.0 && params.presence_penalty == 0.0 {
            return None;
        }
        Some(Self {
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
        })
    }
}

#[derive(Debug)]
pub struct Sampler {
    rng: Mutex<StdRng>,
}

impl Sampler {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    pub fn strategy_from_params(params: &SamplingParams) -> SamplingStrategy {
        if params.temperature <= MIN_TEMPERATURE {
            return SamplingStrategy::ArgMax;
        }

        let k = (params.top_k > 0).then_some(params.top_k as usize);
        let p = (params.top_p < 1.0).then_some(params.top_p);
        let temperature = params.temperature;

        match (k, p) {
            (None, None) => SamplingStrategy::All { temperature },
            (Some(k), None) => SamplingStrategy::TopK { k, temperature },
            (None, Some(p)) => SamplingStrategy::TopP { p, temperature },
            (Some(k), Some(p)) => SamplingStrategy::TopKThenTopP { k, p, temperature },
        }
    }

    pub fn sample_from_params(
        &self,
        logits: &Tensor,
        params: &SamplingParams,
        token_histories: Option<&[Vec<u32>]>,
    ) -> Result<Vec<u32>> {
        let strategy = Self::strategy_from_params(params);
        let penalties = PenaltyConfig::from_sampling_params(params);
        self.sample(logits, &strategy, penalties, token_histories)
    }

    pub fn sample(
        &self,
        logits: &Tensor,
        strategy: &SamplingStrategy,
        penalties: Option<PenaltyConfig>,
        token_histories: Option<&[Vec<u32>]>,
    ) -> Result<Vec<u32>> {
        let logits = logits.to_dtype(DType::F32)?;
        let mut rows = logits.to_vec2::<f32>()?;

        if let Some(histories) = token_histories {
            ensure!(
                histories.len() == rows.len(),
                "token histories length {} must match batch size {}",
                histories.len(),
                rows.len()
            );
        }

        let mut out = Vec::with_capacity(rows.len());
        for (row_idx, row) in rows.iter_mut().enumerate() {
            if let (Some(cfg), Some(histories)) = (penalties, token_histories) {
                Self::apply_penalties(row, &histories[row_idx], cfg);
            }
            out.push(self.sample_row(row, strategy)?);
        }
        Ok(out)
    }

    pub fn apply_penalties(logits: &mut [f32], history: &[u32], cfg: PenaltyConfig) {
        if cfg.frequency_penalty == 0.0 && cfg.presence_penalty == 0.0 {
            return;
        }

        let mut counts = vec![0u32; logits.len()];
        for &token_id in history {
            let idx = token_id as usize;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }

        for (idx, logit) in logits.iter_mut().enumerate() {
            let count = counts[idx] as f32;
            if count == 0.0 {
                continue;
            }
            *logit -= cfg.frequency_penalty * count;
            *logit -= cfg.presence_penalty;
        }
    }

    fn sample_row(&self, logits: &[f32], strategy: &SamplingStrategy) -> Result<u32> {
        ensure!(!logits.is_empty(), "logits row must not be empty");
        match strategy {
            SamplingStrategy::ArgMax => Ok(Self::argmax_index(logits)? as u32),
            SamplingStrategy::All { temperature } => {
                let probs = Self::softmax_scaled(logits, *temperature)?;
                Ok(self.sample_probs_index(&probs)? as u32)
            }
            SamplingStrategy::TopK { k, temperature } => {
                let (indices, probs) = Self::topk_probs(logits, *k, *temperature)?;
                let picked = self.sample_probs_index(&probs)?;
                Ok(indices[picked] as u32)
            }
            SamplingStrategy::TopP { p, temperature } => {
                let (indices, probs) = Self::topp_probs(logits, *p, *temperature)?;
                let picked = self.sample_probs_index(&probs)?;
                Ok(indices[picked] as u32)
            }
            SamplingStrategy::TopKThenTopP { k, p, temperature } => {
                let (indices, probs) = Self::topk_topp_probs(logits, *k, *p, *temperature)?;
                let picked = self.sample_probs_index(&probs)?;
                Ok(indices[picked] as u32)
            }
        }
    }

    fn sample_probs_index(&self, probs: &[f32]) -> Result<usize> {
        let dist = WeightedIndex::new(probs)
            .map_err(|_| anyhow!("invalid probabilities: must contain positive finite values"))?;
        let mut rng = self.rng.lock();
        Ok(dist.sample(&mut *rng))
    }

    fn argmax_index(values: &[f32]) -> Result<usize> {
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("empty values for argmax"))
    }

    fn softmax_scaled(logits: &[f32], temperature: f32) -> Result<Vec<f32>> {
        ensure!(temperature > 0.0, "temperature must be positive");
        let scaled: Vec<f32> = logits.iter().map(|x| *x / temperature).collect();
        Ok(Self::softmax(&scaled))
    }

    fn softmax(values: &[f32]) -> Vec<f32> {
        let max = values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |acc, x| acc.max(x));
        let mut exps: Vec<f32> = values.iter().map(|x| (*x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            let mut probs = vec![0.0; values.len()];
            let idx = values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            probs[idx] = 1.0;
            return probs;
        }
        for x in &mut exps {
            *x /= sum;
        }
        exps
    }

    fn topk_probs(logits: &[f32], k: usize, temperature: f32) -> Result<(Vec<usize>, Vec<f32>)> {
        ensure!(k > 0, "top-k must be positive");
        let probs = Self::softmax_scaled(logits, temperature)?;
        let mut pairs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let kept = pairs
            .into_iter()
            .take(k.min(logits.len()))
            .collect::<Vec<_>>();
        let (indices, mut kept_probs): (Vec<usize>, Vec<f32>) = kept.into_iter().unzip();
        Self::renorm(&mut kept_probs);
        Ok((indices, kept_probs))
    }

    fn topp_probs(logits: &[f32], p: f32, temperature: f32) -> Result<(Vec<usize>, Vec<f32>)> {
        ensure!(p > 0.0 && p <= 1.0, "top-p must be in (0, 1]");
        let probs = Self::softmax_scaled(logits, temperature)?;
        let mut pairs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut cum = 0.0f32;
        let mut kept = Vec::new();
        for (idx, prob) in pairs {
            kept.push((idx, prob));
            cum += prob;
            if cum >= p {
                break;
            }
        }

        let (indices, mut kept_probs): (Vec<usize>, Vec<f32>) = kept.into_iter().unzip();
        Self::renorm(&mut kept_probs);
        Ok((indices, kept_probs))
    }

    fn topk_topp_probs(
        logits: &[f32],
        k: usize,
        p: f32,
        temperature: f32,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let (topk_indices, topk_probs) = Self::topk_probs(logits, k, temperature)?;
        if p >= 1.0 {
            return Ok((topk_indices, topk_probs));
        }

        let mut pairs: Vec<(usize, f32)> = topk_indices.into_iter().zip(topk_probs).collect();
        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut cum = 0.0f32;
        let mut kept = Vec::new();
        for (idx, prob) in pairs {
            kept.push((idx, prob));
            cum += prob;
            if cum >= p {
                break;
            }
        }

        let (indices, mut kept_probs): (Vec<usize>, Vec<f32>) = kept.into_iter().unzip();
        Self::renorm(&mut kept_probs);
        Ok((indices, kept_probs))
    }

    fn renorm(probs: &mut [f32]) {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for prob in probs {
                *prob /= sum;
            }
        }
    }
}
