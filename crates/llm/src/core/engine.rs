use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use candle_core::{DType, Device};
use tokenizers::Tokenizer;
use tracing::info;

use crate::core::prefix_cache::PrefixCacheConfig;
use crate::core::{ModelRunner, Scheduler, Sequence};
use crate::models::{Comm, Qwen3ForCausalLM};
use crate::runner::Sampler;
use crate::utils::config::{EngineConfig, SamplingParams};
use crate::utils::kvcache_allocator::{KVCacheAllocator, KVCachePlan};
use crate::utils::loader::load_qwen3_weights_from_model_path;
use crate::utils::tokenizer::{
    ModelConfig, decode_tokens, encode_prompt, load_model_config, load_tokenizer,
    load_tokenizer_config, resolve_eos_id,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationOutput {
    pub seq_id: usize,
    pub text: String,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedOutput {
    pub seq_id: usize,
    pub token_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepOutput {
    pub tokens: Vec<TokenOutput>,
    pub finished: Vec<FinishedOutput>,
    pub cancelled: Vec<CancelledOutput>,
    pub num_tokens: isize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenOutput {
    pub seq_id: usize,
    pub token_id: u32,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CancelledOutput {
    pub seq_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOutput {
    Token {
        seq_id: usize,
        token_id: u32,
        text: String,
        finished: bool,
    },
    Done(GenerationOutput),
    Cancelled(CancelledOutput),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptInput<'a> {
    Text(&'a str),
    TokenIds(&'a [u32]),
}

#[derive(Debug)]
pub struct Engine {
    pub config: EngineConfig,
    pub model_config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub kv_cache_plan: KVCachePlan,
    pub scheduler: Scheduler,
    pub runner: ModelRunner,
    pub comm: Comm,
    active_requests: HashSet<usize>,
    pending_cancellations: VecDeque<usize>,
}

pub struct EngineStream<'a> {
    engine: &'a mut Engine,
    pending_events: VecDeque<StreamOutput>,
    done: bool,
}

impl Engine {
    pub fn new(mut config: EngineConfig) -> Result<Self> {
        config.validate()?;
        let comm = Comm::bootstrap(config.tensor_parallel_size)?;

        let model_dir = PathBuf::from(&config.model);
        ensure!(
            model_dir.is_dir(),
            "model must be an existing directory: {}",
            model_dir.display()
        );

        let model_config = load_model_config(&model_dir)?;
        config.max_model_len = config
            .max_model_len
            .min(model_config.max_position_embeddings);
        ensure!(
            config.max_num_batched_tokens >= config.max_model_len,
            "max_num_batched_tokens must be >= clamped max_model_len"
        );
        let allocator = KVCacheAllocator::new(&config, &model_config)?;
        let kv_cache_plan = allocator.plan_auto()?;
        KVCacheAllocator::apply_plan(&mut config, &kv_cache_plan);
        ensure!(
            config.max_num_batched_tokens >= config.max_model_len,
            "max_num_batched_tokens must be >= max_model_len after kv cache planning"
        );

        let tokenizer = load_tokenizer(&model_dir)?;
        let tokenizer_config = load_tokenizer_config(&model_dir)?;
        config.eos = resolve_eos_id(&tokenizer, tokenizer_config.as_ref(), config.eos)?;

        let num_kvcache_blocks = usize::try_from(config.num_kvcache_blocks)
            .map_err(|_| anyhow!("num_kvcache_blocks must be positive after kv planning"))?;
        let mut max_cached_blocks = config
            .prefix_cache_max_cached_blocks
            .min(num_kvcache_blocks);
        if config.prefix_cache_enabled && max_cached_blocks == 0 {
            max_cached_blocks = num_kvcache_blocks / 2;
        }
        let prefix_cfg = PrefixCacheConfig {
            enabled: config.prefix_cache_enabled && max_cached_blocks > 0,
            max_cached_blocks,
            backend: config.prefix_cache_backend,
        };
        let scheduler = Scheduler::new(&config, prefix_cfg)?;

        let device = Device::Cpu;
        let dtype = DType::F32;
        let mut model = Qwen3ForCausalLM::from_model_config(
            &model_config,
            dtype,
            &device,
            comm.clone(),
            config.attention_backend_selection()?,
        )?;
        load_qwen3_weights_from_model_path(&mut model, &config.model, &device)?;
        let sampler = Sampler::from_seed(0);
        let runner = ModelRunner::new(
            model,
            sampler,
            num_kvcache_blocks,
            config.kvcache_block_size,
            comm.clone(),
            config.max_num_seqs,
            config.enforce_eager,
        )?;

        Ok(Self {
            config,
            model_config,
            tokenizer,
            kv_cache_plan,
            scheduler,
            runner,
            comm,
            active_requests: HashSet::new(),
            pending_cancellations: VecDeque::new(),
        })
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        encode_prompt(&self.tokenizer, prompt)
    }

    pub fn decode_tokens(&self, token_ids: &[u32]) -> Result<String> {
        decode_tokens(&self.tokenizer, token_ids)
    }

    pub fn add_request(
        &mut self,
        prompt: PromptInput<'_>,
        sampling_params: SamplingParams,
    ) -> Result<usize> {
        let token_ids = match prompt {
            PromptInput::Text(prompt) => self.encode_prompt(prompt)?,
            PromptInput::TokenIds(token_ids) => token_ids.to_vec(),
        };
        self.add_tokenized_request(token_ids, sampling_params)
    }

    pub fn add_tokenized_request(
        &mut self,
        token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<usize> {
        sampling_params.validate()?;
        ensure!(
            !token_ids.is_empty(),
            "prompt must contain at least one token"
        );
        let seq = Sequence::new(token_ids, self.config.kvcache_block_size, sampling_params);
        let seq_id = self.scheduler.add(seq);
        self.active_requests.insert(seq_id);
        Ok(seq_id)
    }

    pub fn is_finished(&self) -> bool {
        self.scheduler.is_finished()
    }

    pub fn step(&mut self) -> Result<StepOutput> {
        let cancelled = self.apply_pending_cancellations();
        if self.scheduler.is_finished() {
            return Ok(StepOutput {
                tokens: Vec::new(),
                finished: Vec::new(),
                cancelled,
                num_tokens: 0,
            });
        }

        let (scheduled_ids, is_prefill) = self.scheduler.schedule()?;
        let (output_ids, num_tokens) = {
            let seqs = self.scheduler.get_running_sequences(&scheduled_ids)?;
            let output_ids = self.runner.run(&seqs, is_prefill)?;
            ensure!(
                output_ids.len() == seqs.len(),
                "runner output size mismatch"
            );
            let num_tokens = if is_prefill {
                seqs.iter()
                    .map(|seq| {
                        if seq.prefill_chunk_tokens > 0 {
                            seq.prefill_chunk_tokens
                        } else {
                            seq.remaining_prefill_tokens()
                        }
                    })
                    .sum::<usize>() as isize
            } else {
                -(seqs.len() as isize)
            };
            (output_ids, num_tokens)
        };

        let postprocess = self
            .scheduler
            .postprocess(&scheduled_ids, &output_ids, is_prefill)?;
        let finished = postprocess
            .finished
            .into_iter()
            .map(|(seq_id, token_ids)| FinishedOutput { seq_id, token_ids })
            .collect::<Vec<_>>();
        let tokens = postprocess
            .tokens
            .into_iter()
            .map(|token| TokenOutput {
                seq_id: token.seq_id,
                token_id: token.token_id,
                finished: token.finished,
            })
            .collect::<Vec<_>>();
        for finished in &finished {
            self.active_requests.remove(&finished.seq_id);
        }

        Ok(StepOutput {
            tokens,
            finished,
            cancelled,
            num_tokens,
        })
    }

    pub fn generate_sync(
        &mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<Vec<GenerationOutput>> {
        ensure!(!prompts.is_empty(), "prompts must not be empty");
        let params = expand_sampling_params(prompts.len(), sampling_params)?;

        let mut submission_order = Vec::with_capacity(prompts.len());
        for (prompt, params) in prompts.iter().zip(params.into_iter()) {
            let seq_id = self.add_request(PromptInput::Text(prompt), params)?;
            submission_order.push(seq_id);
        }

        let mut finished_by_seq = HashMap::with_capacity(submission_order.len());
        let mut prefill_throughput = 0.0f32;
        let mut decode_throughput = 0.0f32;

        while !self.is_finished() {
            let start = Instant::now();
            let StepOutput {
                finished,
                num_tokens,
                ..
            } = self.step()?;
            let elapsed = start.elapsed().as_secs_f32().max(f32::EPSILON);
            if num_tokens > 0 {
                prefill_throughput = num_tokens as f32 / elapsed;
            } else if num_tokens < 0 {
                decode_throughput = (-num_tokens) as f32 / elapsed;
            }
            info!(
                prefill_tok_s = prefill_throughput,
                decode_tok_s = decode_throughput,
                "engine step throughput"
            );
            for output in finished {
                finished_by_seq.insert(output.seq_id, output.token_ids);
            }
        }

        submission_order
            .into_iter()
            .map(|seq_id| {
                let token_ids = finished_by_seq
                    .remove(&seq_id)
                    .ok_or_else(|| anyhow!("missing output for sequence {}", seq_id))?;
                let text = self.decode_tokens(&token_ids)?;
                Ok(GenerationOutput {
                    seq_id,
                    text,
                    token_ids,
                })
            })
            .collect()
    }

    pub fn generate_stream<'a>(
        &'a mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<EngineStream<'a>> {
        ensure!(!prompts.is_empty(), "prompts must not be empty");
        let params = expand_sampling_params(prompts.len(), sampling_params)?;
        for (prompt, params) in prompts.iter().zip(params.into_iter()) {
            let _ = self.add_request(PromptInput::Text(prompt), params)?;
        }
        Ok(EngineStream {
            engine: self,
            pending_events: VecDeque::new(),
            done: false,
        })
    }

    pub fn cancel_request(&mut self, seq_id: usize) -> bool {
        if !self.active_requests.contains(&seq_id) {
            return false;
        }
        if self.pending_cancellations.iter().any(|id| *id == seq_id) {
            return true;
        }
        self.pending_cancellations.push_back(seq_id);
        true
    }

    pub fn cancel_all_requests(&mut self) -> usize {
        let active_ids = self.active_requests.iter().copied().collect::<Vec<_>>();
        let mut queued = 0usize;
        for seq_id in active_ids {
            if self.pending_cancellations.iter().any(|id| *id == seq_id) {
                continue;
            }
            self.pending_cancellations.push_back(seq_id);
            queued += 1;
        }
        queued
    }

    fn apply_pending_cancellations(&mut self) -> Vec<CancelledOutput> {
        let mut cancelled = Vec::new();
        while let Some(seq_id) = self.pending_cancellations.pop_front() {
            if self.scheduler.cancel(seq_id) {
                self.active_requests.remove(&seq_id);
                cancelled.push(CancelledOutput { seq_id });
            }
        }
        cancelled
    }
}

impl<'a> EngineStream<'a> {
    pub fn cancel_request(&mut self, seq_id: usize) -> bool {
        self.engine.cancel_request(seq_id)
    }

    pub fn cancel_all_requests(&mut self) -> usize {
        self.engine.cancel_all_requests()
    }
}

impl<'a> Iterator for EngineStream<'a> {
    type Item = Result<StreamOutput>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(event) = self.pending_events.pop_front() {
                return Some(Ok(event));
            }
            if self.done {
                return None;
            }

            let step = match self.engine.step() {
                Ok(step) => step,
                Err(err) => {
                    self.done = true;
                    return Some(Err(err));
                }
            };

            for cancelled in step.cancelled {
                self.pending_events
                    .push_back(StreamOutput::Cancelled(cancelled));
            }
            for token in step.tokens {
                let text = match self.engine.decode_tokens(&[token.token_id]) {
                    Ok(text) => text,
                    Err(err) => {
                        self.done = true;
                        return Some(Err(err));
                    }
                };
                self.pending_events.push_back(StreamOutput::Token {
                    seq_id: token.seq_id,
                    token_id: token.token_id,
                    text,
                    finished: token.finished,
                });
            }
            for finished in step.finished {
                let text = match self.engine.decode_tokens(&finished.token_ids) {
                    Ok(text) => text,
                    Err(err) => {
                        self.done = true;
                        return Some(Err(err));
                    }
                };
                self.pending_events
                    .push_back(StreamOutput::Done(GenerationOutput {
                        seq_id: finished.seq_id,
                        text,
                        token_ids: finished.token_ids,
                    }));
            }

            if self.engine.is_finished() && self.pending_events.is_empty() {
                self.done = true;
                return None;
            }
        }
    }
}

fn expand_sampling_params(
    prompts_len: usize,
    sampling_params: &[SamplingParams],
) -> Result<Vec<SamplingParams>> {
    ensure!(
        !sampling_params.is_empty(),
        "sampling_params must not be empty"
    );
    if sampling_params.len() == 1 {
        return Ok(vec![sampling_params[0].clone(); prompts_len]);
    }
    ensure!(
        sampling_params.len() == prompts_len,
        "sampling_params size {} must be 1 or match prompts size {}",
        sampling_params.len(),
        prompts_len
    );
    Ok(sampling_params.to_vec())
}
