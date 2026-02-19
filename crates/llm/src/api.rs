use anyhow::Result;

use crate::core::{EngineStream, GenerationOutput};
use crate::{Engine, EngineConfig, SamplingParams};

#[derive(Debug)]
pub struct LLM {
    engine: Engine,
}

impl LLM {
    pub fn new(config: EngineConfig) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(config)?,
        })
    }

    pub fn generate(
        &mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<Vec<GenerationOutput>> {
        self.engine.generate_sync(prompts, sampling_params)
    }

    pub fn generate_stream<'a>(
        &'a mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<EngineStream<'a>> {
        self.engine.generate_stream(prompts, sampling_params)
    }

    pub fn count_prompt_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(self.engine.encode_prompt(prompt)?.len())
    }
}
