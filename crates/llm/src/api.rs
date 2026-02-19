use anyhow::Result;

use crate::core::GenerationOutput;
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
}
