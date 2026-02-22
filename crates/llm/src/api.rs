use anyhow::{Result, ensure};

use crate::core::{EngineStream, GenerationOutput};
use crate::{Engine, EngineConfig, SamplingParams};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug)]
pub struct LLM {
    engine: Engine,
}

impl LLM {
    pub fn from_model(model: impl Into<String>) -> Result<Self> {
        let config = EngineConfig {
            model: model.into(),
            ..EngineConfig::default()
        };
        Self::new(config)
    }

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

    pub fn generate_one(
        &mut self,
        prompt: impl Into<String>,
        sampling: SamplingParams,
    ) -> Result<GenerationOutput> {
        let prompts = vec![prompt.into()];
        let params = vec![sampling];
        let mut outputs = self.generate(&prompts, &params)?;
        ensure!(!outputs.is_empty(), "engine returned no outputs");
        Ok(outputs.remove(0))
    }

    pub fn generate_stream<'a>(
        &'a mut self,
        prompts: &[String],
        sampling_params: &[SamplingParams],
    ) -> Result<EngineStream<'a>> {
        self.engine.generate_stream(prompts, sampling_params)
    }

    pub fn generate_one_stream<'a>(
        &'a mut self,
        prompt: impl Into<String>,
        sampling: SamplingParams,
    ) -> Result<EngineStream<'a>> {
        let prompts = vec![prompt.into()];
        let params = vec![sampling];
        self.generate_stream(&prompts, &params)
    }

    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        sampling: SamplingParams,
    ) -> Result<GenerationOutput> {
        let prompt = chat_messages_to_prompt(messages);
        self.generate_one(prompt, sampling)
    }

    pub fn chat_stream<'a>(
        &'a mut self,
        messages: &[ChatMessage],
        sampling: SamplingParams,
    ) -> Result<EngineStream<'a>> {
        let prompt = chat_messages_to_prompt(messages);
        self.generate_one_stream(prompt, sampling)
    }

    pub fn count_prompt_tokens(&self, prompt: &str) -> Result<usize> {
        Ok(self.engine.encode_prompt(prompt)?.len())
    }
}

fn chat_messages_to_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}
