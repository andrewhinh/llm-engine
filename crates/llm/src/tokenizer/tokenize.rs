use anyhow::{Result, ensure};
use tokenizers::Tokenizer;

use crate::ipc::{ChatRole, PromptPayload, TokenizeRequest, UserTokenizedRequest};
use crate::utils::tokenizer::encode_prompt;

#[derive(Debug, Clone)]
pub struct TokenizeManager {
    tokenizer: Tokenizer,
}

impl TokenizeManager {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn tokenize_batch(
        &self,
        requests: &[TokenizeRequest],
    ) -> Result<Vec<UserTokenizedRequest>> {
        requests
            .iter()
            .map(|request| {
                let prompt = render_prompt(&request.prompt)?;
                let input_ids = encode_prompt(&self.tokenizer, &prompt)?;
                Ok(UserTokenizedRequest {
                    request_id: request.request_id,
                    input_ids,
                    sampling_params: request.sampling_params.clone(),
                })
            })
            .collect()
    }
}

pub fn render_prompt(payload: &PromptPayload) -> Result<String> {
    let prompt = match payload {
        PromptPayload::Text(text) => text.trim().to_string(),
        PromptPayload::Messages(messages) => messages
            .iter()
            .map(|message| format!("{}: {}", role_label(&message.role), message.content.trim()))
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string(),
    };
    ensure!(!prompt.is_empty(), "prompt must not be empty");
    Ok(prompt)
}

fn role_label(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
        ChatRole::Tool => "tool",
    }
}
