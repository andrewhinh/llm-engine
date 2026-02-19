use axum::response::sse::Event;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoiceChunk>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoiceChunk {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

pub fn token_chunk(
    id: &str,
    created: u64,
    model: &str,
    content: String,
    include_role: bool,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChoiceChunk {
            index: 0,
            delta: Delta {
                role: if include_role {
                    Some("assistant")
                } else {
                    None
                },
                content: Some(content),
            },
            finish_reason: None,
        }],
    }
}

pub fn finish_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChoiceChunk {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
    }
}

pub fn chunk_event(chunk: &ChatCompletionChunk) -> Event {
    Event::default()
        .json_data(chunk)
        .expect("serializing chat chunk must succeed")
}

pub fn done_event() -> Event {
    Event::default().data("[DONE]")
}
