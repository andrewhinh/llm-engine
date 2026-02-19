use serde::{Deserialize, Serialize};

use crate::utils::SamplingParams;

pub type RequestId = u64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PromptPayload {
    Text(String),
    Messages(Vec<ChatMessage>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizeRequest {
    pub request_id: RequestId,
    pub prompt: PromptPayload,
    pub sampling_params: SamplingParams,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AbortRequest {
    pub request_id: RequestId,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum FrontendToTokenizerMsg {
    Tokenize(TokenizeRequest),
    Abort(AbortRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FrontendToTokenizerBatch {
    pub data: Vec<FrontendToTokenizerMsg>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UserTokenizedRequest {
    pub request_id: RequestId,
    pub input_ids: Vec<u32>,
    pub sampling_params: SamplingParams,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum TokenizerToSchedulerMsg {
    User(UserTokenizedRequest),
    Exit,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerToSchedulerBatch {
    pub data: Vec<TokenizerToSchedulerMsg>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DetokenizeRequest {
    pub request_id: RequestId,
    pub next_token: u32,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SchedulerToTokenizerMsg {
    Detokenize(DetokenizeRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchedulerToTokenizerBatch {
    pub data: Vec<SchedulerToTokenizerMsg>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UserReply {
    pub request_id: RequestId,
    pub incremental_output: String,
    pub finished: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum TokenizerToFrontendMsg {
    Reply(UserReply),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerToFrontendBatch {
    pub data: Vec<TokenizerToFrontendMsg>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SchedulerRankControlMsg {
    Barrier,
    Idle,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SchedulerRankMsg {
    Forward(TokenizerToSchedulerMsg),
    ForwardBatch(TokenizerToSchedulerBatch),
    Control(SchedulerRankControlMsg),
}
