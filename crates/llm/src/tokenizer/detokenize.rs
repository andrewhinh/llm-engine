use std::collections::HashMap;

use anyhow::Result;
use tokenizers::Tokenizer;

use crate::ipc::{DetokenizeRequest, UserReply};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeStatus {
    pub decoded_ids: Vec<u32>,
    pub decoded_text: String,
    pub read_offset: usize,
    pub surrogate_offset: usize,
    pub sent_offset: usize,
}

impl DecodeStatus {
    fn new() -> Self {
        Self {
            decoded_ids: Vec::new(),
            decoded_text: String::new(),
            read_offset: 0,
            surrogate_offset: 0,
            sent_offset: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetokenizeManager {
    tokenizer: Tokenizer,
    eos_token_id: u32,
    decode_map: HashMap<u64, DecodeStatus>,
}

impl DetokenizeManager {
    pub fn new(tokenizer: Tokenizer, eos_token_id: u32) -> Self {
        Self {
            tokenizer,
            eos_token_id,
            decode_map: HashMap::new(),
        }
    }

    pub fn abort(&mut self, request_id: u64) {
        self.decode_map.remove(&request_id);
    }

    pub fn detokenize_batch(&mut self, requests: &[DetokenizeRequest]) -> Result<Vec<UserReply>> {
        let mut outputs = Vec::with_capacity(requests.len());
        for request in requests {
            let status = self
                .decode_map
                .entry(request.request_id)
                .or_insert_with(DecodeStatus::new);
            if !(request.finished && request.next_token == self.eos_token_id) {
                status.decoded_ids.push(request.next_token);
            }

            let read_ids = &status.decoded_ids[status.surrogate_offset..];
            let surrogate_ids = &status.decoded_ids[status.surrogate_offset..status.read_offset];
            let read_text = decode_ids(&self.tokenizer, read_ids)?;
            let surrogate_text = decode_ids(&self.tokenizer, surrogate_ids)?;
            let mut new_text = read_text
                .strip_prefix(&surrogate_text)
                .unwrap_or(read_text.as_str());

            let output_text = if !new_text.is_empty() && !new_text.ends_with('\u{FFFD}') {
                status.decoded_text.push_str(new_text);
                status.surrogate_offset = status.read_offset;
                status.read_offset = status.decoded_ids.len();
                status.decoded_text.clone()
            } else {
                new_text = find_printable_text(new_text);
                let mut combined = status.decoded_text.clone();
                combined.push_str(new_text);
                combined
            };

            let incremental_output = output_text
                .get(status.sent_offset..)
                .unwrap_or_default()
                .to_string();
            status.sent_offset = output_text.len();
            outputs.push(UserReply {
                request_id: request.request_id,
                incremental_output,
                finished: request.finished,
            });
            if request.finished {
                self.decode_map.remove(&request.request_id);
            }
        }
        Ok(outputs)
    }
}

fn decode_ids(tokenizer: &Tokenizer, token_ids: &[u32]) -> Result<String> {
    if token_ids.is_empty() {
        return Ok(String::new());
    }
    let text = tokenizer
        .decode(token_ids, false)
        .map_err(anyhow::Error::msg)?;
    Ok(text)
}

fn find_printable_text(text: &str) -> &str {
    if text.ends_with('\n') {
        return text;
    }
    if text
        .chars()
        .last()
        .map(|ch| is_cjk(ch as u32))
        .unwrap_or(false)
    {
        return text;
    }
    if penultimate_char(text)
        .map(|ch| is_cjk(ch as u32))
        .unwrap_or(false)
    {
        return truncate_last_char(text);
    }
    match text.rfind(' ') {
        Some(index) => &text[..index + 1],
        None => "",
    }
}

fn penultimate_char(text: &str) -> Option<char> {
    let mut iter = text.chars().rev();
    let _ = iter.next()?;
    iter.next()
}

fn truncate_last_char(text: &str) -> &str {
    match text.char_indices().next_back() {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

fn is_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}
