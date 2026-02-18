use crate::utils::SamplingParams;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: usize,
    pub status: SequenceStatus,
    pub token_ids: Vec<u32>,
    pub last_token: u32,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub block_table: Vec<u32>,
    pub block_size: usize,
    pub sampling_params: SamplingParams,
}

impl Sequence {
    pub fn new(token_ids: Vec<u32>, block_size: usize, sampling_params: SamplingParams) -> Self {
        assert!(!token_ids.is_empty(), "token_ids must not be empty");
        let num_tokens = token_ids.len();
        let last_token = *token_ids.last().unwrap();
        Self {
            id: 0,
            status: SequenceStatus::Waiting,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            block_size,
            sampling_params,
        }
    }

    pub fn len(&self) -> usize {
        self.num_tokens
    }

    pub fn is_empty(&self) -> bool {
        self.num_tokens == 0
    }

    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.token_ids[..self.num_prompt_tokens]
    }

    pub fn completion_token_ids(&self) -> &[u32] {
        &self.token_ids[self.num_prompt_tokens..]
    }

    pub fn num_cached_blocks(&self) -> usize {
        self.num_cached_tokens / self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_tokens.div_ceil(self.block_size)
    }

    pub fn last_block_num_tokens(&self) -> usize {
        self.num_tokens - (self.num_blocks() - 1) * self.block_size
    }

    pub fn block(&self, index: usize) -> &[u32] {
        assert!(index < self.num_blocks(), "block index out of range");
        let start = index * self.block_size;
        let end = usize::min(start + self.block_size, self.num_tokens);
        &self.token_ids[start..end]
    }

    pub fn append_token(&mut self, token_id: u32) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }
}
