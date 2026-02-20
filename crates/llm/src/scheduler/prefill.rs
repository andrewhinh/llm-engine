#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPolicy {
    pub max_prefill_tokens: usize,
    pub max_num_batched_tokens: usize,
}

impl PrefillPolicy {
    pub fn step_budget_tokens(&self) -> usize {
        self.max_prefill_tokens.min(self.max_num_batched_tokens)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillBudget {
    remaining_tokens: usize,
}

impl PrefillBudget {
    pub fn new(policy: PrefillPolicy) -> Self {
        Self {
            remaining_tokens: policy.step_budget_tokens(),
        }
    }

    pub fn remaining_tokens(&self) -> usize {
        self.remaining_tokens
    }

    pub fn is_exhausted(&self) -> bool {
        self.remaining_tokens == 0
    }

    pub fn plan_chunk(
        &self,
        remaining_prefill_tokens: usize,
        max_batch_tokens_left: usize,
    ) -> usize {
        remaining_prefill_tokens
            .min(self.remaining_tokens)
            .min(max_batch_tokens_left)
    }

    pub fn consume(&mut self, consumed_tokens: usize) {
        self.remaining_tokens = self.remaining_tokens.saturating_sub(consumed_tokens);
    }
}
