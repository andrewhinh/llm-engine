#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeExecutionPlan {
    Eager,
    GraphReplay { capture_batch: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeGraphRuntime {
    enabled: bool,
    batches: Vec<usize>,
}

impl DecodeGraphRuntime {
    pub fn new(max_num_seqs: usize, enforce_eager: bool, cuda_supported: bool) -> Self {
        let batches = planned_decode_graph_batches(max_num_seqs);
        let enabled = cuda_supported && !enforce_eager && !batches.is_empty();
        Self { enabled, batches }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn batches(&self) -> &[usize] {
        &self.batches
    }

    pub fn select_capture_batch(&self, decode_batch: usize) -> Option<usize> {
        if !self.enabled || decode_batch == 0 {
            return None;
        }
        self.batches.iter().copied().find(|bs| *bs >= decode_batch)
    }

    pub fn plan(&self, is_prefill: bool, batch: usize) -> DecodeExecutionPlan {
        if is_prefill {
            return DecodeExecutionPlan::Eager;
        }
        match self.select_capture_batch(batch) {
            Some(capture_batch) => DecodeExecutionPlan::GraphReplay { capture_batch },
            None => DecodeExecutionPlan::Eager,
        }
    }
}

pub fn planned_decode_graph_batches(max_num_seqs: usize) -> Vec<usize> {
    if max_num_seqs == 0 {
        return Vec::new();
    }

    let capped = max_num_seqs.min(512);
    let mut graph_bs = Vec::new();
    for bs in [1usize, 2, 4, 8] {
        if bs <= capped {
            graph_bs.push(bs);
        }
    }
    if capped >= 16 {
        graph_bs.extend((16..=capped).step_by(16));
    }
    graph_bs
}
