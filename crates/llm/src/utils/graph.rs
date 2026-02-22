use std::collections::BTreeSet;

#[cfg(feature = "cuda-graph")]
use anyhow::{Result, anyhow};

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
        self.batches.iter().copied().find(|&bs| bs >= decode_batch)
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecodeGraphCaptures {
    captured: BTreeSet<usize>,
}

impl DecodeGraphCaptures {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_captured(&mut self, capture_batch: usize) {
        self.captured.insert(capture_batch);
    }

    pub fn is_captured(&self, capture_batch: usize) -> bool {
        self.captured.contains(&capture_batch)
    }

    pub fn captured_batches(&self) -> Vec<usize> {
        self.captured.iter().copied().collect()
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

#[cfg(feature = "cuda-graph")]
#[derive(Debug)]
pub struct DecodeCudaGraph {
    cu_graph: candle_core::cuda_backend::cudarc::driver::sys::CUgraph,
    cu_graph_exec: candle_core::cuda_backend::cudarc::driver::sys::CUgraphExec,
    stream: candle_core::cuda_backend::cudarc::driver::sys::CUstream,
}

#[cfg(feature = "cuda-graph")]
impl DecodeCudaGraph {
    pub fn begin_capture(device: &candle_core::Device) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys::{CUstreamCaptureMode, lib};

        let cuda = device
            .as_cuda_device()
            .ok_or_else(|| anyhow!("decode graph capture requires cuda device"))?;
        unsafe {
            lib()
                .cuStreamBeginCapture_v2(
                    cuda.cu_stream().clone(),
                    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .result()
                .map_err(|e| anyhow!("cuStreamBeginCapture_v2 failed: {e:?}"))?;
        }
        Ok(())
    }

    pub fn end_capture(device: &candle_core::Device) -> Result<Self> {
        use candle_core::cuda_backend::cudarc::driver::sys::{CUgraphInstantiate_flags, lib};
        use std::mem::MaybeUninit;

        let cuda = device
            .as_cuda_device()
            .ok_or_else(|| anyhow!("decode graph end capture requires cuda device"))?;
        let stream = cuda.cu_stream().clone();
        let mut graph = MaybeUninit::uninit();
        let cu_graph = unsafe {
            lib()
                .cuStreamEndCapture(stream, graph.as_mut_ptr())
                .result()
                .map_err(|e| anyhow!("cuStreamEndCapture failed: {e:?}"))?;
            graph.assume_init()
        };
        let mut graph_exec = MaybeUninit::uninit();
        let cu_graph_exec = unsafe {
            lib()
                .cuGraphInstantiateWithFlags(
                    graph_exec.as_mut_ptr(),
                    cu_graph,
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u32
                        as u64,
                )
                .result()
                .map_err(|e| anyhow!("cuGraphInstantiateWithFlags failed: {e:?}"))?;
            graph_exec.assume_init()
        };
        Ok(Self {
            cu_graph,
            cu_graph_exec,
            stream,
        })
    }

    pub fn launch(&self) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::sys::lib;
        unsafe {
            lib()
                .cuGraphLaunch(self.cu_graph_exec, self.stream)
                .result()
                .map_err(|e| anyhow!("cuGraphLaunch failed: {e:?}"))?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda-graph")]
impl Drop for DecodeCudaGraph {
    fn drop(&mut self) {
        use candle_core::cuda_backend::cudarc::driver::sys::lib;
        unsafe {
            let _ = lib().cuGraphExecDestroy(self.cu_graph_exec);
            let _ = lib().cuGraphDestroy(self.cu_graph);
        }
    }
}
