use std::env;
use std::ops::Range;
use std::sync::{OnceLock, RwLock};

use anyhow::{Result, ensure};
use candle_core::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpInfo {
    pub rank: usize,
    pub world_size: usize,
}

impl TpInfo {
    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        ensure!(
            world_size > 0,
            "tensor parallel world_size must be positive"
        );
        ensure!(
            rank < world_size,
            "tensor parallel rank must be in [0, world_size)"
        );
        Ok(Self { rank, world_size })
    }

    pub fn is_primary(&self) -> bool {
        self.rank == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpBackend {
    None,
    Gloo,
    Nccl,
}

impl TpBackend {
    fn from_env(world_size: usize) -> Result<Self> {
        if world_size == 1 {
            return Ok(Self::None);
        }
        let value = env::var("LLM_TP_BACKEND")
            .unwrap_or_else(|_| "gloo".to_string())
            .to_ascii_lowercase();
        match value.as_str() {
            "gloo" => Ok(Self::Gloo),
            "nccl" => Ok(Self::Nccl),
            _ => anyhow::bail!("unsupported LLM_TP_BACKEND '{}'", value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Comm {
    info: TpInfo,
    backend: TpBackend,
}

impl Comm {
    pub fn from_parts(rank: usize, world_size: usize, backend: TpBackend) -> Result<Self> {
        let info = TpInfo::new(rank, world_size)?;
        set_tp_info(info.rank, info.world_size)?;
        Ok(Self { info, backend })
    }

    pub fn bootstrap(world_size: usize) -> Result<Self> {
        let rank = env::var("LLM_TP_RANK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let info = TpInfo::new(rank, world_size)?;
        let backend = TpBackend::from_env(world_size)?;
        set_tp_info(info.rank, info.world_size)?;
        Ok(Self { info, backend })
    }

    pub fn info(&self) -> TpInfo {
        self.info
    }

    pub fn rank(&self) -> usize {
        self.info.rank
    }

    pub fn world_size(&self) -> usize {
        self.info.world_size
    }

    pub fn backend(&self) -> TpBackend {
        self.backend
    }

    pub fn all_reduce_sum(&self, x: &Tensor) -> Result<Tensor> {
        if self.info.world_size == 1 {
            return Ok(x.clone());
        }
        Ok(x.clone())
    }
}

fn tp_info_cell() -> &'static RwLock<TpInfo> {
    static TP_INFO: OnceLock<RwLock<TpInfo>> = OnceLock::new();
    TP_INFO.get_or_init(|| RwLock::new(TpInfo::new(0, 1).expect("default tp info")))
}

pub fn set_tp_info(rank: usize, world_size: usize) -> Result<()> {
    let info = TpInfo::new(rank, world_size)?;
    let mut guard = tp_info_cell()
        .write()
        .map_err(|_| anyhow::anyhow!("tp info lock poisoned"))?;
    *guard = info;
    Ok(())
}

pub fn get_tp_info() -> TpInfo {
    *tp_info_cell()
        .read()
        .expect("tp info lock poisoned while reading")
}

pub fn try_get_tp_info() -> Option<TpInfo> {
    tp_info_cell().read().ok().map(|guard| *guard)
}

pub fn shard_range(total: usize, rank: usize, world_size: usize) -> Result<Range<usize>> {
    ensure!(world_size > 0, "world_size must be positive");
    ensure!(rank < world_size, "rank must be < world_size");
    ensure!(
        total.is_multiple_of(world_size),
        "dimension {} must be divisible by world_size {}",
        total,
        world_size
    );
    let size = total / world_size;
    let start = rank * size;
    let end = start + size;
    Ok(start..end)
}

pub fn shard_size(total: usize, world_size: usize) -> Result<usize> {
    ensure!(world_size > 0, "world_size must be positive");
    ensure!(
        total.is_multiple_of(world_size),
        "dimension {} must be divisible by world_size {}",
        total,
        world_size
    );
    Ok(total / world_size)
}

pub fn kv_head_shard(total_num_kv_heads: usize, rank: usize, world_size: usize) -> Result<usize> {
    ensure!(total_num_kv_heads > 0, "num_kv_heads must be positive");
    ensure!(world_size > 0, "world_size must be positive");
    ensure!(rank < world_size, "rank must be < world_size");
    if total_num_kv_heads >= world_size {
        ensure!(
            total_num_kv_heads.is_multiple_of(world_size),
            "num_kv_heads must be divisible by world_size when partitioned"
        );
        Ok(total_num_kv_heads / world_size)
    } else {
        ensure!(
            world_size.is_multiple_of(total_num_kv_heads),
            "world_size must be divisible by num_kv_heads when replicated"
        );
        let ranks_per_kv_head = world_size / total_num_kv_heads;
        let kv_rank = rank / ranks_per_kv_head;
        let _ = kv_rank;
        Ok(1)
    }
}
