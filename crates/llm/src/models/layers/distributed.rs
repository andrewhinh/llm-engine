use std::env;
use std::fs;
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
#[cfg(feature = "nccl")]
use candle_core::CustomOp1;
use candle_core::{Device, Tensor};

#[cfg(feature = "nccl")]
use candle_core::cuda_backend::CudaDevice;
#[cfg(feature = "nccl")]
use candle_core::cuda_backend::cudarc::nccl::safe::{Comm as NcclNativeComm, Id, ReduceOp};

pub const LLM_TP_NAMESPACE_ENV: &str = "LLM_TP_NAMESPACE";
pub const LLM_TP_RANK_ENV: &str = "LLM_TP_RANK";
pub const LLM_TP_BACKEND_ENV: &str = "LLM_TP_BACKEND";
pub const LLM_TP_DEVICE_ID_ENV: &str = "LLM_TP_DEVICE_ID";
pub const LLM_TP_NCCL_ID_ENV: &str = "LLM_TP_NCCL_ID";

const GLOO_IO_TIMEOUT: Duration = Duration::from_secs(30);
const GLOO_IO_POLL: Duration = Duration::from_millis(2);

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
        let value = env::var(LLM_TP_BACKEND_ENV)
            .unwrap_or_else(|_| "gloo".to_string())
            .to_ascii_lowercase();
        match value.as_str() {
            "gloo" => Ok(Self::Gloo),
            "nccl" => Ok(Self::Nccl),
            _ => anyhow::bail!("unsupported {} '{}'", LLM_TP_BACKEND_ENV, value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Comm {
    info: TpInfo,
    backend: TpBackend,
    state: CommState,
}

#[derive(Debug, Clone)]
enum CommState {
    None,
    Gloo(Arc<GlooState>),
    #[cfg(feature = "nccl")]
    Nccl(Arc<NcclState>),
}

#[derive(Debug)]
struct GlooState {
    namespace: String,
    op_seq: AtomicU64,
    op_lock: Mutex<()>,
}

#[cfg(feature = "nccl")]
#[derive(Debug)]
struct NcclState {
    comm: Arc<NcclNativeComm>,
}

impl Comm {
    pub fn from_parts(rank: usize, world_size: usize, backend: TpBackend) -> Result<Self> {
        let info = TpInfo::new(rank, world_size)?;
        set_tp_info(info.rank, info.world_size)?;
        let state = build_comm_state(info, backend)?;
        Ok(Self {
            info,
            backend,
            state,
        })
    }

    pub fn bootstrap(world_size: usize) -> Result<Self> {
        let rank = env::var(LLM_TP_RANK_ENV)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let info = TpInfo::new(rank, world_size)?;
        let backend = TpBackend::from_env(world_size)?;
        set_tp_info(info.rank, info.world_size)?;
        let state = build_comm_state(info, backend)?;
        Ok(Self {
            info,
            backend,
            state,
        })
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
        match &self.state {
            CommState::None => Ok(x.clone()),
            CommState::Gloo(state) => self.all_reduce_sum_gloo(state, x),
            #[cfg(feature = "nccl")]
            CommState::Nccl(state) => self.all_reduce_sum_nccl(state, x),
        }
    }

    fn all_reduce_sum_gloo(&self, state: &GlooState, x: &Tensor) -> Result<Tensor> {
        let _guard = state
            .op_lock
            .lock()
            .map_err(|_| anyhow!("tp gloo operation lock poisoned"))?;
        let seq = state.op_seq.fetch_add(1, Ordering::Relaxed);
        let shape = x.dims().to_vec();
        let original_dtype = x.dtype();
        let cpu = x.to_device(&Device::Cpu)?;
        let f32 = if cpu.dtype() == candle_core::DType::F32 {
            cpu
        } else {
            cpu.to_dtype(candle_core::DType::F32)?
        };
        let flat = f32.flatten_all()?;
        let local = flat.to_vec1::<f32>()?;

        let step_dir = gloo_step_dir(&state.namespace, seq);
        fs::create_dir_all(&step_dir).with_context(|| {
            format!(
                "failed to create gloo all-reduce step dir {}",
                step_dir.display()
            )
        })?;
        write_tp_tensor_file(&gloo_rank_file(&step_dir, self.info.rank), &shape, &local)?;

        let reduced = if self.info.rank == 0 {
            let mut sum = local;
            for rank in 1..self.info.world_size {
                let path = gloo_rank_file(&step_dir, rank);
                wait_for_path(&path, GLOO_IO_TIMEOUT)?;
                let (dims, values) = read_tp_tensor_file(&path)?;
                ensure!(dims == shape, "gloo all-reduce shape mismatch");
                sum_in_place(&mut sum, &values)?;
            }
            write_tp_tensor_file(&gloo_output_file(&step_dir), &shape, &sum)?;
            for rank in 1..self.info.world_size {
                let ack_path = gloo_ack_file(&step_dir, rank);
                wait_for_path(&ack_path, GLOO_IO_TIMEOUT)?;
            }
            let _ = fs::remove_dir_all(&step_dir);
            sum
        } else {
            let out_path = gloo_output_file(&step_dir);
            wait_for_path(&out_path, GLOO_IO_TIMEOUT)?;
            let (dims, values) = read_tp_tensor_file(&out_path)?;
            ensure!(dims == shape, "gloo all-reduce output shape mismatch");
            fs::write(gloo_ack_file(&step_dir, self.info.rank), b"ok")
                .context("failed to write gloo all-reduce ack file")?;
            values
        };

        let reduced = Tensor::from_vec(reduced, shape, &Device::Cpu)?;
        let reduced = if original_dtype == candle_core::DType::F32 {
            reduced
        } else {
            reduced.to_dtype(original_dtype)?
        };
        Ok(reduced.to_device(x.device())?)
    }

    #[cfg(feature = "nccl")]
    fn all_reduce_sum_nccl(&self, state: &NcclState, x: &Tensor) -> Result<Tensor> {
        ensure!(
            matches!(x.device(), Device::Cuda(_)),
            "nccl all-reduce requires CUDA tensors"
        );
        if x.dtype() != candle_core::DType::F32 {
            let casted = x.to_dtype(candle_core::DType::F32)?;
            let reduced = apply_nccl_all_reduce(&casted, state.comm.clone())?;
            return reduced.to_dtype(x.dtype()).map_err(|e| anyhow!("{e}"));
        }
        apply_nccl_all_reduce(x, state.comm.clone())
    }
}

fn build_comm_state(info: TpInfo, backend: TpBackend) -> Result<CommState> {
    if info.world_size == 1 {
        return Ok(CommState::None);
    }
    match backend {
        TpBackend::None => Ok(CommState::None),
        TpBackend::Gloo => {
            let namespace = env::var(LLM_TP_NAMESPACE_ENV)
                .unwrap_or_else(|_| format!("llm-engine-tp-{}", std::process::id()));
            Ok(CommState::Gloo(Arc::new(GlooState {
                namespace,
                op_seq: AtomicU64::new(0),
                op_lock: Mutex::new(()),
            })))
        }
        TpBackend::Nccl => {
            #[cfg(feature = "nccl")]
            {
                let nccl_hex = env::var(LLM_TP_NCCL_ID_ENV)
                    .with_context(|| format!("missing {} for nccl backend", LLM_TP_NCCL_ID_ENV))?;
                let id = nccl_id_from_hex(&nccl_hex)?;
                let dev_id = env::var(LLM_TP_DEVICE_ID_ENV)
                    .ok()
                    .and_then(|raw| raw.parse::<usize>().ok())
                    .unwrap_or(info.rank);
                let device = CudaDevice::new(dev_id)
                    .map_err(|e| anyhow!("failed to init cuda device {dev_id} for nccl: {e}"))?;
                let native =
                    NcclNativeComm::from_rank(device.cuda_device(), info.rank, info.world_size, id)
                        .map_err(|e| {
                            anyhow!("failed nccl comm init for rank {}: {e:?}", info.rank)
                        })?;
                Ok(CommState::Nccl(Arc::new(NcclState {
                    comm: Arc::new(native),
                })))
            }
            #[cfg(not(feature = "nccl"))]
            {
                let _ = info;
                anyhow::bail!("nccl backend requested but crate built without `nccl` feature");
            }
        }
    }
}

fn gloo_step_dir(namespace: &str, seq: u64) -> PathBuf {
    PathBuf::from("/tmp").join(format!("{namespace}-tp-step-{seq}"))
}

fn gloo_rank_file(step_dir: &Path, rank: usize) -> PathBuf {
    step_dir.join(format!("rank-{rank}.bin"))
}

fn gloo_output_file(step_dir: &Path) -> PathBuf {
    step_dir.join("out.bin")
}

fn gloo_ack_file(step_dir: &Path, rank: usize) -> PathBuf {
    step_dir.join(format!("ack-{rank}.ok"))
}

fn wait_for_path(path: &Path, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    loop {
        if path.exists() {
            return Ok(());
        }
        if start.elapsed() > timeout {
            anyhow::bail!("timed out waiting for path {}", path.display());
        }
        std::thread::sleep(GLOO_IO_POLL);
    }
}

fn write_tp_tensor_file(path: &Path, dims: &[usize], data: &[f32]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let mut bytes = Vec::with_capacity((dims.len() + data.len() + 2) * 8);
    bytes.extend_from_slice(&(dims.len() as u64).to_le_bytes());
    for &d in dims {
        bytes.extend_from_slice(&(d as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut file = fs::File::create(&tmp)
        .with_context(|| format!("failed to create tp tensor temp file {}", tmp.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("failed to write tp tensor file {}", tmp.display()))?;
    file.flush()
        .with_context(|| format!("failed to flush tp tensor file {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "failed to atomically rename tp tensor file {} -> {}",
            tmp.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn read_tp_tensor_file(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let mut bytes = Vec::new();
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open tp tensor file {}", path.display()))?;
    file.read_to_end(&mut bytes)
        .with_context(|| format!("failed to read tp tensor file {}", path.display()))?;
    let mut cursor = 0usize;
    let dims_len = read_u64(&bytes, &mut cursor)? as usize;
    let mut dims = Vec::with_capacity(dims_len);
    for _ in 0..dims_len {
        dims.push(read_u64(&bytes, &mut cursor)? as usize);
    }
    let data_len = read_u64(&bytes, &mut cursor)? as usize;
    let mut data = Vec::with_capacity(data_len);
    for _ in 0..data_len {
        let raw = read_array_4(&bytes, &mut cursor)?;
        data.push(f32::from_le_bytes(raw));
    }
    Ok((dims, data))
}

fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let end = cursor.saturating_add(8);
    ensure!(end <= bytes.len(), "corrupt tp tensor payload");
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u64::from_le_bytes(raw))
}

fn read_array_4(bytes: &[u8], cursor: &mut usize) -> Result<[u8; 4]> {
    let end = cursor.saturating_add(4);
    ensure!(end <= bytes.len(), "corrupt tp tensor payload");
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(raw)
}

fn sum_in_place(sum: &mut [f32], values: &[f32]) -> Result<()> {
    ensure!(
        sum.len() == values.len(),
        "gloo all-reduce payload length mismatch"
    );
    for (dst, src) in sum.iter_mut().zip(values.iter()) {
        *dst += *src;
    }
    Ok(())
}

#[cfg(feature = "nccl")]
pub fn generate_nccl_id_hex() -> Result<String> {
    let id = Id::new().map_err(|e| anyhow!("failed to generate nccl id: {e:?}"))?;
    let bytes = id.as_bytes();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    Ok(out)
}

#[cfg(feature = "nccl")]
fn nccl_id_from_hex(raw: &str) -> Result<Id> {
    ensure!(raw.len() == 256, "nccl id hex must be exactly 256 chars");
    let mut bytes = [0u8; 128];
    for (idx, chunk) in raw.as_bytes().chunks_exact(2).enumerate() {
        let s = std::str::from_utf8(chunk).context("invalid utf8 in nccl id")?;
        bytes[idx] = u8::from_str_radix(s, 16)
            .with_context(|| format!("invalid hex at nccl id byte index {idx}: {s}"))?;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut signed = [0i8; 128];
        for (dst, src) in signed.iter_mut().zip(bytes.iter()) {
            *dst = *src as i8;
        }
        Ok(Id::uninit(signed))
    }
    #[cfg(target_arch = "aarch64")]
    {
        Ok(Id::uninit(bytes))
    }
}

#[cfg(feature = "nccl")]
#[derive(Debug, Clone)]
struct NcclAllReduceOp {
    comm: Arc<NcclNativeComm>,
}

#[cfg(feature = "nccl")]
unsafe impl Send for NcclAllReduceOp {}
#[cfg(feature = "nccl")]
unsafe impl Sync for NcclAllReduceOp {}

#[cfg(feature = "nccl")]
impl CustomOp1 for NcclAllReduceOp {
    fn name(&self) -> &'static str {
        "nccl-all-reduce"
    }

    fn cpu_fwd(
        &self,
        _s: &candle_core::CpuStorage,
        _l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("nccl all-reduce op is CUDA only")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice;

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let start = l.start_offset();
        let src_full = s.as_cuda_slice::<f32>()?;
        let end = start.saturating_add(elem_count);
        if end > src_full.len() {
            candle_core::bail!(
                "nccl all-reduce slice out of bounds: start={}, elem_count={}, len={}",
                start,
                elem_count,
                src_full.len()
            );
        }
        let src = src_full.slice(start..end);
        let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
        self.comm
            .all_reduce(&src, &mut dst, &ReduceOp::Sum)
            .map_err(candle_core::Error::debug)?;
        let out = candle_core::CudaStorage::wrap_cuda_slice(dst, dev)?;
        Ok((out, l.shape().clone()))
    }
}

#[cfg(feature = "nccl")]
fn apply_nccl_all_reduce(x: &Tensor, comm: Arc<NcclNativeComm>) -> Result<Tensor> {
    let op = NcclAllReduceOp { comm };
    x.apply_op1_no_bwd(&op).map_err(|e| anyhow!("{e}"))
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
