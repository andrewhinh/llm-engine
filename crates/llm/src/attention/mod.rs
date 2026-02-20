pub mod base;
pub mod eager;
pub mod fa;
pub mod fi;

use std::sync::Arc;

use anyhow::Result;
use candle_core::Device;
use tracing::warn;

use crate::utils::{AttentionBackendKind, AttentionBackendSelection};

pub use base::{AttentionBackend, KvCache, KvSlot};
pub use eager::EagerAttentionBackend;
pub use fa::FlashAttentionBackend;
pub use fi::FlashInferBackend;

#[derive(Debug, Clone)]
pub struct AttentionBackends {
    pub prefill: Arc<dyn AttentionBackend>,
    pub decode: Arc<dyn AttentionBackend>,
}

impl AttentionBackends {
    pub fn new(
        selection: AttentionBackendSelection,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            prefill: build_backend(selection.prefill, num_heads, num_kv_heads, head_dim, device)?,
            decode: build_backend(selection.decode, num_heads, num_kv_heads, head_dim, device)?,
        })
    }
}

pub fn build_backend(
    kind: AttentionBackendKind,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Result<Arc<dyn AttentionBackend>> {
    #[cfg(not(any(feature = "flash-attn", feature = "flashinfer")))]
    let _ = device;

    match kind {
        AttentionBackendKind::Eager => Ok(Arc::new(EagerAttentionBackend::new(
            num_heads,
            num_kv_heads,
            head_dim,
        )?)),
        AttentionBackendKind::Fa => {
            #[cfg(feature = "flash-attn")]
            {
                if !matches!(device, Device::Cuda(_)) {
                    warn!("flash-attn selected on non-cuda device, falling back to eager backend");
                    return Ok(Arc::new(EagerAttentionBackend::new(
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )?));
                }
                Ok(Arc::new(FlashAttentionBackend::new(
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    device,
                )?))
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                warn!("flash-attn selected but feature is disabled, falling back to eager backend");
                Ok(Arc::new(EagerAttentionBackend::new(
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?))
            }
        }
        AttentionBackendKind::Fi => {
            #[cfg(feature = "flashinfer")]
            {
                if !matches!(device, Device::Cuda(_)) {
                    warn!("flashinfer selected on non-cuda device, falling back to eager backend");
                    return Ok(Arc::new(EagerAttentionBackend::new(
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    )?));
                }
                Ok(Arc::new(FlashInferBackend::new(
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    device,
                )?))
            }
            #[cfg(not(feature = "flashinfer"))]
            {
                warn!("flashinfer selected but feature is disabled, falling back to eager backend");
                Ok(Arc::new(EagerAttentionBackend::new(
                    num_heads,
                    num_kv_heads,
                    head_dim,
                )?))
            }
        }
    }
}
