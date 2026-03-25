from __future__ import annotations

import torch

from llmeng.distributed import get_tp_info
from llmeng.utils import split_kv_heads

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads, _, _ = split_kv_heads(num_kv_heads, tp_info.size, tp_info.rank)
        self._kv_buffer = torch.empty(
            (num_layers, num_pages, page_size, 2, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._device = device
        self._flat_shape = (num_pages * page_size, local_kv_heads * head_dim)
        self._token_stride = 2 * local_kv_heads * head_dim
        self._value_offset = local_kv_heads * head_dim

    def _layer_buffer(self, index: int) -> torch.Tensor:
        return self._kv_buffer[index]

    def _flat_cache_views(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        layer_buffer = self._layer_buffer(index)
        k_cache = layer_buffer.as_strided(
            self._flat_shape,
            (self._token_stride, 1),
            layer_buffer.storage_offset(),
        )
        v_cache = layer_buffer.as_strided(
            self._flat_shape,
            (self._token_stride, 1),
            layer_buffer.storage_offset() + self._value_offset,
        )
        return k_cache, v_cache

    def k_cache(self, index: int) -> torch.Tensor:
        return self._layer_buffer(index)[:, :, 0, :, :]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._layer_buffer(index)[:, :, 1, :, :]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from llmeng.kernel import store_cache

        k_cache, v_cache = self._flat_cache_views(layer_id)
        store_cache(
            k_cache=k_cache,
            v_cache=v_cache,
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
