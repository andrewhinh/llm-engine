from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from .runtime import (
    as_cute_1d_tensor,
    as_cute_2d_tensor,
    get_current_cuda_stream,
    tile_last_dim,
)

NUM_THREADS = 256
MAX_VECTOR_BYTES = 16


def _get_vector_elems(width: int, element_size: int) -> int:
    max_vector_elems = max(1, MAX_VECTOR_BYTES // element_size)
    vector_elems = max_vector_elems
    while vector_elems > 1:
        if width % vector_elems == 0 and width // vector_elems <= NUM_THREADS:
            return vector_elems
        vector_elems //= 2
    return 1


def _get_rows_per_block(num_rows: int, tiles_per_row: int) -> int:
    rows_per_block = min(NUM_THREADS // tiles_per_row, 8)
    while rows_per_block > 1 and num_rows % rows_per_block != 0:
        rows_per_block //= 2
    return max(rows_per_block, 1)


class _FusedStoreKernel:
    def __init__(self, tile_elems: int, tiles_per_row: int, rows_per_block: int):
        self.tile_elems = tile_elems
        self.tiles_per_row = tiles_per_row
        self.rows_per_block = rows_per_block

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        indices: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        tiled_src = tile_last_dim(src, self.tile_elems)
        tiled_dst = tile_last_dim(dst, self.tile_elems)
        rows, _ = tiled_src.shape[1]
        self.kernel(tiled_src, tiled_dst, indices).launch(
            grid=(rows // self.rows_per_block, 1, 1),
            block=(self.tiles_per_row, self.rows_per_block, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_src: cute.Tensor,
        tiled_dst: cute.Tensor,
        indices: cute.Tensor,
    ) -> None:
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * self.rows_per_block + tidy
        pos = indices[row]
        tiled_dst[(None, (pos, tidx))] = tiled_src[(None, (row, tidx))].load()


class _SplitStoreKernel:
    def __init__(self, tile_elems: int, tiles_per_row: int, rows_per_block: int):
        self.tile_elems = tile_elems
        self.tiles_per_row = tiles_per_row
        self.rows_per_block = rows_per_block

    @cute.jit
    def __call__(
        self,
        k_src: cute.Tensor,
        v_src: cute.Tensor,
        k_dst: cute.Tensor,
        v_dst: cute.Tensor,
        indices: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        tiled_k_src = tile_last_dim(k_src, self.tile_elems)
        tiled_v_src = tile_last_dim(v_src, self.tile_elems)
        tiled_k_dst = tile_last_dim(k_dst, self.tile_elems)
        tiled_v_dst = tile_last_dim(v_dst, self.tile_elems)
        rows, _ = tiled_k_src.shape[1]
        self.kernel(
            tiled_k_src,
            tiled_v_src,
            tiled_k_dst,
            tiled_v_dst,
            indices,
        ).launch(
            grid=(rows // self.rows_per_block, 1, 1),
            block=(self.tiles_per_row, self.rows_per_block, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_k_src: cute.Tensor,
        tiled_v_src: cute.Tensor,
        tiled_k_dst: cute.Tensor,
        tiled_v_dst: cute.Tensor,
        indices: cute.Tensor,
    ) -> None:
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * self.rows_per_block + tidy
        pos = indices[row]
        tiled_k_dst[(None, (pos, tidx))] = tiled_k_src[(None, (row, tidx))].load()
        tiled_v_dst[(None, (pos, tidx))] = tiled_v_src[(None, (row, tidx))].load()


@functools.cache
def _compiled_fused_store(
    width: int,
    device_index: int,
    torch_dtype: torch.dtype,
    index_dtype: torch.dtype,
    tile_elems: int,
    rows_per_block: int,
):
    sample_device = torch.device(f"cuda:{device_index}")
    sample_src = torch.empty(
        (rows_per_block, width), device=sample_device, dtype=torch_dtype
    )
    sample_dst = torch.empty(
        (rows_per_block * 2, width),
        device=sample_device,
        dtype=torch_dtype,
    )
    sample_indices = torch.zeros(
        (rows_per_block,), device=sample_device, dtype=index_dtype
    )
    return cute.compile(
        _FusedStoreKernel(tile_elems, width // tile_elems, rows_per_block),
        as_cute_2d_tensor(sample_src, divisibility=tile_elems),
        as_cute_2d_tensor(sample_dst, divisibility=tile_elems),
        as_cute_1d_tensor(sample_indices),
        get_current_cuda_stream(sample_device),
    )


@functools.cache
def _compiled_split_store(
    width: int,
    device_index: int,
    torch_dtype: torch.dtype,
    index_dtype: torch.dtype,
    tile_elems: int,
    rows_per_block: int,
):
    sample_device = torch.device(f"cuda:{device_index}")
    sample_k_src = torch.empty(
        (rows_per_block, width),
        device=sample_device,
        dtype=torch_dtype,
    )
    sample_v_src = torch.empty(
        (rows_per_block, width),
        device=sample_device,
        dtype=torch_dtype,
    )
    sample_k_dst = torch.empty(
        (rows_per_block * 2, width),
        device=sample_device,
        dtype=torch_dtype,
    )
    sample_v_dst = torch.empty(
        (rows_per_block * 2, width),
        device=sample_device,
        dtype=torch_dtype,
    )
    sample_indices = torch.zeros(
        (rows_per_block,), device=sample_device, dtype=index_dtype
    )
    return cute.compile(
        _SplitStoreKernel(tile_elems, width // tile_elems, rows_per_block),
        as_cute_2d_tensor(sample_k_src, divisibility=tile_elems),
        as_cute_2d_tensor(sample_v_src, divisibility=tile_elems),
        as_cute_2d_tensor(sample_k_dst, divisibility=tile_elems),
        as_cute_2d_tensor(sample_v_dst, divisibility=tile_elems),
        as_cute_1d_tensor(sample_indices),
        get_current_cuda_stream(sample_device),
    )


def _get_adjacent_pair_view(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor | None:
    if first.shape != second.shape or first.ndim != 2:
        return None
    if first.device != second.device or first.dtype != second.dtype:
        return None
    if first.stride(-1) != 1 or second.stride(-1) != 1:
        return None
    if first.stride(0) != second.stride(0):
        return None
    width = first.shape[1]
    if first.stride(0) < width * 2:
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if second.storage_offset() != first.storage_offset() + width:
        return None
    return first.as_strided(
        (first.shape[0], width * 2),
        (first.stride(0), 1),
        first.storage_offset(),
    )


def _validate_inputs(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if k_cache.ndim != 2 or v_cache.ndim != 2:
        raise RuntimeError("cache tensors must be 2D views")
    if k.ndim != 2 or v.ndim != 2:
        raise RuntimeError("k and v must be 2D")
    if k_cache.shape != v_cache.shape:
        raise RuntimeError("k_cache and v_cache must share a shape")
    if k.shape != v.shape:
        raise RuntimeError("k and v must share a shape")
    if k.shape[0] != indices.shape[0]:
        raise RuntimeError("k/v rows must match indices length")
    if k.shape[1] != k_cache.shape[1]:
        raise RuntimeError("k/v width must match cache width")
    tensors = (k_cache, v_cache, indices, k, v)
    if any(t.device.type != "cuda" for t in tensors):
        raise RuntimeError("store_cache tensors must live on CUDA")
    base_device = k_cache.device
    if any(t.device != base_device for t in tensors):
        raise RuntimeError("store_cache tensors must share one CUDA device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("indices must have dtype int32 or int64")
    if k.dtype != v.dtype or k.dtype != k_cache.dtype or v.dtype != v_cache.dtype:
        raise RuntimeError("store_cache tensors must share one dtype")
    if not indices.is_contiguous():
        raise RuntimeError("indices must be contiguous")
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("k and v must have contiguous last dimensions")
    if k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise RuntimeError("cache views must have contiguous last dimensions")


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    _validate_inputs(k_cache, v_cache, indices, k, v)
    total = k.numel()
    if total == 0:
        return
    stream = get_current_cuda_stream(k_cache.device)
    num_rows = k.shape[0]
    fused_cache = _get_adjacent_pair_view(k_cache, v_cache)
    fused_kv = _get_adjacent_pair_view(k, v)
    if fused_cache is not None and fused_kv is not None:
        tile_elems = _get_vector_elems(fused_cache.shape[1], fused_cache.element_size())
        rows_per_block = _get_rows_per_block(
            num_rows,
            fused_cache.shape[1] // tile_elems,
        )
        _compiled_fused_store(
            fused_cache.shape[1],
            k_cache.device.index or 0,
            fused_cache.dtype,
            indices.dtype,
            tile_elems,
            rows_per_block,
        )(
            fused_kv,
            fused_cache,
            indices,
            stream,
        )
        return
    tile_elems = _get_vector_elems(k.shape[1], k.element_size())
    rows_per_block = _get_rows_per_block(num_rows, k.shape[1] // tile_elems)
    _compiled_split_store(
        k.shape[1],
        k_cache.device.index or 0,
        k.dtype,
        indices.dtype,
        tile_elems,
        rows_per_block,
    )(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        stream,
    )
