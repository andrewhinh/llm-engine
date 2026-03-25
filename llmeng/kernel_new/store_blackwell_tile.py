# NOTE: underperforms CuTe-DSL version, so only here for reference.

from __future__ import annotations

import functools

import torch

from llmeng.utils import is_sm100_supported

MAX_ROW_TILE = 8


def _get_row_tile(num_rows: int) -> int:
    for row_tile in (MAX_ROW_TILE, 4, 2):
        if num_rows % row_tile == 0:
            return row_tile
    return 1


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


def supports_store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    if not is_sm100_supported(device=k_cache.device):
        return False
    if k_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if indices.dtype not in (torch.int32, torch.int64):
        return False
    return True


@functools.cache
def _get_store_kernel(
    *,
    width: int,
    row_tile: int,
    index_dtype: torch.dtype,
):
    import cuda.tile as ct

    globals()["ct"] = ct
    tile_index_dtype = ct.int64 if index_dtype == torch.int64 else ct.int32

    def _broadcast_rows(tile):
        return ct.broadcast_to(ct.reshape(tile, (row_tile, 1)), (row_tile, width))

    def _broadcast_cols(tile):
        return ct.broadcast_to(ct.reshape(tile, (1, width)), (row_tile, width))

    @ct.kernel
    def _kernel(src, dst, indices):
        bid_row = ct.bid(0)
        row_offsets = ct.arange(row_tile, dtype=tile_index_dtype) + bid_row * row_tile
        col_offsets = ct.arange(width, dtype=tile_index_dtype)
        dst_rows = ct.gather(indices, row_offsets, check_bounds=False, latency=1)
        values = ct.gather(
            src,
            (_broadcast_rows(row_offsets), _broadcast_cols(col_offsets)),
            check_bounds=False,
            latency=10,
        )
        ct.scatter(
            dst,
            (_broadcast_rows(dst_rows), _broadcast_cols(col_offsets)),
            values,
            check_bounds=False,
            latency=10,
        )

    return _kernel


def _run_store(
    src: torch.Tensor,
    dst: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    import cuda.tile as ct

    row_tile = _get_row_tile(src.shape[0])
    grid = (src.shape[0] // row_tile, 1, 1)
    stream = torch.cuda.current_stream(device=src.device)
    ct.launch(
        stream,
        grid,
        _get_store_kernel(
            width=src.shape[1],
            row_tile=row_tile,
            index_dtype=indices.dtype,
        ),
        (src, dst, indices),
    )


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
    if k.numel() == 0:
        return
    if not supports_store_cache(k_cache, v_cache, indices, k, v):
        raise RuntimeError(
            "Blackwell cuTile store requires SM100+ CUDA tensors with matching dtypes"
        )

    fused_cache = _get_adjacent_pair_view(k_cache, v_cache)
    fused_kv = _get_adjacent_pair_view(k, v)
    if fused_cache is not None and fused_kv is not None:
        _run_store(fused_kv, fused_cache, indices)
        return
    _run_store(k, k_cache, indices)
    _run_store(v, v_cache, indices)
