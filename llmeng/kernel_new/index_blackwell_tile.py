from __future__ import annotations

import functools
from typing import Tuple

import torch

from llmeng.utils import is_sm100_supported

ROW_TILE = 8
COL_TILE = 256


def supports_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    if not is_sm100_supported(device=weights.device):
        return False
    if weights.ndim != 2 or indices.ndim != 1 or output.ndim != 2:
        return False
    if weights.device.type != "cuda":
        return False
    if indices.device != weights.device or output.device != weights.device:
        return False
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if output.dtype != weights.dtype:
        return False
    if indices.dtype not in (torch.int32, torch.int64):
        return False
    return True


@functools.cache
def _get_index_kernel(index_dtype: torch.dtype):
    import cuda.tile as ct

    globals()["ct"] = ct
    tile_index_dtype = ct.int64 if index_dtype == torch.int64 else ct.int32

    def _broadcast_rows(tile):
        return ct.broadcast_to(ct.reshape(tile, (ROW_TILE, 1)), (ROW_TILE, COL_TILE))

    def _broadcast_cols(tile):
        return ct.broadcast_to(ct.reshape(tile, (1, COL_TILE)), (ROW_TILE, COL_TILE))

    @ct.kernel
    def _kernel(weights, indices, output):
        bid_row = ct.bid(0)
        bid_col = ct.bid(1)
        row_offsets = ct.arange(ROW_TILE, dtype=tile_index_dtype) + bid_row * ROW_TILE
        col_offsets = ct.arange(COL_TILE, dtype=tile_index_dtype) + bid_col * COL_TILE
        gathered_rows = ct.gather(indices, row_offsets, padding_value=0)
        values = ct.gather(
            weights,
            (_broadcast_rows(gathered_rows), _broadcast_cols(col_offsets)),
            padding_value=0,
            check_bounds=True,
        )
        ct.scatter(
            output,
            (_broadcast_rows(row_offsets), _broadcast_cols(col_offsets)),
            values,
            check_bounds=True,
        )

    return _kernel


# NOTE: underperforms CuTe-DSL version, so only here for reference.


@functools.cache
def _get_masked_index_kernel(index_dtype: torch.dtype):
    import cuda.tile as ct

    globals()["ct"] = ct
    tile_index_dtype = ct.int64 if index_dtype == torch.int64 else ct.int32

    def _broadcast_rows(tile):
        return ct.broadcast_to(ct.reshape(tile, (ROW_TILE, 1)), (ROW_TILE, COL_TILE))

    def _broadcast_cols(tile):
        return ct.broadcast_to(ct.reshape(tile, (1, COL_TILE)), (ROW_TILE, COL_TILE))

    @ct.kernel
    def _kernel(weights, indices, output, start: int, length: int):
        bid_row = ct.bid(0)
        bid_col = ct.bid(1)
        row_offsets = ct.arange(ROW_TILE, dtype=tile_index_dtype) + bid_row * ROW_TILE
        col_offsets = ct.arange(COL_TILE, dtype=tile_index_dtype) + bid_col * COL_TILE
        gathered_rows = ct.gather(indices, row_offsets, padding_value=0) - start
        valid_rows = (gathered_rows >= 0) & (gathered_rows < length)
        safe_rows = ct.where(valid_rows, gathered_rows, tile_index_dtype(-1))
        values = ct.gather(
            weights,
            (_broadcast_rows(safe_rows), _broadcast_cols(col_offsets)),
            padding_value=0,
            check_bounds=True,
        )
        ct.scatter(
            output,
            (_broadcast_rows(row_offsets), _broadcast_cols(col_offsets)),
            values,
            check_bounds=True,
        )

    return _kernel


def run_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,
) -> torch.Tensor:
    import cuda.tile as ct

    grid = (
        ct.cdiv(output.shape[0], ROW_TILE),
        ct.cdiv(output.shape[1], COL_TILE),
        1,
    )
    stream = torch.cuda.current_stream(device=weights.device)
    if vocab_range is None:
        ct.launch(
            stream,
            grid,
            _get_index_kernel(indices.dtype),
            (weights, indices, output),
        )
    else:
        start, length = vocab_range
        ct.launch(
            stream,
            grid,
            _get_masked_index_kernel(indices.dtype),
            (weights, indices, output, int(start), int(length)),
        )
    return output
