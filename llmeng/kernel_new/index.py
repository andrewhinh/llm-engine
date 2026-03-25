from __future__ import annotations

import functools
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

from llmeng.utils import is_sm100_supported

from .runtime import (
    as_cute_1d_tensor,
    as_cute_2d_tensor,
    get_current_cuda_stream,
    get_tensor_alignment,
    tile_last_dim,
)

WARP_SIZE = 32
NUM_THREADS = 128
WARPS_PER_BLOCK = NUM_THREADS // WARP_SIZE
MAX_VECTOR_BYTES = 16


class _IndexKernel:
    def __init__(
        self,
        tile_elems: int,
        num_splits: int,
        tiles_per_split: int,
    ) -> None:
        self.tile_elems = tile_elems
        self.num_splits = num_splits
        self.tiles_per_split = tiles_per_split
        self.split_shift = num_splits.bit_length() - 1
        self.split_mask = num_splits - 1

    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,
        indices: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        tiled_weights = tile_last_dim(weights, self.tile_elems)
        tiled_output = tile_last_dim(output, self.tile_elems)
        rows, _ = tiled_output.shape[1]
        num_warps = rows * self.num_splits
        self.kernel(tiled_weights, indices, tiled_output).launch(
            grid=((num_warps + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_weights: cute.Tensor,
        indices: cute.Tensor,
        tiled_output: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % WARP_SIZE
        warp_idx = tidx // WARP_SIZE
        logical_warp = bidx * WARPS_PER_BLOCK + warp_idx
        row = logical_warp >> self.split_shift
        rows, _ = tiled_output.shape[1]
        if row < rows:
            split_id = logical_warp & self.split_mask
            tile_base = split_id * self.tiles_per_split
            src_row = indices[row]
            for tile_offset in range(lane, self.tiles_per_split, WARP_SIZE):
                tile = tile_base + tile_offset
                src_tile = tiled_weights[(None, (src_row, tile))]
                dst_tile = tiled_output[(None, (row, tile))]
                dst_tile.store(src_tile.load())


class _MaskedIndexKernel:
    def __init__(
        self,
        tile_elems: int,
        num_splits: int,
        tiles_per_split: int,
    ) -> None:
        self.tile_elems = tile_elems
        self.num_splits = num_splits
        self.tiles_per_split = tiles_per_split
        self.split_shift = num_splits.bit_length() - 1
        self.split_mask = num_splits - 1

    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,
        indices: cute.Tensor,
        output: cute.Tensor,
        start: int,
        length: int,
        stream: cuda.CUstream,
    ) -> None:
        tiled_weights = tile_last_dim(weights, self.tile_elems)
        tiled_output = tile_last_dim(output, self.tile_elems)
        rows, _ = tiled_output.shape[1]
        num_warps = rows * self.num_splits
        self.kernel(
            tiled_weights,
            indices,
            tiled_output,
            start,
            length,
        ).launch(
            grid=((num_warps + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK, 1, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_weights: cute.Tensor,
        indices: cute.Tensor,
        tiled_output: cute.Tensor,
        start: int,
        length: int,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % WARP_SIZE
        warp_idx = tidx // WARP_SIZE
        logical_warp = bidx * WARPS_PER_BLOCK + warp_idx
        row = logical_warp >> self.split_shift
        rows, _ = tiled_output.shape[1]
        if row < rows:
            split_id = logical_warp & self.split_mask
            tile_base = split_id * self.tiles_per_split
            src_row = indices[row] - start
            valid = 0 <= src_row < length
            if valid:
                for tile_offset in range(lane, self.tiles_per_split, WARP_SIZE):
                    tile = tile_base + tile_offset
                    dst_tile = tiled_output[(None, (row, tile))]
                    src_tile = tiled_weights[(None, (src_row, tile))]
                    dst_tile.store(src_tile.load())
            else:
                for tile_offset in range(lane, self.tiles_per_split, WARP_SIZE):
                    tile = tile_base + tile_offset
                    dst_tile = tiled_output[(None, (row, tile))]
                    dst_tile.fill(0)


@functools.cache
def _compiled_index(
    width: int,
    device_index: int,
    torch_dtype: torch.dtype,
    index_dtype: torch.dtype,
    tile_elems: int,
    num_splits: int,
):
    sample_device = torch.device(f"cuda:{device_index}")
    sample_weights = torch.empty((2, width), device=sample_device, dtype=torch_dtype)
    sample_indices = torch.zeros((1,), device=sample_device, dtype=index_dtype)
    sample_output = torch.empty((1, width), device=sample_device, dtype=torch_dtype)
    return cute.compile(
        _IndexKernel(tile_elems, num_splits, (width // tile_elems) // num_splits),
        as_cute_2d_tensor(sample_weights, divisibility=tile_elems),
        as_cute_1d_tensor(sample_indices),
        as_cute_2d_tensor(sample_output, divisibility=tile_elems),
        get_current_cuda_stream(sample_device),
    )


@functools.cache
def _compiled_masked_index(
    width: int,
    device_index: int,
    torch_dtype: torch.dtype,
    index_dtype: torch.dtype,
    tile_elems: int,
    num_splits: int,
):
    sample_device = torch.device(f"cuda:{device_index}")
    sample_weights = torch.empty((2, width), device=sample_device, dtype=torch_dtype)
    sample_indices = torch.zeros((1,), device=sample_device, dtype=index_dtype)
    sample_output = torch.empty((1, width), device=sample_device, dtype=torch_dtype)
    return cute.compile(
        _MaskedIndexKernel(tile_elems, num_splits, (width // tile_elems) // num_splits),
        as_cute_2d_tensor(sample_weights, divisibility=tile_elems),
        as_cute_1d_tensor(sample_indices),
        as_cute_2d_tensor(sample_output, divisibility=tile_elems),
        0,
        1,
        get_current_cuda_stream(sample_device),
    )


def _validate_inputs(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if weights.ndim != 2:
        raise RuntimeError("weights must be 2D")
    if indices.ndim != 1:
        raise RuntimeError("indices must be 1D")
    if output.shape != (indices.shape[0], weights.shape[1]):
        raise RuntimeError("output shape must match (len(indices), embedding_dim)")
    if weights.device.type != "cuda" or indices.device.type != "cuda":
        raise RuntimeError("weights and indices must live on CUDA")
    if output.device != weights.device or indices.device != weights.device:
        raise RuntimeError("weights, indices, and output must share a CUDA device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError("indices must have dtype int32 or int64")
    if weights.dtype != output.dtype:
        raise RuntimeError("weights and output must share a dtype")
    if not weights.is_contiguous():
        raise RuntimeError("weights must be contiguous")
    if not indices.is_contiguous():
        raise RuntimeError("indices must be contiguous")
    if not output.is_contiguous():
        raise RuntimeError("output must be contiguous")


def _get_num_splits(width: int, element_size: int) -> int:
    row_bytes = width * element_size
    if row_bytes % 2048 == 0:
        return 4
    if row_bytes % 1024 == 0:
        return 2
    return 1


def _get_tile_elems(
    weights: torch.Tensor,
    output: torch.Tensor,
    *,
    num_splits: int,
) -> int:
    max_alignment = min(
        get_tensor_alignment(weights, max_alignment=MAX_VECTOR_BYTES),
        get_tensor_alignment(output, max_alignment=MAX_VECTOR_BYTES),
    )
    vector_elems = max(
        1, min(MAX_VECTOR_BYTES, max_alignment) // weights.element_size()
    )
    while vector_elems > 1:
        if weights.shape[1] % vector_elems == 0:
            tiles_per_row = weights.shape[1] // vector_elems
            if tiles_per_row % num_splits == 0:
                return vector_elems
        vector_elems //= 2
    return 1


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])
    _validate_inputs(weights, indices, output)
    total = output.numel()
    if total == 0:
        return output

    if _run_blackwell_tile_indexing(
        weights,
        indices,
        output,
        vocab_range=vocab_range,
    ):
        return output
    return _run_cute_indexing(
        weights,
        indices,
        output,
        vocab_range=vocab_range,
    )


def _run_blackwell_tile_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None,
) -> bool:
    if vocab_range is not None:
        return False
    if not is_sm100_supported(device=weights.device):
        return False
    from . import index_blackwell_tile

    if not index_blackwell_tile.supports_indexing(weights, indices, output):
        return False
    index_blackwell_tile.run_indexing(
        weights,
        indices,
        output,
    )
    return True


def _run_cute_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None,
) -> torch.Tensor:
    num_splits = _get_num_splits(weights.shape[1], weights.element_size())
    tile_elems = _get_tile_elems(weights, output, num_splits=num_splits)
    stream = get_current_cuda_stream(weights.device)
    if vocab_range is None:
        _compiled_index(
            weights.shape[1],
            weights.device.index or 0,
            weights.dtype,
            indices.dtype,
            tile_elems,
            num_splits,
        )(
            weights,
            indices,
            output,
            stream,
        )
    else:
        start, length = vocab_range
        _compiled_masked_index(
            weights.shape[1],
            weights.device.index or 0,
            weights.dtype,
            indices.dtype,
            tile_elems,
            num_splits,
        )(
            weights,
            indices,
            output,
            int(start),
            int(length),
            stream,
        )
    return output
