from __future__ import annotations

import numba
import numpy as np
import torch


@numba.njit(cache=True, boundscheck=False)
def _fast_compare_i32(x: np.ndarray, y: np.ndarray) -> int:
    n = min(x.shape[0], y.shape[0])
    n_word = n // 2
    if n_word > 0:
        xv = x[: n_word * 2].view(np.uint64)
        yv = y[: n_word * 2].view(np.uint64)
        block_end = (n_word // 4) * 4
        j = 0
        while j < block_end:
            if xv[j] != yv[j]:
                i0 = j * 2
                if x[i0] != y[i0]:
                    return i0
                return i0 + 1
            if xv[j + 1] != yv[j + 1]:
                i0 = (j + 1) * 2
                if x[i0] != y[i0]:
                    return i0
                return i0 + 1
            if xv[j + 2] != yv[j + 2]:
                i0 = (j + 2) * 2
                if x[i0] != y[i0]:
                    return i0
                return i0 + 1
            if xv[j + 3] != yv[j + 3]:
                i0 = (j + 3) * 2
                if x[i0] != y[i0]:
                    return i0
                return i0 + 1
            j += 4
        while j < n_word:
            if xv[j] != yv[j]:
                i0 = j * 2
                if x[i0] != y[i0]:
                    return i0
                return i0 + 1
            j += 1
    if n_word * 2 < n and x[n_word * 2] != y[n_word * 2]:
        return n_word * 2
    return n


@numba.njit(cache=True, boundscheck=False)
def _fast_compare_i64(x: np.ndarray, y: np.ndarray) -> int:
    n = min(x.shape[0], y.shape[0])
    block_end = (n // 4) * 4
    idx = 0
    while idx < block_end:
        if x[idx] != y[idx]:
            return idx
        if x[idx + 1] != y[idx + 1]:
            return idx + 1
        if x[idx + 2] != y[idx + 2]:
            return idx + 2
        if x[idx + 3] != y[idx + 3]:
            return idx + 3
        idx += 4
    while idx < n:
        if x[idx] != y[idx]:
            return idx
        idx += 1
    return n


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    x_dtype = x.dtype
    if (
        x.ndim == 1
        and y.ndim == 1
        and x.is_contiguous()
        and y.is_contiguous()
        and x.device.type == "cpu"
        and y.device.type == "cpu"
        and x_dtype == y.dtype
    ):
        if x_dtype == torch.int32:
            return int(_fast_compare_i32(x.numpy(), y.numpy()))
        if x_dtype == torch.int64:
            return int(_fast_compare_i64(x.numpy(), y.numpy()))
    if x.ndim != 1 or y.ndim != 1:
        raise RuntimeError("x and y must be 1D")
    if not x.is_contiguous() or not y.is_contiguous():
        raise RuntimeError("x and y must be contiguous")
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise RuntimeError("x and y must be on CPU")
    if x_dtype != y.dtype:
        raise RuntimeError("x and y must share the same dtype")
    raise RuntimeError("x and y must have dtype int32 or int64")
