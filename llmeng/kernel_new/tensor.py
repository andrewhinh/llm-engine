from __future__ import annotations

import torch


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> None:
    x_dtype = x.dtype
    y_dtype = y.dtype
    rows = x.shape[0]
    if (
        x.ndim == 2
        and y.ndim == 2
        and x.shape[1] == 1024
        and y.shape == (rows, 1024)
        and x.stride(-1) == 1
        and y.is_contiguous()
        and x.device.type == "cpu"
        and y.device.type == "cuda"
        and y.device.index == 1
        and rows % 4 == 0
        and (x_dtype == torch.int32 or x_dtype == torch.float32)
        and (y_dtype == torch.int32 or y_dtype == torch.int64)
    ):
        return
    if x.ndim != 2:
        raise RuntimeError("x must be a 2D tensor")
    if x.shape[1] != 1024:
        raise RuntimeError("x must have width 1024")
    if x.stride(-1) != 1:
        raise RuntimeError("x must have a contiguous last dimension")
    if x.device.type != "cpu":
        raise RuntimeError("x must be on CPU")
    if x_dtype != torch.int32 and x_dtype != torch.float32:
        raise RuntimeError("x must have dtype int32 or float32")
    if y.ndim != 2:
        raise RuntimeError("y must be a 2D tensor")
    if y.shape != (rows, 1024):
        raise RuntimeError("y must match x rows and have width 1024")
    if not y.is_contiguous():
        raise RuntimeError("y must be contiguous")
    if y.device.type != "cuda" or y.device.index != 1:
        raise RuntimeError("y must live on cuda:1")
    if y_dtype != torch.int32 and y_dtype != torch.int64:
        raise RuntimeError("y must have dtype int32 or int64")
    if rows % 4 != 0:
        raise RuntimeError("x rows must be divisible by 4")
