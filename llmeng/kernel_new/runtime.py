# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cuda.core import Device
from cutlass.cute.runtime import from_dlpack
from numba import cuda as _numba_cuda

if TYPE_CHECKING:
    import torch

NUMBA_CUDA: Any = _numba_cuda
_CUTE_1D_TENSOR_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}
_CUTE_2D_TENSOR_CACHE: dict[tuple[int, int], tuple[Any, Any]] = {}


def get_cuda_core_device(device_index: int | None = None) -> Any:
    device = Device() if device_index is None else Device(device_index)
    device.set_current()
    return device


@dataclass(frozen=True)
class TorchCudaStreamAdapter:
    handle: int

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


def get_torch_stream_adapter(
    stream: torch.cuda.Stream | None = None,
) -> TorchCudaStreamAdapter:
    import torch

    stream = stream or torch.cuda.current_stream()
    return TorchCudaStreamAdapter(handle=int(stream.cuda_stream))


def get_current_cuda_stream(
    device: torch.device | int | str | None = None,
) -> cuda.CUstream:
    import torch

    return cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)


def get_tensor_alignment(
    tensor: torch.Tensor,
    *,
    max_alignment: int = 16,
) -> int:
    alignment = max_alignment
    while alignment > tensor.element_size():
        if tensor.data_ptr() % alignment == 0:
            return alignment
        alignment //= 2
    return tensor.element_size()


def as_cute_2d_tensor(
    tensor: torch.Tensor,
    *,
    divisibility: int = 1,
) -> Any:
    if tensor.ndim != 2:
        raise RuntimeError("CuTe store tensors must be 2D")
    divisibility = max(divisibility, 1)
    cache_key = (id(tensor), divisibility)
    cached = _CUTE_2D_TENSOR_CACHE.get(cache_key)
    if cached is not None and cached[0] is tensor:
        return cached[1]
    cute_tensor = (
        from_dlpack(tensor, assumed_align=get_tensor_alignment(tensor))
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(
            mode=1,
            stride_order=tuple(tensor.dim_order()),
            divisibility=divisibility,
        )
    )
    _CUTE_2D_TENSOR_CACHE[cache_key] = (tensor, cute_tensor)
    return cute_tensor


def as_cute_1d_tensor(tensor: torch.Tensor) -> Any:
    if tensor.ndim != 1:
        raise RuntimeError("CuTe index tensors must be 1D")
    cache_key = (id(tensor), 1)
    cached = _CUTE_1D_TENSOR_CACHE.get(cache_key)
    if cached is not None and cached[0] is tensor:
        return cached[1]
    cute_tensor = from_dlpack(
        tensor,
        assumed_align=get_tensor_alignment(tensor, max_alignment=8),
    )
    _CUTE_1D_TENSOR_CACHE[cache_key] = (tensor, cute_tensor)
    return cute_tensor


def tile_last_dim(tensor: Any, tile_width: int) -> Any:
    return cute.zipped_divide(tensor, (1, tile_width))
