# pyright: reportMissingImports=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llmeng.env import ENV

if TYPE_CHECKING:
    import torch
    import torch.distributed

ReduceOp = Literal["sum", "prod", "max", "min", "avg"]


def _required_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _resolve_output_shape(tensor: torch.Tensor, world_size: int) -> tuple[int, ...]:
    if tensor.ndim == 0:
        raise RuntimeError("reduce_scatter requires at least 1D input")
    if tensor.shape[0] % world_size != 0:
        raise RuntimeError(
            "reduce_scatter requires the leading dimension to be divisible by world size"
        )
    return (tensor.shape[0] // world_size, *tensor.shape[1:])


def _resolve_gather_shape(tensor: torch.Tensor, world_size: int) -> tuple[int, ...]:
    if tensor.ndim == 0:
        raise RuntimeError("all_gather requires at least 1D input")
    return (tensor.shape[0] * world_size, *tensor.shape[1:])


class NcclCommunicator:
    def __init__(
        self,
        *,
        communicator: Any,
        world_size: int,
        rank: int,
        device: torch.device,
        scratch_tensor: torch.Tensor | None,
        scratch_handle: Any | None,
        scratch_window: Any | None,
        max_size_bytes: int,
    ) -> None:
        self._comm = communicator
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.max_size_bytes = max_size_bytes
        self._scratch_tensor = scratch_tensor
        self._scratch_handle = scratch_handle
        self._scratch_window = scratch_window
        self._destroyed = False

    def _check_active(self) -> None:
        if self._destroyed:
            raise RuntimeError("NcclCommunicator has been destroyed")

    def _check_cuda_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.device != self.device:
            raise RuntimeError(
                f"{name} must be on communicator device {self.device}, got {tensor.device}"
            )
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    def _stream_ptr(self, tensor: torch.Tensor) -> int:
        import torch

        return int(torch.cuda.current_stream(device=tensor.device).cuda_stream)

    def _close_resource(self, resource: Any) -> None:
        try:
            resource.close()
        except Exception:
            pass

    def _scratch_view_like(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if self._scratch_tensor is None:
            return None
        size_bytes = _required_bytes(tensor)
        if size_bytes == 0 or size_bytes > self.max_size_bytes:
            return None
        return self._scratch_tensor[:size_bytes].view(tensor.dtype).view_as(tensor)

    def all_reduce(self, input: torch.Tensor, op: ReduceOp = "sum") -> None:
        import nccl.bindings as nccl_bindings
        import nccl.core as nccl
        from nccl.core.interop.torch import resolve_tensor

        self._check_active()
        self._check_cuda_tensor("input", input)
        if input.numel() == 0:
            return
        target = input
        scratch = self._scratch_view_like(input)
        if scratch is not None:
            target = scratch
            if target.data_ptr() != input.data_ptr():
                target.copy_(input, non_blocking=True)
        reduce_op = {
            "sum": int(nccl.SUM),
            "prod": int(nccl.PROD),
            "max": int(nccl.MAX),
            "min": int(nccl.MIN),
            "avg": int(nccl.AVG),
        }[op]
        ptr, count, dtype, _ = resolve_tensor(target)
        nccl_bindings.all_reduce(
            ptr,
            ptr,
            count,
            int(dtype),
            reduce_op,
            self._comm.ptr,
            self._stream_ptr(input),
        )
        if target.data_ptr() != input.data_ptr():
            input.copy_(target, non_blocking=True)

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        import nccl.bindings as nccl_bindings
        from nccl.core.interop.torch import resolve_tensor

        self._check_active()
        self._check_cuda_tensor("input", input)
        self._check_cuda_tensor("output", output)
        if input.numel() == 0:
            return
        if output.dtype != input.dtype:
            raise RuntimeError("all_gather requires matching input/output dtypes")
        expected_shape = _resolve_gather_shape(input, self.world_size)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "all_gather output shape must match the gathered leading dimension"
            )
        src_ptr, count, dtype, _ = resolve_tensor(input)
        dst_ptr, _, _, _ = resolve_tensor(output)
        nccl_bindings.all_gather(
            src_ptr,
            dst_ptr,
            count,
            int(dtype),
            self._comm.ptr,
            self._stream_ptr(input),
        )

    def reduce_scatter(
        self,
        input: torch.Tensor,
        output: torch.Tensor | None = None,
        op: ReduceOp = "sum",
    ) -> torch.Tensor:
        import nccl.bindings as nccl_bindings
        import nccl.core as nccl
        from nccl.core.interop.torch import resolve_tensor

        self._check_active()
        self._check_cuda_tensor("input", input)
        if input.numel() == 0:
            if output is None:
                return input.new_empty(_resolve_output_shape(input, self.world_size))
            self._check_cuda_tensor("output", output)
            return output
        if input.numel() % self.world_size != 0:
            raise RuntimeError(
                "reduce_scatter input element count must be divisible by world size"
            )
        if output is None:
            output = input.new_empty(_resolve_output_shape(input, self.world_size))
        self._check_cuda_tensor("output", output)
        if output.dtype != input.dtype:
            raise RuntimeError("reduce_scatter requires matching input/output dtypes")
        expected_shape = _resolve_output_shape(input, self.world_size)
        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "reduce_scatter output shape must match the scattered leading dimension"
            )
        reduce_op = {
            "sum": int(nccl.SUM),
            "prod": int(nccl.PROD),
            "max": int(nccl.MAX),
            "min": int(nccl.MIN),
            "avg": int(nccl.AVG),
        }[op]
        src_ptr, _, dtype, _ = resolve_tensor(input)
        dst_ptr, count, _, _ = resolve_tensor(output)
        nccl_bindings.reduce_scatter(
            src_ptr,
            dst_ptr,
            count,
            int(dtype),
            reduce_op,
            self._comm.ptr,
            self._stream_ptr(input),
        )
        return output

    def get_buffer(self) -> int:
        if self._scratch_tensor is None:
            return 0
        return int(self._scratch_tensor.data_ptr())

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        for resource in (self._scratch_window, self._scratch_handle):
            if resource is None:
                continue
            self._close_resource(resource)
        self._scratch_window = None
        self._scratch_handle = None
        self._scratch_tensor = None
        self._comm.destroy()

    close = destroy


def init_nccl(
    *,
    local_rank: int,
    local_size: int,
    global_rank: int,
    global_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> NcclCommunicator:
    import nccl.core as nccl
    import torch
    import torch.distributed as dist
    from nccl.core.interop.torch import empty as nccl_empty

    if local_rank < 0:
        raise RuntimeError(f"local_rank must be non-negative, got {local_rank}")
    if local_size < 1:
        raise RuntimeError(f"local_size must be positive, got {local_size}")
    if max_size_bytes < 0:
        raise RuntimeError(f"max_size_bytes must be non-negative, got {max_size_bytes}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    max_size_bytes = min(max_size_bytes, ENV.NCCL_MAX_BUFFER_SIZE.value)

    group_rank = dist.get_rank(group=tp_cpu_group)
    group_size = dist.get_world_size(group=tp_cpu_group)
    if global_rank != group_rank or global_size != group_size:
        raise RuntimeError(
            "init_nccl requires global_rank/global_size to match tp_cpu_group scope"
        )

    uid_list = [nccl.get_unique_id().as_bytes if group_rank == 0 else None]
    dist.broadcast_object_list(uid_list, src=0, group=tp_cpu_group)
    uid_bytes = uid_list[0]
    if uid_bytes is None:
        raise RuntimeError("Failed to broadcast NCCL unique id")

    communicator = nccl.Communicator.init(
        nranks=group_size,
        rank=group_rank,
        unique_id=nccl.UniqueId.from_bytes(uid_bytes),
    )

    scratch_tensor = None
    scratch_handle = None
    scratch_window = None
    if max_size_bytes > 0:
        scratch_tensor = nccl_empty(
            (max_size_bytes,),
            dtype=torch.uint8,
            device=device,
        )
        scratch_handle = communicator.register_buffer(scratch_tensor)
        scratch_window = communicator.register_window(
            scratch_tensor, nccl.WindowFlag.CollSymmetric
        )

    return NcclCommunicator(
        communicator=communicator,
        world_size=group_size,
        rank=group_rank,
        device=device,
        scratch_tensor=scratch_tensor,
        scratch_handle=scratch_handle,
        scratch_window=scratch_window,
        max_size_bytes=max_size_bytes,
    )
