from __future__ import annotations

import functools
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

from llmeng.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("llmeng.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


class _PennyComm:
    def __init__(
        self,
        local_rank: int,
        tp_cpu_group: torch.distributed.ProcessGroup,
        max_size_bytes: int = 0,
    ) -> None:
        from llmeng.kernel.penny.custom_all_reduce import CustomAllreduce

        # Penny expects NNODES to partition communication.
        os.environ.setdefault("NNODES", "1")

        self._ca = CustomAllreduce(
            group=tp_cpu_group,
            device=f"cuda:{local_rank}",
            max_size=max_size_bytes,
        )
        if self._ca.disabled:
            raise RuntimeError("Penny custom allreduce is disabled for this config")

        self._group = tp_cpu_group
        self._local_rank = local_rank

    def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None:
        if op != "sum":
            raise ValueError("Penny backend currently only supports sum reduction")

        import torch.distributed as dist

        output = self._ca.custom_all_reduce(input)
        if output is None:
            dist.all_reduce(input, op=dist.ReduceOp.SUM, group=self._group)
            return

        if output.data_ptr() != input.data_ptr():
            input.copy_(output)

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        import torch.distributed as dist

        gathered = self._ca.custom_all_gather(input, out=output)
        if gathered is None:
            dist.all_gather_into_tensor(output, input, group=self._group)
            return

        if gathered.data_ptr() != output.data_ptr():
            output.copy_(gathered)

    def get_buffer(self) -> int:
        # For compatibility with the legacy NCCL extension API. Penny keeps a
        # pool of IPC buffers per process rank in rank_data, while buffer ptrs
        # are an implementation detail of the original backend.
        try:
            return int(self._ca.buffer_ptrs[self._local_rank])
        except Exception:
            return 0


def _init_penny(
    local_rank: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int,
) -> PyNCCLCommunicator | None:
    try:
        return _PennyComm(  # type: ignore[return-value]
            local_rank=local_rank,
            tp_cpu_group=tp_cpu_group,
            max_size_bytes=max_size_bytes,
        )
    except Exception as e:
        warnings.warn(f"Penny backend unavailable for collective ops: {e}")
        return None


def _init_nccl(
    global_rank: int,
    global_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int,
) -> PyNCCLCommunicator:
    import torch

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if global_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert nccl_id is not None, f"Failed to get NCCL unique ID on {global_rank = }"

    return cls(global_rank, global_size, max_size_bytes, nccl_id)


def init_pynccl(
    *,
    local_rank: int,
    local_size: int,
    global_rank: int,
    global_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)
    _ = local_size
    penny_comm = _init_penny(
        local_rank=local_rank,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_size_bytes,
    )
    if penny_comm is not None:
        return penny_comm

    return _init_nccl(
        global_rank=global_rank,
        global_size=global_size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_size_bytes,
    )
