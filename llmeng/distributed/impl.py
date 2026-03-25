from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from llmeng.distributed import DistributedInfo
    from llmeng.kernel import NcclCommunicator, init_nccl


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    group: torch.distributed.ProcessGroup | None = None
    world_size: int = 1

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * self.world_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x, group=self.group)
        return out


@dataclass
class NcclDistributedImpl(DistributedImpl):
    comm: NcclCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        output_shape = list(x.shape)
        output_shape[0] *= self.comm.world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result

    def destroy(self) -> None:
        self.comm.destroy()


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def configure_torch_distributed(
    tp_group: torch.distributed.ProcessGroup | None,
    tp_world_size: int,
) -> None:
    DistributedCommunicator.plugins[0] = TorchDistributedImpl(
        group=tp_group,
        world_size=tp_world_size,
    )


def enable_nccl_distributed(
    tp_info: DistributedInfo,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_bytes: int,
) -> None:
    """
    Enable nccl4py-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    tp_group_size = dist.get_world_size(group=tp_cpu_group)
    if tp_group_size == 1:
        return
    if tp_group_size != tp_info.size:
        return
    tp_group_rank = dist.get_rank(group=tp_cpu_group)

    comm = init_nccl(
        local_rank=tp_info.rank,
        local_size=tp_info.size,
        global_rank=tp_group_rank,
        global_size=tp_group_size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )
    DistributedCommunicator.plugins.append(NcclDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    for plugin in reversed(DistributedCommunicator.plugins[1:]):
        destroy = getattr(plugin, "destroy", None)
        if callable(destroy):
            destroy()
            continue
        comm = getattr(plugin, "comm", None)
        if comm is None:
            continue
        destroy = getattr(comm, "destroy", None) or getattr(comm, "close", None)
        if callable(destroy):
            destroy()
    DistributedCommunicator.plugins = [TorchDistributedImpl()]
