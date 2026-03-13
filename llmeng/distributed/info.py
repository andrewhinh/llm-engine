from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int
    global_rank: int | None = None
    global_size: int | None = None

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        has_global_rank = self.global_rank is not None
        has_global_size = self.global_size is not None
        if has_global_rank != has_global_size:
            raise ValueError("global_rank and global_size must be set together")
        if has_global_rank:
            assert self.global_rank is not None
            assert self.global_size is not None
            assert 0 <= self.global_rank < self.global_size

    def is_primary(self) -> bool:
        return self.rank == 0

    @property
    def resolved_global_rank(self) -> int:
        if self.global_rank is None:
            return self.rank
        return self.global_rank

    @property
    def resolved_global_size(self) -> int:
        if self.global_size is None:
            return self.size
        return self.global_size


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
