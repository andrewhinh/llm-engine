from __future__ import annotations


def div_even(a: int, b: int) -> int:
    """Divides two integers"""
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def split_kv_heads(
    num_kv_heads: int, tp_size: int, tp_rank: int
) -> tuple[int, int, int]:
    """Map a TP rank to KV-head shard info, allowing KV replication."""
    if num_kv_heads >= tp_size:
        return div_even(num_kv_heads, tp_size), tp_rank, tp_size

    assert tp_size % num_kv_heads == 0, (
        f"{tp_size = } must be divisible by {num_kv_heads = } when KV heads are replicated"
    )
    kv_replication = div_even(tp_size, num_kv_heads)
    return 1, tp_rank // kv_replication, num_kv_heads


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()
