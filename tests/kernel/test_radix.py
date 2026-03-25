from __future__ import annotations

import torch
from llmeng.kernel import fast_compare_key

from llmeng.utils import compare_latency_kernel_perf, load_module


def test_fast_compare_key_prefix_match_int32():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    b = torch.tensor([1, 2, 9, 4], dtype=torch.int32)
    assert fast_compare_key(a, b) == 2


def test_fast_compare_key_full_match_shorter_len():
    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    assert fast_compare_key(a, b) == 3


def _ref_fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    common_len = min(x.shape[0], y.shape[0])
    diff = torch.nonzero(x[:common_len] != y[:common_len], as_tuple=False)
    return common_len if diff.numel() == 0 else int(diff[0].item())


def test_fast_compare_key_perf() -> None:
    old_fast_compare_key = load_module(True, "old", "radix").fast_compare_key
    new_fast_compare_key = load_module(False, "new", "radix").fast_compare_key

    for length in [2**10, 2**14, 2**18]:
        a = torch.arange(length, dtype=torch.int32)
        b = a.clone()
        b[-1] = -1
        old_fast_compare_key(a, b)
        new_fast_compare_key(a, b)
        _ref_fast_compare_key(a, b)
        compare_latency_kernel_perf(
            torch_impl=lambda: _ref_fast_compare_key(a, b),
            old_impl=lambda: old_fast_compare_key(a, b),
            new_impl=lambda: new_fast_compare_key(a, b),
            description=f"LEN={length:7d} | fast_compare_key | ",
            warmup=25,
            repetitions=100,
        )
