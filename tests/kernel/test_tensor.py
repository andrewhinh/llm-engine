from __future__ import annotations

import pytest
import torch

from llmeng.kernel import test_tensor as check_tensor_signature
from llmeng.utils import compare_latency_kernel_perf, load_module

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices"
)


def test_tensor_signature_matches_expected_contract() -> None:
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    check_tensor_signature(x, y)


def test_tensor_signature_perf() -> None:
    old_test_tensor = load_module(True, "old", "tensor").test_tensor
    new_test_tensor = load_module(False, "new", "tensor").test_tensor

    for rows in [4, 256, 4096]:
        x = torch.empty((rows, 2048), dtype=torch.int32, device="cpu")[:, :1024]
        y = torch.empty((rows, 1024), dtype=torch.int64, device="cuda:1")
        compare_latency_kernel_perf(
            old_impl=lambda: old_test_tensor(x, y),
            new_impl=lambda: new_test_tensor(x, y),
            description=f"ROWS={rows:6d} | test_tensor | ",
            warmup=100,
            repetitions=1000,
        )
