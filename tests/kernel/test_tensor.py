from __future__ import annotations

from minisgl.kernel import test_tensor as check_tensor_signature
from minisgl.utils import call_if_main
import torch
import pytest


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices"
)


@call_if_main(__name__)
def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    check_tensor_signature(x, y)
