from __future__ import annotations

import pytest
import torch

from llmeng.utils import compare_memory_kernel_perf, is_sm100_supported, load_module

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="requires >=1 CUDA device"
)


def _run_store_cache_perf_case(
    *,
    description_prefix: str = "",
    include_cutile: bool = False,
) -> None:
    HEAD_SIZE = 128
    NUM_TOKENS = 1048576  # 1M
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    kv_cache = torch.randn(
        (NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16
    )
    k_cache = kv_cache[:, 0, :]
    v_cache = kv_cache[:, 1, :]

    old_store = load_module(True, "old", "store").store_cache
    new_store = load_module(False, "new", "store").store_cache
    cutile_store = None
    if include_cutile:
        cutile_store = load_module(False, "new", "store_blackwell_tile").store_cache

    kv_cache_old_copy = kv_cache.clone()
    k_cache_old = kv_cache_old_copy[:, 0, :]
    v_cache_old = kv_cache_old_copy[:, 1, :]

    kv_cache_new_copy = kv_cache.clone()
    k_cache_new = kv_cache_new_copy[:, 0, :]
    v_cache_new = kv_cache_new_copy[:, 1, :]

    if cutile_store is not None:
        kv_cache_cutile_copy = kv_cache.clone()
        k_cache_cutile = kv_cache_cutile_copy[:, 0, :]
        v_cache_cutile = kv_cache_cutile_copy[:, 1, :]

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)
        qkv = torch.randn((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)
        k = qkv[:, :HEAD_SIZE]
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]
        old_store(k_cache_old, v_cache_old, indices, k, v)
        new_store(k_cache_new, v_cache_new, indices, k, v)
        if cutile_store is not None:
            cutile_store(k_cache_cutile, v_cache_cutile, indices, k, v)

        assert torch.all(k_cache_old[indices] == k), bs
        assert torch.all(v_cache_old[indices] == v), bs
        assert torch.all(k_cache_new[indices] == k), bs
        assert torch.all(v_cache_new[indices] == v), bs
        if cutile_store is not None:
            assert torch.all(k_cache_cutile[indices] == k), bs
            assert torch.all(v_cache_cutile[indices] == v), bs

        MEM = bs * HEAD_SIZE * 2 * kv_cache.element_size()

        k = k.contiguous()
        v = v.contiguous()

        @torch.compile()
        def baseline():
            k_cache[indices] = k
            v_cache[indices] = v

        compare_memory_kernel_perf(
            torch_impl=lambda: baseline(),
            old_impl=lambda: old_store(k_cache_old, v_cache_old, indices, k, v),
            new_impl=lambda: new_store(k_cache_new, v_cache_new, indices, k, v),
            cutile_impl=(
                None
                if cutile_store is None
                else lambda: cutile_store(k_cache_cutile, v_cache_cutile, indices, k, v)
            ),
            memory_footprint=MEM,
            description=f"{description_prefix}BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )


def test_store_cache():
    _run_store_cache_perf_case()


@pytest.mark.skipif(
    not is_sm100_supported(device="cuda:0"),
    reason="Blackwell-only perf case",
)
def test_store_cache_blackwell() -> None:
    _run_store_cache_perf_case(
        description_prefix="[blackwell] ",
        include_cutile=True,
    )
