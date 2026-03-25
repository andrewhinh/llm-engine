from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from llmeng.utils import (
    compare_memory_kernel_perf,
    init_logger,
    is_sm100_supported,
    load_module,
)

logger = init_logger(__name__)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="requires >=1 CUDA device"
)


NUM_TOKENS = 131072
EMBED_SIZE = 4096


def ref_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    indices = indices.clone()
    if vocab_range is not None:
        start, length = vocab_range
        assert length <= weights.shape[0]
        indices = indices - start
        indices_mask = (indices < 0) | (indices >= length)
        indices[indices_mask] = 0  # set out-of-vocab indices to zero
        result = F.embedding(indices, weights)
        result[indices_mask] = 0
        return result
    else:
        return F.embedding(indices, weights)


def run_indexing_case(
    vocab_range: Tuple[int, int] | None,
    *,
    extra_kwargs: Dict[str, object] | None = None,
    description_prefix: str = "",
    include_cutile: bool = False,
):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)
    old_index = load_module(True, "old", "index").indexing
    new_index = load_module(False, "new", "index").indexing
    cutile_index = None
    if include_cutile:
        cutile_index = load_module(False, "new", "index_blackwell_tile").run_indexing

    if vocab_range is not None:
        label = f"vocab_range={vocab_range}, "
    else:
        label = ""

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)
        old_result = old_index(weights, indices, vocab_range=vocab_range)
        new_result = new_index(weights, indices, vocab_range=vocab_range)
        expected = ref_indexing(weights, indices, vocab_range=vocab_range)
        assert torch.equal(old_result, expected), f"Old mismatch for BS={bs}"
        assert torch.equal(new_result, expected), f"New mismatch for BS={bs}"
        if cutile_index is not None:
            cutile_output = torch.empty(
                (bs, EMBED_SIZE), device=weights.device, dtype=weights.dtype
            )
            cutile_result = cutile_index(
                weights,
                indices,
                cutile_output,
                vocab_range=vocab_range,
            )
            assert torch.equal(cutile_result, expected), f"CuTile mismatch for BS={bs}"

        mem = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            torch_impl=lambda: ref_indexing(weights, indices, vocab_range=vocab_range),
            old_impl=lambda: old_index(weights, indices, vocab_range=vocab_range),
            new_impl=lambda: new_index(weights, indices, vocab_range=vocab_range),
            cutile_impl=(
                (
                    lambda: cutile_index(
                        weights,
                        indices,
                        cutile_output,
                        vocab_range=vocab_range,
                    )
                )
                if cutile_index is not None
                else None
            ),
            memory_footprint=mem,
            description=f"{description_prefix}BS={bs:6d} | {label}",
            extra_kwargs=extra_kwargs,
        )


def test_indexing():
    run_indexing_case(vocab_range=None)


def test_indexing_with_mask():
    TP = 4
    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    run_indexing_case(
        (MASK_LENGTH, MASK_LENGTH),  # (start, length)
        extra_kwargs={"init_stream": False},
        description_prefix="[masked] ",
    )


@pytest.mark.skipif(
    not is_sm100_supported(device="cuda:0"),
    reason="requires Blackwell SM100+",
)
def test_indexing_blackwell() -> None:
    run_indexing_case(
        vocab_range=None,
        description_prefix="[blackwell] ",
        include_cutile=True,
    )


@pytest.mark.skipif(
    not is_sm100_supported(device="cuda:0"),
    reason="requires Blackwell SM100+",
)
def test_indexing_blackwell_masked() -> None:
    TP = 4
    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    run_indexing_case(
        vocab_range=(MASK_LENGTH, MASK_LENGTH),
        extra_kwargs={"init_stream": False},
        description_prefix="[blackwell-masked] ",
        include_cutile=True,
    )
