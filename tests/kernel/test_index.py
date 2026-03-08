from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel import indexing
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


NUM_TOKENS = 131072


def ref_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
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
):
    EMBED_SIZE = 4096
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    if vocab_range is not None:
        label = f"vocab_range={vocab_range}, "
    else:
        label = ""

    for bs in [2**n for n in range(0, 16)]:
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        result = indexing(
            weights,
            indices,
            vocab_range=vocab_range,
        )
        expected = ref_indexing(
            weights,
            indices,
            vocab_range=vocab_range,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"

        mem = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices, vocab_range=vocab_range),
            baseline=lambda: ref_indexing(weights, indices, vocab_range=vocab_range),
            memory_footprint=mem,
            description=f"BS={bs:6d} | {label}",
            extra_kwargs=extra_kwargs,
        )


@call_if_main(__name__)
def test_indexing():
    run_indexing_case(vocab_range=None)


@call_if_main(__name__)
def test_indexing_with_mask():
    TP = 4
    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # start, length
    run_indexing_case(MASK_RANGE, extra_kwargs={"init_stream": False})
