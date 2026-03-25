from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llmeng.utils import compare_memory_kernel_perf, load_module

moe_new = load_module(False, "new", "fused_moe")

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="requires >=1 CUDA device"
)


def _get_triton_config(num_tokens: int, num_experts: int) -> dict[str, int]:
    if num_tokens <= num_experts:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }


def _make_moe_inputs(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.empty(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ).normal_(0, 0.5)
    w1 = torch.empty(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device="cuda",
        dtype=dtype,
    ).normal_(0, 0.1)
    w2 = torch.empty(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=dtype,
    ).normal_(0, 0.1)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", dtype=torch.float32), dim=-1
    ).to(dtype)
    return hidden_states, w1, w2, topk_weights, topk_ids


def _align_tokens_by_expert(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = topk_ids.device
    flat_expert_ids = topk_ids.reshape(-1)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)
    expert_token_counts = torch.bincount(
        flat_expert_ids.to(torch.int64), minlength=num_experts
    )
    expert_block_counts = (expert_token_counts + block_size - 1) // block_size
    total_blocks = int(expert_block_counts.sum().item())
    total_tokens = topk_ids.numel()
    total_tokens_post_padded = total_blocks * block_size

    sorted_token_ids = torch.full(
        (total_tokens_post_padded,),
        total_tokens,
        device=device,
        dtype=torch.int32,
    )
    expert_ids = torch.empty((total_blocks,), device=device, dtype=torch.int32)

    token_offset = 0
    block_offset = 0
    for expert_id in range(num_experts):
        token_count = int(expert_token_counts[expert_id].item())
        block_count = int(expert_block_counts[expert_id].item())
        if block_count == 0:
            continue
        expert_ids[block_offset : block_offset + block_count] = expert_id
        block_start = block_offset * block_size
        sorted_token_ids[block_start : block_start + token_count] = (
            sorted_token_indices[token_offset : token_offset + token_count].to(
                torch.int32
            )
        )
        token_offset += token_count
        block_offset += block_count

    num_tokens_post_padded = torch.tensor(
        [total_tokens_post_padded], device=device, dtype=torch.int32
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _ref_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    gate_proj, up_proj = w1.chunk(2, dim=1)
    final_hidden_states = torch.zeros_like(hidden_states)

    num_experts = w1.shape[0]
    expert_mask = F.one_hot(topk_ids.to(torch.int64), num_classes=num_experts).permute(
        2, 1, 0
    )

    for expert_id in range(num_experts):
        matched_ks, matched_token_ids = torch.where(expert_mask[expert_id])
        if matched_token_ids.numel() == 0:
            continue
        matched_tokens = hidden_states[matched_token_ids]
        routing_weights = topk_weights[matched_token_ids, matched_ks].unsqueeze(-1)

        gate_output = matched_tokens @ gate_proj[expert_id].transpose(0, 1)
        up_output = matched_tokens @ up_proj[expert_id].transpose(0, 1)
        if apply_router_weight_on_input:
            gate_output = gate_output * routing_weights.to(gate_output.dtype)
            up_output = up_output * routing_weights.to(up_output.dtype)
        expert_output = F.silu(gate_output) * up_output
        expert_output = expert_output @ w2[expert_id].transpose(0, 1)

        if not apply_router_weight_on_input:
            expert_output = expert_output * routing_weights.to(expert_output.dtype)

        final_hidden_states.index_add_(
            0,
            matched_token_ids,
            expert_output.to(hidden_states.dtype),
        )

    return final_hidden_states


def _run_triton_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    num_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    intermediate_size = w2.shape[2]
    config = _get_triton_config(num_tokens, num_experts)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_tokens_by_expert(
        topk_ids, config["BLOCK_SIZE_M"], num_experts
    )

    intermediate_cache1 = torch.empty(
        (num_tokens, topk, w1.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (num_tokens * topk, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (num_tokens, topk, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    out = torch.empty_like(hidden_states)

    moe_new.fused_moe_kernel_triton(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        config,
        compute_type=hidden_states.dtype,
    )

    gate_output, up_output = intermediate_cache1.chunk(2, dim=-1)
    intermediate_cache2.copy_(
        (F.silu(gate_output) * up_output).view(-1, intermediate_size)
    )

    moe_new.fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=hidden_states.dtype,
    )
    moe_new.moe_sum_reduce_triton(intermediate_cache3, out)
    return out


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_experts",
        "hidden_size",
        "intermediate_size",
        "topk",
        "dtype",
        "apply_router_weight_on_input",
    ),
    [
        pytest.param(16, 32, 96, 80, 1, torch.float16, False, id="small_top1_fp16"),
        pytest.param(
            17, 32, 88, 52, 3, torch.bfloat16, True, id="small_tailk_bf16_router_in"
        ),
        pytest.param(192, 16, 128, 96, 4, torch.float16, False, id="grouped_even_fp16"),
        pytest.param(
            255, 8, 160, 112, 2, torch.bfloat16, True, id="grouped_bf16_router_in"
        ),
    ],
)
def test_fused_moe_triton_correctness(
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
    apply_router_weight_on_input: bool,
) -> None:
    torch.manual_seed(
        1000
        + num_tokens
        + 10 * num_experts
        + 100 * hidden_size
        + 1000 * intermediate_size
        + 10000 * topk
        + (1 if dtype == torch.bfloat16 else 0)
        + (2 if apply_router_weight_on_input else 0)
    )
    hidden_states, w1, w2, topk_weights, topk_ids = _make_moe_inputs(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
    )

    actual = _run_triton_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    expected = _ref_fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )

    atol = rtol = 1e-1 if dtype == torch.bfloat16 else 5e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_fused_moe_triton_perf() -> None:
    num_tokens = 512
    num_experts = 32
    hidden_size = 1024
    intermediate_size = 4096
    topk = 8
    dtype = torch.bfloat16
    apply_router_weight_on_input = False
    torch.manual_seed(4321)
    hidden_states, w1, w2, topk_weights, topk_ids = _make_moe_inputs(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        topk,
        dtype,
    )

    element_size = hidden_states.element_size()
    memory_footprint = (
        hidden_states.numel() * element_size
        + w1.numel() * element_size
        + topk_weights.numel() * topk_weights.element_size()
        + topk_ids.numel() * topk_ids.element_size()
        + num_tokens * topk * w1.shape[1] * element_size
        + num_tokens * topk * intermediate_size * element_size
        + w2.numel() * element_size
        + num_tokens * topk * hidden_size * element_size
        + num_tokens * hidden_size * element_size
    )
    compare_memory_kernel_perf(
        torch_impl=lambda: _ref_fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        ),
        old_impl=lambda: _run_triton_fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        ),
        memory_footprint=memory_footprint,
        description=(
            "fused_moe | "
            f"M={num_tokens:4d} | "
            f"E={num_experts:3d} | "
            f"H={hidden_size:4d} | "
            f"I={intermediate_size:5d} | "
            f"topk={topk} | "
        ),
        extra_kwargs={
            "init_stream": False,
            "repetitions": 10,
            "cuda_graph_repetitions": 0,
        },
    )


@pytest.mark.parametrize(
    ("num_tokens", "topk", "hidden_size", "dtype"),
    [
        pytest.param(17, 1, 513, torch.bfloat16, id="single_block"),
        pytest.param(33, 8, 4097, torch.float16, id="multi_block_tail"),
    ],
)
def test_moe_sum_reduce_triton_correctness(
    num_tokens: int,
    topk: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(
        9000
        + num_tokens
        + 10 * topk
        + 100 * hidden_size
        + (1 if dtype == torch.bfloat16 else 0)
    )
    x = torch.randn(
        num_tokens,
        topk,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    actual = torch.empty((num_tokens, hidden_size), device="cuda", dtype=dtype)
    moe_new.moe_sum_reduce_triton(x, actual)
    expected = x.sum(dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


def test_moe_sum_reduce_triton_perf() -> None:
    num_tokens, topk, hidden_size = 4096, 8, 2048
    x = torch.randn(num_tokens, topk, hidden_size, device="cuda", dtype=torch.float16)
    out = torch.empty((num_tokens, hidden_size), device="cuda", dtype=torch.float16)
    memory_footprint = x.numel() * x.element_size() + out.numel() * out.element_size()
    compare_memory_kernel_perf(
        torch_impl=lambda: x.sum(dim=1),
        old_impl=lambda: moe_new.moe_sum_reduce_triton(x, out),
        memory_footprint=memory_footprint,
        description=(
            f"moe_sum_reduce | M={num_tokens:5d} | topk={topk} | H={hidden_size:4d} | "
        ),
        extra_kwargs={
            "init_stream": False,
            "repetitions": 10,
            "cuda_graph_repetitions": 0,
        },
    )
