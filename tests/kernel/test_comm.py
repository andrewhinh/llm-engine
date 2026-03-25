import multiprocessing as mp
import os
import random
import time
import traceback

import pytest
import torch
import torch.distributed as dist
from tqdm import tqdm

from llmeng.distributed.info import set_tp_info
from llmeng.utils import init_logger, load_module

logger = init_logger(__name__)
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >=2 CUDA devices"
)


@torch.no_grad()
def run(tp_size: int, tp_rank: int, port: int, queue: mp.Queue):
    torch.cuda.set_device(tp_rank)
    torch.cuda.set_stream(torch.cuda.Stream(tp_rank))
    stream = torch.cuda.current_stream()
    set_tp_info(tp_rank, tp_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    old_comm = None
    new_comm = None
    try:
        dist.init_process_group(
            world_size=tp_size,
            rank=tp_rank,
            backend="gloo",
        )

        tp_cpu_group = dist.group.WORLD
        assert tp_cpu_group is not None, "CPU group should not be None"
        dtype = torch.float16

        K = 512
        USE_SYMM = 0
        max_size_bytes = 8192 * K * dtype.itemsize if USE_SYMM else 0

        old_module = load_module(True, "old", "pynccl")
        new_module = load_module(False, "new", "nccl")
        if not hasattr(old_module.ENV, "PYNCCL_MAX_BUFFER_SIZE"):
            old_module.ENV.PYNCCL_MAX_BUFFER_SIZE = old_module.ENV.NCCL_MAX_BUFFER_SIZE

        old_comm = old_module.init_pynccl(
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_cpu_group=tp_cpu_group,
            max_size_bytes=max_size_bytes,
        )
        new_comm = new_module.init_nccl(
            local_rank=tp_rank,
            local_size=tp_size,
            global_rank=tp_rank,
            global_size=tp_size,
            tp_cpu_group=tp_cpu_group,
            max_size_bytes=max_size_bytes,
        )

        def bench_performance(name, f, use_graph=False):
            import gc

            gc.collect()
            gc.disable()

            N = 1024
            M = 16
            x = torch.zeros(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
            f(x)
            f(x)

            pbar = tqdm(
                list(range(N)), desc="Capturing cuda graph", disable=tp_rank > 0
            )

            torch.cuda.synchronize()
            if use_graph:
                g = torch.cuda.CUDAGraph()
                graph = torch.cuda.graph(g)
                with graph:
                    for _ in pbar:
                        f(x)
                cur_stream = graph.capture_stream
            else:
                nonlocal stream
                f(x)
                f(x)
                cur_stream = stream

            tic = torch.cuda.Event(enable_timing=True)
            toc = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(cur_stream):
                tic.record(cur_stream)
                if use_graph:
                    for _ in range(M):
                        g.replay()
                else:
                    for _ in range(M):
                        for _ in pbar:
                            f(x)
                toc.record(cur_stream)
            gc.enable()
            toc.synchronize()
            elapsed_time = tic.elapsed_time(toc)
            avg_time = elapsed_time * 1000 / (M * N)
            logger.info(
                f"Rank {tp_rank} {name} all-reduce avg time: {avg_time: .4f} us"
            )
            bandwidth = (8192 * K * dtype.itemsize) / (avg_time * 1e3)
            logger.info(
                f"Rank {tp_rank} {name} all-reduce bandwidth: {bandwidth:.2f} GB/s"
            )
            mem_usage = torch.cuda.memory_allocated() / (1024 * 1024)
            logger.info(f"Rank {tp_rank} {name} memory usage: {mem_usage:.2f} MB")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def test_correctness(name, f):
            N = 4
            x = torch.ones(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
            for _ in range(N):
                f(x)
            ans = pow(tp_size, N)
            y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")

            assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

            x = torch.full((8192 * K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
            if tp_rank == 0:
                torch.cuda.synchronize()
                time.sleep(1)
            f(x)
            ans = (tp_size * (tp_size - 1)) // 2
            y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")
            assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

            x = torch.cat(
                [
                    torch.zeros(
                        (8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"
                    ),
                    torch.ones((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                ]
            )
            f(x)
            y = torch.cat(
                [
                    torch.zeros(
                        (8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"
                    ),
                    torch.full(
                        (8192 * K // 2,), tp_size, dtype=dtype, device=f"cuda:{tp_rank}"
                    ),
                ]
            )
            assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

            if N % 2 != 0:
                f(x)

            logger.info(f"Correctness check for rank {tp_rank} passed ({name})")

        test_correctness("old", lambda x: old_comm.all_reduce(x, "sum"))
        test_correctness("new", lambda x: new_comm.all_reduce(x, "sum"))
        bench_performance("old", lambda x: old_comm.all_reduce(x, "sum"))
        bench_performance("new", lambda x: new_comm.all_reduce(x, "sum"))
        test_correctness("old", lambda x: old_comm.all_reduce(x, "sum"))
        test_correctness("new", lambda x: new_comm.all_reduce(x, "sum"))

        src = torch.full((K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
        torch.cuda.synchronize()
        old_dst = torch.empty((K * tp_size,), dtype=dtype, device=f"cuda:{tp_rank}")
        new_dst = torch.empty((K * tp_size,), dtype=dtype, device=f"cuda:{tp_rank}")
        old_comm.all_gather(old_dst, src)
        new_comm.all_gather(new_dst, src)
        torch.cuda.synchronize()
        expected = torch.arange(tp_size, dtype=dtype, device=f"cuda:{tp_rank}")
        expected = expected.repeat_interleave(K)
        assert torch.allclose(old_dst, expected), (
            f"Rank {tp_rank} old all-gather failed"
        )
        assert torch.allclose(new_dst, expected), (
            f"Rank {tp_rank} new all-gather failed"
        )
        queue.put(None)
    except Exception:
        queue.put(traceback.format_exc())
    finally:
        for comm in (new_comm, old_comm):
            if comm is None:
                continue
            destroy = getattr(comm, "destroy", None) or getattr(comm, "close", None)
            if callable(destroy):
                destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_nccl_comm() -> None:
    tp_size = 2
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = random.randint(20000, 50000)
    p_list = [
        ctx.Process(target=run, args=(tp_size, rank, port, queue))
        for rank in range(tp_size)
    ]
    try:
        for p in p_list:
            p.start()
        results = [queue.get(timeout=300) for _ in p_list]
        for p in p_list:
            p.join(300)
        for p in p_list:
            assert p.exitcode == 0
        assert results == [None] * tp_size
    finally:
        for p in p_list:
            if p.is_alive():
                p.terminate()
                p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    test_nccl_comm()
