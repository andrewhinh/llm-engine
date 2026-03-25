from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from llmeng.utils import init_logger

logger = init_logger(__name__)


def perf_cuda(
    f: Callable[[], Any],
    *,
    init_stream: bool = True,
    repetitions: int = 10,
    cuda_graph_repetitions: int | None = 10,
) -> float:
    import torch

    assert repetitions > 0
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if init_stream:
        stream = torch.cuda.Stream()
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        f()
        if N := cuda_graph_repetitions:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(N):
                    f()
            replay = g.replay
            del g
        else:
            replay = f
            N = 1

        torch.cuda.synchronize()

        replay()
        tic.record()
        for _ in range(repetitions):
            replay()
        toc.record()
        toc.synchronize()
        dur = tic.elapsed_time(toc)
        return dur / (N * repetitions)


def compare_memory_kernel_perf(
    *,
    torch_impl: Callable[[], Any],
    old_impl: Callable[[], Any],
    new_impl: Callable[[], Any] | None = None,
    cutile_impl: Callable[[], Any] | None = None,
    memory_footprint: int,
    description: str = " ",
    extra_kwargs: Dict[str, Any] | None = None,
    need_latency: bool = True,
) -> Tuple[float, float, float | None, float | None]:
    extra_kwargs = extra_kwargs or {}

    def _measure(fn: Callable[[], Any], label: str) -> tuple[float, str]:
        dur = perf_cuda(fn, **extra_kwargs)
        value = memory_footprint / (dur * 1e6)
        latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
        return value, f"{label}: {latency_msg}{value:8.3f} GB/s"

    value_0, message_0 = _measure(torch_impl, "Torch Impl")
    value_1, message_1 = _measure(old_impl, "Old Impl")
    message_parts = [message_0, message_1]

    value_2 = None
    if new_impl is not None:
        value_2, message_2 = _measure(new_impl, "New Impl")
        message_parts.append(message_2)

    value_3 = None
    if cutile_impl is not None:
        value_3, message_3 = _measure(cutile_impl, "cuTile Impl")
        message_parts.append(message_3)

    logger.info(f"{description}{' | '.join(message_parts)}")
    return value_0, value_1, value_2, value_3


def perf_host(
    f: Callable[[], Any],
    *,
    warmup: int = 25,
    repetitions: int = 250,
) -> float:
    assert repetitions > 0
    for _ in range(warmup):
        f()
    tic = time.perf_counter_ns()
    for _ in range(repetitions):
        f()
    toc = time.perf_counter_ns()
    return (toc - tic) / (repetitions * 1e6)


def compare_latency_kernel_perf(
    *,
    torch_impl: Callable[[], Any] | None = None,
    old_impl: Callable[[], Any],
    new_impl: Callable[[], Any],
    description: str = " ",
    warmup: int = 25,
    repetitions: int = 250,
) -> Tuple[float, float, float]:
    message_parts: list[str] = []
    if torch_impl is None:
        dur_0 = float("nan")
    else:
        dur_0 = perf_host(torch_impl, warmup=warmup, repetitions=repetitions)
        message_parts.append(f"Torch Impl: {dur_0:8.3f} ms")
    dur_1 = perf_host(old_impl, warmup=warmup, repetitions=repetitions)
    dur_2 = perf_host(new_impl, warmup=warmup, repetitions=repetitions)
    message_parts.append(f"Old Impl: {dur_1:8.3f} ms")
    message_parts.append(f"New Impl: {dur_2:8.3f} ms")
    logger.info(f"{description}{' | '.join(message_parts)}")
    return dur_0, dur_1, dur_2


def load_module(from_old: bool, package_name: str, module_root_name: str):
    if from_old:
        root = Path("/root/bench_kernel_old")
    else:
        root = Path("/root/bench_kernel_new")
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = f"{package_name}.{module_root_name}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            module_name, root / f"{module_root_name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {module_root_name} module from {root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module
