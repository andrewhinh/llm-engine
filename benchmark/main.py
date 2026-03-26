from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any

from benchmark.constants import CONSTANT_RATES, DATA_TYPES, MODEL_MATRIX
from benchmark.guide import GuideLLM
from benchmark.plot import convert_json_to_html
from llmeng.api import create_model_server_class
from llmeng.modal import app, get_startup_metrics_dict

BenchmarkRate = int | float | list[int | float] | None
BenchmarkJob = tuple[str, BenchmarkRate]
BenchmarkResult = dict[str, Any]

SERVER_CLASSES = {
    (
        engine,
        model_name,
        config["nnode"],
        config["gpu_type"],
        config["n_gpu"],
    ): create_model_server_class(
        engine,
        model_name,
        config["nnode"],
        config["gpu_type"],
        config["n_gpu"],
        f"_{engine}_{model_name.replace('/', '--')}",
    )
    for engine in ("llmeng", "minisgl")
    for model_name, config in MODEL_MATRIX.items()
}


async def run_server_benchmarks(
    *,
    endpoint: str,
    engine: str,
    model_name: str,
    gpu_type: str,
    n_gpu: int,
    benchmark_jobs: Sequence[BenchmarkJob],
    server_id: str,
    startup_metrics_dict_id: str,
    duration: float,
    data_types: dict[str, str],
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for data_type in data_types.values():
        for benchmark_rate_type, benchmark_rate in benchmark_jobs:
            batch = await GuideLLM().run_benchmark.remote.aio(
                endpoint=endpoint,
                model=model_name,
                rate_type=benchmark_rate_type,
                data=data_type,
                duration=duration,
                client_config={
                    "use_chat_completions": True,
                    "extra_body": {
                        "ignore_eos": True,
                    },
                },
                rate=benchmark_rate,
                server_id=server_id,
                startup_metrics_dict_id=startup_metrics_dict_id,
            )
            results.extend(
                [
                    {
                        **item,
                        "engine": engine,
                        "model": model_name,
                        "gpu_type": gpu_type,
                        "n_gpu": n_gpu,
                        "data": data_type,
                    }
                    for item in batch
                ]
            )
    return results


@app.local_entrypoint()
async def main(
    engine: str = "all",
    model: str = "all",
    data: str = "128;1024",
    rate_type: str = "synchronous",
    rate: int | float | None = None,
    duration: float = 120.0,
    results_path: str = "benchmark/results.json",
    html_path: str = "benchmark/results.html",
) -> list[BenchmarkResult]:
    engines = ["llmeng", "minisgl"]
    if engine != "all":
        assert engine in engines, f"Engine {engine} not found in engines"
        engines = [engine]

    models = MODEL_MATRIX
    if model != "all":
        assert model in models, f"Model {model} not found in models"
        models = {model: models[model]}

    data_types = DATA_TYPES
    if data != "all":
        assert data in DATA_TYPES, f"Data {data} not found in DATA_TYPES"
        data_types = {data: DATA_TYPES[data]}

    rate_types = ["constant", "synchronous", "throughput"]
    if rate_type != "all":
        assert rate_type in rate_types, f"Rate type {rate_type} not found in rate_types"

    assert results_path is not None, "Results path must be provided"
    assert html_path is not None, "HTML path must be provided"

    startup_metrics_dict = await get_startup_metrics_dict().hydrate.aio()
    startup_metrics_dict_id = startup_metrics_dict.object_id

    calls: list[Coroutine[Any, Any, list[BenchmarkResult]]] = []
    for engine in engines:
        for model_name, model_config in models.items():
            server_class = SERVER_CLASSES[
                (
                    engine,
                    model_name,
                    model_config["nnode"],
                    model_config["gpu_type"],
                    model_config["n_gpu"],
                )
            ]
            server_id = uuid.uuid4().hex[:4]
            server = server_class(
                model_path=model_name,
                server_id=server_id,
                startup_metrics_dict_id=startup_metrics_dict_id,
                **model_config["server_kwargs"],
            )
            endpoint = f"{await server.serve.get_web_url.aio()}/v1"
            constant_rates = CONSTANT_RATES.get(
                (model_config["gpu_type"], model_config["n_gpu"])
            )

            if rate_type == "all":
                assert constant_rates is not None, (
                    "Missing constant-rate config for "
                    f"{(model_config['gpu_type'], model_config['n_gpu'])}"
                )
                benchmark_jobs = [
                    ("synchronous", None),
                    ("throughput", None),
                    ("constant", constant_rates),
                ]
            elif rate_type == "constant":
                if rate is not None:
                    benchmark_jobs = [(rate_type, [rate])]
                else:
                    assert constant_rates is not None, (
                        "Missing constant-rate config for "
                        f"{(model_config['gpu_type'], model_config['n_gpu'])}"
                    )
                    benchmark_jobs = [(rate_type, constant_rates)]
            else:
                assert rate is None, (
                    "Rate only applies to constant benchmarks in this entrypoint"
                )
                benchmark_jobs = [(rate_type, None)]

            calls.append(
                run_server_benchmarks(
                    endpoint=endpoint,
                    engine=engine,
                    model_name=model_name,
                    gpu_type=model_config["gpu_type"],
                    n_gpu=model_config["n_gpu"],
                    benchmark_jobs=benchmark_jobs,
                    server_id=server_id,
                    startup_metrics_dict_id=startup_metrics_dict_id,
                    duration=duration,
                    data_types=data_types,
                )
            )

    results = [item for batch in await asyncio.gather(*calls) for item in batch]
    with Path(results_path).open("w") as f:
        json.dump(results, f, indent=2)

    convert_json_to_html(results, Path(html_path))

    return results
