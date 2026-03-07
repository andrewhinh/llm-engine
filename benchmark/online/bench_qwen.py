import random
from pathlib import Path

from minisgl.modal import MINUTES, app, image

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


def download_qwen_trace(url: str, logger) -> str:
    file_path = Path(__file__).resolve().parent / "qwen_traceA_blksz_16.jsonl"
    if not file_path.exists():
        import urllib.request

        logger.info(f"Downloading trace from {url} to {file_path}...")
        urllib.request.urlretrieve(url, file_path)
        logger.info("Download completed.")
    return str(file_path)


@app.function(image=image, timeout=10 * MINUTES)
async def run_online_benchmark(server_url: str):
    from openai import AsyncOpenAI as OpenAI
    from transformers import AutoTokenizer

    from minisgl.benchmark.client import (
        benchmark_trace,
        get_model_name,
        process_benchmark_results,
        read_qwen_trace,
        scale_traces,
    )
    from minisgl.utils import init_logger

    logger = init_logger(__name__)
    random.seed(42)

    N = 1000
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]

    async with OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)
        traces = read_qwen_trace(
            download_qwen_trace(URL, logger), tokenizer, n=N, dummy=True
        )
        logger.info(f"Start benchmarking with {N} requests using model {model}...")
        for scale in SCALES:
            scaled_traces = scale_traces(traces, scale)
            results = await benchmark_trace(client, scaled_traces, model)
            process_benchmark_results(results)
        logger.info("Benchmarking completed.")


@app.local_entrypoint()
def main(server_url: str = ""):
    if not server_url:
        from modal import Cls

        model_server = Cls.from_name("mini-sglang", "ModelServer")
        server_url = model_server().serve.get_web_url()

    run_online_benchmark.remote(server_url)
