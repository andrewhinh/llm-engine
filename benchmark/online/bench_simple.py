import asyncio
import random
from typing import List

from minisgl.app import MINUTES, app, image


@app.function(image=image, timeout=10 * MINUTES)
async def run_online_benchmark(server_url: str):
    from openai import AsyncOpenAI as OpenAI
    from transformers import AutoTokenizer

    from minisgl.benchmark.client import (
        benchmark_one_batch,
        generate_prompt,
        get_model_name,
        process_benchmark_results,
    )
    from minisgl.utils import init_logger

    logger = init_logger(__name__)
    random.seed(42)

    async def generate_task(max_bs: int) -> List[str]:
        result = []
        for _ in range(max_bs):
            length = random.randint(1, MAX_INPUT)
            message = generate_prompt(tokenizer, length)
            result.append(message)
            await asyncio.sleep(0)
        return result

    TEST_BS = [64]
    MAX_INPUT = 8192

    async with OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="") as client:
        model = await get_model_name(client)
        tokenizer = AutoTokenizer.from_pretrained(model)

        gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
        msgs = await gen_task
        output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
        logger.info(f"Generated {len(msgs)} test messages")

        logger.info("Running benchmark...")
        for batch_size in TEST_BS:
            results = await benchmark_one_batch(
                client, msgs[:batch_size], output_lengths[:batch_size], model
            )
            process_benchmark_results(results)
    logger.info("Benchmark completed.")


@app.local_entrypoint()
def main(server_url: str = ""):
    if not server_url:
        from modal import Cls

        model_server = Cls.from_name("mini-sglang", "ModelServer")
        server_url = model_server().serve.get_web_url()

    run_online_benchmark.remote(server_url)
