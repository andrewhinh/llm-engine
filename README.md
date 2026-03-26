# llm-engine

A pure Python high-performance LLM inference engine.

![icon](./assets/icon.svg)

## Development

### Installation

- [uv](https://docs.astral.sh/uv)
- [prek](https://prek.j178.dev)

```bash
prek install
```

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install modal==1.3.5
modal setup
modal secret
modal secret create huggingface-secret HF_TOKEN=<your_huggingface_token>
```

### Commands

Serve an OpenAI-compatible API server.
You can specify the following environment variables:

- `NNODES`: # nodes (`1..4`, default=`1`)
- `N_GPU`: # GPUs per node (`1..8`, default=`1`)
- `GPU_TYPE`: GPU type (`l4`, `l40s`, `a100`, `a100-40gb`, `a100-80gb`, `rtx-pro-6000`, `h100/h100!`, `h200`, `b200/b200+`, default=`a100`)
- `RDMA`: whether to use RDMA (`0` or `1`, default=`0`)

```bash
modal serve llmeng/api.py
```

_Note that for multi-node deployment on Hopper and Blackwell chips, your Modal workspace must have RDMA support._

Run the benchmark suite.
Some important parameters:

- `--engine`: engine to benchmark (`llmeng`, `minisgl`, `all`, default=`all`)
- `--model`: model to benchmark (`Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B`, `all`, default=`all`)
- `--data`: data workload (`in_out_128_1024`, `in_out_256_2048`, `in_out_512_512`, `in_out_512_4096`, `in_out_1024_128`, `in_out_1024_1024`, `in_out_2048_256`, `in_out_2048_2048`, `in_out_4096_512`, `all`, default=`in_out_128_1024`)
- `--rate-type`: request workload (`synchronous`, `throughput`, `constant`, `all`, default=`synchronous`)
- `--rate`: only for constant rate type, request rate per second (float, default=`None`)

The results will be saved to `benchmark/results.json` and `benchmark/results.html`.
Work is parallelized across engine/model servers. Within each server, data workloads and rate workloads run sequentially to avoid interference.

```bash
modal run benchmark/main.py
```

Run the tests.
You can specify the following parameters:

- `--target`: a specific test file or test function (e.g., `tests/kernel/test_radix.py::test_fast_compare_key_perf`).
- `--pytest-args`: additional pytest arguments (e.g., `-q`).

```bash
modal run ci.py
```

## Roadmap

- [x] port mini-sglang to Modal
- [ ] ~~replace nccl with penny~~
  - reverted for now due to compatibility issues between nccl4py and nvshmem4py
- [x] rewrite C++/CUDA/Triton in Numba/CuTe-DSL/CuTile
- [x] benchmark against mini-sglang properly
- [ ] add speculative speculative decoding (SSD)

## Credit

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [Penny](https://github.com/SzymonOzog/Penny), worklogs [1](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html), [2](https://szymonozog.github.io/posts/2025-10-26-Penny-worklog-2.html), [3](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html)
- [kernel benchmarking code](https://github.com/ekzhang/kernels)
- Simon Veitner's [blog posts](https://veitner.bearblog.dev/)
- [Getting Memory-Bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
- [cuda-python](https://pypi.org/project/cuda-python/), [nvidia-cutlass-dsl](https://pypi.org/project/nvidia-cutlass-dsl/), [cuda-tile](https://pypi.org/project/cuda-tile/), [nccl4py](https://pypi.org/project/nccl4py/), [nvshmem](https://pypi.org/project/nvshmem4py-cu12/)
- [LLM Almanac](https://modal.com/llm-almanac) and its [code](https://github.com/modal-labs/stopwatch)
- [SSD](https://github.com/tanishqkumar/ssd), [paper](https://arxiv.org/pdf/2603.03251)
