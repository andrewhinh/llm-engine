# llm-engine

A pure Python implementation of Mini-SGLang using Cute-DSL.

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
```

### Commands

For the shell and server clients, you can specify the following environment variables:

- `NNODES`: number of nodes (1..4)
- `N_GPU`: number of GPUs per node (1..8)
- `GPU_TYPE`: GPU type
- `RDMA`: whether to use RDMA (0 or 1)

For multi-node deployment on Hopper and Blackwell chips:

1. Your Modal workspace must have RDMA support.
2. You must pass `--rdma` to the commands below.

Run an interactive shell client:

```bash
modal run -i -m llmeng.shell
```

Serve an OpenAI-compatible API server:

```bash
modal serve llmeng/server.py
```

Run offline benchmarks:

```bash
modal run benchmark/offline/bench.py
modal run benchmark/offline/bench_wildchat.py
```

Run online benchmarks:

1. Deploy the server:

```bash
N_GPU=4 GPU_TYPE=h200 modal deploy llmeng/app.py
```

2. Run the benchmarks:

```bash
modal run benchmark/online/bench_qwen.py
modal run benchmark/online/bench_simple.py
```

## Roadmap

- [x] port mini-sglang to Modal
- [x] replace nccl with penny
- [ ] rewrite C++/CUDA/Triton in Cute-DSL
- [ ] add speculative speculative decoding (SSD)

## Credit

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [Cute-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html), Simon Vietner's blog posts: [1](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/)
- [Penny](https://github.com/SzymonOzog/Penny), worklogs [1](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html), [2](https://szymonozog.github.io/posts/2025-10-26-Penny-worklog-2.html), [3](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html)
- [SSD](https://github.com/tanishqkumar/ssd), [paper](https://arxiv.org/pdf/2603.03251)
- Tristan Hume's [blog post on profiling](https://thume.ca/2023/12/02/tracing-methods/)
