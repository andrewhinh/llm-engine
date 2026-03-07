# mini-sglang

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

Run an interactive shell client:

```bash
modal run -i minisgl/shell.py
```

Serve an OpenAI-compatible API server:

```bash
modal serve minisgl/app.py
```

Run offline benchmarks:

```bash
modal run benchmark/offline/bench.py
modal run benchmark/offline/bench_wildchat.py
```

Run online benchmarks:

1. Deploy the server:

```bash
modal deploy minisgl/app.py
```

2. Run the benchmarks:

```bash
modal run benchmark/online/bench_qwen.py
modal run benchmark/online/bench_simple.py
```

## Roadmap

- [x] port mini-sglang to Modal
  - [ ] clean up code
  - [ ] use bcc and torch profiler for tracing
- [ ] rewrite C/C++/Cuda in Cute-DSL
- [ ] replace nccl with penny
  - [ ] rewrite penny C/C++/Cuda in Cute-DSL
- [ ] add speculative speculative decoding (SSD)

## Credit

- [mini-sglang](https://github.com/sgl-project/mini-sglang)
- [Cute-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html), Simon Vietner's blog posts: [1](https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/)
- [Penny](https://github.com/SzymonOzog/Penny), worklogs [1](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html), [2](https://szymonozog.github.io/posts/2025-10-26-Penny-worklog-2.html), [3](https://szymonozog.github.io/posts/2025-11-11-Penny-worklog-3.html)
- [SSD](https://github.com/tanishqkumar/ssd), [paper](https://arxiv.org/pdf/2603.03251)
- Tristan Hume's [blog post on profiling](https://thume.ca/2023/12/02/tracing-methods/)
