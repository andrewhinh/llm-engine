# sglang-rs

A minimal Rust implementation of SGLang using Candle.

![icon](./assets/icon.svg)

## Development

### Installation

- [rustup](https://rustup.rs)
- [prek](https://prek.j178.dev)

```bash
prek install
```

### Setup

Local:

- [hf cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli#standalone-installer-recommended)

```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
```

To run on Modal:

- [uv](https://docs.astral.sh/uv) for `uvx modal` (or standalone [modal](https://modal.com/docs/guide) CLI)

```bash
uvx modal setup                                                                    # one-time auth setup
GPU_TYPE=H100 GPU_COUNT=1 uvx modal shell scripts/modal_shell.py::dev_shell --pty  # optionally specify GPU_TYPE and GPU_COUNT; then create gpu sandbox, download model, and open shell
```

### Commands

```bash
cargo run
cargo bench
```

## Roadmap

- [ ] port nano-vllm to Rust and Candle
- [ ] OpenAI-compatible API server
- [ ] interactive shell mode
- [ ] hybrid attention backend selection (prefill/decode split)
- [ ] FlashInfer decode path
- [ ] chunked prefill
- [ ] radix cache
- [ ] overlap scheduling
- [ ] tokenizer worker scaling (multi-tokenizer processes)
- [ ] multi-model support (Llama/Qwen2/Qwen3 dense)
- [ ] MoE backend integration
- [ ] Qwen3-MoE support

## Credit

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vllm.rs](https://github.com/guoqingbao/vllm.rs)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)

```

```
