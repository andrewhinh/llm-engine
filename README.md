# llm-engine

A minimal LLM inference engine in Rust.

![icon](./assets/icon.jpg)

## Development

### Installation

- [rustup](https://rustup.rs/)
- [prek](https://prek.j178.dev/installation/)

```bash
prek install
```

### Commands

```bash
cargo test
cargo run
```

## Roadmap

- [x] crate layout and deps
- [ ] config and sampling param types
- [ ] tokenizer wrapper
- [ ] sequence/request state machine
- [ ] kv cache allocator and block manager
- [ ] hash-based prefix cache
- [ ] scheduler (prefill/decode, continuous batching)
- [ ] core model layers (linear, rmsnorm, rope, attention)
- [ ] Qwen-3 model path
- [ ] model runner (prefill/decode prep and sampling)
- [ ] engine loop and sync/stream generation API
- [ ] preemption when KV full
- [ ] tensor parallel baseline (single node, multi-GPU, NCCL path)
- [ ] optional CUDA graph decode path

- [ ] openAI-compatible HTTP server and SSE streaming
- [ ] process model (launcher and frontend/tokenizer/scheduler worker roles) with typed IPC
- [ ] radix-tree prefix cache (keep hash cache fallback)
- [ ] chunked prefill for long prompts
- [ ] overlap scheduling (CPU scheduling overlaps GPU work)
- [ ] attention backend switch (eager/flash-attn/flashinfer paths)
- [ ] session/request cancel and usage stats endpoint

## Credit

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vllm.rs](https://github.com/guoqingbao/vllm.rs)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
