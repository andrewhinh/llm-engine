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

| Status | Task                                                                                      | Target                                                                                       | References                                                                                                                                                                                                            |
| ------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅     | scaffold crate/module layout and deps                                                     | `crates/llm/src/lib.rs`, `crates/llm/Cargo.toml`                                             | `nano-vllm-main/nanovllm/__init__.py`, `vllm.rs-main/src/lib.rs`                                                                                                                                                      |
| ✅     | engine and sampling config types                                                          | `crates/llm/src/utils/config.rs`                                                             | `nano-vllm-main/nanovllm/config.py`, `nano-vllm-main/nanovllm/sampling_params.py`                                                                                                                                     |
| ✅     | tokenizer/model bootstrap (hf config, eos, encode/decode)                                 | `crates/llm/src/utils/tokenizer.rs`, `crates/llm/src/core/engine.rs`                         | `nano-vllm-main/nanovllm/engine/llm_engine.py`, `mini-sglang-main/python/minisgl/tokenizer/server.py`, `vllm.rs-main/src/core/engine.rs`                                                                              |
| ✅     | request/sequence state machine (WAITING/RUNNING/FINISHED, block table, token bookkeeping) | `crates/llm/src/core/sequence.rs`                                                            | `nano-vllm-main/nanovllm/engine/sequence.py`, `vllm.rs-main/src/core/sequence.rs`                                                                                                                                     |
| ✅     | KV allocator planning (num blocks from free mem, config caps)                             | `crates/llm/src/utils/kvcache_allocator.rs`                                                  | `nano-vllm-main/nanovllm/engine/model_runner.py`, `vllm.rs-main/src/utils/kvcache_allocator.rs`                                                                                                                       |
| ✅     | block manager and hash prefix cache (allocate/deallocate/may_append/reuse)                | `crates/llm/src/core/block_manager.rs`, `crates/llm/src/core/prefix_cache/{mod.rs,hash.rs}`  | `nano-vllm-main/nanovllm/engine/block_manager.py`, `vllm.rs-main/src/core/{block_manager.rs,prefix_cache.rs}`                                                                                                         |
| ✅     | scheduler core (prefill first, decode next, continuous batching, preempt on KV pressure)  | `crates/llm/src/core/scheduler.rs`                                                           | `nano-vllm-main/nanovllm/engine/scheduler.py`, `vllm.rs-main/src/core/scheduler.rs`                                                                                                                                   |
| ✅     | core layers, embeddings/lm-head, and qwen3 decoder path                                   | `crates/llm/src/models/layers/*`, `crates/llm/src/models/qwen3.rs`                           | `nano-vllm-main/nanovllm/layers/*.py`, `nano-vllm-main/nanovllm/models/qwen3.py`, `mini-sglang-main/python/minisgl/models/qwen3.py`, `vllm.rs-main/src/models/{layers/{linear.rs,attention.rs},qwen3.rs}`             |
| ✅     | weight loader and sharded weight hooks                                                    | `crates/llm/src/utils/loader.rs`                                                             | `nano-vllm-main/nanovllm/utils/loader.py`, `mini-sglang-main/python/minisgl/models/weight.py`, `vllm.rs-main/src/utils/downloader.rs`                                                                                 |
| ✅     | model runner prefill/decode prep, forward, sample                                         | `crates/llm/src/core/runner.rs`, `crates/llm/src/runner/sampler.rs`                          | `nano-vllm-main/nanovllm/engine/model_runner.py`, `nano-vllm-main/nanovllm/layers/sampler.py`, `mini-sglang-main/python/minisgl/engine/engine.py`, `vllm.rs-main/src/core/runner.rs`                                  |
| ✅     | engine loop, sync generate API, and throughput logs                                       | `crates/llm/src/core/engine.rs`, `crates/llm/src/api.rs`                                     | `nano-vllm-main/nanovllm/engine/llm_engine.py`, `nano-vllm-main/nanovllm/llm.py`                                                                                                                                      |
| ✅     | TP baseline (single-node multi-GPU, NCCL/gloo bootstrap, col/row-parallel ops)            | `crates/llm/src/models/layers/distributed.rs`, `crates/llm/src/core/runner.rs`               | `nano-vllm-main/nanovllm/layers/linear.py`, `nano-vllm-main/nanovllm/engine/model_runner.py`, `vllm.rs-main/src/models/layers/distributed.rs`                                                                         |
| ✅     | optional CUDA graph decode path (capture/replay by bs buckets)                            | `crates/llm/src/utils/graph.rs`, `crates/llm/src/core/runner.rs`                             | `nano-vllm-main/nanovllm/engine/model_runner.py`, `mini-sglang-main/python/minisgl/engine/graph.py`, `vllm.rs-main/src/utils/graph.rs`                                                                                |
| ✅     | stream generate API and request cancel plumbing                                           | `crates/llm/src/core/engine.rs`, `crates/llm/src/server/api.rs`                              | `mini-sglang-main/python/minisgl/server/api_server.py`, `vllm.rs-main/src/server/streaming.rs`                                                                                                                        |
| ✅     | OpenAI-compatible HTTP server and SSE (`/v1`, `/v1/models`, `/v1/chat/completions`)       | `crates/llm/src/server/{mod.rs,api.rs,streaming.rs}`                                         | `mini-sglang-main/python/minisgl/server/api_server.py`, `vllm.rs-main/src/server/{server.rs,streaming.rs}`                                                                                                            |
| ✅     | typed IPC message schema (frontend/tokenizer/backend/scheduler)                           | `crates/llm/src/ipc/messages.rs`                                                             | `mini-sglang-main/python/minisgl/message/{tokenizer.py,backend.py,frontend.py}`, `vllm.rs-main/src/runner/mod.rs`                                                                                                     |
| ✅     | launcher process model (frontend, tokenizer/detokenizer, per-rank scheduler)              | `crates/llm/src/server/launch.rs`                                                            | `mini-sglang-main/python/minisgl/server/launch.py`, `vllm.rs-main/src/runner/runner.rs`                                                                                                                               |
| ✅     | tokenizer/detokenizer workers and batching bridge                                         | `crates/llm/src/tokenizer/{worker.rs,tokenize.rs,detokenize.rs}`                             | `mini-sglang-main/python/minisgl/tokenizer/{server.py,tokenize.py,detokenize.py}`                                                                                                                                     |
| ✅     | scheduler IO and rank0 routing/broadcast between TP workers                               | `crates/llm/src/scheduler/{io.rs,scheduler.rs}`                                              | `mini-sglang-main/python/minisgl/scheduler/{io.py,scheduler.py}`, `vllm.rs-main/src/runner/mod.rs`                                                                                                                    |
| ✅     | radix-tree prefix cache (keep hash cache fallback)                                        | `crates/llm/src/core/prefix_cache/{mod.rs,radix.rs,hash.rs}`                                 | `mini-sglang-main/python/minisgl/kvcache/radix_manager.py`, `mini-sglang-main/python/minisgl/scheduler/cache.py`, `vllm.rs-main/src/core/prefix_cache.rs`                                                             |
| ✅     | chunked prefill manager and prefill token budget controls                                 | `crates/llm/src/scheduler/prefill.rs`                                                        | `mini-sglang-main/python/minisgl/scheduler/prefill.py`, `mini-sglang-main/docs/features.md`, `vllm.rs-main/src/core/scheduler.rs`                                                                                     |
| ✅     | overlap scheduling (CPU metadata/scheduling overlaps GPU execution)                       | `crates/llm/src/scheduler/scheduler.rs`                                                      | `mini-sglang-main/python/minisgl/scheduler/scheduler.py`, `mini-sglang-main/docs/features.md`                                                                                                                         |
| ✅     | attention backend switch (eager/fa/flashinfer, prefill/decode split)                      | `crates/llm/src/attention/{base.rs,eager.rs,fa.rs,fi.rs}`                                    | `mini-sglang-main/python/minisgl/attention/{base.py,fa.py,fi.py}`, `vllm.rs-main/src/models/layers/attention.rs`                                                                                                      |
| ⬜     | shell mode and high-level client wrapper                                                  | `crates/llm/src/server/shell.rs`, `crates/llm/src/server/launch.rs`, `crates/llm/src/api.rs` | `mini-sglang-main/docs/features.md`, `mini-sglang-main/python/minisgl/{shell.py,server/api_server.py}`                                                                                                                |
| ⬜     | benchmark/examples/UML                                                                    | `crates/llm/benches/*`, `crates/llm/examples/*`, `README.md`, `docs/*`                       | `nano-vllm-main/bench.py`, `mini-sglang-main/benchmark/{offline/bench.py,online/bench_qwen.py}`, `mini-sglang-main/python/minisgl/benchmark/{perf.py,client.py}`, `mini-sglang-main/docs/{features.md,structures.md}` |

## Credit

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vllm.rs](https://github.com/guoqingbao/vllm.rs)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
