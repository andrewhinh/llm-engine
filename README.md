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

| Status | Task                                                                                      | Target                                                                                  | References                                                                                                                                                                           |
| ------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ✅     | scaffold crate/module layout and deps                                                     | `crates/llm/src/lib.rs`, `crates/llm/Cargo.toml`                                        | `nano-vllm-main/nanovllm/__init__.py`, `vllm.rs-main/src/lib.rs`                                                                                                                     |
| ✅     | engine and sampling config types                                                          | `crates/llm/src/utils/config.rs`                                                        | `nano-vllm-main/nanovllm/config.py`, `nano-vllm-main/nanovllm/sampling_params.py`, `vllm.rs-main/src/utils/config.rs`                                                                |
| ✅     | tokenizer/model bootstrap (hf config, eos, encode/decode)                                 | `crates/llm/src/utils/tokenizer.rs`, `crates/llm/src/core/engine.rs`                    | `nano-vllm-main/nanovllm/engine/llm_engine.py`, `vllm.rs-main/src/core/engine.rs`, `mini-sglang-main/python/minisgl/tokenizer/server.py`                                             |
| ✅     | request/sequence state machine (WAITING/RUNNING/FINISHED, block table, token bookkeeping) | `crates/llm/src/core/sequence.rs`                                                       | `nano-vllm-main/nanovllm/engine/sequence.py`, `vllm.rs-main/src/core/sequence.rs`, `mini-sglang-main/python/minisgl/core.py`                                                         |
| ✅     | KV allocator planning (num blocks from free mem, config caps)                             | `crates/llm/src/utils/kvcache_allocator.rs`                                             | `nano-vllm-main/nanovllm/engine/model_runner.py`, `vllm.rs-main/src/utils/kvcache_allocator.rs`, `mini-sglang-main/python/minisgl/engine/engine.py`                                  |
| ✅     | block manager and hash prefix cache (allocate/deallocate/may_append/reuse)                | `crates/llm/src/core/block_manager.rs`, `crates/llm/src/core/prefix_cache_hash.rs`      | `nano-vllm-main/nanovllm/engine/block_manager.py`, `vllm.rs-main/src/core/block_manager.rs`, `vllm.rs-main/src/core/prefix_cache.rs`                                                 |
| ✅     | scheduler core (prefill first, decode next, continuous batching, preempt on KV pressure)  | `crates/llm/src/core/scheduler.rs`                                                      | `nano-vllm-main/nanovllm/engine/scheduler.py`, `vllm.rs-main/src/core/scheduler.rs`, `mini-sglang-main/python/minisgl/scheduler/scheduler.py`                                        |
| ⬜     | core layers (linear/rmsnorm/rope/attention and kv write path)                             | `crates/llm/src/models/layers/{linear.rs,norm.rs,rotary.rs,attention.rs,mod.rs}`        | `nano-vllm-main/nanovllm/layers/linear.py`, `nano-vllm-main/nanovllm/layers/attention.py`, `vllm.rs-main/src/models/layers/linear.rs`, `vllm.rs-main/src/models/layers/attention.rs` |
| ⬜     | embeddings/lm-head and runtime attention context metadata                                 | `crates/llm/src/models/layers/{embed_head.rs,context.rs}`                               | `nano-vllm-main/nanovllm/layers/embed_head.py`, `nano-vllm-main/nanovllm/utils/context.py`, `mini-sglang-main/python/minisgl/core.py`                                                |
| ⬜     | Qwen3 model path (decoder layers and packed proj map)                                     | `crates/llm/src/models/qwen3.rs`, `crates/llm/src/models/mod.rs`                        | `nano-vllm-main/nanovllm/models/qwen3.py`, `mini-sglang-main/python/minisgl/models/qwen3.py`, `vllm.rs-main/src/models/qwen3.rs`                                                     |
| ⬜     | weight loader and sharded weight hooks                                                    | `crates/llm/src/utils/loader.rs`                                                        | `nano-vllm-main/nanovllm/utils/loader.py`, `mini-sglang-main/python/minisgl/models/weight.py`, `vllm.rs-main/src/utils/downloader.rs`                                                |
| ⬜     | sampler (temp/topk/topp and penalties API shape)                                          | `crates/llm/src/runner/sampler.rs`                                                      | `nano-vllm-main/nanovllm/layers/sampler.py`, `mini-sglang-main/python/minisgl/engine/sample.py`, `vllm.rs-main/src/utils/logits_processor.rs`                                        |
| ⬜     | model runner prefill/decode prep, forward, sample                                         | `crates/llm/src/core/runner.rs`                                                         | `nano-vllm-main/nanovllm/engine/model_runner.py`, `vllm.rs-main/src/core/runner.rs`, `mini-sglang-main/python/minisgl/engine/engine.py`                                              |
| ⬜     | engine loop, sync generate API, throughput logs                                           | `crates/llm/src/core/engine.rs`, `crates/llm/src/api.rs`                                | `nano-vllm-main/nanovllm/engine/llm_engine.py`, `nano-vllm-main/nanovllm/llm.py`, `vllm.rs-main/src/core/engine.rs`                                                                  |
| ⬜     | stream generate API and request/session cancel plumbing                                   | `crates/llm/src/core/engine.rs`                                                         | `vllm.rs-main/src/core/engine.rs`, `vllm.rs-main/src/server/streaming.rs`, `mini-sglang-main/python/minisgl/server/api_server.py`                                                    |
| ⬜     | TP baseline (single-node multi-GPU, NCCL/gloo bootstrap, col/row-parallel ops)            | `crates/llm/src/models/layers/distributed.rs`, `crates/llm/src/core/runner.rs`          | `nano-vllm-main/nanovllm/layers/linear.py`, `nano-vllm-main/nanovllm/engine/model_runner.py`, `vllm.rs-main/src/models/layers/distributed.rs`                                        |
| ⬜     | optional CUDA graph decode path (capture and replay by bs buckets)                        | `crates/llm/src/utils/graph.rs`, `crates/llm/src/core/runner.rs`                        | `nano-vllm-main/nanovllm/engine/model_runner.py`, `mini-sglang-main/python/minisgl/engine/graph.py`, `vllm.rs-main/src/utils/graph.rs`                                               |
| ⬜     | OpenAI-compatible HTTP server and SSE stream                                              | `crates/llm/src/server/{mod.rs,api.rs,streaming.rs}`                                    | `mini-sglang-main/python/minisgl/server/api_server.py`, `vllm.rs-main/src/server/server.rs`, `vllm.rs-main/src/server/streaming.rs`                                                  |
| ⬜     | typed IPC message schema (frontend/tokenizer/backend/scheduler)                           | `crates/llm/src/ipc/messages.rs`                                                        | `mini-sglang-main/python/minisgl/message/{tokenizer.py,backend.py,frontend.py}`, `vllm.rs-main/src/runner/mod.rs`                                                                    |
| ⬜     | launcher process model (frontend, tokenizer/detokenizer, per-rank scheduler)              | `crates/llm/src/server/launch.rs`                                                       | `mini-sglang-main/python/minisgl/server/launch.py`, `vllm.rs-main/src/runner/runner.rs`                                                                                              |
| ⬜     | tokenizer/detokenizer workers and batching bridge                                         | `crates/llm/src/tokenizer/{worker.rs,tokenize.rs,detokenize.rs}`                        | `mini-sglang-main/python/minisgl/tokenizer/server.py`, `mini-sglang-main/python/minisgl/tokenizer/tokenize.py`                                                                       |
| ⬜     | scheduler IO and rank0 routing/broadcast between TP workers                               | `crates/llm/src/scheduler/{io.rs,scheduler.rs}`                                         | `mini-sglang-main/python/minisgl/scheduler/{io.py,scheduler.py}`, `vllm.rs-main/src/runner/mod.rs`                                                                                   |
| ⬜     | radix-tree prefix cache (keep hash cache fallback selectable)                             | `crates/llm/src/core/prefix_cache_radix.rs`, `crates/llm/src/core/prefix_cache_hash.rs` | `mini-sglang-main/python/minisgl/kvcache/radix_manager.py`, `mini-sglang-main/python/minisgl/scheduler/cache.py`, `vllm.rs-main/src/core/prefix_cache.rs`                            |
| ⬜     | chunked prefill manager and prefill token budget controls                                 | `crates/llm/src/scheduler/prefill.rs`                                                   | `mini-sglang-main/python/minisgl/scheduler/prefill.py`, `mini-sglang-main/docs/features.md`, `vllm.rs-main/src/core/scheduler.rs`                                                    |
| ⬜     | overlap scheduling (CPU metadata/scheduling overlaps GPU execution)                       | `crates/llm/src/scheduler/scheduler.rs`                                                 | `mini-sglang-main/python/minisgl/scheduler/scheduler.py`, `mini-sglang-main/docs/features.md`                                                                                        |
| ⬜     | attention backend switch (eager/fa/flashinfer, prefill/decode split)                      | `crates/llm/src/attention/{base.rs,eager.rs,fa.rs,fi.rs}`                               | `mini-sglang-main/python/minisgl/attention/{base.py,fa.py,fi.py}`, `vllm.rs-main/src/models/layers/attention.rs`                                                                     |
| ⬜     | usage stats endpoint, cancel endpoint, session accounting                                 | `crates/llm/src/server/api.rs`, `crates/llm/src/core/engine.rs`                         | `vllm.rs-main/src/server/server.rs`, `mini-sglang-main/python/minisgl/server/api_server.py`                                                                                          |
| ⬜     | shell mode and high-level client wrapper                                                  | `crates/llm/src/server/shell.rs`, `crates/llm/src/api.rs`                               | `mini-sglang-main/python/minisgl/server/api_server.py`, `mini-sglang-main/docs/features.md`                                                                                          |
| ⬜     | integration tests: scheduler, cache, runner, api stream, cancel                           | `crates/llm/tests/*.rs`                                                                 | `nano-vllm-main/bench.py`, `mini-sglang-main/python/minisgl/benchmark/{perf.py,client.py}`, `vllm.rs-main/example/rust-demo/src/main.rs`                                             |
| ⬜     | perf bench harness and UML diagrams                                                       | `crates/llm/benches/*`, `README.md`                                                     | `nano-vllm-main/bench.py`, `mini-sglang-main/docs/{structures.md,features.md}`                                                                                                       |

## Credit

- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [vllm.rs](https://github.com/guoqingbao/vllm.rs)
- [mini-sglang](https://github.com/sgl-project/mini-sglang)
