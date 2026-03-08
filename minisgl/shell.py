from __future__ import annotations

import sys

from minisgl.modal import MINUTES, app, image, resolve_model_path

n_gpu = 1


@app.function(
    image=image,
    gpu=f"h200:{n_gpu}",
    timeout=60 * MINUTES,
    scaledown_window=5 * MINUTES,
)
def _run_api_shell(
    model_path: str,
    model_source: str,
    dtype: str,
    tp_size: int,
    max_running_req: int,
    memory_ratio: int,
    attention_backend: str,
    moe_backend: str,
    cache_type: str,
    max_extend_tokens: int,
    page_size: int,
    server_host: str,
    server_port: int,
    max_seq_len_override: int,
    num_page_override: int,
    num_tokenizer: int,
    cuda_graph_max_bs: int,
    use_pynccl: bool,
) -> None:
    from modal import interact

    from minisgl.server.api_server import run_api_server
    from minisgl.server.args import build_cli_args, parse_args
    from minisgl.server.launch import start_subprocesses

    resolved_path = resolve_model_path(model_path)
    server_args, run_shell = parse_args(
        build_cli_args(
            model_path=resolved_path,
            model_source=model_source,
            dtype=dtype,
            tp_size=tp_size,
            max_running_req=max_running_req,
            memory_ratio=memory_ratio / 100,
            attention_backend=attention_backend,
            moe_backend=moe_backend,
            cache_type=cache_type,
            max_extend_tokens=max_extend_tokens,
            page_size=page_size,
            server_host=server_host,
            server_port=server_port,
            max_seq_len_override=max_seq_len_override or None,
            num_page_override=num_page_override,
            num_tokenizer=num_tokenizer,
            cuda_graph_max_bs=cuda_graph_max_bs,
            use_pynccl=use_pynccl,
        ),
        run_shell=True,
    )

    def start_backend() -> None:
        start_subprocesses(server_args)

    interact()
    run_api_server(server_args, start_backend, run_shell=run_shell)


@app.local_entrypoint()
def main(
    model_path: str = "Qwen/Qwen3-0.6B",
    model_source: str = "huggingface",
    dtype: str = "auto",
    tp_size: int = n_gpu,
    max_running_req: int = 256,
    memory_ratio: int = 90,
    attention_backend: str = "auto",
    moe_backend: str = "auto",
    cache_type: str = "radix",
    max_extend_tokens: int = 8192,
    page_size: int = 1,
    server_host: str = "0.0.0.0",
    server_port: int = 8000,
    max_seq_len_override: int = 0,
    num_page_override: int = 0,
    num_tokenizer: int = 0,
    cuda_graph_max_bs: int = 0,
    use_pynccl: bool = True,
) -> None:
    if not sys.stdin.isatty():
        raise ValueError(
            "Interactive shell sessions require a TTY. Re-run with `modal run -i minisgl/shell.py`."
        )

    _run_api_shell.remote(
        model_path=model_path,
        model_source=model_source,
        dtype=dtype,
        tp_size=tp_size,
        max_running_req=max_running_req,
        memory_ratio=memory_ratio,
        attention_backend=attention_backend,
        moe_backend=moe_backend,
        cache_type=cache_type,
        max_extend_tokens=max_extend_tokens,
        page_size=page_size,
        server_host=server_host,
        server_port=server_port,
        max_seq_len_override=max_seq_len_override,
        num_page_override=num_page_override,
        num_tokenizer=num_tokenizer,
        cuda_graph_max_bs=cuda_graph_max_bs,
        use_pynccl=use_pynccl,
    )
