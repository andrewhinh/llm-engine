import os
import sys

import modal
import modal.experimental

from llmeng.modal import (
    GPU_TYPE,
    MINUTES,
    MODEL_DIR,
    N_GPU,
    RDMA,
    app,
    image,
    model_volume,
)

SUPPORTED_NNODES = [1, 2, 3, 4]


def _create_shell_runner(cluster_size: int):
    _name = "_run_shell" if cluster_size == 1 else f"_run_shell_clustered{cluster_size}"

    def _run_shell_entry(
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
        nnodes: int,
    ) -> None:
        from llmeng.server.api_server import run_api_server
        from llmeng.server.args import build_cli_args, parse_args
        from llmeng.server.launch import start_subprocesses

        if nnodes != cluster_size:
            if cluster_size == 1:
                raise ValueError(
                    f"nnodes={nnodes} requires clustered mode. Use nnodes in {SUPPORTED_NNODES[1:]}."
                )
            raise ValueError(
                f"Clustered shell requires nnodes={cluster_size}, got {nnodes}."
            )
        node_rank = 0
        master_ip = "127.0.0.1"
        if cluster_size > 1:
            cluster_info = modal.experimental.get_cluster_info()
            node_rank = cluster_info.rank
            master_ip = cluster_info.container_ipv4_ips[0]

        os.environ["NNODES"] = str(nnodes)
        os.environ["LLMENG_NODE_RANK"] = str(node_rank)
        os.environ["LLMENG_DISTRIBUTED_ADDR"] = f"tcp://{master_ip}:{server_port + 1}"
        if nnodes > 1:
            os.environ.setdefault("NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY", "AF_INET6")

        resolved_path = model_path
        local_name = resolved_path.replace("/", "--")
        local_path = os.path.join(MODEL_DIR, local_name)

        if not os.path.exists(local_path):
            from huggingface_hub import snapshot_download

            resolved_path = snapshot_download(resolved_path, local_dir=local_path)
            model_volume.commit()

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
            run_shell=(cluster_size == 1 or node_rank == 0),
        )

        def start_backend() -> None:
            start_subprocesses(server_args)

        if run_shell:
            modal.interact()

        run_api_server(server_args, start_backend, run_shell=run_shell)

    _run_shell_entry.__name__ = _name
    _run_shell_entry.__qualname__ = _name
    decorated = _run_shell_entry
    if cluster_size > 1:
        decorated = modal.experimental.clustered(size=cluster_size, rdma=RDMA)(
            decorated
        )
    return app.function(
        image=image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        timeout=60 * MINUTES,
        scaledown_window=5 * MINUTES,
        serialized=True,
    )(decorated)


SHELL_RUNNER_BY_NNODES: dict[int, modal.Function] = {
    n: _create_shell_runner(n) for n in SUPPORTED_NNODES
}


@app.local_entrypoint()
def main(
    model_path: str = "Qwen/Qwen3-0.6B",
    model_source: str = "huggingface",
    dtype: str = "auto",
    tp_size: int = 0,
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
    nnodes: int = 1,
) -> None:
    if not sys.stdin.isatty():
        raise ValueError(
            "Interactive shell sessions require a TTY. Re-run with `modal run -i -m llmeng.shell`."
        )
    if nnodes not in SHELL_RUNNER_BY_NNODES:
        raise ValueError(f"nnodes must be in {SUPPORTED_NNODES}, got {nnodes}.")
    if tp_size != N_GPU and tp_size > 0:
        raise ValueError(
            f"tp_size must match N_GPU (tp_size={tp_size}, N_GPU={N_GPU})."
        )
    runner = SHELL_RUNNER_BY_NNODES[nnodes]
    runner.remote(
        model_path=model_path,
        model_source=model_source,
        dtype=dtype,
        tp_size=tp_size or N_GPU,
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
        nnodes=nnodes,
    )
