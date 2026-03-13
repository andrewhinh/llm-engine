import os

import modal
import modal.experimental

from llmeng.modal import (
    GPU_TYPE,
    MINUTES,
    MODEL_DIR,
    N_GPU,
    NNODES,
    RDMA,
    app,
    image,
    model_volume,
)


def _create_model_server_class(cluster_size: int):
    class _ModelServer:
        model_path: str = modal.parameter(default="Qwen/Qwen3-14B")
        model_source: str = modal.parameter(default="huggingface")
        dtype_str: str = modal.parameter(default="auto")
        tp_size: int = modal.parameter(default=N_GPU)
        max_running_req: int = modal.parameter(default=256)
        memory_ratio: int = modal.parameter(default=90)
        attention_backend: str = modal.parameter(default="auto")
        moe_backend: str = modal.parameter(default="auto")
        cache_type: str = modal.parameter(default="naive")
        max_extend_tokens: int = modal.parameter(default=8192)
        page_size: int = modal.parameter(default=1)
        server_host: str = modal.parameter(default="0.0.0.0")
        server_port: int = modal.parameter(default=8000)
        max_seq_len_override: int = modal.parameter(default=0)
        num_page_override: int = modal.parameter(default=0)
        num_tokenizer: int = modal.parameter(default=0)
        cuda_graph_max_bs: int = modal.parameter(default=0)
        use_pynccl: bool = modal.parameter(default=True)
        nnodes: int = modal.parameter(default=cluster_size)

        @modal.enter()
        def startup(self):
            import llmeng.server.api_server as api_module
            from llmeng.message import BaseFrontendMsg, BaseTokenizerMsg
            from llmeng.server.api_server import FrontendManager
            from llmeng.server.args import build_cli_args, parse_args
            from llmeng.server.launch import start_subprocesses
            from llmeng.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

            if self.nnodes != cluster_size:
                raise ValueError(f"Expected nnodes={cluster_size}, got {self.nnodes}.")

            node_rank = 0
            master_ip = "127.0.0.1"
            if cluster_size > 1:
                cluster_info = modal.experimental.get_cluster_info()
                node_rank = cluster_info.rank
                master_ip = cluster_info.container_ipv4_ips[0]
                print(f"Cluster info: {cluster_info}, master_ip: {master_ip}")

            self.logger = init_logger(__name__, "modal-server")
            os.environ["NNODES"] = str(cluster_size)
            os.environ["LLMENG_NODE_RANK"] = str(node_rank)
            os.environ["LLMENG_DISTRIBUTED_ADDR"] = (
                f"tcp://{master_ip}:{self.server_port + 1}"
            )
            if cluster_size > 1:
                os.environ.setdefault("NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY", "AF_INET6")

            resolved_path = self.model_path
            local_name = resolved_path.replace("/", "--")
            local_path = os.path.join(MODEL_DIR, local_name)

            if not os.path.exists(local_path):
                from huggingface_hub import snapshot_download

                resolved_path = snapshot_download(resolved_path, local_dir=local_path)
                model_volume.commit()

            self.config, _ = parse_args(
                build_cli_args(
                    model_path=resolved_path,
                    model_source=self.model_source,
                    dtype=self.dtype_str,
                    tp_size=self.tp_size,
                    max_running_req=self.max_running_req,
                    memory_ratio=self.memory_ratio / 100,
                    attention_backend=self.attention_backend,
                    moe_backend=self.moe_backend,
                    cache_type=self.cache_type,
                    max_extend_tokens=self.max_extend_tokens,
                    page_size=self.page_size,
                    server_host=self.server_host,
                    server_port=self.server_port,
                    max_seq_len_override=self.max_seq_len_override or None,
                    num_page_override=self.num_page_override,
                    num_tokenizer=self.num_tokenizer,
                    cuda_graph_max_bs=self.cuda_graph_max_bs,
                    use_pynccl=self.use_pynccl,
                ),
                run_shell=False,
            )

            start_subprocesses(self.config, logger=self.logger)

            self._global_state = FrontendManager(
                config=self.config,
                recv_tokenizer=ZmqAsyncPullQueue(
                    self.config.zmq_frontend_addr,
                    create=True,
                    decoder=BaseFrontendMsg.decoder,
                ),
                send_tokenizer=ZmqAsyncPushQueue(
                    self.config.zmq_tokenizer_addr,
                    create=self.config.frontend_create_tokenizer_link,
                    encoder=BaseTokenizerMsg.encoder,
                ),
            )
            api_module._GLOBAL_STATE = self._global_state
            self.logger.info("Server startup complete")

        @modal.asgi_app()
        def serve(self):
            from llmeng.server.api_server import app as fastapi_app

            return fastapi_app

        @modal.exit()
        def shutdown(self):
            import psutil

            self._global_state.shutdown()

            parent = psutil.Process()
            for child in parent.children(recursive=True):
                child.kill()

            self.logger.info("Server shutdown complete")

    cls_name = (
        "ModelServer" if cluster_size == 1 else f"ModelServerClustered{cluster_size}"
    )
    _ModelServer.__name__ = cls_name
    _ModelServer.__qualname__ = cls_name

    decorated = modal.concurrent(max_inputs=64)(_ModelServer)
    if cluster_size > 1:
        decorated = modal.experimental.clustered(size=cluster_size, rdma=RDMA)(
            decorated
        )
    decorated = app.cls(
        image=image,
        gpu=f"{GPU_TYPE}:{N_GPU}",
        timeout=60 * MINUTES,
        scaledown_window=5 * MINUTES,
        serialized=True,
    )(decorated)
    return decorated


CLASS_NAME_BY_NNODES: dict[int, str] = {}
MODEL_SERVER_BY_NNODES: dict[int, type] = {}

_cls = _create_model_server_class(NNODES)
_name = "ModelServer" if NNODES == 1 else f"ModelServerClustered{NNODES}"
CLASS_NAME_BY_NNODES[NNODES] = _name
MODEL_SERVER_BY_NNODES[NNODES] = _cls
globals()[_name] = _cls
