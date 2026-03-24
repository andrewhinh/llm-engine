import os
from datetime import datetime, timezone

import modal
import modal.experimental

from llmeng.modal import (
    GPU_TYPE,
    HF_CACHE_PATH,
    MINUTES,
    N_GPU,
    NNODES,
    RDMA,
    app,
    get_runtime_image,
    hf_cache_volume,
)


def create_model_server_class(
    engine: str = "llmeng",
    model: str = "Qwen/Qwen3-32B",
    nnode: int = NNODES,
    gpu_type: str = GPU_TYPE,
    n_gpu: int = N_GPU,
    name_suffix: str | None = None,
):
    class _ModelServer:
        model_path: str = modal.parameter(default=model)
        model_source: str = modal.parameter(default="huggingface")
        dtype_str: str = modal.parameter(default="auto")
        tp_size: int = modal.parameter(default=n_gpu)
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
        cuda_graph_max_bs: int = modal.parameter(default=-1)
        use_nccl: bool = modal.parameter(default=True)
        nnodes: int = modal.parameter(default=nnode)
        server_id: str = modal.parameter(default="")
        startup_metrics_dict_id: str = modal.parameter(default="")

        @modal.enter()
        def startup(self):
            import llmeng.server.api_server as api_module
            from llmeng.message import BaseFrontendMsg, BaseTokenizerMsg
            from llmeng.server.api_server import FrontendManager
            from llmeng.server.args import build_cli_args, parse_args
            from llmeng.server.launch import start_subprocesses
            from llmeng.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

            if self.server_id and self.startup_metrics_dict_id:
                startup_metrics = modal.Dict.from_id(self.startup_metrics_dict_id)
                startup_metrics[self.server_id] = datetime.now(timezone.utc).timestamp()

            if self.nnodes != nnode:
                raise ValueError(f"Expected nnodes={nnode}, got {self.nnodes}.")

            node_rank = 0
            master_ip = "127.0.0.1"
            if nnode > 1:
                cluster_info = modal.experimental.get_cluster_info()
                node_rank = cluster_info.rank
                master_ip = cluster_info.container_ipv4_ips[0]
                print(f"Cluster info: {cluster_info}, master_ip: {master_ip}")

            self.logger = init_logger(__name__, "modal-server")
            os.environ["NNODES"] = str(nnode)
            os.environ["LLMENG_NODE_RANK"] = str(node_rank)
            os.environ["LLMENG_DISTRIBUTED_ADDR"] = (
                f"tcp://{master_ip}:{self.server_port + 1}"
            )

            resolved_path = self.model_path
            local_name = resolved_path.replace("/", "--")
            local_path = os.path.join(HF_CACHE_PATH, local_name)

            if not os.path.exists(local_path):
                from huggingface_hub import snapshot_download

                resolved_path = snapshot_download(resolved_path, local_dir=local_path)
                hf_cache_volume.commit()

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
                    cuda_graph_max_bs=(
                        None
                        if self.cuda_graph_max_bs == -1
                        else 0
                        if self.cuda_graph_max_bs == -2
                        else self.cuda_graph_max_bs
                    ),
                    use_nccl=self.use_nccl,
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

    cls_name = "ModelServer" if nnode == 1 else f"ModelServerClustered{nnode}"
    if name_suffix:
        cls_name = f"{cls_name}{name_suffix}"
    _ModelServer.__name__ = cls_name
    _ModelServer.__qualname__ = cls_name

    decorated = modal.concurrent(max_inputs=1000)(_ModelServer)
    if nnode > 1:
        decorated = modal.experimental.clustered(size=nnode, rdma=RDMA)(decorated)
    decorated = app.cls(
        image=get_runtime_image(engine),
        gpu=f"{gpu_type}:{n_gpu}",
        startup_timeout=30 * MINUTES,
        timeout=30 * MINUTES,
        scaledown_window=2 * MINUTES,
        serialized=True,
    )(decorated)
    globals()[cls_name] = decorated
    return decorated
