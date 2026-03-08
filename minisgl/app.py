import modal

from minisgl.modal import MINUTES, app, image, resolve_model_path

n_gpu = 4


@app.cls(
    image=image,
    gpu=f"h200:{n_gpu}",
    timeout=60 * MINUTES,
    scaledown_window=5 * MINUTES,
)
@modal.concurrent(max_inputs=64)
class ModelServer:
    model_path: str = modal.parameter(default="Qwen/Qwen3-32B")
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
    cuda_graph_max_bs: int = modal.parameter(default=0)
    use_pynccl: bool = modal.parameter(default=True)

    @modal.enter()
    def startup(self):
        import minisgl.server.api_server as api_module
        from minisgl.message import BaseFrontendMsg, BaseTokenizerMsg
        from minisgl.server.api_server import FrontendManager
        from minisgl.server.args import build_cli_args, parse_args
        from minisgl.server.launch import start_subprocesses
        from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

        self.logger = init_logger(__name__, "modal-server")

        resolved_path = resolve_model_path(self.model_path)
        print(f"Starting up with model={resolved_path}, tp={self.tp_size}")

        server_args, _ = parse_args(
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
        self.config = server_args

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
        from minisgl.server.api_server import app as fastapi_app

        return fastapi_app

    @modal.exit()
    def shutdown(self):
        import psutil

        if hasattr(self, "_global_state"):
            self._global_state.shutdown()

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()

        if hasattr(self, "logger"):
            self.logger.info("Server shutdown complete")
