from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import replace
from typing import TYPE_CHECKING

from llmeng.distributed import DistributedInfo
from llmeng.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch

    from llmeng.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
            logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def start_subprocesses(
    server_args: ServerArgs, logger: logging.Logger | None = None
) -> None:
    if logger is None:
        logger = init_logger(__name__)

    import multiprocessing as mp

    from llmeng.tokenizer import tokenize_worker

    mp.set_start_method("spawn", force=True)

    local_world_size = server_args.tp_info.size
    nnodes = int(os.environ.get("NNODES", "1"))
    node_rank = int(os.environ.get("LLMENG_NODE_RANK", "0"))
    global_world_size = local_world_size * nnodes
    # a multiprocessing queue to receive ack from subprocesses
    # so that we can guarantee all subprocesses are ready
    ack_queue: mp.Queue[str] = mp.Queue()

    for i in range(local_world_size):
        global_rank = node_rank * local_world_size + i
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(
                rank=i,
                size=local_world_size,
                global_rank=global_rank,
                global_size=global_world_size,
            ),
        )
        mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"llmeng-TP{i}-scheduler",
        ).start()

    num_tokenizers = server_args.num_tokenizer
    # DeTokenizer, only 1
    base_tokenizer_kwargs = dict(
        tokenizer_path=server_args.model_path,
        backend_addr=server_args.zmq_backend_addr,
        frontend_addr=server_args.zmq_frontend_addr,
        local_bs=1,
        create=server_args.tokenizer_create_addr,
        ack_queue=ack_queue,
    )

    mp.Process(
        target=tokenize_worker,
        kwargs={
            **base_tokenizer_kwargs,
            "addr": server_args.zmq_detokenizer_addr,
            "tokenizer_id": num_tokenizers,
        },
        daemon=False,
        name="llmeng-detokenizer-0",
    ).start()
    for i in range(num_tokenizers):
        mp.Process(
            target=tokenize_worker,
            kwargs={
                **base_tokenizer_kwargs,
                "addr": server_args.zmq_tokenizer_addr,
                "tokenizer_id": i,
            },
            daemon=False,
            name=f"llmeng-tokenizer-{i}",
        ).start()

    # Wait for acknowledgments from all worker processes:
    # - world_size schedulers (but only primary rank sends ack)
    # - num_tokenizers tokenizers
    # - 1 detokenizer
    # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
    for _ in range(num_tokenizers + 2):
        logger.info(ack_queue.get())
