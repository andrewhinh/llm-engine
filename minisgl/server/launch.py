from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    import torch

    from minisgl.scheduler import Scheduler

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

    from minisgl.tokenizer import tokenize_worker

    mp.set_start_method("spawn", force=True)

    world_size = server_args.tp_info.size
    # a multiprocessing queue to receive ack from subprocesses
    # so that we can guarantee all subprocesses are ready
    ack_queue: mp.Queue[str] = mp.Queue()

    for i in range(world_size):
        new_args = replace(
            server_args,
            tp_info=DistributedInfo(i, world_size),
        )
        mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"minisgl-TP{i}-scheduler",
        ).start()

    num_tokenizers = server_args.num_tokenizer
    # DeTokenizer, only 1
    mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": server_args.model_path,
            "addr": server_args.zmq_detokenizer_addr,
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "create": server_args.tokenizer_create_addr,
            "tokenizer_id": num_tokenizers,
            "ack_queue": ack_queue,
        },
        daemon=False,
        name="minisgl-detokenizer-0",
    ).start()
    for i in range(num_tokenizers):
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_tokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": i,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name=f"minisgl-tokenizer-{i}",
        ).start()

    # Wait for acknowledgments from all worker processes:
    # - world_size schedulers (but only primary rank sends ack)
    # - num_tokenizers tokenizers
    # - 1 detokenizer
    # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
    for _ in range(num_tokenizers + 2):
        logger.info(ack_queue.get())
