from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple, cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import AliasChoices, BaseModel, Field
from starlette.background import BackgroundTask

from llmeng.core import SamplingParams
from llmeng.env import ENV
from llmeng.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from llmeng.message.utils import unwrap_batch_msg
from llmeng.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE: "FrontendManager | None" = None


def _message_content_to_text(content: str | List[Dict[str, object]]) -> str:
    if isinstance(content, str):
        return content

    text_parts: List[str] = []
    for item in content:
        text = item.get("text")
        if item.get("type") == "text" and isinstance(text, str):
            text_parts.append(text)
        else:
            text_parts.append(json.dumps(item))
    return "".join(text_parts)


def _serialize_message(
    message: "Message | Dict[str, str] | Dict[str, object]",
) -> Dict[str, str]:
    content: str | List[Dict[str, object]] | None
    if isinstance(message, dict):
        role = message.get("role")
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            content = cast(List[Dict[str, object]], raw_content)
        else:
            content = None
    else:
        role = message.role
        content = message.content
    assert role is not None, "Message role must be set"
    assert content is not None, "Message content must be set"
    assert isinstance(content, (str, list)), "Message content must be serializable"
    return {"role": str(role), "content": _message_content_to_text(content)}


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    msgs = unwrap_batch_msg(msg, BatchFrontendMsg)
    if not isinstance(msg, BatchFrontendMsg):
        assert isinstance(msg, UserReply)
        return [msg]
    assert all(isinstance(reply, UserReply) for reply in msgs)
    return [reply for reply in msgs if isinstance(reply, UserReply)]


async def _send_request(
    state: "FrontendManager",
    text: str | List[Message | Dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
    ignore_eos: bool = False,
) -> int:
    uid = state.new_user()
    text_for_msg: str | List[Dict[str, str]] = (
        [_serialize_message(m) for m in text] if isinstance(text, list) else text
    )
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=text_for_msg,
            sampling_params=SamplingParams(
                ignore_eos=ignore_eos,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )
    )
    return uid


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | List[Dict[str, object]]


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = Field(
        default=16,
        validation_alias=AliasChoices("max_tokens", "max_completion_tokens"),
    )
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | List[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llm-engine"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        completion_id = f"chatcmpl-{uid}"
        created = int(time.time())
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {"content": ""}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.config.model_path,
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.config.model_path,
            "choices": [
                {"delta": {"content": ""}, "index": 0, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def complete_chat_completions(self, uid: int) -> Dict[str, object]:
        chunks: List[str] = []
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                chunks.append(ack.incremental_output)
            if ack.finished:
                break
        return {
            "id": f"chatcmpl-{uid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.config.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(chunks)},
                    "finish_reason": "stop",
                }
            ],
        }

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = await _send_request(
        state,
        req.prompt,
        max_tokens=req.max_tokens,
        ignore_eos=req.ignore_eos,
        temperature=0.0,
        top_k=1,
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [_serialize_message(msg) for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    uid = await _send_request(
        state,
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        ignore_eos=req.ignore_eos,
    )

    if not req.stream:
        return await state.complete_chat_completions(uid)

    return StreamingResponse(
        state.stream_with_cancellation(
            state.stream_chat_completions(uid), request, uid
        ),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(
        data=[ModelCard(id=state.config.model_path, root=state.config.model_path)]
    )


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    uid = await _send_request(
        state,
        cast(List[Message | Dict[str, str]], req.messages),
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        ignore_eos=req.ignore_eos,
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


async def read_stdin():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        line = line.decode().rstrip("\n")


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            need_stop = False
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(
                    Message(role="assistant", content=assistant_msg)
                )
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                if need_stop:
                    break
                msg = chunk.decode()
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(
    config: ServerArgs, start_backend: Callable[[], None], run_shell: bool
) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
