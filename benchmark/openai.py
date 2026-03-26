from collections.abc import AsyncGenerator
from typing import Optional, Union

import httpx
from guidellm.backend.openai import OpenAIHTTPBackend
from guidellm.backend.response import ResponseSummary, StreamingTextResponse


class CustomOpenAIHTTPBackend(OpenAIHTTPBackend):
    """A custom OpenAI HTTP backend that increases the number of maximum redirects."""

    def __init__(self, *args, use_chat_completions: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_chat_completions = use_chat_completions

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            client = super()._get_async_client()
            client.max_redirects = 1000
            self._async_client = client

        return self._async_client

    async def text_completions(
        self,
        prompt: Union[str, list[str]],
        request_id: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        **kwargs,
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        if not self.use_chat_completions:
            async for resp in super().text_completions(
                prompt=prompt,
                request_id=request_id,
                prompt_token_count=prompt_token_count,
                output_token_count=output_token_count,
                **kwargs,
            ):
                yield resp
            return

        async for resp in self.chat_completions(
            content=prompt,
            request_id=request_id,
            prompt_token_count=prompt_token_count,
            output_token_count=output_token_count,
            **kwargs,
        ):
            yield resp
