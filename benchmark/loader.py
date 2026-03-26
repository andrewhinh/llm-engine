from collections.abc import Iterator
from typing import Any

from guidellm.request import GenerationRequest, GenerativeRequestLoader


class CustomGenerativeRequestLoader(GenerativeRequestLoader):
    def __init__(
        self,
        extra_body: dict[str, Any] | None = None,
        *,
        use_chat_completions: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.extra_body = extra_body or {}
        self.use_chat_completions = use_chat_completions

    def __iter__(self) -> Iterator[GenerationRequest]:
        for item in super().__iter__():
            item.params.update(self.extra_body)
            if self.use_chat_completions:
                item.request_type = "chat_completions"
            yield item

    def __len__(self) -> int:
        return super().__len__()
