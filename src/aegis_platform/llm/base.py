from collections.abc import AsyncIterator
from typing import Protocol

from .models import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse: ...

    def stream(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[str]: ...
