from collections.abc import AsyncIterator

from aegis_platform.llm.models import LLMRequest, LLMResponse


class StubLLMProvider:
    def __init__(self, content: str = "Hello from the stub LLM") -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=self.content, model="stub", provider="stub")

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        yield self.content


class ChunkedStubLLMProvider(StubLLMProvider):
    """Streams the content in fixed chunks, like a real provider."""

    def __init__(self, chunks: tuple[str, ...] = ("Hel", "lo", " world")) -> None:
        super().__init__("".join(chunks))
        self.chunks = chunks

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        for chunk in self.chunks:
            yield chunk


class FailingStubLLMProvider(StubLLMProvider):
    """Raises ``error`` from generate, and from stream after yielding ``chunks_before_error``."""

    def __init__(self, error: BaseException, chunks_before_error: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.error = error
        self.chunks_before_error = chunks_before_error

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise self.error

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        self.requests.append(request)
        for chunk in self.chunks_before_error:
            yield chunk
        raise self.error
