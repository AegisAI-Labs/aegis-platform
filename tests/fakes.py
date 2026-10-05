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
