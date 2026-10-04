from aegis_platform.llm.models import LLMRequest, LLMResponse


class StubLLMProvider:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="Hello from the stub LLM", model="stub", provider="stub")
