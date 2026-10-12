import pytest

from aegis_platform.llm.base import LLMProvider
from aegis_platform.llm.models import LLMMessage, LLMRequest, LLMResponse
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_fake_provider_contract() -> None:
    provider: LLMProvider = StubLLMProvider()
    request = LLMRequest(messages=[LLMMessage(role="user", content="Hello")])

    response = await provider.generate(request)
    chunks = [chunk async for chunk in provider.stream(request)]

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello from the stub LLM"
    assert chunks == [response.content]
