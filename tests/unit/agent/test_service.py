import pytest

from aegis_platform.agent.service import AgentService
from aegis_platform.llm.models import LLMMessage
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_service_run_llm() -> None:
    provider = StubLLMProvider()
    result = await AgentService(provider).run(" Hello ")

    assert result == {"response": "Hello from the stub LLM", "tool_used": "llm"}
    assert provider.requests[0].messages == [LLMMessage(role="user", content="Hello")]


@pytest.mark.asyncio
async def test_service_run_calculator() -> None:
    provider = StubLLMProvider()
    result = await AgentService(provider).run("calculate 7*9")

    assert result == {"response": "63", "tool_used": "calculator"}
    assert provider.requests == []


@pytest.mark.asyncio
async def test_service_stream() -> None:
    service = AgentService(StubLLMProvider())
    chunks = [chunk async for chunk in service.stream("Hello")]

    assert chunks == ["Hello ", "from ", "the ", "stub ", "LLM "]
