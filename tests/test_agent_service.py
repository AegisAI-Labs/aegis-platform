import pytest

from services.agent.service import AgentService


@pytest.mark.asyncio
async def test_agent_service_run():
    service = AgentService()
    result = await service.run("Hello")
    assert result is not None
    assert result["response"] == "Aegis received your input: Hello"
    assert result["tool_used"] == "none"


@pytest.mark.asyncio
async def test_agent_service_run_calculator():
    service = AgentService()
    result = await service.run("calc: 7*9")
    assert result is not None
    assert result["response"] == "Calculation result: 63"
    assert result["tool_used"] == "calculator"


@pytest.mark.asyncio
async def test_agent_service_stream():
    service = AgentService()
    chunks = []
    async for chunk in service.stream("Hello"):
        chunks.append(chunk)
    assert chunks is not None
    assert "".join(chunks).strip() == "Aegis received your input: Hello"
