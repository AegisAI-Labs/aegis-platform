import pytest

from aegis_platform.agent.graph import app_graph
from aegis_platform.agent.state import AgentState
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_graph_llm() -> None:
    state: AgentState = {"user_input": "Hello", "response": "", "tool_used": ""}
    provider = StubLLMProvider()
    result = await app_graph.ainvoke(state, config={"configurable": {"llm_provider": provider}})

    assert result == {
        "user_input": "Hello",
        "response": "Hello from the stub LLM",
        "tool_used": "llm",
    }
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_graph_calculator() -> None:
    state: AgentState = {"user_input": "calculate 7*9", "response": "", "tool_used": ""}
    provider = StubLLMProvider()
    result = await app_graph.ainvoke(state, config={"configurable": {"llm_provider": provider}})

    assert result == {"user_input": "calculate 7*9", "response": "63", "tool_used": "calculator"}
    assert provider.requests == []
