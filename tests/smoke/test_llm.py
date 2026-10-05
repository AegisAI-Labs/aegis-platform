import pytest

from aegis_platform.agent.graph import app_graph
from aegis_platform.agent.state import AgentState
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_llm_smoke() -> None:
    state: AgentState = {"user_input": "Hello", "response": "", "tool_used": ""}
    result = await app_graph.ainvoke(
        state, config={"configurable": {"llm_provider": StubLLMProvider()}}
    )
    assert result["response"] == "Hello from the stub LLM"
    assert result["tool_used"] == "llm"
