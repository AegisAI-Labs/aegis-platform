import pytest

from aegis_platform.agent.graph import app_graph
from aegis_platform.agent.state import AgentState
from scripts import experiment_day4_latency
from tests.fakes import StubLLMProvider


@pytest.mark.asyncio
async def test_llm_smoke() -> None:
    state: AgentState = {"user_input": "Hello", "response": "", "tool_used": ""}
    result = await app_graph.ainvoke(
        state, config={"configurable": {"llm_provider": StubLLMProvider()}}
    )
    assert result["response"] == "Hello from the stub LLM"
    assert result["tool_used"] == "llm"


@pytest.mark.asyncio
async def test_local_latency_experiments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_day4_latency, "ITERATIONS", 2)

    calculator = await experiment_day4_latency.run_calculator_experiment()
    fake = await experiment_day4_latency.run_fake_llm_experiment()

    assert calculator.iterations == 2
    assert fake.iterations == 2
    assert calculator.minimum >= 0
    assert fake.minimum >= 0


@pytest.mark.asyncio
async def test_latency_experiment_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_day4_latency.settings, "llm_model", "")
    with pytest.raises(ValueError, match="Set LLM_MODEL"):
        await experiment_day4_latency.main()
