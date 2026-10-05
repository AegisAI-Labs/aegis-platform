import pytest

from aegis_platform.agent.graph import calculator_node, route_request, tool_registry
from aegis_platform.agent.state import AgentState


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("calculate 2+3", "calculator"),
        ("Please MULTIPLY 2 by 3", "calculator"),
        ("Hello", "llm"),
        ("calc:2+3", "llm"),
    ],
)
def test_route_request(message: str, expected: str) -> None:
    state: AgentState = {"user_input": message, "response": "", "tool_used": ""}
    assert route_request(state) == expected


def test_route_request_without_calculator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_registry, "has", lambda name: False)
    state: AgentState = {"user_input": "calculate 2+3", "response": "", "tool_used": ""}
    assert route_request(state) == "llm"


def test_calculator_node() -> None:
    state: AgentState = {"user_input": "calculate 2+3", "response": "", "tool_used": ""}
    assert calculator_node(state) == {"response": "5", "tool_used": "calculator"}


def test_calculator_node_uses_registered_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    expressions: list[str] = []

    def fake_calculator(expression: str) -> str:
        expressions.append(expression)
        return "result"

    monkeypatch.setattr(tool_registry, "get", lambda name: fake_calculator)
    state: AgentState = {"user_input": "calculator 7 * 9", "response": "", "tool_used": ""}
    assert calculator_node(state) == {"response": "result", "tool_used": "calculator"}
    assert expressions == ["7 * 9"]


def test_calculator_node_rejects_invalid_expression() -> None:
    state: AgentState = {"user_input": "calculate invalid", "response": "", "tool_used": ""}
    with pytest.raises(ValueError, match="Invalid calculator expression"):
        calculator_node(state)
