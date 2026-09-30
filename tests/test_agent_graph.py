from services.agent.graph import app_graph


def test_app_graph():
    assert app_graph is not None


def test_echo():
    result = app_graph.invoke({"user_input": "Hello"})
    assert result is not None
    assert result["response"] == "Aegis received your input: Hello"
    assert result["tool_used"] == "none"


def test_calculator():
    result = app_graph.invoke({"user_input": "calc:2+3"})
    assert result is not None
    assert result["response"] == "Calculation result: 5"
    assert result["tool_used"] == "calculator"
