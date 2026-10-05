import pytest
from fastapi.testclient import TestClient


def test_agent_run_llm(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the stub LLM", "tool_used": "llm"}


def test_agent_run_calculator(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "calculate 7*9"})
    assert response.status_code == 200
    assert response.json() == {"response": "63", "tool_used": "calculator"}


def test_agent_stream(client: TestClient) -> None:
    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        assert response.status_code == 200
        assert "".join(response.iter_text()) == "Hello from the stub LLM "


def test_agent_run_rejects_missing_message(client: TestClient) -> None:
    response = client.post("/agent/run", json={})
    assert response.status_code == 422


def test_agent_run_propagates_invalid_calculation(client: TestClient) -> None:
    with pytest.raises(ValueError, match="Invalid calculator expression"):
        client.post("/agent/run", json={"message": "calculate invalid"})
