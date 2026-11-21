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
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "".join(response.iter_text()) == (
            'event: token\ndata: {"text": "Hello from the stub LLM"}\n\n'
            'event: done\ndata: {"status": "completed"}\n\n'
        )


def test_agent_run_rejects_missing_message(client: TestClient) -> None:
    response = client.post("/agent/run", json={})
    assert response.status_code == 422


def test_agent_run_maps_invalid_calculation_to_400(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "calculate 2/0"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_tool_input"
    assert error["message"] == "Division by zero"
    assert error["correlation_id"] == response.headers["x-correlation-id"]


def test_agent_run_sends_non_arithmetic_calculate_request_to_llm(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "calculate invalid"})

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the stub LLM", "tool_used": "llm"}
