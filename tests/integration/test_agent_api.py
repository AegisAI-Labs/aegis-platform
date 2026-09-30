from fastapi.testclient import TestClient

from aegis_platform.api.main import app

client = TestClient(app)


def test_agent_run():
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "Aegis received your input: Hello", "tool_used": "none"}


def test_agent_run_calculator():
    response = client.post("/agent/run", json={"message": "calc: 7*9"})
    assert response.status_code == 200
    assert response.json() == {"response": "Calculation result: 63", "tool_used": "calculator"}


def test_agent_stream():
    response = client.post("/agent/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.text.strip() == "Aegis received your input: Hello"


def test_agent_stream_calculator():
    response = client.post("/agent/stream", json={"message": "calc: 7*9"})
    assert response.status_code == 200
    assert response.text.strip() == "Calculation result: 63"
