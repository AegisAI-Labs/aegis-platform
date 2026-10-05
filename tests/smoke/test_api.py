from fastapi.testclient import TestClient


def test_api_smoke(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the stub LLM", "tool_used": "llm"}
