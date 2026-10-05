from fastapi.testclient import TestClient


def test_api_stream_smoke(client: TestClient) -> None:
    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        assert response.status_code == 200
        assert "".join(response.iter_text()) == "Hello from the stub LLM "
