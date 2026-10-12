from fastapi.testclient import TestClient


def test_api_stream_smoke(client: TestClient) -> None:
    with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "".join(response.iter_text()) == (
            'event: token\ndata: {"text": "Hello "}\n\n'
            'event: token\ndata: {"text": "from "}\n\n'
            'event: token\ndata: {"text": "the "}\n\n'
            'event: token\ndata: {"text": "stub "}\n\n'
            'event: token\ndata: {"text": "LLM "}\n\n'
            'event: done\ndata: {"status": "completed"}\n\n'
        )
