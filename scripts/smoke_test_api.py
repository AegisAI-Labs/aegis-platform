from unittest.mock import patch

from fastapi.testclient import TestClient

from aegis_platform.llm.factory import LLMProviderFactory
from scripts.stub_llm_provider import StubLLMProvider


def main() -> None:
    with patch.object(LLMProviderFactory, "create_llm_provider", return_value=StubLLMProvider()):
        from aegis_platform.api.main import app

    print("=== Agent API Smoke Test ===")

    cases = [
        ("Hello", {"response": "Hello from the stub LLM", "tool_used": "llm"}),
        ("calculate 2+3", {"response": "5", "tool_used": "calculator"}),
    ]

    with TestClient(app) as client:
        for message, expected in cases:
            response = client.post("/agent/run", json={"message": message})
            assert response.status_code == 200, response.text
            assert response.json() == expected, response.json()
            print(message)
            print(response.json())

        print("=== Streaming Smoke Test ===")
        with client.stream("POST", "/agent/stream", json={"message": "Hello"}) as response:
            assert response.status_code == 200, response.text
            streamed_text = "".join(response.iter_text())
            assert streamed_text == "Hello from the stub LLM ", streamed_text
            print(response.status_code)
            print(streamed_text, end="")


if __name__ == "__main__":
    main()
