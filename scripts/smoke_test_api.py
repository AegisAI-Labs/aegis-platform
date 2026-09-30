from fastapi.testclient import TestClient

from aegis_platform.api.main import app

client = TestClient(app)

print("=== Agent API Smoke Test ===")

for message in ["Hello", "calc:2+3"]:
    response = client.post(
        "/agent/run",
        json={"message": message},
    )

    print(message)
    print(response.json())
