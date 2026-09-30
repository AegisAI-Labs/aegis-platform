from fastapi.testclient import TestClient

from aegis_platform.api.main import app

client = TestClient(app)

print("=== Streaming Smoke Test ===")

with client.stream(
    "POST",
    "/agent/stream",
    json={"message": "Hello"},
) as response:
    print(response.status_code)

    for chunk in response.iter_text():
        print(chunk, end="")
