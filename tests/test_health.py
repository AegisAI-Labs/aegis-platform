from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient):
    """The general health endpoint should return HTTP 200."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_live_returns_200(client: TestClient):
    """The liveness endpoint should return HTTP 200."""

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_ready_returns_200(client: TestClient):
    """The readiness endpoint should return HTTP 200."""

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
