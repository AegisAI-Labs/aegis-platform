from fastapi.testclient import TestClient

from services.api.main import app

client = TestClient(app)


def test_health_returns_200():
    """The general health endpoint should return HTTP 200."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_live_returns_200():
    """The liveness endpoint should return HTTP 200."""

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_ready_returns_200():
    """The readiness endpoint should return HTTP 200."""

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
