from fastapi.testclient import TestClient


def test_api_smoke(client: TestClient) -> None:
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the stub LLM", "tool_used": "llm"}


def test_observability_metrics_endpoint(client: TestClient) -> None:
    """Test that metrics endpoint returns metrics after agent run."""
    # Run the agent to generate metrics
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200

    # Get metrics
    metrics_response = client.get("/observability/metrics")
    assert metrics_response.status_code == 200
    data = metrics_response.json()
    assert "metrics" in data
    assert isinstance(data["metrics"], list)


def test_observability_events_endpoint(client: TestClient) -> None:
    """Test that events endpoint returns events after agent run."""
    # Run the agent to generate events
    response = client.post("/agent/run", json={"message": "Hello"})
    assert response.status_code == 200

    # Get events
    events_response = client.get("/observability/events")
    assert events_response.status_code == 200
    data = events_response.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_observability_endpoints_empty_initially(client: TestClient) -> None:
    """Test that observability endpoints return empty lists when no agent calls made."""
    # Get metrics without running agent
    metrics_response = client.get("/observability/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json()["metrics"] == []

    # Get events without running agent
    events_response = client.get("/observability/events")
    assert events_response.status_code == 200
    assert events_response.json()["events"] == []
