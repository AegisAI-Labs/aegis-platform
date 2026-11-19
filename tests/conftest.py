from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load local settings before application modules; CI environment variables win.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from aegis_platform.agent.service import AgentService
    from aegis_platform.llm.factory import LLMProviderFactory
    from tests.fakes import StubLLMProvider

    with patch.object(LLMProviderFactory, "create_llm_provider", return_value=StubLLMProvider()):
        api = import_module("aegis_platform.api.main")
    monkeypatch.setattr(api, "agent_service", AgentService(StubLLMProvider()))
    with TestClient(api.app) as test_client:
        yield test_client
