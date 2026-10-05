from collections.abc import Iterator
from importlib import import_module
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aegis_platform.agent.service import AgentService
from aegis_platform.llm.factory import LLMProviderFactory
from tests.fakes import StubLLMProvider


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    with patch.object(LLMProviderFactory, "create_llm_provider", return_value=StubLLMProvider()):
        api = import_module("aegis_platform.api.main")
    monkeypatch.setattr(api, "agent_service", AgentService(StubLLMProvider()))
    with TestClient(api.app) as test_client:
        yield test_client
