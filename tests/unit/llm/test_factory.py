from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aegis_platform.config.settings import Settings
from aegis_platform.llm.factory import LLMProviderFactory


def test_factory_creates_openai_provider() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="OpenAI",
        openai_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=12.0,
    )
    with patch("aegis_platform.llm.factory.OpenAIProvider") as provider:
        result = LLMProviderFactory.create_llm_provider(settings)

    assert result is provider.return_value
    provider.assert_called_once_with(api_key="test-key", model="test-model", timeout_seconds=12.0)


def test_factory_requires_openai_key() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="OpenAI API key is not configured"):
        LLMProviderFactory.create_llm_provider(settings)


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings(_env_file=None, llm_provider="unknown")
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
        LLMProviderFactory.create_llm_provider(settings)


def test_settings_reject_invalid_timeout() -> None:
    with pytest.raises(ValidationError, match="llm_timeout_seconds"):
        Settings(_env_file=None, llm_timeout_seconds="not-a-number")  # type: ignore[arg-type]
