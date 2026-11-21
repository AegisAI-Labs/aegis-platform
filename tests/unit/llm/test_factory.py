from unittest.mock import patch

import pytest
from pydantic import ValidationError

from aegis_platform.config.settings import Settings
from aegis_platform.llm.factory import LLMProviderFactory


def test_factory_creates_openai_provider() -> None:
    settings = Settings(
        llm_provider="OpenAI",
        openai_api_key="test-key",
        llm_model="test-model",
        llm_timeout_seconds=12.0,
    )
    with patch("aegis_platform.llm.factory.OpenAIProvider") as provider:
        result = LLMProviderFactory.create_llm_provider(settings)

    assert result is provider.return_value
    provider.assert_called_once_with(api_key="test-key", model="test-model", timeout_seconds=12.0)


def test_factory_creates_nvidia_provider() -> None:
    settings = Settings(
        llm_provider="NVIDIA",
        nvidia_api_key="test-key",
        nvidia_model="test-model",
        llm_timeout_seconds=12.0,
    )
    with patch("aegis_platform.llm.factory.NvidiaProvider") as provider:
        result = LLMProviderFactory.create_llm_provider(settings)

    assert result is provider.return_value
    provider.assert_called_once_with(api_key="test-key", model="test-model", timeout_seconds=12.0)


def test_factory_requires_openai_key() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="OpenAI API key is not configured"):
        LLMProviderFactory.create_llm_provider(settings)


def test_factory_rejects_unsupported_provider() -> None:
    settings = Settings(llm_provider="unknown")
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown"):
        LLMProviderFactory.create_llm_provider(settings)


def test_settings_reject_invalid_timeout() -> None:
    with pytest.raises(ValidationError, match="llm_timeout_seconds"):
        Settings(llm_timeout_seconds="not-a-number")  # type: ignore[arg-type]


def test_settings_retry_defaults_match_previous_policy() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.llm_retry_max_attempts == 3
    assert settings.llm_retry_deadline_seconds == 100.0
    assert settings.llm_retry_initial_delay_seconds == 1.0
    assert settings.llm_retry_max_delay_seconds == 5.0
    assert settings.llm_retry_jitter is True


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("openai", "openai-model"), (" NVIDIA ", "nvidia-model"), ("other", "openai-model")],
)
def test_settings_active_model(provider: str, expected: str) -> None:
    settings = Settings(
        llm_provider=provider, llm_model="openai-model", nvidia_model="nvidia-model"
    )

    assert settings.active_model == expected


def test_settings_reject_invalid_retry_attempts() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_retry_max_attempts=0)
