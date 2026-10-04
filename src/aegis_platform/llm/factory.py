from aegis_platform.config.settings import Settings
from aegis_platform.llm.providers.openai import OpenAIProvider

from .base import LLMProvider


class LLMProviderFactory:
    @staticmethod
    def create_llm_provider(settings: Settings) -> LLMProvider:
        """
        Create an instance of the specified LLM provider.

        Args:
            settings (Settings): The application settings containing LLM provider configuration.

        Returns:
            LLMProvider: An instance of the requested LLM provider.

        Raises:
            ValueError: If the specified provider is not supported.
        """
        provider = settings.llm_provider.lower()  # Normalize the provider name

        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OpenAI API key is not configured in settings for LLM provider = openai"
                )
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.llm_model,
                timeout_seconds=settings.llm_timeout_seconds,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")
