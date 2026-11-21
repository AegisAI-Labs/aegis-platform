from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "nvidia"

    # openai configuration
    openai_api_key: str = Field(
        default="",
        repr=False,
    )

    llm_model: str = "gpt-4o-mini"

    # nvidia configuration
    nvidia_api_key: str = Field(
        default="",
        repr=False,
    )

    nvidia_model: str = "openai/gpt-oss-20b"

    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )

    # Retry policy for RetryingLLMProvider (ADR-007). Provider SDK retries are disabled,
    # so these settings are the only retry layer.
    llm_retry_max_attempts: int = Field(default=3, ge=1)  # includes the first attempt
    llm_retry_deadline_seconds: float = Field(default=100.0, gt=0)  # whole logical request
    llm_retry_initial_delay_seconds: float = Field(default=1.0, ge=0)
    llm_retry_max_delay_seconds: float = Field(default=5.0, ge=0)
    llm_retry_jitter: bool = True

    # In-memory observability window (development/test sinks).
    observability_max_records: int = Field(default=10_000, ge=1)

    @property
    def active_model(self) -> str:
        """Model configured for the selected LLM provider."""
        if self.llm_provider.strip().lower() == "nvidia":
            return self.nvidia_model
        return self.llm_model


settings = Settings()
