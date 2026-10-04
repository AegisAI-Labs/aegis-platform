from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # Provider credentials
    openai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
