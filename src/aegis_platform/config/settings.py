from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openai"

    openai_api_key: str = Field(
        default="",
        repr=False,
    )

    llm_model: str = "gpt-4o-mini"

    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )


settings = Settings()
