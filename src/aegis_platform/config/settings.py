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


settings = Settings()
