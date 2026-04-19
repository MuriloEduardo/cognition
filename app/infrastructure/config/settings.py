from pydantic import AmqpDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rabbitmq_url: AmqpDsn
    rabbitmq_prefetch_count: int = 10
    rabbitmq_reconnect_delay: float = 5.0
    rabbitmq_max_retries: int = 5

    app_name: str = "cognition"
    log_level: str = "INFO"

    # LLM Configuration
    openai_api_key: str
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.7
    default_max_tokens: int = 1024
