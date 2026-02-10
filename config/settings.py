from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://salesbot:salesbot_secure_2026@127.0.0.1:5432/salesbot"

    # Redis
    redis_url: str = "redis://:redis_secure_2026@127.0.0.1:6379/0"

    # GigaChat
    gigachat_client_id: str = ""
    gigachat_auth_key: str = ""
    gigachat_scope: str = "GIGACHAT_API_PERS"
    gigachat_auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    gigachat_api_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model: str = "GigaChat"

    # Telegram (main chat bot)
    telegram_bot_token: str = ""

    # Monitoring
    monitoring_bot_token: str = ""
    monitoring_admin_chat_id: str = ""
    monitoring_interval: int = 300  # seconds between checks

    # Gateway
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000
    jwt_secret: str = "CHANGE_ME_TO_RANDOM_SECRET"
    rate_limit_per_minute: int = 30

    # Sentry
    sentry_dsn: str = ""

    # Debug
    debug: bool = False
    log_level: str = "INFO"

    # Cursor API (Repair Agent)
    cursor_api_key: str = ""
    cursor_api_url: str = "https://api.cursor.com/v0"
    github_repo_url: str = ""
    github_repo_branch: str = "main"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
