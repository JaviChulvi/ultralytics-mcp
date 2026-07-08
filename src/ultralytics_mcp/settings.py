"""Service configuration.

Deliberately holds no platform credential: every request must carry its own
(FR-002 — hosted multi-user service, per-request Authorization header).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ULTRALYTICS_MCP_")

    platform_base_url: str = "https://platform.ultralytics.com"
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    default_page_size: int = 20
    max_page_size: int = 50
    max_response_bytes: int = 8192
    log_level: str = "INFO"


settings = Settings()
