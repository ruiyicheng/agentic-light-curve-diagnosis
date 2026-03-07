"""Centralized configuration management.

Uses pydantic-settings for environment variable loading and validation.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra env vars not defined here
    )

    # Directory paths
    artifacts_dir: Path = Path("./artifacts")
    data_dir: Path = Path("./data")
    skills_dir: Path = Path("./skills")

    # OpenAI configuration
    openai_model: str = "gpt-4o"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    # Analysis parameters
    default_period_min_days: float = 10 / 24 / 60  # 10 minutes
    default_period_max_days: float = 2000.0
    default_period_grid_size: int = 1_000_000
    default_chunk_size: int = 8192

    # Plotting defaults
    default_dpi: int = 200
    default_figure_format: str = "png"

    # BLS defaults
    bls_default_duration: float = 0.2  # days

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Use lru_cache to avoid reloading .env on every call.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
