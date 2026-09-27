"""Application settings.

Values load from environment variables first and the repo-root `.env` second, so an
exported variable always overrides the file. Each layer owns one settings group.
"""

import functools
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_DATABASE = "neo4j"
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONNECTION_POOL_SIZE = 100


class Neo4jSettings(BaseSettings):
    """Connection settings for the Neo4j instance, read from `NEO4J_*` variables."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=ENV_FILE, extra="ignore")

    uri: str
    username: str
    password: SecretStr
    database: str = DEFAULT_DATABASE
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS  # seconds
    max_connection_pool_size: int = DEFAULT_MAX_CONNECTION_POOL_SIZE


class Settings(BaseModel):
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)


@functools.cache
def get_settings() -> Settings:
    return Settings()
