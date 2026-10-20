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
DEFAULT_ONTOLOGY_DIR = PROJECT_ROOT / "ontology"

DEFAULT_DATABASE = "neo4j"
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONNECTION_POOL_SIZE = 100

# Equal to the openai SDK defaults, so unset values behave like a plain client.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 2


class Neo4jSettings(BaseSettings):
    """Connection settings for the Neo4j instance, read from `NEO4J_*` variables."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=ENV_FILE, extra="ignore")

    uri: str
    username: str
    password: SecretStr
    database: str = DEFAULT_DATABASE
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS  # seconds
    max_connection_pool_size: int = DEFAULT_MAX_CONNECTION_POOL_SIZE


class OntologySettings(BaseSettings):
    """Location of the ontology files, read from `MUON_*` variables."""

    model_config = SettingsConfigDict(env_prefix="MUON_", env_file=ENV_FILE, extra="ignore")

    ontology_dir: Path = DEFAULT_ONTOLOGY_DIR


class ProviderSettings(BaseSettings):
    """Where requests to 1 OpenAI-compatible endpoint go: URL, key and model."""

    base_url: str
    api_key: SecretStr
    model: str
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS  # seconds
    max_retries: int = DEFAULT_MAX_RETRIES


class LLMSettings(ProviderSettings):
    """Chat provider, read from `MUON_LLM_*` variables."""

    model_config = SettingsConfigDict(env_prefix="MUON_LLM_", env_file=ENV_FILE, extra="ignore")


class EmbeddingSettings(ProviderSettings):
    """Embedding provider, read from `MUON_EMBEDDING_*` variables.

    `dimensions` is `None` when unset, so callers omit the parameter for models that
    reject it.
    """

    model_config = SettingsConfigDict(
        env_prefix="MUON_EMBEDDING_", env_file=ENV_FILE, extra="ignore"
    )

    dimensions: int | None = Field(default=None, gt=0)


class Settings(BaseModel):
    """Settings every graph command needs.

    `LLMSettings` and `EmbeddingSettings` stay out, so a command that calls no model
    runs without `MUON_LLM_*` or `MUON_EMBEDDING_*` variables.
    """

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    ontology: OntologySettings = Field(default_factory=OntologySettings)


@functools.cache
def get_settings() -> Settings:
    return Settings()
