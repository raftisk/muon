"""Application settings.

Values load from environment variables first and the repo-root `.env` second, so an
exported variable always overrides the file. Each layer owns one settings group.
"""

import functools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
DEFAULT_UNIVERSES_DIR = PROJECT_ROOT / "muon" / "universes"
DEFAULT_SCHEMA_CACHE_DIR = PROJECT_ROOT / "data" / "schema"

DEFAULT_DATABASE = "neo4j"
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONNECTION_POOL_SIZE = 100

# Equal to the openai SDK defaults, so unset values behave like a plain client.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_RETRIES = 2

DEFAULT_MAX_TOOL_CALLS = 10
DEFAULT_MAX_CYPHER_CALLS = 2
DEFAULT_MAX_CHECK_RETRIES = 1

# The values mirror `openai.types.shared.ReasoningEffort`.
ReasoningLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


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


class UniverseSettings(BaseSettings):
    """Locations of the universe plugins and the schema cache, read from `MUON_*` variables."""

    model_config = SettingsConfigDict(env_prefix="MUON_", env_file=ENV_FILE, extra="ignore")

    universes_dir: Path = DEFAULT_UNIVERSES_DIR
    schema_cache_dir: Path = DEFAULT_SCHEMA_CACHE_DIR


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


class AgentSettings(ProviderSettings):
    """Agent provider and loop limits, read from `MUON_AGENT_*` variables.

    `max_tool_calls` caps all tool calls of 1 question, `max_cypher_calls` caps the
    `run_cypher` calls, and `max_check_retries` is the number of generator reruns after
    a failed citation check (0 disables the rerun).

    `retrieve_model` and `generate_model` override `model` for their step, and an unset
    one falls back to `model`. A `None` reasoning level sends no `reasoning_effort`, so
    the provider default applies.
    """

    model_config = SettingsConfigDict(env_prefix="MUON_AGENT_", env_file=ENV_FILE, extra="ignore")

    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)
    max_cypher_calls: int = Field(default=DEFAULT_MAX_CYPHER_CALLS, gt=0)
    max_check_retries: int = Field(default=DEFAULT_MAX_CHECK_RETRIES, ge=0)
    retrieve_model: str | None = None
    generate_model: str | None = None
    retrieve_reasoning: ReasoningLevel | None = None
    generate_reasoning: ReasoningLevel | None = None

    def resolve_retrieve_model(self) -> str:
        return self.retrieve_model or self.model

    def resolve_generate_model(self) -> str:
        return self.generate_model or self.model


class Settings(BaseModel):
    """Settings every graph command needs.

    `LLMSettings`, `EmbeddingSettings` and `AgentSettings` stay out, so a command that
    calls no model runs without `MUON_LLM_*`, `MUON_EMBEDDING_*` or `MUON_AGENT_*`
    variables.
    """

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    ontology: OntologySettings = Field(default_factory=OntologySettings)
    universes: UniverseSettings = Field(default_factory=UniverseSettings)


@functools.cache
def get_settings() -> Settings:
    return Settings()
