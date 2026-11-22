"""Application settings.

Values load from environment variables first and the repo-root `.env` second, so an
exported variable always overrides the file. Each layer owns one settings group.
"""

import functools
from enum import StrEnum
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

# The values mirror `openai.types.shared.ReasoningEffort`.
ReasoningLevel = Literal["none", "low", "medium", "high", "xhigh", "max"]
DEFAULT_REASONING: ReasoningLevel = "none"


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
    """Where requests to 1 OpenAI-compatible endpoint go: URL and key. Subclasses add the model."""

    base_url: str
    api_key: SecretStr
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS  # seconds
    max_retries: int = DEFAULT_MAX_RETRIES


class LLMSettings(ProviderSettings):
    """Chat provider, read from `MUON_LLM_*` variables."""

    model_config = SettingsConfigDict(env_prefix="MUON_LLM_", env_file=ENV_FILE, extra="ignore")

    model: str


class EmbeddingSettings(ProviderSettings):
    """Embedding provider, read from `MUON_EMBEDDING_*` variables.

    `dimensions` is `None` when unset, so callers omit the parameter for models that
    reject it.
    """

    model_config = SettingsConfigDict(
        env_prefix="MUON_EMBEDDING_", env_file=ENV_FILE, extra="ignore"
    )

    model: str
    dimensions: int | None = Field(default=None, gt=0)


class AgentStep(StrEnum):
    """The model steps of `ask()`, in pipeline order."""

    RESOLVE = "resolve"
    RETRIEVE = "retrieve"
    GENERATE = "generate"


class AgentSettings(ProviderSettings):
    """Agent provider and loop limits, read from `MUON_AGENT_*` variables.

    `max_tool_calls` caps all tool calls of 1 question, `max_cypher_calls` caps the
    `run_cypher` calls.

    The 3 steps are `AgentStep.RESOLVE`, `RETRIEVE` and `GENERATE`. `resolve_model`,
    `retrieve_model` and `generate_model` override `default_model` for their step, and an
    unset one falls back to `default_model`. Each reasoning level defaults to `"none"`, so
    the provider default never applies. Read them through `model_for` and `reasoning_for`.
    """

    model_config = SettingsConfigDict(env_prefix="MUON_AGENT_", env_file=ENV_FILE, extra="ignore")

    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, gt=0)
    max_cypher_calls: int = Field(default=DEFAULT_MAX_CYPHER_CALLS, gt=0)
    default_model: str
    resolve_model: str | None = None
    retrieve_model: str | None = None
    generate_model: str | None = None
    resolve_reasoning: ReasoningLevel = DEFAULT_REASONING
    retrieve_reasoning: ReasoningLevel = DEFAULT_REASONING
    generate_reasoning: ReasoningLevel = DEFAULT_REASONING

    def model_for(self, step: AgentStep) -> str:
        """Return the model of `step`, or `default_model` when the step sets none."""
        match step:
            case AgentStep.RESOLVE:
                return self.resolve_model or self.default_model
            case AgentStep.RETRIEVE:
                return self.retrieve_model or self.default_model
            case AgentStep.GENERATE:
                return self.generate_model or self.default_model

    def reasoning_for(self, step: AgentStep) -> ReasoningLevel:
        match step:
            case AgentStep.RESOLVE:
                return self.resolve_reasoning
            case AgentStep.RETRIEVE:
                return self.retrieve_reasoning
            case AgentStep.GENERATE:
                return self.generate_reasoning


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
