import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from muon.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SCHEMA_CACHE_DIR,
    DEFAULT_UNIVERSES_DIR,
    PROJECT_ROOT,
    AgentSettings,
    EmbeddingSettings,
    LLMSettings,
    Neo4jSettings,
    OntologySettings,
    Settings,
    UniverseSettings,
    get_settings,
)

ENV_PREFIXES = ("NEO4J_", "MUON_")
PASSWORD = "file-secret"
LLM_KEY = "llm-secret"
EMBEDDING_KEY = "embedding-secret"


@pytest.fixture(autouse=True)
def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(ENV_PREFIXES):
            monkeypatch.delenv(name)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "NEO4J_URI=neo4j://file-host:7687\n"
        "NEO4J_USERNAME=file-user\n"
        f"NEO4J_PASSWORD={PASSWORD}\n"
        "NEO4J_DATABASE=filedb\n"
    )
    return path


@pytest.fixture
def fresh_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_loads_values_from_env_file(env_file: Path) -> None:
    settings = Neo4jSettings(_env_file=env_file)

    assert settings.uri == "neo4j://file-host:7687"
    assert settings.username == "file-user"
    assert settings.password.get_secret_value() == PASSWORD
    assert settings.database == "filedb"


def test_password_hidden_in_repr(env_file: Path) -> None:
    settings = Neo4jSettings(_env_file=env_file)

    assert PASSWORD not in repr(settings)


def test_env_var_overrides_env_file(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j://env-host:7687")

    settings = Neo4jSettings(_env_file=env_file)

    assert settings.uri == "neo4j://env-host:7687"


def test_missing_required_variable_names_it(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("NEO4J_URI=neo4j://file-host:7687\nNEO4J_USERNAME=file-user\n")

    with pytest.raises(ValidationError, match="password"):
        Neo4jSettings(_env_file=path)


@pytest.mark.usefixtures("fresh_settings_cache")
def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j://env-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")

    assert get_settings() is get_settings()


def test_ontology_dir_defaults_to_repo_ontology(tmp_path: Path) -> None:
    settings = OntologySettings(_env_file=tmp_path / ".env")

    assert settings.ontology_dir == PROJECT_ROOT / "ontology"


def test_ontology_dir_env_var_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUON_ONTOLOGY_DIR", str(tmp_path))

    settings = OntologySettings(_env_file=tmp_path / ".env")

    assert settings.ontology_dir == tmp_path


@pytest.fixture
def provider_env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "MUON_LLM_BASE_URL=https://chat.test/v1\n"
        f"MUON_LLM_API_KEY={LLM_KEY}\n"
        "MUON_LLM_MODEL=chat-model\n"
        "MUON_EMBEDDING_BASE_URL=https://embed.test/v1\n"
        f"MUON_EMBEDDING_API_KEY={EMBEDDING_KEY}\n"
        "MUON_EMBEDDING_MODEL=embed-model\n"
    )
    return path


def test_llm_settings_load_from_env_file(provider_env_file: Path) -> None:
    settings = LLMSettings(_env_file=provider_env_file)

    assert settings.base_url == "https://chat.test/v1"
    assert settings.api_key.get_secret_value() == LLM_KEY
    assert settings.model == "chat-model"


def test_provider_settings_default_to_sdk_timeout_and_retries(provider_env_file: Path) -> None:
    settings = LLMSettings(_env_file=provider_env_file)

    assert settings.timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert settings.max_retries == DEFAULT_MAX_RETRIES


def test_provider_api_keys_hidden_in_repr(provider_env_file: Path) -> None:
    llm = LLMSettings(_env_file=provider_env_file)
    embedding = EmbeddingSettings(_env_file=provider_env_file)

    assert LLM_KEY not in repr(llm)
    assert EMBEDDING_KEY not in repr(embedding)


def test_missing_llm_model_names_it(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("MUON_LLM_BASE_URL=https://chat.test/v1\nMUON_LLM_API_KEY=key\n")

    with pytest.raises(ValidationError, match="model"):
        LLMSettings(_env_file=path)


def test_embedding_dimensions_unset_is_none(provider_env_file: Path) -> None:
    settings = EmbeddingSettings(_env_file=provider_env_file)

    assert settings.dimensions is None


def test_embedding_dimensions_read_from_env(
    provider_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUON_EMBEDDING_DIMENSIONS", "1024")

    settings = EmbeddingSettings(_env_file=provider_env_file)

    assert settings.dimensions == 1024


def test_embedding_dimensions_reject_zero(
    provider_env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUON_EMBEDDING_DIMENSIONS", "0")

    with pytest.raises(ValidationError, match="dimensions"):
        EmbeddingSettings(_env_file=provider_env_file)


def test_chat_and_embeddings_load_separate_providers(provider_env_file: Path) -> None:
    llm = LLMSettings(_env_file=provider_env_file)
    embedding = EmbeddingSettings(_env_file=provider_env_file)

    assert (llm.base_url, llm.api_key.get_secret_value()) == ("https://chat.test/v1", LLM_KEY)
    assert (embedding.base_url, embedding.api_key.get_secret_value()) == (
        "https://embed.test/v1",
        EMBEDDING_KEY,
    )


def test_settings_aggregate_holds_no_provider_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j://env-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")

    settings = Settings()

    assert set(Settings.model_fields) == {"neo4j", "ontology", "universes"}
    assert settings.neo4j.uri == "neo4j://env-host:7687"


def test_universe_settings_default_to_repo_folders(tmp_path: Path) -> None:
    settings = UniverseSettings(_env_file=tmp_path / ".env")

    assert settings.universes_dir == DEFAULT_UNIVERSES_DIR == PROJECT_ROOT / "muon" / "universes"
    assert settings.schema_cache_dir == DEFAULT_SCHEMA_CACHE_DIR == PROJECT_ROOT / "data" / "schema"


def test_universe_env_vars_override_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MUON_UNIVERSES_DIR", str(tmp_path / "plugins"))
    monkeypatch.setenv("MUON_SCHEMA_CACHE_DIR", str(tmp_path / "cache"))

    settings = UniverseSettings(_env_file=tmp_path / ".env")

    assert settings.universes_dir == tmp_path / "plugins"
    assert settings.schema_cache_dir == tmp_path / "cache"


def test_settings_build_with_only_neo4j_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j://env-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")

    settings = Settings()

    assert settings.universes.universes_dir == DEFAULT_UNIVERSES_DIR


def build_agent_settings(**overrides: int) -> AgentSettings:
    return AgentSettings(
        _env_file=None,
        base_url="https://agent.test/v1",
        api_key=SecretStr("k"),
        model="m",
        **overrides,
    )


def test_agent_settings_default_limits() -> None:
    settings = build_agent_settings()

    assert settings.max_tool_calls == 10
    assert settings.max_cypher_calls == 2
    assert settings.max_check_retries == 1


def test_agent_env_vars_override_the_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUON_AGENT_BASE_URL", "https://agent.test/v1")
    monkeypatch.setenv("MUON_AGENT_API_KEY", "k")
    monkeypatch.setenv("MUON_AGENT_MODEL", "m")
    monkeypatch.setenv("MUON_AGENT_MAX_CYPHER_CALLS", "4")

    assert AgentSettings(_env_file=None).max_cypher_calls == 4


def test_agent_settings_reject_zero_tool_calls() -> None:
    with pytest.raises(ValidationError):
        build_agent_settings(max_tool_calls=0)


def test_agent_settings_accept_zero_check_retries() -> None:
    assert build_agent_settings(max_check_retries=0).max_check_retries == 0


def test_settings_build_without_agent_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "neo4j://env-host:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "env-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "env-secret")

    assert "agent" not in Settings.model_fields
    Settings()
