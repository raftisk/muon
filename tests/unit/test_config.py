import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from muon.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    EmbeddingSettings,
    LLMSettings,
    Neo4jSettings,
    OntologySettings,
    Settings,
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

    assert set(Settings.model_fields) == {"neo4j", "ontology"}
    assert settings.neo4j.uri == "neo4j://env-host:7687"
