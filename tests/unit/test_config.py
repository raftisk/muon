import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from muon.config import PROJECT_ROOT, Neo4jSettings, OntologySettings, get_settings

ENV_PREFIXES = ("NEO4J_", "MUON_")
PASSWORD = "file-secret"


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
