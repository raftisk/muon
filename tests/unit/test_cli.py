import os
from dataclasses import fields

import pytest

from muon import cli
from muon.config import Neo4jSettings, Settings
from muon.graph import GraphConnectionError
from muon.ingestion.runner import RunReport
from muon.ingestion.seed import ABoxError
from muon.ontology.model import OntologyError

PREFIX = "muontoy"
SECRET = "secret-check"
CONNECTION_MESSAGE = "Failed to connect to Neo4j at uri='neo4j://x' database='neo4j'"
REPORT = RunReport(
    constraints_ensured=2,
    nodes_created=6,
    nodes_deleted=0,
    relationships_created=5,
    relationships_deleted=0,
    labels_added=24,
    labels_removed=0,
    properties_set=30,
)


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    loaded = Settings(
        neo4j=Neo4jSettings(uri="neo4j://x", username="neo4j", password=SECRET, _env_file=None)
    )
    monkeypatch.setattr(cli, "get_settings", lambda: loaded)
    return loaded


def fake_run_seed(outcome: RunReport | Exception):
    async def run(settings: Settings, prefix: str) -> RunReport:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return run


@pytest.mark.usefixtures("settings")
def test_seed_prints_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "run_seed", fake_run_seed(REPORT))

    exit_code = cli.main(["seed", PREFIX])

    output = capsys.readouterr().out
    assert exit_code == cli.EXIT_SUCCESS
    assert f"seed {PREFIX}: done" in output
    for field in fields(RunReport):
        assert f"{field.name.replace('_', ' ')}: {getattr(REPORT, field.name)}" in output


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    "error",
    [
        ABoxError("individuals/muontoy.ttl", ["muontoy:x: expected 1 rdf:type, found 0"]),
        OntologyError("Missing universe T-Box universes/muontoy.ttl"),
        GraphConnectionError(CONNECTION_MESSAGE),
    ],
)
def test_seed_failure_prints_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    monkeypatch.setattr(cli, "run_seed", fake_run_seed(error))

    exit_code = cli.main(["seed", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert f"seed {PREFIX} failed: {error}" in captured.err
    assert SECRET not in captured.err
    assert captured.out == ""


def test_invalid_settings_print_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "get_settings", lambda: Neo4jSettings(_env_file=None))

    exit_code = cli.main(["seed", PREFIX])

    assert exit_code == cli.EXIT_FAILURE
    assert "invalid settings" in capsys.readouterr().err


def test_missing_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2
