import os
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from muon import cli
from muon.config import Neo4jSettings, OntologySettings, Settings
from muon.graph import GraphConnectionError, WriteSummary
from muon.ingestion.runner import RunReport
from muon.ingestion.seed import ABoxError
from muon.ontology.model import OntologyError

PREFIX = "muontoy"
SECRET = "secret-check"
WRONG_PASSWORD = "wrong-password"
NON_ASCII_PASSWORD = "pässwörd-ß"
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

PURGE_SUMMARY = WriteSummary(
    nodes_created=0,
    nodes_deleted=3,
    relationships_created=0,
    relationships_deleted=2,
    labels_added=0,
    labels_removed=0,
    properties_set=0,
)
EMPTY_PURGE_SUMMARY = WriteSummary(
    nodes_created=0,
    nodes_deleted=0,
    relationships_created=0,
    relationships_deleted=0,
    labels_added=0,
    labels_removed=0,
    properties_set=0,
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


class RecordingPurge:
    """Stands in for `cli.run_purge` and records the prefixes it ran for."""

    def __init__(self, outcome: WriteSummary | Exception) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    async def __call__(self, settings: Settings, prefix: str) -> WriteSummary:
        self.calls.append(prefix)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_prompt(reply: str | BaseException) -> Callable[[], str]:
    def prompt() -> str:
        if isinstance(reply, BaseException):
            raise reply
        return reply

    return prompt


def arrange_purge(
    monkeypatch: pytest.MonkeyPatch, reply: str | BaseException, outcome: WriteSummary | Exception
) -> RecordingPurge:
    purge = RecordingPurge(outcome)
    monkeypatch.setattr(cli, "prompt_password", fake_prompt(reply))
    monkeypatch.setattr(cli, "run_purge", purge)
    return purge


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    ("summary", "nodes_deleted", "relationships_deleted"),
    [(PURGE_SUMMARY, 3, 2), (EMPTY_PURGE_SUMMARY, 0, 0)],
)
def test_purge_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    summary: WriteSummary,
    nodes_deleted: int,
    relationships_deleted: int,
) -> None:
    purge = arrange_purge(monkeypatch, SECRET, summary)

    exit_code = cli.main(["purge", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert purge.calls == [PREFIX]
    assert captured.out.splitlines() == [
        f"purge {PREFIX}: done",
        f"nodes deleted: {nodes_deleted}",
        f"relationships deleted: {relationships_deleted}",
    ]
    assert SECRET not in captured.out + captured.err


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize("typed", [WRONG_PASSWORD, "", NON_ASCII_PASSWORD])
def test_purge_wrong_password_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], typed: str
) -> None:
    purge = arrange_purge(monkeypatch, typed, PURGE_SUMMARY)

    exit_code = cli.main(["purge", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert purge.calls == []
    assert f"purge {PREFIX} failed:" in captured.err
    assert captured.out == ""
    assert SECRET not in captured.err
    if typed:
        assert typed not in captured.err


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
def test_purge_aborted_prompt_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    interruption: BaseException,
) -> None:
    purge = arrange_purge(monkeypatch, interruption, PURGE_SUMMARY)

    exit_code = cli.main(["purge", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert purge.calls == []
    assert f"purge {PREFIX} aborted" in captured.err
    assert captured.out == ""


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    "error",
    [
        OntologyError("Missing universe T-Box universes/muontoy.ttl"),
        OntologyError("Invalid universe prefix 'Bad Prefix': expected ^[a-z][a-z0-9]*$"),
        GraphConnectionError(CONNECTION_MESSAGE),
    ],
)
def test_purge_failure_prints_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    arrange_purge(monkeypatch, SECRET, error)

    exit_code = cli.main(["purge", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert f"purge {PREFIX} failed: {error}" in captured.err
    assert SECRET not in captured.err
    assert captured.out == ""


def test_purge_invalid_settings_print_error_before_prompting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "get_settings", lambda: Neo4jSettings(_env_file=None))
    purge = arrange_purge(monkeypatch, AssertionError("the prompt must not open"), PURGE_SUMMARY)

    exit_code = cli.main(["purge", PREFIX])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert purge.calls == []
    assert f"purge {PREFIX} failed: invalid settings" in captured.err


async def test_run_purge_with_missing_tbox_opens_no_client(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    def fail_on_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("no client may open before the T-Box loads")

    monkeypatch.setattr(cli, "Neo4jGraphClient", fail_on_client)
    empty_dir_settings = settings.model_copy(
        update={"ontology": OntologySettings(ontology_dir=tmp_path, _env_file=None)}
    )

    with pytest.raises(OntologyError):
        await cli.run_purge(empty_dir_settings, PREFIX)
