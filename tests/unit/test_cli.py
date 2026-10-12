import os
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import pytest

from muon import cli
from muon.config import Neo4jSettings, OntologySettings, Settings
from muon.graph import GraphConnectionError, WriteSummary
from muon.ingestion.bulbapedia.api import BulbapediaAPIError
from muon.ingestion.bulbapedia.fetch import FetchError, FetchReport, FetchRequest
from muon.ingestion.bulbapedia.pages import PageStoreError
from muon.ingestion.bulbapedia.species.extract import ResidueKind
from muon.ingestion.bulbapedia.species.forms import OverrideError
from muon.ingestion.bulbapedia.species.run import (
    BatchCounts,
    PageResult,
    PageStatus,
    ParseReport,
    ParseRequest,
    ParseRun,
    ParseRunError,
    ResidueItem,
)
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


FETCH_REPORT = FetchReport(
    page_type="species",
    listed=3,
    fetched=1,
    skipped=2,
    added=("Mew (Pokémon)",),
    changed=(),
    removed=("Bulbasaur (Pokémon)",),
    redirects=(("Mew", "Mew (Pokémon)"),),
    requests=4,
    elapsed_seconds=2.25,
)


class RecordingFetch:
    """Stands in for `cli.run_fetch` and records the requests it ran."""

    def __init__(self, outcome: FetchReport | Exception) -> None:
        self.outcome = outcome
        self.requests: list[FetchRequest] = []

    def __call__(self, request: FetchRequest) -> FetchReport:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def arrange_fetch(
    monkeypatch: pytest.MonkeyPatch, outcome: FetchReport | Exception
) -> RecordingFetch:
    fetch = RecordingFetch(outcome)
    monkeypatch.setattr(cli, "run_fetch", fetch)
    return fetch


def test_fetch_parses_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = arrange_fetch(monkeypatch, FETCH_REPORT)

    cli.main(["fetch", "bulbapedia", "ability", "--title", "A (Ability)", "--title", "B (Ability)"])
    cli.main(["fetch", "bulbapedia", "species", "--update"])

    assert fetch.requests == [
        FetchRequest(page_type="ability", titles=("A (Ability)", "B (Ability)")),
        FetchRequest(page_type="species", is_update=True),
    ]


def test_fetch_runs_without_neo4j_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)

    def fail_on_settings() -> Settings:
        raise AssertionError("fetch must not read the Neo4j settings")

    monkeypatch.setattr(cli, "get_settings", fail_on_settings)
    arrange_fetch(monkeypatch, FETCH_REPORT)

    exit_code = cli.main(["fetch", "bulbapedia", "species"])

    assert exit_code == cli.EXIT_SUCCESS
    assert "fetch bulbapedia species: done" in capsys.readouterr().out


def test_fetch_prints_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arrange_fetch(monkeypatch, FETCH_REPORT)

    cli.main(["fetch", "bulbapedia", "species"])

    assert capsys.readouterr().out.splitlines() == [
        "fetch bulbapedia species: done",
        "listed: 3",
        "fetched: 1",
        "skipped: 2",
        "added: 1",
        "changed: 0",
        "removed: 1",
        "redirects: 1",
        "requests: 4",
        "elapsed seconds: 2.2",
    ]


def test_fetch_update_lists_the_titles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arrange_fetch(monkeypatch, FETCH_REPORT)

    cli.main(["fetch", "bulbapedia", "species", "--update"])

    assert capsys.readouterr().out.splitlines()[-3:] == [
        "added: Mew (Pokémon)",
        "removed: Bulbasaur (Pokémon)",
        "redirect: Mew -> Mew (Pokémon)",
    ]


@pytest.mark.parametrize(
    ("argv", "invalid"),
    [
        (["fetch", "bulbapedia", "moves"], "'moves'"),
        (["fetch", "serebii", "species"], "'serebii'"),
    ],
)
def test_fetch_rejects_unknown_choices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    invalid: str,
) -> None:
    fetch = arrange_fetch(monkeypatch, FETCH_REPORT)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    error = capsys.readouterr().err
    assert exit_info.value.code == 2
    assert f"invalid choice: {invalid}" in error
    assert fetch.requests == []


def test_fetch_lists_the_valid_page_types(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["fetch", "bulbapedia", "moves"])

    error = capsys.readouterr().err
    assert all(page_type in error for page_type in ("ability", "item", "species"))


@pytest.mark.parametrize(
    "error",
    [
        BulbapediaAPIError("GET api.php for titles=X failed with HTTP 403"),
        FetchError("Category:Pokémon lists no page ending in ' (Pokémon)'"),
        PageStoreError("Cannot read manifest data/bulbapedia/manifest.json"),
    ],
)
def test_fetch_failure_prints_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    arrange_fetch(monkeypatch, error)

    exit_code = cli.main(["fetch", "bulbapedia", "species"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert f"fetch bulbapedia species failed: {error}" in captured.err
    assert captured.out == ""


def build_counts(owners: int, stubs: int, edges: int) -> BatchCounts:
    return BatchCounts(
        owner_nodes={"PokemonSpecies": owners}, stubs={"Type": stubs}, edges={"hasType": edges}
    )


def build_parse_report(failed_titles: tuple[str, ...] = ()) -> ParseReport:
    pages = (
        PageResult(
            title="Diglett (Pokémon)",
            page_id=2482,
            revision_id=4630045,
            status=PageStatus.PARSED,
            reason=None,
        ),
        PageResult(
            title="Browt (Pokémon)",
            page_id=352932,
            revision_id=4538484,
            status=PageStatus.SKIPPED,
            reason="no Pokémon Infobox",
        ),
        *(
            PageResult(
                title=title,
                page_id=1,
                revision_id=2,
                status=PageStatus.FAILED,
                reason="no A-Box node for type1",
            )
            for title in failed_titles
        ),
    )
    residue = (
        ResidueItem(
            id="2482:evolution:1",
            page_title="Diglett (Pokémon)",
            revision_id=4630045,
            kind=ResidueKind.EVOLUTION,
            section_path="Game data > Evolution data",
            form_ids=(),
            wikitext="x",
            text="x",
        ),
    )
    return ParseReport(
        pages=pages,
        pages_parsed=1,
        pages_skipped=1,
        pages_failed=len(failed_titles),
        species_counts=build_counts(2, 5, 17),
        evolution_counts=build_counts(1, 2, 3),
        chunk_counts=build_counts(4, 6, 8),
        chunks_per_aspect={"biology": 3, "trivia": 1},
        dropped_templates=(),
        pages_without_chunks=(),
        form_candidates=(),
        overrides_applied=(),
        notes=(),
        unowned_evolution_pairs=(),
        pages_without_learnset=(),
        evolutions_total=1,
        evolutions_structured=1,
        evolution_rule_share=1.0,
        mega_forms_with_stone=0,
        mega_form_changes=0,
        abox_stub_misses=0,
        evolution_species_stub_misses=0,
        residue=residue,
    )


class RecordingParse:
    """Stands in for `cli.run_parse` and records the requests it ran."""

    def __init__(self, outcome: ParseReport | Exception) -> None:
        self.outcome = outcome
        self.requests: list[ParseRequest] = []

    def __call__(self, request: ParseRequest) -> ParseRun:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return ParseRun(outputs=(), report=self.outcome)


def arrange_parse(
    monkeypatch: pytest.MonkeyPatch, outcome: ParseReport | Exception
) -> RecordingParse:
    parse = RecordingParse(outcome)
    monkeypatch.setattr(cli, "run_parse", parse)
    return parse


def test_parse_parses_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    parse = arrange_parse(monkeypatch, build_parse_report())

    cli.main(["parse", "bulbapedia", "species"])
    cli.main(["parse", "bulbapedia", "species", "--title", "A (Pokémon)", "--title", "B (Pokémon)"])

    assert parse.requests == [
        ParseRequest(),
        ParseRequest(titles=("A (Pokémon)", "B (Pokémon)")),
    ]


def test_parse_runs_without_neo4j_settings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)

    def fail_on_settings() -> Settings:
        raise AssertionError("parse must not read the Neo4j settings")

    monkeypatch.setattr(cli, "get_settings", fail_on_settings)
    arrange_parse(monkeypatch, build_parse_report())

    exit_code = cli.main(["parse", "bulbapedia", "species"])

    assert exit_code == cli.EXIT_SUCCESS
    assert "parse bulbapedia species: done" in capsys.readouterr().out


def test_parse_prints_counts_and_residue_per_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arrange_parse(monkeypatch, build_parse_report())

    cli.main(["parse", "bulbapedia", "species"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[:-1] == [
        "parse bulbapedia species: done",
        "pages parsed: 1",
        "pages skipped: 1",
        "pages failed: 0",
        "species owner nodes: 2",
        "species stubs: 5",
        "species edges: 17",
        "evolution owner nodes: 1",
        "evolution stubs: 2",
        "evolution edges: 3",
        "chunk owner nodes: 4",
        "chunk stubs: 6",
        "chunk edges: 8",
        "residue evolution: 1",
    ]
    assert lines[-1].startswith("elapsed seconds: ")


def test_parse_exits_1_when_a_page_failed_and_names_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arrange_parse(monkeypatch, build_parse_report(failed_titles=("Gamma (Pokémon)",)))

    exit_code = cli.main(["parse", "bulbapedia", "species"])

    assert exit_code == cli.EXIT_FAILURE
    assert "failed: Gamma (Pokémon): no A-Box node for type1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        ParseRunError("Titles that are not pinned: ['Ghost']"),
        OverrideError("FORM_OVERRIDES names pages that are not pinned: ['Ghost']"),
        OntologyError("Missing universe T-Box"),
        ABoxError("pkmn.ttl", ["bad"]),
        PageStoreError("No page file"),
    ],
)
def test_parse_reports_run_errors_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    arrange_parse(monkeypatch, error)

    exit_code = cli.main(["parse", "bulbapedia", "species"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert captured.err.startswith("parse bulbapedia species failed:")
    assert captured.out == ""


@pytest.mark.parametrize(
    ("argv", "invalid"),
    [
        (["parse", "bulbapedia", "ability"], "'ability'"),
        (["parse", "serebii", "species"], "'serebii'"),
    ],
)
def test_parse_rejects_unknown_choices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    invalid: str,
) -> None:
    parse = arrange_parse(monkeypatch, build_parse_report())

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert f"invalid choice: {invalid}" in capsys.readouterr().err
    assert parse.requests == []
