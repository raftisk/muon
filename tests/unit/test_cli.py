import argparse
import os
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from types import TracebackType
from typing import Any

import httpx
import pytest

from muon import cli
from muon.config import Neo4jSettings, OntologySettings, Settings
from muon.graph import GraphConnectionError, WriteSummary
from muon.ingestion.batch import NodeRef
from muon.ingestion.bulbapedia.api import BulbapediaAPIError
from muon.ingestion.bulbapedia.export import ExportError
from muon.ingestion.bulbapedia.fetch import FetchError, FetchReport, FetchRequest
from muon.ingestion.bulbapedia.moves.batches import MachineFieldError
from muon.ingestion.bulbapedia.moves.ingest import (
    MoveIngestReport,
    MoveIngestRun,
    MoveIngestStop,
    MovePass,
    MovePassTotals,
)
from muon.ingestion.bulbapedia.moves.page import TableError
from muon.ingestion.bulbapedia.moves.run import BatchCounts as MoveBatchCounts
from muon.ingestion.bulbapedia.moves.run import (
    MachineConflictError,
    MoveParseReport,
    MoveParseRequest,
    MoveParseRun,
)
from muon.ingestion.bulbapedia.moves.run import ParseRunError as MoveParseRunError
from muon.ingestion.bulbapedia.pages import PageStoreError
from muon.ingestion.bulbapedia.species.extract import ResidueKind
from muon.ingestion.bulbapedia.species.forms import OverrideError
from muon.ingestion.bulbapedia.species.ingest import (
    CensusReport,
    IngestPass,
    IngestReport,
    IngestRequest,
    IngestRun,
    IngestStop,
    MissingSeedError,
    PageFailure,
    PassProgress,
    PassTotals,
)
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
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotFetchReport
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
EXPORT_FILE = "Bulbapedia-export.xml"
EXPORT_REPORT = FetchReport(
    page_type="move",
    listed=2,
    fetched=2,
    skipped=0,
    added=("Shadow Ball (move)",),
    changed=("Tackle (move)",),
    removed=("Cut (move)",),
    redirects=(),
    requests=0,
    elapsed_seconds=0.5,
    ignored=46,
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


def test_fetch_parses_the_export_file(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = arrange_fetch(monkeypatch, EXPORT_REPORT)

    cli.main(["fetch", "bulbapedia", "move", "--from-export", EXPORT_FILE])

    assert fetch.requests == [FetchRequest(page_type="move", export_path=Path(EXPORT_FILE))]


@pytest.mark.parametrize("extra", [["--update"], ["--title", "Tackle (move)"]])
def test_fetch_rejects_export_with_update_or_title(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    fetch = arrange_fetch(monkeypatch, EXPORT_REPORT)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["fetch", "bulbapedia", "move", "--from-export", EXPORT_FILE, *extra])

    assert exit_info.value.code == 2
    assert cli.EXPORT_CONFLICT_MESSAGE in capsys.readouterr().err
    assert fetch.requests == []


def test_export_run_needs_no_neo4j_setting_and_no_http_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)

    def fail_on_settings() -> Settings:
        raise AssertionError("an export run must not read the Neo4j settings")

    def fail_on_http_client() -> None:
        raise AssertionError("an export run must not build an HTTP client")

    imports: list[FetchRequest] = []

    def record_import(request: FetchRequest, store: object) -> FetchReport:
        imports.append(request)
        return EXPORT_REPORT

    monkeypatch.setattr(cli, "get_settings", fail_on_settings)
    monkeypatch.setattr(cli, "build_http_client", fail_on_http_client)
    monkeypatch.setattr(cli, "import_pages", record_import)
    path = tmp_path / EXPORT_FILE

    exit_code = cli.main(["fetch", "bulbapedia", "move", "--from-export", str(path)])

    assert exit_code == cli.EXIT_SUCCESS
    assert imports == [FetchRequest(page_type="move", export_path=path)]
    assert "fetch bulbapedia move: done" in capsys.readouterr().out


def test_export_run_prints_ignored_and_titles(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arrange_fetch(monkeypatch, EXPORT_REPORT)

    cli.main(["fetch", "bulbapedia", "move", "--from-export", EXPORT_FILE])

    lines = capsys.readouterr().out.splitlines()
    assert "ignored: 46" in lines
    assert lines[-3:] == [
        "added: Shadow Ball (move)",
        "changed: Tackle (move)",
        "removed: Cut (move)",
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


class RecordingPokeapiFetch:
    """Stands in for `cli.run_fetch_pokeapi` and records each `is_update` it ran with."""

    def __init__(self, outcome: SnapshotFetchReport | Exception) -> None:
        self.outcome = outcome
        self.updates: list[bool] = []

    def __call__(self, is_update: bool) -> SnapshotFetchReport:
        self.updates.append(is_update)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


SNAPSHOT_REPORT = SnapshotFetchReport(commit="8a92bfa", files_downloaded=8, requests=9)


def arrange_pokeapi_fetch(
    monkeypatch: pytest.MonkeyPatch, outcome: SnapshotFetchReport | Exception
) -> RecordingPokeapiFetch:
    fetch = RecordingPokeapiFetch(outcome)
    monkeypatch.setattr(cli, "run_fetch_pokeapi", fetch)
    return fetch


def test_fetch_pokeapi_runs_with_and_without_update(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = arrange_pokeapi_fetch(monkeypatch, SNAPSHOT_REPORT)

    assert cli.main(["fetch", "pokeapi"]) == cli.EXIT_SUCCESS
    assert cli.main(["fetch", "pokeapi", "--update"]) == cli.EXIT_SUCCESS

    assert fetch.updates == [False, True]
    output = capsys.readouterr().out
    assert "commit: 8a92bfa" in output
    assert "files downloaded: 8" in output
    assert "requests: 9" in output


@pytest.mark.parametrize(
    "extra", [["move"], ["--title", "Tackle (move)"], ["--from-export", "export.xml"]]
)
def test_fetch_pokeapi_rejects_bulbapedia_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    fetch = arrange_pokeapi_fetch(monkeypatch, SNAPSHOT_REPORT)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["fetch", "pokeapi", *extra])

    assert exit_info.value.code == 2
    assert cli.POKEAPI_OPTIONS_MESSAGE in capsys.readouterr().err
    assert fetch.updates == []


def test_fetch_bulbapedia_needs_a_page_type(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch = arrange_fetch(monkeypatch, FETCH_REPORT)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["fetch", "bulbapedia"])

    assert exit_info.value.code == 2
    assert cli.MISSING_PAGE_TYPE_MESSAGE in capsys.readouterr().err
    assert fetch.requests == []


@pytest.mark.parametrize(
    "error",
    [SnapshotError("moves.csv does not match its pin"), httpx.ConnectError("no route")],
)
def test_fetch_pokeapi_failure_prints_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    arrange_pokeapi_fetch(monkeypatch, error)

    assert cli.main(["fetch", "pokeapi"]) == cli.EXIT_FAILURE

    assert f"fetch pokeapi failed: {error}" in capsys.readouterr().err


def build_move_report(pages_failed: int = 0) -> MoveParseReport:
    return MoveParseReport(
        pages=(),
        pages_parsed=848,
        pages_skipped=110,
        pages_failed=pages_failed,
        skipped_per_reason={},
        notes=(),
        batch_counts=MoveBatchCounts(
            owner_nodes={"Move": 848, "Machine": 2173}, stubs={}, edges={}, claims={}
        ),
        chunk_counts=MoveBatchCounts(
            owner_nodes={"TextChunk": 9176}, stubs={}, edges={}, claims={}
        ),
        chunks_per_aspect={},
        dropped_templates=(),
        removed_tables_total=0,
        removed_tables_per_page={},
        math_formulas_total=0,
        math_formulas_per_page={},
        moves_per_flag={},
        cross_check=(),
        moves_without_pokeapi_row=(),
        unmatched_pokeapi_moves=(),
        skipped_pokeapi_machines=0,
        snapshot_commit="8a92bfa",
        snapshot_files={},
    )


class RecordingMoveParse:
    """Stands in for `cli.run_move_parse` and records the requests it ran."""

    def __init__(self, outcome: MoveParseReport | Exception) -> None:
        self.outcome = outcome
        self.requests: list[MoveParseRequest] = []

    def __call__(self, request: MoveParseRequest) -> MoveParseRun:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return MoveParseRun(outputs=(), report=self.outcome)


def test_parse_move_reaches_the_move_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parse = RecordingMoveParse(build_move_report())
    monkeypatch.setattr(cli, "run_move_parse", parse)

    exit_code = cli.main(["parse", "bulbapedia", "move", "--title", "Shadow Ball (move)"])

    assert exit_code == cli.EXIT_SUCCESS
    assert parse.requests == [MoveParseRequest(titles=("Shadow Ball (move)",))]
    output = capsys.readouterr().out
    assert "pages parsed: 848" in output
    assert "owner nodes Machine: 2173" in output


def test_parse_move_exits_1_when_a_page_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_move_parse", RecordingMoveParse(build_move_report(1)))

    assert cli.main(["parse", "bulbapedia", "move"]) == cli.EXIT_FAILURE


@pytest.mark.parametrize(
    "error",
    [
        TableError("FLAG_OVERRIDES names dig, which is no in-scope move"),
        MachineFieldError("tm4 names 'X-Y', which is not part of Generation IV"),
        MachineConflictError("tm30-gold-silver from A and B"),
        SnapshotError("no PokeAPI snapshot"),
        MoveParseRunError("Titles that are not pinned move pages: ['X']"),
    ],
)
def test_parse_move_failure_prints_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    monkeypatch.setattr(cli, "run_move_parse", RecordingMoveParse(error))

    assert cli.main(["parse", "bulbapedia", "move"]) == cli.EXIT_FAILURE

    assert f"parse bulbapedia move failed: {error}" in capsys.readouterr().err


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
        ExportError("Cannot read export missing.xml: No such file or directory"),
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
        removed_tables_total=0,
        removed_tables_per_page={},
        math_formulas_total=0,
        math_formulas_per_page={},
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


def build_ingest_run(
    stop: IngestStop | None = None, failed_titles: tuple[str, ...] = ()
) -> IngestRun:
    passes = tuple(
        PassTotals(
            ingest_pass=ingest_pass,
            batches_written=2,
            counters={"nodes_created": 7, "nodes_deleted": 0},
        )
        for ingest_pass in IngestPass
    )
    report = IngestReport(
        titles=(),
        is_full_run=True,
        passes=passes,
        changed_pages=(),
        failed_pages=tuple(PageFailure(page_title=title, reason="bad") for title in failed_titles),
        stop=stop,
        census=None
        if stop
        else CensusReport(
            labels={"Entity": 5, "PokemonSpecies": 2}, relationship_types={"HAS_TYPE": 2}
        ),
    )
    return IngestRun(report=report, elapsed_seconds=dict.fromkeys(IngestPass, 1.5))


class RecordingIngest:
    """Stands in for `cli.run_ingest` and `cli.write_ingest_report`, and records both."""

    def __init__(self, outcome: IngestRun | Exception) -> None:
        self.outcome = outcome
        self.titles: list[tuple[str, ...]] = []
        self.reports: list[IngestReport] = []

    async def run(self, settings: Settings, run: ParseRun, titles: tuple[str, ...]) -> IngestRun:
        self.titles.append(titles)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def write_report(self, report: IngestReport) -> Path:
        self.reports.append(report)
        return Path("species-ingest-report.json")


def arrange_ingest(
    monkeypatch: pytest.MonkeyPatch,
    outcome: IngestRun | Exception,
    parse_outcome: ParseReport | Exception | None = None,
) -> tuple[RecordingParse, RecordingIngest]:
    parse = arrange_parse(monkeypatch, parse_outcome or build_parse_report())
    ingest = RecordingIngest(outcome)
    monkeypatch.setattr(cli, "run_ingest", ingest.run)
    monkeypatch.setattr(cli, "write_ingest_report", ingest.write_report)
    return parse, ingest


INGEST_ARGV = ["ingest", "bulbapedia", "species"]


@pytest.mark.usefixtures("settings")
def test_ingest_passes_its_titles_to_the_parse_and_the_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse, ingest = arrange_ingest(monkeypatch, build_ingest_run())

    cli.main([*INGEST_ARGV, "--title", "A (Pokémon)", "--title", "B (Pokémon)"])
    cli.main(INGEST_ARGV)

    titles = ("A (Pokémon)", "B (Pokémon)")
    assert parse.requests == [ParseRequest(titles=titles), ParseRequest()]
    assert ingest.titles == [titles, ()]


@pytest.mark.usefixtures("settings")
def test_ingest_prints_the_parse_counts_the_pass_rows_and_the_census(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_run = build_ingest_run()
    _, ingest = arrange_ingest(monkeypatch, ingest_run)

    exit_code = cli.main(INGEST_ARGV)

    lines = capsys.readouterr().out.splitlines()
    assert exit_code == cli.EXIT_SUCCESS
    assert ingest.reports == [ingest_run.report]
    assert lines.index("parse bulbapedia species: done") < lines.index(
        "ingest bulbapedia species: done"
    )
    assert "pages parsed: 1" in lines
    assert lines[lines.index("ingest bulbapedia species: done") + 1 :] == [
        "species: 2 batches in 1.5 s",
        "species nodes created: 7",
        "species nodes deleted: 0",
        "evolution: 2 batches in 1.5 s",
        "evolution nodes created: 7",
        "evolution nodes deleted: 0",
        "chunk: 2 batches in 1.5 s",
        "chunk nodes created: 7",
        "chunk nodes deleted: 0",
        "nodes Entity: 5",
        "nodes PokemonSpecies: 2",
        "relationships HAS_TYPE: 2",
    ]


@pytest.mark.usefixtures("settings")
def test_ingest_exits_1_for_a_stopped_run_and_still_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stop = IngestStop(page_title="Beta (Pokémon)", ingest_pass=IngestPass.EVOLUTION, error="boom")
    ingest_run = build_ingest_run(stop=stop)
    _, ingest = arrange_ingest(monkeypatch, ingest_run)

    exit_code = cli.main(INGEST_ARGV)

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_FAILURE
    assert "ingest bulbapedia species: stopped" in out
    assert "stopped: Beta (Pokémon) (evolution): boom" in out
    assert "nodes Entity" not in out
    assert ingest.reports == [ingest_run.report]


@pytest.mark.usefixtures("settings")
def test_ingest_exits_1_when_a_page_failed_to_parse_and_names_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_run = build_ingest_run(failed_titles=("Gamma (Pokémon)",))
    arrange_ingest(monkeypatch, ingest_run, build_parse_report(failed_titles=("Gamma (Pokémon)",)))

    exit_code = cli.main(INGEST_ARGV)

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_FAILURE
    assert "failed: Gamma (Pokémon): bad" in out
    assert "failed: Gamma (Pokémon): no A-Box node for type1" in out


@pytest.mark.usefixtures("settings")
def test_ingest_tells_the_user_to_seed_when_the_seed_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = MissingSeedError([NodeRef(class_iri="https://muon.dev/ns/pkmn#Type", id="fire")])
    _, ingest = arrange_ingest(monkeypatch, missing)

    exit_code = cli.main(INGEST_ARGV)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert captured.err.startswith("ingest bulbapedia species failed: 1 A-Box nodes are missing")
    assert "`muon seed pkmn`" in captured.err
    assert captured.out == ""
    assert ingest.reports == []


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    "error",
    [
        GraphConnectionError(CONNECTION_MESSAGE),
        OntologyError("Missing universe T-Box"),
        ABoxError("pkmn.ttl", ["bad"]),
    ],
)
def test_ingest_reports_run_errors_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    _, ingest = arrange_ingest(monkeypatch, error)

    exit_code = cli.main(INGEST_ARGV)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert captured.err.startswith("ingest bulbapedia species failed:")
    assert SECRET not in captured.err
    assert captured.out == ""
    assert ingest.reports == []


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    "error",
    [ParseRunError("Titles that are not pinned: ['Ghost']"), PageStoreError("No page file")],
)
def test_ingest_stops_on_a_parse_error_before_the_graph(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    _, ingest = arrange_ingest(monkeypatch, build_ingest_run(), parse_outcome=error)

    exit_code = cli.main(INGEST_ARGV)

    assert exit_code == cli.EXIT_FAILURE
    assert capsys.readouterr().err.startswith("ingest bulbapedia species failed:")
    assert ingest.titles == []


def test_ingest_stops_on_invalid_settings_before_the_parse(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(os.environ):
        if name.startswith("NEO4J_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(cli, "get_settings", lambda: Neo4jSettings(_env_file=None))
    parse, ingest = arrange_ingest(monkeypatch, build_ingest_run())

    exit_code = cli.main(INGEST_ARGV)

    assert exit_code == cli.EXIT_FAILURE
    assert "ingest bulbapedia species failed: invalid settings" in capsys.readouterr().err
    assert parse.requests == []
    assert ingest.titles == []


@pytest.mark.parametrize(
    ("argv", "invalid"),
    [
        (["ingest", "bulbapedia", "ability"], "'ability'"),
        (["ingest", "serebii", "species"], "'serebii'"),
    ],
)
def test_ingest_rejects_unknown_choices(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    invalid: str,
) -> None:
    parse, _ = arrange_ingest(monkeypatch, build_ingest_run())

    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert f"invalid choice: {invalid}" in capsys.readouterr().err
    assert parse.requests == []


def test_describe_target_names_the_prefix_or_the_source_and_page_type() -> None:
    seed = argparse.Namespace(command="seed", prefix=PREFIX)
    ingest = argparse.Namespace(command="ingest", source="bulbapedia", page_type="species")

    assert cli.describe_target(seed) == PREFIX
    assert cli.describe_target(ingest) == "bulbapedia species"


def test_a_progress_line_names_the_pass_the_batches_and_the_seconds() -> None:
    progress = PassProgress(IngestPass.CHUNK, done=100, total=1025, elapsed_seconds=3.14)

    assert cli.format_progress(progress) == "chunk: 100/1025 batches, 3.1 s"


class FakeClient:
    """A stand-in for `Neo4jGraphClient` that records its settings and closes cleanly."""

    def __init__(self, neo4j: Neo4jSettings) -> None:
        self.neo4j = neo4j
        self.is_closed = False

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.is_closed = True


async def test_run_ingest_gives_the_run_a_client_the_titles_and_the_progress_printer(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    clients: list[FakeClient] = []
    requests: list[IngestRequest] = []
    ingest_run = build_ingest_run()

    def build_client(neo4j: Neo4jSettings) -> FakeClient:
        clients.append(FakeClient(neo4j))
        return clients[-1]

    async def fake_ingest_species(client: Any, run: ParseRun, request: IngestRequest) -> IngestRun:
        requests.append(request)
        assert client is clients[0]
        return ingest_run

    monkeypatch.setattr(cli, "Neo4jGraphClient", build_client)
    monkeypatch.setattr(cli, "ingest_species", fake_ingest_species)
    run = ParseRun(outputs=(), report=build_parse_report())

    result = await cli.run_ingest(settings, run, ("A (Pokémon)",))

    assert result is ingest_run
    assert clients[0].neo4j == settings.neo4j
    assert clients[0].is_closed
    assert requests[0].titles == ("A (Pokémon)",)
    assert requests[0].ontology_dir == settings.ontology.ontology_dir
    assert requests[0].on_progress is cli.print_progress


def build_move_ingest_run(stop: MoveIngestStop | None = None) -> MoveIngestRun:
    passes = tuple(
        MovePassTotals(ingest_pass=ingest_pass, batches_written=848, counters={"nodes_created": 3})
        for ingest_pass in MovePass
    )
    report = MoveIngestReport(
        titles=(),
        is_full_run=True,
        passes=passes,
        changed_pages=(),
        failed_pages=(),
        stop=stop,
        census=None if stop else CensusReport(labels={"Move": 906}, relationship_types={}),
        stub_ids=None if stop else {"Machine": (), "Move": ("g-max-wildfire",)},
    )
    return MoveIngestRun(report=report, elapsed_seconds=dict.fromkeys(MovePass, 2.5))


class RecordingMoveIngest:
    """Stands in for `cli.run_move_ingest` and `cli.write_move_ingest_report`."""

    def __init__(self, outcome: MoveIngestRun | Exception) -> None:
        self.outcome = outcome
        self.titles: list[tuple[str, ...]] = []
        self.reports: list[MoveIngestReport] = []

    async def run(
        self, settings: Settings, run: MoveParseRun, titles: tuple[str, ...]
    ) -> MoveIngestRun:
        self.titles.append(titles)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def write_report(self, report: MoveIngestReport) -> Path:
        self.reports.append(report)
        return Path("move-ingest-report.json")


def arrange_move_ingest(
    monkeypatch: pytest.MonkeyPatch,
    outcome: MoveIngestRun | Exception,
    parse_outcome: MoveParseReport | Exception | None = None,
) -> tuple[RecordingMoveParse, RecordingMoveIngest]:
    parse = RecordingMoveParse(parse_outcome or build_move_report())
    ingest = RecordingMoveIngest(outcome)
    monkeypatch.setattr(cli, "run_move_parse", parse)
    monkeypatch.setattr(cli, "run_move_ingest", ingest.run)
    monkeypatch.setattr(cli, "write_move_ingest_report", ingest.write_report)
    return parse, ingest


MOVE_INGEST_ARGV = ["ingest", "bulbapedia", "move"]


@pytest.mark.usefixtures("settings")
def test_move_ingest_passes_its_titles_to_the_parse_and_the_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parse, ingest = arrange_move_ingest(monkeypatch, build_move_ingest_run())

    exit_code = cli.main([*MOVE_INGEST_ARGV, "--title", "Shadow Ball (move)"])

    assert exit_code == cli.EXIT_SUCCESS
    assert parse.requests == [MoveParseRequest(titles=("Shadow Ball (move)",))]
    assert ingest.titles == [("Shadow Ball (move)",)]


@pytest.mark.usefixtures("settings")
def test_move_ingest_prints_the_passes_the_census_and_the_stubs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ingest_run = build_move_ingest_run()
    _, ingest = arrange_move_ingest(monkeypatch, ingest_run)

    cli.main(MOVE_INGEST_ARGV)

    lines = capsys.readouterr().out.splitlines()
    assert ingest.reports == [ingest_run.report]
    assert "pages parsed: 848" in lines
    assert lines[lines.index("ingest bulbapedia move: done") + 1 :] == [
        "move: 848 batches in 2.5 s",
        "move nodes created: 3",
        "chunk: 848 batches in 2.5 s",
        "chunk nodes created: 3",
        "nodes Move: 906",
        "stubs Machine: 0",
        "stubs Move: 1",
        "stub Move: g-max-wildfire",
    ]


@pytest.mark.usefixtures("settings")
def test_move_ingest_exits_1_for_a_stopped_run_and_still_writes_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stop = MoveIngestStop(page_title="Surf (move)", ingest_pass=MovePass.CHUNK, error="boom")
    _, ingest = arrange_move_ingest(monkeypatch, build_move_ingest_run(stop))

    exit_code = cli.main(MOVE_INGEST_ARGV)

    assert exit_code == cli.EXIT_FAILURE
    assert "stopped: Surf (move) (chunk): boom" in capsys.readouterr().out
    assert len(ingest.reports) == 1


@pytest.mark.usefixtures("settings")
def test_move_ingest_exits_1_when_a_page_failed_to_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    arrange_move_ingest(monkeypatch, build_move_ingest_run(), build_move_report(pages_failed=1))

    assert cli.main(MOVE_INGEST_ARGV) == cli.EXIT_FAILURE


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    "error",
    [
        MissingSeedError([NodeRef(class_iri="https://muon.dev/ns/pkmn#Type", id="fire")]),
        GraphConnectionError(CONNECTION_MESSAGE),
    ],
)
def test_move_ingest_reports_run_errors_with_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    _, ingest = arrange_move_ingest(monkeypatch, error)

    exit_code = cli.main(MOVE_INGEST_ARGV)

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert captured.err.startswith("ingest bulbapedia move failed:")
    assert SECRET not in captured.err
    assert ingest.reports == []


@pytest.mark.usefixtures("settings")
def test_move_ingest_stops_on_a_parse_error_before_the_graph(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    error = TableError("TYPE_OVERRIDES names growth, which is no in-scope move")
    _, ingest = arrange_move_ingest(monkeypatch, build_move_ingest_run(), error)

    exit_code = cli.main(MOVE_INGEST_ARGV)

    assert exit_code == cli.EXIT_FAILURE
    assert capsys.readouterr().err == f"ingest bulbapedia move failed: {error}\n"
    assert ingest.titles == []
