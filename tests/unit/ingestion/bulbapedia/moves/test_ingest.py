"""The move ingest run on a synthetic page store, with a client that records every write."""

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from muon.graph import GraphQueryError, Statement
from muon.ingestion.bulbapedia.batching import MACHINE, MOVE, PKMN, TEXT_CHUNK
from muon.ingestion.bulbapedia.ingest import (
    CensusReport,
    IngestRequest,
    MissingSeedError,
    PassProgress,
)
from muon.ingestion.bulbapedia.moves import ingest
from muon.ingestion.bulbapedia.moves import page as move_page
from muon.ingestion.bulbapedia.moves.ingest import (
    MOVE_INGEST_REPORT_FILE_NAME,
    MoveIngestRun,
    MoveIngestStop,
    MovePass,
    ingest_moves,
    list_move_constraint_classes,
    select_move_batch,
    write_move_ingest_report,
)
from muon.ingestion.bulbapedia.moves.run import MoveParseRun
from muon.ingestion.runner import (
    LIST_STUB_IDS,
    MERGE_NODES_UPSERT,
    PRUNE_CLAIMED_IN,
    PRUNE_CLAIMED_OUT,
)
from tests.unit.ingestion.bulbapedia.moves.test_pokeapi import TABLES, write_snapshot
from tests.unit.ingestion.bulbapedia.moves.test_run import (
    BROKEN,
    ONTOLOGY_DIR,
    SHADOW_BALL,
    TWIN,
    build_store,
    default_wikitexts,
    move_wikitext,
    run_parse,
)
from tests.unit.ingestion.recording_client import EMPTY_SUMMARY, RecordingGraphClient

# A relationship type that only the claims of 1 pass hold, so a batch can be told apart.
PASS_MARKERS = {MovePass.MOVE: "TEACHES_MOVE", MovePass.CHUNK: "DESCRIBES"}
BATCHES_PER_PASS = 2
PASS_ORDER = [MovePass.MOVE] * BATCHES_PER_PASS + [MovePass.CHUNK] * BATCHES_PER_PASS
# The census reads 2 queries and the stub list 1 per class, after the seed check.
READS_AFTER_A_FULL_RUN = 1 + 2 + 2

ParseRunner = Callable[..., MoveParseRun]


class FailingGraphClient(RecordingGraphClient):
    """Raises `error` on the `fail_at`-th batch write, counted from 1."""

    def __init__(self, fail_at: int, error: Exception, **canned: Any) -> None:
        super().__init__(**canned)
        self.fail_at = fail_at
        self.error = error

    async def execute_write_batch(self, statements: Sequence[Statement]) -> list[Any]:
        if len(self.batches) + 1 == self.fail_at:
            raise self.error
        return await super().execute_write_batch(statements)


@pytest.fixture
def parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParseRunner:
    """Parse Shadow Ball, Twin Ball and a page that fails, with the parser tables emptied."""
    monkeypatch.setattr(move_page, "PARSER_TABLES", {})
    monkeypatch.setattr(move_page, "MOVE_HITS", {})
    twin = move_wikitext("Twin Ball").replace("tm#2=30", "tm#2=31")
    broken = move_wikitext("Broken", type_name="Cosmic")
    wikitexts = {**default_wikitexts(), TWIN: twin, BROKEN: broken}
    store = build_store(tmp_path / "data", wikitexts)
    snapshot = write_snapshot(tmp_path / "pokeapi", TABLES)

    def run(*titles: str) -> MoveParseRun:
        return run_parse(store, snapshot, titles)

    return run


def build_request(*titles: str) -> tuple[IngestRequest, list[PassProgress]]:
    progress: list[PassProgress] = []
    request = IngestRequest(ontology_dir=ONTOLOGY_DIR, titles=titles, on_progress=progress.append)
    return request, progress


async def run_ingest(
    client: RecordingGraphClient, run: MoveParseRun, *titles: str
) -> MoveIngestRun:
    request, _ = build_request(*titles)
    return await ingest_moves(client, run, request)


def classify_batch(statements: Sequence[Statement]) -> MovePass:
    relationship_types = {
        row["type"]
        for statement in statements
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    (ingest_pass,) = [p for p, marker in PASS_MARKERS.items() if marker in relationship_types]
    return ingest_pass


async def test_every_pass_1_batch_is_written_before_any_pass_2_batch(parse: ParseRunner) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, parse())

    assert [classify_batch(batch) for batch in client.batches] == PASS_ORDER


async def test_pages_are_written_in_parse_order_within_a_pass(parse: ParseRunner) -> None:
    client = RecordingGraphClient()
    run = parse()

    await run_ingest(client, run)

    written_ids = []
    for batch in client.batches[:BATCHES_PER_PASS]:
        (merge_nodes,) = [s for s in batch if s.query is MERGE_NODES_UPSERT]
        written_ids.append({row["id"] for row in merge_nodes.parameters["nodes"]})
    assert written_ids == [{node.id for node in o.batch.owner_nodes} for o in run.outputs]
    assert [o.extract.id for o in run.outputs] == ["shadow-ball", "twin-ball"]


async def test_constraints_are_ensured_once_before_the_first_batch(parse: ParseRunner) -> None:
    client = RecordingGraphClient()
    run = parse()

    await run_ingest(client, run)

    kinds = [type(call) for call in client.calls]
    first_batch = kinds.index(list)
    assert set(kinds[:first_batch]) == {str}
    assert set(kinds[first_batch:]) == {list}
    assert len(client.writes) == len(list_move_constraint_classes(run.outputs))


def test_constraint_classes_cover_every_scope_and_stub_class(parse: ParseRunner) -> None:
    run = parse()

    classes = list_move_constraint_classes(run.outputs)

    batches = [select_move_batch(o, p) for o in run.outputs for p in MovePass]
    assert {MOVE, MACHINE, TEXT_CHUNK} <= classes
    assert {stub.class_iri for batch in batches for stub in batch.stub_nodes} <= classes


def test_select_move_batch_returns_the_batch_of_the_pass(parse: ParseRunner) -> None:
    output = parse().outputs[0]

    assert select_move_batch(output, MovePass.MOVE) is output.batch
    assert select_move_batch(output, MovePass.CHUNK) is output.chunk_batch


async def test_a_missing_seed_node_stops_the_run_before_any_write(parse: ParseRunner) -> None:
    missing = [{"class_iri": PKMN + "Type", "id": "ghost"}]
    client = RecordingGraphClient(read_records=[missing])

    with pytest.raises(MissingSeedError, match="`muon seed pkmn`"):
        await run_ingest(client, parse())

    assert client.calls == []


@pytest.mark.parametrize(
    ("fail_at", "pass_index", "passes_run"),
    [(2, 1, 1), (3, 0, 2)],
)
@pytest.mark.parametrize("error", [GraphQueryError("boom"), ValueError("bad batch")])
async def test_a_failed_batch_write_stops_every_later_batch_and_the_census(
    parse: ParseRunner, fail_at: int, pass_index: int, passes_run: int, error: Exception
) -> None:
    client = FailingGraphClient(fail_at, error)
    run = parse()

    result = await run_ingest(client, run)

    assert len(client.batches) == fail_at - 1
    assert result.report.stop == MoveIngestStop(
        page_title=run.outputs[pass_index].page.title,
        ingest_pass=list(MovePass)[passes_run - 1],
        error=str(error),
    )
    assert [p.ingest_pass for p in result.report.passes] == list(MovePass)[:passes_run]
    assert result.report.census is None
    assert result.report.stub_ids is None
    assert len(client.reads) == 1


async def test_a_page_that_failed_to_parse_is_listed_with_its_reason(parse: ParseRunner) -> None:
    result = await run_ingest(RecordingGraphClient(), parse())

    (failure,) = result.report.failed_pages
    assert failure.page_title == BROKEN
    assert "Cosmic" in failure.reason


async def test_progress_comes_at_the_end_of_each_pass(parse: ParseRunner) -> None:
    request, progress = build_request()

    await ingest_moves(RecordingGraphClient(), parse(), request)

    assert [(p.ingest_pass, p.done, p.total) for p in progress] == [
        (MovePass.MOVE, 2, 2),
        (MovePass.CHUNK, 2, 2),
    ]


async def test_progress_comes_every_n_batches(
    parse: ParseRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "PROGRESS_EVERY", 1)
    request, progress = build_request()

    await ingest_moves(RecordingGraphClient(), parse(), request)

    assert [(p.ingest_pass, p.done) for p in progress] == [
        (ingest_pass, done) for ingest_pass in MovePass for done in (1, 2)
    ]


async def test_changed_pages_list_the_batches_that_created_or_deleted_something(
    parse: ParseRunner,
) -> None:
    client = RecordingGraphClient(summary=replace(EMPTY_SUMMARY, nodes_created=1))
    run = parse()

    result = await run_ingest(client, run)

    titles = [o.page.title for o in run.outputs]
    changes = [(c.page_title, c.ingest_pass) for c in result.report.changed_pages]
    assert changes == [(title, p) for p in MovePass for title in titles]
    for totals in result.report.passes:
        assert totals.batches_written == BATCHES_PER_PASS


async def test_census_and_stub_ids_are_read_when_nothing_stopped(parse: ParseRunner) -> None:
    client = RecordingGraphClient(
        read_records=[
            [],
            [{"label": "Move", "count": 3}],
            [{"type": "HAS_TYPE", "count": 2}],
            [],
            [{"id": "g-max-wildfire"}],
        ]
    )

    result = await run_ingest(client, parse())

    assert len(client.reads) == READS_AFTER_A_FULL_RUN
    assert [query for query, _ in client.reads[3:]] == [LIST_STUB_IDS, LIST_STUB_IDS]
    assert result.report.census == CensusReport(
        labels={"Move": 3}, relationship_types={"HAS_TYPE": 2}
    )
    assert result.report.stub_ids == {"Machine": (), "Move": ("g-max-wildfire",)}


async def test_a_title_run_writes_only_those_pages(parse: ParseRunner) -> None:
    client = RecordingGraphClient()

    result = await run_ingest(client, parse(SHADOW_BALL), SHADOW_BALL)

    assert result.report.titles == (SHADOW_BALL,)
    assert result.report.is_full_run is False
    assert len(client.batches) == len(MovePass)


async def test_the_report_file_has_sorted_keys_and_no_time(
    parse: ParseRunner, tmp_path: Path
) -> None:
    result = await run_ingest(RecordingGraphClient(), parse())

    path = write_move_ingest_report(result.report, tmp_path / "ingested")

    assert path.name == MOVE_INGEST_REPORT_FILE_NAME
    text = path.read_text()
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert "seconds" not in text
    assert write_move_ingest_report(result.report, tmp_path / "again").read_bytes() == (
        path.read_bytes()
    )
