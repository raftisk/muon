"""The species ingest run on the synthetic pages, with a client that records every write."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from muon.graph import GraphQueryError, Statement
from muon.ingestion.bulbapedia.pages import PageStore
from muon.ingestion.bulbapedia.species import ingest
from muon.ingestion.bulbapedia.species.batches import (
    EVOLUTION,
    FORM_CHANGE,
    LEARNSET_ENTRY,
    PKMN,
    POKEMON_SPECIES,
    TEXT_CHUNK,
)
from muon.ingestion.bulbapedia.species.ingest import (
    INGEST_REPORT_FILE_NAME,
    CensusReport,
    IngestPass,
    IngestRequest,
    IngestRun,
    IngestStop,
    MissingSeedError,
    PassProgress,
    ingest_species,
    list_constraint_classes,
    select_batch,
    write_ingest_report,
)
from muon.ingestion.bulbapedia.species.run import ParseRun
from muon.ingestion.runner import MERGE_NODES_UPSERT, PRUNE_CLAIMED_IN, PRUNE_CLAIMED_OUT
from tests.unit.ingestion.bulbapedia.species.synthetic_pages import (
    ALPHA,
    BETA,
    GAMMA,
    ONTOLOGY_DIR,
    run_parse,
)
from tests.unit.ingestion.recording_client import EMPTY_SUMMARY, RecordingGraphClient

pytestmark = pytest.mark.usefixtures("no_overrides")

# A relationship type that only the claims of 1 pass hold, so a batch can be told apart.
PASS_MARKERS = {
    IngestPass.SPECIES: "CAN_LEARN",
    IngestPass.EVOLUTION: "EVOLVES_TO",
    IngestPass.CHUNK: "DESCRIBES",
}
PASS_ORDER = [IngestPass.SPECIES] * 2 + [IngestPass.EVOLUTION] * 2 + [IngestPass.CHUNK] * 2
BATCHES_PER_PASS = 2


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


def build_request(*titles: str) -> tuple[IngestRequest, list[PassProgress]]:
    progress: list[PassProgress] = []
    request = IngestRequest(ontology_dir=ONTOLOGY_DIR, titles=titles, on_progress=progress.append)
    return request, progress


async def run_ingest(client: RecordingGraphClient, run: ParseRun, *titles: str) -> IngestRun:
    request, _ = build_request(*titles)
    return await ingest_species(client, run, request)


def classify_batch(statements: Sequence[Statement]) -> IngestPass:
    relationship_types = {
        row["type"]
        for statement in statements
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    (ingest_pass,) = [p for p, marker in PASS_MARKERS.items() if marker in relationship_types]
    return ingest_pass


async def test_every_batch_of_a_pass_is_written_before_the_next_pass(store: PageStore) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run_parse(store))

    assert [classify_batch(batch) for batch in client.batches] == PASS_ORDER


async def test_pages_are_written_in_parse_order_within_a_pass(store: PageStore) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run_parse(store))

    owner_ids = []
    for batch in client.batches[:BATCHES_PER_PASS]:
        (merge_nodes,) = [s for s in batch if s.query is MERGE_NODES_UPSERT]
        owner_ids.append([row["id"] for row in merge_nodes.parameters["nodes"]])
    assert "alpha" in owner_ids[0]
    assert "alpha" not in owner_ids[1]
    assert "beta" in owner_ids[1]


async def test_constraints_are_ensured_once_before_the_first_batch(store: PageStore) -> None:
    client = RecordingGraphClient()
    run = run_parse(store)

    await run_ingest(client, run)

    kinds = [type(call) for call in client.calls]
    first_batch = kinds.index(list)
    assert set(kinds[:first_batch]) == {str}
    assert set(kinds[first_batch:]) == {list}
    assert (
        len(client.writes) == len(set(client.writes)) == len(list_constraint_classes(run.outputs))
    )
    assert all(write.startswith("CREATE CONSTRAINT") for write in client.writes)


async def test_constraint_classes_cover_every_scope_and_stub_class(store: PageStore) -> None:
    run = run_parse(store)

    classes = list_constraint_classes(run.outputs)

    batches = [select_batch(o, p) for o in run.outputs for p in IngestPass]
    stubs = {stub.class_iri for batch in batches for stub in batch.stub_nodes}
    assert {POKEMON_SPECIES, LEARNSET_ENTRY, FORM_CHANGE, EVOLUTION, TEXT_CHUNK} <= classes
    assert stubs <= classes


def test_select_batch_returns_the_batch_of_the_pass(store: PageStore) -> None:
    output = run_parse(store).outputs[0]

    assert select_batch(output, IngestPass.SPECIES) is output.species_batch
    assert select_batch(output, IngestPass.EVOLUTION) is output.evolution_batch
    assert select_batch(output, IngestPass.CHUNK) is output.chunk_batch


async def test_a_missing_seed_node_stops_the_run_before_any_write(store: PageStore) -> None:
    missing = [{"class_iri": PKMN + "Type", "id": f"type-{number}"} for number in range(7)]
    client = RecordingGraphClient(read_records=[missing])

    with pytest.raises(MissingSeedError) as caught:
        await run_ingest(client, run_parse(store))

    assert client.calls == []
    assert len(caught.value.missing) == 7
    message = str(caught.value)
    assert message.startswith("7 A-Box nodes are missing from the graph")
    assert "Type 'type-0'" in message
    assert "Type 'type-4'" in message
    assert "type-5" not in message
    assert "and 2 more" in message
    assert "`muon seed pkmn`" in message


@pytest.mark.parametrize(
    ("fail_at", "page", "ingest_pass", "passes_run"),
    [
        (2, BETA, IngestPass.SPECIES, 1),
        (3, ALPHA, IngestPass.EVOLUTION, 2),
        (5, ALPHA, IngestPass.CHUNK, 3),
    ],
)
@pytest.mark.parametrize("error", [GraphQueryError("boom"), ValueError("bad batch")])
async def test_a_failed_batch_write_stops_every_later_batch_and_the_census(
    store: PageStore,
    fail_at: int,
    page: str,
    ingest_pass: IngestPass,
    passes_run: int,
    error: Exception,
) -> None:
    client = FailingGraphClient(fail_at, error)

    result = await run_ingest(client, run_parse(store))

    assert len(client.batches) == fail_at - 1
    assert result.report.stop == IngestStop(
        page_title=page, ingest_pass=ingest_pass, error=str(error)
    )
    assert [p.ingest_pass for p in result.report.passes] == list(IngestPass)[:passes_run]
    assert result.report.census is None
    assert len(client.reads) == 1


async def test_a_page_that_failed_to_parse_is_listed_with_its_reason(store: PageStore) -> None:
    result = await run_ingest(RecordingGraphClient(), run_parse(store))

    (failure,) = result.report.failed_pages
    assert failure.page_title == GAMMA
    assert "Imaginary" in failure.reason


async def test_progress_comes_at_the_end_of_each_pass(store: PageStore) -> None:
    client = RecordingGraphClient()
    request, progress = build_request()

    await ingest_species(client, run_parse(store), request)

    assert [(p.ingest_pass, p.done, p.total) for p in progress] == [
        (IngestPass.SPECIES, 2, 2),
        (IngestPass.EVOLUTION, 2, 2),
        (IngestPass.CHUNK, 2, 2),
    ]
    assert all(p.elapsed_seconds >= 0 for p in progress)


async def test_progress_comes_every_n_batches_without_a_repeat_at_the_end(
    store: PageStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "PROGRESS_EVERY", 1)
    request, progress = build_request()

    await ingest_species(RecordingGraphClient(), run_parse(store), request)

    assert [(p.ingest_pass, p.done) for p in progress] == [
        (ingest_pass, done) for ingest_pass in IngestPass for done in (1, 2)
    ]


async def test_changed_pages_list_the_batches_that_created_or_deleted_something(
    store: PageStore,
) -> None:
    summary = replace(EMPTY_SUMMARY, nodes_created=1, properties_set=3)
    client = RecordingGraphClient(summary=summary)

    result = await run_ingest(client, run_parse(store))

    report = result.report
    changes = [(c.page_title, c.ingest_pass) for c in report.changed_pages]
    assert changes == [(page, ingest_pass) for ingest_pass in IngestPass for page in (ALPHA, BETA)]
    for change, statements in zip(report.changed_pages, client.batches, strict=True):
        assert change.counters["nodes_created"] == len(statements)
    for totals in report.passes:
        assert totals.batches_written == BATCHES_PER_PASS
        assert totals.counters["nodes_created"] == sum(
            change.counters["nodes_created"]
            for change in report.changed_pages
            if change.ingest_pass is totals.ingest_pass
        )


async def test_a_run_that_changes_nothing_lists_no_page_and_sums_to_zero(store: PageStore) -> None:
    result = await run_ingest(RecordingGraphClient(), run_parse(store))

    assert result.report.changed_pages == ()
    assert result.report.stop is None
    for totals in result.report.passes:
        assert totals.counters["nodes_created"] == 0
        assert totals.counters["relationships_deleted"] == 0


async def test_the_census_is_read_when_nothing_stopped(store: PageStore) -> None:
    client = RecordingGraphClient(
        read_records=[
            [],
            [{"label": "PokemonSpecies", "count": 2}, {"label": "Entity", "count": 3}],
            [{"type": "HAS_TYPE", "count": 2}],
        ]
    )

    result = await run_ingest(client, run_parse(store))

    assert len(client.reads) == 3
    assert result.report.census == CensusReport(
        labels={"Entity": 3, "PokemonSpecies": 2}, relationship_types={"HAS_TYPE": 2}
    )


async def test_a_title_run_reports_its_titles_and_writes_only_those_pages(store: PageStore) -> None:
    client = RecordingGraphClient()

    result = await run_ingest(client, run_parse(store, ALPHA), ALPHA)

    assert result.report.titles == (ALPHA,)
    assert result.report.is_full_run is False
    assert len(client.batches) == len(IngestPass)


async def test_a_full_run_has_no_titles(store: PageStore) -> None:
    result = await run_ingest(RecordingGraphClient(), run_parse(store))

    assert result.report.titles == ()
    assert result.report.is_full_run is True
    assert set(result.elapsed_seconds) == set(IngestPass)


async def test_the_report_file_has_sorted_keys_and_no_time(
    store: PageStore, tmp_path: Path
) -> None:
    first = await run_ingest(RecordingGraphClient(), run_parse(store))
    second = await run_ingest(RecordingGraphClient(), run_parse(store))

    first_path = write_ingest_report(first.report, tmp_path / "first")
    second_path = write_ingest_report(second.report, tmp_path / "second")

    text = first_path.read_text(encoding="utf-8")
    assert first_path == tmp_path / "first" / INGEST_REPORT_FILE_NAME
    assert text == json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    assert "seconds" not in text
    assert text == second_path.read_text(encoding="utf-8")
