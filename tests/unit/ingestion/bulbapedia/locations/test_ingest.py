"""The location ingest run on a synthetic page store, with a client that records every write."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from muon.config import DEFAULT_UNIVERSES_DIR
from muon.graph import GraphQueryError, Statement
from muon.ingestion.bulbapedia.batching import LOCATION, TEXT_CHUNK
from muon.ingestion.bulbapedia.ingest import CensusReport, IngestRequest, IngestStop
from muon.ingestion.bulbapedia.locations.batches import ITEM_FIND, LOCATION_AREA, SETTLEMENT
from muon.ingestion.bulbapedia.locations.ingest import (
    LOCATION_INGEST_REPORT_FILE_NAME,
    STUB_CLASSES,
    LocationIngestRun,
    LocationPass,
    ingest_locations,
    list_location_constraint_classes,
    select_location_batch,
    write_location_ingest_report,
)
from muon.ingestion.bulbapedia.locations.run import LocationParseRun
from muon.ingestion.runner import LIST_STUB_IDS, PRUNE_CLAIMED_IN, PRUNE_CLAIMED_OUT
from tests.unit.ingestion.bulbapedia.locations.test_run import ONTOLOGY_DIR, SOOTOPOLIS, run_parse
from tests.unit.ingestion.recording_client import RecordingGraphClient

# A relationship type that only the claims of 1 pass hold, so a batch can be told apart.
PASS_MARKERS = {
    LocationPass.LOCATION: "ADJACENT_TO",
    LocationPass.CHUNK: "DESCRIBES",
    LocationPass.FIND: "AT_LOCATION",
}


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
def run(tmp_path: Path) -> LocationParseRun:
    return run_parse(tmp_path)


async def run_ingest(client: RecordingGraphClient, run: LocationParseRun) -> LocationIngestRun:
    request = IngestRequest(
        ontology_dir=ONTOLOGY_DIR,
        universes_dir=DEFAULT_UNIVERSES_DIR,
        titles=(),
        on_progress=lambda _: None,
    )
    return await ingest_locations(client, run, request)


def classify_batch(statements: Sequence[Statement]) -> LocationPass | None:
    """Return the pass of a batch from its claims, or `None` for a batch with no claim."""
    relationship_types = {
        row["type"]
        for statement in statements
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    passes = [p for p, marker in PASS_MARKERS.items() if marker in relationship_types]
    return passes[0] if passes else None


async def test_batches_are_written_pass_by_pass(run: LocationParseRun) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run)

    passes = [classify_batch(batch) for batch in client.batches]
    owners = len(run.outputs) - 1
    assert passes.count(LocationPass.LOCATION) == owners
    assert passes.count(LocationPass.CHUNK) == len(run.outputs)
    assert passes.count(LocationPass.FIND) == owners
    assert passes.count(None) == 2  # the A-Box Region page claims nothing in passes 1 and 3
    first_chunk = passes.index(LocationPass.CHUNK)
    first_find = passes.index(LocationPass.FIND)
    assert LocationPass.LOCATION not in passes[first_chunk:]
    assert LocationPass.CHUNK not in passes[first_find:]


def test_select_location_batch_returns_the_batch_of_the_pass(run: LocationParseRun) -> None:
    output = run.outputs[0]

    assert select_location_batch(output, LocationPass.LOCATION) is output.batch
    assert select_location_batch(output, LocationPass.CHUNK) is output.chunk_batch
    assert select_location_batch(output, LocationPass.FIND) is output.find_batch


def test_constraint_classes_cover_scope_and_stub_classes(run: LocationParseRun) -> None:
    classes = list_location_constraint_classes(run.outputs)

    assert {SETTLEMENT, LOCATION_AREA, TEXT_CHUNK, ITEM_FIND, LOCATION} <= classes


def test_the_stub_class_is_the_location_family_key() -> None:
    assert STUB_CLASSES == (LOCATION,)


@pytest.mark.parametrize("error", [GraphQueryError("boom"), ValueError("bad batch")])
async def test_a_failed_write_stops_the_run_before_the_census(
    run: LocationParseRun, error: Exception
) -> None:
    client = FailingGraphClient(len(run.outputs) + 1, error)

    result = await run_ingest(client, run)

    assert result.report.stop is not None
    assert result.report.stop.ingest_pass == LocationPass.CHUNK
    assert result.report.stop == IngestStop(
        page_title=result.report.stop.page_title, ingest_pass=LocationPass.CHUNK, error=str(error)
    )
    assert result.report.census is None
    assert result.report.stub_ids is None


async def test_census_and_location_stubs_are_read_when_nothing_stopped(
    run: LocationParseRun,
) -> None:
    client = RecordingGraphClient(
        read_records=[
            [],
            [{"label": "Location", "count": 3}],
            [{"type": "LOCATED_IN", "count": 4}],
            [{"id": "cave-of-origin", "label": "Location"}],
        ]
    )

    result = await run_ingest(client, run)

    assert [query for query, _ in client.reads[3:]] == [LIST_STUB_IDS]
    assert result.report.census == CensusReport(
        labels={"Location": 3}, relationship_types={"LOCATED_IN": 4}
    )
    assert result.report.stub_ids == {"Location": ("cave-of-origin",)}


async def test_the_report_file_has_sorted_keys_and_no_time(
    run: LocationParseRun, tmp_path: Path
) -> None:
    result = await run_ingest(RecordingGraphClient(), run)

    path = write_location_ingest_report(result.report, tmp_path / "ingested")

    assert path.name == LOCATION_INGEST_REPORT_FILE_NAME
    text = path.read_text()
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert "stub_ids" in json.loads(text)
    assert "seconds" not in text


def test_the_synthetic_run_parses_sootopolis_city(run: LocationParseRun) -> None:
    assert SOOTOPOLIS in [output.page.title for output in run.outputs]
