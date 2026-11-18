"""The item ingest run on a synthetic page store, with a client that records every write."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from muon.config import DEFAULT_UNIVERSES_DIR
from muon.graph import GraphQueryError, Statement
from muon.ingestion.bulbapedia.batching import BERRY, HELD_ITEM, ITEM, TEXT_CHUNK
from muon.ingestion.bulbapedia.ingest import CensusReport, IngestRequest, IngestStop
from muon.ingestion.bulbapedia.items.ingest import (
    ITEM_INGEST_REPORT_FILE_NAME,
    STUB_CLASSES,
    ItemIngestRun,
    ItemPass,
    ingest_items,
    list_item_constraint_classes,
    select_item_batch,
    write_item_ingest_report,
)
from muon.ingestion.bulbapedia.items.run import ItemParseRun
from muon.ingestion.runner import LIST_STUB_IDS, PRUNE_CLAIMED_IN, PRUNE_CLAIMED_OUT
from tests.unit.ingestion.bulbapedia.items.test_run import ONTOLOGY_DIR, ORAN, run_parse
from tests.unit.ingestion.recording_client import RecordingGraphClient

# A relationship type that only the claims of 1 pass hold, so a batch can be told apart.
PASS_MARKERS = {ItemPass.ITEM: "INTRODUCED_IN", ItemPass.CHUNK: "DESCRIBES"}


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
def run(tmp_path: Path) -> ItemParseRun:
    return run_parse(tmp_path)


async def run_ingest(client: RecordingGraphClient, run: ItemParseRun) -> ItemIngestRun:
    request = IngestRequest(
        ontology_dir=ONTOLOGY_DIR,
        universes_dir=DEFAULT_UNIVERSES_DIR,
        titles=(),
        on_progress=lambda _: None,
    )
    return await ingest_items(client, run, request)


def classify_batch(statements: Sequence[Statement]) -> ItemPass:
    relationship_types = {
        row["type"]
        for statement in statements
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    (ingest_pass,) = [p for p, marker in PASS_MARKERS.items() if marker in relationship_types]
    return ingest_pass


async def test_every_item_batch_is_written_before_any_chunk_batch(run: ItemParseRun) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run)

    passes = [classify_batch(batch) for batch in client.batches]
    assert passes == [ItemPass.ITEM] * len(run.outputs) + [ItemPass.CHUNK] * len(run.outputs)


def test_select_item_batch_returns_the_batch_of_the_pass(run: ItemParseRun) -> None:
    output = run.outputs[0]

    assert select_item_batch(output, ItemPass.ITEM) is output.batch
    assert select_item_batch(output, ItemPass.CHUNK) is output.chunk_batch


def test_constraint_classes_cover_scope_and_stub_classes(run: ItemParseRun) -> None:
    classes = list_item_constraint_classes(run.outputs)

    assert {BERRY, HELD_ITEM, ITEM, TEXT_CHUNK} <= classes


def test_the_stub_class_is_the_item_family_key() -> None:
    assert STUB_CLASSES == (ITEM,)


@pytest.mark.parametrize("error", [GraphQueryError("boom"), ValueError("bad batch")])
async def test_a_failed_write_stops_the_run_before_the_census(
    run: ItemParseRun, error: Exception
) -> None:
    client = FailingGraphClient(len(run.outputs) + 1, error)

    result = await run_ingest(client, run)

    assert result.report.stop is not None
    assert result.report.stop.ingest_pass == ItemPass.CHUNK
    assert result.report.stop == IngestStop(
        page_title=result.report.stop.page_title, ingest_pass=ItemPass.CHUNK, error=str(error)
    )
    assert result.report.census is None
    assert result.report.stub_ids is None


async def test_census_and_item_stubs_are_read_when_nothing_stopped(run: ItemParseRun) -> None:
    client = RecordingGraphClient(
        read_records=[
            [],
            [{"label": "Item", "count": 3}],
            [{"type": "DESCRIBES", "count": 4}],
            [{"id": "moon-stone", "label": "Item"}],
        ]
    )

    result = await run_ingest(client, run)

    assert [query for query, _ in client.reads[3:]] == [LIST_STUB_IDS]
    assert result.report.census == CensusReport(
        labels={"Item": 3}, relationship_types={"DESCRIBES": 4}
    )
    assert result.report.stub_ids == {"Item": ("moon-stone",)}


async def test_the_report_file_has_sorted_keys_and_no_time(
    run: ItemParseRun, tmp_path: Path
) -> None:
    result = await run_ingest(RecordingGraphClient(), run)

    path = write_item_ingest_report(result.report, tmp_path / "ingested")

    assert path.name == ITEM_INGEST_REPORT_FILE_NAME
    text = path.read_text()
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert "stub_ids" in json.loads(text)
    assert "seconds" not in text


def test_the_synthetic_run_parses_the_oran_berry(run: ItemParseRun) -> None:
    assert ORAN in [output.page.title for output in run.outputs]
