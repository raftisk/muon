"""The ability ingest run on a synthetic page store, with a client that records every write."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from muon.graph import GraphQueryError, Statement
from muon.ingestion.bulbapedia.abilities.ingest import (
    ABILITY_INGEST_REPORT_FILE_NAME,
    AbilityIngestRun,
    AbilityPass,
    ingest_abilities,
    select_ability_batch,
    write_ability_ingest_report,
)
from muon.ingestion.bulbapedia.abilities.run import AbilityParseRun
from muon.ingestion.bulbapedia.ingest import CensusReport, IngestRequest, IngestStop
from muon.ingestion.runner import LIST_STUB_IDS, PRUNE_CLAIMED_IN, PRUNE_CLAIMED_OUT
from tests.unit.ingestion.bulbapedia.abilities.test_run import ONTOLOGY_DIR, TEST_GUARD, run_parse
from tests.unit.ingestion.recording_client import RecordingGraphClient

# A relationship type that only the claims of 1 pass hold, so a batch can be told apart.
PASS_MARKERS = {AbilityPass.ABILITY: "INTRODUCED_IN", AbilityPass.CHUNK: "DESCRIBES"}


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
def run(tmp_path: Path) -> AbilityParseRun:
    return run_parse(tmp_path)


async def run_ingest(client: RecordingGraphClient, run: AbilityParseRun) -> AbilityIngestRun:
    request = IngestRequest(ontology_dir=ONTOLOGY_DIR, titles=(), on_progress=lambda _: None)
    return await ingest_abilities(client, run, request)


def classify_batch(statements: Sequence[Statement]) -> AbilityPass:
    relationship_types = {
        row["type"]
        for statement in statements
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    (ingest_pass,) = [p for p, marker in PASS_MARKERS.items() if marker in relationship_types]
    return ingest_pass


async def test_the_ability_pass_is_written_before_the_chunk_pass(run: AbilityParseRun) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run)

    assert [classify_batch(batch) for batch in client.batches] == list(AbilityPass)


async def test_no_batch_claims_has_ability(run: AbilityParseRun) -> None:
    client = RecordingGraphClient()

    await run_ingest(client, run)

    claimed = {
        row["type"]
        for batch in client.batches
        for statement in batch
        if statement.query in (PRUNE_CLAIMED_OUT, PRUNE_CLAIMED_IN)
        for row in statement.parameters["claims"]
    }
    assert "HAS_ABILITY" not in claimed


def test_select_ability_batch_returns_the_batch_of_the_pass(run: AbilityParseRun) -> None:
    output = run.outputs[0]

    assert select_ability_batch(output, AbilityPass.ABILITY) is output.batch
    assert select_ability_batch(output, AbilityPass.CHUNK) is output.chunk_batch


@pytest.mark.parametrize("error", [GraphQueryError("boom"), ValueError("bad batch")])
async def test_a_failed_write_stops_the_run_before_the_census(
    run: AbilityParseRun, error: Exception
) -> None:
    client = FailingGraphClient(2, error)

    result = await run_ingest(client, run)

    assert result.report.stop == IngestStop(
        page_title=TEST_GUARD, ingest_pass=AbilityPass.CHUNK, error=str(error)
    )
    assert result.report.census is None
    assert result.report.stub_ids is None


async def test_census_and_ability_stubs_are_read_when_nothing_stopped(
    run: AbilityParseRun,
) -> None:
    client = RecordingGraphClient(
        read_records=[
            [],
            [{"label": "Ability", "count": 3}],
            [{"type": "DESCRIBES", "count": 4}],
            [{"id": "trace"}],
        ]
    )

    result = await run_ingest(client, run)

    assert [query for query, _ in client.reads[3:]] == [LIST_STUB_IDS]
    assert result.report.census == CensusReport(
        labels={"Ability": 3}, relationship_types={"DESCRIBES": 4}
    )
    assert result.report.stub_ids == {"Ability": ("trace",)}


async def test_the_report_file_has_sorted_keys_and_no_time(
    run: AbilityParseRun, tmp_path: Path
) -> None:
    result = await run_ingest(RecordingGraphClient(), run)

    path = write_ability_ingest_report(result.report, tmp_path / "ingested")

    assert path.name == ABILITY_INGEST_REPORT_FILE_NAME
    text = path.read_text()
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert "stub_ids" in json.loads(text)
    assert "seconds" not in text
