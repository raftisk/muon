"""Location ingest: the batches of a `LocationParseRun` go into the graph in 4 passes.

`ingest_locations` checks the seed and ensures the constraints of every class in the run once,
then writes every pass 1 batch (Location, its class and LocationArea), then every pass 2
batch (TextChunk), then every pass 3 batch (ItemFind), then every pass 4 batch (Encounter), and
reads the census of the universe and the ids of the Location family nodes that are still
stubs. A page owns its Location through the `Location` key label, so it finds the stub a species
page wrote. The write loop,
the stop rule and the report models are the shared ones of `bulbapedia/ingest.py`.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import LOCATION
from muon.ingestion.bulbapedia.ingest import (
    INGEST_OUTPUT_ROOT,
    IngestRequest,
    PassItem,
    PlanSummary,
    StubIngestReport,
    build_ingest_report,
    list_batch_classes,
    prepare_target,
    should_count,
    write_passes,
    write_report_file,
)
from muon.ingestion.bulbapedia.locations.run import LocationPageOutput, LocationParseRun

LOCATION_INGEST_REPORT_FILE_NAME = "location-ingest-report.json"
STUB_CLASSES = (LOCATION,)


class LocationPass(StrEnum):
    """The passes in write order: chunks, finds and encounters point at the nodes pass 1 owns."""

    LOCATION = "location"
    CHUNK = "chunk"
    FIND = "find"
    ENCOUNTER = "encounter"


@dataclass(frozen=True)
class LocationIngestRun:
    report: StubIngestReport
    elapsed_seconds: Mapping[str, float]
    plan: PlanSummary | None = None


def select_location_batch(output: LocationPageOutput, ingest_pass: LocationPass) -> GraphBatch:
    if ingest_pass is LocationPass.LOCATION:
        return output.batch
    if ingest_pass is LocationPass.CHUNK:
        return output.chunk_batch
    if ingest_pass is LocationPass.FIND:
        return output.find_batch
    return output.encounter_batch


def list_location_constraint_classes(outputs: Iterable[LocationPageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    return list_batch_classes(
        select_location_batch(output, ingest_pass)
        for output in outputs
        for ingest_pass in LocationPass
    )


async def ingest_locations(
    client: GraphClient, run: LocationParseRun, request: IngestRequest
) -> LocationIngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step, the census or the stub list is
    not caught.
    """
    target = await prepare_target(client, request, list_location_constraint_classes(run.outputs))
    passes = [
        (
            ingest_pass,
            [
                PassItem(output.page.title, select_location_batch(output, ingest_pass))
                for output in run.outputs
            ],
        )
        for ingest_pass in LocationPass
    ]
    pass_run = await write_passes(target, passes, request.on_progress)
    is_counted = should_count(pass_run)
    census = await target.runner.count_universe() if is_counted else None
    stub_ids = await target.runner.list_stub_ids(STUB_CLASSES) if is_counted else None
    report = build_ingest_report(request, pass_run, run.report.pages, census)
    return LocationIngestRun(
        report=StubIngestReport(**dict(report), stub_ids=stub_ids),
        elapsed_seconds=pass_run.elapsed_seconds,
        plan=pass_run.plan,
    )


def write_location_ingest_report(report: StubIngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    return write_report_file(report, LOCATION_INGEST_REPORT_FILE_NAME, root)
