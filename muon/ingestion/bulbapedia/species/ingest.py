"""Species ingest: the batches of a `ParseRun` go into the graph in 4 passes.

`ingest_species` checks the seed and ensures the constraints of every class in the run
once, then writes every species batch, then every evolution batch, then every chunk batch,
then every encounter batch, and reads the census of the universe and the ids of the
Location nodes that are still stubs, which the location page run will own. The write loop,
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
    StubIngestReport,
    build_ingest_report,
    list_batch_classes,
    prepare_runner,
    write_passes,
    write_report_file,
)
from muon.ingestion.bulbapedia.species.run import PageOutput, ParseRun

INGEST_REPORT_FILE_NAME = "species-ingest-report.json"
STUB_CLASSES = (LOCATION,)


class IngestPass(StrEnum):
    """The passes in write order: a later pass points at nodes an earlier pass owns."""

    SPECIES = "species"
    EVOLUTION = "evolution"
    CHUNK = "chunk"
    ENCOUNTER = "encounter"


@dataclass(frozen=True)
class IngestRun:
    report: StubIngestReport
    elapsed_seconds: Mapping[str, float]


def select_batch(output: PageOutput, ingest_pass: IngestPass) -> GraphBatch:
    if ingest_pass is IngestPass.SPECIES:
        return output.species_batch
    if ingest_pass is IngestPass.EVOLUTION:
        return output.evolution_batch
    if ingest_pass is IngestPass.CHUNK:
        return output.chunk_batch
    return output.encounter_batch


def list_constraint_classes(outputs: Iterable[PageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    return list_batch_classes(
        select_batch(output, ingest_pass) for output in outputs for ingest_pass in IngestPass
    )


async def ingest_species(client: GraphClient, run: ParseRun, request: IngestRequest) -> IngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step or the census is not caught.
    """
    runner = await prepare_runner(
        client, request.ontology_dir, list_constraint_classes(run.outputs)
    )
    passes = [
        (
            ingest_pass,
            [
                PassItem(output.page.title, select_batch(output, ingest_pass))
                for output in run.outputs
            ],
        )
        for ingest_pass in IngestPass
    ]
    pass_run = await write_passes(runner, passes, request.on_progress)
    is_stopped = pass_run.stop is not None
    census = None if is_stopped else await runner.count_universe()
    stub_ids = None if is_stopped else await runner.list_stub_ids(STUB_CLASSES)
    report = build_ingest_report(request, pass_run, run.report.pages, census)
    return IngestRun(
        report=StubIngestReport(**dict(report), stub_ids=stub_ids),
        elapsed_seconds=pass_run.elapsed_seconds,
    )


def write_ingest_report(report: StubIngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    return write_report_file(report, INGEST_REPORT_FILE_NAME, root)
