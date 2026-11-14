"""Move ingest: the batches of a `MoveParseRun` go into the graph in 2 passes.

`ingest_moves` checks the seed and ensures the constraints of every class in the run once,
then writes every pass 1 batch (Move and Machine), then every pass 2 batch (TextChunk), and
reads the census of the universe and the ids of the Move and Machine nodes that are still
stubs. The write loop, the stop rule and the report models are the shared ones of
`bulbapedia/ingest.py`.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import MACHINE, MOVE
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
from muon.ingestion.bulbapedia.moves.run import MovePageOutput, MoveParseRun

MOVE_INGEST_REPORT_FILE_NAME = "move-ingest-report.json"
STUB_CLASSES = (MOVE, MACHINE)


class MovePass(StrEnum):
    """The passes in write order: chunks point at the Move nodes pass 1 owns."""

    MOVE = "move"
    CHUNK = "chunk"


@dataclass(frozen=True)
class MoveIngestRun:
    report: StubIngestReport
    elapsed_seconds: Mapping[str, float]


def select_move_batch(output: MovePageOutput, ingest_pass: MovePass) -> GraphBatch:
    if ingest_pass is MovePass.MOVE:
        return output.batch
    return output.chunk_batch


def list_move_constraint_classes(outputs: Iterable[MovePageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    return list_batch_classes(
        select_move_batch(output, ingest_pass) for output in outputs for ingest_pass in MovePass
    )


async def ingest_moves(
    client: GraphClient, run: MoveParseRun, request: IngestRequest
) -> MoveIngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step, the census or the stub list is
    not caught.
    """
    runner = await prepare_runner(
        client, request.ontology_dir, list_move_constraint_classes(run.outputs)
    )
    passes = [
        (
            ingest_pass,
            [
                PassItem(output.page.title, select_move_batch(output, ingest_pass))
                for output in run.outputs
            ],
        )
        for ingest_pass in MovePass
    ]
    pass_run = await write_passes(runner, passes, request.on_progress)
    is_stopped = pass_run.stop is not None
    census = None if is_stopped else await runner.count_universe()
    stub_ids = None if is_stopped else await runner.list_stub_ids(STUB_CLASSES)
    report = build_ingest_report(request, pass_run, run.report.pages, census)
    return MoveIngestRun(
        report=StubIngestReport(**dict(report), stub_ids=stub_ids),
        elapsed_seconds=pass_run.elapsed_seconds,
    )


def write_move_ingest_report(report: StubIngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    return write_report_file(report, MOVE_INGEST_REPORT_FILE_NAME, root)
