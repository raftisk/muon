"""Move ingest: the batches of a `MoveParseRun` go into the graph in 2 passes.

Map:
- `ingest_moves` is the entry point. It checks that the graph holds the A-Box, ensures the
  `id` constraints of every class in the run once, writes every pass 1 batch (Move and
  Machine), then every pass 2 batch (TextChunk), and reads the census of the universe and
  the ids of the Move and Machine nodes that are still stubs.
- `write_move_pass` writes 1 pass through `Runner.write` in upsert mode, 1 transaction per
  batch, with the claim prune of each batch inside its transaction.
- The first batch that raises a `GraphError` or a `ValueError` stops the run: no later
  batch of any pass is written, and the report holds the batches written so far.
- `MoveIngestReport` holds counters and names, never a time, so equal runs give equal
  files. The seconds per pass travel in `MoveIngestRun` for the printed summary only.

The seed check, the change counters and the census model come from the species ingest.
This module knows the pkmn universe, as every ingestion module of a source does.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient, GraphError
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import MACHINE, MOVE, UNIVERSE
from muon.ingestion.bulbapedia.moves.run import MovePageOutput, MoveParseRun
from muon.ingestion.bulbapedia.pages import write_text_atomically
from muon.ingestion.bulbapedia.runs import PageStatus, ReportModel, dump_json
from muon.ingestion.bulbapedia.species.ingest import (
    INGEST_OUTPUT_ROOT,
    PROGRESS_EVERY,
    CensusReport,
    IngestRequest,
    PageFailure,
    PassProgress,
    build_census_report,
    check_seed,
    is_change,
)
from muon.ingestion.runner import Runner, WriteMode
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model

MOVE_INGEST_REPORT_FILE_NAME = "move-ingest-report.json"
STUB_CLASSES = (MOVE, MACHINE)


class MovePass(StrEnum):
    """The passes in write order: chunks point at the Move nodes pass 1 owns."""

    MOVE = "move"
    CHUNK = "chunk"


class MovePassTotals(ReportModel):
    """The batches written in 1 pass and the `RunReport` counters summed over them."""

    ingest_pass: MovePass
    batches_written: int
    counters: Mapping[str, int]


class MovePageChange(ReportModel):
    """The counters of 1 batch that created or deleted a node or a relationship."""

    page_title: str
    ingest_pass: MovePass
    counters: Mapping[str, int]


class MoveIngestStop(ReportModel):
    page_title: str
    ingest_pass: MovePass
    error: str


class MoveIngestReport(ReportModel):
    """`stub_ids` maps a class label to the ids of its nodes with only `id` and `name`."""

    titles: tuple[str, ...]
    is_full_run: bool
    passes: tuple[MovePassTotals, ...]
    changed_pages: tuple[MovePageChange, ...]
    failed_pages: tuple[PageFailure, ...]
    stop: MoveIngestStop | None
    census: CensusReport | None
    stub_ids: Mapping[str, tuple[str, ...]] | None


@dataclass(frozen=True)
class MoveIngestRun:
    report: MoveIngestReport
    elapsed_seconds: Mapping[MovePass, float]


@dataclass(frozen=True)
class MovePassResult:
    totals: MovePassTotals
    changes: tuple[MovePageChange, ...]
    stop: MoveIngestStop | None


def select_move_batch(output: MovePageOutput, ingest_pass: MovePass) -> GraphBatch:
    if ingest_pass is MovePass.MOVE:
        return output.batch
    return output.chunk_batch


def list_move_constraint_classes(outputs: Iterable[MovePageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    classes: set[str] = set()
    for output in outputs:
        for ingest_pass in MovePass:
            batch = select_move_batch(output, ingest_pass)
            classes |= batch.scope
            classes.update(stub.class_iri for stub in batch.stub_nodes)
    return frozenset(classes)


def build_move_totals(
    ingest_pass: MovePass, written: int, counters: Mapping[str, int]
) -> MovePassTotals:
    return MovePassTotals(ingest_pass=ingest_pass, batches_written=written, counters=dict(counters))


async def write_move_pass(
    runner: Runner,
    outputs: Sequence[MovePageOutput],
    ingest_pass: MovePass,
    on_progress: Callable[[PassProgress], None],
) -> MovePassResult:
    """Write the batch of `ingest_pass` of every output, in order, and stop at a failure.

    Progress goes to `on_progress` every `PROGRESS_EVERY` batches and after the last one.
    """
    started = time.monotonic()
    counters: Counter[str] = Counter()
    changes: list[MovePageChange] = []
    written = 0
    for output in outputs:
        try:
            report = await runner.write(
                select_move_batch(output, ingest_pass),
                WriteMode.UPSERT,
                should_ensure_constraints=False,
            )
        except (GraphError, ValueError) as error:
            stop = MoveIngestStop(
                page_title=output.page.title, ingest_pass=ingest_pass, error=str(error)
            )
            totals = build_move_totals(ingest_pass, written, counters)
            return MovePassResult(totals, tuple(changes), stop)
        written += 1
        counters.update(asdict(report))
        if is_change(report):
            changes.append(
                MovePageChange(
                    page_title=output.page.title, ingest_pass=ingest_pass, counters=asdict(report)
                )
            )
        if written % PROGRESS_EVERY == 0 or written == len(outputs):
            elapsed = time.monotonic() - started
            on_progress(PassProgress(ingest_pass, written, len(outputs), elapsed))
    return MovePassResult(build_move_totals(ingest_pass, written, counters), tuple(changes), None)


async def ingest_moves(
    client: GraphClient, run: MoveParseRun, request: IngestRequest
) -> MoveIngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step, the census or the stub list is
    not caught.
    """
    model = load_ontology_model(request.ontology_dir, UNIVERSE)
    materializer = LabelMaterializer(model)
    await check_seed(client, materializer, read_abox(model, request.ontology_dir))

    runner = Runner(client, materializer)
    await runner.ensure_constraints(list_move_constraint_classes(run.outputs))

    results: list[MovePassResult] = []
    seconds: dict[MovePass, float] = {}
    for ingest_pass in MovePass:
        started = time.monotonic()
        result = await write_move_pass(runner, run.outputs, ingest_pass, request.on_progress)
        seconds[ingest_pass] = time.monotonic() - started
        results.append(result)
        if result.stop is not None:
            break

    stop = results[-1].stop
    census = None if stop is not None else await runner.count_universe()
    stub_ids = None if stop is not None else await runner.list_stub_ids(STUB_CLASSES)
    report = MoveIngestReport(
        titles=tuple(sorted(request.titles)),
        is_full_run=not request.titles,
        passes=tuple(result.totals for result in results),
        changed_pages=tuple(change for result in results for change in result.changes),
        failed_pages=tuple(
            PageFailure(page_title=page.title, reason=page.reason or "")
            for page in run.report.pages
            if page.status is PageStatus.FAILED
        ),
        stop=stop,
        census=build_census_report(census),
        stub_ids=stub_ids,
    )
    return MoveIngestRun(report=report, elapsed_seconds=seconds)


def write_move_ingest_report(report: MoveIngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    path = root / MOVE_INGEST_REPORT_FILE_NAME
    write_text_atomically(path, dump_json(report.model_dump(mode="json")))
    return path
