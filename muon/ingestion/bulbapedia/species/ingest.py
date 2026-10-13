"""Species ingest: the batches of a `ParseRun` go into the graph in 3 passes.

Map:
- `ingest_species` is the entry point. It checks that the graph holds the A-Box
  (`check_seed`), ensures the `id` constraints of every class in the run once, writes
  every species batch, then every evolution batch, then every chunk batch, and reads the
  census of the universe.
- `write_pass` writes 1 pass through `Runner.write` in upsert mode, 1 transaction per
  batch. The batch claims delete what the page no longer gives inside that transaction.
- The first batch that raises a `GraphError` or a `ValueError` stops the run: no later
  batch of any pass is written, and the report holds the batches written so far. Every
  batch is idempotent, so a re-run converges.
- `IngestReport` holds counters and names, never a time, so equal runs give equal files.
  The seconds per pass travel in `IngestRun` for the printed summary only.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient, GraphError
from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.bulbapedia.pages import DEFAULT_ROOT, write_text_atomically
from muon.ingestion.bulbapedia.species.batches import UNIVERSE
from muon.ingestion.bulbapedia.species.run import (
    PageOutput,
    PageStatus,
    ParseRun,
    ReportModel,
    dump_json,
)
from muon.ingestion.runner import Runner, RunReport, UniverseCensus, WriteMode
from muon.ingestion.seed import find_missing_seed_nodes, read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import extract_local_name, load_ontology_model

INGEST_OUTPUT_ROOT = DEFAULT_ROOT / "ingested"
INGEST_REPORT_FILE_NAME = "species-ingest-report.json"
PROGRESS_EVERY = 100
MISSING_KEYS_SHOWN = 5
SEED_HINT = "muon seed pkmn"
CHANGE_COUNTERS = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
)


class IngestPass(StrEnum):
    """The passes in write order: a later pass points at nodes an earlier pass owns."""

    SPECIES = "species"
    EVOLUTION = "evolution"
    CHUNK = "chunk"


class MissingSeedError(Exception):
    """The graph lacks A-Box nodes. The message gives the count, some keys and the fix."""

    def __init__(self, missing: Sequence[NodeRef]) -> None:
        self.missing = tuple(missing)
        shown = ", ".join(
            f"{extract_local_name(ref.class_iri)} {ref.id!r}"
            for ref in self.missing[:MISSING_KEYS_SHOWN]
        )
        hidden = len(self.missing) - MISSING_KEYS_SHOWN
        more = f" and {hidden} more" if hidden > 0 else ""
        super().__init__(
            f"{len(self.missing)} A-Box nodes are missing from the graph: {shown}{more}. "
            f"Run `{SEED_HINT}` first."
        )


@dataclass(frozen=True)
class PassProgress:
    ingest_pass: IngestPass
    done: int
    total: int
    elapsed_seconds: float


@dataclass(frozen=True)
class IngestRequest:
    ontology_dir: Path
    titles: tuple[str, ...]
    on_progress: Callable[[PassProgress], None]


class PassTotals(ReportModel):
    """The batches written in 1 pass and the `RunReport` counters summed over them."""

    ingest_pass: IngestPass
    batches_written: int
    counters: Mapping[str, int]


class PageChange(ReportModel):
    """The counters of 1 batch that created or deleted a node or a relationship."""

    page_title: str
    ingest_pass: IngestPass
    counters: Mapping[str, int]


class PageFailure(ReportModel):
    page_title: str
    reason: str


class IngestStop(ReportModel):
    page_title: str
    ingest_pass: IngestPass
    error: str


class CensusReport(ReportModel):
    labels: Mapping[str, int]
    relationship_types: Mapping[str, int]


class IngestReport(ReportModel):
    titles: tuple[str, ...]
    is_full_run: bool
    passes: tuple[PassTotals, ...]
    changed_pages: tuple[PageChange, ...]
    failed_pages: tuple[PageFailure, ...]
    stop: IngestStop | None
    census: CensusReport | None


@dataclass(frozen=True)
class IngestRun:
    report: IngestReport
    elapsed_seconds: Mapping[IngestPass, float]


@dataclass(frozen=True)
class PassResult:
    totals: PassTotals
    changes: tuple[PageChange, ...]
    stop: IngestStop | None


def select_batch(output: PageOutput, ingest_pass: IngestPass) -> GraphBatch:
    if ingest_pass is IngestPass.SPECIES:
        return output.species_batch
    if ingest_pass is IngestPass.EVOLUTION:
        return output.evolution_batch
    return output.chunk_batch


def list_constraint_classes(outputs: Iterable[PageOutput]) -> frozenset[str]:
    """Return the scope classes and the stub classes of every batch of the run."""
    classes: set[str] = set()
    for output in outputs:
        for ingest_pass in IngestPass:
            batch = select_batch(output, ingest_pass)
            classes |= batch.scope
            classes.update(stub.class_iri for stub in batch.stub_nodes)
    return frozenset(classes)


async def check_seed(
    client: GraphClient, materializer: LabelMaterializer, abox: GraphBatch
) -> None:
    """Raise `MissingSeedError` unless every A-Box node is in the graph."""
    missing = await find_missing_seed_nodes(client, materializer, abox)
    if missing:
        raise MissingSeedError(missing)


def is_change(report: RunReport) -> bool:
    counters = asdict(report)
    return any(counters[name] for name in CHANGE_COUNTERS)


async def write_pass(
    runner: Runner,
    outputs: Sequence[PageOutput],
    ingest_pass: IngestPass,
    on_progress: Callable[[PassProgress], None],
) -> PassResult:
    """Write the batch of `ingest_pass` of every output, in order, and stop at a failure.

    Progress goes to `on_progress` every `PROGRESS_EVERY` batches and after the last one.
    """
    started = time.monotonic()
    counters: Counter[str] = Counter()
    changes: list[PageChange] = []
    written = 0
    for output in outputs:
        try:
            report = await runner.write(
                select_batch(output, ingest_pass), WriteMode.UPSERT, should_ensure_constraints=False
            )
        except (GraphError, ValueError) as error:
            stop = IngestStop(
                page_title=output.page.title, ingest_pass=ingest_pass, error=str(error)
            )
            return PassResult(build_totals(ingest_pass, written, counters), tuple(changes), stop)
        written += 1
        counters.update(asdict(report))
        if is_change(report):
            changes.append(
                PageChange(
                    page_title=output.page.title, ingest_pass=ingest_pass, counters=asdict(report)
                )
            )
        if written % PROGRESS_EVERY == 0 or written == len(outputs):
            elapsed = time.monotonic() - started
            on_progress(PassProgress(ingest_pass, written, len(outputs), elapsed))
    return PassResult(build_totals(ingest_pass, written, counters), tuple(changes), None)


def build_totals(ingest_pass: IngestPass, written: int, counters: Mapping[str, int]) -> PassTotals:
    return PassTotals(ingest_pass=ingest_pass, batches_written=written, counters=dict(counters))


def build_census_report(census: UniverseCensus | None) -> CensusReport | None:
    if census is None:
        return None
    return CensusReport(
        labels=dict(census.labels), relationship_types=dict(census.relationship_types)
    )


async def ingest_species(client: GraphClient, run: ParseRun, request: IngestRequest) -> IngestRun:
    """Write the batches of `run` into the graph and report what happened.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph. A
    `GraphError` from the seed check, the constraint step or the census is not caught.
    """
    model = load_ontology_model(request.ontology_dir, UNIVERSE)
    materializer = LabelMaterializer(model)
    await check_seed(client, materializer, read_abox(model, request.ontology_dir))

    runner = Runner(client, materializer)
    await runner.ensure_constraints(list_constraint_classes(run.outputs))

    results: list[PassResult] = []
    seconds: dict[IngestPass, float] = {}
    for ingest_pass in IngestPass:
        started = time.monotonic()
        result = await write_pass(runner, run.outputs, ingest_pass, request.on_progress)
        seconds[ingest_pass] = time.monotonic() - started
        results.append(result)
        if result.stop is not None:
            break

    stop = results[-1].stop
    census = None if stop is not None else await runner.count_universe()
    report = IngestReport(
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
    )
    return IngestRun(report=report, elapsed_seconds=seconds)


def write_ingest_report(report: IngestReport, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    path = root / INGEST_REPORT_FILE_NAME
    write_text_atomically(path, dump_json(report.model_dump(mode="json")))
    return path
