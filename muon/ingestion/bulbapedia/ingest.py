"""Ingest code that every Bulbapedia page type shares: the batches of a parse run go into the
graph pass by pass.

Map:
- `prepare_runner` checks that the graph holds the A-Box (`check_seed`) and ensures the
  `id` constraints of every class the run's batches name, once.
- `write_passes` writes the passes in order through `write_pass`: `Runner.write` in upsert
  mode, 1 transaction per batch, with the claim prune of each batch inside its transaction.
- The first batch that raises a `GraphError` or a `ValueError` stops the run: no later batch
  of any pass is written, and the report holds the batches written so far. Every batch is
  idempotent, so a re-run converges.
- `IngestReport` holds counters and names, never a time, so equal runs give equal files.
  The seconds per pass travel in `PassRun` for the printed summary only.

Each page type names its passes with its own `StrEnum`; the report models store the pass
as its string value, which dumps to the same JSON.
"""

import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from muon.graph import GraphClient, GraphError
from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.bulbapedia.batching import UNIVERSE
from muon.ingestion.bulbapedia.pages import DEFAULT_ROOT, write_text_atomically
from muon.ingestion.bulbapedia.runs import PageResult, PageStatus, ReportModel, dump_json
from muon.ingestion.runner import Runner, RunReport, UniverseCensus, WriteMode
from muon.ingestion.seed import find_missing_seed_nodes, read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import extract_local_name, load_ontology_model

INGEST_OUTPUT_ROOT = DEFAULT_ROOT / "ingested"
PROGRESS_EVERY = 100
MISSING_KEYS_SHOWN = 5
SEED_HINT = "muon seed pkmn"
CHANGE_COUNTERS = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
)


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
    """Progress of 1 pass. `ingest_pass` is the pass enum of the page type that writes."""

    ingest_pass: StrEnum
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

    ingest_pass: str
    batches_written: int
    counters: Mapping[str, int]


class PageChange(ReportModel):
    """The counters of 1 batch that created or deleted a node or a relationship."""

    page_title: str
    ingest_pass: str
    counters: Mapping[str, int]


class PageFailure(ReportModel):
    page_title: str
    reason: str


class IngestStop(ReportModel):
    page_title: str
    ingest_pass: str
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
class PassItem:
    """1 batch to write and the page it came from."""

    page_title: str
    batch: GraphBatch


@dataclass(frozen=True)
class PassResult:
    totals: PassTotals
    changes: tuple[PageChange, ...]
    stop: IngestStop | None


@dataclass(frozen=True)
class PassRun:
    """The passes written, in order, up to and including the 1st one that stopped."""

    results: tuple[PassResult, ...]
    elapsed_seconds: Mapping[str, float]

    @property
    def stop(self) -> IngestStop | None:
        return self.results[-1].stop if self.results else None


async def check_seed(
    client: GraphClient, materializer: LabelMaterializer, abox: GraphBatch
) -> None:
    """Raise `MissingSeedError` unless every A-Box node is in the graph."""
    missing = await find_missing_seed_nodes(client, materializer, abox)
    if missing:
        raise MissingSeedError(missing)


def list_batch_classes(batches: Iterable[GraphBatch]) -> frozenset[str]:
    """Return the scope classes and the stub classes of the batches."""
    classes: set[str] = set()
    for batch in batches:
        classes |= batch.scope
        classes.update(stub.class_iri for stub in batch.stub_nodes)
    return frozenset(classes)


async def prepare_runner(
    client: GraphClient, ontology_dir: Path, constraint_classes: frozenset[str]
) -> Runner:
    """Check the seed, then ensure the `id` constraints of `constraint_classes` once.

    Raises `MissingSeedError` before any write when the A-Box is not in the graph.
    """
    model = load_ontology_model(ontology_dir, UNIVERSE)
    materializer = LabelMaterializer(model)
    await check_seed(client, materializer, read_abox(model, ontology_dir))
    runner = Runner(client, materializer)
    await runner.ensure_constraints(constraint_classes)
    return runner


def is_change(report: RunReport) -> bool:
    counters = asdict(report)
    return any(counters[name] for name in CHANGE_COUNTERS)


def build_totals(ingest_pass: StrEnum, written: int, counters: Mapping[str, int]) -> PassTotals:
    return PassTotals(ingest_pass=ingest_pass, batches_written=written, counters=dict(counters))


async def write_pass(
    runner: Runner,
    items: Sequence[PassItem],
    ingest_pass: StrEnum,
    on_progress: Callable[[PassProgress], None],
) -> PassResult:
    """Write the batches of 1 pass, in order, and stop at a failure.

    Progress goes to `on_progress` every `PROGRESS_EVERY` batches and after the last one.
    """
    started = time.monotonic()
    counters: Counter[str] = Counter()
    changes: list[PageChange] = []
    written = 0
    for item in items:
        try:
            report = await runner.write(
                item.batch, WriteMode.UPSERT, should_ensure_constraints=False
            )
        except (GraphError, ValueError) as error:
            stop = IngestStop(page_title=item.page_title, ingest_pass=ingest_pass, error=str(error))
            return PassResult(build_totals(ingest_pass, written, counters), tuple(changes), stop)
        written += 1
        counters.update(asdict(report))
        if is_change(report):
            changes.append(
                PageChange(
                    page_title=item.page_title, ingest_pass=ingest_pass, counters=asdict(report)
                )
            )
        if written % PROGRESS_EVERY == 0 or written == len(items):
            elapsed = time.monotonic() - started
            on_progress(PassProgress(ingest_pass, written, len(items), elapsed))
    return PassResult(build_totals(ingest_pass, written, counters), tuple(changes), None)


async def write_passes(
    runner: Runner,
    passes: Sequence[tuple[StrEnum, Sequence[PassItem]]],
    on_progress: Callable[[PassProgress], None],
) -> PassRun:
    """Write the passes in order and stop after the 1st pass that stops."""
    results: list[PassResult] = []
    seconds: dict[str, float] = {}
    for ingest_pass, items in passes:
        started = time.monotonic()
        result = await write_pass(runner, items, ingest_pass, on_progress)
        seconds[ingest_pass] = time.monotonic() - started
        results.append(result)
        if result.stop is not None:
            break
    return PassRun(results=tuple(results), elapsed_seconds=seconds)


def build_census_report(census: UniverseCensus | None) -> CensusReport | None:
    if census is None:
        return None
    return CensusReport(
        labels=dict(census.labels), relationship_types=dict(census.relationship_types)
    )


def list_failed_pages(pages: Iterable[PageResult]) -> tuple[PageFailure, ...]:
    return tuple(
        PageFailure(page_title=page.title, reason=page.reason or "")
        for page in pages
        if page.status is PageStatus.FAILED
    )


def build_ingest_report(
    request: IngestRequest,
    pass_run: PassRun,
    pages: Iterable[PageResult],
    census: UniverseCensus | None,
) -> IngestReport:
    return IngestReport(
        titles=tuple(sorted(request.titles)),
        is_full_run=not request.titles,
        passes=tuple(result.totals for result in pass_run.results),
        changed_pages=tuple(change for result in pass_run.results for change in result.changes),
        failed_pages=list_failed_pages(pages),
        stop=pass_run.stop,
        census=build_census_report(census),
    )


def write_report_file(report: ReportModel, file_name: str, root: Path = INGEST_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    path = root / file_name
    write_text_atomically(path, dump_json(report.model_dump(mode="json")))
    return path
