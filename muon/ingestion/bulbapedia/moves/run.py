"""A parse run over the pinned move pages: extracts, pass 1 batches, cross-check and report.

`parse_move_pages` reads the page cache, the PokeAPI snapshot and the ontology, never the
network. It parses every pinned page, because the table checks and the machine id check
need all of them, and keeps the outputs and report entries of the requested titles. The
order is fixed: request check, snapshot index, pages, `check_tables`, `check_machine_fields`,
batches, machine id conflicts, cross-check, report. A page that breaks a rule fails alone
and the run goes on; a request, snapshot, table, machine field or machine conflict error
stops the run before anything is written. Output files are byte-identical across runs.

The cross-check reads the extracts and changes no batch (P7).
"""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import (
    MACHINE,
    UNIVERSE,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.moves.batches import build_move_batch, check_machine_fields
from muon.ingestion.bulbapedia.moves.page import (
    MoveNote,
    MoveNoteKind,
    MovePageExtract,
    MovePageParseError,
    SkippedPage,
    SkipReason,
    check_tables,
    parse_move_page,
)
from muon.ingestion.bulbapedia.moves.pokeapi import (
    CrossCheckDifference,
    PokeapiIndex,
    build_pokeapi_index,
    cross_check,
    list_unmatched_pokeapi_moves,
    read_effect_text,
)
from muon.ingestion.bulbapedia.pages import (
    DEFAULT_ROOT,
    Page,
    PagePin,
    PageStore,
    write_text_atomically,
)
from muon.ingestion.pokeapi.snapshot import (
    UPDATE_HINT,
    SnapshotError,
    SnapshotManifest,
    SnapshotStore,
)
from muon.ingestion.seed import read_abox
from muon.ontology.model import extract_local_name, load_ontology_model

PAGE_TYPE = "move"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "parsed"
PAGE_OUTPUT_DIR = "move"
REPORT_FILE_NAME = "move-report.json"
OUTPUT_SUFFIX = ".json"
JSON_INDENT = 2
EFFECT_TEXT_FIELD = "effect_text"
NO_POKEAPI_ROW = "no PokeAPI move"
NO_ENGLISH_TEXT = "no English short effect"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not a pinned move page."""


class MachineConflictError(Exception):
    """2 pages give the same Machine id."""


class ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PageStatus(StrEnum):
    PARSED = "parsed"
    SKIPPED = "skipped"
    FAILED = "failed"


class PageResult(ReportModel):
    title: str
    page_id: int
    revision_id: int
    status: PageStatus
    reason: str | None


class ReportNote(ReportModel):
    page_title: str
    kind: MoveNoteKind
    field: str
    detail: str


class BatchCounts(ReportModel):
    """Counts by class local name (owners and stubs) and property local name (edges, claims)."""

    owner_nodes: Mapping[str, int]
    stubs: Mapping[str, int]
    edges: Mapping[str, int]
    claims: Mapping[str, int]


class MoveParseReport(ReportModel):
    pages: tuple[PageResult, ...]
    pages_parsed: int
    pages_skipped: int
    pages_failed: int
    skipped_per_reason: Mapping[str, int]
    notes: tuple[ReportNote, ...]
    batch_counts: BatchCounts
    moves_per_flag: Mapping[str, int]
    cross_check: tuple[CrossCheckDifference, ...]
    moves_without_pokeapi_row: tuple[str, ...]
    unmatched_pokeapi_moves: tuple[str, ...]
    skipped_pokeapi_machines: int
    snapshot_commit: str
    snapshot_files: Mapping[str, str]


@dataclass(frozen=True)
class MoveParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoveRunInputs:
    request: MoveParseRequest
    store: PageStore
    snapshot: SnapshotStore
    ontology_dir: Path


@dataclass(frozen=True)
class MovePageOutput:
    page: Page
    extract: MovePageExtract
    batch: GraphBatch


@dataclass(frozen=True)
class MoveParseRun:
    outputs: tuple[MovePageOutput, ...]
    report: MoveParseReport


@dataclass(frozen=True)
class ParsedPages:
    """Extracts by title, and the result of every page that has no extract."""

    extracts: Mapping[str, MovePageExtract]
    results: Mapping[str, PageResult]


@dataclass(frozen=True)
class BuiltBatches:
    batches: Mapping[str, GraphBatch]
    failures: Mapping[str, str]


@dataclass(frozen=True)
class SnapshotRead:
    manifest: SnapshotManifest
    index: PokeapiIndex


def build_result(page: Page, status: PageStatus, reason: str | None = None) -> PageResult:
    return PageResult(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        status=status,
        reason=reason,
    )


def check_request(request: MoveParseRequest, pins: Iterable[PagePin]) -> None:
    pinned = {pin.title for pin in pins}
    unknown = sorted(set(request.titles) - pinned)
    if unknown:
        raise ParseRunError(f"Titles that are not pinned move pages: {unknown}")


def read_snapshot(snapshot: SnapshotStore) -> SnapshotRead:
    manifest = snapshot.read_manifest()
    if manifest is None:
        raise SnapshotError(f"no PokeAPI snapshot at {snapshot.root}: {UPDATE_HINT}")
    return SnapshotRead(manifest=manifest, index=build_pokeapi_index(snapshot))


def parse_pages(pages: Iterable[Page]) -> ParsedPages:
    """Parse every page: an out-of-scope page is skipped, a page that breaks a rule fails."""
    extracts: dict[str, MovePageExtract] = {}
    results: dict[str, PageResult] = {}
    for page in pages:
        try:
            parsed = parse_move_page(page)
        except MovePageParseError as error:
            results[page.title] = build_result(page, PageStatus.FAILED, str(error))
            continue
        if isinstance(parsed, SkippedPage):
            results[page.title] = build_result(page, PageStatus.SKIPPED, parsed.reason.value)
            continue
        extracts[page.title] = parsed
    return ParsedPages(extracts=extracts, results=results)


def add_effect_note(extract: MovePageExtract, index: PokeapiIndex) -> MovePageExtract:
    """Add a `no_pokeapi_match` note to a move that gets no `effectText`."""
    if read_effect_text(index, extract.id) is not None:
        return extract
    detail = NO_POKEAPI_ROW if extract.id not in index.moves else NO_ENGLISH_TEXT
    note = MoveNote(kind=MoveNoteKind.NO_POKEAPI_MATCH, field=EFFECT_TEXT_FIELD, detail=detail)
    return extract.model_copy(update={"notes": (*extract.notes, note)})


def build_batches(
    extracts: Mapping[str, MovePageExtract], abox: AboxIndex, index: PokeapiIndex
) -> BuiltBatches:
    batches: dict[str, GraphBatch] = {}
    failures: dict[str, str] = {}
    for title, extract in extracts.items():
        try:
            batches[title] = build_move_batch(extract, abox, read_effect_text(index, extract.id))
        except BatchBuildError as error:
            failures[title] = str(error)
    return BuiltBatches(batches=batches, failures=failures)


def check_machine_conflicts(batches: Mapping[str, GraphBatch]) -> None:
    """Fail when 2 pages own the same Machine id, naming the id and both pages."""
    pages_by_machine: defaultdict[str, list[str]] = defaultdict(list)
    for title, batch in batches.items():
        for node in batch.owner_nodes:
            if node.class_iri == MACHINE:
                pages_by_machine[node.id].append(title)
    conflicts = [
        f"{machine_id} from {' and '.join(sorted(titles))}"
        for machine_id, titles in sorted(pages_by_machine.items())
        if len(titles) > 1
    ]
    if conflicts:
        raise MachineConflictError(f"Machine ids given by more than 1 page: {'; '.join(conflicts)}")


def count_batches(batches: Iterable[GraphBatch]) -> BatchCounts:
    owners: Counter[str] = Counter()
    stubs: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    claims: Counter[str] = Counter()
    for batch in batches:
        owners.update(extract_local_name(node.class_iri) for node in batch.owner_nodes)
        stubs.update(extract_local_name(node.class_iri) for node in batch.stub_nodes)
        edges.update(extract_local_name(edge.property_iri) for edge in batch.edges)
        claims.update(extract_local_name(claim.property_iri) for claim in batch.claims)
    return BatchCounts(
        owner_nodes=dict(owners), stubs=dict(stubs), edges=dict(edges), claims=dict(claims)
    )


def to_report_notes(extract: MovePageExtract) -> list[ReportNote]:
    return [
        ReportNote(page_title=extract.title, kind=note.kind, field=note.field, detail=note.detail)
        for note in extract.notes
    ]


@dataclass(frozen=True)
class ReportInputs:
    """What the report reads beyond the requested outputs and results."""

    snapshot: SnapshotRead
    version_group_ids: Mapping[str, str]
    in_scope_ids: frozenset[str]


def build_report(
    outputs: tuple[MovePageOutput, ...], results: tuple[PageResult, ...], inputs: ReportInputs
) -> MoveParseReport:
    """Collect the outcomes, notes, counts and cross-check of the requested pages."""
    index = inputs.snapshot.index
    flags: Counter[str] = Counter(flag for o in outputs for flag in o.extract.flags)
    skipped: Counter[str] = Counter(
        r.reason or "" for r in results if r.status is PageStatus.SKIPPED
    )
    return MoveParseReport(
        pages=results,
        pages_parsed=sum(r.status is PageStatus.PARSED for r in results),
        pages_skipped=sum(r.status is PageStatus.SKIPPED for r in results),
        pages_failed=sum(r.status is PageStatus.FAILED for r in results),
        skipped_per_reason={reason.value: skipped[reason.value] for reason in SkipReason},
        notes=tuple(note for o in outputs for note in to_report_notes(o.extract)),
        batch_counts=count_batches(o.batch for o in outputs),
        moves_per_flag=dict(sorted(flags.items())),
        cross_check=tuple(
            difference
            for o in outputs
            for difference in cross_check(o.extract, index, inputs.version_group_ids)
        ),
        moves_without_pokeapi_row=tuple(
            o.extract.id for o in outputs if o.extract.id not in index.moves
        ),
        unmatched_pokeapi_moves=list_unmatched_pokeapi_moves(index, inputs.in_scope_ids),
        skipped_pokeapi_machines=index.skipped_machine_count,
        snapshot_commit=inputs.snapshot.manifest.commit,
        snapshot_files=inputs.snapshot.manifest.files,
    )


def resolve_status(title: str, page: Page, parsed: ParsedPages, built: BuiltBatches) -> PageResult:
    if title in built.failures:
        return build_result(page, PageStatus.FAILED, built.failures[title])
    if title in parsed.results:
        return parsed.results[title]
    return build_result(page, PageStatus.PARSED)


def parse_move_pages(inputs: MoveRunInputs) -> MoveParseRun:
    """Parse the pinned move pages and build the pass 1 batch of each.

    Raises `ParseRunError`, `SnapshotError`, `TableError`, `MachineFieldError` or
    `MachineConflictError` before anything is written. The caller writes the outputs with
    `write_outputs`.
    """
    pins = inputs.store.read_manifest().list_pins(PAGE_TYPE)
    check_request(inputs.request, pins)
    snapshot = read_snapshot(inputs.snapshot)
    requested = set(inputs.request.titles) or {pin.title for pin in pins}
    pages = {pin.title: inputs.store.read_page(PAGE_TYPE, pin) for pin in pins}
    parsed = parse_pages(pages.values())
    check_tables(parsed.extracts.values())
    model = load_ontology_model(inputs.ontology_dir, UNIVERSE)
    seed = read_abox(model, inputs.ontology_dir)
    check_machine_fields(seed)
    abox = build_abox_index(seed)
    extracts = {
        title: add_effect_note(extract, snapshot.index)
        for title, extract in parsed.extracts.items()
    }
    built = build_batches(extracts, abox, snapshot.index)
    check_machine_conflicts(built.batches)
    outputs = tuple(
        MovePageOutput(page=pages[title], extract=extracts[title], batch=built.batches[title])
        for title in pages
        if title in requested and title in built.batches
    )
    results = tuple(
        resolve_status(title, page, parsed, built)
        for title, page in pages.items()
        if title in requested
    )
    report_inputs = ReportInputs(
        snapshot=snapshot,
        version_group_ids={
            label: node_id
            for (class_iri, label), node_id in abox.ids.items()
            if class_iri == VERSION_GROUP
        },
        in_scope_ids=frozenset(extract.id for extract in extracts.values()),
    )
    return MoveParseRun(outputs, build_report(outputs, results, report_inputs))


def dump_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=JSON_INDENT) + "\n"


def serialize_batch(batch: GraphBatch) -> dict[str, object]:
    """Dump a batch with its scope sorted: pydantic dumps a frozenset in hash order."""
    data = batch.model_dump(mode="json")
    data["scope"] = sorted(batch.scope)
    return data


def serialize_output(output: MovePageOutput) -> str:
    return dump_json(
        {
            "title": output.page.title,
            "page_id": output.page.page_id,
            "revision_id": output.page.revision_id,
            "extract": output.extract.model_dump(mode="json"),
            "batch": serialize_batch(output.batch),
        }
    )


def write_outputs(run: MoveParseRun, root: Path, is_full_run: bool) -> None:
    """Write the page outputs and the report under `root`, atomically.

    A full run also deletes the output files of every page it produced no output for, so
    the directory depends on the run and not on the runs before it.
    """
    output_dir = root / PAGE_OUTPUT_DIR
    kept = set()
    for output in run.outputs:
        name = f"{output.page.page_id}{OUTPUT_SUFFIX}"
        kept.add(name)
        write_text_atomically(output_dir / name, serialize_output(output))
    write_text_atomically(root / REPORT_FILE_NAME, dump_json(run.report.model_dump(mode="json")))
    if is_full_run:
        for path in output_dir.glob(f"*{OUTPUT_SUFFIX}"):
            if path.name not in kept:
                path.unlink()
