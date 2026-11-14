"""A parse run over the pinned move pages: extracts, pass 1 and 2 batches, cross-check and
report.

`parse_move_pages` reads the page cache, the PokeAPI snapshot and the ontology, never the
network. It parses every pinned page, because the table checks and the machine id check
need all of them, and keeps the outputs and report entries of the requested titles. The
order is fixed: request check, snapshot index, pages, `check_tables`, `check_machine_fields`,
`check_game_codes`, `check_type_overrides`, batches, machine id conflicts, cross-check,
report. A page that breaks a rule fails alone and the run goes on; a request, snapshot,
table, machine field, game code or machine conflict error stops the run before anything is
written. Output files are byte-identical across runs.

The cross-check reads the extracts and changes no batch (P7).
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    MACHINE,
    UNIVERSE,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    ChunkBuildError,
    SpeciesResolver,
    build_abox_index,
)
from muon.ingestion.bulbapedia.moves.batches import (
    MoveChunkBuild,
    MoveChunkInputs,
    build_move_batch,
    build_move_chunk_batch,
    check_game_codes,
    check_machine_fields,
    check_type_overrides,
)
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
from muon.ingestion.bulbapedia.pages import Page, PagePin, PageStore, write_text_atomically
from muon.ingestion.bulbapedia.runs import (
    OUTPUT_SUFFIX,
    SPECIES_PAGE_TYPE,
    BatchCounts,
    PageResult,
    PageStatus,
    ReportModel,
    TemplateCount,
    build_result,
    build_species_resolver,
    count_batches,
    count_nonzero,
    dump_json,
    find_unpinned_titles,
    rank_templates,
    serialize_batch,
    write_page_files,
)
from muon.ingestion.pokeapi.snapshot import (
    UPDATE_HINT,
    SnapshotError,
    SnapshotManifest,
    SnapshotStore,
)
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

PAGE_TYPE = "move"
PAGE_OUTPUT_DIR = "move"
REPORT_FILE_NAME = "move-report.json"
EFFECT_TEXT_FIELD = "effect_text"
NO_POKEAPI_ROW = "no PokeAPI move"
NO_ENGLISH_TEXT = "no English short effect"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not a pinned move page."""


class MachineConflictError(Exception):
    """2 pages give the same Machine id."""


class ReportNote(ReportModel):
    page_title: str
    kind: MoveNoteKind
    field: str
    detail: str


class MoveParseReport(ReportModel):
    pages: tuple[PageResult, ...]
    pages_parsed: int
    pages_skipped: int
    pages_failed: int
    skipped_per_reason: Mapping[str, int]
    notes: tuple[ReportNote, ...]
    batch_counts: BatchCounts
    chunk_counts: BatchCounts
    chunks_per_aspect: Mapping[str, int]
    dropped_templates: tuple[TemplateCount, ...]
    removed_tables_total: int
    removed_tables_per_page: Mapping[str, int]
    math_formulas_total: int
    math_formulas_per_page: Mapping[str, int]
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
    chunk_batch: GraphBatch
    chunk_notes: tuple[MoveNote, ...]


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
    chunk_builds: Mapping[str, MoveChunkBuild]
    failures: Mapping[str, str]


@dataclass(frozen=True)
class BatchInputs:
    """What both passes resolve against."""

    abox: AboxIndex
    index: PokeapiIndex
    resolve_species: SpeciesResolver


@dataclass(frozen=True)
class SnapshotRead:
    manifest: SnapshotManifest
    index: PokeapiIndex


def check_request(request: MoveParseRequest, pins: Iterable[PagePin]) -> None:
    pinned = {pin.title for pin in pins}
    unknown = find_unpinned_titles(request.titles, pinned)
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
    extracts: Mapping[str, MovePageExtract],
    pages: Mapping[str, Page],
    inputs: BatchInputs,
) -> BuiltBatches:
    """Build both passes of every page. A page that fails either pass gets no batch."""
    batches: dict[str, GraphBatch] = {}
    chunk_builds: dict[str, MoveChunkBuild] = {}
    failures: dict[str, str] = {}
    for title, extract in extracts.items():
        effect_text = read_effect_text(inputs.index, extract.id)
        chunk_inputs = MoveChunkInputs(inputs.abox, inputs.resolve_species, pages[title].permalink)
        try:
            batch = build_move_batch(extract, inputs.abox, effect_text)
            chunk_build = build_move_chunk_batch(extract, chunk_inputs)
        except (BatchBuildError, ChunkBuildError) as error:
            failures[title] = str(error)
            continue
        batches[title] = batch
        chunk_builds[title] = chunk_build
    return BuiltBatches(batches=batches, chunk_builds=chunk_builds, failures=failures)


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


def to_report_notes(output: MovePageOutput) -> list[ReportNote]:
    title = output.extract.title
    return [
        ReportNote(page_title=title, kind=note.kind, field=note.field, detail=note.detail)
        for note in (*output.extract.notes, *output.chunk_notes)
    ]


def count_templates(outputs: Iterable[MovePageOutput]) -> tuple[TemplateCount, ...]:
    """Sum the dropped templates of the pages, most dropped first, then by name."""
    counts: Counter[str] = Counter()
    for output in outputs:
        counts.update(output.extract.dropped_templates)
    return rank_templates(counts)


def count_per_page(
    outputs: Iterable[MovePageOutput], read_count: Callable[[MovePageExtract], int]
) -> dict[str, int]:
    """Map each page title whose extract gives a nonzero count to that count."""
    return count_nonzero((o.extract.title, read_count(o.extract)) for o in outputs)


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
    aspects: Counter[str] = Counter(
        str(node.properties[ASPECT]) for o in outputs for node in o.chunk_batch.owner_nodes
    )
    tables = count_per_page(outputs, lambda extract: extract.removed_tables)
    formulas = count_per_page(outputs, lambda extract: extract.math_formulas)
    skipped: Counter[str] = Counter(
        r.reason or "" for r in results if r.status is PageStatus.SKIPPED
    )
    return MoveParseReport(
        pages=results,
        pages_parsed=sum(r.status is PageStatus.PARSED for r in results),
        pages_skipped=sum(r.status is PageStatus.SKIPPED for r in results),
        pages_failed=sum(r.status is PageStatus.FAILED for r in results),
        skipped_per_reason={reason.value: skipped[reason.value] for reason in SkipReason},
        notes=tuple(note for o in outputs for note in to_report_notes(o)),
        batch_counts=count_batches(o.batch for o in outputs),
        chunk_counts=count_batches(o.chunk_batch for o in outputs),
        chunks_per_aspect=dict(sorted(aspects.items())),
        dropped_templates=count_templates(outputs),
        removed_tables_total=sum(tables.values()),
        removed_tables_per_page=tables,
        math_formulas_total=sum(formulas.values()),
        math_formulas_per_page=formulas,
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
    """Parse the pinned move pages and build the pass 1 and pass 2 batches of each.

    Raises `ParseRunError`, `SnapshotError`, `TableError`, `MachineFieldError`,
    `GameCodeError` or `MachineConflictError` before anything is written. The caller writes
    the outputs with `write_outputs`.
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
    check_game_codes(seed)
    check_type_overrides(seed)
    abox = build_abox_index(seed)
    species_pins = inputs.store.read_manifest().list_pins(SPECIES_PAGE_TYPE)
    extracts = {
        title: add_effect_note(extract, snapshot.index)
        for title, extract in parsed.extracts.items()
    }
    batch_inputs = BatchInputs(abox, snapshot.index, build_species_resolver(species_pins))
    built = build_batches(extracts, pages, batch_inputs)
    check_machine_conflicts(built.batches)
    outputs = tuple(
        MovePageOutput(
            page=pages[title],
            extract=extracts[title],
            batch=built.batches[title],
            chunk_batch=built.chunk_builds[title].batch,
            chunk_notes=built.chunk_builds[title].notes,
        )
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


def serialize_output(output: MovePageOutput) -> str:
    return dump_json(
        {
            "title": output.page.title,
            "page_id": output.page.page_id,
            "revision_id": output.page.revision_id,
            "extract": output.extract.model_dump(mode="json"),
            "batch": serialize_batch(output.batch),
            "chunk_batch": serialize_batch(output.chunk_batch),
        }
    )


def write_outputs(run: MoveParseRun, root: Path, is_full_run: bool) -> None:
    """Write the page outputs and the report under `root`, atomically. A full run also
    deletes the output files of every page it produced no output for."""
    files = (
        (f"{output.page.page_id}{OUTPUT_SUFFIX}", serialize_output(output))
        for output in run.outputs
    )
    write_page_files(root / PAGE_OUTPUT_DIR, files, is_full_run)
    write_text_atomically(root / REPORT_FILE_NAME, dump_json(run.report.model_dump(mode="json")))
