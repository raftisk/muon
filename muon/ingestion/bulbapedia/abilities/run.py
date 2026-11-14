"""A parse run over the pinned ability pages: extracts, chunks, pass 1 and 2 batches, the
species cross-check and the report.

`parse_ability_pages` reads the page cache, the PokeAPI snapshot, the ontology and the
species parse output, never the network. The order is fixed: request check, snapshot
index, pages, `check_sup_codes`, chunks, pass 1 for every page, pass 2 for every page,
cross-check, report. A page that breaks a rule fails alone and the run goes on; a request,
snapshot, `sup` code or species output error stops the run before anything is written.
Output files are byte-identical across runs.

The cross-check reads the extracts and the species output and changes no batch.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.abilities.batches import (
    AbilityChunkBuild,
    AbilityChunkInputs,
    build_ability_batch,
    build_ability_chunk_batch,
    check_sup_codes,
)
from muon.ingestion.bulbapedia.abilities.chunks import (
    build_ability_version_groups,
    build_chunk_notes,
    read_ability_chunks,
)
from muon.ingestion.bulbapedia.abilities.crosscheck import (
    HolderDifference,
    SpeciesAbilityIndex,
    cross_check_species,
    read_species_abilities,
)
from muon.ingestion.bulbapedia.abilities.page import (
    AbilityNote,
    AbilityNoteKind,
    AbilityPageExtract,
    AbilityPageParseError,
    SkippedPage,
    SkipReason,
    parse_ability_page,
)
from muon.ingestion.bulbapedia.abilities.pokeapi import (
    PokeapiAbilityIndex,
    build_pokeapi_index,
    read_effect_text,
)
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    UNIVERSE,
    AboxIndex,
    BatchBuildError,
    ChunkBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.pages import Page, PagePin, PageStore, write_text_atomically
from muon.ingestion.bulbapedia.runs import (
    DEFAULT_OUTPUT_ROOT,
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
from muon.ingestion.bulbapedia.wikitext import split_sections
from muon.ingestion.pokeapi.snapshot import (
    UPDATE_HINT,
    SnapshotError,
    SnapshotManifest,
    SnapshotStore,
)
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

PAGE_TYPE = "ability"
PAGE_OUTPUT_DIR = "ability"
REPORT_FILE_NAME = "ability-report.json"
SPECIES_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / SPECIES_PAGE_TYPE
EFFECT_TEXT_FIELD = "effect_text"
NO_POKEAPI_ROW = "no PokeAPI ability"
NO_ENGLISH_TEXT = "no English short effect"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not a pinned ability page."""


class ReportNote(ReportModel):
    page_title: str
    kind: AbilityNoteKind
    field: str
    detail: str


class AbilityParseReport(ReportModel):
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
    abilities_per_category: Mapping[str, int]
    species_output_missing: bool
    holder_differences: tuple[HolderDifference, ...]
    unowned_species_stubs: tuple[str, ...]
    abilities_without_pokeapi_row: tuple[str, ...]
    snapshot_commit: str
    snapshot_files: Mapping[str, str]


@dataclass(frozen=True)
class AbilityParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbilityRunInputs:
    request: AbilityParseRequest
    store: PageStore
    snapshot: SnapshotStore
    ontology_dir: Path
    species_output_dir: Path = SPECIES_OUTPUT_DIR


@dataclass(frozen=True)
class AbilityPageOutput:
    page: Page
    extract: AbilityPageExtract
    batch: GraphBatch
    chunk_batch: GraphBatch
    chunk_notes: tuple[AbilityNote, ...]


@dataclass(frozen=True)
class AbilityParseRun:
    outputs: tuple[AbilityPageOutput, ...]
    report: AbilityParseReport


@dataclass(frozen=True)
class ParsedPages:
    """Extracts by title, and the result of every page that has no extract."""

    extracts: Mapping[str, AbilityPageExtract]
    results: Mapping[str, PageResult]


@dataclass(frozen=True)
class BuiltBatches:
    batches: Mapping[str, GraphBatch]
    chunk_builds: Mapping[str, AbilityChunkBuild]
    failures: Mapping[str, str]


@dataclass(frozen=True)
class SnapshotRead:
    manifest: SnapshotManifest
    index: PokeapiAbilityIndex


@dataclass(frozen=True)
class ReportInputs:
    """What the report reads beyond the requested outputs and results."""

    snapshot: SnapshotRead
    species_index: SpeciesAbilityIndex | None
    in_scope_ids: frozenset[str]


def check_request(request: AbilityParseRequest, pins: Iterable[PagePin]) -> None:
    unknown = find_unpinned_titles(request.titles, (pin.title for pin in pins))
    if unknown:
        raise ParseRunError(f"Titles that are not pinned ability pages: {unknown}")


def read_snapshot(snapshot: SnapshotStore) -> SnapshotRead:
    manifest = snapshot.read_manifest()
    if manifest is None:
        raise SnapshotError(f"no PokeAPI snapshot at {snapshot.root}: {UPDATE_HINT}")
    return SnapshotRead(manifest=manifest, index=build_pokeapi_index(snapshot))


def parse_pages(pages: Iterable[Page]) -> ParsedPages:
    """Parse every page: a page without the infobox is skipped, one that breaks a rule fails."""
    extracts: dict[str, AbilityPageExtract] = {}
    results: dict[str, PageResult] = {}
    for page in pages:
        try:
            parsed = parse_ability_page(page)
        except AbilityPageParseError as error:
            results[page.title] = build_result(page, PageStatus.FAILED, str(error))
            continue
        if isinstance(parsed, SkippedPage):
            results[page.title] = build_result(page, PageStatus.SKIPPED, parsed.reason.value)
            continue
        extracts[page.title] = parsed
    return ParsedPages(extracts=extracts, results=results)


def build_effect_notes(
    extract: AbilityPageExtract, index: PokeapiAbilityIndex
) -> tuple[AbilityNote, ...]:
    """Return a `no_pokeapi_match` note for an ability that gets no `effectText`."""
    if read_effect_text(index, extract.id) is not None:
        return ()
    detail = NO_POKEAPI_ROW if extract.id not in index.identifiers else NO_ENGLISH_TEXT
    return (
        AbilityNote(kind=AbilityNoteKind.NO_POKEAPI_MATCH, field=EFFECT_TEXT_FIELD, detail=detail),
    )


def add_chunks(
    extract: AbilityPageExtract,
    page: Page,
    version_groups: Mapping[str, tuple[str, ...]],
    index: PokeapiAbilityIndex,
) -> AbilityPageExtract:
    """Return the extract with its chunks, their cleanup counts and its notes."""
    reading = read_ability_chunks(
        split_sections(page.wikitext), extract.desc_rows, extract.id, version_groups
    )
    return extract.model_copy(
        update={
            "chunks": reading.chunks,
            "dropped_templates": reading.dropped_templates,
            "removed_tables": reading.removed_tables,
            "math_formulas": reading.math_formulas,
            "notes": (
                *extract.notes,
                *build_effect_notes(extract, index),
                *build_chunk_notes(reading),
            ),
        }
    )


@dataclass(frozen=True)
class BatchInputs:
    """What both passes resolve against."""

    abox: AboxIndex
    index: PokeapiAbilityIndex
    pages: Mapping[str, Page]
    species_pins: tuple[PagePin, ...]


def build_batches(extracts: Mapping[str, AbilityPageExtract], inputs: BatchInputs) -> BuiltBatches:
    """Build pass 1 of every page, then pass 2 of every page. A page that fails either pass
    gets no batch."""
    batches: dict[str, GraphBatch] = {}
    failures: dict[str, str] = {}
    for title, extract in extracts.items():
        effect_text = read_effect_text(inputs.index, extract.id)
        try:
            batches[title] = build_ability_batch(extract, inputs.abox, effect_text)
        except BatchBuildError as error:
            failures[title] = str(error)
    resolve_species = build_species_resolver(inputs.species_pins)
    chunk_builds: dict[str, AbilityChunkBuild] = {}
    for title in batches:
        chunk_inputs = AbilityChunkInputs(
            inputs.abox, resolve_species, inputs.pages[title].permalink
        )
        try:
            chunk_builds[title] = build_ability_chunk_batch(extracts[title], chunk_inputs)
        except (BatchBuildError, ChunkBuildError) as error:
            failures[title] = str(error)
    return BuiltBatches(
        batches={title: batches[title] for title in chunk_builds},
        chunk_builds=chunk_builds,
        failures=failures,
    )


def to_report_notes(output: AbilityPageOutput) -> list[ReportNote]:
    title = output.extract.title
    return [
        ReportNote(page_title=title, kind=note.kind, field=note.field, detail=note.detail)
        for note in (*output.extract.notes, *output.chunk_notes)
    ]


def count_templates(outputs: Iterable[AbilityPageOutput]) -> tuple[TemplateCount, ...]:
    counts: Counter[str] = Counter()
    for output in outputs:
        counts.update(output.extract.dropped_templates)
    return rank_templates(counts)


def count_per_page(
    outputs: Iterable[AbilityPageOutput], read_count: Callable[[AbilityPageExtract], int]
) -> dict[str, int]:
    return count_nonzero((o.extract.title, read_count(o.extract)) for o in outputs)


def list_unowned_species_stubs(
    species_index: SpeciesAbilityIndex | None, in_scope_ids: frozenset[str]
) -> tuple[str, ...]:
    """Return the Ability ids the species output points at that no ability page owns."""
    if species_index is None:
        return ()
    return tuple(sorted(set(species_index.species_by_ability) - in_scope_ids))


def build_report(
    outputs: tuple[AbilityPageOutput, ...],
    results: tuple[PageResult, ...],
    inputs: ReportInputs,
) -> AbilityParseReport:
    """Collect the outcomes, notes, counts and cross-checks of the requested pages."""
    index = inputs.snapshot.index
    aspects: Counter[str] = Counter(
        str(node.properties[ASPECT]) for o in outputs for node in o.chunk_batch.owner_nodes
    )
    categories: Counter[str] = Counter(
        category for o in outputs for category in o.extract.ability_categories
    )
    tables = count_per_page(outputs, lambda extract: extract.removed_tables)
    formulas = count_per_page(outputs, lambda extract: extract.math_formulas)
    skipped: Counter[str] = Counter(
        r.reason or "" for r in results if r.status is PageStatus.SKIPPED
    )
    differences = (
        ()
        if inputs.species_index is None
        else cross_check_species((o.extract for o in outputs), inputs.species_index)
    )
    return AbilityParseReport(
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
        abilities_per_category=dict(sorted(categories.items())),
        species_output_missing=inputs.species_index is None,
        holder_differences=differences,
        unowned_species_stubs=list_unowned_species_stubs(inputs.species_index, inputs.in_scope_ids),
        abilities_without_pokeapi_row=tuple(
            o.extract.id for o in outputs if o.extract.id not in index.identifiers
        ),
        snapshot_commit=inputs.snapshot.manifest.commit,
        snapshot_files=inputs.snapshot.manifest.files,
    )


def resolve_status(title: str, page: Page, parsed: ParsedPages, built: BuiltBatches) -> PageResult:
    if title in built.failures:
        return build_result(page, PageStatus.FAILED, built.failures[title])
    if title in parsed.results:
        return parsed.results[title]
    return build_result(page, PageStatus.PARSED)


def parse_ability_pages(inputs: AbilityRunInputs) -> AbilityParseRun:
    """Parse the pinned ability pages and build the pass 1 and pass 2 batches of each.

    Raises `ParseRunError`, `SnapshotError`, `SupCodeError` or `CrossCheckError` before
    anything is written. The caller writes the outputs with `write_outputs`.
    """
    manifest = inputs.store.read_manifest()
    pins = manifest.list_pins(PAGE_TYPE)
    check_request(inputs.request, pins)
    snapshot = read_snapshot(inputs.snapshot)
    requested = set(inputs.request.titles) or {pin.title for pin in pins}
    pages = {pin.title: inputs.store.read_page(PAGE_TYPE, pin) for pin in pins}
    parsed = parse_pages(pages.values())
    model = load_ontology_model(inputs.ontology_dir, UNIVERSE)
    seed = read_abox(model, inputs.ontology_dir)
    check_sup_codes(seed)
    version_groups = build_ability_version_groups(seed)
    extracts = {
        title: add_chunks(extract, pages[title], version_groups, snapshot.index)
        for title, extract in parsed.extracts.items()
    }
    batch_inputs = BatchInputs(
        abox=build_abox_index(seed),
        index=snapshot.index,
        pages=pages,
        species_pins=tuple(manifest.list_pins(SPECIES_PAGE_TYPE)),
    )
    built = build_batches(extracts, batch_inputs)
    species_index = read_species_abilities(inputs.species_output_dir)
    outputs = tuple(
        AbilityPageOutput(
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
        species_index=species_index,
        in_scope_ids=frozenset(extract.id for extract in extracts.values()),
    )
    return AbilityParseRun(outputs, build_report(outputs, results, report_inputs))


def serialize_output(output: AbilityPageOutput) -> str:
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


def write_outputs(run: AbilityParseRun, root: Path, is_full_run: bool) -> None:
    """Write the page outputs and the report under `root`, atomically. A full run also
    deletes the output files of every page it produced no output for."""
    files = (
        (f"{output.page.page_id}{OUTPUT_SUFFIX}", serialize_output(output))
        for output in run.outputs
    )
    write_page_files(root / PAGE_OUTPUT_DIR, files, is_full_run)
    write_text_atomically(root / REPORT_FILE_NAME, dump_json(run.report.model_dump(mode="json")))
