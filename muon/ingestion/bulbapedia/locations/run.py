"""A parse run over the pinned location pages: extracts, pass 1 batches, the report and the
candidates file.

`parse_location_pages` reads the page cache and the ontology, never the network. It reads
and parses every pin, because the location index and the class of every referenced place
need all of them, builds the batches of the requested titles and keeps 1 page output at a
time. A page that breaks a rule fails alone and the run goes on; a table that names something
the A-Box lacks stops the run before anything is written. Output files are byte-identical
across runs.

The candidates file lists every pinned title and every title the pages name that no pin has,
so that the next Special:Export replaces the pins with a complete set.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from muon.files import write_text_atomically
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    CORE,
    MENTIONED_IN,
    REGION,
    UNIVERSE,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    ChunkBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.locations.batches import (
    CLASS_IRIS,
    LOCATION_FAMILY,
    VERSION,
    LocationBuildInputs,
    build_location_batch,
    build_location_chunk_batch,
    build_location_id,
    list_unresolved_mentions,
)
from muon.ingestion.bulbapedia.locations.chunks import LocationChunkReading, read_location_chunks
from muon.ingestion.bulbapedia.locations.identity import (
    LOCATION_INFOBOXES,
    LocationIndex,
    LocationVerdict,
    build_location_index,
    classify_location,
    list_candidate_titles,
    normalize_title,
)
from muon.ingestion.bulbapedia.locations.page import (
    ABOX_REGION_TITLES,
    LocationNoteKind,
    LocationPageExtract,
    LocationTableError,
    SkippedPage,
    SkipReason,
    check_category_table,
    parse_location_page,
)
from muon.ingestion.bulbapedia.pages import Page, PageStore
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
    dump_json,
    find_unpinned_titles,
    rank_templates,
    serialize_batch,
    write_page_files,
)
from muon.ingestion.bulbapedia.wikitext import mask_comments, split_sections
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

PAGE_TYPE = "location"
FOREIGN_PAGE_TYPES = ("species", "item", "move", "ability")
PAGE_OUTPUT_DIR = "location"
REPORT_FILE_NAME = "location-report.json"
CANDIDATES_FILE_NAME = "location-candidates.txt"
ABOX_REGION_CLASS = "abox_region"
PART_OF = CORE + "partOf"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not a pinned location page."""


class ReportNote(ReportModel):
    page_title: str
    kind: LocationNoteKind
    field: str
    detail: str


class LocationParseReport(ReportModel):
    pages: tuple[PageResult, ...]
    pages_parsed: int
    pages_skipped: int
    pages_failed: int
    skipped_per_reason: Mapping[str, int]
    locations_per_class: Mapping[str, int]
    notes: tuple[ReportNote, ...]
    areas: int
    batch_counts: BatchCounts
    duplicate_ids: tuple[str, ...]
    unowned_stub_titles: tuple[str, ...]
    candidate_count: int
    chunk_counts: BatchCounts
    chunks_per_aspect: Mapping[str, int]
    mentions: int
    unresolved_mentions: int
    dropped_templates: tuple[TemplateCount, ...]
    removed_tables_total: int
    math_formulas_total: int
    long_chunks: tuple[str, ...]


@dataclass(frozen=True)
class LocationParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocationRunInputs:
    request: LocationParseRequest
    store: PageStore
    ontology_dir: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT


@dataclass(frozen=True)
class LocationPageOutput:
    """A page with its extract, both batches and its chunk reading. The page has no text."""

    page: Page
    extract: LocationPageExtract
    batch: GraphBatch
    chunk_batch: GraphBatch
    chunk_reading: LocationChunkReading


@dataclass(frozen=True)
class LocationParseRun:
    outputs: tuple[LocationPageOutput, ...]
    report: LocationParseReport
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class ParsedPage:
    """A page without its text, what the parser made of it and the chunks it holds."""

    page: Page
    result: LocationPageExtract | SkippedPage
    chunks: LocationChunkReading | None


def check_request(request: LocationParseRequest, titles: Iterable[str]) -> None:
    unknown = find_unpinned_titles(request.titles, titles)
    if unknown:
        raise ParseRunError(f"Titles that are not pinned location pages: {unknown}")


def build_group_versions(seed: GraphBatch) -> dict[str, tuple[str, ...]]:
    """Map each A-Box VersionGroup label to the labels of its Versions, from the `partOf`
    edges."""
    names = {node.node_ref: node.name for node in seed.nodes}
    versions: dict[str, list[str]] = {}
    for edge in seed.edges:
        if (
            edge.property_iri == PART_OF
            and edge.start.class_iri == VERSION
            and edge.end.class_iri == VERSION_GROUP
        ):
            versions.setdefault(names[edge.end], []).append(names[edge.start])
    return {group: tuple(sorted(labels)) for group, labels in sorted(versions.items())}


def check_abox_regions(abox: AboxIndex) -> None:
    """Raise `LocationTableError` when `ABOX_REGION_TITLES` names a Region the A-Box lacks."""
    missing = sorted(title for title in ABOX_REGION_TITLES if (REGION, title) not in abox.ids)
    if missing:
        raise LocationTableError(f"Region titles the A-Box lacks: {missing}")


def strip_page(page: Page) -> Page:
    return page.model_copy(update={"wikitext": ""})


def parse_page(page: Page, index: LocationIndex) -> ParsedPage:
    """Parse 1 page and read its chunks, then drop its text."""
    result = parse_location_page(page, index)
    if isinstance(result, SkippedPage):
        return ParsedPage(strip_page(page), result, None)
    sections = split_sections(mask_comments(page.wikitext))
    chunks = read_location_chunks(sections, build_location_id(result.title))
    return ParsedPage(strip_page(page), result, chunks)


def parse_pages(pages: Iterable[Page], index: LocationIndex) -> tuple[ParsedPage, ...]:
    """Parse every page in turn and drop its text."""
    return tuple(parse_page(page, index) for page in pages)


def list_extracts(parsed: Iterable[ParsedPage]) -> tuple[LocationPageExtract, ...]:
    return tuple(p.result for p in parsed if isinstance(p.result, LocationPageExtract))


def map_page_classes(extracts: Iterable[LocationPageExtract]) -> dict[str, str]:
    """Map the normalized title of each owned page to its class IRI."""
    return {
        normalize_title(extract.title): CLASS_IRIS[extract.leaf_class]
        for extract in extracts
        if extract.leaf_class is not None
    }


def build_candidate_titles(
    index: LocationIndex, extracts: Iterable[LocationPageExtract]
) -> tuple[str, ...]:
    """Return every pinned title and every title the pages name that no pin has, minus the
    titles other page types pin, sorted."""
    named = {
        title
        for extract in extracts
        for title in (
            *extract.parent_titles,
            *(neighbour.title for neighbour in extract.neighbours),
            *extract.place_titles,
        )
    }
    tests = (classify_location(index, title) for title in sorted(named))
    unknown = [test.title for test in tests if test.verdict is LocationVerdict.UNKNOWN_LOCATION]
    return list_candidate_titles(index, unknown)


def build_outputs(
    parsed: Sequence[ParsedPage], requested: set[str], inputs: LocationBuildInputs
) -> tuple[tuple[LocationPageOutput, ...], tuple[PageResult, ...]]:
    """Build the batch of every requested page. A skipped or failed page gets a result and
    no output."""
    outputs: list[LocationPageOutput] = []
    results: list[PageResult] = []
    for item in parsed:
        if item.page.title not in requested:
            continue
        if isinstance(item.result, SkippedPage) or item.chunks is None:
            reason = item.result.reason.value if isinstance(item.result, SkippedPage) else None
            results.append(build_result(item.page, PageStatus.SKIPPED, reason))
            continue
        try:
            batch = build_location_batch(item.result, inputs)
            chunk_batch = build_location_chunk_batch(item.result, item.chunks, inputs)
        except (BatchBuildError, ChunkBuildError) as error:
            results.append(build_result(item.page, PageStatus.FAILED, str(error)))
            continue
        outputs.append(LocationPageOutput(item.page, item.result, batch, chunk_batch, item.chunks))
        results.append(build_result(item.page, PageStatus.PARSED))
    return tuple(outputs), tuple(results)


def list_unowned_stub_titles(outputs: Iterable[LocationPageOutput]) -> tuple[str, ...]:
    """Return the titles of the Location family references that no output owns."""
    owned = {build_location_id(output.extract.title) for output in outputs}
    names = {
        node.id: node.name
        for output in outputs
        for node in output.batch.nodes
        if node.class_iri in LOCATION_FAMILY and node.id not in owned
    }
    return tuple(sorted(names.values()))


def find_duplicate_ids(extracts: Iterable[LocationPageExtract]) -> tuple[str, ...]:
    """Return the ids that 2 pages share, each with its titles."""
    titles: dict[str, list[str]] = {}
    for extract in extracts:
        titles.setdefault(build_location_id(extract.title), []).append(extract.title)
    return tuple(
        f"{node_id}: {', '.join(sorted(names))}"
        for node_id, names in sorted(titles.items())
        if len(names) > 1
    )


def count_edges(batches: Iterable[GraphBatch], property_iri: str) -> int:
    return sum(edge.property_iri == property_iri for batch in batches for edge in batch.edges)


def list_long_chunks(outputs: Iterable[LocationPageOutput]) -> tuple[str, ...]:
    """Return `title: section` for each chunk above the long chunk length."""
    return tuple(
        f"{output.extract.title}: {chunk.section_path}"
        for output in outputs
        for chunk in output.chunk_reading.long_chunks
    )


def count_chunk_aspects(outputs: Iterable[LocationPageOutput]) -> dict[str, int]:
    aspects: Counter[str] = Counter(
        str(node.properties[ASPECT]) for o in outputs for node in o.chunk_batch.owner_nodes
    )
    return dict(sorted(aspects.items()))


def build_report(
    outputs: tuple[LocationPageOutput, ...],
    results: tuple[PageResult, ...],
    extracts: Sequence[LocationPageExtract],
    candidates: Sequence[str],
    unresolved_mentions: int,
) -> LocationParseReport:
    skipped: Counter[str] = Counter(
        result.reason or "" for result in results if result.status is PageStatus.SKIPPED
    )
    classes: Counter[str] = Counter(
        output.extract.leaf_class.value if output.extract.leaf_class else ABOX_REGION_CLASS
        for output in outputs
    )
    dropped: Counter[str] = Counter()
    for output in outputs:
        dropped.update(output.chunk_reading.dropped_templates)
    return LocationParseReport(
        pages=results,
        pages_parsed=sum(r.status is PageStatus.PARSED for r in results),
        pages_skipped=sum(r.status is PageStatus.SKIPPED for r in results),
        pages_failed=sum(r.status is PageStatus.FAILED for r in results),
        skipped_per_reason={reason.value: skipped[reason.value] for reason in SkipReason},
        locations_per_class=dict(sorted(classes.items())),
        notes=tuple(
            ReportNote(page_title=o.extract.title, kind=n.kind, field=n.field, detail=n.detail)
            for o in outputs
            for n in o.extract.notes
        ),
        areas=sum(len(o.extract.areas) for o in outputs),
        batch_counts=count_batches(o.batch for o in outputs),
        duplicate_ids=find_duplicate_ids(extracts),
        unowned_stub_titles=list_unowned_stub_titles(outputs),
        candidate_count=len(candidates),
        chunk_counts=count_batches(o.chunk_batch for o in outputs),
        chunks_per_aspect=count_chunk_aspects(outputs),
        mentions=count_edges((o.chunk_batch for o in outputs), MENTIONED_IN),
        unresolved_mentions=unresolved_mentions,
        dropped_templates=rank_templates(dropped),
        removed_tables_total=sum(o.chunk_reading.removed_tables for o in outputs),
        math_formulas_total=sum(o.chunk_reading.math_formulas for o in outputs),
        long_chunks=list_long_chunks(outputs),
    )


def parse_location_pages(inputs: LocationRunInputs) -> LocationParseRun:
    """Parse the pinned location pages and build the pass 1 and 2 batches of each requested page.

    Raises `ParseRunError` or `LocationTableError` before anything is written. The caller
    writes the outputs with `write_outputs`. Every pin is parsed whatever the request, so a
    `--title` run resolves classes as a full run does; the candidates cover every page.
    """
    manifest = inputs.store.read_manifest()
    pins = manifest.list_pins(PAGE_TYPE)
    check_request(inputs.request, (pin.title for pin in pins))
    requested = set(inputs.request.titles) or {pin.title for pin in pins}
    model = load_ontology_model(inputs.ontology_dir, UNIVERSE)
    seed = read_abox(model, inputs.ontology_dir)
    abox = build_abox_index(seed)
    group_versions = build_group_versions(seed)
    check_category_table(group_versions)
    check_abox_regions(abox)
    foreign = [pin.title for name in FOREIGN_PAGE_TYPES for pin in manifest.list_pins(name)]
    pages = [inputs.store.read_page(PAGE_TYPE, pin) for pin in pins]
    index = build_location_index(pages, foreign, LOCATION_INFOBOXES)
    parsed = parse_pages(pages, index)
    extracts = list_extracts(parsed)
    resolver = build_species_resolver(manifest.list_pins(SPECIES_PAGE_TYPE))
    build_inputs = LocationBuildInputs(
        abox, index, map_page_classes(extracts), group_versions, resolver
    )
    outputs, results = build_outputs(parsed, requested, build_inputs)
    candidates = build_candidate_titles(index, extracts)
    unresolved = sum(
        len(list_unresolved_mentions(output.chunk_reading, build_inputs)) for output in outputs
    )
    report = build_report(outputs, results, extracts, candidates, unresolved)
    return LocationParseRun(outputs, report, candidates)


def serialize_output(output: LocationPageOutput) -> str:
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


def write_outputs(run: LocationParseRun, root: Path, is_full_run: bool) -> None:
    """Write the page outputs, the report and the candidates under `root`, atomically. A full
    run also deletes the output files of every page it produced no output for."""
    files = (
        (f"{output.page.page_id}{OUTPUT_SUFFIX}", serialize_output(output))
        for output in run.outputs
    )
    write_page_files(root / PAGE_OUTPUT_DIR, files, is_full_run)
    write_text_atomically(root / REPORT_FILE_NAME, dump_json(run.report.model_dump(mode="json")))
    write_text_atomically(root / CANDIDATES_FILE_NAME, "\n".join(run.candidates) + "\n")
