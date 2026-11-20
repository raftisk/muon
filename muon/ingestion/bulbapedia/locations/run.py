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

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

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
    ENCOUNTER_KIND,
    HAS_ENCOUNTER,
    IN_VERSION,
    IS_HIDDEN,
    LOCATION_FAMILY,
    METHOD,
    VERSION,
    EncounterBuild,
    FindBuild,
    LocationBuildInputs,
    build_encounter_batch,
    build_find_batch,
    build_location_batch,
    build_location_chunk_batch,
    build_location_id,
    list_unresolved_mentions,
)
from muon.ingestion.bulbapedia.locations.chunks import LocationChunkReading, read_location_chunks
from muon.ingestion.bulbapedia.locations.encounter_templates import check_catch_templates
from muon.ingestion.bulbapedia.locations.encounters import (
    EncounterReading,
    EncounterSkipReason,
    read_encounters,
)
from muon.ingestion.bulbapedia.locations.finds import (
    FindFact,
    FindReading,
    FindSkipReason,
    check_version_codes,
    read_item_finds,
)
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
    SPECIES_TITLE_SUFFIX,
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
from muon.ingestion.bulbapedia.species.forms import build_species_id
from muon.ingestion.bulbapedia.wikitext import mask_comments, split_sections
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotStore
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

PAGE_TYPE = "location"
FOREIGN_PAGE_TYPES = ("species", "item", "move", "ability")
PAGE_OUTPUT_DIR = "location"
REPORT_FILE_NAME = "location-report.json"
CANDIDATES_FILE_NAME = "location-candidates.txt"
COVERAGE_FILE_NAME = "location-coverage.json"
ABOX_REGION_CLASS = "abox_region"
ITEM_PAGE_TYPE = "item"
TEXT_ENCODING = "utf-8"
NO_METHOD = "none"
UNMAPPED_REPORT_LIMIT = 40
POKEAPI_LOCATIONS_FILE = "locations.csv"
POKEAPI_IDENTIFIER_COLUMN = "identifier"
POKEAPI_REGION_COLUMN = "region_id"
# PokeAPI region ids of Kanto to Paldea; the other regions are side games.
CORE_REGION_IDS = frozenset(range(1, 11))
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
    find_counts: BatchCounts
    finds_per_method: Mapping[str, int]
    hidden_finds: int
    find_skips: Mapping[str, int]
    duplicate_finds: int
    item_stub_names: tuple[str, ...]
    is_item_source_missing: bool
    pokeapi_unmatched: tuple[str, ...]
    is_pokeapi_source_missing: bool
    encounter_counts: BatchCounts
    encounters_per_kind: Mapping[str, int]
    encounter_skips: Mapping[str, int]
    unmapped_encounter_values: Mapping[str, int]
    merged_encounters: int
    duplicate_encounters: int
    unresolved_species: tuple[str, ...]
    coverage_pairs: int


@dataclass(frozen=True)
class LocationParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocationRunInputs:
    request: LocationParseRequest
    store: PageStore
    ontology_dir: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    snapshot: SnapshotStore = field(default_factory=SnapshotStore)


@dataclass(frozen=True)
class LocationPageOutput:
    """A page with its extract, both batches and its chunk reading. The page has no text."""

    page: Page
    extract: LocationPageExtract
    batch: GraphBatch
    chunk_batch: GraphBatch
    chunk_reading: LocationChunkReading
    find_batch: GraphBatch
    find_build: FindBuild
    find_reading: FindReading
    encounter_batch: GraphBatch
    encounter_build: EncounterBuild
    encounter_reading: EncounterReading


@dataclass(frozen=True, order=True)
class CoverageRow:
    """The species with a detailed Encounter at 1 Location in 1 Version."""

    location_id: str
    version_id: str
    species_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocationParseRun:
    outputs: tuple[LocationPageOutput, ...]
    report: LocationParseReport
    candidates: tuple[str, ...]
    coverage: tuple[CoverageRow, ...] = ()


@dataclass(frozen=True)
class ParsedPage:
    """A page without its text, what the parser made of it and the chunks it holds."""

    page: Page
    result: LocationPageExtract | SkippedPage
    chunks: LocationChunkReading | None
    finds: FindReading | None
    encounters: EncounterReading | None


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
        return ParsedPage(strip_page(page), result, None, None, None)
    sections = split_sections(mask_comments(page.wikitext))
    chunks = read_location_chunks(sections, build_location_id(result.title))
    finds = read_item_finds(sections, result.title, index)
    encounters = read_encounters(sections, result.title, index)
    return ParsedPage(strip_page(page), result, chunks, finds, encounters)


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
        if (
            isinstance(item.result, SkippedPage)
            or item.chunks is None
            or item.finds is None
            or item.encounters is None
        ):
            reason = item.result.reason.value if isinstance(item.result, SkippedPage) else None
            results.append(build_result(item.page, PageStatus.SKIPPED, reason))
            continue
        try:
            batch = build_location_batch(item.result, inputs)
            chunk_batch = build_location_chunk_batch(item.result, item.chunks, inputs)
            find_build = build_find_batch(item.result, item.finds.facts, inputs)
            encounter_build = build_encounter_batch(item.result, item.encounters.facts, inputs)
        except (BatchBuildError, ChunkBuildError) as error:
            results.append(build_result(item.page, PageStatus.FAILED, str(error)))
            continue
        outputs.append(
            LocationPageOutput(
                item.page,
                item.result,
                batch,
                chunk_batch,
                item.chunks,
                find_build.batch,
                find_build,
                item.finds,
                encounter_build.batch,
                encounter_build,
                item.encounters,
            )
        )
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


class OwnerExtract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    item_id: str
    name: str


class OwnerFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    extract: OwnerExtract


def read_item_owners(output_root: Path) -> dict[str, tuple[str, str]]:
    """Return the Item owners of the item parse outputs: the lowercase name of each item to its
    id and name. A missing output folder gives an empty mapping."""
    folder = output_root / ITEM_PAGE_TYPE
    owners: dict[str, tuple[str, str]] = {}
    for path in sorted(folder.glob(f"*{OUTPUT_SUFFIX}")) if folder.is_dir() else []:
        try:
            owner = OwnerFile.model_validate(json.loads(path.read_text(TEXT_ENCODING)))
        except (json.JSONDecodeError, ValidationError) as error:
            raise ParseRunError(f"cannot read item output {path}: {error}") from error
        owners[owner.extract.name.lower()] = (owner.extract.item_id, owner.extract.name)
    return owners


@dataclass(frozen=True)
class PokeAPILocation:
    """1 PokeAPI location of a core region."""

    identifier: str
    region_id: int


def read_pokeapi_locations(store: SnapshotStore) -> tuple[PokeAPILocation, ...] | None:
    """Return the locations of the core regions of the pinned PokeAPI snapshot, or `None` when
    the snapshot lacks `locations.csv`. PokeAPI is a coverage check only, never a source."""
    try:
        rows = store.read_table(POKEAPI_LOCATIONS_FILE)
    except SnapshotError:
        return None
    return tuple(
        PokeAPILocation(row[POKEAPI_IDENTIFIER_COLUMN], int(row[POKEAPI_REGION_COLUMN]))
        for row in rows
        if row[POKEAPI_REGION_COLUMN] and int(row[POKEAPI_REGION_COLUMN]) in CORE_REGION_IDS
    )


def list_pokeapi_unmatched(
    locations: Iterable[PokeAPILocation], extracts: Iterable[LocationPageExtract]
) -> tuple[str, ...]:
    """Return the identifiers of the PokeAPI locations that no Location owner has an id or an
    alias id for, sorted."""
    owned: set[str] = set()
    for extract in extracts:
        if extract.leaf_class is None:
            continue
        owned.add(build_location_id(extract.title))
        owned.update(build_location_id(alias) for alias in extract.aliases)
    return tuple(sorted({loc.identifier for loc in locations} - owned))


class IdOverlapError(Exception):
    """A Location family id equals a species id: the id prefixes of the encounter claims would
    overlap. The message names both."""

    def __init__(self, location_id: str, location_title: str, species_title: str) -> None:
        super().__init__(
            f"Location {location_title!r} and species {species_title!r} share the id "
            f"{location_id!r}; encounter claims filter on id prefixes"
        )


def check_id_overlap(outputs: Iterable[LocationPageOutput], species_titles: Iterable[str]) -> None:
    """Raise `IdOverlapError` when the id of a Location owner or of an area is the id of a
    pinned species."""
    species = {
        build_species_id(title.removesuffix(SPECIES_TITLE_SUFFIX)): title
        for title in species_titles
    }
    for output in outputs:
        for node in output.batch.owner_nodes:
            if node.id in species:
                raise IdOverlapError(node.id, output.extract.title, species[node.id])


def build_coverage(outputs: Iterable[LocationPageOutput]) -> tuple[CoverageRow, ...]:
    """Return the (location, version) pairs with at least 1 Encounter and the species that have
    one, sorted. An Encounter at an area counts for the Location of its page."""
    pairs: dict[tuple[str, str], set[str]] = {}
    for output in outputs:
        location_id = build_location_id(output.extract.title)
        versions = {
            edge.start.id: edge.end.id
            for edge in output.encounter_batch.edges
            if edge.property_iri == IN_VERSION
        }
        for edge in output.encounter_batch.edges:
            if edge.property_iri == HAS_ENCOUNTER:
                key = (location_id, versions[edge.end.id])
                pairs.setdefault(key, set()).add(edge.start.id)
    return tuple(
        CoverageRow(location_id, version_id, tuple(sorted(species)))
        for (location_id, version_id), species in sorted(pairs.items())
    )


def serialize_coverage(rows: Iterable[CoverageRow]) -> str:
    return dump_json(
        [
            {
                "location_id": r.location_id,
                "version_id": r.version_id,
                "species_ids": list(r.species_ids),
            }
            for r in rows
        ]
    )


def count_finds(facts: Iterable[FindFact]) -> Counter[str]:
    return Counter(fact.method or NO_METHOD for fact in facts)


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
    is_item_source_missing: bool,
    pokeapi_locations: tuple[PokeAPILocation, ...] | None,
    coverage: Sequence[CoverageRow],
) -> LocationParseReport:
    skipped: Counter[str] = Counter(
        result.reason or "" for result in results if result.status is PageStatus.SKIPPED
    )
    classes: Counter[str] = Counter(
        output.extract.leaf_class.value if output.extract.leaf_class else ABOX_REGION_CLASS
        for output in outputs
    )
    dropped: Counter[str] = Counter()
    finds_per_method: Counter[str] = Counter()
    per_kind: Counter[str] = Counter()
    encounter_skips: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for output in outputs:
        encounter_skips.update(output.encounter_reading.skips)
        unmapped.update(output.encounter_reading.unmapped)
        per_kind.update(
            f"{node.properties[ENCOUNTER_KIND]}/{node.properties.get(METHOD, NO_METHOD)}"
            for node in output.encounter_batch.owner_nodes
        )
        dropped.update(output.chunk_reading.dropped_templates)
        finds_per_method.update(
            str(node.properties.get(METHOD, NO_METHOD)) for node in output.find_batch.owner_nodes
        )
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
        find_counts=count_batches(o.find_batch for o in outputs),
        finds_per_method=dict(sorted(finds_per_method.items())),
        hidden_finds=sum(
            node.properties.get(IS_HIDDEN) is True
            for o in outputs
            for node in o.find_batch.owner_nodes
        ),
        find_skips={
            reason.value: sum(o.find_reading.skips[reason.value] for o in outputs)
            for reason in FindSkipReason
        },
        duplicate_finds=sum(o.find_build.duplicates for o in outputs),
        item_stub_names=tuple(sorted({n for o in outputs for n in o.find_build.item_stub_names})),
        is_item_source_missing=is_item_source_missing,
        pokeapi_unmatched=list_pokeapi_unmatched(pokeapi_locations or (), extracts),
        is_pokeapi_source_missing=pokeapi_locations is None,
        encounter_counts=count_batches(o.encounter_batch for o in outputs),
        encounters_per_kind=dict(sorted(per_kind.items())),
        encounter_skips={
            reason.value: encounter_skips[reason.value] for reason in EncounterSkipReason
        },
        unmapped_encounter_values=dict(
            sorted(unmapped.items(), key=lambda pair: (-pair[1], pair[0]))[:UNMAPPED_REPORT_LIMIT]
        ),
        merged_encounters=sum(o.encounter_build.merges for o in outputs),
        duplicate_encounters=sum(o.encounter_build.duplicates for o in outputs),
        unresolved_species=tuple(
            sorted({s for o in outputs for s in o.encounter_build.unresolved_species})
        ),
        coverage_pairs=len(coverage),
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
    version_labels = [label for labels in group_versions.values() for label in labels]
    check_version_codes(version_labels)
    check_catch_templates(version_labels)
    foreign = [pin.title for name in FOREIGN_PAGE_TYPES for pin in manifest.list_pins(name)]
    pages = [inputs.store.read_page(PAGE_TYPE, pin) for pin in pins]
    index = build_location_index(pages, foreign, LOCATION_INFOBOXES)
    parsed = parse_pages(pages, index)
    extracts = list_extracts(parsed)
    resolver = build_species_resolver(manifest.list_pins(SPECIES_PAGE_TYPE))
    item_owners = read_item_owners(inputs.output_root)
    build_inputs = LocationBuildInputs(
        abox, index, map_page_classes(extracts), group_versions, resolver, item_owners
    )
    outputs, results = build_outputs(parsed, requested, build_inputs)
    candidates = build_candidate_titles(index, extracts)
    unresolved = sum(
        len(list_unresolved_mentions(output.chunk_reading, build_inputs)) for output in outputs
    )
    check_id_overlap(outputs, (pin.title for pin in manifest.list_pins(SPECIES_PAGE_TYPE)))
    coverage = build_coverage(outputs)
    report = build_report(
        outputs,
        results,
        extracts,
        candidates,
        unresolved,
        not item_owners,
        read_pokeapi_locations(inputs.snapshot),
        coverage,
    )
    return LocationParseRun(outputs, report, candidates, coverage)


def serialize_output(output: LocationPageOutput) -> str:
    return dump_json(
        {
            "title": output.page.title,
            "page_id": output.page.page_id,
            "revision_id": output.page.revision_id,
            "extract": output.extract.model_dump(mode="json"),
            "batch": serialize_batch(output.batch),
            "chunk_batch": serialize_batch(output.chunk_batch),
            "find_batch": serialize_batch(output.find_batch),
            "encounter_batch": serialize_batch(output.encounter_batch),
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
    write_text_atomically(root / COVERAGE_FILE_NAME, serialize_coverage(run.coverage))
