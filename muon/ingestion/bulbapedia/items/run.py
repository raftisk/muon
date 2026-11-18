"""A parse run over the pinned item pages: extracts, chunks, pass 1 and 2 batches and the
report.

`parse_item_pages` reads the page cache, the ontology and, for the stub list, the species,
move and ability parse outputs, never the network. Each page is read, parsed, chunked and
built in turn, so the run holds 1 page text at a time. A page that breaks a rule fails alone
and the run goes on; a request or game code error stops the run before anything is written.
Output files are byte-identical across runs.

The unowned stub list reads the species, move and ability outputs and changes no batch. A
missing output folder is named in the report and the run goes on.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from muon.files import write_text_atomically
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    BERRY,
    HAS_FLAVOR,
    HELD_ITEM,
    ITEM,
    MENTIONED_IN,
    UNIVERSE,
    AboxIndex,
    BatchBuildError,
    ChunkBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.items.batches import (
    ItemChunkInputs,
    build_item_batch,
    build_item_chunk_batch,
)
from muon.ingestion.bulbapedia.items.chunks import ItemChunkReading, read_item_chunks
from muon.ingestion.bulbapedia.items.page import (
    ItemNoteKind,
    ItemPageExtract,
    ItemPageParseError,
    SkippedPage,
    SkipReason,
    parse_item_page,
)
from muon.ingestion.bulbapedia.moves.batches import check_game_codes
from muon.ingestion.bulbapedia.pages import Page, PagePin, PageStore
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
from muon.ingestion.seed import read_abox
from muon.ontology.model import extract_local_name, load_ontology_model

PAGE_TYPE = "item"
PAGE_OUTPUT_DIR = "item"
REPORT_FILE_NAME = "item-report.json"
STUB_SOURCE_PAGE_TYPES = ("species", "move", "ability")
ITEM_FAMILY = frozenset({ITEM, HELD_ITEM, BERRY})
TEXT_ENCODING = "utf-8"
NO_CATEGORY = "none"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not a pinned item page."""


class ReportNote(ReportModel):
    page_title: str
    kind: ItemNoteKind
    field: str
    detail: str


class UnknownCodeEntry(ReportModel):
    page_title: str
    section_path: str
    cell: str


class ItemParseReport(ReportModel):
    pages: tuple[PageResult, ...]
    pages_parsed: int
    pages_skipped: int
    pages_failed: int
    skipped_per_reason: Mapping[str, int]
    items_per_class: Mapping[str, int]
    items_per_category: Mapping[str, int]
    notes: tuple[ReportNote, ...]
    absent_fling_power: int
    flavor_edges: int
    batch_counts: BatchCounts
    chunk_counts: BatchCounts
    chunks_per_aspect: Mapping[str, int]
    mentions: int
    dropped_templates: tuple[TemplateCount, ...]
    removed_tables_total: int
    math_formulas_total: int
    unknown_codes: tuple[UnknownCodeEntry, ...]
    missing_stub_sources: tuple[str, ...]
    unowned_stub_ids: tuple[str, ...]


@dataclass(frozen=True)
class ItemParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ItemRunInputs:
    request: ItemParseRequest
    store: PageStore
    ontology_dir: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT


@dataclass(frozen=True)
class ItemPageOutput:
    """A page with its extract (sections cleared) and both batches. The page has no text."""

    page: Page
    extract: ItemPageExtract
    batch: GraphBatch
    chunk_batch: GraphBatch
    chunk_reading: ItemChunkReading


@dataclass(frozen=True)
class ItemParseRun:
    outputs: tuple[ItemPageOutput, ...]
    report: ItemParseReport


@dataclass(frozen=True)
class BuildContext:
    """What every page of the run resolves against."""

    abox: AboxIndex
    chunk_inputs: ItemChunkInputs


class StubOutputModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class StubNode(StubOutputModel):
    class_iri: str
    id: str


class StubBatch(StubOutputModel):
    nodes: tuple[StubNode, ...] = ()
    scope: tuple[str, ...] = ()


def check_request(request: ItemParseRequest, pins: Iterable[PagePin]) -> None:
    unknown = find_unpinned_titles(request.titles, (pin.title for pin in pins))
    if unknown:
        raise ParseRunError(f"Titles that are not pinned item pages: {unknown}")


def read_item_stub_ids(output_root: Path) -> tuple[Mapping[str, str], tuple[str, ...]]:
    """Return the Item family stubs of the species, move and ability outputs, and the output
    folders that are missing.

    The stubs map an id to its class local name. A node is a stub when its class is in the
    Item family and not in the scope of its batch. Output files need only the batches they
    hold; other keys are ignored.
    """
    stubs: dict[str, str] = {}
    missing: list[str] = []
    for page_type in STUB_SOURCE_PAGE_TYPES:
        folder = output_root / page_type
        paths = sorted(folder.glob(f"*{OUTPUT_SUFFIX}")) if folder.is_dir() else []
        if not paths:
            missing.append(page_type)
        for path in paths:
            stubs.update(read_file_stubs(path))
    return dict(sorted(stubs.items())), tuple(missing)


def read_file_stubs(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(TEXT_ENCODING))
        batches = [
            StubBatch.model_validate(value)
            for value in data.values()
            if isinstance(value, dict) and "nodes" in value
        ]
    except (json.JSONDecodeError, ValidationError) as error:
        raise ParseRunError(f"cannot read output {path}: {error}") from error
    return {
        node.id: extract_local_name(node.class_iri)
        for batch in batches
        for node in batch.nodes
        if node.class_iri in ITEM_FAMILY and node.class_iri not in batch.scope
    }


def build_page_output(
    page: Page, extract: ItemPageExtract, context: BuildContext
) -> ItemPageOutput:
    """Build both passes of 1 page. Raises `BatchBuildError` or `ChunkBuildError`."""
    reading = read_item_chunks(extract.sections, extract.item_id)
    batch = build_item_batch(extract, context.abox)
    chunk_batch = build_item_chunk_batch(extract, reading, context.chunk_inputs)
    return ItemPageOutput(
        page=page.model_copy(update={"wikitext": ""}),
        extract=extract.model_copy(update={"sections": ()}),
        batch=batch,
        chunk_batch=chunk_batch,
        chunk_reading=reading,
    )


def parse_pins(
    pins: Iterable[PagePin], store: PageStore, context: BuildContext
) -> tuple[tuple[ItemPageOutput, ...], tuple[PageResult, ...]]:
    """Parse every pin in turn. A skipped or failed page gets a result and no output."""
    outputs: list[ItemPageOutput] = []
    results: list[PageResult] = []
    for pin in pins:
        page = store.read_page(PAGE_TYPE, pin)
        try:
            parsed = parse_item_page(page)
            if isinstance(parsed, SkippedPage):
                results.append(build_result(page, PageStatus.SKIPPED, parsed.reason.value))
                continue
            outputs.append(build_page_output(page, parsed, context))
        except (ItemPageParseError, BatchBuildError, ChunkBuildError) as error:
            results.append(build_result(page, PageStatus.FAILED, str(error)))
            continue
        results.append(build_result(page, PageStatus.PARSED))
    return tuple(outputs), tuple(results)


def count_edges(batches: Iterable[GraphBatch], property_iri: str) -> int:
    return sum(edge.property_iri == property_iri for batch in batches for edge in batch.edges)


def list_unknown_codes(outputs: Iterable[ItemPageOutput]) -> tuple[UnknownCodeEntry, ...]:
    return tuple(
        UnknownCodeEntry(
            page_title=output.page.title, section_path=code.section_path, cell=code.cell
        )
        for output in outputs
        for code in output.chunk_reading.unknown_codes
    )


def build_report(
    outputs: tuple[ItemPageOutput, ...],
    results: tuple[PageResult, ...],
    output_root: Path,
) -> ItemParseReport:
    """Collect the outcomes, counts and the unowned stub list of the requested pages."""
    stubs, missing = read_item_stub_ids(output_root)
    owned = {output.extract.item_id for output in outputs}
    aspects: Counter[str] = Counter(
        str(node.properties[ASPECT]) for o in outputs for node in o.chunk_batch.owner_nodes
    )
    skipped: Counter[str] = Counter(
        r.reason or "" for r in results if r.status is PageStatus.SKIPPED
    )
    dropped: Counter[str] = Counter()
    for output in outputs:
        dropped.update(output.chunk_reading.dropped_templates)
    return ItemParseReport(
        pages=results,
        pages_parsed=sum(r.status is PageStatus.PARSED for r in results),
        pages_skipped=sum(r.status is PageStatus.SKIPPED for r in results),
        pages_failed=sum(r.status is PageStatus.FAILED for r in results),
        skipped_per_reason={reason.value: skipped[reason.value] for reason in SkipReason},
        items_per_class=dict(sorted(Counter(o.extract.item_class.value for o in outputs).items())),
        items_per_category=dict(
            sorted(Counter(o.extract.category or NO_CATEGORY for o in outputs).items())
        ),
        notes=tuple(
            ReportNote(page_title=o.extract.title, kind=n.kind, field=n.field, detail=n.detail)
            for o in outputs
            for n in o.extract.notes
        ),
        absent_fling_power=sum(o.extract.fling_power is None for o in outputs),
        flavor_edges=count_edges((o.batch for o in outputs), HAS_FLAVOR),
        batch_counts=count_batches(o.batch for o in outputs),
        chunk_counts=count_batches(o.chunk_batch for o in outputs),
        chunks_per_aspect=dict(sorted(aspects.items())),
        mentions=count_edges((o.chunk_batch for o in outputs), MENTIONED_IN),
        dropped_templates=rank_templates(dropped),
        removed_tables_total=sum(o.chunk_reading.removed_tables for o in outputs),
        math_formulas_total=sum(o.chunk_reading.math_formulas for o in outputs),
        unknown_codes=list_unknown_codes(outputs),
        missing_stub_sources=missing,
        unowned_stub_ids=tuple(sorted(set(stubs) - owned)),
    )


def parse_item_pages(inputs: ItemRunInputs) -> ItemParseRun:
    """Parse the pinned item pages and build the pass 1 and pass 2 batches of each.

    Raises `ParseRunError` or `GameCodeError` before anything is written. The caller writes
    the outputs with `write_outputs`. The unowned stub list covers the items of the requested
    pages, so a `--title` run lists the stubs of every other item as unowned.
    """
    manifest = inputs.store.read_manifest()
    pins = manifest.list_pins(PAGE_TYPE)
    check_request(inputs.request, pins)
    requested = set(inputs.request.titles) or {pin.title for pin in pins}
    model = load_ontology_model(inputs.ontology_dir, UNIVERSE)
    seed = read_abox(model, inputs.ontology_dir)
    check_game_codes(seed)
    abox = build_abox_index(seed)
    resolver = build_species_resolver(manifest.list_pins(SPECIES_PAGE_TYPE))
    context = BuildContext(abox=abox, chunk_inputs=ItemChunkInputs(abox=abox, resolver=resolver))
    outputs, results = parse_pins(
        (pin for pin in pins if pin.title in requested), inputs.store, context
    )
    return ItemParseRun(outputs, build_report(outputs, results, inputs.output_root))


def serialize_output(output: ItemPageOutput) -> str:
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


def write_outputs(run: ItemParseRun, root: Path, is_full_run: bool) -> None:
    """Write the page outputs and the report under `root`, atomically. A full run also
    deletes the output files of every page it produced no output for."""
    files = (
        (f"{output.page.page_id}{OUTPUT_SUFFIX}", serialize_output(output))
        for output in run.outputs
    )
    write_page_files(root / PAGE_OUTPUT_DIR, files, is_full_run)
    write_text_atomically(root / REPORT_FILE_NAME, dump_json(run.report.model_dump(mode="json")))
