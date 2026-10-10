"""A parse run over the pinned species pages: extracts, batches, residue, report and files.

`parse_species_pages` reads the page cache only. It parses every pinned page, because the
species index that resolves evolution references needs all of them, and keeps the outputs
and report entries of the requested titles. A page that breaks a rule fails alone and the
run goes on; an override that matches nothing and an unknown `--title` stop the run before
anything is written. Output files are byte-identical across runs: sets are sorted, JSON
keys are sorted and the report holds no time.
"""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.bulbapedia.pages import (
    DEFAULT_ROOT,
    Page,
    PagePin,
    PageStore,
    write_text_atomically,
)
from muon.ingestion.bulbapedia.species.batches import (
    CONDITION,
    POKEMON_SPECIES,
    UNIVERSE,
    AboxIndex,
    BatchBuildError,
    EvolutionBuild,
    SpeciesIndex,
    build_abox_index,
    build_change_id,
    build_evolution_batch,
    build_species_batch,
    build_species_index,
    resolve_species,
)
from muon.ingestion.bulbapedia.species.evolution import (
    MEGA_KIND,
    pick_mega_stone,
    read_mega_stones,
)
from muon.ingestion.bulbapedia.species.extract import (
    FormCandidateResult,
    NoteKind,
    PageNote,
    PageParseError,
    ResidueKind,
    SpeciesPageExtract,
)
from muon.ingestion.bulbapedia.species.forms import check_override_titles, rank_mega_forms
from muon.ingestion.bulbapedia.species.page import parse_species_page, read_infobox
from muon.ingestion.bulbapedia.wikitext import split_sections
from muon.ingestion.seed import read_abox
from muon.ontology.model import extract_local_name, load_ontology_model

PAGE_TYPE = "species"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "parsed"
PAGE_OUTPUT_DIR = "species"
REPORT_FILE_NAME = "species-report.json"
OUTPUT_SUFFIX = ".json"
JSON_INDENT = 2
SHARE_DIGITS = 4
NO_INFOBOX_REASON = "no Pokémon Infobox"
RESIDUE_ID_TEMPLATE = "{page_id}:{kind}:{number}"


class ParseRunError(Exception):
    """A request the run cannot start: a title that is not pinned."""


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


class ResidueItem(ReportModel):
    """A residue item with the identity feature 7 reads it by."""

    id: str
    page_title: str
    revision_id: int
    kind: ResidueKind
    section_path: str
    form_ids: tuple[str, ...]
    wikitext: str
    text: str


class ReportNote(ReportModel):
    page_title: str
    kind: NoteKind
    section_path: str
    detail: str


class FormCandidateEntry(ReportModel):
    page_title: str
    candidate: FormCandidateResult


class UnownedPair(ReportModel):
    """An evolution a page shows for a species of another page, which that page did not emit."""

    page_title: str
    evolution_id: str


class BatchCounts(ReportModel):
    """Counts by class local name (owner nodes and stubs) and property local name (edges)."""

    owner_nodes: Mapping[str, int]
    stubs: Mapping[str, int]
    edges: Mapping[str, int]


class ParseReport(ReportModel):
    pages: tuple[PageResult, ...]
    pages_parsed: int
    pages_skipped: int
    pages_failed: int
    species_counts: BatchCounts
    evolution_counts: BatchCounts
    form_candidates: tuple[FormCandidateEntry, ...]
    overrides_applied: tuple[ReportNote, ...]
    notes: tuple[ReportNote, ...]
    unowned_evolution_pairs: tuple[UnownedPair, ...]
    pages_without_learnset: tuple[str, ...]
    evolutions_total: int
    evolutions_structured: int
    evolution_rule_share: float
    mega_forms_with_stone: int
    mega_form_changes: int
    abox_stub_misses: int
    evolution_species_stub_misses: int
    residue: tuple[ResidueItem, ...]


@dataclass(frozen=True)
class ParseRequest:
    titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageOutput:
    page: Page
    extract: SpeciesPageExtract
    species_batch: GraphBatch
    evolution_batch: GraphBatch | None
    notes: tuple[PageNote, ...]
    residue: tuple[ResidueItem, ...]


@dataclass(frozen=True)
class ParseRun:
    outputs: tuple[PageOutput, ...]
    report: ParseReport


@dataclass(frozen=True)
class RunInputs:
    request: ParseRequest
    store: PageStore
    ontology_dir: Path


@dataclass(frozen=True)
class ParsedPages:
    """The pages of a run by title: extracts, and the result of every page that has none."""

    extracts: Mapping[str, SpeciesPageExtract]
    results: Mapping[str, PageResult]


def build_residue_items(extract: SpeciesPageExtract) -> tuple[ResidueItem, ...]:
    """Number the residue of a page per kind in page order: `{page id}:{kind}:{n}`."""
    counts: Counter[ResidueKind] = Counter()
    items: list[ResidueItem] = []
    for residue in extract.residue:
        counts[residue.kind] += 1
        items.append(
            ResidueItem(
                id=RESIDUE_ID_TEMPLATE.format(
                    page_id=extract.page_id, kind=residue.kind.value, number=counts[residue.kind]
                ),
                page_title=extract.title,
                revision_id=extract.revision_id,
                kind=residue.kind,
                section_path=residue.section_path,
                form_ids=residue.form_ids,
                wikitext=residue.wikitext,
                text=residue.text,
            )
        )
    return tuple(items)


def build_result(page: Page, status: PageStatus, reason: str | None = None) -> PageResult:
    return PageResult(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        status=status,
        reason=reason,
    )


def check_request(request: ParseRequest, pins: Iterable[PagePin]) -> None:
    """Stop a run before it writes anything: unknown titles and stale overrides."""
    pinned = {pin.title for pin in pins}
    unknown = sorted(set(request.titles) - pinned)
    if unknown:
        raise ParseRunError(f"Titles that are not pinned: {unknown}")
    check_override_titles(pinned)


def parse_pages(pages: Iterable[Page]) -> ParsedPages:
    """Parse every page. A page without an infobox is skipped and a page that breaks a rule
    fails; an `OverrideError` is not caught, since it stops the run."""
    extracts: dict[str, SpeciesPageExtract] = {}
    results: dict[str, PageResult] = {}
    for page in pages:
        try:
            extract = parse_species_page(page)
        except PageParseError as error:
            results[page.title] = build_result(page, PageStatus.FAILED, str(error))
            continue
        if extract is None:
            results[page.title] = build_result(page, PageStatus.SKIPPED, NO_INFOBOX_REASON)
        else:
            extracts[page.title] = extract
    return ParsedPages(extracts, results)


def build_species_batches(
    extracts: Mapping[str, SpeciesPageExtract], abox: AboxIndex
) -> tuple[dict[str, GraphBatch], dict[str, str]]:
    """Build pass 1 for every page. Returns the batches and the reason of each failed page."""
    batches: dict[str, GraphBatch] = {}
    failures: dict[str, str] = {}
    for title, extract in extracts.items():
        try:
            batches[title] = build_species_batch(extract, abox)
        except BatchBuildError as error:
            failures[title] = str(error)
    return batches, failures


def find_unowned_pairs(
    extracts: Mapping[str, SpeciesPageExtract],
    builds: Mapping[str, EvolutionBuild],
    index: SpeciesIndex,
) -> tuple[UnownedPair, ...]:
    """Find the evolutions a page shows for another page's species that no page emitted."""
    emitted = {
        node.id
        for build in builds.values()
        if build.batch is not None
        for node in build.batch.owner_nodes
    }
    unowned: set[tuple[str, str]] = set()
    for title, extract in extracts.items():
        own_ids = {form.id for form in extract.forms}
        for fact in extract.evolutions:
            source = resolve_species(index, fact.source)
            target = resolve_species(index, fact.target)
            if source is None or target is None or source[0] in own_ids:
                continue
            evolution_id = build_change_id(source[0], target[0])
            if evolution_id not in emitted:
                unowned.add((title, evolution_id))
    return tuple(UnownedPair(page_title=t, evolution_id=i) for t, i in sorted(unowned))


def count_batch(batches: Iterable[GraphBatch]) -> BatchCounts:
    owners: Counter[str] = Counter()
    stubs: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    for batch in batches:
        owners.update(extract_local_name(node.class_iri) for node in batch.owner_nodes)
        stubs.update(extract_local_name(node.class_iri) for node in batch.stub_nodes)
        edges.update(extract_local_name(edge.property_iri) for edge in batch.edges)
    return BatchCounts(owner_nodes=dict(owners), stubs=dict(stubs), edges=dict(edges))


def count_mega_forms_with_stone(page: Page, extract: SpeciesPageExtract) -> int:
    """Count the Mega form nodes of a page that the infobox gives a stone."""
    infobox = read_infobox(split_sections(page.wikitext)[0])
    if infobox is None:
        return 0
    stones = read_mega_stones(infobox)
    ranks = rank_mega_forms(extract.candidates)
    node_indices = {form.form_index for form in extract.forms if not form.is_default}
    return sum(
        pick_mega_stone(stones, rank) is not None
        for index, rank in ranks.items()
        if index in node_indices
    )


@dataclass(frozen=True)
class ReportContext:
    """What the report needs from the whole run, not only from the requested pages."""

    seed_refs: frozenset[NodeRef]
    owner_refs: frozenset[NodeRef]
    unowned: tuple[UnownedPair, ...]


def count_stub_misses(outputs: Iterable[PageOutput], context: ReportContext) -> tuple[int, int]:
    """Count the A-Box stubs that match no seed node and the species stubs of evolution
    batches that match no species owner of the run."""
    abox_classes = {ref.class_iri for ref in context.seed_refs}
    abox_misses = 0
    species_misses = 0
    for output in outputs:
        abox_misses += sum(
            stub.class_iri in abox_classes and stub.node_ref not in context.seed_refs
            for stub in output.species_batch.stub_nodes
        )
        if output.evolution_batch is not None:
            species_misses += sum(
                stub.node_ref not in context.owner_refs
                for stub in output.evolution_batch.stub_nodes
                if stub.class_iri == POKEMON_SPECIES
            )
    return abox_misses, species_misses


def to_report_notes(title: str, notes: Iterable[PageNote]) -> list[ReportNote]:
    return [
        ReportNote(
            page_title=title, kind=note.kind, section_path=note.section_path, detail=note.detail
        )
        for note in notes
    ]


def build_report(
    outputs: tuple[PageOutput, ...], results: tuple[PageResult, ...], context: ReportContext
) -> ParseReport:
    """Collect the outcomes, notes, residue and counts of the requested pages."""
    notes = [n for o in outputs for n in to_report_notes(o.extract.title, o.notes)]
    abox_misses, species_misses = count_stub_misses(outputs, context)
    evolution_batches = [o.evolution_batch for o in outputs if o.evolution_batch is not None]
    evolutions = [node for batch in evolution_batches for node in batch.owner_nodes]
    structured = sum(CONDITION not in node.properties for node in evolutions)
    return ParseReport(
        pages=results,
        pages_parsed=sum(r.status is PageStatus.PARSED for r in results),
        pages_skipped=sum(r.status is PageStatus.SKIPPED for r in results),
        pages_failed=sum(r.status is PageStatus.FAILED for r in results),
        species_counts=count_batch(o.species_batch for o in outputs),
        evolution_counts=count_batch(evolution_batches),
        form_candidates=tuple(
            FormCandidateEntry(page_title=o.extract.title, candidate=candidate)
            for o in outputs
            for candidate in o.extract.candidates
        ),
        overrides_applied=tuple(n for n in notes if n.kind is NoteKind.OVERRIDE_APPLIED),
        notes=tuple(n for n in notes if n.kind is not NoteKind.OVERRIDE_APPLIED),
        unowned_evolution_pairs=context.unowned,
        pages_without_learnset=tuple(
            o.extract.title
            for o in outputs
            if any(n.kind is NoteKind.MISSING_LEARNSET for n in o.notes)
        ),
        evolutions_total=len(evolutions),
        evolutions_structured=structured,
        evolution_rule_share=(
            round(structured / len(evolutions), SHARE_DIGITS) if evolutions else 0.0
        ),
        mega_forms_with_stone=sum(count_mega_forms_with_stone(o.page, o.extract) for o in outputs),
        mega_form_changes=sum(
            change.form_kind == MEGA_KIND for o in outputs for change in o.extract.form_changes
        ),
        abox_stub_misses=abox_misses,
        evolution_species_stub_misses=species_misses,
        residue=tuple(item for o in outputs for item in o.residue),
    )


def build_output(
    page: Page, extract: SpeciesPageExtract, batches: tuple[GraphBatch, EvolutionBuild]
) -> PageOutput:
    species_batch, build = batches
    return PageOutput(
        page=page,
        extract=extract,
        species_batch=species_batch,
        evolution_batch=build.batch,
        notes=(*extract.notes, *build.notes),
        residue=build_residue_items(extract),
    )


def parse_species_pages(inputs: RunInputs) -> ParseRun:
    """Parse the pinned species pages and build both batches of each.

    Raises `ParseRunError` for a title that is not pinned and `OverrideError` for an
    override that matches nothing, both before any page is read. The caller writes the
    outputs with `write_outputs`.
    """
    pins = inputs.store.read_manifest().list_pins(PAGE_TYPE)
    check_request(inputs.request, pins)
    requested = set(inputs.request.titles) or {pin.title for pin in pins}
    pages = {pin.title: inputs.store.read_page(PAGE_TYPE, pin) for pin in pins}
    parsed = parse_pages(pages.values())
    model = load_ontology_model(inputs.ontology_dir, UNIVERSE)
    seed = read_abox(model, inputs.ontology_dir)
    abox = build_abox_index(seed)
    species_batches, failures = build_species_batches(parsed.extracts, abox)
    extracts = {title: parsed.extracts[title] for title in species_batches}
    index = build_species_index(extracts.values())
    builds = {
        title: build_evolution_batch(extract, abox, index) for title, extract in extracts.items()
    }
    results = {**parsed.results}
    for title, page in pages.items():
        if title in failures:
            results[title] = build_result(page, PageStatus.FAILED, failures[title])
        elif title in extracts:
            results[title] = build_result(page, PageStatus.PARSED)
    context = ReportContext(
        seed_refs=frozenset(node.node_ref for node in seed.nodes),
        owner_refs=frozenset(
            node.node_ref
            for batch in species_batches.values()
            for node in batch.owner_nodes
            if node.class_iri == POKEMON_SPECIES
        ),
        unowned=tuple(
            pair
            for pair in find_unowned_pairs(extracts, builds, index)
            if pair.page_title in requested
        ),
    )
    outputs = tuple(
        build_output(pages[title], extracts[title], (species_batches[title], builds[title]))
        for title in pages
        if title in requested and title in extracts
    )
    ordered_results = tuple(results[title] for title in pages if title in requested)
    return ParseRun(outputs, build_report(outputs, ordered_results, context))


def dump_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=JSON_INDENT) + "\n"


def serialize_batch(batch: GraphBatch | None) -> dict[str, object] | None:
    """Dump a batch with its scope sorted: pydantic dumps a frozenset in hash order."""
    if batch is None:
        return None
    data = batch.model_dump(mode="json")
    data["scope"] = sorted(batch.scope)
    return data


def serialize_output(output: PageOutput) -> str:
    return dump_json(
        {
            "title": output.page.title,
            "page_id": output.page.page_id,
            "revision_id": output.page.revision_id,
            "extract": output.extract.model_dump(mode="json"),
            "species_batch": serialize_batch(output.species_batch),
            "evolution_batch": serialize_batch(output.evolution_batch),
            "residue": [item.model_dump(mode="json") for item in output.residue],
        }
    )


def serialize_report(report: ParseReport) -> str:
    return dump_json(report.model_dump(mode="json"))


def write_outputs(run: ParseRun, root: Path, is_full_run: bool) -> None:
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
    write_text_atomically(root / REPORT_FILE_NAME, serialize_report(run.report))
    if is_full_run:
        for path in output_dir.glob(f"*{OUTPUT_SUFFIX}"):
            if path.name not in kept:
                path.unlink()
