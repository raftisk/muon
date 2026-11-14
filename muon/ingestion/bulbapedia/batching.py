"""pkmn batch building shared by the Bulbapedia page types.

Every page type resolves the names the hand-written A-Box owns through `AboxIndex`, builds
stubs and edges the same way and claims its edge patterns with `ClaimRule`s. A page type
owns its scope, its claim tables and its own records; this module owns what they share,
including the machine id, which species stubs and move owners must agree on.

Text chunks are built here too: their id, their record and their `describes`,
`mentionedIn` and `fromWork` edges follow 1 rule for every page type. A page type passes
in how a `{{p}}` mention finds its species and turns the misses into its own notes.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from muon.ingestion.batch import (
    ClaimDirection,
    EdgeClaim,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
    PropertyValue,
    build_node_id,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ontology.model import extract_local_name

UNIVERSE = "pkmn"
PKMN = "https://muon.dev/ns/pkmn#"
CORE = "https://muon.dev/ns/core#"

TYPE = PKMN + "Type"
GENERATION = PKMN + "Generation"
VERSION_GROUP = PKMN + "VersionGroup"
MOVE = PKMN + "Move"
MACHINE = PKMN + "Machine"
ABILITY = PKMN + "Ability"
REGION = CORE + "Region"
LOCATION = CORE + "Location"
TEXT_CHUNK = CORE + "TextChunk"

HAS_TYPE = PKMN + "hasType"
INTRODUCED_IN = PKMN + "introducedIn"
TEACHES_MOVE = PKMN + "teachesMove"
IN_VERSION_GROUP = PKMN + "inVersionGroup"
FOUND_IN = CORE + "foundIn"
ALIASES = CORE + "aliases"
DESCRIBES = CORE + "describes"
MENTIONED_IN = CORE + "mentionedIn"
FROM_WORK = CORE + "fromWork"
TEXT = CORE + "text"
ASPECT = CORE + "aspect"
SOURCE = CORE + "source"
SECTION = CORE + "section"
SOURCE_URL = CORE + "sourceUrl"

GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERALS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))

CHUNK_SOURCE = "bulbapedia"
CHUNK_ID_TEMPLATE = "{source}:{label}:{subject}:{aspect}:{n}"
CHUNK_NAME_TEMPLATE = "{subject} {aspect} {n}"
TEXT_HASH_LENGTH = 8
ENCOUNTER_ID_SEPARATOR = ":"
# Texts shown in a game: their id ends with a text hash and they name their origin through
# `fromWork`, so they carry no `source` (shape X1).
WORK_ASPECTS = frozenset({ChunkAspect.POKEDEX, ChunkAspect.FLAVOR})
SPECIES_MENTION = "p"
MENTION_CLASSES: Mapping[str, str] = {"m": MOVE, "a": ABILITY, "type": TYPE, "t": TYPE}
MENTION_NOUNS: Mapping[str, str] = {
    SPECIES_MENTION: "species",
    "m": "move",
    "a": "ability",
    "type": "A-Box Type",
    "t": "A-Box Type",
}

SpeciesResolver = Callable[[str], NodeRecord | None]


@dataclass(frozen=True)
class ClaimRule:
    """An edge pattern that a batch claims at every anchor of a class."""

    property_iri: str
    direction: ClaimDirection
    other_class_iri: str | None
    owns_other_end: bool


OUT = ClaimDirection.OUT
IN = ClaimDirection.IN


class BatchBuildError(Exception):
    """A page names something the A-Box lacks. The message names all 4 fields."""

    def __init__(self, page_title: str, section_path: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: no A-Box node for {field} in {section_path}: {value!r}")
        self.page_title = page_title
        self.section_path = section_path
        self.field = field
        self.value = value


@dataclass(frozen=True)
class AboxIndex:
    """The ids of the hand-written A-Box nodes by class and label, and the generation of
    each region."""

    ids: Mapping[tuple[str, str], str]
    region_generations: Mapping[str, str]


class AboxContext(Protocol):
    """What `require_abox` needs to name a failed page: its title and the A-Box index."""

    @property
    def page_title(self) -> str: ...

    @property
    def abox(self) -> AboxIndex: ...


@dataclass(frozen=True)
class Parts:
    nodes: tuple[NodeRecord, ...] = ()
    edges: tuple[EdgeRecord, ...] = ()


class AboxMissError(Exception):
    def __init__(self, class_iri: str, label: str) -> None:
        super().__init__(f"No A-Box node of class {class_iri} labelled {label!r}")
        self.class_iri = class_iri
        self.label = label


def build_abox_index(abox: GraphBatch) -> AboxIndex:
    """Index the nodes of `read_abox()` by (class, label) and map each region to the label
    of the generation its `introducedIn` edge names."""
    ids = {(node.class_iri, node.name): node.id for node in abox.nodes}
    names = {node.node_ref: node.name for node in abox.nodes}
    region_generations = {
        names[edge.start]: names[edge.end]
        for edge in abox.edges
        if edge.property_iri == INTRODUCED_IN
        and edge.start.class_iri == REGION
        and edge.end.class_iri == GENERATION
    }
    return AboxIndex(ids=ids, region_generations=region_generations)


def resolve_abox(index: AboxIndex, class_iri: str, label: str) -> NodeRecord:
    """Return the stub record of an A-Box node. It carries the A-Box id and name."""
    node_id = index.ids.get((class_iri, label))
    if node_id is None:
        raise AboxMissError(class_iri, label)
    return NodeRecord(class_iri=class_iri, id=node_id, name=label)


def require_abox(
    context: AboxContext, class_iri: str, label: str, location: tuple[str, str]
) -> NodeRecord:
    """Resolve an A-Box node for a page; a miss fails the page. `location` is the section
    path and the field that named it."""
    try:
        return resolve_abox(context.abox, class_iri, label)
    except AboxMissError as error:
        section_path, field_name = location
        raise BatchBuildError(context.page_title, section_path, field_name, label) from error


def to_roman(number: int) -> str:
    numeral = ""
    for value, symbol in ROMAN_NUMERALS:
        count, number = divmod(number, value)
        numeral += symbol * count
    return numeral


def build_stub(class_iri: str, name: str) -> NodeRecord:
    """Return a stub for a node another page owns: id from the name, name as shown."""
    return NodeRecord(class_iri=class_iri, id=build_node_id(name), name=name)


def build_edge(
    property_iri: str,
    start: NodeRecord,
    end: NodeRecord,
    properties: Mapping[str, PropertyValue] | None = None,
) -> EdgeRecord:
    return EdgeRecord(
        property_iri=property_iri,
        start=start.node_ref,
        end=end.node_ref,
        properties=properties or {},
    )


def drop_none(values: Mapping[str, PropertyValue | None]) -> dict[str, PropertyValue]:
    return {key: value for key, value in values.items() if value is not None}


def join_parts(parts: Iterable[Parts]) -> Parts:
    collected = list(parts)
    return Parts(
        nodes=tuple(node for part in collected for node in part.nodes),
        edges=tuple(edge for part in collected for edge in part.edges),
    )


def build_claims(anchors: Iterable[NodeRef], rules: Sequence[ClaimRule]) -> tuple[EdgeClaim, ...]:
    """Return every rule as a claim at every anchor, anchor by anchor."""
    return tuple(
        EdgeClaim(
            anchor=anchor,
            property_iri=rule.property_iri,
            direction=rule.direction,
            other_class_iri=rule.other_class_iri,
            owns_other_end=rule.owns_other_end,
        )
        for anchor in anchors
        for rule in rules
    )


def build_owner_claims(
    nodes: Iterable[NodeRecord], rules_by_class: Mapping[str, tuple[ClaimRule, ...]]
) -> tuple[EdgeClaim, ...]:
    """Claim, at each owner record, the rules of its class. A class with no rules claims nothing."""
    return tuple(
        claim
        for node in nodes
        for claim in build_claims([node.node_ref], rules_by_class.get(node.class_iri, ()))
    )


def list_owner_records(parts: Parts, scope: frozenset[str]) -> list[NodeRecord]:
    return [node for node in parts.nodes if node.class_iri in scope]


def assemble_batch(
    scope: frozenset[str], parts: Parts, claims: Iterable[EdgeClaim] = ()
) -> GraphBatch:
    """Build a sorted batch. A node or edge repeated across parts is kept once, and an owner
    record wins over a stub of the same key. A repeated claim is kept once."""
    nodes: dict[NodeRef, NodeRecord] = {}
    for node in parts.nodes:
        kept = nodes.get(node.node_ref)
        if kept is None or (node.class_iri in scope and kept.class_iri not in scope):
            nodes[node.node_ref] = node
    edges = {(e.property_iri, e.start, e.end): e for e in reversed(parts.edges)}
    claims_by_key = {claim.sort_key: claim for claim in reversed(tuple(claims))}
    return GraphBatch(
        universe=UNIVERSE,
        scope=scope,
        nodes=tuple(sorted(nodes.values(), key=lambda node: (node.class_iri, node.id))),
        edges=tuple(
            sorted(
                edges.values(),
                key=lambda e: (
                    e.property_iri,
                    e.start.class_iri,
                    e.start.id,
                    e.end.class_iri,
                    e.end.id,
                ),
            )
        ),
        claims=tuple(claims_by_key[key] for key in sorted(claims_by_key)),
    )


def build_machine_id(kind: str, number_text: str, version_group_id: str) -> str:
    """Return the Machine id `{kind}{number}-{version group}`, e.g. `tm074-legends-z-a`.

    `number_text` keeps the digits the source gives, so `TM00` and `TM074` stay distinct
    from `TM0` and `TM74`. Species stubs and move owners must build the same id.
    """
    return f"{kind.lower()}{number_text}-{version_group_id}"


def build_encounter_id(
    species_id: str, location_id: str, version_id: str, method: str | None
) -> str:
    """Return the Encounter id `{species}:{location}:{version}[:{method}]`, e.g.
    `diglett:vermilion-city:heartgold:rock-smash`. An encounter with no method has no
    method segment."""
    segments = (species_id, location_id, version_id, *((method,) if method else ()))
    return ENCOUNTER_ID_SEPARATOR.join(segments)


class ChunkBuildError(Exception):
    """2 different texts of 1 subject and aspect share a chunk id. The message holds both."""

    def __init__(self, page_title: str, digest: str, first: str, second: str) -> None:
        super().__init__(f"{page_title}: texts share the hash {digest}: {first!r}, {second!r}")
        self.page_title = page_title
        self.digest = digest


@dataclass(frozen=True)
class MentionContext:
    """What a mention resolves against: the A-Box for types and a page type's species lookup."""

    abox: AboxIndex
    resolve_species: SpeciesResolver


def hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:TEXT_HASH_LENGTH]


def number_chunks(page_title: str, facts: Sequence[ChunkFact]) -> tuple[tuple[ChunkFact, str], ...]:
    """Pair each fact with the last segment of its id.

    The segment counts per (subject, aspect) from 1 in page order, and for an aspect in
    `WORK_ASPECTS` it is the hash of the text. Raises `ChunkBuildError` when 2 texts of 1
    subject and aspect share a hash.
    """
    counts: Counter[tuple[str, ChunkAspect]] = Counter()
    texts_by_hash: dict[tuple[str, ChunkAspect, str], str] = {}
    numbered: list[tuple[ChunkFact, str]] = []
    for fact in facts:
        if fact.aspect in WORK_ASPECTS:
            digest = hash_chunk_text(fact.text)
            earlier = texts_by_hash.setdefault((fact.subject_id, fact.aspect, digest), fact.text)
            if earlier != fact.text:
                raise ChunkBuildError(page_title, digest, earlier, fact.text)
            numbered.append((fact, digest))
            continue
        counts[(fact.subject_id, fact.aspect)] += 1
        numbered.append((fact, str(counts[(fact.subject_id, fact.aspect)])))
    return tuple(numbered)


def build_chunk_record(
    fact: ChunkFact, label: str, subject: NodeRecord, source_url: str
) -> NodeRecord:
    """Build the chunk node. A chunk of an aspect in `WORK_ASPECTS` names its origin through
    `fromWork` and carries no `source`."""
    chunk_id = CHUNK_ID_TEMPLATE.format(
        source=CHUNK_SOURCE,
        label=extract_local_name(subject.class_iri).lower(),
        subject=subject.id,
        aspect=fact.aspect.value,
        n=label,
    )
    properties: dict[str, PropertyValue] = {
        TEXT: fact.text,
        ASPECT: fact.aspect.value,
        SECTION: fact.section_path,
        SOURCE_URL: source_url,
    }
    if fact.aspect not in WORK_ASPECTS:
        properties[SOURCE] = CHUNK_SOURCE
    name = CHUNK_NAME_TEMPLATE.format(subject=subject.name, aspect=fact.aspect.value, n=label)
    return NodeRecord(class_iri=TEXT_CHUNK, id=chunk_id, name=name, properties=properties)


def resolve_mention(mention: MentionRef, context: MentionContext) -> NodeRecord | None:
    """Return the stub that a mention names, or `None` when nothing owns the target.

    A species goes through the page type's resolver and a type through the A-Box. A move or
    an ability is trusted as written, since its page may not be in the run yet.
    """
    if mention.template == SPECIES_MENTION:
        return context.resolve_species(mention.target)
    class_iri = MENTION_CLASSES[mention.template]
    if class_iri == TYPE:
        try:
            return resolve_abox(context.abox, class_iri, mention.target)
        except AboxMissError:
            return None
    return build_stub(class_iri, mention.target) if build_node_id(mention.target) else None


def describe_unresolved(mention: MentionRef) -> str:
    noun = MENTION_NOUNS[mention.template]
    return "{{" + f"{mention.template}|{mention.target}" + "}} names no " + noun


def build_mention_parts(
    fact: ChunkFact, chunk: NodeRecord, subject: NodeRecord, context: MentionContext
) -> tuple[Parts, tuple[MentionRef, ...]]:
    """Build the `mentionedIn` edges of a chunk and return the mentions that resolve to
    nothing. The subject never mentions itself."""
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    unresolved: list[MentionRef] = []
    for mention in fact.mentions:
        entity = resolve_mention(mention, context)
        if entity is None:
            unresolved.append(mention)
        elif entity.node_ref != subject.node_ref:
            nodes.append(entity)
            edges.append(build_edge(MENTIONED_IN, entity, chunk))
    return Parts(tuple(nodes), tuple(edges)), tuple(unresolved)


def build_work_parts(
    fact: ChunkFact, chunk: NodeRecord, abox: AboxIndex, work_class: str
) -> tuple[Parts, tuple[str, ...]]:
    """Build the `fromWork` edges of a chunk to the A-Box nodes of `work_class` that its
    `versions` name, and return the names the A-Box lacks."""
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    missing: list[str] = []
    for name in fact.versions:
        try:
            work = resolve_abox(abox, work_class, name)
        except AboxMissError:
            missing.append(name)
            continue
        nodes.append(work)
        edges.append(build_edge(FROM_WORK, chunk, work))
    return Parts(tuple(nodes), tuple(edges)), tuple(missing)
