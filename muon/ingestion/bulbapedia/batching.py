"""pkmn batch building shared by the Bulbapedia page types.

Every page type resolves the names the hand-written A-Box owns through `AboxIndex`, builds
stubs and edges the same way and claims its edge patterns with `ClaimRule`s. A page type
owns its scope, its claim tables and its own records; this module owns what they share,
including the machine id, which species stubs and move owners must agree on.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping, Sequence
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

UNIVERSE = "pkmn"
PKMN = "https://muon.dev/ns/pkmn#"
CORE = "https://muon.dev/ns/core#"

TYPE = PKMN + "Type"
GENERATION = PKMN + "Generation"
VERSION_GROUP = PKMN + "VersionGroup"
MOVE = PKMN + "Move"
MACHINE = PKMN + "Machine"
REGION = CORE + "Region"

HAS_TYPE = PKMN + "hasType"
INTRODUCED_IN = PKMN + "introducedIn"
TEACHES_MOVE = PKMN + "teachesMove"
IN_VERSION_GROUP = PKMN + "inVersionGroup"
ALIASES = CORE + "aliases"

GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERALS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


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
