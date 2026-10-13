"""The contract between producers and the graph, in ontology terms.

A `GraphBatch` holds class and property IRIs, never Neo4j labels or relationship
types; the label materializer supplies those when the batch is written. Edge
property names are the exception: they exist in Neo4j only and travel as Neo4j
names. A node is keyed by its asserted class and its `id`, which `build_node_id`
derives from the English name.
"""

import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NODE_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*(:[a-z0-9]+(-[a-z0-9]+)*)*$")
APOSTROPHES: Final = re.compile("['’]")
NON_ID_CHARACTERS = re.compile(r"[^a-z0-9]+")
ID_SEPARATOR = "-"
UNICODE_DECOMPOSITION: Final = "NFKD"

PropertyValue = str | int | float | bool | tuple[str, ...]


def build_node_id(name: str) -> str:
    """Return the node `id` for an English name, e.g. "Béta Unit" -> "beta-unit".

    Accents and apostrophes are stripped ("Let's Go" -> "lets-go", "Farfetch'd" ->
    "farfetchd"), the rest is lowercased and each run of characters outside `[a-z0-9]`
    becomes one hyphen. The result is empty when `name` holds no letter or digit.
    """
    decomposed = unicodedata.normalize(UNICODE_DECOMPOSITION, name)
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    unquoted = APOSTROPHES.sub("", unaccented)
    return NON_ID_CHARACTERS.sub(ID_SEPARATOR, unquoted.lower()).strip(ID_SEPARATOR)


class NodeRef(BaseModel):
    """The key of a node: its asserted class and its `id`."""

    model_config = ConfigDict(frozen=True)

    class_iri: str
    id: str


class NodeRecord(BaseModel):
    """One node. `properties` maps datatype property IRIs to values."""

    model_config = ConfigDict(frozen=True)

    class_iri: str
    id: str
    name: str
    properties: Mapping[str, PropertyValue] = Field(default_factory=dict)

    @property
    def node_ref(self) -> NodeRef:
        return NodeRef(class_iri=self.class_iri, id=self.id)


class EdgeRecord(BaseModel):
    """One edge, typed by its most specific object property.

    A producer uses the super-property instead when an edge property carries what a
    sub-property would say. `properties` maps Neo4j edge property names to values.
    """

    model_config = ConfigDict(frozen=True)

    property_iri: str
    start: NodeRef
    end: NodeRef
    properties: Mapping[str, PropertyValue] = Field(default_factory=dict)


class ClaimDirection(StrEnum):
    """Which end of the claimed edges the anchor sits at."""

    OUT = "out"
    IN = "in"


class EdgeClaim(BaseModel):
    """The edges of one property at one anchor that a batch owns.

    The batch lists every edge of `property_iri` at `anchor`, in `direction`, whose
    other end carries the label of `other_class_iri`; `None` means any node of the
    universe. A write in upsert mode deletes the matching edges that the batch does
    not list. With `owns_other_end`, it also deletes the node at the other end of
    each deleted edge, unless the batch lists that node as an owner record.
    """

    model_config = ConfigDict(frozen=True)

    anchor: NodeRef
    property_iri: str
    direction: ClaimDirection
    other_class_iri: str | None = None
    owns_other_end: bool = False

    @property
    def sort_key(self) -> tuple[str, str, str, str, str]:
        """The identity of a claim and the order a producer sorts claims in."""
        return (
            self.anchor.class_iri,
            self.anchor.id,
            self.property_iri,
            self.direction.value,
            self.other_class_iri or "",
        )


class GraphBatch(BaseModel):
    """One unit of validation and one write transaction.

    `scope` holds the classes the batch owns. A node record of any other class is a
    stub: a node the batch references but does not own, with `name` and no
    properties. Node keys and edge keys (property IRI, start, end) are unique, and
    every edge end matches a node record, owner or stub.

    `claims` name the edge patterns the batch is complete for, so a re-write can
    delete what the batch no longer lists. A claim anchors at a node record of the
    batch, and its identity is its `sort_key`. The batch checks claim keys, not
    their order; a claim that owns its other end names a class in `scope`.
    """

    model_config = ConfigDict(frozen=True)

    universe: str
    scope: frozenset[str]
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]
    claims: tuple[EdgeClaim, ...] = ()

    @property
    def owner_nodes(self) -> tuple[NodeRecord, ...]:
        return tuple(node for node in self.nodes if node.class_iri in self.scope)

    @property
    def stub_nodes(self) -> tuple[NodeRecord, ...]:
        return tuple(node for node in self.nodes if node.class_iri not in self.scope)

    @model_validator(mode="after")
    def check_nodes(self) -> Self:
        node_refs: set[NodeRef] = set()
        for node in self.nodes:
            if node.node_ref in node_refs:
                raise ValueError(f"Duplicate node key {node.class_iri} {node.id!r}")
            node_refs.add(node.node_ref)

        for stub in self.stub_nodes:
            if stub.properties:
                first_property = next(iter(stub.properties))
                raise ValueError(
                    f"Stub {stub.class_iri} {stub.id!r} is outside scope and carries "
                    f"{first_property}; a stub carries name only"
                )
        return self

    @model_validator(mode="after")
    def check_edges(self) -> Self:
        node_refs = {node.node_ref for node in self.nodes}
        edge_keys: set[tuple[str, NodeRef, NodeRef]] = set()
        for edge in self.edges:
            missing_ends = [end for end in (edge.start, edge.end) if end not in node_refs]
            if missing_ends:
                end = missing_ends[0]
                raise ValueError(
                    f"Edge {edge.property_iri} refers to a missing node {end.class_iri} {end.id!r}"
                )
            edge_key = (edge.property_iri, edge.start, edge.end)
            if edge_key in edge_keys:
                raise ValueError(
                    f"Duplicate edge {edge.property_iri} from {edge.start.class_iri} "
                    f"{edge.start.id!r} to {edge.end.class_iri} {edge.end.id!r}"
                )
            edge_keys.add(edge_key)
        return self

    @model_validator(mode="after")
    def check_claims(self) -> Self:
        node_refs = {node.node_ref for node in self.nodes}
        claim_keys: set[tuple[str, str, str, str, str]] = set()
        for claim in self.claims:
            anchor = claim.anchor
            label = f"{claim.property_iri} at {anchor.class_iri} {anchor.id!r}"
            if anchor not in node_refs:
                raise ValueError(f"Claim {label} refers to a missing anchor node")
            if claim.sort_key in claim_keys:
                raise ValueError(
                    f"Duplicate claim {label} ({claim.direction.value}, "
                    f"{claim.other_class_iri or 'any class'})"
                )
            claim_keys.add(claim.sort_key)
            if claim.owns_other_end and claim.other_class_iri not in self.scope:
                raise ValueError(
                    f"Claim {label} owns its other end, which needs an other class in "
                    f"scope; got {claim.other_class_iri}"
                )
        return self
