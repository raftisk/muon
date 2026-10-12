"""Runner: the only writer of graph data.

Compiles a `GraphBatch` into fixed Cypher statements and writes them in one
transaction. Labels and relationship types travel as parameters through dynamic
labels (`$(...)`), with names from the label materializer. A node merges on its
class label, the universe label and `id`; an edge merges on (start, type, end) and
carries its properties. A stub merges on the same key as a node and is set on
create only, so it never changes a node that exists. It also purges a universe in
transactions of at most `PURGE_BATCH_SIZE` nodes and counts what the universe holds.

In upsert mode, a batch with claims first deletes the edges it owns and no longer
lists. A claim row names an anchor, a relationship type and a direction; its
statement matches the edges there and deletes those missing from the row's `kept`
list. A claim that owns its other end also deletes the node at the other end of each
deleted edge, unless the batch lists that node as an owner record. Matching collects
the stale edges and nodes before any delete, so a node that 2 claims reach is tested
before it goes and deleted once. There is 1 statement per direction because Cypher
takes no relationship direction as a parameter.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any, Final, LiteralString

from muon.graph import GraphClient, Statement, WriteSummary
from muon.ingestion.batch import ClaimDirection, GraphBatch, NodeRecord, NodeRef
from muon.ingestion.schema import ensure_id_constraints
from muon.ontology.materializer import NAME_PROPERTY, LabelMaterializer

ID_PROPERTY = "id"
PURGE_BATCH_SIZE = 10_000


class WriteMode(StrEnum):
    """`upsert` merges and never deletes.

    `sync` treats the batch as the full truth for its scope: it also deletes the
    universe's scope nodes the batch does not list, and the relationships between
    2 scope nodes the batch does not list. A sync batch holds no stubs.
    """

    UPSERT = "upsert"
    SYNC = "sync"


@dataclass(frozen=True)
class UniverseCensus:
    """What a universe holds: nodes per label and relationships per type, sorted by name.

    A relationship counts by its start node, so one that ends outside the universe counts too.
    """

    labels: Mapping[str, int]
    relationship_types: Mapping[str, int]


@dataclass(frozen=True)
class RunReport:
    """Counters of one batch write, summed over its statements."""

    constraints_ensured: int
    nodes_created: int
    nodes_deleted: int
    relationships_created: int
    relationships_deleted: int
    labels_added: int
    labels_removed: int
    properties_set: int


# Parameter and row keys in these statements must match the builders below.
PRUNE_NODES: LiteralString = """
MATCH (n:$($universe))
WHERE any(label IN labels(n) WHERE label IN $scope_labels)
  AND NOT any(label IN labels(n) WHERE label IN $scope_labels AND n.id IN $ids_by_label[label])
DETACH DELETE n
"""

PRUNE_EDGES: LiteralString = """
MATCH (a:$($universe))-[r]->(b:$($universe))
WHERE any(label IN labels(a) WHERE label IN $scope_labels)
  AND any(label IN labels(b) WHERE label IN $scope_labels)
  AND NOT any(edge IN $edges WHERE edge.type = type(r)
    AND edge.start_label IN labels(a) AND edge.start_id = a.id
    AND edge.end_label IN labels(b) AND edge.end_id = b.id)
DELETE r
"""

# Row keys must match `build_claim_rows`. A stale edge is an edge at the claimed
# anchor that the batch does not list in `kept`.
PRUNE_CLAIMED_HEAD: LiteralString = """
UNWIND $claims AS claim
"""

PRUNE_CLAIMED_OUT_MATCH: LiteralString = """
MATCH (anchor:$([claim.anchor_label, $universe]) {id: claim.anchor_id})
  -[r:$(claim.type)]->(other:$($universe))
"""

PRUNE_CLAIMED_IN_MATCH: LiteralString = """
MATCH (anchor:$([claim.anchor_label, $universe]) {id: claim.anchor_id})
  <-[r:$(claim.type)]-(other:$($universe))
"""

PRUNE_CLAIMED_TAIL: LiteralString = """
WHERE (claim.other_label IS NULL OR claim.other_label IN labels(other))
  AND NOT any(kept IN claim.kept WHERE kept.label IN labels(other) AND kept.id = other.id)
WITH collect(DISTINCT r) AS stale_edges,
  collect(DISTINCT CASE
    WHEN claim.owns_other_end
      AND NOT any(label IN labels(other) WHERE other.id IN coalesce($owner_ids_by_label[label], []))
    THEN other
  END) AS orphans
FOREACH (stale_edge IN stale_edges | DELETE stale_edge)
FOREACH (orphan IN orphans | DETACH DELETE orphan)
"""

PRUNE_CLAIMED_OUT: LiteralString = PRUNE_CLAIMED_HEAD + PRUNE_CLAIMED_OUT_MATCH + PRUNE_CLAIMED_TAIL

PRUNE_CLAIMED_IN: LiteralString = PRUNE_CLAIMED_HEAD + PRUNE_CLAIMED_IN_MATCH + PRUNE_CLAIMED_TAIL

PRUNE_CLAIMED_BY_DIRECTION: Final[Mapping[ClaimDirection, LiteralString]] = {
    ClaimDirection.OUT: PRUNE_CLAIMED_OUT,
    ClaimDirection.IN: PRUNE_CLAIMED_IN,
}

MERGE_NODES_HEAD: LiteralString = """
UNWIND $nodes AS row
MERGE (n:$(row.key_labels) {id: row.id})
SET n:$(row.labels)
REMOVE n:$([label IN labels(n) WHERE label IN $class_labels AND NOT label IN row.labels])
"""

MERGE_NODES_SYNC: LiteralString = MERGE_NODES_HEAD + "SET n = row.properties\n"

MERGE_NODES_UPSERT: LiteralString = MERGE_NODES_HEAD + "SET n += row.properties\n"

# A stub never changes a node that exists: a reference by an ancestor class must
# not remove the owner's subclass label or overwrite its name.
MERGE_STUBS: LiteralString = """
UNWIND $stubs AS row
MERGE (n:$(row.key_labels) {id: row.id})
ON CREATE SET n:$(row.labels), n = row.properties
"""

MERGE_EDGES_HEAD: LiteralString = """
UNWIND $edges AS row
MATCH (a:$([row.start_label, $universe]) {id: row.start_id})
MATCH (b:$([row.end_label, $universe]) {id: row.end_id})
MERGE (a)-[r:$(row.type)]->(b)
"""

MERGE_EDGES_SYNC: LiteralString = MERGE_EDGES_HEAD + "SET r = row.properties\n"

MERGE_EDGES_UPSERT: LiteralString = MERGE_EDGES_HEAD + "SET r += row.properties\n"

PURGE_UNIVERSE: LiteralString = """
MATCH (n:$($universe))
WITH n LIMIT $batch_size
DETACH DELETE n
"""

COUNT_LABELS: LiteralString = """
MATCH (n:$($universe))
UNWIND labels(n) AS label
RETURN label, count(*) AS count
"""

COUNT_RELATIONSHIP_TYPES: LiteralString = """
MATCH (:$($universe))-[r]->()
RETURN type(r) AS type, count(*) AS count
"""


class Runner:
    """Writes batches of one universe through a `GraphClient` and purges that universe."""

    def __init__(self, client: GraphClient, materializer: LabelMaterializer) -> None:
        self.client = client
        self.materializer = materializer

    async def write(
        self, batch: GraphBatch, mode: WriteMode, should_ensure_constraints: bool = True
    ) -> RunReport:
        """Ensure the scope and stub class constraints, then write the batch in one transaction.

        A caller that ensured the constraints of every batch up front passes
        `should_ensure_constraints=False`; the report then counts 0 constraints.
        """
        if batch.universe != self.materializer.universe_label:
            raise ValueError(
                f"Batch universe {batch.universe!r} does not match the ontology universe "
                f"{self.materializer.universe_label!r}"
            )
        if mode is WriteMode.SYNC and batch.stub_nodes:
            stub = batch.stub_nodes[0]
            raise ValueError(
                f"Sync mode takes no stubs: {stub.class_iri} {stub.id!r} is outside scope"
            )
        if mode is WriteMode.SYNC and batch.claims:
            claim = batch.claims[0]
            raise ValueError(
                f"Sync mode takes no claims: {claim.property_iri} at "
                f"{claim.anchor.class_iri} {claim.anchor.id!r}"
            )
        constraints_ensured = 0
        if should_ensure_constraints:
            stub_classes = {stub.class_iri for stub in batch.stub_nodes}
            constraints_ensured = await self.ensure_constraints(batch.scope | stub_classes)
        statements = build_statements(batch, mode, self.materializer)
        summaries = await self.client.execute_write_batch(statements)
        return summarize_run(summaries, constraints_ensured)

    async def ensure_constraints(self, class_iris: Iterable[str]) -> int:
        """Ensure the `id` constraint of each class and return how many classes it covered."""
        return await ensure_id_constraints(self.client, self.materializer, class_iris)

    async def purge(self, batch_size: int = PURGE_BATCH_SIZE) -> WriteSummary:
        """Delete every node carrying the universe label, with its relationships.

        Each transaction deletes at most `batch_size` nodes, and the loop ends after a
        transaction that deletes none. An interrupted purge leaves part of the
        universe; running it again finishes the job. Constraints, indexes and the
        nodes of other universes stay. A relationship between a purged node and a
        node of another universe goes with the purged node.
        """
        parameters = {"universe": self.materializer.universe_label, "batch_size": batch_size}
        summaries: list[WriteSummary] = []
        while not summaries or summaries[-1].nodes_deleted > 0:
            summaries.append(await self.client.execute_write(PURGE_UNIVERSE, parameters))
        return add_summaries(summaries)

    async def count_universe(self) -> UniverseCensus:
        """Count the nodes per label and the relationships per type of the universe."""
        parameters = {"universe": self.materializer.universe_label}
        label_records = await self.client.execute_read(COUNT_LABELS, parameters)
        type_records = await self.client.execute_read(COUNT_RELATIONSHIP_TYPES, parameters)
        return UniverseCensus(
            labels={
                record["label"]: record["count"] for record in sort_records(label_records, "label")
            },
            relationship_types={
                record["type"]: record["count"] for record in sort_records(type_records, "type")
            },
        )


def build_statements(
    batch: GraphBatch, mode: WriteMode, materializer: LabelMaterializer
) -> list[Statement]:
    """Return the statements of `mode` in run order, each with the parameters it uses."""
    universe = materializer.universe_label
    node_parameters = {
        "nodes": build_node_rows(batch.owner_nodes, materializer),
        "class_labels": sorted(materializer.list_ontology_labels()),
    }
    edge_rows = build_edge_rows(batch, materializer)
    edge_parameters = {"universe": universe, "edges": edge_rows}
    if mode is WriteMode.UPSERT:
        stub_statements = (
            [Statement(MERGE_STUBS, {"stubs": build_node_rows(batch.stub_nodes, materializer)})]
            if batch.stub_nodes
            else []
        )
        return [
            *build_prune_statements(batch, materializer),
            Statement(MERGE_NODES_UPSERT, node_parameters),
            *stub_statements,
            Statement(MERGE_EDGES_UPSERT, edge_parameters),
        ]

    scope_labels = sorted(materializer.map_class_label(class_iri) for class_iri in batch.scope)
    prune_nodes_parameters = {
        "universe": universe,
        "scope_labels": scope_labels,
        "ids_by_label": build_ids_by_label(batch, materializer),
    }
    prune_edges_parameters = {
        "universe": universe,
        "scope_labels": scope_labels,
        "edges": edge_rows,
    }
    return [
        Statement(PRUNE_NODES, prune_nodes_parameters),
        Statement(PRUNE_EDGES, prune_edges_parameters),
        Statement(MERGE_NODES_SYNC, node_parameters),
        Statement(MERGE_EDGES_SYNC, edge_parameters),
    ]


def build_prune_statements(batch: GraphBatch, materializer: LabelMaterializer) -> list[Statement]:
    """Return the claim prune of each direction that has claims, `out` first."""
    parameters = {
        "universe": materializer.universe_label,
        "owner_ids_by_label": build_ids_by_label(batch, materializer),
    }
    statements: list[Statement] = []
    for direction, query in PRUNE_CLAIMED_BY_DIRECTION.items():
        claim_rows = build_claim_rows(batch, materializer, direction)
        if claim_rows:
            statements.append(Statement(query, {**parameters, "claims": claim_rows}))
    return statements


def build_claim_rows(
    batch: GraphBatch, materializer: LabelMaterializer, direction: ClaimDirection
) -> list[dict[str, Any]]:
    """Return one row per claim of `direction`, in claim order.

    `kept` lists the other ends of the batch's own edges that the claim matches: same
    property, anchor at the claimed end and, when the claim names a class, an other
    end that carries that class label.
    """
    other_ends = index_other_ends(batch, direction)
    rows: list[dict[str, Any]] = []
    for claim in batch.claims:
        if claim.direction is not direction:
            continue
        other_label = (
            materializer.map_class_label(claim.other_class_iri) if claim.other_class_iri else None
        )
        kept_ends = other_ends.get((claim.property_iri, claim.anchor), [])
        rows.append(
            {
                "anchor_label": materializer.map_class_label(claim.anchor.class_iri),
                "anchor_id": claim.anchor.id,
                "type": materializer.map_relationship_type(claim.property_iri),
                "other_label": other_label,
                "owns_other_end": claim.owns_other_end,
                "kept": [
                    {"label": materializer.map_class_label(end.class_iri), "id": end.id}
                    for end in kept_ends
                    if other_label is None
                    or other_label in materializer.map_class_labels(end.class_iri)
                ],
            }
        )
    return rows


def index_other_ends(
    batch: GraphBatch, direction: ClaimDirection
) -> dict[tuple[str, NodeRef], list[NodeRef]]:
    """Map (property IRI, anchor) to the other ends of the batch edges, in edge order."""
    other_ends: dict[tuple[str, NodeRef], list[NodeRef]] = {}
    for edge in batch.edges:
        anchor, other = (
            (edge.start, edge.end) if direction is ClaimDirection.OUT else (edge.end, edge.start)
        )
        other_ends.setdefault((edge.property_iri, anchor), []).append(other)
    return other_ends


def build_node_rows(
    nodes: Sequence[NodeRecord], materializer: LabelMaterializer
) -> list[dict[str, Any]]:
    return [
        {
            "key_labels": [
                materializer.map_class_label(node.class_iri),
                materializer.universe_label,
            ],
            "labels": list(materializer.map_class_labels(node.class_iri)),
            "id": node.id,
            "properties": build_node_properties(node, materializer),
        }
        for node in nodes
    ]


def build_node_properties(node: NodeRecord, materializer: LabelMaterializer) -> dict[str, Any]:
    """Return the full property map of a node: `id`, `name` and its datatype properties."""
    properties: dict[str, Any] = {ID_PROPERTY: node.id, NAME_PROPERTY: node.name}
    for property_iri, value in node.properties.items():
        property_name = materializer.map_property_name(property_iri)
        properties[property_name] = list(value) if isinstance(value, tuple) else value
    return properties


def build_edge_rows(batch: GraphBatch, materializer: LabelMaterializer) -> list[dict[str, Any]]:
    return [
        {
            "type": materializer.map_relationship_type(edge.property_iri),
            "start_label": materializer.map_class_label(edge.start.class_iri),
            "start_id": edge.start.id,
            "end_label": materializer.map_class_label(edge.end.class_iri),
            "end_id": edge.end.id,
            "properties": {
                name: list(value) if isinstance(value, tuple) else value
                for name, value in edge.properties.items()
            },
        }
        for edge in batch.edges
    ]


def build_ids_by_label(batch: GraphBatch, materializer: LabelMaterializer) -> dict[str, list[str]]:
    """Map every scope label to the ids the batch lists, so no scope label is missing."""
    ids_by_label: dict[str, list[str]] = {
        materializer.map_class_label(class_iri): [] for class_iri in batch.scope
    }
    for node in batch.owner_nodes:
        ids_by_label[materializer.map_class_label(node.class_iri)].append(node.id)
    return ids_by_label


def sort_records(records: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record[key])


def add_summaries(summaries: Iterable[WriteSummary]) -> WriteSummary:
    """Return the field-wise sum of the summaries."""
    summary_list = list(summaries)
    return WriteSummary(
        **{
            field.name: sum(getattr(summary, field.name) for summary in summary_list)
            for field in fields(WriteSummary)
        }
    )


def summarize_run(summaries: list[WriteSummary], constraints_ensured: int) -> RunReport:
    return RunReport(constraints_ensured=constraints_ensured, **asdict(add_summaries(summaries)))
