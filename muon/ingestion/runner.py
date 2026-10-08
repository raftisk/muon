"""Runner: the only writer of graph data.

Compiles a `GraphBatch` into fixed Cypher statements and writes them in one
transaction. Labels and relationship types travel as parameters through dynamic
labels (`$(...)`), with names from the label materializer. A node merges on its
class label, the universe label and `id`; an edge merges on (start, type, end).
It also purges a universe: every node with the universe label, in one statement.
"""

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any, LiteralString

from muon.graph import GraphClient, Statement, WriteSummary
from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.schema import ensure_id_constraints
from muon.ontology.materializer import NAME_PROPERTY, LabelMaterializer

ID_PROPERTY = "id"


class WriteMode(StrEnum):
    """`upsert` merges and never deletes.

    `sync` treats the batch as the full truth for its scope: it also deletes the
    universe's scope nodes the batch does not list, and the relationships between
    2 scope nodes the batch does not list.
    """

    UPSERT = "upsert"
    SYNC = "sync"


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

MERGE_NODES_HEAD: LiteralString = """
UNWIND $nodes AS row
MERGE (n:$(row.key_labels) {id: row.id})
SET n:$(row.labels)
REMOVE n:$([label IN labels(n) WHERE label IN $class_labels AND NOT label IN row.labels])
"""

MERGE_NODES_SYNC: LiteralString = MERGE_NODES_HEAD + "SET n = row.properties\n"

MERGE_NODES_UPSERT: LiteralString = MERGE_NODES_HEAD + "SET n += row.properties\n"

MERGE_EDGES: LiteralString = """
UNWIND $edges AS row
MATCH (a:$([row.start_label, $universe]) {id: row.start_id})
MATCH (b:$([row.end_label, $universe]) {id: row.end_id})
MERGE (a)-[:$(row.type)]->(b)
"""

PURGE_UNIVERSE: LiteralString = """
MATCH (n:$($universe))
DETACH DELETE n
"""


class Runner:
    """Writes batches of one universe through a `GraphClient` and purges that universe."""

    def __init__(self, client: GraphClient, materializer: LabelMaterializer) -> None:
        self.client = client
        self.materializer = materializer

    async def write(self, batch: GraphBatch, mode: WriteMode) -> RunReport:
        """Ensure the scope constraints, then write the batch in one transaction."""
        if batch.universe != self.materializer.universe_label:
            raise ValueError(
                f"Batch universe {batch.universe!r} does not match the ontology universe "
                f"{self.materializer.universe_label!r}"
            )
        constraints_ensured = await ensure_id_constraints(
            self.client, self.materializer, batch.scope
        )
        statements = build_statements(batch, mode, self.materializer)
        summaries = await self.client.execute_write_batch(statements)
        return summarize_run(summaries, constraints_ensured)

    async def purge(self) -> WriteSummary:
        """Delete every node carrying the universe label, with its relationships.

        Constraints, indexes and the nodes of other universes stay. A relationship
        between a purged node and a node of another universe goes with the purged node.
        """
        return await self.client.execute_write(
            PURGE_UNIVERSE, {"universe": self.materializer.universe_label}
        )


def build_statements(
    batch: GraphBatch, mode: WriteMode, materializer: LabelMaterializer
) -> list[Statement]:
    """Return the statements of `mode` in run order, each with the parameters it uses."""
    universe = materializer.universe_label
    node_parameters = {
        "nodes": build_node_rows(batch, materializer),
        "class_labels": sorted(materializer.list_ontology_labels()),
    }
    edge_rows = build_edge_rows(batch, materializer)
    merge_edges = Statement(MERGE_EDGES, {"universe": universe, "edges": edge_rows})
    if mode is WriteMode.UPSERT:
        return [Statement(MERGE_NODES_UPSERT, node_parameters), merge_edges]

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
        merge_edges,
    ]


def build_node_rows(batch: GraphBatch, materializer: LabelMaterializer) -> list[dict[str, Any]]:
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
        for node in batch.nodes
    ]


def build_node_properties(node: NodeRecord, materializer: LabelMaterializer) -> dict[str, Any]:
    """Return the full property map of a node: `id`, `name` and its datatype properties."""
    properties: dict[str, Any] = {ID_PROPERTY: node.id, NAME_PROPERTY: node.name}
    for property_iri, value in node.properties.items():
        property_name = materializer.map_property_name(property_iri)
        properties[property_name] = list(value) if isinstance(value, tuple) else value
    return properties


def build_edge_rows(batch: GraphBatch, materializer: LabelMaterializer) -> list[dict[str, str]]:
    return [
        {
            "type": materializer.map_relationship_type(edge.property_iri),
            "start_label": materializer.map_class_label(edge.start.class_iri),
            "start_id": edge.start.id,
            "end_label": materializer.map_class_label(edge.end.class_iri),
            "end_id": edge.end.id,
        }
        for edge in batch.edges
    ]


def build_ids_by_label(batch: GraphBatch, materializer: LabelMaterializer) -> dict[str, list[str]]:
    """Map every scope label to the ids the batch lists, so no scope label is missing."""
    ids_by_label: dict[str, list[str]] = {
        materializer.map_class_label(class_iri): [] for class_iri in batch.scope
    }
    for node in batch.nodes:
        ids_by_label[materializer.map_class_label(node.class_iri)].append(node.id)
    return ids_by_label


def summarize_run(summaries: list[WriteSummary], constraints_ensured: int) -> RunReport:
    totals = {
        field.name: sum(getattr(summary, field.name) for summary in summaries)
        for field in fields(WriteSummary)
    }
    return RunReport(constraints_ensured=constraints_ensured, **totals)
