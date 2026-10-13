"""The claim prune of upsert writes, against the live Neo4j in `.env`.

Every test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards. Each test writes a batch, then an edited copy of it, and reads the
universe back.
"""

from typing import LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import ClaimDirection, EdgeClaim, EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.runner import Runner, RunReport, WriteMode
from tests.fixtures.toy_ontology import TOY_UNIVERSE

pytestmark = pytest.mark.integration

TOY = "https://muon.dev/ns/muontoy#"
TOY_GADGET = TOY + "ToyGadget"
TOY_WIDGET = TOY + "ToyWidget"
TOY_COLOR = TOY + "ToyColor"
TOY_HAS_COLOR = TOY + "hasColor"
TOY_PAIRS_WITH = TOY + "pairsWith"

ALPHA = NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha")
BETA = NodeRecord(class_iri=TOY_GADGET, id="beta", name="Beta")
GADGET_STUB = NodeRecord(class_iri=TOY_GADGET, id="g1", name="G1")
FIRST_WIDGET = NodeRecord(class_iri=TOY_WIDGET, id="w1", name="W1")
SECOND_WIDGET = NodeRecord(class_iri=TOY_WIDGET, id="w2", name="W2")
RED = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
BLUE = NodeRecord(class_iri=TOY_COLOR, id="blue", name="Blue")

READ_NODE_IDS: LiteralString = "MATCH (n:$($universe)) RETURN n.id AS id"

READ_EDGES: LiteralString = """
MATCH (a:$($universe))-[r]->(b:$($universe))
RETURN a.id AS start, type(r) AS type, b.id AS end
"""


def build_batch(
    scope: frozenset[str],
    nodes: tuple[NodeRecord, ...],
    edges: tuple[EdgeRecord, ...] = (),
    claims: tuple[EdgeClaim, ...] = (),
) -> GraphBatch:
    return GraphBatch(universe=TOY_UNIVERSE, scope=scope, nodes=nodes, edges=edges, claims=claims)


def build_edge(property_iri: str, start: NodeRecord, end: NodeRecord) -> EdgeRecord:
    return EdgeRecord(property_iri=property_iri, start=start.node_ref, end=end.node_ref)


def build_claim(
    anchor: NodeRecord,
    property_iri: str,
    direction: ClaimDirection,
    other_class_iri: str | None,
    owns_other_end: bool = False,
) -> EdgeClaim:
    return EdgeClaim(
        anchor=anchor.node_ref,
        property_iri=property_iri,
        direction=direction,
        other_class_iri=other_class_iri,
        owns_other_end=owns_other_end,
    )


async def read_node_ids(client: Neo4jGraphClient) -> set[str]:
    records = await client.execute_read(READ_NODE_IDS, {"universe": TOY_UNIVERSE})
    return {record["id"] for record in records}


async def read_edges(client: Neo4jGraphClient) -> set[tuple[str, str, str]]:
    records = await client.execute_read(READ_EDGES, {"universe": TOY_UNIVERSE})
    return {(record["start"], record["type"], record["end"]) for record in records}


async def test_claimed_edge_goes_and_its_other_end_stays(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    scope = frozenset({TOY_GADGET, TOY_COLOR})
    claim = build_claim(ALPHA, TOY_HAS_COLOR, ClaimDirection.OUT, TOY_COLOR)
    first = build_batch(
        scope,
        (ALPHA, RED, BLUE),
        (build_edge(TOY_HAS_COLOR, ALPHA, RED), build_edge(TOY_HAS_COLOR, ALPHA, BLUE)),
        (claim,),
    )
    second = build_batch(scope, (ALPHA, RED), (build_edge(TOY_HAS_COLOR, ALPHA, RED),), (claim,))

    await toy_runner.write(first, WriteMode.UPSERT)
    await toy_runner.write(second, WriteMode.UPSERT)

    assert await read_edges(graph_client) == {("alpha", "HAS_COLOR", "red")}
    assert await read_node_ids(graph_client) == {"alpha", "red", "blue"}


async def test_owned_other_end_of_a_stub_anchor_goes_with_its_edge(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    scope = frozenset({TOY_WIDGET})
    claim = build_claim(GADGET_STUB, TOY_PAIRS_WITH, ClaimDirection.IN, TOY_WIDGET, True)
    first = build_batch(
        scope,
        (FIRST_WIDGET, SECOND_WIDGET, GADGET_STUB),
        (
            build_edge(TOY_PAIRS_WITH, FIRST_WIDGET, GADGET_STUB),
            build_edge(TOY_PAIRS_WITH, SECOND_WIDGET, GADGET_STUB),
        ),
        (claim,),
    )
    second = build_batch(
        scope,
        (FIRST_WIDGET, GADGET_STUB),
        (build_edge(TOY_PAIRS_WITH, FIRST_WIDGET, GADGET_STUB),),
        (claim,),
    )

    await toy_runner.write(first, WriteMode.UPSERT)
    await toy_runner.write(second, WriteMode.UPSERT)

    assert await read_node_ids(graph_client) == {"w1", "g1"}
    assert await read_edges(graph_client) == {("w1", "PAIRS_WITH", "g1")}


async def test_owner_record_keeps_its_node_when_its_edge_goes(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    scope = frozenset({TOY_WIDGET})
    claim = build_claim(GADGET_STUB, TOY_PAIRS_WITH, ClaimDirection.IN, TOY_WIDGET, True)
    first = build_batch(
        scope,
        (FIRST_WIDGET, SECOND_WIDGET, GADGET_STUB),
        (
            build_edge(TOY_PAIRS_WITH, FIRST_WIDGET, GADGET_STUB),
            build_edge(TOY_PAIRS_WITH, SECOND_WIDGET, GADGET_STUB),
        ),
        (claim,),
    )
    second = build_batch(
        scope,
        (FIRST_WIDGET, SECOND_WIDGET, GADGET_STUB),
        (build_edge(TOY_PAIRS_WITH, FIRST_WIDGET, GADGET_STUB),),
        (claim,),
    )

    await toy_runner.write(first, WriteMode.UPSERT)
    await toy_runner.write(second, WriteMode.UPSERT)

    assert await read_node_ids(graph_client) == {"w1", "w2", "g1"}
    assert await read_edges(graph_client) == {("w1", "PAIRS_WITH", "g1")}


async def test_claim_with_any_class_deletes_edges_to_every_class(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    scope = frozenset({TOY_GADGET, TOY_WIDGET})
    claim = build_claim(ALPHA, TOY_PAIRS_WITH, ClaimDirection.OUT, None)
    first = build_batch(
        scope,
        (ALPHA, BETA, FIRST_WIDGET),
        (build_edge(TOY_PAIRS_WITH, ALPHA, BETA), build_edge(TOY_PAIRS_WITH, ALPHA, FIRST_WIDGET)),
        (claim,),
    )
    second = build_batch(scope, (ALPHA,), claims=(claim,))

    await toy_runner.write(first, WriteMode.UPSERT)
    await toy_runner.write(second, WriteMode.UPSERT)

    assert await read_edges(graph_client) == set()
    assert await read_node_ids(graph_client) == {"alpha", "beta", "w1"}


async def test_same_batch_written_twice_changes_nothing(toy_runner: Runner) -> None:
    scope = frozenset({TOY_WIDGET})
    claim = build_claim(GADGET_STUB, TOY_PAIRS_WITH, ClaimDirection.IN, TOY_WIDGET, True)
    batch = build_batch(
        scope,
        (FIRST_WIDGET, GADGET_STUB),
        (build_edge(TOY_PAIRS_WITH, FIRST_WIDGET, GADGET_STUB),),
        (claim,),
    )

    await toy_runner.write(batch, WriteMode.UPSERT)
    report = await toy_runner.write(batch, WriteMode.UPSERT)

    assert report == RunReport(
        constraints_ensured=report.constraints_ensured,
        nodes_created=0,
        nodes_deleted=0,
        relationships_created=0,
        relationships_deleted=0,
        labels_added=0,
        labels_removed=0,
        properties_set=report.properties_set,
    )


async def test_census_counts_toy_labels_and_relationship_types(toy_runner: Runner) -> None:
    batch = build_batch(
        frozenset({TOY_GADGET, TOY_COLOR}),
        (ALPHA, RED, BLUE),
        (build_edge(TOY_HAS_COLOR, ALPHA, RED), build_edge(TOY_HAS_COLOR, ALPHA, BLUE)),
    )
    await toy_runner.write(batch, WriteMode.UPSERT)

    census = await toy_runner.count_universe()

    assert census.labels["ToyGadget"] == 1
    assert census.labels["ToyColor"] == 2
    assert census.labels[TOY_UNIVERSE] == 3
    assert dict(census.relationship_types) == {"HAS_COLOR": 2}
    assert list(census.labels) == sorted(census.labels)
