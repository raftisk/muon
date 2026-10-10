"""Runner writes of stubs and edge properties, against the live Neo4j in `.env`.

Every test writes `muontoy` nodes only, and the `toy_runner` fixture purges that
universe afterwards.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord, PropertyValue
from muon.ingestion.runner import Runner, WriteMode
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology

pytestmark = pytest.mark.integration

TOY = "https://muon.dev/ns/muontoy#"
TOY_GADGET = TOY + "ToyGadget"
TOY_WIDGET = TOY + "ToyWidget"
TOY_COLOR = TOY + "ToyColor"
TOY_HAS_COLOR = TOY + "hasColor"
TOY_WEIGHT = TOY + "weight"

READ_NODE: LiteralString = """
MATCH (n:$($label):$($universe) {id: $id})
RETURN labels(n) AS labels, properties(n) AS properties
"""

READ_COLOR_EDGE: LiteralString = """
MATCH (:$($universe) {id: $start_id})-[r:HAS_COLOR]->(:$($universe) {id: $end_id})
RETURN properties(r) AS properties
"""


@pytest.fixture
async def toy_runner(graph_client: Neo4jGraphClient, tmp_path: Path) -> AsyncIterator[Runner]:
    model = load_ontology_model(copy_toy_ontology(tmp_path), TOY_UNIVERSE)
    runner = Runner(graph_client, LabelMaterializer(model))
    yield runner
    await runner.purge()


async def read_node(client: Neo4jGraphClient, label: str, node_id: str) -> dict[str, Any]:
    """Return the labels and properties of the 1 toy node with `label` and `node_id`."""
    (record,) = await client.execute_read(
        READ_NODE, {"label": label, "universe": TOY_UNIVERSE, "id": node_id}
    )
    return record


async def read_color_edge(client: Neo4jGraphClient, start_id: str, end_id: str) -> dict[str, Any]:
    """Return the properties of the 1 `HAS_COLOR` edge from `start_id` to `end_id`."""
    (record,) = await client.execute_read(
        READ_COLOR_EDGE, {"universe": TOY_UNIVERSE, "start_id": start_id, "end_id": end_id}
    )
    return dict(record["properties"])


def build_batch(
    scope: frozenset[str], nodes: tuple[NodeRecord, ...], edges: tuple[EdgeRecord, ...] = ()
) -> GraphBatch:
    return GraphBatch(universe=TOY_UNIVERSE, scope=scope, nodes=nodes, edges=edges)


def build_color_edge(
    gadget: NodeRecord, color: NodeRecord, **properties: PropertyValue
) -> EdgeRecord:
    return EdgeRecord(
        property_iri=TOY_HAS_COLOR,
        start=gadget.node_ref,
        end=color.node_ref,
        properties=properties,
    )


async def test_stub_creates_missing_node(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    gamma = NodeRecord(class_iri=TOY_WIDGET, id="gamma", name="Gamma")

    await toy_runner.write(build_batch(frozenset({TOY_COLOR}), (gamma,)), WriteMode.UPSERT)

    node = await read_node(graph_client, "ToyWidget", "gamma")
    assert set(node["labels"]) == {"ToyWidget", "ToyGadget", "Object", "Entity", TOY_UNIVERSE}
    assert node["properties"] == {"id": "gamma", "name": "Gamma"}


async def test_stub_keeps_owner_node(toy_runner: Runner, graph_client: Neo4jGraphClient) -> None:
    alpha = NodeRecord(class_iri=TOY_WIDGET, id="alpha", name="Alpha", properties={TOY_WEIGHT: 1.5})
    await toy_runner.write(build_batch(frozenset({TOY_WIDGET}), (alpha,)), WriteMode.UPSERT)

    for stub_class in (TOY_WIDGET, TOY_GADGET):
        stub = NodeRecord(class_iri=stub_class, id="alpha", name="Other")
        await toy_runner.write(build_batch(frozenset({TOY_COLOR}), (stub,)), WriteMode.UPSERT)

    node = await read_node(graph_client, "ToyGadget", "alpha")
    assert "ToyWidget" in node["labels"]
    assert node["properties"] == {"id": "alpha", "name": "Alpha", "weight": 1.5}


async def test_owner_fills_stub(toy_runner: Runner, graph_client: Neo4jGraphClient) -> None:
    red = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
    beta_stub = NodeRecord(class_iri=TOY_GADGET, id="beta", name="Beta")
    color_batch = build_batch(
        frozenset({TOY_COLOR}), (red, beta_stub), (build_color_edge(beta_stub, red),)
    )
    await toy_runner.write(color_batch, WriteMode.UPSERT)

    beta = NodeRecord(class_iri=TOY_GADGET, id="beta", name="Beta", properties={TOY_WEIGHT: 2.0})
    await toy_runner.write(build_batch(frozenset({TOY_GADGET}), (beta,)), WriteMode.UPSERT)

    node = await read_node(graph_client, "ToyGadget", "beta")
    assert node["properties"]["weight"] == 2.0
    assert await read_color_edge(graph_client, "beta", "red") == {}


async def test_edge_properties_upsert_merges_and_sync_replaces(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    alpha = NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha")
    red = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
    scope = frozenset({TOY_GADGET, TOY_COLOR})

    upsert_edges = (
        build_color_edge(alpha, red, shade="dark"),
        build_color_edge(alpha, red, gloss=True),
    )
    for upsert_edge in upsert_edges:
        await toy_runner.write(build_batch(scope, (alpha, red), (upsert_edge,)), WriteMode.UPSERT)
    merged = await read_color_edge(graph_client, "alpha", "red")

    sync_edge = build_color_edge(alpha, red, gloss=False)
    await toy_runner.write(build_batch(scope, (alpha, red), (sync_edge,)), WriteMode.SYNC)
    replaced = await read_color_edge(graph_client, "alpha", "red")

    assert merged == {"shade": "dark", "gloss": True}
    assert replaced == {"gloss": False}
