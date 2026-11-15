"""Schema introspection against the live Neo4j in `.env`.

Every test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards.
"""

from typing import LiteralString

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.runner import Runner
from muon.retrieval.schema import GraphSchema, NodeLabelSchema, PropertySchema, introspect_schema

pytestmark = pytest.mark.integration

WRITE_TOY_GRAPH: LiteralString = """
CREATE (w1:ToyWidget:ToyGadget:Object:Entity:muontoy
  {id: 'w1', name: 'W1', nickname: 'alpha', serial: 1})
CREATE (w2:ToyWidget:ToyGadget:Object:Entity:muontoy
  {id: 'w2', name: 'W2', nickname: 'beta', serial: 2})
CREATE (w3:ToyWidget:ToyGadget:Object:Entity:muontoy {id: 'w3', name: 'W3', nickname: 'alpha'})
CREATE (g1:ToyGadget:Object:Entity:muontoy {id: 'g1', name: 'G1', nickname: 'gizmo', serial: 3})
CREATE (c1:ToyColor:Concept:Entity:muontoy {id: 'c1', name: 'C1'})
CREATE (c2:ToyColor:Concept:Entity:muontoy {id: 'c2', name: 'C2'})
CREATE (w1)-[:HAS_COLOR]->(c1)
CREATE (w2)-[:HAS_COLOR]->(c2)
CREATE (w1)-[:PAIRS_WITH {strength: 'high'}]->(g1)
"""


def find_node_label(schema: GraphSchema, label: str) -> NodeLabelSchema:
    return next(node_label for node_label in schema.node_labels if node_label.label == label)


def find_property(node_label: NodeLabelSchema, name: str) -> PropertySchema:
    return next(property_ for property_ in node_label.properties if property_.name == name)


@pytest.fixture
async def toy_schema(graph_client: Neo4jGraphClient, toy_runner: Runner) -> GraphSchema:
    await graph_client.execute_write(WRITE_TOY_GRAPH)
    return await introspect_schema(graph_client, toy_runner.materializer)


async def test_leaf_classes_carry_counts_and_ancestors(toy_schema: GraphSchema) -> None:
    counts = {node_label.label: node_label.count for node_label in toy_schema.node_labels}

    assert counts == {"ToyColor": 2, "ToyGadget": 1, "ToyWidget": 3}
    assert find_node_label(toy_schema, "ToyWidget").ancestor_labels == (
        "Entity",
        "Object",
        "ToyGadget",
    )


async def test_a_property_missing_on_one_node_is_optional(toy_schema: GraphSchema) -> None:
    widget = find_node_label(toy_schema, "ToyWidget")

    assert find_property(widget, "serial").is_mandatory is False
    assert find_property(widget, "nickname").is_mandatory is True


async def test_enum_values_skip_the_nodes_of_subclasses(toy_schema: GraphSchema) -> None:
    widget_nickname = find_property(find_node_label(toy_schema, "ToyWidget"), "nickname")
    gadget_nickname = find_property(find_node_label(toy_schema, "ToyGadget"), "nickname")

    assert widget_nickname.enum_values == ("alpha", "beta")
    assert gadget_nickname.enum_values == ("gizmo",)


async def test_key_properties_list_no_values(toy_schema: GraphSchema) -> None:
    widget = find_node_label(toy_schema, "ToyWidget")

    assert find_property(widget, "id").enum_values is None
    assert find_property(widget, "name").enum_values is None


async def test_patterns_count_leaf_class_pairs(toy_schema: GraphSchema) -> None:
    patterns = [
        (pattern.start_label, pattern.type, pattern.end_label, pattern.count)
        for pattern in toy_schema.patterns
    ]

    assert patterns == [
        ("ToyWidget", "HAS_COLOR", "ToyColor", 2),
        ("ToyWidget", "PAIRS_WITH", "ToyGadget", 1),
    ]


async def test_relationship_property_values_are_sampled(toy_schema: GraphSchema) -> None:
    pairs_with = next(
        relationship
        for relationship in toy_schema.relationship_types
        if relationship.type == "PAIRS_WITH"
    )
    strength = next(
        property_ for property_ in pairs_with.properties if property_.name == "strength"
    )

    assert strength.enum_values == ("high",)
    assert strength.is_mandatory is True
