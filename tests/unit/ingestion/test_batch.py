import re

import pytest
from pydantic import ValidationError

from muon.ingestion.batch import (
    NODE_ID_PATTERN,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
    PropertyValue,
    build_node_id,
)

EX = "https://example.org/fixture#"
WIDGET = EX + "Widget"
COLOR = EX + "Color"
GADGET = EX + "Gadget"
HAS_COLOR = EX + "hasColor"


@pytest.mark.parametrize(
    ("name", "node_id"),
    [
        ("Pokémon", "pokemon"),
        ("Mr. Mime", "mr-mime"),
        ("  Fire  ", "fire"),
        ("Béta Unit", "beta-unit"),
        ("Let's Go Pikachu", "lets-go-pikachu"),
        ("Can't Escape", "cant-escape"),
        ("Farfetch’d", "farfetchd"),
        ("!!!", ""),
    ],
)
def test_build_node_id(name: str, node_id: str) -> None:
    result = build_node_id(name)

    assert result == node_id
    assert result == "" or NODE_ID_PATTERN.fullmatch(result)


def build_batch(nodes: tuple[NodeRecord, ...], edges: tuple[EdgeRecord, ...] = ()) -> GraphBatch:
    return GraphBatch(universe="ex", scope=frozenset({WIDGET, COLOR}), nodes=nodes, edges=edges)


WIDGET_NODE = NodeRecord(class_iri=WIDGET, id="alpha", name="Alpha", properties={EX + "size": 3})
COLOR_NODE = NodeRecord(class_iri=COLOR, id="red", name="Red")
COLOR_EDGE = EdgeRecord(
    property_iri=HAS_COLOR,
    start=NodeRef(class_iri=WIDGET, id="alpha"),
    end=NodeRef(class_iri=COLOR, id="red"),
)


def test_valid_batch_builds() -> None:
    batch = build_batch((WIDGET_NODE, COLOR_NODE), (COLOR_EDGE,))

    assert batch.nodes[0].node_ref == NodeRef(class_iri=WIDGET, id="alpha")
    assert batch.nodes[0].properties == {EX + "size": 3}


def test_duplicate_node_key_raises() -> None:
    duplicate = NodeRecord(class_iri=WIDGET, id="alpha", name="Alpha again")

    with pytest.raises(ValidationError, match="Duplicate node key .*alpha"):
        build_batch((WIDGET_NODE, duplicate))


def test_edge_to_missing_node_raises() -> None:
    with pytest.raises(ValidationError, match="missing node .*red"):
        build_batch((WIDGET_NODE,), (COLOR_EDGE,))


def test_node_outside_scope_is_stub() -> None:
    stub = NodeRecord(class_iri=GADGET, id="beta", name="Beta")

    batch = build_batch((WIDGET_NODE, stub))

    assert batch.stub_nodes == (stub,)
    assert batch.owner_nodes == (WIDGET_NODE,)


def test_stub_with_properties_raises() -> None:
    stub = NodeRecord(class_iri=GADGET, id="beta", name="Beta", properties={EX + "size": 1})
    message = f"Stub {GADGET} 'beta' is outside scope and carries {EX}size"

    with pytest.raises(ValidationError, match=re.escape(message)):
        build_batch((stub,))


@pytest.mark.parametrize("second_properties", [{}, {"shade": "dark"}])
def test_duplicate_edge_raises(second_properties: dict[str, PropertyValue]) -> None:
    duplicate = EdgeRecord(
        property_iri=HAS_COLOR,
        start=COLOR_EDGE.start,
        end=COLOR_EDGE.end,
        properties=second_properties,
    )
    message = f"Duplicate edge {HAS_COLOR} from {WIDGET} 'alpha' to {COLOR} 'red'"

    with pytest.raises(ValidationError, match=re.escape(message)):
        build_batch((WIDGET_NODE, COLOR_NODE), (COLOR_EDGE, duplicate))


def test_edge_properties_default_empty() -> None:
    assert COLOR_EDGE.properties == {}


def test_edge_properties_kept() -> None:
    edge = EdgeRecord(
        property_iri=HAS_COLOR,
        start=COLOR_EDGE.start,
        end=COLOR_EDGE.end,
        properties={"slot": 1, "isHidden": True},
    )

    batch = build_batch((WIDGET_NODE, COLOR_NODE), (edge,))

    assert batch.edges[0].properties == {"slot": 1, "isHidden": True}


def test_edge_may_end_at_stub() -> None:
    batch = GraphBatch(
        universe="ex",
        scope=frozenset({WIDGET}),
        nodes=(WIDGET_NODE, COLOR_NODE),
        edges=(COLOR_EDGE,),
    )

    assert batch.stub_nodes == (COLOR_NODE,)
    assert batch.edges == (COLOR_EDGE,)


def test_batch_is_immutable() -> None:
    batch = build_batch((WIDGET_NODE,))

    with pytest.raises(ValidationError):
        batch.universe = "other"  # type: ignore[misc]
