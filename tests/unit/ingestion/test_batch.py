import pytest
from pydantic import ValidationError

from muon.ingestion.batch import (
    NODE_ID_PATTERN,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
    build_node_id,
)

EX = "https://example.org/fixture#"
WIDGET = EX + "Widget"
COLOR = EX + "Color"
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


def test_node_class_outside_scope_raises() -> None:
    stray = NodeRecord(class_iri=EX + "Gadget", id="beta", name="Beta")

    with pytest.raises(ValidationError, match="Gadget 'beta' has a class outside scope"):
        build_batch((stray,))


def test_batch_is_immutable() -> None:
    batch = build_batch((WIDGET_NODE,))

    with pytest.raises(ValidationError):
        batch.universe = "other"  # type: ignore[misc]
