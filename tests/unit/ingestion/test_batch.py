import re

import pytest
from pydantic import ValidationError

from muon.ingestion.batch import (
    NODE_ID_PATTERN,
    ClaimDirection,
    EdgeClaim,
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


GADGET_STUB = NodeRecord(class_iri=GADGET, id="beta", name="Beta")
OWNER_CLAIM = EdgeClaim(
    anchor=WIDGET_NODE.node_ref,
    property_iri=HAS_COLOR,
    direction=ClaimDirection.OUT,
    other_class_iri=COLOR,
)
STUB_CLAIM = EdgeClaim(
    anchor=GADGET_STUB.node_ref,
    property_iri=HAS_COLOR,
    direction=ClaimDirection.IN,
    other_class_iri=WIDGET,
    owns_other_end=True,
)


def build_claim_batch(*claims: EdgeClaim) -> GraphBatch:
    return GraphBatch(
        universe="ex",
        scope=frozenset({WIDGET, COLOR}),
        nodes=(WIDGET_NODE, COLOR_NODE, GADGET_STUB),
        edges=(),
        claims=claims,
    )


def test_claims_default_empty() -> None:
    assert build_batch((WIDGET_NODE,)).claims == ()


def test_claim_on_owner_and_on_stub() -> None:
    batch = build_claim_batch(OWNER_CLAIM, STUB_CLAIM)

    assert batch.claims == (OWNER_CLAIM, STUB_CLAIM)
    assert OWNER_CLAIM.owns_other_end is False
    assert OWNER_CLAIM.sort_key == (WIDGET, "alpha", HAS_COLOR, "out", COLOR)


def test_claim_with_any_class_sorts_with_empty_other_class() -> None:
    claim = EdgeClaim(
        anchor=WIDGET_NODE.node_ref, property_iri=HAS_COLOR, direction=ClaimDirection.IN
    )

    assert claim.other_class_iri is None
    assert claim.sort_key == (WIDGET, "alpha", HAS_COLOR, "in", "")


def test_claim_without_anchor_node_raises() -> None:
    claim = EdgeClaim(
        anchor=NodeRef(class_iri=WIDGET, id="missing"),
        property_iri=HAS_COLOR,
        direction=ClaimDirection.OUT,
    )
    message = f"Claim {HAS_COLOR} at {WIDGET} 'missing' refers to a missing anchor node"

    with pytest.raises(ValidationError, match=re.escape(message)):
        build_claim_batch(claim)


def test_duplicate_claim_raises() -> None:
    duplicate = EdgeClaim(
        anchor=WIDGET_NODE.node_ref,
        property_iri=HAS_COLOR,
        direction=ClaimDirection.OUT,
        other_class_iri=COLOR,
        owns_other_end=True,
    )
    message = f"Duplicate claim {HAS_COLOR} at {WIDGET} 'alpha'"

    with pytest.raises(ValidationError, match=re.escape(message)):
        build_claim_batch(OWNER_CLAIM, duplicate)


def test_claims_differing_in_direction_or_class_are_distinct() -> None:
    inbound = OWNER_CLAIM.model_copy(update={"direction": ClaimDirection.IN})
    any_class = OWNER_CLAIM.model_copy(update={"other_class_iri": None})

    batch = build_claim_batch(OWNER_CLAIM, inbound, any_class)

    assert len(batch.claims) == 3


@pytest.mark.parametrize("other_class_iri", [None, GADGET])
def test_owned_other_end_outside_scope_raises(other_class_iri: str | None) -> None:
    claim = EdgeClaim(
        anchor=WIDGET_NODE.node_ref,
        property_iri=HAS_COLOR,
        direction=ClaimDirection.OUT,
        other_class_iri=other_class_iri,
        owns_other_end=True,
    )
    message = f"Claim {HAS_COLOR} at {WIDGET} 'alpha' owns its other end"

    with pytest.raises(ValidationError, match=re.escape(message)):
        build_claim_batch(claim)


def test_claims_survive_json_round_trip() -> None:
    batch = build_claim_batch(OWNER_CLAIM, STUB_CLAIM)

    restored = GraphBatch.model_validate(batch.model_dump(mode="json"))

    assert restored == batch
    assert restored.claims[1].direction is ClaimDirection.IN


def test_json_without_claims_key_loads_with_no_claims() -> None:
    payload = build_batch((WIDGET_NODE,)).model_dump(mode="json")
    del payload["claims"]

    assert GraphBatch.model_validate(payload).claims == ()
