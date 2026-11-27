"""The write policy check: 1 broken batch per rule gives only that rule id."""

import pytest

from muon.ingestion.batch import (
    ClaimDirection,
    EdgeClaim,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
)
from muon.ingestion.policy import ClaimRule, WritePolicy, check_policy

TOY = "https://example.org/toy#"
EFFECT = TOY + "Effect"
OWNER = TOY + "Owner"
TAG = TOY + "Tag"
LEVEL = TOY + "level"
EXTRACTED_BY = TOY + "extractedBy"
HAS_TAG = TOY + "hasTag"
HAS_EFFECT = TOY + "hasEffect"

CLAIM_RULE = ClaimRule(
    property_iri=HAS_EFFECT,
    direction=ClaimDirection.OUT,
    other_class_iri=EFFECT,
    owns_other_end=True,
)
POLICY = WritePolicy(
    job="toy_effects",
    owned_classes=frozenset({EFFECT}),
    owner_properties=frozenset({LEVEL, EXTRACTED_BY}),
    required_owner_properties=frozenset({EXTRACTED_BY}),
    edge_properties=frozenset({HAS_TAG, HAS_EFFECT}),
    claim_rules=(CLAIM_RULE,),
)

OWNER_STUB = NodeRecord(class_iri=OWNER, id="owner", name="Owner")
TAG_STUB = NodeRecord(class_iri=TAG, id="tag", name="Tag")
VALID_PROPERTIES = {LEVEL: 2, EXTRACTED_BY: "toy_effects:v1:model"}
VALID_CLAIM = EdgeClaim(
    anchor=OWNER_STUB.node_ref,
    property_iri=HAS_EFFECT,
    direction=ClaimDirection.OUT,
    other_class_iri=EFFECT,
    owns_other_end=True,
)


def build_batch(**changes: object) -> GraphBatch:
    """The batch that follows `POLICY`, with `changes` to its fields.

    `properties` and `tag_property` change the effect node and the edge to the tag.
    """
    effect = NodeRecord(
        class_iri=EFFECT,
        id="effect",
        name="Effect",
        properties=changes.pop("properties", VALID_PROPERTIES),
    )
    tag_property = changes.pop("tag_property", HAS_TAG)
    fields: dict[str, object] = {
        "universe": "toy",
        "scope": frozenset({EFFECT}),
        "nodes": (effect, OWNER_STUB, TAG_STUB),
        "edges": (
            EdgeRecord(property_iri=HAS_EFFECT, start=OWNER_STUB.node_ref, end=effect.node_ref),
            EdgeRecord(property_iri=tag_property, start=effect.node_ref, end=TAG_STUB.node_ref),
        ),
        "claims": (VALID_CLAIM,),
    }
    return GraphBatch.model_validate(fields | changes)


def list_rule_ids(batch: GraphBatch) -> list[str]:
    return [violation.rule_id for violation in check_policy(batch, POLICY)]


def test_batch_that_follows_the_policy_passes() -> None:
    assert check_policy(build_batch(), POLICY) == ()


def test_wider_scope_fails_p1() -> None:
    batch = build_batch(scope=frozenset({EFFECT, TOY + "Ability"}))

    violations = check_policy(batch, POLICY)

    assert [violation.rule_id for violation in violations] == ["P1"]
    assert "Ability" in violations[0].message


def test_stub_creation_fails_p2() -> None:
    assert list_rule_ids(build_batch(can_create_stubs=True)) == ["P2"]


def test_unlisted_owner_property_fails_p3() -> None:
    batch = build_batch(properties=VALID_PROPERTIES | {TOY + "power": 90})

    violations = check_policy(batch, POLICY)

    assert [violation.rule_id for violation in violations] == ["P3"]
    assert violations[0].property_name == "power"
    assert violations[0].value == "90"


def test_missing_required_property_fails_p4() -> None:
    violations = check_policy(build_batch(properties={LEVEL: 2}), POLICY)

    assert [violation.rule_id for violation in violations] == ["P4"]
    assert violations[0].property_name == "extractedBy"


def test_unlisted_edge_property_fails_p5() -> None:
    assert list_rule_ids(build_batch(tag_property=TOY + "hasColor")) == ["P5"]


@pytest.mark.parametrize(
    "change",
    [
        {"property_iri": HAS_TAG},
        {"direction": ClaimDirection.IN},
        {"other_class_iri": None, "owns_other_end": False},
        {"owns_other_end": False},
    ],
    ids=["property", "direction", "other-class", "ownership"],
)
def test_claim_with_no_matching_rule_fails_p6(change: dict[str, object]) -> None:
    claim = VALID_CLAIM.model_copy(update=change)

    assert list_rule_ids(build_batch(claims=(claim,))) == ["P6"]


def test_policy_dumps_its_sets_in_sorted_order() -> None:
    dumped = POLICY.model_dump(mode="json")

    assert dumped["owner_properties"] == [EXTRACTED_BY, LEVEL]
    assert WritePolicy.model_validate(dumped) == POLICY
