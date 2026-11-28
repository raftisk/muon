"""The write policy of a job: what 1 job may write, checked on a batch before the Validator.

A `GraphBatch` states what it owns. A `WritePolicy` states what its producer may own, so
a producer bug that widens a batch fails before anything reads the graph. The rule ids are
P1 to P6, 1 function each.
"""

from pydantic import BaseModel, ConfigDict, field_serializer

from muon.ingestion.batch import ClaimDirection, EdgeClaim, GraphBatch, NodeRecord
from muon.ontology.model import extract_local_name
from muon.validation.report import Violation

RULE_POLICY_SCOPE = "P1"
RULE_POLICY_STUBS = "P2"
RULE_POLICY_PROPERTY = "P3"
RULE_POLICY_REQUIRED = "P4"
RULE_POLICY_EDGE = "P5"
RULE_POLICY_CLAIM = "P6"


class ClaimRule(BaseModel):
    """1 claim shape a job may make. A claim matches when all 4 fields are equal."""

    model_config = ConfigDict(frozen=True)

    property_iri: str
    direction: ClaimDirection
    other_class_iri: str | None
    owns_other_end: bool

    def matches(self, claim: EdgeClaim) -> bool:
        return (
            claim.property_iri == self.property_iri
            and claim.direction == self.direction
            and claim.other_class_iri == self.other_class_iri
            and claim.owns_other_end == self.owns_other_end
        )


class WritePolicy(BaseModel):
    """What the job `job` may write.

    `owned_classes` is the exact scope of its batches. `owner_properties` are the datatype
    properties an owner node may carry, `required_owner_properties` those every owner
    node must carry, and `edge_properties` the object properties of its edges.
    """

    model_config = ConfigDict(frozen=True)

    job: str
    owned_classes: frozenset[str]
    owner_properties: frozenset[str]
    required_owner_properties: frozenset[str]
    edge_properties: frozenset[str]
    claim_rules: tuple[ClaimRule, ...]

    @field_serializer(
        "owned_classes", "owner_properties", "required_owner_properties", "edge_properties"
    )
    def serialize_sorted(self, iris: frozenset[str]) -> list[str]:
        """Dump a set in sorted order: a frozenset iterates in hash order."""
        return sorted(iris)


def check_policy(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    """Return 1 violation per breach of `policy` by `batch`, in rule order."""
    return (
        *check_policy_scope(batch, policy),
        *check_stub_creation(batch, policy),
        *check_owner_properties(batch, policy),
        *check_required_properties(batch, policy),
        *check_edge_properties(batch, policy),
        *check_claims(batch, policy),
    )


def check_policy_scope(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    if batch.scope == policy.owned_classes:
        return ()
    message = (
        f"Job {policy.job!r} owns {format_names(policy.owned_classes)}, "
        f"but the batch scope is {format_names(batch.scope)}"
    )
    return (Violation(rule_id=RULE_POLICY_SCOPE, value=format_names(batch.scope), message=message),)


def check_stub_creation(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    if not batch.can_create_stubs:
        return ()
    message = f"Job {policy.job!r} may not create stubs, but the batch sets can_create_stubs"
    return (Violation(rule_id=RULE_POLICY_STUBS, value=str(True), message=message),)


def check_owner_properties(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    return tuple(
        build_property_violation(RULE_POLICY_PROPERTY, node, property_iri, policy)
        for node in batch.owner_nodes
        for property_iri in sorted(set(node.properties) - policy.owner_properties)
    )


def check_required_properties(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    return tuple(
        build_property_violation(RULE_POLICY_REQUIRED, node, property_iri, policy)
        for node in batch.owner_nodes
        for property_iri in sorted(policy.required_owner_properties - set(node.properties))
    )


def build_property_violation(
    rule_id: str, node: NodeRecord, property_iri: str, policy: WritePolicy
) -> Violation:
    """A P3 violation for a property `node` may not carry, or P4 for one it lacks."""
    name = extract_local_name(property_iri)
    value = node.properties.get(property_iri)
    problem = (
        f"may not set {name} to {value!r}"
        if rule_id == RULE_POLICY_PROPERTY
        else f"needs {name} on every owner node"
    )
    return Violation(
        rule_id=rule_id,
        node=node.node_ref,
        property_name=name,
        value=None if value is None else str(value),
        message=f"Job {policy.job!r} {problem}: {extract_local_name(node.class_iri)} {node.id!r}",
    )


def check_edge_properties(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    return tuple(
        Violation(
            rule_id=RULE_POLICY_EDGE,
            node=edge.start,
            property_name=extract_local_name(edge.property_iri),
            value=edge.end.id,
            message=f"Job {policy.job!r} may not write "
            f"{extract_local_name(edge.property_iri)} edges: "
            f"{edge.start.id!r} to {edge.end.id!r}",
        )
        for edge in batch.edges
        if edge.property_iri not in policy.edge_properties
    )


def check_claims(batch: GraphBatch, policy: WritePolicy) -> tuple[Violation, ...]:
    return tuple(
        build_claim_violation(claim, policy)
        for claim in batch.claims
        if not any(rule.matches(claim) for rule in policy.claim_rules)
    )


def build_claim_violation(claim: EdgeClaim, policy: WritePolicy) -> Violation:
    name = extract_local_name(claim.property_iri)
    other = extract_local_name(claim.other_class_iri) if claim.other_class_iri else "any class"
    return Violation(
        rule_id=RULE_POLICY_CLAIM,
        node=claim.anchor,
        property_name=name,
        message=f"Job {policy.job!r} has no claim rule for {name} at {claim.anchor.id!r} "
        f"(direction {claim.direction.value}, other class {other}, "
        f"owns_other_end {claim.owns_other_end})",
    )


def format_names(iris: frozenset[str]) -> str:
    return ", ".join(sorted(extract_local_name(iri) for iri in iris)) or "no class"
