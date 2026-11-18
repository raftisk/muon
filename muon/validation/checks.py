"""Gate steps 1 to 4: checks of a batch against the T-Box and the A-Box, with no graph read."""

from collections import defaultdict

from rdflib.namespace import RDFS

from muon.ingestion.batch import GraphBatch, NodeRecord, NodeRef
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, extract_local_name
from muon.validation.report import (
    RULE_ABOX,
    RULE_EDGE_TYPING,
    RULE_ID,
    RULE_IRI,
    Violation,
)

LABEL_IRI = str(RDFS.label)


def check_iris(batch: GraphBatch, model: OntologyModel) -> tuple[Violation, ...]:
    """Check that every class and property IRI of `batch` is declared in the T-Box.

    Scope classes, node classes and claim `other_class_iri` must be classes. Node
    property IRIs must be datatype properties or `rdfs:label`. Edge and claim property
    IRIs must be object properties.
    """
    violations: list[Violation] = []
    for class_iri in sorted(batch.scope - model.classes):
        violations.append(
            Violation(
                rule_id=RULE_IRI,
                value=class_iri,
                message=f"Scope class {class_iri} is not a class in the T-Box",
            )
        )
    for node in batch.nodes:
        violations.extend(find_node_iri_violations(node, model))
    for edge in batch.edges:
        if edge.property_iri not in model.object_properties:
            violations.append(
                Violation(
                    rule_id=RULE_IRI,
                    node=edge.start,
                    property_name=extract_local_name(edge.property_iri),
                    value=edge.property_iri,
                    message=f"Edge property {edge.property_iri} is not an object property "
                    f"in the T-Box",
                )
            )
    for claim in batch.claims:
        if claim.property_iri not in model.object_properties:
            violations.append(
                Violation(
                    rule_id=RULE_IRI,
                    property_name=extract_local_name(claim.property_iri),
                    value=claim.property_iri,
                    message=f"Claim property {claim.property_iri} at {claim.anchor.class_iri} "
                    f"{claim.anchor.id!r} is not an object property in the T-Box",
                )
            )
        if claim.other_class_iri is not None and claim.other_class_iri not in model.classes:
            violations.append(
                Violation(
                    rule_id=RULE_IRI,
                    property_name=extract_local_name(claim.property_iri),
                    value=claim.other_class_iri,
                    message=f"Claim other class {claim.other_class_iri} at "
                    f"{claim.anchor.class_iri} {claim.anchor.id!r} is not a class in the T-Box",
                )
            )
    return tuple(violations)


def find_node_iri_violations(node: NodeRecord, model: OntologyModel) -> list[Violation]:
    violations: list[Violation] = []
    if node.class_iri not in model.classes:
        violations.append(
            Violation(
                rule_id=RULE_IRI,
                node=node.node_ref,
                value=node.class_iri,
                message=f"Node class {node.class_iri} is not a class in the T-Box",
            )
        )
    for property_iri in node.properties:
        if property_iri in model.datatype_properties or property_iri == LABEL_IRI:
            continue
        violations.append(
            Violation(
                rule_id=RULE_IRI,
                node=node.node_ref,
                property_name=extract_local_name(property_iri),
                value=property_iri,
                message=f"Node property {property_iri} is not a datatype property in the T-Box",
            )
        )
    return violations


def check_ids(
    batch: GraphBatch, model: OntologyModel, materializer: LabelMaterializer
) -> tuple[Violation, ...]:
    """Flag 2 node records that share an `id` and a key label, or form an ancestor pair.

    Such records would be 1 Neo4j node with 2 identities. Unrelated classes may share an
    `id`: names repeat across labels by design.
    """
    refs_by_id: dict[str, list[NodeRef]] = defaultdict(list)
    for node in batch.nodes:
        refs_by_id[node.id].append(node.node_ref)

    violations: list[Violation] = []
    for node_id, refs in refs_by_id.items():
        for position, first in enumerate(refs):
            for second in refs[position + 1 :]:
                key_label = materializer.map_key_label(first.class_iri)
                if is_same_identity(first, second, model, materializer):
                    violations.append(
                        Violation(
                            rule_id=RULE_ID,
                            node=first,
                            value=node_id,
                            message=f"Nodes {first.class_iri} and {second.class_iri} share the "
                            f"id {node_id!r} and the key label {key_label!r}, or one class is "
                            f"an ancestor of the other",
                        )
                    )
    return tuple(violations)


def is_same_identity(
    first: NodeRef, second: NodeRef, model: OntologyModel, materializer: LabelMaterializer
) -> bool:
    if materializer.map_key_label(first.class_iri) == materializer.map_key_label(second.class_iri):
        return True
    return is_ancestor_pair(first.class_iri, second.class_iri, model)


def is_ancestor_pair(first_class: str, second_class: str, model: OntologyModel) -> bool:
    first_ancestors = model.ancestors.get(first_class, frozenset())
    second_ancestors = model.ancestors.get(second_class, frozenset())
    return second_class in first_ancestors or first_class in second_ancestors


def check_edge_typing(batch: GraphBatch, model: OntologyModel) -> tuple[Violation, ...]:
    """Check every edge end against the `rdfs:domain` and `rdfs:range` of its property.

    An end passes when its class or one of its ancestors is allowed. A property with no
    domain or range is not checked on that end.
    """
    violations: list[Violation] = []
    for edge in batch.edges:
        for end, allowed_by_property, role in (
            (edge.start, model.domains, "domain"),
            (edge.end, model.ranges, "range"),
        ):
            allowed = allowed_by_property.get(edge.property_iri)
            if allowed is None:
                continue
            if {end.class_iri, *model.ancestors.get(end.class_iri, frozenset())} & allowed:
                continue
            violations.append(
                Violation(
                    rule_id=RULE_EDGE_TYPING,
                    node=edge.start,
                    property_name=extract_local_name(edge.property_iri),
                    value=end.id,
                    message=f"Edge {edge.property_iri} from {edge.start.class_iri} "
                    f"{edge.start.id!r} to {edge.end.class_iri} {edge.end.id!r} breaks the "
                    f"{role}: {end.class_iri} is not one of {', '.join(sorted(allowed))}",
                )
            )
    return tuple(violations)


def check_abox_stubs(batch: GraphBatch, abox: GraphBatch) -> tuple[Violation, ...]:
    """Flag a stub of an A-Box class that matches no A-Box individual by `id`."""
    abox_classes = {node.class_iri for node in abox.nodes}
    abox_refs = {node.node_ref for node in abox.nodes}
    return tuple(
        Violation(
            rule_id=RULE_ABOX,
            node=stub.node_ref,
            value=stub.id,
            message=f"Stub {stub.class_iri} {stub.id!r} is not an individual of the A-Box",
        )
        for stub in batch.stub_nodes
        if stub.class_iri in abox_classes and stub.node_ref not in abox_refs
    )
