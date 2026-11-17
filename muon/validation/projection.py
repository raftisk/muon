"""Projects a `GraphBatch` to an rdflib graph that the shapes validate.

A node becomes the IRI `<namespace>node/<class local name>/<id>`. An edge becomes 1
triple, plus a reified node when it carries edge properties the projection table does
not consume.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.term import Node

from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord, NodeRef, PropertyValue
from muon.ontology.model import OntologyModel, extract_local_name
from muon.plugins import UniverseManifest

NODE_PATH = "node/"
ID_PROPERTY_NAME = "id"
BOOLEAN_TRUE = "true"
BOOLEAN_FALSE = "false"

Triple = tuple[Node, URIRef, Node]


@dataclass(frozen=True)
class ProjectionRules:
    """The universe namespace and the edge property to sub-property table.

    `sub_properties` is keyed by (property IRI, edge property name) and maps the edge
    property value as a string to a sub-property IRI.
    """

    namespace: str
    sub_properties: Mapping[tuple[str, str], Mapping[str, str]]


@dataclass(frozen=True)
class ProjectedBatch:
    """The projected graph and the node each term stands for.

    A reified edge node maps to the start node of its edge.
    """

    graph: Graph
    nodes_by_term: Mapping[Node, NodeRef]


@dataclass(frozen=True)
class EdgeProjection:
    triples: tuple[Triple, ...]
    reified_node: BNode | None


def build_projection_rules(manifest: UniverseManifest) -> ProjectionRules:
    """Resolve the manifest's projection table against its namespace."""
    namespace = manifest.namespace
    return ProjectionRules(
        namespace=namespace,
        sub_properties={
            (namespace + entry.property, entry.edge_property): {
                value: namespace + local_name for value, local_name in entry.values.items()
            }
            for entry in manifest.projection
        },
    )


def project_batch(
    batch: GraphBatch, model: OntologyModel, rules: ProjectionRules
) -> ProjectedBatch:
    """Build the RDF graph of `batch` and the table from its terms back to node keys."""
    graph = Graph()
    nodes_by_term: dict[Node, NodeRef] = {}
    for node in batch.nodes:
        nodes_by_term[build_node_term(node.node_ref, rules)] = node.node_ref
        for triple in project_node(node, model, rules):
            graph.add(triple)
    for edge in batch.edges:
        projection = project_edge(edge, model, rules)
        for triple in projection.triples:
            graph.add(triple)
        if projection.reified_node is not None:
            nodes_by_term[projection.reified_node] = edge.start
    return ProjectedBatch(graph=graph, nodes_by_term=nodes_by_term)


def build_node_term(node_ref: NodeRef, rules: ProjectionRules) -> URIRef:
    class_name = extract_local_name(node_ref.class_iri)
    return URIRef(f"{rules.namespace}{NODE_PATH}{class_name}/{node_ref.id}")


def project_node(node: NodeRecord, model: OntologyModel, rules: ProjectionRules) -> list[Triple]:
    term = build_node_term(node.node_ref, rules)
    triples: list[Triple] = [
        (term, RDF.type, URIRef(node.class_iri)),
        (term, URIRef(rules.namespace + ID_PROPERTY_NAME), Literal(node.id)),
        (term, RDFS.label, Literal(node.name)),
    ]
    for property_iri, value in node.properties.items():
        datatype_iri = model.datatype_ranges.get(property_iri)
        triples.extend(
            (term, URIRef(property_iri), literal)
            for literal in project_literal(value, datatype_iri)
        )
    return triples


def project_literal(value: PropertyValue, datatype_iri: str | None) -> list[Literal]:
    """Return 1 literal per value, typed by `datatype_iri` or else by the Python type.

    An `xsd:string` value becomes a plain literal, the form the shapes' `sh:in` lists and
    `sh:hasValue` values use; rdflib does not equate it with a typed `xsd:string`.
    """
    values = value if isinstance(value, tuple) else (value,)
    return [
        build_literal(format_lexical_form(element), datatype_iri or infer_datatype(element))
        for element in values
    ]


def build_literal(lexical_form: str, datatype_iri: str) -> Literal:
    if datatype_iri == str(XSD.string):
        return Literal(lexical_form)
    return Literal(lexical_form, datatype=URIRef(datatype_iri))


def infer_datatype(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return str(XSD.boolean)
    if isinstance(value, int):
        return str(XSD.integer)
    if isinstance(value, float):
        return str(XSD.decimal)
    return str(XSD.string)


def format_lexical_form(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return BOOLEAN_TRUE if value else BOOLEAN_FALSE
    if isinstance(value, float):
        return format(Decimal(str(value)), "f")
    return str(value)


def project_edge(edge: EdgeRecord, model: OntologyModel, rules: ProjectionRules) -> EdgeProjection:
    """Project an edge, replacing its property when the projection table maps an edge property.

    The consumed edge property is dropped. Any other edge property goes onto a reified node.
    """
    property_iri, remaining = apply_projection_table(edge, rules)
    start = build_node_term(edge.start, rules)
    end = build_node_term(edge.end, rules)
    triples: list[Triple] = [(start, URIRef(property_iri), end)]
    if not remaining:
        return EdgeProjection(triples=tuple(triples), reified_node=None)

    reified_node = BNode()
    triples.extend(
        [
            (reified_node, RDF.subject, start),
            (reified_node, RDF.predicate, URIRef(property_iri)),
            (reified_node, RDF.object, end),
        ]
    )
    for name, value in remaining.items():
        edge_property = rules.namespace + name
        datatype_iri = model.datatype_ranges.get(edge_property)
        triples.extend(
            (reified_node, URIRef(edge_property), literal)
            for literal in project_literal(value, datatype_iri)
        )
    return EdgeProjection(triples=tuple(triples), reified_node=reified_node)


def apply_projection_table(
    edge: EdgeRecord, rules: ProjectionRules
) -> tuple[str, Mapping[str, PropertyValue]]:
    """Return the edge's projected property IRI and its edge properties left over."""
    for name, value in edge.properties.items():
        mapped = rules.sub_properties.get((edge.property_iri, name), {}).get(str(value))
        if mapped is not None:
            remaining = {key: item for key, item in edge.properties.items() if key != name}
            return mapped, remaining
    return edge.property_iri, edge.properties
