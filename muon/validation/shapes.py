"""Shape loading, selection by batch scope and the pyshacl run."""

from collections.abc import Mapping
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, SH
from rdflib.plugins.parsers.notation3 import BadSyntax
from rdflib.term import Node

from muon.ingestion.batch import NodeRef
from muon.ontology.model import TURTLE_FORMAT, OntologyError, OntologyModel, extract_local_name
from muon.validation.projection import ProjectedBatch
from muon.validation.report import Violation

SHAPES_DIR_NAME = "shapes"
SHAPES_SUFFIX = ".shacl.ttl"
SCOPE_PROPERTY_NAME = "scope"
ALWAYS_VALUE_NAME = "Always"
UNKNOWN_RULE_ID = "SHACL"


def load_shapes(ontology_dir: Path, universe: str) -> Graph:
    """Parse `shapes/<universe>.shacl.ttl` under `ontology_dir`."""
    path = ontology_dir / SHAPES_DIR_NAME / f"{universe}{SHAPES_SUFFIX}"
    if not path.is_file():
        raise OntologyError(f"Missing shapes file {path}")
    try:
        return Graph().parse(path, format=TURTLE_FORMAT)
    except BadSyntax as error:
        raise OntologyError(f"Failed to parse {path}: {error}") from error


def select_shapes(
    shapes: Graph, scope: frozenset[str], model: OntologyModel, shapes_namespace: str
) -> Graph:
    """Keep the shapes that apply to a batch of `scope`, plus the prefix declarations.

    A shape applies when one of its scope values is `Always`, is in `scope` or is a
    subclass of a class in `scope`. The result holds the concise bounded description of
    every kept shape and of every subject with `sh:declare`, which the SPARQL constraints
    need for their prefixes.
    """
    scope_property = URIRef(shapes_namespace + SCOPE_PROPERTY_NAME)
    always_value = shapes_namespace + ALWAYS_VALUE_NAME
    selected = Graph()
    for shape in set(shapes.subjects(scope_property, None)):
        values = {str(value) for value in shapes.objects(shape, scope_property)}
        if any(is_in_scope(value, scope, model, always_value) for value in values):
            selected += shapes.cbd(shape)
    for declaring_subject in set(shapes.subjects(SH.declare, None)):
        selected += shapes.cbd(declaring_subject)
    return selected


def is_in_scope(value: str, scope: frozenset[str], model: OntologyModel, always_value: str) -> bool:
    if value == always_value or value in scope:
        return True
    return bool(scope & model.ancestors.get(value, frozenset()))


def run_shapes(projected: ProjectedBatch, shapes: Graph, tbox: Graph) -> tuple[Violation, ...]:
    """Validate the projected graph against `shapes`, with `tbox` as the ontology graph."""
    _, results_graph, _ = validate(
        projected.graph,
        shacl_graph=shapes,
        ont_graph=tbox,
        advanced=True,
        inference="none",
    )
    results = sorted(results_graph.subjects(RDF.type, SH.ValidationResult), key=str)
    return tuple(
        read_result(results_graph, result, shapes, projected.nodes_by_term) for result in results
    )


def read_result(
    results_graph: Graph, result: Node, shapes: Graph, nodes_by_term: Mapping[Node, NodeRef]
) -> Violation:
    source_shape = results_graph.value(result, SH.sourceShape)
    rule_id = read_rule_id(shapes, source_shape)
    focus_node = results_graph.value(result, SH.focusNode)
    path = results_graph.value(result, SH.resultPath)
    value = results_graph.value(result, SH.value)
    message = results_graph.value(result, SH.resultMessage)
    return Violation(
        rule_id=rule_id,
        node=nodes_by_term.get(focus_node) if focus_node is not None else None,
        property_name=extract_local_name(str(path)) if isinstance(path, URIRef) else None,
        value=str(value) if value is not None else None,
        message=str(message) if isinstance(message, Literal) else rule_id,
    )


def read_rule_id(shapes: Graph, source_shape: Node | None) -> str:
    """Return the `sh:name` of the shape that produced a result, or its IRI when it has none."""
    if source_shape is None:
        return UNKNOWN_RULE_ID
    name = shapes.value(source_shape, SH.name)
    return str(name) if name is not None else str(source_shape)
