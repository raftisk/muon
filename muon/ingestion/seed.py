"""Seed path: a universe's hand-written A-Box, loaded into the graph.

`seed_universe` reads the T-Box and `individuals/<prefix>.ttl`, then writes the
batch in sync mode, so the graph mirrors the file after every run.

`read_abox` maps the A-Box with no knowledge of the universe:
every typed subject becomes a node, every datatype literal a node property and
every object property value an edge. It collects every problem into one
`ABoxError`, so nothing reaches the graph from a file it cannot map in full.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, XSD
from rdflib.plugins.parsers.notation3 import BadSyntax
from rdflib.term import Node

from muon.graph import GraphClient
from muon.ingestion.batch import (
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
    PropertyValue,
    build_node_id,
)
from muon.ingestion.runner import Runner, RunReport, WriteMode
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import TTL_SUFFIX, TURTLE_FORMAT, OntologyModel, load_ontology_model

INDIVIDUALS_DIR_NAME = "individuals"
LABEL_LANGUAGE = "en"
IGNORED_PREDICATES = frozenset({str(RDFS.comment), str(SKOS.definition)})
RESERVED_PREDICATES = frozenset({str(RDF.type), str(RDFS.label)})
LITERAL_CONVERTERS: Mapping[URIRef | None, Callable[[Literal], PropertyValue]] = {
    None: str,
    XSD.string: str,
    RDF.langString: str,
    XSD.integer: lambda literal: int(literal.toPython()),
    XSD.decimal: lambda literal: float(literal.toPython()),
    XSD.boolean: lambda literal: bool(literal.toPython()),
}


class ABoxError(Exception):
    """An A-Box the seed cannot map. `problems` holds every problem found."""

    def __init__(self, source: str, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        lines = [f"Invalid A-Box {source}:", *(f"- {problem}" for problem in self.problems)]
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class Individual:
    iri: str
    node_ref: NodeRef
    name: str


async def seed_universe(client: GraphClient, ontology_dir: Path, universe: str) -> RunReport:
    """Load the A-Box of `universe` and write it in sync mode.

    Both ttl reads finish before the first database call, so an invalid file
    leaves the graph untouched.
    """
    model = load_ontology_model(ontology_dir, universe)
    batch = read_abox(model, ontology_dir)
    return await Runner(client, LabelMaterializer(model)).write(batch, WriteMode.SYNC)


def read_abox(model: OntologyModel, ontology_dir: Path) -> GraphBatch:
    """Read `individuals/<universe>.ttl` under `ontology_dir` into a batch."""
    path = ontology_dir / INDIVIDUALS_DIR_NAME / f"{model.universe}{TTL_SUFFIX}"
    if not path.is_file():
        raise ABoxError(str(path), ["the file does not exist"])
    try:
        abox = Graph().parse(path, format=TURTLE_FORMAT)
    except BadSyntax as error:
        raise ABoxError(str(path), [f"the file does not parse: {error}"]) from error
    return build_abox_batch(model, abox, str(path))


def build_abox_batch(model: OntologyModel, abox: Graph, source: str) -> GraphBatch:
    """Map a parsed A-Box to a batch whose scope is the set of asserted classes.

    Raises `ABoxError` naming `source` and listing every problem.
    """
    problems: list[str] = []
    individuals: dict[str, Individual] = {}
    properties_by_iri: dict[str, dict[str, PropertyValue]] = {}
    for subject in sorted(set(abox.subjects()), key=str):
        individual, individual_problems = read_individual(model, abox, subject)
        properties, property_problems = read_node_properties(model, abox, subject)
        problems.extend(individual_problems + property_problems)
        if individual is not None:
            individuals[individual.iri] = individual
            properties_by_iri[individual.iri] = properties

    problems.extend(find_duplicate_keys(individuals.values()))
    edges, edge_problems = read_edges(model, abox, individuals)
    problems.extend(edge_problems)
    if problems:
        raise ABoxError(source, problems)

    nodes = [
        NodeRecord(
            class_iri=individual.node_ref.class_iri,
            id=individual.node_ref.id,
            name=individual.name,
            properties=properties_by_iri[individual.iri],
        )
        for individual in individuals.values()
    ]
    return GraphBatch(
        universe=model.universe,
        scope=frozenset(node.class_iri for node in nodes),
        nodes=tuple(sorted(nodes, key=lambda node: (node.class_iri, node.id))),
        edges=tuple(sorted(edges, key=build_edge_sort_key)),
    )


def read_individual(
    model: OntologyModel, abox: Graph, subject: Node
) -> tuple[Individual | None, list[str]]:
    """Read the class, name and `id` of a subject, or the problems that block them."""
    if not isinstance(subject, URIRef):
        return None, [f"{subject}: the subject is not an IRI"]

    problems: list[str] = []
    classes = sorted(str(class_iri) for class_iri in abox.objects(subject, RDF.type))
    if len(classes) != 1:
        problems.append(f"{subject}: expected 1 rdf:type, found {len(classes)} {classes}")
    problems.extend(
        f"{subject}: class {class_iri} is not declared in the T-Box"
        for class_iri in classes
        if class_iri not in model.classes
    )

    names = [
        str(label)
        for label in abox.objects(subject, RDFS.label)
        if isinstance(label, Literal) and label.language == LABEL_LANGUAGE
    ]
    if len(names) != 1:
        problems.append(
            f"{subject}: expected 1 rdfs:label in @{LABEL_LANGUAGE}, found {len(names)}"
        )
    elif not build_node_id(names[0]):
        problems.append(f"{subject}: the label {names[0]!r} gives an empty id")

    if problems:
        return None, problems
    name = names[0]
    node_ref = NodeRef(class_iri=classes[0], id=build_node_id(name))
    return Individual(iri=str(subject), node_ref=node_ref, name=name), []


def find_duplicate_keys(individuals: Iterable[Individual]) -> list[str]:
    iris_by_ref: dict[NodeRef, list[str]] = {}
    for individual in individuals:
        iris_by_ref.setdefault(individual.node_ref, []).append(individual.iri)
    return [
        f"{node_ref.class_iri} id {node_ref.id!r} is shared by {sorted(iris)}"
        for node_ref, iris in iris_by_ref.items()
        if len(iris) > 1
    ]


def read_node_properties(
    model: OntologyModel, abox: Graph, subject: Node
) -> tuple[dict[str, PropertyValue], list[str]]:
    """Read the datatype property values of a subject and flag undeclared predicates.

    Object property values are left to `read_edges`.
    """
    skipped_predicates = RESERVED_PREDICATES | IGNORED_PREDICATES | model.object_properties
    problems: list[str] = []
    values_by_property: dict[str, list[PropertyValue]] = {}
    for predicate, value in abox.predicate_objects(subject):
        predicate_iri = str(predicate)
        if predicate_iri in skipped_predicates:
            continue
        if predicate_iri not in model.datatype_properties:
            problems.append(
                f"{subject} {predicate_iri}: the predicate is not declared in the T-Box"
            )
            continue
        converted = convert_literal(value) if isinstance(value, Literal) else None
        if converted is None:
            problems.append(f"{subject} {predicate_iri}: unsupported value {value!r}")
            continue
        values_by_property.setdefault(predicate_iri, []).append(converted)

    properties, merge_problems = merge_property_values(str(subject), values_by_property)
    return properties, problems + merge_problems


def merge_property_values(
    subject: str, values_by_property: Mapping[str, list[PropertyValue]]
) -> tuple[dict[str, PropertyValue], list[str]]:
    """Keep single values; turn repeated strings into a sorted tuple."""
    properties: dict[str, PropertyValue] = {}
    problems: list[str] = []
    for predicate_iri, values in values_by_property.items():
        if len(values) == 1:
            properties[predicate_iri] = values[0]
        elif all(isinstance(value, str) for value in values):
            properties[predicate_iri] = tuple(sorted(str(value) for value in values))
        else:
            problems.append(
                f"{subject} {predicate_iri}: {len(values)} values, but only strings may repeat"
            )
    return properties, problems


def convert_literal(literal: Literal) -> PropertyValue | None:
    """Return the Python value of a literal, or None for an unsupported or ill-typed one."""
    converter = LITERAL_CONVERTERS.get(literal.datatype)
    if converter is None or literal.ill_typed:
        return None
    return converter(literal)


def read_edges(
    model: OntologyModel, abox: Graph, individuals: Mapping[str, Individual]
) -> tuple[list[EdgeRecord], list[str]]:
    """Read every object property value of an individual as an edge.

    A value that is a subject of the file but not a valid individual already has
    its own problems, so it gets no second message here.
    """
    subjects = {str(subject) for subject in abox.subjects()}
    edges: list[EdgeRecord] = []
    problems: list[str] = []
    for subject, predicate, value in abox:
        start = individuals.get(str(subject))
        if str(predicate) not in model.object_properties or start is None:
            continue
        end = individuals.get(str(value)) if isinstance(value, URIRef) else None
        if end is None:
            if isinstance(value, Literal) or str(value) not in subjects:
                problems.append(
                    f"{subject} {predicate}: {value!r} is not an individual of the file"
                )
            continue
        edges.append(
            EdgeRecord(property_iri=str(predicate), start=start.node_ref, end=end.node_ref)
        )
    return edges, problems


def build_edge_sort_key(edge: EdgeRecord) -> tuple[str, str, str, str, str]:
    return (edge.property_iri, edge.start.class_iri, edge.start.id, edge.end.class_iri, edge.end.id)
