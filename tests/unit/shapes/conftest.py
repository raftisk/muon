"""Fixtures for the pkmn SHACL shape tests.

`validate_fixture` is the only entry point for running shapes against a Turtle snippet.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from pyshacl import validate
from rdflib import Graph, Literal
from rdflib.namespace import SH

ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"
SHAPES_PATH = ONTOLOGY_DIR / "shapes" / "pkmn.shacl.ttl"
CORE_TTL_PATH = ONTOLOGY_DIR / "core" / "muon.ttl"
PKMN_TTL_PATH = ONTOLOGY_DIR / "universes" / "pkmn.ttl"
ABOX_TTL_PATH = ONTOLOGY_DIR / "individuals" / "pkmn.ttl"
TURTLE_FORMAT = "turtle"

FIXTURE_PREFIXES = """\
@prefix pkmn: <https://muon.dev/ns/pkmn#> .
@prefix mo: <https://muon.dev/ns/core#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix ex: <https://example.org/fixture#> .
"""


@dataclass(frozen=True)
class ShapeReport:
    conforms: bool
    rule_ids: frozenset[str]
    messages: tuple[str, ...]


@pytest.fixture(scope="session")
def shapes_graph() -> Graph:
    return Graph().parse(SHAPES_PATH, format=TURTLE_FORMAT)


@pytest.fixture(scope="session")
def tbox_graph() -> Graph:
    graph = Graph()
    graph.parse(CORE_TTL_PATH, format=TURTLE_FORMAT)
    graph.parse(PKMN_TTL_PATH, format=TURTLE_FORMAT)
    return graph


@pytest.fixture(scope="session")
def abox_graph() -> Graph:
    return Graph().parse(ABOX_TTL_PATH, format=TURTLE_FORMAT)


def read_rule_ids(results_graph: Graph, shapes: Graph) -> frozenset[str]:
    """Map each result's source shape to its `sh:name`, the rule id."""
    rule_ids: set[str] = set()
    for source_shape in results_graph.objects(predicate=SH.sourceShape):
        for name in shapes.objects(source_shape, SH.name):
            rule_ids.add(str(name))
    return frozenset(rule_ids)


def read_messages(results_graph: Graph) -> tuple[str, ...]:
    messages = results_graph.objects(predicate=SH.resultMessage)
    return tuple(sorted(str(message) for message in messages if isinstance(message, Literal)))


@pytest.fixture
def validate_fixture(shapes_graph: Graph, tbox_graph: Graph):
    def run(data_ttl: str) -> ShapeReport:
        data_graph = Graph().parse(data=FIXTURE_PREFIXES + data_ttl, format=TURTLE_FORMAT)
        conforms, results_graph, _ = validate(
            data_graph,
            shacl_graph=shapes_graph,
            ont_graph=tbox_graph,
            advanced=True,
            inference="none",
        )
        return ShapeReport(
            conforms=bool(conforms),
            rule_ids=read_rule_ids(results_graph, shapes_graph),
            messages=read_messages(results_graph),
        )

    return run
