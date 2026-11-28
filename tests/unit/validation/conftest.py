"""Fixtures for the Validator tests: the pkmn T-Box model, its A-Box and a batch builder."""

from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from rdflib import Graph

from muon.config import DEFAULT_UNIVERSES_DIR
from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord, NodeRef, PropertyValue
from muon.ingestion.seed import read_abox
from muon.ontology.model import OntologyModel, load_ontology_model, load_tbox
from muon.plugins import discover_universes
from muon.validation.projection import ProjectionRules, build_projection_rules
from muon.validation.shapes import load_shapes
from muon.validation.validator import ValidatorInputs, build_validator
from tests.unit.ingestion.recording_client import RecordingGraphClient

ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"
UNIVERSE = "pkmn"
PKMN = "https://muon.dev/ns/pkmn#"
MO = "https://muon.dev/ns/core#"


def build_node(
    class_iri: str, node_id: str, properties: Mapping[str, PropertyValue] | None = None
) -> NodeRecord:
    """A node whose name is its id."""
    return NodeRecord(class_iri=class_iri, id=node_id, name=node_id, properties=properties or {})


def build_edge(
    property_iri: str,
    start: NodeRecord,
    end: NodeRecord,
    properties: Mapping[str, PropertyValue] | None = None,
) -> EdgeRecord:
    return EdgeRecord(
        property_iri=property_iri,
        start=start.node_ref,
        end=end.node_ref,
        properties=properties or {},
    )


def build_test_batch(
    scope: Iterable[str], nodes: Iterable[NodeRecord], edges: Iterable[EdgeRecord] = ()
) -> GraphBatch:
    return GraphBatch(
        universe=UNIVERSE, scope=frozenset(scope), nodes=tuple(nodes), edges=tuple(edges)
    )


def build_ref(class_iri: str, node_id: str) -> NodeRef:
    return NodeRef(class_iri=class_iri, id=node_id)


@pytest.fixture(scope="session")
def pkmn_model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, UNIVERSE)


@pytest.fixture(scope="session")
def pkmn_abox(pkmn_model: OntologyModel) -> GraphBatch:
    return read_abox(pkmn_model, ONTOLOGY_DIR)


@pytest.fixture(scope="session")
def pkmn_tbox() -> Graph:
    return load_tbox(ONTOLOGY_DIR, UNIVERSE)


@pytest.fixture(scope="session")
def pkmn_shapes() -> Graph:
    return load_shapes(ONTOLOGY_DIR, UNIVERSE)


@pytest.fixture(scope="session")
def pkmn_rules(pkmn_abox: GraphBatch) -> ProjectionRules:
    plugin = discover_universes(DEFAULT_UNIVERSES_DIR, ONTOLOGY_DIR).get(UNIVERSE)
    return build_projection_rules(plugin.manifest, pkmn_abox)


@pytest.fixture(scope="session")
def pkmn_validator_inputs() -> ValidatorInputs:
    plugin = discover_universes(DEFAULT_UNIVERSES_DIR, ONTOLOGY_DIR).get(UNIVERSE)
    return build_validator(RecordingGraphClient(), ONTOLOGY_DIR, plugin).inputs
