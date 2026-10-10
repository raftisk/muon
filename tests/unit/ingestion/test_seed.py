from pathlib import Path

import pytest
from rdflib import Graph

from muon.ingestion.batch import GraphBatch, NodeRef, PropertyValue
from muon.ingestion.runner import MERGE_EDGES_SYNC, MERGE_NODES_SYNC, PRUNE_EDGES, PRUNE_NODES
from muon.ingestion.seed import ABoxError, build_abox_batch, read_abox, seed_universe
from muon.ontology.model import OntologyModel, load_ontology_model
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.unit.ingestion.recording_client import RecordingGraphClient

REAL_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"
CORE = "https://muon.dev/ns/core#"
PKMN = "https://muon.dev/ns/pkmn#"
TOY = "https://muon.dev/ns/muontoy#"
REAL_NODE_COUNT = 235
REAL_EDGE_COUNT = 364
REAL_CLASS_COUNT = 19
INLINE_SOURCE = "inline.ttl"
TOY_PREFIXES = """\
@prefix muontoy: <https://muon.dev/ns/muontoy#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
"""


@pytest.fixture(scope="module")
def toy_ontology_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return copy_toy_ontology(tmp_path_factory.mktemp("ontology"))


@pytest.fixture(scope="module")
def toy_model(toy_ontology_dir: Path) -> OntologyModel:
    return load_ontology_model(toy_ontology_dir, TOY_UNIVERSE)


@pytest.fixture(scope="module")
def real_batch() -> GraphBatch:
    model = load_ontology_model(REAL_ONTOLOGY_DIR, "pkmn")
    return read_abox(model, REAL_ONTOLOGY_DIR)


def read_inline(model: OntologyModel, body: str) -> GraphBatch:
    abox = Graph().parse(data=TOY_PREFIXES + body, format="turtle")
    return build_abox_batch(model, abox, INLINE_SOURCE)


def test_real_abox_counts(real_batch: GraphBatch) -> None:
    assert len(real_batch.nodes) == REAL_NODE_COUNT
    assert len(real_batch.edges) == REAL_EDGE_COUNT
    assert len(real_batch.scope) == REAL_CLASS_COUNT
    assert real_batch.universe == "pkmn"


def test_real_abox_universe_node(real_batch: GraphBatch) -> None:
    node = next(node for node in real_batch.nodes if node.id == "pokemon")

    assert node.class_iri == "https://muon.dev/ns/core#Universe"
    assert node.name == "Pokémon"


def test_real_abox_alias_and_key(real_batch: GraphBatch) -> None:
    stealth_rock = next(node for node in real_batch.nodes if node.id == "stealth-rock")
    escape = NodeRef(class_iri=PKMN + "VolatileCondition", id="cant-escape")

    assert stealth_rock.properties[CORE + "aliases"] == ("Pointed Stones",)
    assert escape in {node.node_ref for node in real_batch.nodes}


def test_real_abox_type_chart_edges(real_batch: GraphBatch) -> None:
    fire = NodeRef(class_iri=PKMN + "Type", id="fire")
    targets = {
        edge.end.id
        for edge in real_batch.edges
        if edge.start == fire and edge.property_iri == PKMN + "superEffectiveAgainst"
    }

    assert targets == {"grass", "ice", "bug", "steel"}


def test_toy_abox_properties(toy_model: OntologyModel, toy_ontology_dir: Path) -> None:
    batch = read_abox(toy_model, toy_ontology_dir)
    nodes = {node.id: node for node in batch.nodes}

    assert len(batch.nodes) == 6
    assert len(batch.edges) == 5
    assert nodes["alpha"].properties == {
        TOY + "weight": 1.5,
        TOY + "serial": 7,
        TOY + "isBroken": False,
        TOY + "nickname": ("a-one", "a-prime"),
    }
    assert nodes["beta-unit"].name == "Béta Unit"
    assert batch.scope == {TOY + "ToyGadget", TOY + "ToyColor"}


def test_property_types_are_exact(toy_model: OntologyModel, toy_ontology_dir: Path) -> None:
    alpha = next(
        node for node in read_abox(toy_model, toy_ontology_dir).nodes if node.id == "alpha"
    )

    assert type(alpha.properties[TOY + "weight"]) is float
    assert type(alpha.properties[TOY + "serial"]) is int
    assert type(alpha.properties[TOY + "isBroken"]) is bool


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ('muontoy:x a muontoy:Unknown ; rdfs:label "X"@en .', "is not declared in the T-Box"),
        ('muontoy:x rdfs:label "X"@en .', "expected 1 rdf:type, found 0"),
        (
            'muontoy:x a muontoy:ToyGadget, muontoy:ToyColor ; rdfs:label "X"@en .',
            "expected 1 rdf:type, found 2",
        ),
        ("muontoy:x a muontoy:ToyGadget .", "expected 1 rdfs:label in @en, found 0"),
        ('muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en, "Y"@en .', "found 2"),
        ('muontoy:x a muontoy:ToyGadget ; rdfs:label "!!!"@en .', "gives an empty id"),
        (
            'muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en ; muontoy:unknown 3 .',
            "muontoy#unknown: the predicate is not declared",
        ),
        (
            'muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en .\n'
            "muontoy:x muontoy:hasColor muontoy:nowhere .",
            "is not an individual of the file",
        ),
        (
            'muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en ; muontoy:serial 1, 2 .',
            "only strings may repeat",
        ),
        (
            'muontoy:x a muontoy:ToyGadget ; rdfs:label "Beta Unit"@en .\n'
            'muontoy:y a muontoy:ToyGadget ; rdfs:label "Béta Unit"@en .',
            "id 'beta-unit' is shared by",
        ),
    ],
)
def test_invalid_abox_raises(toy_model: OntologyModel, body: str, fragment: str) -> None:
    with pytest.raises(ABoxError) as error_info:
        read_inline(toy_model, body)

    assert fragment in str(error_info.value)
    assert INLINE_SOURCE in str(error_info.value)


def test_all_problems_in_one_error(toy_model: OntologyModel) -> None:
    body = (
        "muontoy:x a muontoy:ToyGadget .\n"
        'muontoy:y a muontoy:Unknown ; rdfs:label "Y"@en .\n'
        'muontoy:z a muontoy:ToyGadget ; rdfs:label "Z"@en ; muontoy:unknown 3 .\n'
    )

    with pytest.raises(ABoxError) as error_info:
        read_inline(toy_model, body)

    assert len(error_info.value.problems) == 3


def read_inline_aliases(model: OntologyModel, aliases: str) -> PropertyValue:
    body = (
        "@prefix mo: <https://muon.dev/ns/core#> .\n"
        f'muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en ; mo:aliases {aliases} .'
    )
    return read_inline(model, body).nodes[0].properties[CORE + "aliases"]


def test_single_alias_loads_as_a_tuple(toy_model: OntologyModel) -> None:
    assert read_inline_aliases(toy_model, '"one"@en') == ("one",)


def test_two_aliases_load_sorted(toy_model: OntologyModel) -> None:
    assert read_inline_aliases(toy_model, '"two"@en, "one"@en') == ("one", "two")


def test_ignored_predicates_are_not_properties(toy_model: OntologyModel) -> None:
    batch = read_inline(
        toy_model,
        'muontoy:x a muontoy:ToyGadget ; rdfs:label "X"@en ; '
        'rdfs:comment "note" ; skos:definition "definition" .',
    )

    assert batch.nodes[0].properties == {}


def test_missing_abox_file_names_path(toy_model: OntologyModel, tmp_path: Path) -> None:
    with pytest.raises(ABoxError) as error_info:
        read_abox(toy_model, tmp_path)

    assert str(tmp_path / "individuals" / f"{TOY_UNIVERSE}.ttl") in str(error_info.value)


async def test_seed_universe_writes_constraints_then_batch(tmp_path: Path) -> None:
    ontology_dir = copy_toy_ontology(tmp_path)
    client = RecordingGraphClient()

    report = await seed_universe(client, ontology_dir, TOY_UNIVERSE)

    assert len(client.writes) == 2
    assert [statement.query for statement in client.batches[0]] == [
        PRUNE_NODES,
        PRUNE_EDGES,
        MERGE_NODES_SYNC,
        MERGE_EDGES_SYNC,
    ]
    assert report.constraints_ensured == 2


async def test_seed_universe_with_invalid_abox_writes_nothing(tmp_path: Path) -> None:
    ontology_dir = copy_toy_ontology(tmp_path)
    abox_path = ontology_dir / "individuals" / f"{TOY_UNIVERSE}.ttl"
    abox_path.write_text(abox_path.read_text() + "muontoy:alpha muontoy:unknown 3 .\n")
    client = RecordingGraphClient()

    with pytest.raises(ABoxError, match="the predicate is not declared"):
        await seed_universe(client, ontology_dir, TOY_UNIVERSE)

    assert client.calls == []
