"""A hand-written species batch for Diglett, the target the species parser emits.

Values come from the Diglett page at revision 4630045.
"""

from pathlib import Path

import pytest

from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord, NodeRef, PropertyValue
from muon.ingestion.runner import (
    MERGE_EDGES_UPSERT,
    MERGE_NODES_UPSERT,
    MERGE_STUBS,
    WriteMode,
    build_statements,
)
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model

REAL_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"
PKMN = "https://muon.dev/ns/pkmn#"
SPECIES = PKMN + "PokemonSpecies"
LEARNSET_ENTRY = PKMN + "LearnsetEntry"
EARTHQUAKE_ENTRY_ID = "diglett:earthquake:level:40:scarlet-violet"
# (class local name, id, name) of every node the page references but does not own.
STUBS = (
    ("Type", "ground", "Ground"),
    ("Type", "steel", "Steel"),
    ("Ability", "sand-veil", "Sand Veil"),
    ("Ability", "arena-trap", "Arena Trap"),
    ("Ability", "tangling-hair", "Tangling Hair"),
    ("Ability", "sand-force", "Sand Force"),
    ("EggGroup", "field", "Field"),
    ("Stat", "speed", "Speed"),
    ("Move", "earthquake", "Earthquake"),
    ("VersionGroup", "scarlet-violet", "Scarlet-Violet"),
)
ABOX_CLASSES = frozenset({PKMN + "Type", PKMN + "EggGroup", PKMN + "Stat", PKMN + "VersionGroup"})
ABOX_STUB_COUNT = 5


def build_ref(class_name: str, node_id: str) -> NodeRef:
    return NodeRef(class_iri=PKMN + class_name, id=node_id)


def build_edge(
    property_name: str, start: NodeRef, end: NodeRef, **properties: PropertyValue
) -> EdgeRecord:
    return EdgeRecord(
        property_iri=PKMN + property_name, start=start, end=end, properties=properties
    )


def build_diglett_batch() -> GraphBatch:
    diglett = NodeRecord(
        class_iri=SPECIES,
        id="diglett",
        name="Diglett",
        properties={
            PKMN + "pokedexNumber": 50,
            PKMN + "height": 0.2,
            PKMN + "weight": 0.8,
            PKMN + "isDefaultForm": True,
        },
    )
    alolan_diglett = NodeRecord(
        class_iri=SPECIES,
        id="diglett-alola",
        name="Alolan Diglett",
        properties={
            PKMN + "pokedexNumber": 50,
            PKMN + "height": 0.2,
            PKMN + "weight": 1.0,
            PKMN + "isDefaultForm": False,
        },
    )
    # The name is a test value: the species parser sets the name rule for reified nodes.
    earthquake_entry = NodeRecord(
        class_iri=LEARNSET_ENTRY,
        id=EARTHQUAKE_ENTRY_ID,
        name="Diglett Earthquake level 40",
        properties={PKMN + "method": "level", PKMN + "level": 40},
    )
    stubs = tuple(
        NodeRecord(class_iri=PKMN + class_name, id=node_id, name=name)
        for class_name, node_id, name in STUBS
    )
    diglett_ref = diglett.node_ref
    alolan_ref = alolan_diglett.node_ref
    entry_ref = earthquake_entry.node_ref
    edges = (
        build_edge("hasType", diglett_ref, build_ref("Type", "ground"), slot=1),
        build_edge("hasType", alolan_ref, build_ref("Type", "ground"), slot=1),
        build_edge("hasType", alolan_ref, build_ref("Type", "steel"), slot=2),
        build_edge("hasAbility", diglett_ref, build_ref("Ability", "sand-veil"), isHidden=False),
        build_edge("hasAbility", diglett_ref, build_ref("Ability", "arena-trap"), isHidden=False),
        build_edge("hasAbility", diglett_ref, build_ref("Ability", "sand-force"), isHidden=True),
        build_edge("hasAbility", alolan_ref, build_ref("Ability", "sand-veil"), isHidden=False),
        build_edge("hasAbility", alolan_ref, build_ref("Ability", "tangling-hair"), isHidden=False),
        build_edge("hasAbility", alolan_ref, build_ref("Ability", "sand-force"), isHidden=True),
        build_edge("inEggGroup", diglett_ref, build_ref("EggGroup", "field")),
        build_edge("inEggGroup", alolan_ref, build_ref("EggGroup", "field")),
        build_edge("yieldsEv", diglett_ref, build_ref("Stat", "speed"), amount=1),
        build_edge("yieldsEv", alolan_ref, build_ref("Stat", "speed"), amount=1),
        build_edge("hasForm", diglett_ref, alolan_ref, region="Alola"),
        build_edge("hasLearnsetEntry", diglett_ref, entry_ref),
        build_edge("teachesMove", entry_ref, build_ref("Move", "earthquake")),
        build_edge("inVersionGroup", entry_ref, build_ref("VersionGroup", "scarlet-violet")),
    )
    return GraphBatch(
        universe="pkmn",
        scope=frozenset({SPECIES, LEARNSET_ENTRY}),
        nodes=(diglett, alolan_diglett, earthquake_entry, *stubs),
        edges=edges,
    )


@pytest.fixture(scope="module")
def pkmn_model() -> OntologyModel:
    return load_ontology_model(REAL_ONTOLOGY_DIR, "pkmn")


def test_diglett_batch_uses_declared_terms(pkmn_model: OntologyModel) -> None:
    batch = build_diglett_batch()
    node_property_iris = {iri for node in batch.nodes for iri in node.properties}

    assert {node.class_iri for node in batch.nodes} <= pkmn_model.classes
    assert {edge.property_iri for edge in batch.edges} <= pkmn_model.object_properties
    assert node_property_iris <= pkmn_model.datatype_properties


def test_diglett_stubs() -> None:
    batch = build_diglett_batch()

    assert {(stub.class_iri, stub.id) for stub in batch.stub_nodes} == {
        (PKMN + class_name, node_id) for class_name, node_id, _ in STUBS
    }


def test_diglett_abox_stubs_match_seed_keys(pkmn_model: OntologyModel) -> None:
    seed_keys = {node.node_ref for node in read_abox(pkmn_model, REAL_ONTOLOGY_DIR).nodes}
    abox_stubs = {
        stub.node_ref for stub in build_diglett_batch().stub_nodes if stub.class_iri in ABOX_CLASSES
    }

    assert len(abox_stubs) == ABOX_STUB_COUNT
    assert abox_stubs <= seed_keys


def test_diglett_batch_compiles(pkmn_model: OntologyModel) -> None:
    statements = build_statements(
        build_diglett_batch(), WriteMode.UPSERT, LabelMaterializer(pkmn_model)
    )
    properties_by_edge = {
        (row["type"], row["start_id"], row["end_id"]): row["properties"]
        for row in statements[-1].parameters["edges"]
    }

    assert [statement.query for statement in statements] == [
        MERGE_NODES_UPSERT,
        MERGE_STUBS,
        MERGE_EDGES_UPSERT,
    ]
    assert properties_by_edge[("HAS_TYPE", "diglett-alola", "ground")] == {"slot": 1}
    assert properties_by_edge[("HAS_TYPE", "diglett-alola", "steel")] == {"slot": 2}
    assert properties_by_edge[("HAS_ABILITY", "diglett", "sand-force")] == {"isHidden": True}
    assert properties_by_edge[("HAS_FORM", "diglett", "diglett-alola")] == {"region": "Alola"}
    assert properties_by_edge[("YIELDS_EV", "diglett", "speed")] == {"amount": 1}
