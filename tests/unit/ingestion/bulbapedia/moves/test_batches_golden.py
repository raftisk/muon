"""Shadow Ball end to end: pinned page, A-Box and PokeAPI snapshot to its pass 1 batch."""

from collections.abc import Callable
from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    HAS_TYPE,
    IN_VERSION_GROUP,
    INTRODUCED_IN,
    MACHINE,
    MOVE,
    TEACHES_MOVE,
    build_abox_index,
)
from muon.ingestion.bulbapedia.moves.batches import (
    ACCURACY,
    EFFECT_TEXT,
    HAS_DAMAGE_CLASS,
    HAS_FLAG,
    IS_MIRRORABLE,
    IS_PROTECTABLE,
    MACHINE_KIND,
    MACHINE_NUMBER,
    MOVE_OWNER_CLAIMS,
    MOVE_SCOPE,
    POWER,
    PP,
    PRIORITY,
    TARGET,
    build_move_batch,
)
from muon.ingestion.bulbapedia.moves.page import MovePageExtract
from muon.ingestion.bulbapedia.moves.pokeapi import build_pokeapi_index, read_effect_text
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotStore
from muon.ingestion.runner import MERGE_NODES_UPSERT, WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
EFFECT = "Has a chance to lower the target’s Special Defense by one stage."
TM30_GROUPS = (
    "gold-silver",
    "crystal",
    "ruby-sapphire",
    "emerald",
    "firered-leafgreen",
    "colosseum",
    "xd",
    "diamond-pearl",
    "platinum",
    "heartgold-soulsilver",
    "black-white",
    "black-2-white-2",
    "x-y",
    "omega-ruby-alpha-sapphire",
    "sun-moon",
    "ultra-sun-ultra-moon",
)
OTHER_MACHINES = {
    "tm43-lets-go-pikachu-lets-go-eevee": ("TM43", "TM", 43, "lets-go-pikachu-lets-go-eevee"),
    "tr33-sword-shield": ("TR33", "TR", 33, "sword-shield"),
    "tm30-brilliant-diamond-and-shining-pearl": (
        "TM30",
        "TM",
        30,
        "brilliant-diamond-and-shining-pearl",
    ),
    "tm114-scarlet-violet": ("TM114", "TM", 114, "scarlet-violet"),
    "tm074-legends-z-a": ("TM074", "TM", 74, "legends-z-a"),
}


@pytest.fixture(scope="module")
def batch(read_extract: Callable[[str], MovePageExtract]) -> GraphBatch:
    try:
        index = build_pokeapi_index(SnapshotStore())
    except SnapshotError as error:
        pytest.skip(f"the PokeAPI snapshot is not on disk: {error}")
    model = load_ontology_model(ONTOLOGY_DIR, "pkmn")
    abox = build_abox_index(read_abox(model, ONTOLOGY_DIR))
    return build_move_batch(
        read_extract("Shadow Ball"), abox, read_effect_text(index, "shadow-ball")
    )


def test_the_move_owner(batch: GraphBatch) -> None:
    owners = [node for node in batch.nodes if node.class_iri == MOVE]

    assert batch.scope == MOVE_SCOPE
    assert [(node.id, node.name) for node in owners] == [("shadow-ball", "Shadow Ball")]
    properties = dict(owners[0].properties)
    aliases = properties.pop(ALIASES)
    assert properties == {
        POWER: 80,
        ACCURACY: 100,
        PP: 15,
        PRIORITY: 0,
        TARGET: "any_adjacent",
        IS_PROTECTABLE: True,
        IS_MIRRORABLE: True,
        EFFECT_TEXT: EFFECT,
    }
    assert isinstance(aliases, tuple)
    assert {"Ball'Ombre", "Spukball", "Palla Ombra", "シャドーボール"} <= set(aliases)
    assert "Shadow Ball" not in aliases


def test_the_move_edges(batch: GraphBatch) -> None:
    out_edges = sorted(
        (edge.property_iri, edge.end.id) for edge in batch.edges if edge.start.class_iri == MOVE
    )

    assert out_edges == sorted(
        [
            (HAS_TYPE, "ghost"),
            (HAS_DAMAGE_CLASS, "special"),
            (HAS_FLAG, "ballistic"),
            (INTRODUCED_IN, "generation-ii"),
        ]
    )


def test_the_21_machines(batch: GraphBatch) -> None:
    machines = {node.id: node for node in batch.nodes if node.class_iri == MACHINE}
    expected = {f"tm30-{group}": ("TM30", "TM", 30, group) for group in TM30_GROUPS}
    expected.update(OTHER_MACHINES)
    teaches = {(e.start.id, e.end.id) for e in batch.edges if e.property_iri == TEACHES_MOVE}
    groups = {e.start.id: e.end.id for e in batch.edges if e.property_iri == IN_VERSION_GROUP}

    assert len(machines) == 21
    assert {
        machine_id: (
            node.name,
            node.properties[MACHINE_KIND],
            node.properties[MACHINE_NUMBER],
            groups[machine_id],
        )
        for machine_id, node in machines.items()
    } == expected
    assert teaches == {(machine_id, "shadow-ball") for machine_id in expected}
    assert all(len(node.properties) == 2 for node in machines.values())


def test_the_claims(batch: GraphBatch) -> None:
    anchor = NodeRef(class_iri=MOVE, id="shadow-ball")
    expected = {
        (rule.property_iri, rule.direction, rule.other_class_iri, rule.owns_other_end)
        for rule in MOVE_OWNER_CLAIMS[MOVE]
    }

    assert {claim.anchor for claim in batch.claims} == {anchor}
    assert len(batch.claims) == len(expected)
    assert {
        (c.property_iri, c.direction, c.other_class_iri, c.owns_other_end) for c in batch.claims
    } == expected


def test_the_batch_compiles(batch: GraphBatch) -> None:
    model = load_ontology_model(ONTOLOGY_DIR, "pkmn")

    statements = build_statements(batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert MERGE_NODES_UPSERT in [statement.query for statement in statements]
