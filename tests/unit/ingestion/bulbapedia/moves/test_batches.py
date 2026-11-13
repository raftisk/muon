from pathlib import Path

import pytest

from muon.ingestion.batch import ClaimDirection, GraphBatch, NodeRef
from muon.ingestion.bulbapedia.batching import (
    GENERATION,
    HAS_TYPE,
    IN_VERSION_GROUP,
    INTRODUCED_IN,
    MACHINE,
    MOVE,
    TEACHES_MOVE,
    TYPE,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.moves import batches
from muon.ingestion.bulbapedia.moves.batches import (
    ACCURACY,
    EFFECT_TEXT,
    HAS_DAMAGE_CLASS,
    HAS_FLAG,
    MACHINE_KIND,
    MACHINE_NUMBER,
    MIN_HITS,
    MOVE_DAMAGE_CLASS,
    MOVE_FLAG,
    MOVE_SCOPE,
    POWER,
    MachineFieldError,
    build_move_batch,
    check_machine_fields,
)
from muon.ingestion.bulbapedia.moves.page import MachineFact, MachineField, MovePageExtract
from muon.ingestion.runner import (
    MERGE_EDGES_UPSERT,
    MERGE_NODES_UPSERT,
    MERGE_STUBS,
    PRUNE_CLAIMED_IN,
    PRUNE_CLAIMED_OUT,
    WriteMode,
    build_statements,
)
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
EFFECT = "Has a chance to lower the target’s Special Defense by one stage."


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def seed(model: OntologyModel) -> GraphBatch:
    return read_abox(model, ONTOLOGY_DIR)


@pytest.fixture(scope="module")
def abox(seed: GraphBatch) -> AboxIndex:
    return build_abox_index(seed)


def build_extract(**values: object) -> MovePageExtract:
    base: dict[str, object] = {
        "title": "Testmove (move)",
        "page_id": 1,
        "revision_id": 2,
        "id": "testmove",
        "name": "Testmove",
        "type_name": "Ghost",
        "damage_class_name": "Special",
        "generation_label": "Generation II",
        "pp": 15,
        "priority": 0,
        "target": "any_adjacent",
        "power": 80,
        "accuracy": None,
        "is_protectable": True,
        "flags": ("ballistic", "two-turn"),
        "machines": (
            MachineFact(
                kind="TM", number_text="30", version_group_label="Gold-Silver", field="tm2"
            ),
            MachineFact(
                kind="TM", number_text="074", version_group_label="Legends: Z-A", field="tmza"
            ),
        ),
    }
    return MovePageExtract.model_validate({**base, **values})


def find_node_ids(batch: GraphBatch, class_iri: str) -> list[str]:
    return [node.id for node in batch.nodes if node.class_iri == class_iri]


def list_edges(batch: GraphBatch, property_iri: str) -> list[tuple[str, str]]:
    return [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == property_iri]


def test_the_move_owner_has_no_empty_values(abox: AboxIndex) -> None:
    batch = build_move_batch(build_extract(), abox, EFFECT)

    move = next(node for node in batch.nodes if node.class_iri == MOVE)
    assert batch.scope == MOVE_SCOPE
    assert (move.id, move.name) == ("testmove", "Testmove")
    assert move.properties[POWER] == 80
    assert move.properties[EFFECT_TEXT] == EFFECT
    assert ACCURACY not in move.properties
    assert MIN_HITS not in move.properties


def test_no_effect_text_leaves_the_property_out(abox: AboxIndex) -> None:
    batch = build_move_batch(build_extract(), abox, None)

    move = next(node for node in batch.nodes if node.class_iri == MOVE)
    assert EFFECT_TEXT not in move.properties


def test_a_box_stubs_and_move_edges(abox: AboxIndex) -> None:
    batch = build_move_batch(build_extract(), abox, None)

    assert list_edges(batch, HAS_TYPE) == [("testmove", find_node_ids(batch, TYPE)[0])]
    assert list_edges(batch, HAS_DAMAGE_CLASS) == [("testmove", "special")]
    assert sorted(end for _, end in list_edges(batch, HAS_FLAG)) == sorted(
        find_node_ids(batch, MOVE_FLAG)
    )
    assert len(find_node_ids(batch, MOVE_FLAG)) == 2
    assert list_edges(batch, INTRODUCED_IN) == [("testmove", "generation-ii")]
    assert find_node_ids(batch, MOVE_DAMAGE_CLASS) == ["special"]
    assert find_node_ids(batch, GENERATION) == ["generation-ii"]


def test_machines_are_owners_with_kind_number_and_edges(abox: AboxIndex) -> None:
    batch = build_move_batch(build_extract(), abox, None)

    machines = [node for node in batch.nodes if node.class_iri == MACHINE]
    assert [(m.id, m.name, m.properties) for m in machines] == [
        ("tm074-legends-z-a", "TM074", {MACHINE_KIND: "TM", MACHINE_NUMBER: 74}),
        ("tm30-gold-silver", "TM30", {MACHINE_KIND: "TM", MACHINE_NUMBER: 30}),
    ]
    assert sorted(list_edges(batch, TEACHES_MOVE)) == [
        ("tm074-legends-z-a", "testmove"),
        ("tm30-gold-silver", "testmove"),
    ]
    assert sorted(list_edges(batch, IN_VERSION_GROUP)) == [
        ("tm074-legends-z-a", "legends-z-a"),
        ("tm30-gold-silver", "gold-silver"),
    ]
    assert sorted(find_node_ids(batch, VERSION_GROUP)) == ["gold-silver", "legends-z-a"]


def test_claims_are_the_5_move_rules(abox: AboxIndex) -> None:
    batch = build_move_batch(build_extract(), abox, None)

    anchor = NodeRef(class_iri=MOVE, id="testmove")
    assert {claim.anchor for claim in batch.claims} == {anchor}
    assert {
        (claim.property_iri, claim.direction, claim.other_class_iri, claim.owns_other_end)
        for claim in batch.claims
    } == {
        (HAS_TYPE, ClaimDirection.OUT, TYPE, False),
        (HAS_DAMAGE_CLASS, ClaimDirection.OUT, MOVE_DAMAGE_CLASS, False),
        (HAS_FLAG, ClaimDirection.OUT, MOVE_FLAG, False),
        (INTRODUCED_IN, ClaimDirection.OUT, GENERATION, False),
        (TEACHES_MOVE, ClaimDirection.IN, MACHINE, True),
    }


@pytest.mark.parametrize(
    "values",
    [
        {"type_name": "Shadowy"},
        {"flags": ("sparkly",)},
        {"generation_label": "Generation XI"},
        {
            "machines": (
                MachineFact(kind="TM", number_text="1", version_group_label="Gold", field="tm2"),
            )
        },
    ],
)
def test_a_name_the_a_box_lacks_fails_the_page(abox: AboxIndex, values: dict[str, object]) -> None:
    with pytest.raises(BatchBuildError, match="Testmove \\(move\\)"):
        build_move_batch(build_extract(**values), abox, None)


def test_machine_fields_match_the_a_box(seed: GraphBatch) -> None:
    check_machine_fields(seed)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"tmza": MachineField("TM", ("Legends Z-A",))}, "no A-Box VersionGroup"),
        ({"tm4": MachineField("TM", ("X-Y",))}, "not part of Generation IV"),
    ],
)
def test_a_wrong_machine_field_fails(
    monkeypatch: pytest.MonkeyPatch,
    seed: GraphBatch,
    fields: dict[str, MachineField],
    message: str,
) -> None:
    monkeypatch.setattr(batches, "MACHINE_FIELDS", fields)

    with pytest.raises(MachineFieldError, match=message):
        check_machine_fields(seed)


def test_the_move_batch_compiles_into_upsert_statements(
    abox: AboxIndex, model: OntologyModel
) -> None:
    batch = build_move_batch(build_extract(), abox, EFFECT)

    statements = build_statements(batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert [s.query for s in statements] == [
        PRUNE_CLAIMED_OUT,
        PRUNE_CLAIMED_IN,
        MERGE_NODES_UPSERT,
        MERGE_STUBS,
        MERGE_EDGES_UPSERT,
    ]
