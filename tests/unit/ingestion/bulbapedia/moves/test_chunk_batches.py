from pathlib import Path

import pytest

from muon.ingestion.batch import ClaimDirection, GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import (
    DESCRIBES,
    FROM_WORK,
    MENTIONED_IN,
    MOVE,
    SOURCE,
    TEXT_CHUNK,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
    hash_chunk_text,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.moves import batches
from muon.ingestion.bulbapedia.moves.batches import (
    MOVE_CHUNK_SCOPE,
    GameCodeError,
    MoveChunkBuild,
    MoveChunkInputs,
    build_move_chunk_batch,
    check_game_codes,
)
from muon.ingestion.bulbapedia.moves.page import MoveNoteKind
from muon.ingestion.bulbapedia.species.batches import POKEMON_SPECIES
from muon.ingestion.runner import MERGE_NODES_UPSERT, MERGE_STUBS, WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model
from tests.unit.ingestion.bulbapedia.moves.test_batches import build_extract

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
SOURCE_URL = "https://bulbapedia.bulbagarden.net/w/index.php?oldid=1"
FLAVOR_TEXT = "Hurls a shadowy blob at the target."


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def seed(model: OntologyModel) -> GraphBatch:
    return read_abox(model, ONTOLOGY_DIR)


@pytest.fixture(scope="module")
def abox(seed: GraphBatch) -> AboxIndex:
    return build_abox_index(seed)


def resolve_species(target: str) -> NodeRecord | None:
    return (
        NodeRecord(class_iri=POKEMON_SPECIES, id="gengar", name="Gengar")
        if target == "Gengar"
        else None
    )


def build_fact(aspect: ChunkAspect, text: str, *mentions: MentionRef) -> ChunkFact:
    return ChunkFact(
        subject_id="testmove",
        aspect=aspect,
        section_path="Description" if aspect is ChunkAspect.FLAVOR else "Effect",
        text=text,
        mentions=mentions,
        versions=("Gold-Silver", "Legends: Z-A") if aspect is ChunkAspect.FLAVOR else (),
    )


def build_chunks(abox: AboxIndex, *facts: ChunkFact) -> MoveChunkBuild:
    inputs = MoveChunkInputs(abox=abox, resolve_species=resolve_species, source_url=SOURCE_URL)
    return build_move_chunk_batch(build_extract(chunks=facts), inputs)


def list_edges(batch: GraphBatch, property_iri: str) -> list[tuple[str, str]]:
    return [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == property_iri]


def test_chunk_ids_properties_and_describes(abox: AboxIndex) -> None:
    build = build_chunks(
        abox,
        build_fact(ChunkAspect.EFFECT, "Deals damage."),
        build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT),
        build_fact(ChunkAspect.TRIVIA, "A fact."),
    )
    digest = hash_chunk_text(FLAVOR_TEXT)
    chunks = {node.id: node for node in build.batch.owner_nodes}

    assert build.batch.scope == MOVE_CHUNK_SCOPE == {TEXT_CHUNK}
    assert list(chunks) == sorted(
        [
            "bulbapedia:move:testmove:effect:1",
            f"bulbapedia:move:testmove:flavor:{digest}",
            "bulbapedia:move:testmove:trivia:1",
        ]
    )
    assert SOURCE not in chunks[f"bulbapedia:move:testmove:flavor:{digest}"].properties
    assert chunks["bulbapedia:move:testmove:effect:1"].properties[SOURCE] == "bulbapedia"
    assert {end for _, end in list_edges(build.batch, DESCRIBES)} == {"testmove"}
    assert len(list_edges(build.batch, DESCRIBES)) == 3


def test_a_flavor_chunk_links_to_its_version_groups(abox: AboxIndex) -> None:
    build = build_chunks(abox, build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT))

    assert sorted(end for _, end in list_edges(build.batch, FROM_WORK)) == [
        "gold-silver",
        "legends-z-a",
    ]
    assert {n.class_iri for n in build.batch.stub_nodes} == {MOVE, VERSION_GROUP}


def test_a_flavor_label_the_a_box_lacks_fails_the_page(abox: AboxIndex) -> None:
    fact = build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT).model_copy(update={"versions": ("Pokopia",)})

    with pytest.raises(BatchBuildError, match="Pokopia"):
        build_chunks(abox, fact)


def test_mentions_resolve_skip_the_subject_and_note_the_misses(abox: AboxIndex) -> None:
    fact = build_fact(
        ChunkAspect.EFFECT,
        "Text.",
        MentionRef(template="p", target="Gengar"),
        MentionRef(template="m", target="Testmove"),
        MentionRef(template="m", target="Never-Ending Nightmare"),
        MentionRef(template="t", target="Stellar"),
    )

    build = build_chunks(abox, fact)

    assert sorted(start for start, _ in list_edges(build.batch, MENTIONED_IN)) == [
        "gengar",
        "never-ending-nightmare",
    ]
    assert [(note.kind, note.detail) for note in build.notes] == [
        (MoveNoteKind.UNRESOLVED_MENTION, "{{t|Stellar}} names no A-Box Type")
    ]


def test_a_page_with_no_chunk_keeps_its_anchor_and_claim(abox: AboxIndex) -> None:
    batch = build_chunks(abox).batch

    assert batch.owner_nodes == ()
    assert [node.id for node in batch.stub_nodes] == ["testmove"]
    assert [(c.anchor.id, c.property_iri, c.direction) for c in batch.claims] == [
        ("testmove", DESCRIBES, ClaimDirection.IN)
    ]


def test_game_codes_match_the_a_box(seed: GraphBatch) -> None:
    check_game_codes(seed)


def test_a_wrong_game_code_label_fails(monkeypatch: pytest.MonkeyPatch, seed: GraphBatch) -> None:
    monkeypatch.setattr(batches, "GAME_CODES", {"ZA": ("Legends Z-A",)})

    with pytest.raises(GameCodeError, match="ZA names 'Legends Z-A'"):
        check_game_codes(seed)


def test_the_chunk_batch_compiles_into_upsert_statements(
    abox: AboxIndex, model: OntologyModel
) -> None:
    batch = build_chunks(abox, build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT)).batch

    statements = build_statements(batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert {MERGE_NODES_UPSERT, MERGE_STUBS} <= {s.query for s in statements}
