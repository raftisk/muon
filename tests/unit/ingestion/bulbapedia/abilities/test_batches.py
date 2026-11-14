from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.abilities.batches import (
    ABILITY_CHUNK_SCOPE,
    ABILITY_SCOPE,
    EFFECT_TEXT,
    AbilityChunkBuild,
    AbilityChunkInputs,
    build_ability_batch,
    build_ability_chunk_batch,
    check_sup_codes,
)
from muon.ingestion.bulbapedia.abilities.page import AbilityNoteKind, AbilityPageExtract
from muon.ingestion.bulbapedia.batching import (
    ABILITY,
    ALIASES,
    DESCRIBES,
    FROM_WORK,
    GENERATION,
    INTRODUCED_IN,
    MENTIONED_IN,
    SOURCE,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
    hash_chunk_text,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.species.batches import POKEMON_SPECIES
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
SOURCE_URL = "https://bulbapedia.bulbagarden.net/w/index.php?oldid=1"
FLAVOR_TEXT = "Guards in tests."


@pytest.fixture(scope="module")
def seed() -> GraphBatch:
    return read_abox(load_ontology_model(ONTOLOGY_DIR, "pkmn"), ONTOLOGY_DIR)


@pytest.fixture(scope="module")
def abox(seed: GraphBatch) -> AboxIndex:
    return build_abox_index(seed)


def build_extract(**values: object) -> AbilityPageExtract:
    base: dict[str, object] = {
        "title": "Test Guard (Ability)",
        "page_id": 1,
        "revision_id": 2,
        "id": "test-guard",
        "name": "Test Guard",
        "generation_label": "Generation III",
        "aliases": ("Garde Test",),
    }
    return AbilityPageExtract.model_validate({**base, **values})


def build_fact(aspect: ChunkAspect, text: str, *mentions: MentionRef) -> ChunkFact:
    return ChunkFact(
        subject_id="test-guard",
        aspect=aspect,
        section_path="Infobox" if aspect is ChunkAspect.FLAVOR else "Effect",
        text=text,
        mentions=mentions,
        versions=("Ruby-Sapphire", "Legends: Arceus") if aspect is ChunkAspect.FLAVOR else (),
    )


def resolve_species(target: str) -> NodeRecord | None:
    if target != "Pikachu":
        return None
    return NodeRecord(class_iri=POKEMON_SPECIES, id="pikachu", name="Pikachu")


def build_chunks(abox: AboxIndex, *facts: ChunkFact) -> AbilityChunkBuild:
    inputs = AbilityChunkInputs(abox=abox, resolve_species=resolve_species, source_url=SOURCE_URL)
    return build_ability_chunk_batch(build_extract(chunks=facts), inputs)


def list_edges(batch: GraphBatch, property_iri: str) -> list[tuple[str, str]]:
    return [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == property_iri]


def test_pass_1_owns_the_ability_with_its_generation_and_claim(abox: AboxIndex) -> None:
    batch = build_ability_batch(build_extract(), abox, "Guards.")

    (ability,) = batch.owner_nodes
    assert batch.scope == ABILITY_SCOPE
    assert (ability.id, ability.name) == ("test-guard", "Test Guard")
    assert ability.properties == {EFFECT_TEXT: "Guards.", ALIASES: ("Garde Test",)}
    assert list_edges(batch, INTRODUCED_IN) == [("test-guard", "generation-iii")]
    assert [(c.property_iri, c.other_class_iri) for c in batch.claims] == [
        (INTRODUCED_IN, GENERATION)
    ]


def test_an_ability_without_effect_text_or_aliases_has_no_such_property(abox: AboxIndex) -> None:
    batch = build_ability_batch(build_extract(aliases=()), abox, None)

    assert batch.owner_nodes[0].properties == {}


def test_a_generation_the_a_box_lacks_fails_the_page(abox: AboxIndex) -> None:
    with pytest.raises(BatchBuildError, match="Generation XX"):
        build_ability_batch(build_extract(generation_label="Generation XX"), abox, None)


def test_chunk_ids_follow_the_ability_and_link_to_their_works(abox: AboxIndex) -> None:
    build = build_chunks(
        abox,
        build_fact(ChunkAspect.EFFECT, "It guards."),
        build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT),
    )
    digest = hash_chunk_text(FLAVOR_TEXT)
    chunks = {node.id: node for node in build.batch.owner_nodes}

    assert build.batch.scope == ABILITY_CHUNK_SCOPE
    assert sorted(chunks) == [
        "bulbapedia:ability:test-guard:effect:1",
        f"bulbapedia:ability:test-guard:flavor:{digest}",
    ]
    assert SOURCE not in chunks[f"bulbapedia:ability:test-guard:flavor:{digest}"].properties
    assert {end for _, end in list_edges(build.batch, DESCRIBES)} == {"test-guard"}
    assert sorted(end for _, end in list_edges(build.batch, FROM_WORK)) == [
        "legends-arceus",
        "ruby-sapphire",
    ]
    assert {n.class_iri for n in build.batch.stub_nodes} == {ABILITY, VERSION_GROUP}


def test_mentions_skip_the_ability_itself_and_note_the_misses(abox: AboxIndex) -> None:
    fact = build_fact(
        ChunkAspect.EFFECT,
        "Text.",
        MentionRef(template="p", target="Pikachu"),
        MentionRef(template="a", target="Test Guard"),
        MentionRef(template="p", target="Missingno"),
    )

    build = build_chunks(abox, fact)

    assert [start for start, _ in list_edges(build.batch, MENTIONED_IN)] == ["pikachu"]
    assert [note.kind for note in build.notes] == [AbilityNoteKind.UNRESOLVED_MENTION]


def test_every_sup_code_names_a_box_version_groups(seed: GraphBatch) -> None:
    check_sup_codes(seed)
