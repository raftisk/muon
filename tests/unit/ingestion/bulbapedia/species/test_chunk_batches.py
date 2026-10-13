from pathlib import Path

import pytest

from muon.ingestion.batch import NODE_ID_PATTERN, EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.species.batches import (
    ABILITY,
    ASPECT,
    CHUNK_SCOPE,
    DESCRIBES,
    FROM_WORK,
    MENTIONED_IN,
    MOVE,
    POKEMON_SPECIES,
    SECTION,
    SOURCE,
    SOURCE_URL,
    TEXT,
    TEXT_CHUNK,
    TYPE,
    VERSION,
    AboxIndex,
    ChunkBuild,
    ChunkBuildError,
    ChunkInputs,
    SpeciesIndex,
    build_abox_index,
    build_chunk_batch,
    hash_chunk_text,
    number_chunks,
)
from muon.ingestion.bulbapedia.species.extract import (
    ChunkAspect,
    ChunkFact,
    MentionRef,
    NoteKind,
    SpeciesPageExtract,
)
from muon.ingestion.runner import WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model
from tests.unit.ingestion.bulbapedia.species.claim_summary import (
    CHUNK_ANCHOR_PATTERNS,
    CHUNK_CLAIM_PATTERNS,
    claims_by_anchor,
)
from tests.unit.ingestion.bulbapedia.species.test_batches import build_facts

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
SOURCE_URL_VALUE = (
    "https://bulbapedia.bulbagarden.net/w/index.php?title=Diglett_(Pok%C3%A9mon)&oldid=1"
)
SPECIES = SpeciesIndex(
    names={
        "diglett": "Diglett",
        "diglett-alola": "Alolan Diglett",
        "dugtrio": "Dugtrio",
        "onix": "Onix",
        "klefki": "Klefki",
    },
    ids_by_name={"Diglett": "diglett", "Dugtrio": "dugtrio", "Onix": "onix", "Klefki": "klefki"},
)


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def inputs(model: OntologyModel) -> ChunkInputs:
    abox: AboxIndex = build_abox_index(read_abox(model, ONTOLOGY_DIR))
    return ChunkInputs(abox=abox, species=SPECIES, source_url=SOURCE_URL_VALUE)


def build_fact(text: str = "Diglett digs.", **fields: object) -> ChunkFact:
    values: dict[str, object] = {
        "subject_id": "diglett",
        "aspect": ChunkAspect.BIOLOGY,
        "section_path": "Biology",
        "text": text,
        "mentions": (),
    }
    return ChunkFact.model_validate({**values, **fields})


DIGLETT_FORMS = (
    build_facts("diglett", "Diglett"),
    build_facts("diglett-alola", "Alolan Diglett", form_index=2, is_default=False),
)


def build_extract(*chunks: ChunkFact) -> SpeciesPageExtract:
    return SpeciesPageExtract(
        title="Diglett (Pokémon)",
        page_id=2482,
        revision_id=1,
        forms=DIGLETT_FORMS,
        candidates=(),
        learnset=(),
        form_changes=(),
        evolutions=(),
        residue=(),
        notes=(),
        chunks=chunks,
    )


def build_batch(inputs: ChunkInputs, *chunks: ChunkFact) -> GraphBatch:
    batch = build_chunk_batch(build_extract(*chunks), inputs).batch
    return batch


def node_of(batch: GraphBatch, node_id: str) -> NodeRecord:
    return next(node for node in batch.nodes if node.id == node_id)


def edges_of(batch: GraphBatch, property_iri: str) -> list[tuple[str, str]]:
    return [
        (edge.start.id, edge.end.id) for edge in batch.edges if edge.property_iri == property_iri
    ]


def edge_of(batch: GraphBatch, property_iri: str) -> EdgeRecord:
    (edge,) = [edge for edge in batch.edges if edge.property_iri == property_iri]
    return edge


def test_a_chunk_has_the_core_properties_a_name_and_a_checked_id(inputs: ChunkInputs) -> None:
    batch = build_batch(inputs, build_fact("Diglett digs."))

    chunk = node_of(batch, "bulbapedia:pokemonspecies:diglett:biology:1")
    assert NODE_ID_PATTERN.match(chunk.id)
    assert chunk.class_iri == TEXT_CHUNK
    assert chunk.name == "Diglett biology 1"
    assert chunk.properties == {
        TEXT: "Diglett digs.",
        ASPECT: "biology",
        SECTION: "Biology",
        SOURCE_URL: SOURCE_URL_VALUE,
        SOURCE: "bulbapedia",
    }


def test_the_batch_has_the_chunk_scope_and_the_universe(inputs: ChunkInputs) -> None:
    batch = build_batch(inputs, build_fact())

    assert batch.universe == "pkmn"
    assert batch.scope == CHUNK_SCOPE == {TEXT_CHUNK}
    assert {node.class_iri for node in batch.owner_nodes} == {TEXT_CHUNK}
    assert {node.class_iri for node in batch.stub_nodes} == {POKEMON_SPECIES}


def test_ordinals_count_per_subject_and_aspect_in_page_order(inputs: ChunkInputs) -> None:
    batch = build_batch(
        inputs,
        build_fact("one.", section_path="Biology"),
        build_fact("two.", section_path="Biology > Forms"),
        build_fact("alola.", subject_id="diglett-alola"),
        build_fact("trivia.", aspect=ChunkAspect.TRIVIA, section_path="Trivia"),
    )

    ids = [node.id for node in batch.owner_nodes]
    assert ids == [
        "bulbapedia:pokemonspecies:diglett-alola:biology:1",
        "bulbapedia:pokemonspecies:diglett:biology:1",
        "bulbapedia:pokemonspecies:diglett:biology:2",
        "bulbapedia:pokemonspecies:diglett:trivia:1",
    ]
    assert node_of(batch, ids[0]).name == "Alolan Diglett biology 1"


def test_a_pokedex_chunk_has_a_hash_id_no_source_and_version_edges(inputs: ChunkInputs) -> None:
    text = "Its skin is very thin."
    fact = build_fact(
        text,
        aspect=ChunkAspect.POKEDEX,
        section_path="Game data > Pokédex entries",
        versions=("Gold", "HeartGold", "Y"),
    )

    batch = build_batch(inputs, fact)

    digest = hash_chunk_text(text)
    chunk = node_of(batch, f"bulbapedia:pokemonspecies:diglett:pokedex:{digest}")
    assert len(digest) == 8
    assert NODE_ID_PATTERN.match(chunk.id)
    assert SOURCE not in chunk.properties
    assert chunk.properties[ASPECT] == "pokedex"
    assert chunk.name == f"Diglett pokedex {digest}"
    assert edges_of(batch, FROM_WORK) == [
        (chunk.id, "gold"),
        (chunk.id, "heartgold"),
        (chunk.id, "y"),
    ]
    assert {node.class_iri for node in batch.stub_nodes} == {POKEMON_SPECIES, VERSION}


def test_every_chunk_has_one_describes_edge_to_its_subject(inputs: ChunkInputs) -> None:
    batch = build_batch(inputs, build_fact("a."), build_fact("b.", subject_id="diglett-alola"))

    assert edges_of(batch, DESCRIBES) == [
        ("bulbapedia:pokemonspecies:diglett-alola:biology:1", "diglett-alola"),
        ("bulbapedia:pokemonspecies:diglett:biology:1", "diglett"),
    ]
    assert node_of(batch, "diglett-alola").name == "Alolan Diglett"


def test_mentions_become_edges_from_the_entities_to_the_chunk(inputs: ChunkInputs) -> None:
    fact = build_fact(
        "Mentions.",
        mentions=(
            MentionRef(template="p", target="Onix"),
            MentionRef(template="m", target="Telekinesis"),
            MentionRef(template="a", target="Tangling Hair"),
            MentionRef(template="type", target="Steel"),
            MentionRef(template="t", target="Ground"),
        ),
    )

    batch = build_batch(inputs, fact)

    chunk_id = "bulbapedia:pokemonspecies:diglett:biology:1"
    assert edges_of(batch, MENTIONED_IN) == [
        ("tangling-hair", chunk_id),
        ("telekinesis", chunk_id),
        ("onix", chunk_id),
        ("ground", chunk_id),
        ("steel", chunk_id),
    ]
    assert node_of(batch, "onix").class_iri == POKEMON_SPECIES
    assert node_of(batch, "onix").name == "Onix"
    assert node_of(batch, "telekinesis").class_iri == MOVE
    assert node_of(batch, "tangling-hair").class_iri == ABILITY
    assert node_of(batch, "steel").class_iri == TYPE
    assert all(node.properties == {} for node in batch.stub_nodes), (
        "stubs carry a name and no property"
    )


def test_a_species_mention_resolves_through_the_index_to_the_stub_with_its_name(
    inputs: ChunkInputs,
) -> None:
    fact = build_fact(mentions=(MentionRef(template="p", target="Dugtrio"),))

    batch = build_batch(inputs, fact)

    assert (node_of(batch, "dugtrio").name, node_of(batch, "dugtrio").class_iri) == (
        "Dugtrio",
        POKEMON_SPECIES,
    )


def test_two_mentions_of_one_entity_give_one_edge(inputs: ChunkInputs) -> None:
    fact = build_fact(
        mentions=(MentionRef(template="p", target="Onix"), MentionRef(template="p", target="Onix"))
    )

    assert len(edges_of(build_batch(inputs, fact), MENTIONED_IN)) == 1


def test_a_mention_of_the_subject_gives_no_edge(inputs: ChunkInputs) -> None:
    fact = build_fact(
        mentions=(
            MentionRef(template="p", target="Diglett"),
            MentionRef(template="p", target="Onix"),
        )
    )

    batch = build_batch(inputs, fact)

    assert [start for start, _ in edges_of(batch, MENTIONED_IN)] == ["onix"]


def test_a_mention_of_the_base_form_still_counts_for_a_form_subject(inputs: ChunkInputs) -> None:
    fact = build_fact(
        subject_id="diglett-alola", mentions=(MentionRef(template="p", target="Diglett"),)
    )

    assert [start for start, _ in edges_of(build_batch(inputs, fact), MENTIONED_IN)] == ["diglett"]


def test_an_unknown_species_is_noted_with_no_edge_and_no_stub(inputs: ChunkInputs) -> None:
    fact = build_fact(
        section_path="Biology > Forms",
        mentions=(MentionRef(template="p", target="Nidoran"),),
    )

    build = build_chunk_build(inputs, fact)

    assert edges_of(build.batch, MENTIONED_IN) == []
    assert "nidoran" not in {node.id for node in build.batch.nodes}
    (note,) = build.notes
    assert note.kind is NoteKind.UNRESOLVED_MENTION
    assert note.section_path == "Biology > Forms"
    assert note.detail == "{{p|Nidoran}} names no species"


@pytest.mark.parametrize("template", ["type", "t"])
def test_an_unknown_type_is_noted_and_does_not_fail_the_page(
    inputs: ChunkInputs, template: str
) -> None:
    fact = build_fact(mentions=(MentionRef(template=template, target="Shadow"),))

    build = build_chunk_build(inputs, fact)

    assert edges_of(build.batch, MENTIONED_IN) == []
    (note,) = build.notes
    assert note.detail == "{{" + template + "|Shadow}} names no A-Box Type"


def test_a_move_outside_the_run_is_trusted_as_written(inputs: ChunkInputs) -> None:
    fact = build_fact(mentions=(MentionRef(template="m", target="Telekinesis"),))

    batch = build_batch(inputs, fact)

    assert node_of(batch, "telekinesis").name == "Telekinesis"
    assert edges_of(batch, MENTIONED_IN)[0][0] == "telekinesis"


def test_a_move_target_with_no_letter_is_noted_not_failed(inputs: ChunkInputs) -> None:
    fact = build_fact(mentions=(MentionRef(template="m", target="???"),))

    build = build_chunk_build(inputs, fact)

    assert [note.kind for note in build.notes] == [NoteKind.UNRESOLVED_MENTION]


def build_chunk_build(inputs: ChunkInputs, *chunks: ChunkFact) -> ChunkBuild:
    return build_chunk_batch(build_extract(*chunks), inputs)


def pokedex_fact(versions: tuple[str, ...], text: str = "Entry text.") -> ChunkFact:
    return build_fact(
        text,
        aspect=ChunkAspect.POKEDEX,
        section_path="Game data > Pokédex entries",
        versions=versions,
    )


def test_an_unknown_version_gets_no_edge_and_a_note(inputs: ChunkInputs) -> None:
    build = build_chunk_build(inputs, pokedex_fact(("Gold", "Pokémon Mansion")))

    assert edges_of(build.batch, FROM_WORK)[0][1] == "gold"
    assert len(edges_of(build.batch, FROM_WORK)) == 1
    (note,) = build.notes
    assert note.kind is NoteKind.UNKNOWN_VERSION
    assert note.detail == "Pokémon Mansion is no A-Box Version"


def test_a_pokedex_chunk_with_no_known_version_is_dropped_with_a_note(inputs: ChunkInputs) -> None:
    build = build_chunk_build(inputs, pokedex_fact(("Pokémon Mansion",)), build_fact("Kept."))

    assert [node.id for node in build.batch.owner_nodes] == [
        "bulbapedia:pokemonspecies:diglett:biology:1"
    ]
    (note,) = build.notes
    assert note.detail == "Pokémon Mansion is no A-Box Version; row dropped"


def assert_batch_of_anchors_only(batch: GraphBatch) -> None:
    """The batch of a page with no chunk: the page species as stubs, their claims, no owner."""
    assert batch.scope == CHUNK_SCOPE
    assert not batch.owner_nodes
    assert {node.id for node in batch.stub_nodes} == {"diglett", "diglett-alola"}
    assert batch.edges == ()
    assert claims_by_anchor(batch) == {
        "diglett": CHUNK_ANCHOR_PATTERNS,
        "diglett-alola": CHUNK_ANCHOR_PATTERNS,
    }


def test_a_page_without_a_chunk_gets_a_batch_of_anchors_and_claims(inputs: ChunkInputs) -> None:
    build = build_chunk_build(inputs)

    assert_batch_of_anchors_only(build.batch)
    assert build.notes == ()


def test_a_page_whose_chunks_are_all_dropped_gets_a_batch_of_anchors_and_claims(
    inputs: ChunkInputs,
) -> None:
    build = build_chunk_build(inputs, pokedex_fact(("Pokémon Mansion",)))

    assert_batch_of_anchors_only(build.batch)
    assert len(build.notes) == 1


def test_two_texts_with_the_same_hash_fail_the_page_with_both_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "muon.ingestion.bulbapedia.species.batches.hash_chunk_text", lambda text: "deadbeef"
    )
    facts = [pokedex_fact(("Gold",), "First text."), pokedex_fact(("Gold",), "Second text.")]

    with pytest.raises(ChunkBuildError, match="First text.*Second text") as error:
        number_chunks("Diglett (Pokémon)", facts)

    assert "Diglett (Pokémon)" in str(error.value)
    assert "deadbeef" in str(error.value)


def test_the_hash_is_the_first_8_hex_characters_of_the_sha256() -> None:
    assert hash_chunk_text("abc") == "ba7816bf"


def test_sorting_is_by_class_and_id_for_nodes_and_by_property_for_edges(
    inputs: ChunkInputs,
) -> None:
    batch = build_batch(
        inputs,
        build_fact("b.", mentions=(MentionRef(template="p", target="Onix"),)),
        build_fact("a.", aspect=ChunkAspect.TRIVIA, section_path="Trivia"),
    )

    assert [(node.class_iri, node.id) for node in batch.nodes] == sorted(
        (node.class_iri, node.id) for node in batch.nodes
    )
    properties = [edge.property_iri for edge in batch.edges]
    assert properties == sorted(properties)


def test_a_chunk_batch_compiles_to_upsert_statements(
    model: OntologyModel, inputs: ChunkInputs
) -> None:
    fact = build_fact(
        "Compiles.",
        mentions=(
            MentionRef(template="p", target="Onix"),
            MentionRef(template="type", target="Steel"),
        ),
    )
    batch = build_batch(inputs, fact, pokedex_fact(("Gold", "Sun")))

    statements = build_statements(batch, WriteMode.UPSERT, LabelMaterializer(model))

    assert statements


def test_chunk_properties_are_the_core_text_properties_only(inputs: ChunkInputs) -> None:
    batch = build_batch(inputs, build_fact(), pokedex_fact(("Gold",)))

    keys = {key for node in batch.owner_nodes for key in node.properties}

    assert keys == {TEXT, ASPECT, SECTION, SOURCE_URL, SOURCE}


def test_a_chunk_batch_claims_describes_at_every_form_and_two_patterns_at_each_chunk(
    inputs: ChunkInputs,
) -> None:
    batch = build_batch(inputs, build_fact("One."), pokedex_fact(("Gold",)))

    chunk_ids = {node.id for node in batch.owner_nodes}
    assert len(chunk_ids) == 2
    claims = claims_by_anchor(batch)
    assert set(claims) == {"diglett", "diglett-alola", *chunk_ids}
    assert claims["diglett"] == claims["diglett-alola"] == CHUNK_ANCHOR_PATTERNS
    assert all(claims[chunk_id] == CHUNK_CLAIM_PATTERNS for chunk_id in chunk_ids)
