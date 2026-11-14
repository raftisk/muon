"""Move text chunks on pinned pages: the spec scenarios with the final texts."""

from collections.abc import Callable

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import (
    ASPECT,
    DESCRIBES,
    FROM_WORK,
    MENTIONED_IN,
    SECTION,
    SOURCE,
    SOURCE_URL,
    TEXT,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact
from muon.ingestion.bulbapedia.moves.batches import MoveChunkInputs, build_move_chunk_batch
from muon.ingestion.bulbapedia.moves.page import MovePageExtract, parse_move_page
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.runner import MERGE_NODES_UPSERT, WriteMode, build_statements
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.unit.ingestion.bulbapedia.moves.conftest import ONTOLOGY_DIR

pytestmark = pytest.mark.pages

PageChunkReader = Callable[[str], tuple[MovePageExtract, GraphBatch]]

GOLDEN_MOVES = (
    "Shadow Ball",
    "Hidden Power",
    "Aromatherapy",
    "Crush Grip",
    "Dig",
    "Fly",
    "Roost",
    "Glaive Rush",
    "Tera Blast",
    "Baton Pass",
    "Apple Acid",
    "Aqua Jet",
)
SHADOW_BALL_GROUPS = {
    "An attack that may lower Spcl. Def.": {"gold-silver", "crystal"},
    "Hurls a black blob that may lower Sp. Def.": {"ruby-sapphire", "emerald"},
    "A shadowy blob is hurled at the foe. May also lower the foe's Sp. Def.": {"firered-leafgreen"},
    "The user hurls a shadowy blob at the foe. It may also lower the foe's Sp. Def stat.": {
        "diamond-pearl",
        "platinum",
        "heartgold-soulsilver",
    },
    "The user hurls a shadowy blob at the target. It may also lower the target's Sp. Def stat.": {
        "black-white",
        "black-2-white-2",
    },
    "The user hurls a shadowy blob at the target. This may also lower the target's Sp. Def stat.": {
        "x-y",
        "omega-ruby-alpha-sapphire",
        "sun-moon",
        "ultra-sun-ultra-moon",
        "lets-go-pikachu-lets-go-eevee",
        "sword-shield",
        "brilliant-diamond-and-shining-pearl",
    },
    (
        "The user hurls a shadowy blob at the target to deal damage. This may also lower the "
        "target’s defensive stats."
    ): {"legends-arceus"},
    (
        "The user attacks by hurling a shadowy blob at the target. This may also lower the "
        "target’s Sp. Def stat."
    ): {"scarlet-violet"},
    "The user attacks by hurling a shadowy blob. This may also lower targets' Sp. Def stats.": {
        "legends-z-a"
    },
}


@pytest.fixture(scope="module")
def read_page_chunks(
    read_pinned_page: Callable[[str], Page],
    chunk_inputs: Callable[[Page], MoveChunkInputs],
) -> PageChunkReader:
    """Parse the pinned page of a move name and build its pass 2 batch."""

    def read(name: str) -> tuple[MovePageExtract, GraphBatch]:
        page = read_pinned_page(f"{name} (move)")
        extract = parse_move_page(page)
        assert isinstance(extract, MovePageExtract), f"{name} is out of scope"
        return extract, build_move_chunk_batch(extract, chunk_inputs(page)).batch

    return read


def list_chunks(extract: MovePageExtract, aspect: ChunkAspect) -> list[ChunkFact]:
    return [chunk for chunk in extract.chunks if chunk.aspect is aspect]


def find_chunk(batch: GraphBatch, chunk_id: str) -> NodeRecord:
    (chunk,) = (node for node in batch.owner_nodes if node.id == chunk_id)
    return chunk


def list_edges(batch: GraphBatch, property_iri: str) -> list[tuple[str, str]]:
    return sorted((e.start.id, e.end.id) for e in batch.edges if e.property_iri == property_iri)


def test_shadow_ball_effect_chunks(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Shadow Ball")
    first, second = list_chunks(extract, ChunkAspect.EFFECT)

    assert first.section_path == "Effect"
    assert first.text.endswith("Pokémon with Bulletproof are unaffected by Shadow Ball.")
    assert second.section_path == "Effect > Generation VIII > Pokémon Legends: Arceus"


def test_shadow_ball_flavor_chunks_link_to_their_version_groups(
    read_page_chunks: PageChunkReader,
) -> None:
    extract, batch = read_page_chunks("Shadow Ball")
    groups_by_text = {
        str(find_chunk(batch, chunk_id).properties[TEXT]): set()
        for chunk_id, _ in list_edges(batch, FROM_WORK)
    }
    for chunk_id, group in list_edges(batch, FROM_WORK):
        groups_by_text[str(find_chunk(batch, chunk_id).properties[TEXT])].add(group)

    assert len(list_chunks(extract, ChunkAspect.FLAVOR)) == len(SHADOW_BALL_GROUPS) == 9
    assert len(list_edges(batch, FROM_WORK)) == 20
    assert groups_by_text == SHADOW_BALL_GROUPS


def test_shadow_ball_trivia_chunks_show_template_words(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Shadow Ball")
    trivia = [chunk.text for chunk in list_chunks(extract, ChunkAspect.TRIVIA)]

    assert len(trivia) == 5
    assert "Cooltrainer" in trivia[2]
    assert "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!" in trivia[3]


def test_shadow_ball_mentions_and_describes(read_page_chunks: PageChunkReader) -> None:
    _, batch = read_page_chunks("Shadow Ball")

    assert list_edges(batch, MENTIONED_IN) == [
        ("bulletproof", "bulbapedia:move:shadow-ball:effect:1"),
        ("muk", "bulbapedia:move:shadow-ball:trivia:3"),
        ("night-shade", "bulbapedia:move:shadow-ball:trivia:5"),
    ]
    assert sorted(start for start, _ in list_edges(batch, DESCRIBES)) == sorted(
        node.id for node in batch.owner_nodes
    )
    assert {end for _, end in list_edges(batch, DESCRIBES)} == {"shadow-ball"}


def test_shadow_ball_chunk_properties(read_page_chunks: PageChunkReader) -> None:
    _, batch = read_page_chunks("Shadow Ball")

    for chunk in batch.owner_nodes:
        assert {TEXT, ASPECT, SECTION, SOURCE_URL} <= set(chunk.properties)
        is_flavor = chunk.properties[ASPECT] == ChunkAspect.FLAVOR.value
        assert (SOURCE in chunk.properties) is not is_flavor


def test_hidden_power_removes_its_table_and_keeps_legends_prose(
    read_page_chunks: PageChunkReader,
) -> None:
    extract, _ = read_page_chunks("Hidden Power")

    assert extract.removed_tables == 1
    assert "main" in extract.dropped_templates
    assert "Effect > Generation VIII > Pokémon Legends: Arceus" in {
        chunk.section_path for chunk in extract.chunks
    }


def test_aromatherapy_shows_the_status_words(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Aromatherapy")

    assert "of all status conditions (sleep, poison, paralysis, freeze, and burn)." in (
        extract.chunks[0].text
    )


def test_crush_grip_keeps_its_latex_source(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Crush Grip")
    formula = r"Power = 1 + 120 \times \dfrac {CurrentHP_{target}}{MaxHP_{target}}"

    assert extract.math_formulas == 2
    assert any(formula in chunk.text for chunk in extract.chunks)


@pytest.mark.parametrize(
    ("name", "text"),
    [("Dig", "1st turn: Burrow 2nd turn: Attack"), ("Fly", "1st turn: Fly 2nd turn: Attack")],
)
def test_a_2_line_row_without_captions_gives_1_text(
    read_page_chunks: PageChunkReader, name: str, text: str
) -> None:
    extract, _ = read_page_chunks(name)
    (chunk,) = (c for c in list_chunks(extract, ChunkAspect.FLAVOR) if c.text == text)

    assert chunk.versions == ("Gold-Silver", "Crystal")


@pytest.mark.parametrize("name", ["Fly", "Roost"])
def test_tm_and_hm_texts_are_gone(read_page_chunks: PageChunkReader, name: str) -> None:
    extract, _ = read_page_chunks(name)
    texts = [chunk.text for chunk in list_chunks(extract, ChunkAspect.FLAVOR)]

    assert texts
    assert not any("TM" in text or "HM" in text or "Teaches" in text for text in texts)


def test_glaive_rush_keeps_its_captions_as_lead_ins(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Glaive Rush")
    (scarlet_violet,) = (
        c for c in list_chunks(extract, ChunkAspect.FLAVOR) if c.versions == ("Scarlet-Violet",)
    )

    assert scarlet_violet.text.startswith("Prior to Version 1.2.0: The user throws")
    assert " Version 1.2.0 onwards: The user throws" in scarlet_violet.text


def test_tera_blast_keeps_its_description_subsections(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Tera Blast")

    assert {chunk.section_path for chunk in list_chunks(extract, ChunkAspect.FLAVOR)} == {
        "Description > Stellar type",
        "Description > Non-Stellar type",
    }


@pytest.mark.parametrize(
    ("name", "excluded"), [("Baton Pass", "Contests"), ("Apple Acid", "Pokémon Champions")]
)
def test_excluded_effect_sections_give_no_chunk(
    read_page_chunks: PageChunkReader, name: str, excluded: str
) -> None:
    extract, _ = read_page_chunks(name)

    assert extract.chunks
    assert not any(excluded in chunk.section_path for chunk in extract.chunks)


def test_baton_pass_leaves_no_notice_markup(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Baton Pass")

    assert extract.chunks[0].text.startswith("Baton Pass switches out the user")
    assert not any("{{" in chunk.text or "}}" in chunk.text for chunk in extract.chunks)


def test_aqua_jet_keeps_its_legends_z_a_prose(read_page_chunks: PageChunkReader) -> None:
    extract, _ = read_page_chunks("Aqua Jet")

    assert "Effect > Generation IX > Pokémon Legends: Z-A" in {
        chunk.section_path for chunk in list_chunks(extract, ChunkAspect.EFFECT)
    }


@pytest.mark.parametrize("name", GOLDEN_MOVES)
def test_every_golden_chunk_batch_compiles_into_upsert_statements(
    read_page_chunks: PageChunkReader, name: str
) -> None:
    _, batch = read_page_chunks(name)
    materializer = LabelMaterializer(load_ontology_model(ONTOLOGY_DIR, "pkmn"))

    statements = build_statements(batch, WriteMode.UPSERT, materializer)

    assert MERGE_NODES_UPSERT in {statement.query for statement in statements}
