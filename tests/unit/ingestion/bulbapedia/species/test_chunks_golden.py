"""Extract-level expectations of the chunk scenarios, checked on pinned pages."""

import re
from collections.abc import Callable

import pytest

from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.species.extract import (
    ChunkAspect,
    ChunkFact,
    MentionRef,
    NoteKind,
    PageNote,
    SpeciesPageExtract,
)
from muon.ingestion.bulbapedia.species.page import parse_species_page

pytestmark = pytest.mark.pages

CHUNK_TITLES = tuple(
    f"{name} (Pokémon)"
    for name in ("Diglett", "Rotom", "Deoxys", "Tauros", "Oricorio", "Meowth", "Vivillon", "Minior")
)
DIGLETT = "Diglett (Pokémon)"
POKEDEX = "Game data > Pokédex entries"
SKIN_ENTRY = (
    "Its skin is very thin. If it is exposed to light, its blood heats up, causing it to grow weak."
)
SIDE_GAME_VERSIONS = {"Stadium", "Stadium 2", "Colosseum", "XD", "Battle Revolution", "Pokopia"}
# Markup that rendering must leave out; `<` and `>` can still occur in quoted text.
MARKUP_LEAK = re.compile(r"\{\{|\}\}|\[\[|\]\]|<ref|<div|<small|<sup|<gallery|\{\||\|\}")

Extracts = dict[str, SpeciesPageExtract]


@pytest.fixture(scope="module")
def extracts(read_pinned_page: Callable[[str], Page]) -> Extracts:
    parsed: Extracts = {}
    for title in CHUNK_TITLES:
        extract = parse_species_page(read_pinned_page(title))
        assert extract is not None, f"{title} has no Pokémon Infobox"
        parsed[title] = extract
    return parsed


def chunks_at(extract: SpeciesPageExtract, section_path: str) -> list[ChunkFact]:
    return [chunk for chunk in extract.chunks if chunk.section_path == section_path]


def chunk_with(chunks: list[ChunkFact], text: str) -> ChunkFact:
    (found,) = [chunk for chunk in chunks if text in chunk.text]
    return found


def notes_of(extract: SpeciesPageExtract, kind: NoteKind) -> list[PageNote]:
    return [note for note in extract.notes if note.kind is kind]


def subjects_at(extract: SpeciesPageExtract, section_path: str) -> set[str]:
    return {chunk.subject_id for chunk in chunks_at(extract, section_path)}


def test_diglett_biology_chunks_describe_diglett(extracts: Extracts) -> None:
    chunks = chunks_at(extracts[DIGLETT], "Biology")

    assert len(chunks) == 3
    assert {(chunk.subject_id, chunk.aspect) for chunk in chunks} == {
        ("diglett", ChunkAspect.BIOLOGY)
    }
    onix = chunk_with(chunks, "burrows left behind by Onix")
    assert MentionRef(template="p", target="Onix") in onix.mentions
    assert "caves" in onix.text
    assert "<ref" not in onix.text


def test_diglett_forms_section_merges_the_short_paragraph(extracts: Extracts) -> None:
    extract = extracts[DIGLETT]
    chunks = chunks_at(extract, "Biology > Forms")

    assert len(chunks) == 4
    assert subjects_at(extract, "Biology > Forms") == {"diglett"}
    first = chunks[0]
    assert first.text.startswith("Diglett has a regional form: Alolan Diglett. In Alola,")
    assert len(first.text) > 100


def test_diglett_forms_section_mentions(extracts: Extracts) -> None:
    chunks = chunks_at(extracts[DIGLETT], "Biology > Forms")

    assert chunk_with(chunks, "Tangling Hair").mentions == (
        MentionRef(template="a", target="Tangling Hair"),
    )
    assert set(chunk_with(chunks, "Meltan").mentions) == {
        MentionRef(template="p", target="Klefki"),
        MentionRef(template="p", target="Meltan"),
        MentionRef(template="type", target="Steel"),
    }


def test_diglett_trivia_gives_one_chunk_per_top_level_bullet(extracts: Extracts) -> None:
    chunks = chunks_at(extracts[DIGLETT], "Trivia")

    assert len(chunks) == 6
    assert all(chunk.aspect is ChunkAspect.TRIVIA for chunk in chunks)
    assert chunks[1].text.startswith("Diglett and its evolution are the only Pokémon")
    assert MentionRef(template="p", target="Dugtrio") in chunks[1].mentions
    assert chunks[2].text.endswith(
        "Alolan Diglett has the lowest base HP and Defense stats of all Steel-type Pokémon."
    )


def test_diglett_design_quirks_chunks(extracts: Extracts) -> None:
    chunks = chunks_at(extracts[DIGLETT], "Trivia > Design quirks")

    assert len(chunks) == 3
    assert all(chunk.aspect is ChunkAspect.TRIVIA for chunk in chunks)


def test_diglett_origin_chunks(extracts: Extracts) -> None:
    extract = extracts[DIGLETT]
    origin = chunks_at(extract, "Trivia > Origin")
    name_origin = chunks_at(extract, "Trivia > Origin > Name origin")
    concept = chunks_at(extract, "Trivia > Concept and development")

    assert all(chunk.aspect is ChunkAspect.ORIGIN for chunk in (*origin, *name_origin, *concept))
    assert "based on an artificial mole used in the Whac-A-Mole arcade games" in origin[0].text
    assert len(name_origin) == 1
    assert concept[0].text.startswith("Diglett was designed by Shigeki Morimoto. In the")
    assert len(concept[0].text) > 100


def test_diglett_pokedex_text_shared_across_games_is_one_chunk(extracts: Extracts) -> None:
    extract = extracts[DIGLETT]
    chunks = [chunk for chunk in extract.chunks if chunk.text == SKIN_ENTRY]

    (chunk,) = chunks
    assert (chunk.subject_id, chunk.aspect, chunk.section_path) == (
        "diglett",
        ChunkAspect.POKEDEX,
        POKEDEX,
    )
    assert {"Gold", "HeartGold", "Y"} <= set(chunk.versions)


def test_diglett_side_game_rows_give_no_chunk_and_no_note(extracts: Extracts) -> None:
    extract = extracts[DIGLETT]
    versions = {version for chunk in extract.chunks for version in chunk.versions}

    assert not versions & SIDE_GAME_VERSIONS
    assert not any("Pokémon Red, Silver, or Crystal" in chunk.text for chunk in extract.chunks)
    assert notes_of(extract, NoteKind.UNMATCHED_FORM_LABEL) == []


def test_diglett_alolan_entries_describe_the_alolan_form(extracts: Extracts) -> None:
    extract = extracts[DIGLETT]
    alolan = [chunk for chunk in extract.chunks if chunk.subject_id == "diglett-alola"]
    base = [chunk for chunk in extract.chunks if chunk.aspect is ChunkAspect.POKEDEX]

    assert {chunk.section_path for chunk in alolan} == {f"{POKEDEX} > Alolan Form"}
    assert chunk_with(alolan, "altered form of whiskers made of metal").versions == ("Sun",)
    brilliant = chunk_with(base, "This Pokémon lives underground")
    assert (brilliant.subject_id, brilliant.section_path) == ("diglett", POKEDEX)


def test_rotom_form_subsection_titles_equal_the_node_name(extracts: Extracts) -> None:
    extract = extracts["Rotom (Pokémon)"]

    assert subjects_at(extract, "Biology > Forms > Heat Rotom") == {"rotom-heat"}
    assert subjects_at(extract, "Biology > Forms > Wash Rotom") == {"rotom-wash"}


def test_deoxys_form_subsection_titles_equal_the_infobox_name(extracts: Extracts) -> None:
    extract = extracts["Deoxys (Pokémon)"]

    assert subjects_at(extract, "Biology > Forms > Attack Forme") == {"deoxys-attack"}
    assert subjects_at(extract, "Biology > Forms > Normal Forme") == {"deoxys"}


def test_tauros_form_title_is_contained_in_one_node_name(extracts: Extracts) -> None:
    extract = extracts["Tauros (Pokémon)"]

    assert subjects_at(extract, "Biology > Forms > Combat Breed") == {"tauros-paldea-combat"}


def test_oricorio_straight_and_curly_apostrophes_match(extracts: Extracts) -> None:
    extract = extracts["Oricorio (Pokémon)"]

    assert subjects_at(extract, "Biology > Forms > Pa'u Style") == {"oricorio-pau"}
    assert {
        chunk.subject_id for chunk in extract.chunks if chunk.section_path.endswith("Pa'u Style")
    } == {"oricorio-pau"}


def test_meowth_gigantamax_entries_stay_on_the_base_form_with_a_note(extracts: Extracts) -> None:
    extract = extracts["Meowth (Pokémon)"]
    path = f"{POKEDEX} > Gigantamax"

    assert subjects_at(extract, path) == {"meowth"}
    assert path in [note.section_path for note in notes_of(extract, NoteKind.UNMATCHED_FORM_LABEL)]


def test_vivillon_pattern_table_gives_no_text(extracts: Extracts) -> None:
    extract = extracts["Vivillon (Pokémon)"]
    forms = chunks_at(extract, "Biology > Forms")

    assert forms
    assert {chunk.subject_id for chunk in forms} == {"vivillon"}
    assert not any("{|" in chunk.text or "File:" in chunk.text for chunk in forms)


def test_minior_core_subsection_stays_on_the_base_form_with_a_note(extracts: Extracts) -> None:
    extract = extracts["Minior (Pokémon)"]

    assert subjects_at(extract, "Biology > Forms > Core") == {"minior"}
    assert "Biology > Forms > Core" in [
        note.section_path for note in notes_of(extract, NoteKind.UNMATCHED_FORM_LABEL)
    ]


@pytest.mark.parametrize("title", CHUNK_TITLES)
def test_chunks_describe_a_form_of_their_page_and_keep_markup_out(
    extracts: Extracts, title: str
) -> None:
    extract = extracts[title]
    form_ids = {form.id for form in extract.forms}

    assert extract.chunks
    for chunk in extract.chunks:
        assert chunk.subject_id in form_ids
        assert chunk.text == chunk.text.strip()
        assert "  " not in chunk.text
        assert "\n" not in chunk.text
        assert not MARKUP_LEAK.search(chunk.text), chunk.text[:120]
        assert (chunk.aspect is ChunkAspect.POKEDEX) == bool(chunk.versions)
