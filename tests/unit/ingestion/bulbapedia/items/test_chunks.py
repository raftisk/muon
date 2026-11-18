import pytest

from muon.ingestion.bulbapedia.chunking import MIN_PARAGRAPH_LENGTH, ChunkAspect
from muon.ingestion.bulbapedia.descriptions import UnknownGameCode
from muon.ingestion.bulbapedia.items.chunks import (
    ItemChunkReading,
    read_item_aspect,
    read_item_chunks,
)
from muon.ingestion.bulbapedia.items.page import rebase_core_sections
from muon.ingestion.bulbapedia.wikitext import split_sections

ITEM_ID = "black-sludge"
PROSE = "y" * MIN_PARAGRAPH_LENGTH


def read_page(wikitext: str) -> ItemChunkReading:
    sections = rebase_core_sections(split_sections(wikitext))
    return read_item_chunks(sections, ITEM_ID)


def build_row(games: str, text: str) -> str:
    return "{{movedescentry|" + games + "|" + text + "}}\n"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("Effect",), ChunkAspect.EFFECT),
        (("Effect", "Held item"), ChunkAspect.EFFECT),
        (("Effect", "Bag item"), ChunkAspect.EFFECT),
        (("Effect", "Generation IX", "Pokémon Champions"), None),
        (("Effect", "In contests"), None),
        (("Effect", "Ingredient"), None),
        (("Effect", "Generation III", "In Ingredient dishes"), None),
        (("Trivia",), ChunkAspect.TRIVIA),
        (("Description",), None),
        (("Acquisition",), None),
    ],
)
def test_read_item_aspect(path: tuple[str, ...], expected: ChunkAspect | None) -> None:
    assert read_item_aspect(path) == expected


def test_effect_subsections_come_out_with_the_excluded_ones_left_off() -> None:
    reading = read_page(
        "==In the core series games==\n===Effect===\n====Held item====\n"
        f"{PROSE}\n====Ingredient====\nBlender text.\n====Pokémon Champions====\nChamps.\n"
    )

    assert [(chunk.aspect, chunk.section_path) for chunk in reading.chunks] == [
        (ChunkAspect.EFFECT, "Effect > Held item")
    ]


def test_description_rows_of_a_subsection_give_flavor_chunks_under_the_rebased_path() -> None:
    reading = read_page(
        "==In the core series games==\n===Description===\n====Tag information====\n"
        + build_row("{{gameabbrev9|SV}}", "Tag text.")
    )

    assert [(chunk.section_path, chunk.text, chunk.versions) for chunk in reading.chunks] == [
        ("Description > Tag information", "Tag text.", ("Scarlet-Violet",))
    ]
    assert {chunk.aspect for chunk in reading.chunks} == {ChunkAspect.FLAVOR}


def test_colosseum_and_champions_rows_give_nothing() -> None:
    reading = read_page(
        "==In the core series games==\n===Description===\n"
        + build_row("{{gameabbrev3|RSE}}{{gameabbrevss|Colo}}", "Rows.")
        + build_row("{{gameabbrevss|XD}}", "Side text.")
        + build_row("{{gameabbrevchamp|CHP}}", "Champions text.")
    )

    assert [(chunk.text, chunk.versions) for chunk in reading.chunks] == [
        ("Rows.", ("Ruby-Sapphire", "Emerald"))
    ]


def test_svza_row_reads_both_games() -> None:
    reading = read_page(
        "==In the core series games==\n===Description===\n"
        + build_row("{{gameabbrev9|SVZA}}", "Shared.")
    )

    assert reading.chunks[0].versions == ("Scarlet-Violet", "Legends: Z-A")


def test_unknown_game_codes_are_reported() -> None:
    reading = read_page(
        "==In the core series games==\n===Description===\n"
        + build_row("{{gameabbrev9|QQ}}", "Lost.")
    )

    assert reading.chunks == ()
    assert reading.unknown_codes == (UnknownGameCode("Description", "QQ"),)


def test_trivia_bullets_give_trivia_chunks_and_flavor_comes_last() -> None:
    reading = read_page(
        "==In the core series games==\n===Description===\n"
        + build_row("{{gameabbrev9|SV}}", "Flavor.")
        + "==Trivia==\n* First fact.\n* Second fact.\n"
    )

    assert [chunk.aspect for chunk in reading.chunks] == [
        ChunkAspect.TRIVIA,
        ChunkAspect.TRIVIA,
        ChunkAspect.FLAVOR,
    ]
