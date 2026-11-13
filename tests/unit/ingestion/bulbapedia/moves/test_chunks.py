import pytest

from muon.ingestion.bulbapedia.chunking import (
    LONG_CHUNK_LENGTH,
    MIN_PARAGRAPH_LENGTH,
    ChunkAspect,
    ChunkFact,
    MentionRef,
)
from muon.ingestion.bulbapedia.moves.chunks import (
    MoveChunkReading,
    UnknownGameCode,
    read_move_aspect,
    read_move_chunks,
    read_row_text,
    split_game_codes,
)
from muon.ingestion.bulbapedia.wikitext import split_sections

MOVE_ID = "shadow-ball"
PROSE = "y" * MIN_PARAGRAPH_LENGTH


def read_page(wikitext: str) -> MoveChunkReading:
    return read_move_chunks(split_sections(wikitext), MOVE_ID)


def summarize(chunks: tuple[ChunkFact, ...]) -> list[tuple[str, str, str]]:
    return [(chunk.aspect.value, chunk.section_path, chunk.text) for chunk in chunks]


def build_row(games: str, text: str) -> str:
    return "{{movedescentry|" + games + "|" + text + "}}\n"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("Effect",), ChunkAspect.EFFECT),
        (("Effect", "Generation IX", "Pokémon Legends: Z-A"), ChunkAspect.EFFECT),
        (("Effect", "Outside of battle"), ChunkAspect.EFFECT),
        (("Effect", "Glitches"), ChunkAspect.EFFECT),
        (("Effect", "Generation IX", "Pokémon Champions"), None),
        (("Effect", "In contests"), None),
        (("Effect", "In Contests"), None),
        (("Effect", "Contests"), None),
        (("Trivia",), ChunkAspect.TRIVIA),
        (("Trivia", "In other languages"), ChunkAspect.TRIVIA),
        (("Description",), None),
        (("Learnset",), None),
    ],
)
def test_read_move_aspect(path: tuple[str, ...], expected: ChunkAspect | None) -> None:
    assert read_move_aspect(path) == expected


def test_champions_and_contest_subsections_give_no_chunk() -> None:
    reading = read_page(
        f"==Effect==\n===Generation IX===\n{PROSE}\n"
        "====Pokémon Champions====\nChampions text.\n"
        "===In contests===\nContest text.\n===Contests===\nMore contest text.\n"
    )

    assert summarize(reading.chunks) == [("effect", "Effect > Generation IX", PROSE)]


def test_legends_and_other_effect_subsections_give_effect_chunks() -> None:
    reading = read_page(
        "==Effect==\n===Generation IX===\n====Pokémon Legends: Z-A====\nZ-A text.\n"
        "===Outside of battle===\nField text.\n===Glitches===\nGlitch text.\n"
        "==Trivia==\n* A fact.\n"
    )

    assert summarize(reading.chunks) == [
        ("effect", "Effect > Generation IX > Pokémon Legends: Z-A", "Z-A text."),
        ("effect", "Effect > Outside of battle", "Field text."),
        ("effect", "Effect > Glitches", "Glitch text."),
        ("trivia", "Trivia", "A fact."),
    ]


def test_a_table_gives_no_chunk_and_is_counted() -> None:
    reading = read_page(f"==Effect==\n{PROSE}\n<div>\n{{|\n|Bug\n\n|Fire\n|}}\n</div>\nAfter.")

    assert [chunk.text for chunk in reading.chunks] == [f"{PROSE} After."]
    assert reading.removed_tables == 1


def test_math_keeps_its_latex_source_and_is_counted() -> None:
    formula = r"Power = 1 + 120 \times \dfrac {CurrentHP_{target}}{MaxHP_{target}}"

    reading = read_page(f"==Effect==\nIts power is <math>{formula}</math>.")

    assert formula in reading.chunks[0].text
    assert reading.math_formulas == 1


@pytest.mark.parametrize(
    ("cell", "labels", "rest"),
    [
        (
            "SwShBDSPLA",
            ("Sword-Shield", "Brilliant Diamond and Shining Pearl", "Legends: Arceus"),
            "",
        ),
        ("RSEFRLG", ("Ruby-Sapphire", "Emerald", "FireRed-LeafGreen"), ""),
        ("PtHGSS", ("Platinum", "HeartGold-SoulSilver"), ""),
        ("SVZA", ("Scarlet-Violet", "Legends: Z-A"), ""),
        ("BWB2W2", ("Black-White", "Black 2-White 2"), ""),
        ("XYQQ", ("X-Y",), "QQ"),
    ],
)
def test_split_game_codes(cell: str, labels: tuple[str, ...], rest: str) -> None:
    assert split_game_codes(cell) == (labels, rest)


def test_a_row_links_only_its_main_series_games() -> None:
    reading = read_page(
        "==Description==\n"
        + build_row("{{gameabbrev3|RSE}}{{gameabbrevss|Colo}}", "Hurls a blob.")
        + build_row("{{gameabbrevss|XD}}", "Side text.")
    )

    assert [(chunk.text, chunk.versions) for chunk in reading.chunks] == [
        ("Hurls a blob.", ("Ruby-Sapphire", "Emerald"))
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<small>(move)</small> Cuts.<br><small>(TM)</small> Teaches Cut.", "Cuts."),
        ("Flies up.<br>{{tt|*|HM}} Teaches Fly.", "Flies up."),
        ("Lands.<br>{{tt|*|TM, DP only}} Teaches Roost.", "Lands."),
        ("1st turn: Burrow<br>2nd turn: Attack", "1st turn: Burrow 2nd turn: Attack"),
        (
            "<small>(Prior to Version 1.2.0)</small> Old.<br>"
            "<small>(Version 1.2.0 onwards)</small> New.",
            "Prior to Version 1.2.0: Old. Version 1.2.0 onwards: New.",
        ),
        ("<small>(TM)</small> Teaches only.", None),
    ],
)
def test_read_row_text(raw: str, expected: str | None) -> None:
    block = read_row_text(raw)

    assert (None if block is None else block.text) == expected


def test_a_row_keeps_its_mentions() -> None:
    block = read_row_text("Stronger than {{m|Tackle}} against {{type|Ghost}} foes.")

    assert block is not None
    assert block.mentions == (
        MentionRef(template="m", target="Tackle"),
        MentionRef(template="type", target="Ghost"),
    )


def test_rows_with_the_same_text_give_1_chunk_with_the_union_of_their_groups() -> None:
    reading = read_page(
        "==Description==\n===Stellar type===\n"
        + build_row("{{gameabbrev8|SwSh}}", "Same text.")
        + "===Non-Stellar type===\n"
        + build_row("{{gameabbrev9|SVZA}}{{gameabbrev8|SwSh}}", "Same text.")
        + build_row("{{gameabbrev9|SV}}", "Other text.")
    )

    assert [(c.section_path, c.text, c.versions) for c in reading.chunks] == [
        (
            "Description > Stellar type",
            "Same text.",
            ("Sword-Shield", "Scarlet-Violet", "Legends: Z-A"),
        ),
        ("Description > Non-Stellar type", "Other text.", ("Scarlet-Violet",)),
    ]
    assert {chunk.aspect for chunk in reading.chunks} == {ChunkAspect.FLAVOR}


def test_an_unknown_game_code_is_reported_and_a_row_with_no_group_left_gives_nothing() -> None:
    reading = read_page(
        "==Description==\n"
        + build_row("{{gameabbrev9|QQ}}", "Lost text.")
        + build_row("{{gameabbrev9|SVQQ}}", "Kept text.")
    )

    assert [(chunk.text, chunk.versions) for chunk in reading.chunks] == [
        ("Kept text.", ("Scarlet-Violet",))
    ]
    assert reading.unknown_codes == (
        UnknownGameCode("Description", "QQ"),
        UnknownGameCode("Description", "SVQQ"),
    )


def test_prose_chunks_come_before_flavor_chunks() -> None:
    reading = read_page(
        "==Description==\n" + build_row("{{gameabbrev9|SV}}", "Flavor.") + "==Effect==\nEffect."
    )

    assert [chunk.aspect for chunk in reading.chunks] == [ChunkAspect.EFFECT, ChunkAspect.FLAVOR]


def test_a_chunk_above_the_long_length_is_reported_and_kept_whole() -> None:
    text = "z" * (LONG_CHUNK_LENGTH + 1)

    reading = read_page(f"==Trivia==\n* {text}\n* short")

    assert [chunk.text for chunk in reading.long_chunks] == [text]
    assert len(reading.chunks) == 2


def test_dropped_templates_are_counted_by_name() -> None:
    reading = read_page(
        "==Effect==\na {{roundy|x}} b {{roundy|y}}\n==Description==\n"
        + build_row("{{gameabbrev9|SV}}", "c {{status|Burn}}")
    )

    assert reading.dropped_templates == {"roundy": 2, "status": 1}
