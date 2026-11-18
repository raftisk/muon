import pytest

from muon.ingestion.bulbapedia.chunking import MentionRef
from muon.ingestion.bulbapedia.descriptions import (
    DescriptionReading,
    UnknownGameCode,
    read_description_rows,
    read_row_text,
    split_game_codes,
)
from muon.ingestion.bulbapedia.wikitext import split_sections


def build_row(games: str, text: str) -> str:
    return "{{movedescentry|" + games + "|" + text + "}}\n"


def read_rows(wikitext: str) -> DescriptionReading:
    return read_description_rows(split_sections(wikitext))


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
    reading = read_rows(
        "==Description==\n"
        + build_row("{{gameabbrev3|RSE}}{{gameabbrevss|Colo}}", "Hurls a blob.")
        + build_row("{{gameabbrevss|XD}}", "Side text.")
    )

    assert [(row.block.text, row.labels) for row in reading.rows] == [
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


def test_an_unknown_game_code_is_reported_and_a_row_with_no_group_left_gives_nothing() -> None:
    reading = read_rows(
        "==Description==\n"
        + build_row("{{gameabbrev9|QQ}}", "Lost text.")
        + build_row("{{gameabbrev9|SVQQ}}", "Kept text.")
    )

    assert [(row.block.text, row.labels) for row in reading.rows] == [
        ("Kept text.", ("Scarlet-Violet",))
    ]
    assert reading.unknown_codes == (
        UnknownGameCode("Description", "QQ"),
        UnknownGameCode("Description", "SVQQ"),
    )
