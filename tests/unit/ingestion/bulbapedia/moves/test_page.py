from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from muon.ingestion.bulbapedia.moves.page import (
    MACHINE_FIELDS,
    MachineFact,
    MoveNoteKind,
    MovePageExtract,
    MovePageParseError,
    SkippedPage,
    SkipReason,
    parse_move_page,
    read_machines,
    read_number,
)
from muon.ingestion.bulbapedia.pages import Page, compute_sha1

PAGE_TITLE = "Testmove (move)"

BASE_INFOBOX = {
    "name": "Testmove",
    "type": "Ghost",
    "damagecategory": "Special",
    "basepp": "15",
    "power": "80",
    "accuracy": "100",
    "gen": "II",
    "target": "anyadjacent",
    "protect": "yes",
    "mirrormove": "no",
}


def build_page(infobox: dict[str, str], tail: str = "") -> Page:
    wikitext = (
        "{{MoveInfobox\n"
        + "".join(f"|{key}={value}\n" for key, value in infobox.items())
        + "}}\nLead.\n"
        + tail
    )
    return Page(
        title=PAGE_TITLE,
        page_id=1,
        revision_id=2,
        timestamp=datetime(2026, 10, 20, tzinfo=UTC),
        sha1=compute_sha1(wikitext),
        page_type="move",
        wikitext=wikitext,
    )


def parse(infobox: dict[str, str], tail: str = "") -> MovePageExtract:
    extract = parse_move_page(build_page(infobox, tail))
    assert isinstance(extract, MovePageExtract)
    return extract


def test_values_come_from_the_infobox() -> None:
    extract = parse(BASE_INFOBOX)

    assert extract.id == "testmove"
    assert (extract.power, extract.accuracy, extract.pp, extract.priority) == (80, 100, 15, 0)
    assert extract.target == "any_adjacent"
    assert (extract.type_name, extract.damage_class_name) == ("Ghost", "Special")
    assert extract.generation_label == "Generation II"
    assert (extract.is_protectable, extract.is_mirrorable) == (True, False)
    assert extract.notes == ()


@pytest.mark.parametrize(
    ("raw", "expected", "has_note"),
    [
        ("{{tt|80|70 prior to Champions}}", 70, True),
        ("{{tt|95|85 prior to Generation V; 90 prior to Pokémon Champions}}", 90, True),
        ("{{tt|60|50 in Legends: Arceus}}", 60, False),
        ("{{tt|90|80 prior to Generation V}}", 90, False),
        ("+1{{tt|*|0 prior to Generation IV}}", 1, False),
        ("{{tt|−1|0 prior to Let's Go}}", -1, False),
        ("—", None, False),
        ("&mdash;", None, False),
        ("N/A", None, False),
        ("''Varies''", None, False),
        ("", None, False),
    ],
)
def test_number_rules(raw: str, expected: int | None, has_note: bool) -> None:
    reading = read_number(PAGE_TITLE, {"power": raw}, "power")

    assert reading.value == expected
    assert [note.kind for note in reading.notes] == (
        [MoveNoteKind.CHAMPIONS_VALUE] if has_note else []
    )


def test_a_value_that_is_not_a_number_fails_the_page() -> None:
    with pytest.raises(MovePageParseError, match="power"):
        parse({**BASE_INFOBOX, "power": "strong"})


def test_priority_defaults_to_zero() -> None:
    assert parse(BASE_INFOBOX).priority == 0


def test_an_unknown_target_fails_the_page() -> None:
    with pytest.raises(MovePageParseError, match="target"):
        parse({**BASE_INFOBOX, "target": "everyone"})


@pytest.mark.parametrize("key", ["type", "damagecategory", "basepp", "gen", "target"])
def test_a_missing_required_value_fails_the_page(key: str) -> None:
    infobox = {name: value for name, value in BASE_INFOBOX.items() if name != key}

    with pytest.raises(MovePageParseError, match=key):
        parse(infobox)


def test_a_missing_boolean_gives_no_value_and_a_note() -> None:
    infobox = {name: value for name, value in BASE_INFOBOX.items() if name != "protect"}

    extract = parse(infobox)

    assert extract.is_protectable is None
    assert [(note.kind, note.field) for note in extract.notes] == [
        (MoveNoteKind.MISSING_BOOLEAN, "protect")
    ]


@pytest.mark.parametrize(
    ("infobox", "reason"),
    [
        ({**BASE_INFOBOX, "type": "Shadow"}, SkipReason.SHADOW),
    ],
)
def test_shadow_moves_are_skipped(infobox: dict[str, str], reason: SkipReason) -> None:
    skipped = parse_move_page(build_page(infobox))

    assert skipped == SkippedPage(title=PAGE_TITLE, page_id=1, move_id="testmove", reason=reason)


@pytest.mark.parametrize(
    ("category", "reason"),
    [
        ("Z-Moves", SkipReason.Z_MOVE),
        ("Max Moves", SkipReason.MAX_MOVE),
        ("G-Max Moves", SkipReason.G_MAX_MOVE),
    ],
)
def test_z_max_and_g_max_moves_are_skipped(category: str, reason: SkipReason) -> None:
    skipped = parse_move_page(build_page(BASE_INFOBOX, f"[[Category:{category}]]\n"))

    assert isinstance(skipped, SkippedPage)
    assert skipped.reason is reason


def test_a_page_without_an_infobox_is_skipped() -> None:
    page = build_page({}).model_copy(update={"wikitext": "Excavate is a move."})

    skipped = parse_move_page(page)

    assert skipped == SkippedPage(
        title=PAGE_TITLE, page_id=1, move_id=None, reason=SkipReason.NO_INFOBOX
    )


def test_machine_fields_give_1_fact_per_version_group() -> None:
    machines, notes = read_machines(
        PAGE_TITLE, {"tm4": "yes", "tm#4": "30", "tr8": "yes", "tr#8": "33"}
    )

    assert machines == (
        MachineFact(kind="TM", number_text="30", version_group_label="Diamond-Pearl", field="tm4"),
        MachineFact(kind="TM", number_text="30", version_group_label="Platinum", field="tm4"),
        MachineFact(
            kind="TM", number_text="30", version_group_label="HeartGold-SoulSilver", field="tm4"
        ),
        MachineFact(kind="TR", number_text="33", version_group_label="Sword-Shield", field="tr8"),
    )
    assert notes == ()


def test_machine_numbers_keep_their_digits() -> None:
    machines, _ = read_machines(
        PAGE_TITLE, {"tm8": "yes", "tm#8": "00", "tmza": "yes", "tm#za": "074"}
    )

    assert [machine.number_text for machine in machines] == ["00", "074"]


def test_a_number_without_its_flag_gives_a_note_and_no_machine() -> None:
    machines, notes = read_machines(PAGE_TITLE, {"tm#pe": "30"})

    assert machines == ()
    assert [(note.kind, note.field) for note in notes] == [
        (MoveNoteKind.NUMBER_WITHOUT_FLAG, "tm#pe")
    ]


def test_an_unknown_machine_field_gives_a_note() -> None:
    _, notes = read_machines(PAGE_TITLE, {"tmswsh": "yes", "mtla": "yes"})

    assert [(note.kind, note.field) for note in notes] == [(MoveNoteKind.UNKNOWN_FIELD, "tmswsh")]


def test_a_flag_without_a_number_fails_the_page() -> None:
    with pytest.raises(MovePageParseError, match="tm#9"):
        read_machines(PAGE_TITLE, {"tm9": "yes"})


def test_every_machine_field_has_a_version_group() -> None:
    assert all(spec.version_group_labels for spec in MACHINE_FIELDS.values())


@pytest.mark.pages
def test_shadow_ball_values_aliases_and_machines(
    read_extract: Callable[[str], MovePageExtract],
) -> None:
    extract = read_extract("Shadow Ball")

    assert extract.id == "shadow-ball"
    assert (extract.power, extract.accuracy, extract.pp, extract.priority) == (80, 100, 15, 0)
    assert (extract.target, extract.generation_label) == ("any_adjacent", "Generation II")
    assert (extract.is_protectable, extract.is_mirrorable) == (True, True)
    assert {"Ball'Ombre", "Spukball", "Palla Ombra", "シャドーボール"} <= set(extract.aliases)
    assert "Shadow Ball" not in extract.aliases
    assert not any("(" in alias or "*" in alias for alias in extract.aliases)
    assert "Cień Kulisty" not in extract.aliases
    labels = [(m.kind, m.number_text, m.version_group_label) for m in extract.machines]
    assert len(labels) == 21
    assert labels.count(("TM", "30", "Gold-Silver")) == 1
    assert ("TM", "43", "Let's Go Pikachu-Let's Go Eevee") in labels
    assert ("TR", "33", "Sword-Shield") in labels
    assert ("TM", "30", "Brilliant Diamond and Shining Pearl") in labels
    assert ("TM", "114", "Scarlet-Violet") in labels
    assert ("TM", "074", "Legends: Z-A") in labels
    assert sum(m.number_text == "30" for m in extract.machines) == 17


@pytest.mark.pages
@pytest.mark.parametrize(
    ("name", "field", "expected"),
    [
        ("Slash", "power", 70),
        ("Crabhammer", "accuracy", 90),
        ("Make It Rain", "accuracy", 100),
        ("Night Slash", "pp", 15),
        ("Swift", "accuracy", None),
        ("Swords Dance", "power", None),
        ("Bide", "priority", 1),
        ("Vital Throw", "priority", -1),
    ],
)
def test_pinned_values(
    read_extract: Callable[[str], MovePageExtract], name: str, field: str, expected: int | None
) -> None:
    assert getattr(read_extract(name), field) == expected


@pytest.mark.pages
def test_doodle_has_no_protect_value_and_a_note(
    read_extract: Callable[[str], MovePageExtract],
) -> None:
    extract = read_extract("Doodle")

    assert extract.generation_label == "Generation IX"
    assert extract.is_protectable is None
    assert (MoveNoteKind.MISSING_BOOLEAN, "protect") in [(n.kind, n.field) for n in extract.notes]


@pytest.mark.pages
def test_dive_and_rock_smash_machines(read_extract: Callable[[str], MovePageExtract]) -> None:
    dive = [(m.kind, m.number_text, m.version_group_label) for m in read_extract("Dive").machines]
    smash = [
        (m.kind, m.number_text, m.version_group_label) for m in read_extract("Rock Smash").machines
    ]

    assert ("HM", "08", "Ruby-Sapphire") in dive
    assert ("HM", "08", "Emerald") in dive
    assert ("HM", "07", "Omega Ruby-Alpha Sapphire") in dive
    assert ("HM", "08", "FireRed-LeafGreen") not in dive
    assert sum(kind == "HM" and number == "06" for kind, number, _ in smash) == 9
    assert ("TM", "94", "X-Y") in smash


@pytest.mark.pages
@pytest.mark.parametrize("name", ["Victory Dance", "Lunar Blessing", "Take Heart"])
def test_a_lets_go_number_without_its_flag(
    read_extract: Callable[[str], MovePageExtract], name: str
) -> None:
    extract = read_extract(name)

    assert not any(m.field == "tmpe" for m in extract.machines)
    assert (MoveNoteKind.NUMBER_WITHOUT_FLAG, "tm#pe") in [(n.kind, n.field) for n in extract.notes]


@pytest.mark.pages
@pytest.mark.parametrize(
    ("title", "reason"),
    [
        ("10,000,000 Volt Thunderbolt (move)", SkipReason.Z_MOVE),
        ("Max Flare (move)", SkipReason.MAX_MOVE),
        ("G-Max Wildfire (move)", SkipReason.G_MAX_MOVE),
        ("Shadow Rush (move)", SkipReason.SHADOW),
        ("Excavate (move)", SkipReason.NO_INFOBOX),
    ],
)
def test_pinned_pages_out_of_scope(
    read_pinned_page: Callable[[str], Page], title: str, reason: SkipReason
) -> None:
    skipped = parse_move_page(read_pinned_page(title))

    assert isinstance(skipped, SkippedPage)
    assert skipped.reason is reason
