from collections.abc import Mapping

import pytest

from muon.ingestion.bulbapedia.species.extract import LearnMethod, LearnsetRow, NoteKind
from muon.ingestion.bulbapedia.species.forms import FormInputs, FormSet, resolve_forms
from muon.ingestion.bulbapedia.species.learnset import list_learnset_titles, read_learnset
from muon.ingestion.bulbapedia.species.page import PageParseError
from muon.ingestion.bulbapedia.wikitext import split_sections

PAGE_TITLE = "Testmon (Pokémon)"
BASE_INFOBOX = {"name": "Testmon", "type1": "Ground"}
ALOLA_INFOBOX = {
    **BASE_INFOBOX,
    "form2": "Alolan Form",
    "image2": "0050Testmon-Alola.png",
    "form2type1": "Ground",
}

LEVEL_TABLE = """====By [[Level|leveling up]]====
{{learnlist/levelh/9|Testmon|Ground|Ground|1}}
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
{{learnlist/level9|40|Earthquake|Ground|Physical|100|100|10||'''}}
{{learnlist/level9|{{tt|Evo.|Learned upon evolving}}|Mud Shot|Ground|Special|55|95|15}}
{{learnlist/level9|{{tt|Rem.|Move Reminder}}|Fissure|Ground|Physical|—|30|5}}
{{learnlist/levelf/9|Testmon|Ground|Ground|1}}
"""


def page(*tables: str) -> str:
    return "==Game data==\n===Learnset===\n" + "".join(tables)


def build_forms(infobox: Mapping[str, str], wikitext: str) -> tuple[FormSet, tuple]:
    sections = split_sections(wikitext)
    forms = resolve_forms(FormInputs(PAGE_TITLE, infobox, (), list_learnset_titles(sections)))
    return forms, sections


def read_rows(
    wikitext: str, infobox: Mapping[str, str] = BASE_INFOBOX
) -> tuple[list[LearnsetRow], list[NoteKind]]:
    forms, sections = build_forms(infobox, wikitext)
    rows, notes = read_learnset(sections, forms, PAGE_TITLE)
    return list(rows), [note.kind for note in notes]


def row(form_id: str, move: str, method: LearnMethod, **fields: object) -> LearnsetRow:
    values = {"level": None, "machine": None, **fields}
    return LearnsetRow(form_id=form_id, move=move, method=method, **values)  # type: ignore[arg-type]


def test_level_rows_with_numbers_and_markers() -> None:
    rows, notes = read_rows(page(LEVEL_TABLE))

    assert rows == [
        row("testmon", "Sand Attack", LearnMethod.LEVEL, level=1),
        row("testmon", "Earthquake", LearnMethod.LEVEL, level=40),
        row("testmon", "Mud Shot", LearnMethod.EVOLUTION),
        row("testmon", "Fissure", LearnMethod.REMINDER),
    ]
    assert notes == []


def test_machine_egg_and_special_rows() -> None:
    wikitext = page(
        """====By [[TM]]====
{{learnlist/tmh/9|Testmon|Ground|Ground|1}}
{{learnlist/tm9|TM149|Earthquake|Ground|Physical|100|100|10||'''}}
{{learnlist/tmf/9|Testmon|Ground|Ground|1}}
====By {{pkmn|breeding}}====
{{learnlist/breedh/9|Testmon|Ground|Ground|1}}
{{learnlist/breed9|{{MSP/H|0206|Dunsparce}}|Ancient Power|Rock|Special|60|100|5}}
====Special moves====
{{learnlist/event9|[[Area|Wild Tera Pokémon]]|Rock Tomb|Rock|Physical|60|95|15}}
"""
    )

    rows, _ = read_rows(wikitext)

    assert rows == [
        row("testmon", "Earthquake", LearnMethod.MACHINE, machine="TM149"),
        row("testmon", "Ancient Power", LearnMethod.EGG),
        row("testmon", "Rock Tomb", LearnMethod.SPECIAL),
    ]


def test_rows_of_other_games_and_prevo_rows_are_ignored() -> None:
    wikitext = page(
        """====By leveling up====
{{learnlist/levelh/9|Testmon|Ground|Ground|1}}
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
{{learnlist/levelf/9|Testmon|Ground|Ground|1}}
{{learnlist/levelh/ZA|Testmon|Ground|Ground|1}}
{{learnlist/levelZA|1|Mud Shot|Ground|Special|55|95|15}}
{{learnlist/levelf/9|Testmon|Ground|Ground|1}}
{{learnlist/level8|1|Dig|Ground|Physical|80|100|10}}
{{learnlist/levelLA|1|Bulldoze|Ground|Physical|60|100|20}}
====By TM====
{{learnlist/tmZA|TM001|Take Down|Normal|Physical|90|85|20}}
{{learnlist/tm8|TM001|Take Down|Normal|Physical|90|85|20}}
====By a prior Evolution====
{{learnlist/prevo9|Digletto|Ground|Ground|1}}
{{learnlist/prevo9null}}
{{learnlist/breed9null}}
"""
    )

    rows, notes = read_rows(wikitext)

    assert rows == [row("testmon", "Sand Attack", LearnMethod.LEVEL, level=1)]
    assert notes == []


def test_rows_under_form_subsections_belong_to_those_forms() -> None:
    wikitext = page(
        """====By leveling up====
=====Testmon=====
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
=====Alolan Testmon=====
{{learnlist/level9|1|Metal Claw|Steel|Physical|50|95|35}}
{{learnlist/level9|4|Growl|Normal|Status|—|100|40}}
====By TM====
=====Alolan Testmon=====
{{learnlist/tm9|TM001|Take Down|Normal|Physical|90|85|20}}
"""
    )

    rows, _ = read_rows(wikitext, ALOLA_INFOBOX)

    assert [(r.form_id, r.move) for r in rows] == [
        ("testmon", "Sand Attack"),
        ("testmon-alola", "Metal Claw"),
        ("testmon-alola", "Growl"),
        ("testmon-alola", "Take Down"),
    ]


def test_a_regional_title_gives_its_rows_to_every_form_of_the_region() -> None:
    infobox = {
        "name": "Tauros",
        "form2": "Paldean Form (Combat Breed)",
        "form3": "Paldean Form (Blaze Breed)",
        "image2": "0128Tauros-Paldea Combat.png",
        "image3": "0128Tauros-Paldea Blaze.png",
        "form2type1": "Fighting",
        "form3type1": "Fighting",
        "type1": "Normal",
    }
    wikitext = page(
        """====By leveling up====
=====Paldean Tauros=====
{{learnlist/level9|1|Tackle|Normal|Physical|40|100|35}}
"""
    )

    rows, _ = read_rows(wikitext, infobox)

    assert [r.form_id for r in rows] == ["tauros-paldea-combat", "tauros-paldea-blaze"]


def test_rows_under_an_unmatched_title_are_dropped_without_a_missing_note() -> None:
    wikitext = page(
        """====By leveling up====
=====Partner Testmon=====
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
"""
    )
    forms, sections = build_forms(BASE_INFOBOX, wikitext)

    rows, notes = read_learnset(sections, forms, PAGE_TITLE)

    assert rows == ()
    assert notes == ()
    assert [n.kind for n in forms.notes] == [NoteKind.UNMATCHED_SUBSECTION]


def test_repeated_rows_collapse() -> None:
    wikitext = page(
        """====By leveling up====
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
{{learnlist/level9|20|Sand Attack|Ground|Status|—|100|15}}
"""
    )

    rows, _ = read_rows(wikitext)

    assert [(r.move, r.level) for r in rows] == [("Sand Attack", 1), ("Sand Attack", 20)]


def test_a_page_with_only_other_game_rows_gets_a_missing_learnset_note() -> None:
    wikitext = page(
        """====By leveling up====
{{learnlist/levelZA|1|Sand Attack|Ground|Status|—|100|15}}
"""
    )

    rows, notes = read_rows(wikitext)

    assert rows == []
    assert notes == [NoteKind.MISSING_LEARNSET]


def test_a_page_without_a_learnset_section_gets_a_missing_learnset_note() -> None:
    assert read_rows("==Trivia==\ntext")[1] == [NoteKind.MISSING_LEARNSET]


def test_list_learnset_titles_keeps_form_subsections_with_scarlet_violet_rows() -> None:
    sections = split_sections(
        page(
            """====By leveling up====
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
=====Testmon=====
{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}
=====Alolan Testmon=====
{{learnlist/level9|1|Metal Claw|Steel|Physical|50|95|35}}
=====Pokémon Legends: Z-A=====
{{learnlist/levelZA|1|Mud Shot|Ground|Special|55|95|15}}
====By TM====
=====Alolan Testmon=====
{{learnlist/tm9|TM001|Take Down|Normal|Physical|90|85|20}}
"""
        )
    )

    assert list_learnset_titles(sections) == ("Testmon", "Alolan Testmon", "Alolan Testmon")


@pytest.mark.parametrize(
    ("cell", "field"),
    [("{{tt|Hmm|x}}", "level"), ("", "level"), ("abc", "level")],
)
def test_an_unreadable_level_fails_the_page(cell: str, field: str) -> None:
    wikitext = page(
        f"====By leveling up====\n{{{{learnlist/level9|{cell}|Dig|Ground|Physical|80|100|10}}}}\n"
    )

    with pytest.raises(PageParseError) as error:
        read_rows(wikitext)

    assert error.value.field == field
    assert error.value.page_title == PAGE_TITLE


def test_a_machine_that_is_not_a_tm_number_fails_the_page() -> None:
    wikitext = page("====By TM====\n{{learnlist/tm9|Dig|Dig|Ground|Physical|80|100|10}}\n")

    with pytest.raises(PageParseError, match="machine"):
        read_rows(wikitext)


def test_a_row_without_a_move_fails_the_page() -> None:
    wikitext = page("====By leveling up====\n{{learnlist/level9|1}}\n")

    with pytest.raises(PageParseError, match="move"):
        read_rows(wikitext)
