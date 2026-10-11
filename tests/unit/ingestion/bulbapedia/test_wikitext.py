import mwparserfromhell
import pytest

from muon.ingestion.bulbapedia.wikitext import (
    LEAD_LEVEL,
    PLAIN_TEXT_SEPARATOR,
    TEXT_TEMPLATES,
    find_sections,
    find_templates,
    normalize_heading,
    normalize_template_name,
    read_params,
    read_small_captions,
    split_sections,
    to_plain_text,
)

NESTED_PAGE = """lead text
==Game data==
game text
===Stats===
====Base stats====
stats text
===Learnset===
learnset text
==Trivia==
trivia text
"""

# Markup left open in one section pairs with markup in a later one and hides the headings
# between them from a full parse. On Abra a stray `''` does it.
UNBALANCED_PAGES = {
    "template": "{{Pokémon Infobox\n|name=Abra\n==Game data==\n===Learnset===\nrows\n==Trivia==\n",
    "italic": "==Game data==\nA ''move.\n===Learnset===\nrows\n==Trivia==\nSee ''Abra''.\n",
}


def titles(wikitext: str) -> list[tuple[str, ...]]:
    return [section.path for section in split_sections(wikitext)]


def test_split_sections_puts_the_lead_first() -> None:
    lead = split_sections(NESTED_PAGE)[0]

    assert (lead.level, lead.title, lead.path) == (LEAD_LEVEL, "", ())
    assert lead.text == "lead text\n"


def test_split_sections_keeps_the_open_ancestors_in_the_path() -> None:
    assert titles(NESTED_PAGE) == [
        (),
        ("Game data",),
        ("Game data", "Stats"),
        ("Game data", "Stats", "Base stats"),
        ("Game data", "Learnset"),
        ("Trivia",),
    ]


def test_split_sections_gives_each_section_its_body_only() -> None:
    bodies = {section.title: section.text for section in split_sections(NESTED_PAGE)}

    assert bodies["Base stats"] == "\nstats text\n"
    assert bodies["Game data"] == "\ngame text\n"
    assert bodies["Trivia"] == "\ntrivia text\n"


def test_split_sections_without_headings_returns_the_lead_only() -> None:
    sections = split_sections("just a lead")

    assert [section.path for section in sections] == [()]
    assert sections[0].text == "just a lead"


@pytest.mark.parametrize("wikitext", UNBALANCED_PAGES.values(), ids=UNBALANCED_PAGES.keys())
def test_split_sections_finds_headings_after_unbalanced_markup(wikitext: str) -> None:
    sections = split_sections(wikitext)

    assert [section.path for section in sections[1:]] == [
        ("Game data",),
        ("Game data", "Learnset"),
        ("Trivia",),
    ]
    assert sections[2].parse().strip() == "rows"


def test_split_sections_ignores_a_heading_inside_a_comment() -> None:
    wikitext = "==Game data==\ntable\n|}<!--\n\n====Mega Excadrill====\nrow\n-->\n==Trivia==\n"

    sections = split_sections(wikitext)

    assert [section.path for section in sections[1:]] == [("Game data",), ("Trivia",)]
    assert "====Mega Excadrill====" in sections[1].text
    assert "-->" in sections[1].text


def test_split_sections_ignores_headings_in_an_unterminated_comment() -> None:
    wikitext = "==Game data==\ntext\n<!--\n==Hidden==\n===Also hidden===\n"

    assert [section.title for section in split_sections(wikitext)] == ["", "Game data"]


def test_split_sections_allows_a_comment_after_and_inside_a_heading() -> None:
    wikitext = "==Game data== <!-- note -->\ntext\n==By <!--x-->Level==\nrows\n"

    sections = split_sections(wikitext)

    assert [section.title for section in sections[1:]] == ["Game data", "By Level"]
    assert sections[1].text.strip() == "text"


def test_split_sections_reads_levels_two_to_six() -> None:
    wikitext = "==A==\n===B===\n======F======\n"

    assert [section.level for section in split_sections(wikitext)] == [LEAD_LEVEL, 2, 3, 6]


def test_split_sections_ignores_unbalanced_equal_signs() -> None:
    wikitext = "x = y\n=Not a heading=\n== Real ==\n"

    assert [section.title for section in split_sections(wikitext)] == ["", "Real"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("By {{pkmn|breeding}}", "By breeding"),
        ("[[Stat]]s", "Stats"),
        ("[[Move|Moves]] &amp; TMs", "Moves & TMs"),
        ("Alolan&nbsp;Diglett", "Alolan Diglett"),
        ("  Spaced   out  ", "Spaced out"),
        ("Mega {{p|Charizard}}<ref>note</ref>", "Mega Charizard"),
    ],
)
def test_normalize_heading(raw: str, expected: str) -> None:
    assert normalize_heading(raw) == expected


def test_find_sections_matches_a_path_prefix() -> None:
    sections = split_sections(NESTED_PAGE)

    found = find_sections(sections, ("Game data", "Stats"))

    assert [section.path[-1] for section in found] == ["Stats", "Base stats"]
    assert find_sections(sections, ("Missing",)) == ()


def test_normalize_template_name() -> None:
    assert normalize_template_name(" Evobox-2\n") == "evobox-2"
    assert normalize_template_name("Learnlist/level9") == "learnlist/level9"
    assert normalize_template_name("Pokémon_Infobox") == "pokémon infobox"


def test_find_templates_matches_names_in_any_case() -> None:
    code = mwparserfromhell.parse("{{Evobox-2|a=1}} {{evobox-2|a=2}} {{Other|a=3}}")

    found = find_templates(code, ["Evobox-2"])

    assert [read_params(template)["a"] for template in found] == ["1", "2"]


def test_find_templates_reaches_nested_templates_in_document_order() -> None:
    code = mwparserfromhell.parse("{{Outer|x={{inner|n=1}}}} {{Inner|n=2}}")

    found = find_templates(code, ["inner"])

    assert [read_params(template)["n"] for template in found] == ["1", "2"]


def test_read_params_strips_names_and_keeps_raw_values() -> None:
    code = mwparserfromhell.parse("{{Box | name = Diglett | note = {{tt|a|b}} <!--c--> |1x}}")
    template = find_templates(code, ["box"])[0]

    assert read_params(template) == {
        "name": "Diglett",
        "note": "{{tt|a|b}} <!--c-->",
        "1": "1x",
    }


def test_read_params_keeps_the_last_repeat() -> None:
    template = find_templates(mwparserfromhell.parse("{{Box|a=1|a=2}}"), ["box"])[0]

    assert read_params(template) == {"a": "2"}


TEMPLATE_CASES = [
    ("{{tt|Evo.|Evolves}}", "Evo."),
    ("{{color2|000|Fire Stone|text}}", "text"),
    ("{{color2|000|Fire Stone}}", "Fire Stone"),
    ("{{mcolor|Surf}}", "Surf"),
    ("{{mcolor|Surf|x|Shown}}", "Shown"),
    ("{{pcolor|Diglett}}", "Diglett"),
    ("{{acolor|Sand Veil}}", "Sand Veil"),
    ("{{tcolor|Ground}}", "Ground"),
    ("{{p|Diglett}}", "Diglett"),
    ("{{m|Earthquake}}", "Earthquake"),
    ("{{a|Sand Veil}}", "Sand Veil"),
    ("{{type|Ground}}", "Ground"),
    ("{{t|Ground}}", "Ground"),
    ("{{gen|VII}}", "VII"),
    ("{{rf|Cite}}", "Cite"),
    ("{{pkmn|breeding}}", "breeding"),
    ("{{OBP|Diglett|Sand}}", "Diglett"),
    ("{{ScPkmn}}", "Pokémon"),
    ("{{!}}", "|"),
]


def test_every_text_template_has_a_case() -> None:
    covered = {
        normalize_template_name(case.split("|")[0].strip("{}")) for case, _ in TEMPLATE_CASES
    }

    assert covered == set(TEXT_TEMPLATES)


@pytest.mark.parametrize(("value", "expected"), TEMPLATE_CASES)
def test_to_plain_text_renders_text_templates(value: str, expected: str) -> None:
    assert to_plain_text(value) == expected


def test_to_plain_text_drops_other_templates() -> None:
    assert to_plain_text("a {{bag/s|Fire Stone}} b") == "a b"


def test_to_plain_text_renders_nested_template_values() -> None:
    assert to_plain_text("{{tt|{{p|Diglett}}|caption}}") == "Diglett"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[[Ground (type)|Ground]]", "Ground"),
        ("[[Diglett]]", "Diglett"),
        ("[[File:0050Diglett.png|40px]]x", "x"),
        ("[[Image:Old.png]]x", "x"),
        ("x<ref name=a>note</ref>y", "xy"),
        ("x<ref name=a />y", "xy"),
        ("x<sup>1</sup>y", "xy"),
        ("x<small>caption</small>y", "xy"),
        ("x<!-- hidden -->y", "xy"),
        ("a&nbsp;b", "a b"),
        ("&eacute;", "é"),
        ('<span style="x">kept</span>', "kept"),
        ("''Romaji''", "Romaji"),
        ("a   b \t c", "a b c"),
    ],
)
def test_to_plain_text(value: str, expected: str) -> None:
    assert to_plain_text(value) == expected


def test_to_plain_text_turns_br_into_the_separator() -> None:
    assert to_plain_text("a<br>b<br />c") == f"a{PLAIN_TEXT_SEPARATOR}b{PLAIN_TEXT_SEPARATOR}c"


def test_to_plain_text_collapses_whitespace_per_line() -> None:
    assert to_plain_text("a  b <br>  c   d  ") == "a b\nc d"


def test_read_small_captions_returns_each_caption_in_order() -> None:
    value = "Gorilla Tactics<small>Galarian Form</small> or Zen<small>{{tt|Zen|x}} mode</small>"

    assert read_small_captions(value) == ("Galarian Form", "Zen mode")


def test_read_small_captions_returns_nothing_without_captions() -> None:
    assert read_small_captions("Sand Veil") == ()
