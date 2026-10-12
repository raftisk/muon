import mwparserfromhell
import pytest

from muon.ingestion.bulbapedia.wikitext import (
    LEAD_LEVEL,
    PLAIN_TEXT_SEPARATOR,
    TEXT_TEMPLATES,
    CapturedTemplate,
    TextRule,
    find_sections,
    find_templates,
    normalize_heading,
    normalize_template_name,
    read_params,
    read_plain_text,
    read_small_captions,
    render_rule,
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


# The cases marked "API" hold the text of Bulbapedia's own rendering of the example, read
# once through `action=expandtemplates` on 2026-10-12.
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
    ("{{p|Dugtrio|its evolution}}", "its evolution"),  # API
    ("{{m|Dig}}", "Dig"),  # API
    ("{{m|Fusion Flare|an enormous flame}}", "an enormous flame"),  # API
    ("{{a|Tangling Hair}}", "Tangling Hair"),  # API
    ("{{a|Sturdy|extremely hard}}", "extremely hard"),  # API
    ("{{type|Ground}}", "Ground-type"),  # API
    ("{{type|Dragon|Dragon-type}}", "Dragon-type"),  # API
    ("{{t|Steel}}", "Steel"),  # API
    ("{{t|Grass|type}}", "type"),  # API
    ("{{gen|VII}}", "VII"),
    ("{{rf|Cite}}", "Cite"),
    ("{{pkmn|breeding}}", "breeding"),
    ("{{OBP|Diglett|Sand}}", "Diglett"),
    ("{{OBP|Pokémon|species}}", "Pokémon"),  # API
    ("{{OBP|Ticketed event|GO|Ticketed Special Research}}", "Ticketed Special Research"),  # API
    ("{{wp|mole (animal)|mole}}", "mole"),  # API
    ("{{wp|skunk}}", "skunk"),  # API
    ("{{g|Sun and Moon}}", "Pokémon Sun and Moon"),  # API
    ("{{g|Black 2 and White 2|s}}", "Pokémon Black 2 and White 2"),  # API
    ("{{game|Red and Blue|s}}", "Pokémon Red and Blue"),  # API
    ("{{game|Black and White|s|Pokémon Black}}", "Pokémon Black"),  # API
    ("{{game|Platinum||Platinum}}", "Platinum"),  # API
    ("{{DL|List of Pokémon by habitat|Cave Pokémon|caves}}", "caves"),  # API
    ("{{DL|PML arc|Chapter 1: New Adventure}}", "Chapter 1: New Adventure"),  # API
    ("{{stat|Special Attack}}", "Special Attack"),  # API
    ("{{stat|Attack|physically}}", "physically"),  # API
    ("{{pkmn2|event}}", "event"),  # API
    ("{{pkmn2|Talking|talk}}", "talk"),  # API
    ("{{egg|Field}}", "Field"),  # API
    ("{{egg3|Flying}}", "Flying Egg Group"),  # API
    ("{{Shiny}}", "Shiny"),  # API
    ("{{game3|Black and White|Pokémon White|s}}", "Pokémon White"),  # API
    ("{{game3|FireRed and LeafGreen|FireRed|s}}", "FireRed"),  # API
    ("{{2v2|HeartGold|SoulSilver}}", "HeartGold and SoulSilver"),  # API
    ("{{2v2|Black|White|2}}", "Black 2 and White 2"),  # API
    ("{{B2W2}}", "Pokémon Black 2 and White 2"),  # API
    ("{{B2W2|Black 2}}", "Black 2"),  # API
    ("{{cat|Generation I Pokémon}}", "Generation I Pokémon"),  # API
    (
        "{{cat|Pokémon with a base stat total of 680|base stat total of 680}}",
        "base stat total of 680",
    ),  # API
    ("{{form|Castform|Sunny Form}}", "Sunny Form"),  # API
    ("{{TCG|HP}}", "HP"),  # API
    ("{{TCG|Sword & Shield|Sword & Shield set}}", "Sword & Shield set"),  # API
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


def test_to_plain_text_drops_layout_templates_that_carry_no_prose() -> None:
    value = "a {{bulbanews|On the Origin of Species}} b {{left clear}} c {{aniseries|BW}} d"

    assert to_plain_text(value) == "a b c d"


def test_to_plain_text_drops_other_templates() -> None:
    assert to_plain_text("a {{bag/s|Fire Stone}} b") == "a b"


def test_to_plain_text_renders_nested_template_values() -> None:
    assert to_plain_text("{{tt|{{p|Diglett}}|caption}}") == "Diglett"


def test_read_plain_text_captures_typed_templates_with_their_targets() -> None:
    value = (
        "{{p|Dugtrio|its evolution}} learns {{m|Dig}}, has {{a|Sand Veil}}, "
        "{{type|Ground}} {{t|Steel}}"
    )

    result = read_plain_text(value, frozenset({"p", "m", "a", "type", "t"}))

    assert result.text == "its evolution learns Dig, has Sand Veil, Ground-type Steel"
    assert result.captured == (
        CapturedTemplate("p", "Dugtrio"),
        CapturedTemplate("m", "Dig"),
        CapturedTemplate("a", "Sand Veil"),
        CapturedTemplate("type", "Ground"),
        CapturedTemplate("t", "Steel"),
    )


def test_read_plain_text_captures_only_the_requested_names() -> None:
    result = read_plain_text("{{p|Onix}} and {{m|Dig}}", frozenset({"m"}))

    assert result.captured == (CapturedTemplate("m", "Dig"),)


def test_read_plain_text_strips_the_target_and_skips_an_empty_one() -> None:
    result = read_plain_text("{{p| Onix }} {{p|}} {{p}}", frozenset({"p"}))

    assert result.captured == (CapturedTemplate("p", "Onix"),)


def test_read_plain_text_captures_a_template_nested_in_a_displayed_value() -> None:
    result = read_plain_text("{{wp|Mole|the {{p|Diglett}}}}", frozenset({"p"}))

    assert result.text == "the Diglett"
    assert result.captured == (CapturedTemplate("p", "Diglett"),)


@pytest.mark.parametrize(
    "value",
    [
        "x<ref>{{p|Onix}}</ref>y",
        "x<small>{{p|Onix}}</small>y",
        "x<sup>{{p|Onix}}</sup>y",
        "x<!-- {{p|Onix}} -->y",
        "x[[File:Onix.png|thumb|{{p|Onix}}]]y",
        "x{{bag/s|{{p|Onix}}}}y",
    ],
)
def test_read_plain_text_ignores_templates_in_removed_parts(value: str) -> None:
    result = read_plain_text(value, frozenset({"p"}))

    assert result.text == "xy"
    assert result.captured == ()


def test_read_plain_text_lists_dropped_templates_in_page_order_with_repeats() -> None:
    result = read_plain_text("a {{roundy|x}} b {{aniseries|BW}} c {{Roundy|y}} {{p|Onix}}")

    assert result.text == "a b c Onix"
    assert result.dropped == ("roundy", "aniseries", "roundy")


def test_read_plain_text_does_not_list_templates_inside_removed_parts() -> None:
    result = read_plain_text("x<ref>{{roundy}}</ref>[[File:A.png|{{aniseries|BW}}]]y")

    assert result.dropped == ()


def test_read_plain_text_matches_to_plain_text() -> None:
    value = "{{p|Dugtrio|its evolution}}<br>{{m|Dig}}  [[File:A.png]] {{roundy}}"

    assert read_plain_text(value).text == to_plain_text(value)


def test_render_rule_wraps_only_the_last_parameter() -> None:
    rule = TextRule(params=("2", "1"), literal="none", prefix="Pre ", suffix=" post")
    wrapped = mwparserfromhell.parse("{{x|Named}}").filter_templates()[0]
    shown = mwparserfromhell.parse("{{x|Named|Shown}}").filter_templates()[0]
    bare = mwparserfromhell.parse("{{x}}").filter_templates()[0]

    assert render_rule(wrapped, rule, frozenset()).text == "Pre Named post"
    assert render_rule(shown, rule, frozenset()).text == "Shown"
    assert render_rule(bare, rule, frozenset()).text == "none"


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
