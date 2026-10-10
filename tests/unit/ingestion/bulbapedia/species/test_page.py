from collections.abc import Mapping

import pytest

from muon.ingestion.bulbapedia.species.extract import (
    AbilitySlot,
    FormFacts,
    NoteKind,
    PageNote,
    Residue,
    ResidueKind,
)
from muon.ingestion.bulbapedia.species.forms import FormInputs, FormSet, resolve_forms
from muon.ingestion.bulbapedia.species.page import (
    FactInputs,
    PageParseError,
    list_stats_titles,
    parse_generation_range,
    read_abilities,
    read_aliases,
    read_base_stats,
    read_categories,
    read_form_facts,
    read_genus,
    read_infobox,
    split_top_level,
)
from muon.ingestion.bulbapedia.wikitext import split_sections

PAGE_TITLE = "Testmon (Pokémon)"

BASE_INFOBOX = {
    "name": "Testmon",
    "ndex": "0050",
    "type1": "Ground",
    "ability1": "Sand Veil",
    "egggroupn": "1",
    "egggroup1": "Field",
    "eggcycles": "20",
    "catchrate": "255",
    "expyield": "53",
    "lv100exp": "1,000,000",
    "gendercode": "127",
    "friendship": "70",
    "color": "Brown",
    "body": "05",
    "generation": "1",
    "height-m": "0.2",
    "weight-kg": "0.8",
    "category": "Mole",
    "evsp": "1",
}

STATS_TEMPLATE = (
    "{{Stats|type=Ground|HP=10|Attack=55|Defense=25|SpAtk=35|SpDef=45|Speed=95|Special=45}}"
)


def stats_page(*subsections: tuple[str, str]) -> str:
    """Wikitext with a Base stats section; each pair is (heading line, body)."""
    body = "".join(f"\n{heading}\n{text}\n" for heading, text in subsections)
    return f"==Game data==\n===Stats===\n===={'Base stats'}===={body}"


DEFAULT_STATS_PAGE = stats_page(("=====Testmon=====", STATS_TEMPLATE))


def infobox_with(**fields: str) -> dict[str, str]:
    return {**BASE_INFOBOX, **fields}


def build_inputs(
    infobox: Mapping[str, str],
    wikitext: str = DEFAULT_STATS_PAGE,
    categories: frozenset[str] = frozenset(),
) -> FactInputs:
    sections = split_sections(wikitext)
    forms = resolve_forms(
        FormInputs(PAGE_TITLE, infobox, list_stats_titles(sections), learnset_titles=())
    )
    return FactInputs(PAGE_TITLE, infobox, forms, sections, categories)


def read_facts(
    infobox: Mapping[str, str],
    wikitext: str = DEFAULT_STATS_PAGE,
    categories: frozenset[str] = frozenset(),
) -> dict[str, FormFacts]:
    facts, _, _ = read_form_facts(build_inputs(infobox, wikitext, categories))
    return {form.id: form for form in facts}


def read_all(
    infobox: Mapping[str, str], wikitext: str = DEFAULT_STATS_PAGE
) -> tuple[dict[str, FormFacts], tuple[Residue, ...], tuple[PageNote, ...]]:
    facts, residue, notes = read_form_facts(build_inputs(infobox, wikitext))
    return {form.id: form for form in facts}, residue, notes


def ability_names(facts: FormFacts) -> list[tuple[str, bool]]:
    return [(slot.name, slot.is_hidden) for slot in facts.abilities]


def read_form_set(infobox: Mapping[str, str], wikitext: str = DEFAULT_STATS_PAGE) -> FormSet:
    return build_inputs(infobox, wikitext).forms


def test_read_infobox_strips_comments_and_reads_params() -> None:
    lead = split_sections("{{Pokémon Infobox\n|name=Rotom<!-- x -->\n|ndex=0479\n}}\ntext")[0]

    assert read_infobox(lead) == {"name": "Rotom", "ndex": "0479"}


def test_read_infobox_returns_none_without_the_template() -> None:
    lead = split_sections("{{New Pokémon Infobox|name=Browt}}")[0]

    assert read_infobox(lead) is None


def test_read_categories() -> None:
    wikitext = "[[Category:Legendary Pokémon]]\n[[category: Mythical Pokémon |x]]\n[[Mew]]"

    assert read_categories(wikitext) == {"Legendary Pokémon", "Mythical Pokémon"}


def test_species_values_come_from_the_infobox() -> None:
    facts = read_facts(infobox_with(type2="Steel", egggroup2="Water 1", egggroupn="2"))["testmon"]

    assert (facts.pokedex_number, facts.generation) == (50, 1)
    assert facts.types == ("Ground", "Steel")
    assert facts.egg_groups == ("Field", "Water 1")
    assert (facts.catch_rate, facts.base_exp_yield, facts.egg_cycles) == (255, 53, 20)
    assert facts.base_friendship == 70
    assert (facts.color, facts.shape) == ("Brown", "blob")
    assert (facts.height, facts.weight) == (0.2, 0.8)
    assert facts.genus == "Mole"
    assert facts.is_default
    assert facts.region is None
    assert facts.base_stats.speed == 95


@pytest.mark.parametrize(
    ("code", "female_rate", "is_genderless"),
    [
        ("0", 0.0, False),
        ("31", 0.125, False),
        ("63", 0.25, False),
        ("127", 0.5, False),
        ("191", 0.75, False),
        ("223", 0.875, False),
        ("254", 1.0, False),
        ("255", None, True),
    ],
)
def test_gender_code(code: str, female_rate: float | None, is_genderless: bool) -> None:
    facts = read_facts(infobox_with(gendercode=code))["testmon"]

    assert (facts.female_rate, facts.is_genderless) == (female_rate, is_genderless)


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        ("600,000", "erratic"),
        ("800000", "fast"),
        ("1,000,000", "medium_fast"),
        ("1059860", "medium_slow"),
        ("1,250,000", "slow"),
        ("1,640,000", "fluctuating"),
    ],
)
def test_growth_rate(total: str, expected: str) -> None:
    assert read_facts(infobox_with(lv100exp=total))["testmon"].growth_rate == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [("01", "ball"), ("08", "quadruped"), ("13", "bug_wings"), ("14", "armor"), ("", None)],
)
def test_body_shape(code: str, expected: str | None) -> None:
    assert read_facts(infobox_with(body=code))["testmon"].shape == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gendercode", "100"),
        ("lv100exp", "1,111,111"),
        ("body", "15"),
        ("catchrate", "fast"),
        ("friendship", ""),
        ("ndex", "0050a"),
    ],
)
def test_unreadable_values_fail_the_page(field: str, value: str) -> None:
    with pytest.raises(PageParseError) as error:
        read_facts(infobox_with(**{field: value}))

    assert error.value.page_title == PAGE_TITLE
    assert error.value.field == field
    assert field in str(error.value)
    assert PAGE_TITLE in str(error.value)


def test_a_missing_ndex_fails_the_page() -> None:
    infobox = {key: value for key, value in BASE_INFOBOX.items() if key != "ndex"}

    with pytest.raises(PageParseError, match="ndex"):
        read_facts(infobox)


def test_a_missing_type1_fails_the_page() -> None:
    infobox = {key: value for key, value in BASE_INFOBOX.items() if key != "type1"}

    with pytest.raises(PageParseError, match="type1"):
        read_facts(infobox)


def test_category_flags() -> None:
    categories = frozenset({"Legendary Pokémon", "First partner Pokémon", "Kanto Pokémon"})

    facts = read_facts(BASE_INFOBOX, categories=categories)["testmon"]

    assert (facts.is_legendary, facts.is_mythical, facts.is_starter) == (True, False, True)


FORM_INFOBOX = infobox_with(
    form2="Alolan Form",
    image2="0050Testmon-Alola.png",
    form2type1="Ground",
    form2type2="Steel",
    **{"height-m2": "0.4", "weight-kg2": "???.?"},
)


def test_a_form_takes_its_own_types_and_measures() -> None:
    facts = read_facts(FORM_INFOBOX)

    alola = facts["testmon-alola"]
    assert alola.types == ("Ground", "Steel")
    assert alola.height == 0.4
    assert not alola.is_default
    assert alola.region == "Alola"
    assert alola.generation is None
    assert alola.aliases == ()
    assert facts["testmon"].types == ("Ground",)


def test_a_non_numeric_measure_is_left_out_and_noted() -> None:
    facts, _, notes = read_all(FORM_INFOBOX)

    assert facts["testmon-alola"].weight is None
    assert [(note.kind, note.section_path) for note in notes] == [
        (NoteKind.DROPPED_VALUE, "Pokémon Infobox > weight-kg2")
    ]
    assert "testmon-alola" in notes[0].detail


def test_a_form_without_measures_copies_the_base_values() -> None:
    infobox = infobox_with(form2="Alolan Form", image2="0050Testmon-Alola.png", form2type1="Ground")

    alola = read_facts(infobox)["testmon-alola"]

    assert (alola.height, alola.weight) == (0.2, 0.8)


def test_color_and_shape_are_not_copied_when_forms_set_their_own() -> None:
    infobox = {**FORM_INFOBOX, "formcolors": "yes", "formbody": "yes"}

    facts = read_facts(infobox)

    assert (facts["testmon"].color, facts["testmon"].shape) == ("Brown", "blob")
    assert (facts["testmon-alola"].color, facts["testmon-alola"].shape) == (None, None)
    copied = read_facts(FORM_INFOBOX)["testmon-alola"]
    assert (copied.color, copied.shape) == ("Brown", "blob")


def test_ev_yield_copies_the_base_without_evforms() -> None:
    infobox = infobox_with(evat="1", evsp="2", evat2="3")

    facts = read_facts({**FORM_INFOBOX, **infobox})

    assert facts["testmon"].ev_yield == {"evat": 1, "evsp": 2}
    assert facts["testmon-alola"].ev_yield == {"evat": 1, "evsp": 2}


def test_ev_yield_reads_own_keys_for_forms_evforms_covers() -> None:
    infobox = {
        **FORM_INFOBOX,
        "evforms": "2",
        "evat": "1",
        "evsp": "1",
        "evat2": "2*",
        "evde2": "0*",
    }

    facts = read_facts(infobox)

    assert facts["testmon"].ev_yield == {"evat": 1, "evsp": 1}
    assert facts["testmon-alola"].ev_yield == {"evat": 2}


def test_an_ev_amount_above_three_fails_the_page() -> None:
    with pytest.raises(PageParseError, match="evat"):
        read_facts(infobox_with(evat="4"))


def test_abilities_of_a_base_and_a_regional_form_with_a_shared_hidden_ability() -> None:
    infobox = {
        **FORM_INFOBOX,
        "ability2": "Arena Trap",
        "ability2-1": "Sand Veil",
        "ability2-2": "Tangling Hair",
        "abilityd": "Sand Force",
    }

    facts = read_facts(infobox)

    assert ability_names(facts["testmon"]) == [
        ("Sand Veil", False),
        ("Arena Trap", False),
        ("Sand Force", True),
    ]
    assert ability_names(facts["testmon-alola"]) == [
        ("Sand Veil", False),
        ("Tangling Hair", False),
        ("Sand Force", True),
    ]


def test_mega_forms_take_abilitym_in_order_and_no_hidden_ability() -> None:
    infobox = infobox_with(
        name="Testmon",
        ability1="Blaze",
        abilityd="Solar Power",
        form2="Mega Testmon X",
        form3="Mega Testmon Y",
        image2="0050Testmon-Mega X.png",
        image3="0050Testmon-Mega Y.png",
        form2type1="Fire",
        form3type1="Fire",
        abilitym="Tough Claws",
        abilitym2="Drought",
    )

    facts = read_facts(infobox)

    assert ability_names(facts["testmon"]) == [("Blaze", False), ("Solar Power", True)]
    assert ability_names(facts["testmon-mega-x"]) == [("Tough Claws", False)]
    assert ability_names(facts["testmon-mega-y"]) == [("Drought", False)]


TAUROS_LIKE = infobox_with(
    name="Tauros",
    form2="Paldean Form (Combat Breed)",
    form3="Paldean Form (Blaze Breed)",
    image2="0128Tauros-Paldea Combat.png",
    image3="0128Tauros-Paldea Blaze.png",
    form2type1="Fighting",
    form3type1="Fighting",
    ability1="Intimidate",
    abilityd="Sheer Force",
    abilitydcaption="Tauros",
    abilityd2="Cud Chew",
    abilityd2caption="Paldean Form",
)


def test_hidden_abilities_follow_their_captions() -> None:
    wikitext = stats_page(("=====Tauros=====", STATS_TEMPLATE))

    facts = read_facts(TAUROS_LIKE, wikitext)

    assert ability_names(facts["tauros"]) == [("Intimidate", False), ("Sheer Force", True)]
    for form_id in ("tauros-paldea-combat", "tauros-paldea-blaze"):
        assert ability_names(facts[form_id]) == [("Intimidate", False), ("Cud Chew", True)]


MEOWTH_LIKE = infobox_with(
    name="Meowth",
    form2="Alolan Form",
    form3="Galarian Form",
    image2="0052Meowth-Alola.png",
    image3="0052Meowth-Galar.png",
    form2type1="Dark",
    form3type1="Steel",
    ability1="Pickup",
    ability2="Technician",
    ability2note=(
        "<sup>{{color2|000|Generation IV|Gen IV+}}</sup> <small>(Meowth)</small><br>"
        "{{acolor|Pickup|000}} or {{acolor|Technician|000}} <small>(Alolan Form)</small><br>"
        "{{acolor|Pickup|000}} or {{acolor|Tough Claws|000}} <small>(Galarian Form)</small>"
    ),
    abilityd="Unnerve",
    abilitydcaption="Meowth and Galarian Form<br>",
    abilityd2="Rattled",
    abilityd2caption="Alolan Form<br>",
)


def test_ability_note_naming_forms_gives_residue_and_no_visible_ability() -> None:
    wikitext = stats_page(("=====Meowth=====", STATS_TEMPLATE))

    facts, residue, _ = read_all(MEOWTH_LIKE, wikitext)

    assert ability_names(facts["meowth"]) == [
        ("Pickup", False),
        ("Technician", False),
        ("Unnerve", True),
    ]
    assert ability_names(facts["meowth-alola"]) == [("Rattled", True)]
    assert ability_names(facts["meowth-galar"]) == [("Unnerve", True)]
    assert [(r.kind, r.form_ids, r.section_path) for r in residue] == [
        (ResidueKind.ABILITIES, ("meowth-alola",), "Pokémon Infobox > ability2note"),
        (ResidueKind.ABILITIES, ("meowth-galar",), "Pokémon Infobox > ability2note"),
    ]
    assert "Tough Claws" in residue[0].text
    assert "Tough Claws" in residue[0].wikitext


def test_ability_note_without_form_captions_gives_no_residue() -> None:
    infobox = infobox_with(ability2="Arena Trap", ability2note="<sup>{{color2|000|Gen IV}}</sup>")

    _, residue, _ = read_all(infobox)

    assert residue == ()


def test_a_small_caption_on_a_form_ability_moves_it_to_the_named_form() -> None:
    infobox = infobox_with(
        name="Darmanitan",
        form1="Standard Mode",
        form2="Zen Mode",
        form3="Galarian Form",
        image2="0555Darmanitan-Zen.png",
        image3="0555Darmanitan-Galar.png",
        form2type1="Fire",
        form3type1="Ice",
        ability1="Sheer Force",
        **{
            "ability2-1": "Gorilla Tactics{{!}}{{color2|000|Gorilla Tactics (Ability)}}<br/>"
            "<small>Galarian Form</small>"
        },
        abilityd="Zen Mode",
    )

    facts = read_facts(infobox, stats_page(("=====Darmanitan=====", STATS_TEMPLATE)))

    assert ability_names(facts["darmanitan-galar"])[0] == ("Gorilla Tactics", False)
    assert ability_names(facts["darmanitan-zen"])[0] == ("Sheer Force", False)
    assert ability_names(facts["darmanitan"])[0] == ("Sheer Force", False)


def test_a_caption_that_names_no_form_covers_the_unnamed_forms() -> None:
    infobox = {**FORM_INFOBOX, "abilityd": "Sand Force", "abilitydcaption": "Gen VI+"}

    facts = read_facts(infobox)

    assert ("Sand Force", True) in ability_names(facts["testmon"])
    assert ("Sand Force", True) in ability_names(facts["testmon-alola"])


def test_the_same_ability_twice_keeps_one_visible_slot_and_a_note() -> None:
    infobox = infobox_with(ability1="Levitate", ability2="Levitate", abilityd="Levitate")

    abilities = read_abilities(infobox, read_form_set(infobox))

    assert abilities.abilities["testmon"] == (AbilitySlot(name="Levitate", is_hidden=False),)
    assert [note.kind for note in abilities.notes] == [NoteKind.DUPLICATE_ABILITY] * 2


def test_an_ability_value_is_cut_at_the_first_template_or_tag() -> None:
    infobox = infobox_with(ability1="Sand Veil{{!}}x", ability2="Arena Trap<br/>y")

    names = ability_names(read_facts(infobox)["testmon"])

    assert names == [("Sand Veil", False), ("Arena Trap", False)]


def test_base_stats_from_form_subsections() -> None:
    wikitext = stats_page(
        ("=====Testmon=====", STATS_TEMPLATE),
        ("=====Alolan Testmon=====", "{{Stats|HP=10|Attack=1|Defense=2|SpAtk=3|SpDef=4|Speed=5}}"),
    )

    facts = read_facts(FORM_INFOBOX, wikitext)

    assert facts["testmon"].base_stats.attack == 55
    assert facts["testmon-alola"].base_stats.model_dump() == {
        "hp": 10,
        "attack": 1,
        "defense": 2,
        "sp_attack": 3,
        "sp_defense": 4,
        "speed": 5,
    }


def test_base_stats_from_a_list_subsection_cover_every_listed_form() -> None:
    infobox = infobox_with(
        name="Rotom",
        form2="Heat Rotom",
        form3="Wash Rotom",
        form4="Fan Rotom",
        image2="0479Rotom-Heat.png",
        image3="0479Rotom-Wash.png",
        image4="0479Rotom-Fan.png",
        form2type1="Fire",
        form3type1="Water",
        form4type1="Flying",
    )
    wikitext = stats_page(
        ("=====Rotom=====", STATS_TEMPLATE),
        (
            "=====Heat, Wash, and Fan Rotom=====",
            "{{Stats|HP=50|Attack=65|Defense=107|SpAtk=105|SpDef=107|Speed=86}}",
        ),
    )

    facts = read_facts(infobox, wikitext)

    assert [
        facts[i].base_stats.speed for i in ("rotom", "rotom-heat", "rotom-wash", "rotom-fan")
    ] == [
        95,
        86,
        86,
        86,
    ]


def test_a_form_with_no_subsection_copies_the_base_stats() -> None:
    infobox = infobox_with(form2="Alolan Form", image2="0050Testmon-Alola.png", form2type1="Ground")

    facts = read_facts(infobox)

    assert facts["testmon-alola"].base_stats == facts["testmon"].base_stats


def ranged_page(*sections: tuple[str, str]) -> str:
    return stats_page(*sections)


def test_generation_ranges_under_a_form_title_pick_the_range_with_generation_ix() -> None:
    old = "{{Stats|HP=1|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"
    wikitext = stats_page(
        ("=====Testmon=====", ""),
        ("======Generations I-V======", old),
        ("======Generation VI onward======", STATS_TEMPLATE),
    )

    assert read_facts(BASE_INFOBOX, wikitext)["testmon"].base_stats.hp == 10


def test_generation_ranges_directly_under_base_stats_belong_to_the_base_node() -> None:
    old = "{{Stats|HP=1|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"
    wikitext = stats_page(
        ("=====Generation V=====", old),
        ("=====Generation VI onwards=====", STATS_TEMPLATE),
    )

    assert read_facts(BASE_INFOBOX, wikitext)["testmon"].base_stats.hp == 10


def test_version_sections_pick_the_open_ended_one() -> None:
    first = "{{Stats|HP=1|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"
    wikitext = stats_page(
        ("=====Version 1.0.0=====", first),
        ("=====Version 1.0.1+=====", STATS_TEMPLATE),
    )

    assert read_facts(BASE_INFOBOX, wikitext)["testmon"].base_stats.hp == 10


def test_a_form_without_a_range_covering_generation_ix_takes_its_latest_range() -> None:
    infobox = infobox_with(
        form2="Mega Testmon",
        image2="0050Testmon-Mega.png",
        form2type1="Ground",
        abilitym="Tough Claws",
    )
    gen_six = "{{Stats|HP=6|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"
    gen_seven = "{{Stats|HP=7|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"
    wikitext = stats_page(
        ("=====Testmon=====", STATS_TEMPLATE),
        ("=====Mega Testmon=====", ""),
        ("======Generation VI======", gen_six),
        ("======Generation VII======", gen_seven),
    )

    assert read_facts(infobox, wikitext)["testmon-mega"].base_stats.hp == 7


def test_stats_templates_in_every_spelling_and_pe_stats_are_read_or_ignored() -> None:
    wikitext = stats_page(
        (
            "=====Testmon=====",
            "{{Stats/PE|HP=99}}{{Base Stats|HP=11|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}",
        ),
    )

    assert read_facts(BASE_INFOBOX, wikitext)["testmon"].base_stats.hp == 11


def test_base_stats_directly_in_the_base_stats_section() -> None:
    wikitext = f"==Game data==\n===Stats===\n====Base stats====\n{STATS_TEMPLATE}\n"

    assert read_facts(BASE_INFOBOX, wikitext)["testmon"].base_stats.speed == 95


def test_a_base_node_without_stats_fails_the_page() -> None:
    with pytest.raises(PageParseError, match="base stats"):
        read_facts(BASE_INFOBOX, stats_page(("=====Testmon=====", "no template here")))


def test_a_stat_that_is_not_a_number_fails_the_page() -> None:
    broken = "{{Stats|HP=??|Attack=1|Defense=1|SpAtk=1|SpDef=1|Speed=1}}"

    with pytest.raises(PageParseError) as error:
        read_facts(BASE_INFOBOX, stats_page(("=====Testmon=====", broken)))

    assert error.value.field == "HP"
    assert error.value.value == "??"
    assert "Base stats > Testmon" in error.value.section_path


def test_read_base_stats_returns_only_nodes_with_a_subsection() -> None:
    infobox = infobox_with(form2="Alolan Form", image2="0050Testmon-Alola.png", form2type1="Ground")
    inputs = build_inputs(infobox)

    stats = read_base_stats(inputs.sections, inputs.forms, PAGE_TITLE)

    assert list(stats) == ["testmon"]


def test_list_stats_titles_leaves_out_generation_ranges_and_deeper_sections() -> None:
    sections = split_sections(
        stats_page(
            ("=====Testmon=====", ""),
            ("======Generations I-V======", ""),
            ("=====Alolan Testmon=====", ""),
            ("=====Generation VI onward=====", ""),
            ("=====Version 1.0.1+=====", ""),
        )
    )

    assert list_stats_titles(sections) == ("Testmon", "Alolan Testmon")


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Generations I-V", (1, 5)),
        ("Generation I to V", (1, 5)),
        ("Generation VI onward", (6, None)),
        ("Generation VII onwards", (7, None)),
        ("Generation V", (5, 5)),
        ("Generations IV-VIII", (4, 8)),
        ("Generation IX", (9, 9)),
        ("Version 1.0.1+", (9, None)),
        ("Version 1.0.0", (9, 9)),
        ("Alolan Diglett", None),
        ("Pokémon Legends: Z-A", None),
    ],
)
def test_parse_generation_range(title: str, expected: tuple[int, int | None] | None) -> None:
    assert parse_generation_range(title) == expected


def genus_forms() -> FormSet:
    infobox = infobox_with(
        name="Darmanitan",
        form1="Standard Mode",
        form2="Zen Mode",
        form3="Galarian Form",
        image2="0555Darmanitan-Zen.png",
        image3="0555Darmanitan-Galar.png",
        form2type1="Fire",
        form3type1="Ice",
    )
    return read_form_set(infobox, stats_page(("=====Darmanitan=====", STATS_TEMPLATE)))


def test_genus_strips_the_pokemon_suffix() -> None:
    infobox = infobox_with(category="Mole {{ScPkmn}}")

    genus, residue = read_genus(infobox, read_form_set(infobox))

    assert genus == {"testmon": "Mole"}
    assert residue == ()


def test_genus_with_a_generation_caption_gives_the_base_value_to_every_form() -> None:
    forms = genus_forms()
    infobox = {"category": "{{tt|Scratch Cat|SCRATCHCAT before Generation III}}"}

    genus, residue = read_genus(infobox, forms)

    assert set(genus.values()) == {"Scratch Cat"}
    assert residue == ()


def test_genus_captions_naming_forms_send_the_text_to_those_forms() -> None:
    forms = genus_forms()
    infobox = {
        "category": (
            "{{tt|Blazing|Standard Mode, Zen Mode}} Pokémon<br/>{{tt|Zen Charm|Galarian Form}}"
        )
    }

    genus, residue = read_genus(infobox, forms)

    assert genus == {
        "darmanitan": "Blazing",
        "darmanitan-zen": "Blazing",
        "darmanitan-galar": "Zen Charm",
    }
    assert residue == ()


def test_a_form_no_genus_caption_covers_is_residue() -> None:
    forms = genus_forms()
    infobox = {"category": "{{tt|Blazing|Standard Mode}} Pokémon"}

    genus, residue = read_genus(infobox, forms)

    assert genus == {"darmanitan": "Blazing", "darmanitan-zen": None, "darmanitan-galar": None}
    assert [(r.kind, r.form_ids) for r in residue] == [
        (ResidueKind.GENUS, ("darmanitan-zen",)),
        (ResidueKind.GENUS, ("darmanitan-galar",)),
    ]
    assert residue[0].wikitext == "{{tt|Blazing|Standard Mode}} Pokémon"


def test_a_missing_category_gives_no_genus() -> None:
    forms = genus_forms()

    genus, residue = read_genus({}, forms)

    assert set(genus.values()) == {None}
    assert residue == ()


LANGUAGES_PAGE = """==In other languages==
{{Other languages|type=Ground|type2=Ground
|ja=ディグダ ''Digda''|jameaning=From ''dig''
|fr=Taupiqueur|frmeaning=From ''taupe'' (mole)
|es=Testmon|esmeaning=Same as English name
|de=Digda|demeaning=Same as Japanese name
|zh_cmn=地鼠 ''Dìshǔ''|zh_cmnmeaning=From 地
|ru=Диглетт ''Diglett''<br>Глетт{{tt|*|older}}<ref group=n>note</ref>
}}
==Trivia==
text
"""


def test_aliases_follow_the_template_then_jname_and_tmname() -> None:
    sections = split_sections(LANGUAGES_PAGE)
    infobox = {"jname": "ディグダ", "tmname": "Digda"}

    aliases = read_aliases(sections, infobox, "Testmon")

    assert aliases == (
        "ディグダ",
        "Digda",
        "Taupiqueur",
        "地鼠",
        "Dìshǔ",
        "Диглетт",
        "Diglett",
        "Глетт",
    )


def test_aliases_drop_the_english_name_and_repeats() -> None:
    sections = split_sections(
        "==In other languages==\n{{Other languages|es=Testmon|it=Testmon|fr=A|de=A}}"
    )

    assert read_aliases(sections, {"jname": "A", "tmname": "Testmon"}, "Testmon") == ("A",)


def test_aliases_survive_an_unclosed_italic_that_swallows_the_next_parameters() -> None:
    wikitext = (
        "==In other languages==\n"
        "{{Other languages|ko=다태우지네 ''Datae-ujine|komeaning=From ''da'' (all)\n"
        "|zh_cmn=焚焰蚣 ''Fényàngōng''|zh_cmnmeaning=x}}"
    )

    aliases = read_aliases(split_sections(wikitext), {}, "Testmon")

    assert aliases == ("다태우지네", "Datae-ujine", "焚焰蚣", "Fényàngōng")


def test_aliases_clean_templates_inside_romanizations_and_stray_quotes() -> None:
    wikitext = (
        "==In other languages==\n"
        "{{Other languages|ko=파이리 ''{{tt|Fairi<!--c-->|Romanized as Fairit}}''"
        "|ru=Аэродактиль ''Aerodaktil'''}}"
    )

    aliases = read_aliases(split_sections(wikitext), {}, "Testmon")

    assert aliases == ("파이리", "Fairi", "Аэродактиль", "Aerodaktil")


def test_aliases_without_the_section_come_from_the_infobox_only() -> None:
    aliases = read_aliases(
        split_sections("lead"), {"jname": "ケンタロス", "tmname": "Kentauros"}, "Tauros"
    )

    assert aliases == ("ケンタロス", "Kentauros")


def test_aliases_belong_to_the_base_node_only() -> None:
    wikitext = DEFAULT_STATS_PAGE + LANGUAGES_PAGE
    infobox = {**FORM_INFOBOX, "jname": "ディグダ"}

    facts = read_facts(infobox, wikitext)

    assert "ディグダ" in facts["testmon"].aliases
    assert facts["testmon-alola"].aliases == ()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("a|b|c", ["a", "b", "c"]),
        ("a|b={{tt|x|y}}|c=[[L|t]]", ["a", "b={{tt|x|y}}", "c=[[L|t]]"]),
        ("a|b=''x", ["a", "b=''x"]),
    ],
)
def test_split_top_level(body: str, expected: list[str]) -> None:
    assert split_top_level(body) == expected
