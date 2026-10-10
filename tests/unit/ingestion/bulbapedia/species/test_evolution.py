import mwparserfromhell
import pytest

from muon.ingestion.bulbapedia.species.evolution import (
    METHOD_RULES,
    FormChangeInputs,
    MethodFacts,
    apply_method_rules,
    read_evobox_edges,
    read_evolutions,
    read_form_changes,
    read_method_refs,
    read_method_text,
    select_method,
)
from muon.ingestion.bulbapedia.species.extract import (
    Requirement,
    RequirementRole,
    ResidueKind,
    SpeciesRef,
)
from muon.ingestion.bulbapedia.species.forms import FormInputs, resolve_forms
from muon.ingestion.bulbapedia.wikitext import find_templates, split_sections

RARE_CANDY = "{{Bag/s|Rare Candy|SV}}"
LEVEL_26 = RARE_CANDY + "<br>{{color2|000|Level|Level 26}}"


def read_edges(wikitext: str, name: str) -> list[tuple[str, str, str, str, str]]:
    code = mwparserfromhell.parse(wikitext)
    template = find_templates(code, [name])[0]
    return [
        (e.source.name, e.source.code, e.target.name, e.target.code, e.method)
        for e in read_evobox_edges(template)
    ]


def test_evobox_2_reads_the_arrow_and_the_form_sprite_codes() -> None:
    wikitext = """{{Evobox-2
|type1=Ground
|no1=0050Diglett-Alola
|name1=Diglett
|form1=Alolan Form
|evo1=LEVEL
|no2=0051Dugtrio-Alola
|name2=Dugtrio
|form2=Alolan Form}}""".replace("LEVEL", LEVEL_26)

    assert read_edges(wikitext, "evobox-2") == [("Diglett", "Alola", "Dugtrio", "Alola", LEVEL_26)]


def test_evobox_3_reads_both_arrows() -> None:
    wikitext = """{{Evobox-3
|no1=0004|name1=Charmander|evo1=A
|no2=0005|name2=Charmeleon|evo2=B
|no3=0006|name3=Charizard}}"""

    assert read_edges(wikitext, "evobox-3") == [
        ("Charmander", "", "Charmeleon", "", "A"),
        ("Charmeleon", "", "Charizard", "", "B"),
    ]


def test_1branch2_links_the_single_stage_to_each_variant() -> None:
    wikitext = """{{Evobox/1branch2
|no1=0412|name1=Burmy|form1=Plant Cloak|evo1a=F
|no2a=0413|name2a=Wormadam|evo1b=M
|no2b=0414|name2b=Mothim}}"""

    assert read_edges(wikitext, "evobox/1branch2") == [
        ("Burmy", "", "Wormadam", "", "F"),
        ("Burmy", "", "Mothim", "", "M"),
    ]


def test_1branch3_reads_three_variants() -> None:
    wikitext = """{{Evobox/1branch3
|no1=0236|name1=Tyrogue
|evo1a=A|no2a=0106|name2a=Hitmonlee
|evo1b=B|no2b=0107|name2b=Hitmonchan
|evo1c=C|no2c=0237|name2c=Hitmontop}}"""

    assert [(edge[2], edge[4]) for edge in read_edges(wikitext, "evobox/1branch3")] == [
        ("Hitmonlee", "A"),
        ("Hitmonchan", "B"),
        ("Hitmontop", "C"),
    ]


def test_2branch2_links_the_single_stage_and_then_each_variant() -> None:
    wikitext = """{{Evobox/2branch2
|no1=0172|name1=Pichu|evo1=ONE
|no2=0025|name2=Pikachu|evo2a=A|evo2b=B
|no3a=0026|name3a=Raichu
|no3b=0026Raichu-Alola|name3b=Raichu}}"""

    assert read_edges(wikitext, "evobox/2branch2") == [
        ("Pichu", "", "Pikachu", "", "ONE"),
        ("Pikachu", "", "Raichu", "", "A"),
        ("Pikachu", "", "Raichu", "Alola", "B"),
    ]


def test_2branch1_links_variant_to_the_same_variant() -> None:
    wikitext = """{{Evobox/2branch1
|no1=0265|name1=Wurmple|evo1a=A|evo1b=B
|no2a=0266|name2a=Silcoon|no2b=0268|name2b=Cascoon
|evo2a=C|evo2b=D
|no3a=0267|name3a=Beautifly|no3b=0269|name3b=Dustox}}"""

    assert [
        (e[0], e[2], e[4])
        for e in read_edges(wikitext, "evobox/2branch2".replace("2branch2", "2branch1"))
    ] == [
        ("Wurmple", "Silcoon", "A"),
        ("Wurmple", "Cascoon", "B"),
        ("Silcoon", "Beautifly", "C"),
        ("Cascoon", "Dustox", "D"),
    ]


def test_evobox_1_has_no_arrow() -> None:
    wikitext = "{{Evobox-1|no1=0550Basculin-Red|name1=Basculin|form1=Red-Striped Form}}"

    assert read_edges(wikitext, "evobox-1") == []


def test_an_arrow_to_a_missing_stage_is_left_out() -> None:
    wikitext = "{{Evobox-2|no1=0001|name1=A|evo1=X|no2=0002}}"

    assert read_edges(wikitext, "evobox-2") == []


def test_select_method_drops_a_other_game_alternative() -> None:
    trade = "[[File:Link Trade icon SV.png|40px]]<br>{{color2|000|Trade}}"
    linking = (
        "{{bag/s|Linking Cord|LA}}<br>{{color2|000|Linking Cord}}"
        "<br><small>{{gameIcon|LA}} only</small>"
    )
    method = f"{trade}<br><big><big>→</big></big><br>{linking}"

    selected = select_method(method)

    assert selected is not None
    assert read_method_text(selected) == "Trade"


def test_select_method_keeps_an_alternative_marked_for_scarlet_violet() -> None:
    method = "B<small>{{gameIcon|Sc}}{{gameIcon|V}} only</small>"

    assert select_method(method) == method


def test_select_method_gives_nothing_when_two_alternatives_remain() -> None:
    method = "{{color2|000|Trade}}<br><big><big>→</big></big><br>{{color2|000|Linking Cord}}"

    assert select_method(method) is None


def test_select_method_splits_on_or() -> None:
    method = "Level 10<br>or<br>Level 20<br>or<br>Level 30"

    assert select_method(method) is None
    assert select_method("Level 10<br>or<br>Level 20 {{gameIcon|LA}} only") is not None


def test_select_method_drops_alternatives_limited_to_hisui_or_alola() -> None:
    assert select_method("Level 36 in {{color2|000|Hisui}}") is None
    assert select_method("{{Thunder}} in {{color2|000|Alola}}") is None


def test_outside_of_a_region_is_removed_from_the_text() -> None:
    alternative = (
        "{{Bag/s|Rare Candy|SV}}<br>{{color2|000|Level|Level 37}}"
        "<br>outside of {{color2|000|Hisui}}"
    )

    assert select_method(alternative) == alternative
    assert read_method_text(alternative) == "Level 37"


def test_read_method_text_keeps_small_text_and_drops_decoration() -> None:
    method = (
        "{{bag/s|Rare Candy|SV}} + {{bag/s|Soothe Bell|SV}} + [[File:Night Icon SV.png|40px]]"
        "<br>{{color2|000|Level|Level up}} with high {{color2|000|friendship}}"
        "<br><small>at {{color2|000|time|night}}</small>"
    )

    assert read_method_text(method) == "Level up with high friendship at night"


def test_read_method_text_drops_a_parenthetical_that_names_a_game() -> None:
    method = (
        "{{Bag/s|Rare Candy|SV}}<br>{{color2|000|Level|Level 14}}"
        "<br><small>(17 in {{gameIcon|LA}})</small>"
    )

    assert read_method_text(method) == "Level 14"


def test_read_method_refs_classifies_typed_templates_and_links() -> None:
    method = (
        "{{bag/s|Razor Claw|SV}} {{color2|000|Razor Claw}} {{color2|000|Trade}} "
        "{{color2|000|Double Hit (move)|Double Hit}} {{mcolor|Mimic|000}} "
        "{{color2|000|Remoraid (Pokémon)|Remoraid}} {{pcolor|Karrablast|000}} {{p|Shelmet}} "
        "{{color2|000|Dark (type)|Dark-type}} {{type|Fairy}}"
    )

    refs = read_method_refs(method)

    assert refs.items == ("Razor Claw",)
    assert sorted(refs.moves) == ["Double Hit", "Mimic"]
    assert sorted(refs.species) == ["Karrablast", "Remoraid", "Shelmet"]
    assert sorted(refs.types) == ["Dark", "Fairy"]


def level_up(extra: str) -> str:
    return RARE_CANDY + "<br>{{color2|000|Level|Level up}}" + extra


def read_facts(method: str) -> MethodFacts:
    facts = apply_method_rules(method)
    assert facts is not None
    return facts


def test_rule_level() -> None:
    assert read_facts(LEVEL_26) == MethodFacts("level", min_level=26)


@pytest.mark.parametrize(
    ("small", "time_of_day"),
    [
        ("at {{color2|000|time|night}}", "night"),
        ("during the {{color2|000|time|day}}", "day"),
        ("during the evening", "dusk"),
    ],
)
def test_rule_level_at_a_time_of_day(small: str, time_of_day: str) -> None:
    method = f"{RARE_CANDY}<br>Level 25<br><small>{small}</small>"

    assert read_facts(method) == MethodFacts("level", min_level=25, time_of_day=time_of_day)


def test_rule_level_for_a_gender() -> None:
    method = (
        f"{RARE_CANDY} + [[File:Male icon HOME.png]]<br>Level 20"
        "<br><small>({{color2|000|Gender|male}})</small>"
    )

    assert read_facts(method) == MethodFacts("level", min_level=20, gender="male")
    assert read_facts("Level 30 Female") == MethodFacts("level", min_level=30, gender="female")


def test_rule_item_used() -> None:
    method = "{{Bag/s|Fire Stone|SV}}<br>{{color2|000|Fire Stone}}"

    assert read_facts(method) == MethodFacts(
        "item",
        requirements=(Requirement(role=RequirementRole.USE, name="Fire Stone", species=None),),
    )


def test_rule_item_used_with_a_gender() -> None:
    method = (
        "{{Bag/s|Dawn Stone|SV}}<br>{{color2|000|Dawn Stone}}"
        "<br><small>({{color2|000|Gender|male}})</small>"
    )

    facts = read_facts(method)

    assert (facts.trigger, facts.gender) == ("item", "male")
    assert facts.requirements[0].name == "Dawn Stone"


def test_rule_item_needs_a_typed_item() -> None:
    assert apply_method_rules("Spin holding a Sweet") is None
    assert apply_method_rules("{{color2|000|Trade}}") == MethodFacts("trade")


def test_rule_trade() -> None:
    assert read_facts(
        "[[File:Link Trade icon SV.png|40px]]<br>{{color2|000|Trade}}"
    ) == MethodFacts("trade")


def test_rule_trade_holding_an_item() -> None:
    method = (
        "[[File:x.png]] + {{bag/s|Metal Coat|SV}}<br>{{color2|000|Trade}}"
        "<br>holding {{color2|000|Metal Coat}}"
    )

    assert read_facts(method) == MethodFacts(
        "trade",
        requirements=(Requirement(role=RequirementRole.HOLD, name="Metal Coat", species=None),),
    )
    while_holding = method.replace("holding", "while holding")
    assert read_facts(while_holding).requirements[0].role is RequirementRole.HOLD


def test_rule_trade_for_a_species() -> None:
    method = (
        "[[File:x.png]]<br>{{color2|000|Trade}}<br><small>for {{pcolor|Karrablast|000}}</small>"
    )

    assert read_facts(method) == MethodFacts(
        "trade",
        requirements=(
            Requirement(
                role=RequirementRole.TRADED_FOR,
                name="Karrablast",
                species=SpeciesRef(name="Karrablast", code=""),
            ),
        ),
    )


def test_rule_friendship_with_and_without_a_time_of_day() -> None:
    friendship = level_up(" with<br>high {{color2|000|friendship}}")

    assert read_facts(friendship) == MethodFacts("friendship")
    at_night = friendship + "<br><small>at {{color2|000|time|night}}</small>"
    assert read_facts(at_night) == MethodFacts("friendship", time_of_day="night")


def test_rule_level_up_holding_an_item() -> None:
    method = (
        "{{bag/s|Razor Fang|SV}}<br>{{color2|000|Level|Level up}} holding {{color2|000|Razor Fang}}"
        "<br><small>at {{color2|000|time|night}}</small>"
    )

    assert read_facts(method) == MethodFacts(
        "level",
        time_of_day="night",
        requirements=(Requirement(role=RequirementRole.HOLD, name="Razor Fang", species=None),),
    )


def test_rule_level_up_knowing_a_move() -> None:
    method = level_up("<br>knowing {{color2|000|Double Hit (move)|Double Hit}}")

    assert read_facts(method) == MethodFacts(
        "level",
        requirements=(
            Requirement(role=RequirementRole.KNOWN_MOVE, name="Double Hit", species=None),
        ),
    )


def test_rule_friendship_knowing_a_move_of_a_type() -> None:
    method = level_up(
        " with high {{color2|000|friendship}} knowing a {{color2|000|Fairy (type)|Fairy-type}} move"
    )

    assert read_facts(method) == MethodFacts(
        "friendship",
        requirements=(
            Requirement(role=RequirementRole.KNOWN_MOVE_TYPE, name="Fairy", species=None),
        ),
    )


def test_rule_level_with_a_species_in_the_party() -> None:
    method = level_up("<br>with {{color2|000|Remoraid (Pokémon)|Remoraid}} in {{color2|000|party}}")

    facts = read_facts(method)

    assert facts.trigger == "level"
    assert facts.min_level is None
    assert facts.requirements == (
        Requirement(
            role=RequirementRole.PARTY_SPECIES,
            name="Remoraid",
            species=SpeciesRef(name="Remoraid", code=""),
        ),
    )
    with_level = method.replace("Level up", "Level 20")
    assert read_facts(with_level).min_level == 20


def test_rule_level_with_a_type_in_the_party() -> None:
    method = (
        f"{RARE_CANDY}<br>{{{{color2|000|Level|Level 32}}}}<br><small>with a "
        "{{color2|000|Dark (type)|Dark-type}} Pokémon in the {{color2|000|party}}</small>"
    )

    assert read_facts(method) == MethodFacts(
        "level",
        min_level=32,
        requirements=(Requirement(role=RequirementRole.PARTY_TYPE, name="Dark", species=None),),
    )


@pytest.mark.parametrize(
    "method",
    [
        "Level 20<br><small>({{color2|000|Attack}} > {{color2|000|Defense}})</small>",
        "Level up whilst outside of its Poké Ball after walking 1,000 steps",
        "Use a {{bag/s|Peat Block|SV}} during a full moon",
        "Land three critical hits in one battle",
        "Level up knowing {{color2|000|Mimic (move)|Mimic}} in Galar",
    ],
)
def test_a_method_no_rule_reads_is_left_alone(method: str) -> None:
    assert apply_method_rules(method) is None


def test_every_rule_pattern_is_anchored() -> None:
    assert all(rule.pattern.pattern.startswith("^") for rule in METHOD_RULES)
    assert all(rule.pattern.pattern.endswith("$") for rule in METHOD_RULES)


def evolution_page(box: str) -> list:
    return list(split_sections(f"==Game data==\n===Evolution data===\n====Testmon====\n{box}\n"))


def test_read_evolutions_builds_structured_facts() -> None:
    sections = evolution_page(
        f"{{{{Evobox-2|no1=0050Diglett-Alola|name1=Diglett|evo1={LEVEL_26}|no2=0051Dugtrio-Alola|name2=Dugtrio}}}}"
    )

    evolutions, residue = read_evolutions(sections)

    assert residue == ()
    (evolution,) = evolutions
    assert evolution.source == SpeciesRef(name="Diglett", code="Alola")
    assert evolution.target == SpeciesRef(name="Dugtrio", code="Alola")
    assert (evolution.trigger, evolution.min_level, evolution.condition) == ("level", 26, None)
    assert evolution.section_path == "Game data > Evolution data > Testmon"


def test_read_evolutions_keeps_an_unread_method_as_condition_and_residue() -> None:
    method = "Level 20<br><small>(Attack > Defense)</small>"
    sections = evolution_page(
        f"{{{{Evobox-2|no1=0236|name1=Tyrogue|evo1={method}|no2=0106|name2=Hitmonlee}}}}"
    )

    evolutions, residue = read_evolutions(sections)

    (evolution,) = evolutions
    assert evolution.trigger is None
    assert evolution.condition == "Level 20 (Attack > Defense)"
    assert (
        evolution.min_level,
        evolution.time_of_day,
        evolution.gender,
        evolution.requirements,
    ) == (
        None,
        None,
        None,
        (),
    )
    (item,) = residue
    assert item.kind is ResidueKind.EVOLUTION
    assert item.wikitext == method
    assert item.text == evolution.condition


def test_read_evolutions_condition_joins_the_text_of_every_alternative() -> None:
    method = "{{color2|000|Trade}}<br><big><big>→</big></big><br>{{color2|000|Linking Cord}}"
    sections = evolution_page(f"{{{{Evobox-2|no1=0001|name1=A|evo1={method}|no2=0002|name2=B}}}}")

    evolutions, residue = read_evolutions(sections)

    assert evolutions[0].condition == "Trade or Linking Cord"
    assert len(residue) == 1


def test_read_evolutions_gives_a_section_with_an_unread_evobox_one_residue_item() -> None:
    sections = split_sections(
        "==Game data==\n===Evolution data===\nEevee evolves.\n"
        "{{Evobox/Setup|133Eevee|Eevee}}\n|-\n{{Evobox/Setup|133Eevee|Vaporeon}}\n"
    )

    evolutions, residue = read_evolutions(sections)

    assert evolutions == ()
    (item,) = residue
    assert item.kind is ResidueKind.EVOLUTION
    assert item.section_path == "Game data > Evolution data"
    assert "{{Evobox/Setup|133Eevee|Eevee}}" in item.wikitext
    assert "Eevee evolves." in item.text


def test_read_evolutions_reads_boxes_next_to_an_unread_one() -> None:
    sections = evolution_page(
        f"{{{{Evobox-2|no1=0001|name1=A|evo1={LEVEL_26}|no2=0002|name2=B}}}}{{{{Evobox/Meowth-Alola|x}}}}"
    )

    evolutions, residue = read_evolutions(sections)

    assert len(evolutions) == 1
    assert len(residue) == 1


def test_read_evolutions_collapses_repeated_boxes_and_ignores_other_sections() -> None:
    box = f"{{{{Evobox-2|no1=0001|name1=A|evo1={LEVEL_26}|no2=0002|name2=B}}}}"
    sections = split_sections(
        f"==Game data==\n===Evolution data===\n====One====\n{box}\n====Two====\n{box}\n"
        f"===Stats===\n{box}\n"
    )

    evolutions, _ = read_evolutions(sections)

    assert len(evolutions) == 2
    assert {evolution.section_path for evolution in evolutions} == {
        "Game data > Evolution data > One",
        "Game data > Evolution data > Two",
    }


def build_change_inputs(infobox: dict[str, str], form_data: str = "") -> FormChangeInputs:
    sections = split_sections(f"==Game data==\n===Form data===\n{form_data}\n")
    forms = resolve_forms(FormInputs("Testmon (Pokémon)", infobox, (), ()))
    return FormChangeInputs(sections, infobox, forms)


MEGA_INFOBOX = {
    "name": "Charizard",
    "form2": "Mega Charizard X",
    "form3": "Mega&nbsp;Charizard&nbsp;Y",
    "form4": "Gigantamax",
    "image2": "0006Charizard-Mega X.png",
    "image3": "0006Charizard-Mega Y.png",
    "form2type1": "Fire",
    "form3type1": "Fire",
    "form4type1": "Fire",
    "mega": "Charizardite X",
    "mega2": "Charizardite Y",
}


def test_mega_forms_take_the_stones_in_infobox_order() -> None:
    changes, residue = read_form_changes(build_change_inputs(MEGA_INFOBOX))

    assert residue == ()
    assert [(c.from_id, c.to_id, c.form_kind, c.held_item) for c in changes] == [
        ("charizard", "charizard-mega-x", "mega", "Charizardite X"),
        ("charizard", "charizard-mega-y", "mega", "Charizardite Y"),
    ]
    assert all(c.is_battle_only and c.trigger is None for c in changes)


def test_one_stone_is_shared_by_every_mega_form() -> None:
    infobox = {
        "name": "Tatsugiri",
        "form2": "Mega Tatsugiri (Curly Form)",
        "form3": "Mega Tatsugiri (Droopy Form)",
        "form4": "Mega Tatsugiri (Stretchy Form)",
        "image2": "0978Tatsugiri-Mega Curly.png",
        "image3": "0978Tatsugiri-Mega Droopy.png",
        "image4": "0978Tatsugiri-Mega Stretchy.png",
        "form2type1": "Dragon",
        "form3type1": "Dragon",
        "form4type1": "Dragon",
        "mega": "Tatsugirinite",
    }

    changes, residue = read_form_changes(build_change_inputs(infobox))

    assert residue == ()
    assert {c.held_item for c in changes} == {"Tatsugirinite"}
    assert len(changes) == 3


def test_a_mega_form_beyond_the_listed_stones_is_residue() -> None:
    infobox = {**MEGA_INFOBOX, "form4": "Mega Charizard Z", "image4": "0006Charizard-Mega Z.png"}

    changes, residue = read_form_changes(build_change_inputs(infobox))

    assert len(changes) == 2
    assert [r.form_ids for r in residue] == [("charizard-mega-z",)]


def test_mega_none_makes_the_mega_form_residue() -> None:
    infobox = {
        "name": "Rayquaza",
        "form2": "Mega Rayquaza",
        "image2": "0384Rayquaza-Mega.png",
        "abilitym": "Delta Stream",
        "mega": "None",
    }
    form_data = "Rayquaza can Mega Evolve if it knows Dragon Ascent."

    changes, residue = read_form_changes(build_change_inputs(infobox, form_data))

    assert changes == ()
    (item,) = residue
    assert item.kind is ResidueKind.FORM_CHANGE
    assert item.form_ids == ("rayquaza-mega",)
    assert item.section_path == "Game data > Form data"
    assert item.wikitext == form_data


GROUDON_INFOBOX = {
    "name": "Groudon",
    "form2": "Primal Reversion",
    "image2": "0383Groudon-Primal.png",
    "form2type1": "Ground",
}
GROUDON_FORMES = """{{Evobox/Formes
|forme1=Groudon
|loc1=Default
|item1=Red Orb
|forme2=Primal Groudon
|loc2={{color2|000|Held item|Held}} Red Orb
}}"""


def test_a_primal_form_takes_the_orb_its_row_names() -> None:
    changes, residue = read_form_changes(build_change_inputs(GROUDON_INFOBOX, GROUDON_FORMES))

    assert residue == ()
    (change,) = changes
    assert (change.to_id, change.form_kind, change.trigger) == (
        "groudon-primal",
        "primal",
        "switch_in",
    )
    assert (change.is_battle_only, change.held_item) == (True, "Red Orb")


def test_a_primal_row_item_wins_over_its_location() -> None:
    form_data = GROUDON_FORMES.replace("|loc2=", "|item2=Blue Orb\n|loc2=")

    changes, _ = read_form_changes(build_change_inputs(GROUDON_INFOBOX, form_data))

    assert changes[0].held_item == "Blue Orb"


def test_a_primal_form_without_a_row_is_residue() -> None:
    changes, residue = read_form_changes(build_change_inputs(GROUDON_INFOBOX, "No template."))

    assert changes == ()
    assert [r.form_ids for r in residue] == [("groudon-primal",)]


def test_other_form_nodes_are_residue_with_the_form_data_text() -> None:
    infobox = {
        "name": "Rotom",
        "form2": "Heat Rotom",
        "form3": "Wash Rotom",
        "image2": "0479Rotom-Heat.png",
        "image3": "0479Rotom-Wash.png",
        "form2type1": "Fire",
        "form3type1": "Water",
    }
    sections = split_sections(
        "==Game data==\n===Form data===\nRotom has forms.\n"
        "====Appliances====\n{{Evobox/Formes|forme1=Heat Rotom|loc1=Heat}}\n"
    )
    forms = resolve_forms(FormInputs("Rotom (Pokémon)", infobox, (), ()))

    changes, residue = read_form_changes(FormChangeInputs(sections, infobox, forms))

    assert changes == ()
    assert [r.form_ids for r in residue] == [("rotom-heat",), ("rotom-wash",)]
    assert "Rotom has forms." in residue[0].wikitext
    assert "{{Evobox/Formes|forme1=Heat Rotom|loc1=Heat}}" in residue[0].wikitext
    assert residue[0].text.startswith("Rotom has forms.")


def test_regional_and_default_forms_give_nothing() -> None:
    infobox = {
        "name": "Diglett",
        "form2": "Alolan Form",
        "image2": "0050Diglett-Alola.png",
        "form2type1": "Ground",
    }

    changes, residue = read_form_changes(build_change_inputs(infobox))

    assert (changes, residue) == ((), ())


def test_a_page_without_form_data_still_gives_residue_with_no_text() -> None:
    infobox = {
        "name": "Testmon",
        "form2": "Hot Form",
        "image2": "0001Testmon-Hot.png",
        "form2type1": "Fire",
    }
    forms = resolve_forms(FormInputs("Testmon (Pokémon)", infobox, (), ()))

    _, residue = read_form_changes(FormChangeInputs(split_sections("lead"), infobox, forms))

    assert [(r.form_ids, r.wikitext, r.text) for r in residue] == [(("testmon-hot",), "", "")]
