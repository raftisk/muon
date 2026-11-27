import pytest

from tests.unit.shapes.games_fixtures import GENERATION_NODE, IN_FIRST_GENERATION
from tests.unit.shapes.move_fixtures import build_move

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'


def species(name: str, dex: int) -> str:
    return (
        f'ex:{name} a pkmn:PokemonSpecies ; pkmn:id "{name}" ; pkmn:pokedexNumber {dex} ; '
        "pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.5 ."
    )


BASE = FIRE + GENERATION_NODE + species("charmander", 4) + species("charmeleon", 5)

TARGETS = {
    "Item": GENERATION_NODE
    + 'ex:thing a pkmn:Item ; pkmn:id "thing" ; pkmn:introducedIn ex:gen1 .',
    "Move": (
        build_move("move", damage_class_local_name="physical")
        + 'ex:physical a pkmn:MoveDamageClass ; pkmn:id "physical" .'
        + GENERATION_NODE
    ),
    "Type": FIRE,
    "Location": 'ex:thing a mo:Location ; pkmn:id "thing" .',
    "PokemonSpecies": species("thing", 6),
}
ROLE_TARGETS = [
    ("use", "Item", "ex:thing"),
    ("hold", "Item", "ex:thing"),
    ("knownMove", "Move", "ex:move"),
    ("knownMoveType", "Type", "ex:fire"),
    ("partyType", "Type", "ex:fire"),
    ("atLocation", "Location", "ex:thing"),
    ("fieldMove", "Move", "ex:move"),
    ("fieldItem", "Item", "ex:thing"),
    ("partySpecies", "PokemonSpecies", "ex:thing"),
    ("tradedFor", "PokemonSpecies", "ex:thing"),
    ("fusesWith", "PokemonSpecies", "ex:thing"),
]


def form_change(kind: str, battle_only: str) -> str:
    return (
        'ex:change a pkmn:FormChange ; pkmn:id "charmander:charmeleon" ; '
        "pkmn:fromSpecies ex:charmander ; pkmn:toSpecies ex:charmeleon ; "
        f'pkmn:formKind "{kind}" ; pkmn:isBattleOnly {battle_only} ; {IN_FIRST_GENERATION} .'
    )


def requires_edge(role: str, target: str) -> str:
    return (
        "[] rdf:subject ex:change ; rdf:predicate pkmn:requires ; "
        f'rdf:object {target} ; pkmn:role "{role}" .'
    )


def test_evolution_with_both_species_conforms(validate_fixture):
    data = BASE + (
        'ex:evolution a pkmn:Evolution ; pkmn:id "charmander:charmeleon" ; '
        f"pkmn:fromSpecies ex:charmander ; pkmn:toSpecies ex:charmeleon ; {IN_FIRST_GENERATION} ."
    )
    assert validate_fixture(data).conforms


def test_evolution_without_target_species_fails(validate_fixture):
    data = BASE + (
        'ex:evolution a pkmn:Evolution ; pkmn:id "charmander:charmeleon" ; '
        f"pkmn:fromSpecies ex:charmander ; {IN_FIRST_GENERATION} ."
    )
    assert validate_fixture(data).rule_ids == {"V1"}


def test_evolution_with_two_source_species_fails(validate_fixture):
    data = BASE + (
        'ex:evolution a pkmn:Evolution ; pkmn:id "charmander:charmeleon" ; '
        "pkmn:fromSpecies ex:charmander , ex:charmeleon ; pkmn:toSpecies ex:charmeleon ; "
        f"{IN_FIRST_GENERATION} ."
    )
    assert validate_fixture(data).rule_ids == {"V1"}


def test_mega_form_change_battle_only_conforms(validate_fixture):
    assert validate_fixture(BASE + form_change("mega", "true")).conforms


def test_alternate_form_change_not_battle_only_conforms(validate_fixture):
    assert validate_fixture(BASE + form_change("alternate", "false")).conforms


def test_alternate_form_change_battle_only_fails(validate_fixture):
    assert validate_fixture(BASE + form_change("alternate", "true")).rule_ids == {"V2"}


def test_mega_form_change_not_battle_only_fails(validate_fixture):
    assert validate_fixture(BASE + form_change("mega", "false")).rule_ids == {"V2"}


def test_unknown_form_kind_fails(validate_fixture):
    assert "V2" in validate_fixture(BASE + form_change("giga", "true")).rule_ids


def test_form_change_without_species_fails(validate_fixture):
    data = GENERATION_NODE + (
        'ex:change a pkmn:FormChange ; pkmn:id "x" ; '
        f'pkmn:formKind "mega" ; pkmn:isBattleOnly true ; {IN_FIRST_GENERATION} .'
    )
    assert validate_fixture(data).rule_ids == {"V1"}


@pytest.mark.parametrize(("role", "target_class", "target"), ROLE_TARGETS)
def test_role_with_matching_target_conforms(validate_fixture, role, target_class, target):
    data = BASE + TARGETS[target_class] + requires_edge(role, target)
    assert validate_fixture(data).conforms


def test_hold_pointing_at_a_move_fails(validate_fixture):
    data = BASE + TARGETS["Move"] + requires_edge("hold", "ex:move")
    assert validate_fixture(data).rule_ids == {"R1"}


def test_field_move_pointing_at_an_item_fails(validate_fixture):
    data = BASE + TARGETS["Item"] + requires_edge("fieldMove", "ex:thing")
    assert validate_fixture(data).rule_ids == {"R1"}


def test_unknown_role_fails(validate_fixture):
    data = BASE + TARGETS["Item"] + requires_edge("steal", "ex:thing")
    assert validate_fixture(data).rule_ids == {"R1"}


def test_missing_role_fails(validate_fixture):
    data = BASE + TARGETS["Item"] + "[] rdf:predicate pkmn:requires ; rdf:object ex:thing ."
    assert validate_fixture(data).rule_ids == {"R1"}
