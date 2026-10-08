from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    build_version,
    build_version_group,
)

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
SPECIES = (
    'ex:charmander a pkmn:PokemonSpecies ; pkmn:id "charmander" ; pkmn:pokedexNumber 4 ; '
    "pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.125 ."
)
GROUP = build_version_group("rb", "red-blue")
VERSION = build_version("red", "red", "rb")
AREA = 'ex:cave-1f a pkmn:LocationArea ; pkmn:id "cave-1f" .'
REFERENCES = FIRE + SPECIES + GENERATION_NODE + GROUP + VERSION + AREA


def encounter(properties: str = "pkmn:minLevel 5 ; pkmn:maxLevel 10 ; pkmn:chance 20") -> str:
    return (
        'ex:encounter a pkmn:Encounter ; pkmn:id "charmander:cave-1f:red" ; '
        f"pkmn:atLocation ex:cave-1f ; pkmn:inVersion ex:red ; {properties} ."
    )


LINK = "ex:charmander pkmn:hasEncounter ex:encounter ."


def test_complete_encounter_conforms(validate_fixture):
    assert validate_fixture(REFERENCES + encounter() + LINK).conforms


def test_encounter_without_levels_conforms(validate_fixture):
    assert validate_fixture(REFERENCES + encounter("pkmn:chance 20") + LINK).conforms


def test_min_level_above_max_level_fails(validate_fixture):
    data = REFERENCES + encounter("pkmn:minLevel 30 ; pkmn:maxLevel 20") + LINK
    assert validate_fixture(data).rule_ids == {"C1"}


def test_chance_zero_fails(validate_fixture):
    assert validate_fixture(REFERENCES + encounter("pkmn:chance 0") + LINK).rule_ids == {"C1"}


def test_chance_above_100_fails(validate_fixture):
    assert validate_fixture(REFERENCES + encounter("pkmn:chance 101") + LINK).rule_ids == {"C1"}


def test_encounter_without_incoming_link_fails(validate_fixture):
    assert validate_fixture(REFERENCES + encounter()).rule_ids == {"C1"}


def test_encounter_with_two_incoming_links_fails(validate_fixture):
    other = (
        'ex:squirtle a pkmn:PokemonSpecies ; pkmn:id "squirtle" ; pkmn:pokedexNumber 7 ; '
        "pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.125 . "
        "ex:squirtle pkmn:hasEncounter ex:encounter ."
    )
    assert validate_fixture(REFERENCES + encounter() + LINK + other).rule_ids == {"C1"}


def test_encounter_without_version_fails(validate_fixture):
    data = (
        REFERENCES
        + 'ex:encounter a pkmn:Encounter ; pkmn:id "x" ; pkmn:atLocation ex:cave-1f .'
        + LINK
    )
    assert validate_fixture(data).rule_ids == {"C1"}


def test_encounter_at_a_plain_location_conforms(validate_fixture):
    data = REFERENCES + 'ex:route a mo:Route ; pkmn:id "route-1" .'
    data += encounter().replace("ex:cave-1f", "ex:route") + LINK
    assert validate_fixture(data).conforms
