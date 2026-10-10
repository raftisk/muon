from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    build_version,
    build_version_group,
)

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
GROUP = build_version_group("rb", "red-blue")
VERSION = build_version("red", "red", "rb")
SPECIES = (
    'ex:charmander a pkmn:PokemonSpecies ; pkmn:id "charmander" ; pkmn:pokedexNumber 4 ; '
    "pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.125 ."
)
REFERENCES = FIRE + GENERATION_NODE + GROUP + VERSION + SPECIES


def chunk(properties: str) -> str:
    return f'ex:chunk a mo:TextChunk ; pkmn:id "chunk" ; mo:text "Text." ; {properties} .'


def test_pokedex_chunk_with_version_conforms(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "pokedex" ; mo:describes ex:charmander ; mo:fromWork ex:red'
    )
    assert validate_fixture(data).conforms


def test_pokedex_chunk_with_version_group_fails(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "pokedex" ; mo:describes ex:charmander ; mo:fromWork ex:rb'
    )
    assert validate_fixture(data).rule_ids == {"X2"}


def test_pokedex_chunk_describing_a_type_fails(validate_fixture):
    data = REFERENCES + chunk('mo:aspect "pokedex" ; mo:describes ex:fire ; mo:fromWork ex:red')
    assert validate_fixture(data).rule_ids == {"X2"}


def test_pokedex_chunk_without_work_fails(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "pokedex" ; mo:describes ex:charmander ; mo:source "pokeapi"'
    )
    assert validate_fixture(data).rule_ids == {"X2"}


def test_flavor_chunk_with_version_group_conforms(validate_fixture):
    data = REFERENCES + chunk('mo:aspect "flavor" ; mo:describes ex:fire ; mo:fromWork ex:rb')
    assert validate_fixture(data).conforms


def test_flavor_chunk_with_version_fails(validate_fixture):
    data = REFERENCES + chunk('mo:aspect "flavor" ; mo:describes ex:fire ; mo:fromWork ex:red')
    assert validate_fixture(data).rule_ids == {"X2"}


def test_biology_chunk_with_source_conforms(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "biology" ; mo:describes ex:charmander ; mo:source "bulbapedia"'
    )
    assert validate_fixture(data).conforms


def test_biology_chunk_with_work_fails(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "biology" ; mo:describes ex:charmander ; mo:fromWork ex:rb'
    )
    assert validate_fixture(data).rule_ids == {"X2"}


def test_chunk_with_neither_source_nor_work_fails(validate_fixture):
    data = REFERENCES + chunk('mo:aspect "biology" ; mo:describes ex:charmander')
    assert validate_fixture(data).rule_ids == {"X1"}


def test_chunk_with_source_and_work_fails(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "flavor" ; mo:describes ex:fire ; mo:fromWork ex:rb ; mo:source "bulbapedia"'
    )
    assert validate_fixture(data).rule_ids == {"X1"}


def test_unknown_aspect_fails(validate_fixture):
    data = REFERENCES + chunk('mo:aspect "gossip" ; mo:describes ex:fire ; mo:source "bulbapedia"')
    assert "X1" in validate_fixture(data).rule_ids


def test_two_describes_fail(validate_fixture):
    data = REFERENCES + chunk(
        'mo:aspect "biology" ; mo:describes ex:charmander , ex:fire ; mo:source "bulbapedia"'
    )
    assert validate_fixture(data).rule_ids == {"X1"}
