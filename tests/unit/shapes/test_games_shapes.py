import pytest

from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    build_version,
    build_version_group,
)

SECOND_GENERATION = 'ex:gen2 a pkmn:Generation ; pkmn:id "gen2" ; mo:sequenceNumber 2 .'
GROUP = build_version_group("rb", "red-blue")
CHAIN = GENERATION_NODE + GROUP + build_version("red", "red", "rb")


def test_full_chain_conforms(validate_fixture):
    assert validate_fixture(CHAIN).conforms


def test_version_without_group_fails(validate_fixture):
    data = GENERATION_NODE + GROUP + 'ex:red a pkmn:Version ; pkmn:id "red" .'
    assert validate_fixture(data).rule_ids == {"G1"}


def test_version_with_two_groups_fails(validate_fixture):
    data = (
        GENERATION_NODE
        + GROUP
        + build_version_group("gs", "gold-silver")
        + 'ex:red a pkmn:Version ; pkmn:id "red" ; mo:partOf ex:rb , ex:gs .'
    )
    assert validate_fixture(data).rule_ids == {"G1"}


def test_version_part_of_a_generation_fails(validate_fixture):
    data = GENERATION_NODE + GROUP + 'ex:red a pkmn:Version ; pkmn:id "red" ; mo:partOf ex:gen1 .'
    assert validate_fixture(data).rule_ids == {"G1"}


def test_group_without_generation_fails(validate_fixture):
    assert validate_fixture('ex:rb a pkmn:VersionGroup ; pkmn:id "red-blue" .').rule_ids == {"G2"}


def test_group_with_two_generations_fails(validate_fixture):
    data = (
        GENERATION_NODE
        + SECOND_GENERATION
        + 'ex:rb a pkmn:VersionGroup ; pkmn:id "red-blue" ; mo:partOf ex:gen1 , ex:gen2 .'
    )
    assert validate_fixture(data).rule_ids == {"G2"}


def test_group_part_of_a_version_fails(validate_fixture):
    data = (
        GENERATION_NODE
        + GROUP
        + build_version("red", "red", "rb")
        + 'ex:other a pkmn:VersionGroup ; pkmn:id "other" ; mo:partOf ex:red .'
    )
    assert validate_fixture(data).rule_ids == {"G2"}


def build_generation(properties: str) -> str:
    return f'ex:gen3 a pkmn:Generation ; pkmn:id "gen3" {properties} .'


def test_generation_with_one_sequence_number_conforms(validate_fixture):
    assert validate_fixture(build_generation("; mo:sequenceNumber 3")).conforms


@pytest.mark.parametrize(
    "properties",
    ["", "; mo:sequenceNumber 3 , 4", "; mo:sequenceNumber 0"],
    ids=["missing", "repeated", "zero"],
)
def test_generation_with_wrong_sequence_number_fails(validate_fixture, properties):
    assert validate_fixture(build_generation(properties)).rule_ids == {"G4"}
