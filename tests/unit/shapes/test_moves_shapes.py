from tests.unit.shapes.games_fixtures import GENERATION_NODE, build_version_group
from tests.unit.shapes.move_fixtures import build_move

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
ICE = 'ex:ice a pkmn:Type ; pkmn:id "ice" .'
SPECIAL = 'ex:special a pkmn:MoveDamageClass ; pkmn:id "special" .'
GROUP = build_version_group("sv", "scarlet-violet")
EARTHQUAKE = build_move("earthquake")
REFERENCES = FIRE + ICE + SPECIAL + GENERATION_NODE + GROUP + EARTHQUAKE
MOVE_REFERENCES = FIRE + ICE + SPECIAL + GENERATION_NODE


def test_move_with_type_and_class_conforms(validate_fixture):
    assert validate_fixture(MOVE_REFERENCES + build_move("tackle")).conforms


def test_move_in_one_transaction_with_its_edges_conforms(validate_fixture):
    assert validate_fixture(REFERENCES).conforms


def test_move_with_two_types_fails(validate_fixture):
    data = MOVE_REFERENCES + build_move("tackle", extra="pkmn:hasType ex:ice")
    assert validate_fixture(data).rule_ids == {"M1"}


def test_move_without_type_fails(validate_fixture):
    data = MOVE_REFERENCES + build_move("tackle", type_local_name=None)
    assert validate_fixture(data).rule_ids == {"M1"}


def test_move_without_damage_class_fails(validate_fixture):
    data = MOVE_REFERENCES + build_move("tackle", damage_class_local_name=None)
    assert validate_fixture(data).rule_ids == {"M1"}


def test_name_only_move_fails(validate_fixture):
    data = 'ex:bare a pkmn:Move ; pkmn:id "bare" ; rdfs:label "Bare"@en .'
    report = validate_fixture(data)
    assert report.rule_ids == {"M1"}
    assert len(report.messages) == 2


def test_complete_machine_conforms(validate_fixture):
    data = REFERENCES + (
        'ex:tm114 a pkmn:Machine ; pkmn:id "tm114-scarlet-violet" ; '
        "pkmn:inVersionGroup ex:sv ; pkmn:teachesMove ex:earthquake ."
    )
    assert validate_fixture(data).conforms


def test_machine_without_version_group_fails(validate_fixture):
    data = REFERENCES + (
        'ex:tm114 a pkmn:Machine ; pkmn:id "tm114-scarlet-violet" ; '
        "pkmn:teachesMove ex:earthquake ."
    )
    assert validate_fixture(data).rule_ids == {"C2"}


def test_machine_without_move_fails(validate_fixture):
    data = REFERENCES + (
        'ex:tm114 a pkmn:Machine ; pkmn:id "tm114-scarlet-violet" ; pkmn:inVersionGroup ex:sv .'
    )
    assert validate_fixture(data).rule_ids == {"C2"}


def test_learnset_entry_with_one_version_group_conforms(validate_fixture):
    data = REFERENCES + (
        'ex:entry a pkmn:LearnsetEntry ; pkmn:id "diglett:earthquake:level:40:scarlet-violet" ; '
        "pkmn:inVersionGroup ex:sv ."
    )
    assert validate_fixture(data).conforms


def test_learnset_entry_with_two_version_groups_fails(validate_fixture):
    data = (
        REFERENCES
        + build_version_group("other", "sword-shield")
        + 'ex:entry a pkmn:LearnsetEntry ; pkmn:id "diglett:earthquake:level:40:scarlet-violet" ; '
        "pkmn:inVersionGroup ex:sv , ex:other ."
    )
    assert validate_fixture(data).rule_ids == {"C2"}
