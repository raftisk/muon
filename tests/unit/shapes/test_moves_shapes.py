import pytest

from tests.unit.shapes.games_fixtures import GENERATION_NODE, build_version_group
from tests.unit.shapes.move_fixtures import MOVE_PP, MOVE_PRIORITY, MOVE_TARGET, build_move

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
ICE = 'ex:ice a pkmn:Type ; pkmn:id "ice" .'
SPECIAL = 'ex:special a pkmn:MoveDamageClass ; pkmn:id "special" .'
MULTIHIT = 'ex:multihit a pkmn:MoveFlag ; pkmn:id "multihit" .'
GROUP = build_version_group("sv", "scarlet-violet")
EARTHQUAKE = build_move("earthquake")
REFERENCES = FIRE + ICE + SPECIAL + GENERATION_NODE + GROUP + EARTHQUAKE
MOVE_REFERENCES = FIRE + ICE + SPECIAL + GENERATION_NODE + MULTIHIT
REQUIRED_PAIRS = {
    "pp": f"pkmn:pp {MOVE_PP}",
    "priority": f"pkmn:priority {MOVE_PRIORITY}",
    "target": f'pkmn:target "{MOVE_TARGET}"',
    "introducedIn": "pkmn:introducedIn ex:gen1",
}
TARGET_CODES = (
    "any_adjacent",
    "self",
    "all_adjacent_foes",
    "any",
    "all_adjacent",
    "entire_field",
    "user_side",
    "user_and_allies",
    "random_foe",
    "varies",
    "all",
    "foe_side",
    "adjacent_ally",
    "user_or_adjacent_ally",
    "adjacent_foe",
    "all_allies",
)
MACHINE_LINKS = "pkmn:inVersionGroup ex:sv ; pkmn:teachesMove ex:earthquake"
MACHINE_NUMBERING = 'pkmn:machineKind "TM" ; pkmn:machineNumber 114'


def replace_pair(move: str, pair: str, replacement: str) -> str:
    """Swap 1 predicate-object pair of a `build_move` snippet; an empty replacement drops it."""
    old = f" ; {pair}"
    assert old in move, pair
    return move.replace(old, f" ; {replacement}" if replacement else "")


def build_machine(node_id: str, values: str) -> str:
    return f'ex:machine a pkmn:Machine ; pkmn:id "{node_id}" ; {values} .'


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
    assert report.rule_ids == {"M1", "M2"}
    assert len(report.messages) == 6


@pytest.mark.parametrize("property_name", REQUIRED_PAIRS)
def test_move_without_required_value_fails(validate_fixture, property_name):
    move = replace_pair(build_move("tackle"), REQUIRED_PAIRS[property_name], "")
    report = validate_fixture(MOVE_REFERENCES + move)
    assert report.rule_ids == {"M2"}
    assert any(f"1 {property_name}" in message for message in report.messages)


def test_move_with_unknown_target_lists_the_codes(validate_fixture):
    move = replace_pair(build_move("tackle"), REQUIRED_PAIRS["target"], 'pkmn:target "anyadjacent"')
    report = validate_fixture(MOVE_REFERENCES + move)
    assert report.rule_ids == {"M2"}
    assert all(code in report.messages[0] for code in TARGET_CODES)


def test_multihit_move_conforms(validate_fixture):
    extra = "pkmn:hasFlag ex:multihit ; pkmn:minHits 2 ; pkmn:maxHits 5"
    assert validate_fixture(MOVE_REFERENCES + build_move("bullet-seed", extra=extra)).conforms


@pytest.mark.parametrize(
    "extra",
    [
        "pkmn:hasFlag ex:multihit ; pkmn:minHits 2",
        "pkmn:hasFlag ex:multihit ; pkmn:maxHits 5",
        "pkmn:hasFlag ex:multihit ; pkmn:minHits 5 ; pkmn:maxHits 2",
        "pkmn:hasFlag ex:multihit ; pkmn:minHits 0 ; pkmn:maxHits 2",
        "pkmn:minHits 2 ; pkmn:maxHits 5",
    ],
)
def test_wrong_hit_counts_fail(validate_fixture, extra):
    report = validate_fixture(MOVE_REFERENCES + build_move("bullet-seed", extra=extra))
    assert report.rule_ids == {"M2"}


@pytest.mark.parametrize(
    "move",
    [
        build_move("tackle", extra="pkmn:accuracy 0"),
        build_move("tackle", extra="pkmn:accuracy 101"),
        build_move("tackle", extra="pkmn:power 0"),
        build_move("tackle", extra="pkmn:critStage 4"),
        replace_pair(build_move("tackle"), REQUIRED_PAIRS["priority"], "pkmn:priority 6"),
        replace_pair(build_move("tackle"), REQUIRED_PAIRS["priority"], "pkmn:priority -8"),
        replace_pair(build_move("tackle"), REQUIRED_PAIRS["pp"], "pkmn:pp 0"),
    ],
)
def test_move_value_out_of_range_fails(validate_fixture, move):
    assert validate_fixture(MOVE_REFERENCES + move).rule_ids == {"M2"}


def test_move_values_in_range_conform(validate_fixture):
    extra = (
        "pkmn:accuracy 100 ; pkmn:power 1 ; pkmn:critStage 3 ; "
        "pkmn:minTurns 2 ; pkmn:maxTurns 3 ; pkmn:isProtectable true ; pkmn:isMirrorable false"
    )
    move = replace_pair(
        build_move("outrage", extra=extra), REQUIRED_PAIRS["priority"], "pkmn:priority -7"
    )
    data = MOVE_REFERENCES + move
    assert validate_fixture(data).conforms


@pytest.mark.parametrize(
    "extra",
    [
        "pkmn:minTurns 2",
        "pkmn:maxTurns 3",
        "pkmn:minTurns 0 ; pkmn:maxTurns 3",
        "pkmn:minTurns 3 ; pkmn:maxTurns 2",
    ],
)
def test_wrong_move_turns_fail(validate_fixture, extra):
    report = validate_fixture(MOVE_REFERENCES + build_move("outrage", extra=extra))
    assert report.rule_ids == {"M2"}


@pytest.mark.parametrize("extra", ["pkmn:isProtectable true , false", 'pkmn:isMirrorable "yes"'])
def test_wrong_move_booleans_fail(validate_fixture, extra):
    report = validate_fixture(MOVE_REFERENCES + build_move("tackle", extra=extra))
    assert report.rule_ids == {"M2"}


def test_complete_machine_conforms(validate_fixture):
    data = REFERENCES + build_machine(
        "tm114-scarlet-violet", f"{MACHINE_LINKS} ; {MACHINE_NUMBERING}"
    )
    assert validate_fixture(data).conforms


def test_gold_silver_machine_conforms(validate_fixture):
    values = (
        "pkmn:inVersionGroup ex:gs ; pkmn:teachesMove ex:earthquake ; "
        'rdfs:label "TM30"@en ; pkmn:machineKind "TM" ; pkmn:machineNumber 30'
    )
    data = REFERENCES + build_version_group("gs", "gold-silver")
    assert validate_fixture(data + build_machine("tm30-gold-silver", values)).conforms


def test_machine_number_zero_conforms(validate_fixture):
    values = (
        "pkmn:inVersionGroup ex:swsh ; pkmn:teachesMove ex:earthquake ; "
        'pkmn:machineKind "TM" ; pkmn:machineNumber 0'
    )
    data = REFERENCES + build_version_group("swsh", "sword-shield")
    assert validate_fixture(data + build_machine("tm00-sword-shield", values)).conforms


@pytest.mark.parametrize(
    "values",
    [
        f"pkmn:teachesMove ex:earthquake ; {MACHINE_NUMBERING}",
        f"pkmn:inVersionGroup ex:sv ; {MACHINE_NUMBERING}",
        f'{MACHINE_LINKS} ; pkmn:machineKind "tm" ; pkmn:machineNumber 114',
        f"{MACHINE_LINKS} ; pkmn:machineNumber 114",
        f'{MACHINE_LINKS} ; pkmn:machineKind "TM"',
        f'{MACHINE_LINKS} ; pkmn:machineKind "TM" ; pkmn:machineNumber -1',
    ],
)
def test_incomplete_machine_fails(validate_fixture, values):
    data = REFERENCES + build_machine("tm114-scarlet-violet", values)
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
