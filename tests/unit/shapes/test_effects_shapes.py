import pytest

from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    IN_FIRST_GENERATION,
    build_generation,
)

EFFECT_CLASSES = ["MoveEffect", "AbilityEffect", "ItemEffect"]


def effect(
    properties: str,
    effect_class: str = "MoveEffect",
    scope: str = IN_FIRST_GENERATION,
    local_name: str = "effect",
) -> str:
    pairs = " ; ".join(pair for pair in (properties, scope) if pair)
    node = f'ex:{local_name} a pkmn:{effect_class} ; pkmn:id "{local_name}" ; {pairs} .'
    return GENERATION_NODE + node


@pytest.mark.parametrize("effect_class", EFFECT_CLASSES)
def test_valid_effect_conforms(validate_fixture, effect_class):
    data = effect('pkmn:probability 30 ; pkmn:statStages -1 ; pkmn:target "foe"', effect_class)
    assert validate_fixture(data).conforms


@pytest.mark.parametrize("effect_class", EFFECT_CLASSES)
def test_probability_above_100_fails(validate_fixture, effect_class):
    report = validate_fixture(effect("pkmn:probability 150", effect_class))
    assert report.rule_ids == {"E1"}


def test_negative_probability_fails(validate_fixture):
    assert validate_fixture(effect("pkmn:probability -1")).rule_ids == {"E1"}


def test_stat_stages_above_six_fails(validate_fixture):
    assert validate_fixture(effect("pkmn:statStages 7")).rule_ids == {"E1"}


def test_stat_stages_below_minus_six_fails(validate_fixture):
    assert validate_fixture(effect("pkmn:statStages -7")).rule_ids == {"E1"}


def test_hp_threshold_above_one_fails(validate_fixture):
    data = effect("pkmn:hpThreshold 2.0", "AbilityEffect")
    assert validate_fixture(data).rule_ids == {"E1"}


def test_hp_threshold_half_conforms(validate_fixture):
    data = effect('pkmn:hpThreshold 0.5 ; pkmn:trigger "low_hp"', "AbilityEffect")
    assert validate_fixture(data).conforms


def test_unknown_trigger_fails(validate_fixture):
    data = effect('pkmn:trigger "on_sneeze"', "AbilityEffect")
    assert validate_fixture(data).rule_ids == {"E1"}


def test_unknown_target_fails(validate_fixture):
    assert validate_fixture(effect('pkmn:target "everyone"')).rule_ids == {"E1"}


def test_effect_text_on_ability_conforms(validate_fixture):
    data = 'ex:static a pkmn:Ability ; pkmn:id "static" ; pkmn:effectText "May paralyze." .'
    assert validate_fixture(data).conforms


def test_effect_text_on_berry_conforms(validate_fixture):
    data = GENERATION_NODE + (
        'ex:oran a pkmn:Berry ; pkmn:id "oran" ; pkmn:introducedIn ex:gen1 ; '
        'pkmn:effectText "Restores 10 HP." .'
    )
    assert validate_fixture(data).conforms


def test_effect_text_on_type_fails(validate_fixture):
    data = 'ex:fire a pkmn:Type ; pkmn:id "fire" ; pkmn:effectText "Hot." .'
    assert validate_fixture(data).rule_ids == {"X3"}


def test_effect_text_on_effect_node_fails(validate_fixture):
    assert validate_fixture(effect('pkmn:effectText "Hot."')).rule_ids == {"X3"}


def test_overworld_move_effect_without_action_edge_conforms(validate_fixture):
    assert validate_fixture(effect("pkmn:isOverworld true")).conforms


def test_unseen_fist_effect_conforms(validate_fixture):
    contact = 'ex:contact a pkmn:MoveFlag ; pkmn:id "contact" .'
    properties = 'pkmn:isProtectable false ; pkmn:trigger "on_attack" ; pkmn:triggeredBy ex:contact'
    assert validate_fixture(contact + effect(properties, "AbilityEffect")).conforms


@pytest.mark.parametrize("effect_class", EFFECT_CLASSES)
@pytest.mark.parametrize(
    "scope",
    ["", "pkmn:introducedIn ex:gen1 , ex:gen3"],
    ids=["no-start", "two-starts"],
)
def test_effect_without_exactly_one_start_fails(validate_fixture, effect_class, scope):
    data = build_generation(3) + effect("pkmn:statStages 1", effect_class, scope)
    assert validate_fixture(data).rule_ids == {"G3"}


def test_effect_with_two_ends_fails(validate_fixture):
    scope = IN_FIRST_GENERATION + " ; pkmn:lastValidIn ex:gen3 , ex:gen4"
    data = build_generation(3) + build_generation(4) + effect("", scope=scope)
    assert validate_fixture(data).rule_ids == {"G3"}


def test_effect_that_ends_before_it_starts_fails_and_names_both(validate_fixture):
    scope = "pkmn:introducedIn ex:gen4 ; pkmn:lastValidIn ex:gen3"
    data = build_generation(3) + build_generation(4) + effect("", "ItemEffect", scope)

    report = validate_fixture(data)

    assert report.rule_ids == {"G3"}
    assert any("gen3" in message and "gen4" in message for message in report.messages)


def test_effect_that_starts_and_ends_in_1_generation_conforms(validate_fixture):
    # Sitrus Berry restored 30 HP in Generation III only.
    scope = "pkmn:introducedIn ex:gen3 ; pkmn:lastValidIn ex:gen3"
    data = build_generation(3) + effect('pkmn:trigger "low_hp"', "ItemEffect", scope)
    assert validate_fixture(data).conforms


@pytest.mark.parametrize(
    ("effect_class", "properties"),
    [
        ("MoveEffect", "pkmn:isOverworld true , false"),
        ("AbilityEffect", "pkmn:isOverworld true , false"),
        ("AbilityEffect", 'pkmn:isProtectable "no"'),
        ("AbilityEffect", "pkmn:isProtectable true , false"),
    ],
)
def test_wrong_effect_booleans_fail(validate_fixture, effect_class, properties):
    assert validate_fixture(effect(properties, effect_class)).rule_ids == {"E1"}


ATTACK = 'ex:attack a pkmn:Stat ; pkmn:id "attack" .'
DEFENSE = 'ex:defense a pkmn:Stat ; pkmn:id "defense" .'
HP = 'ex:hp a pkmn:Stat ; pkmn:id "hp" .'
CRITICAL = 'ex:critical a pkmn:BattleStat ; pkmn:id "critical-hit-ratio" .'
BURN = 'ex:burn a pkmn:StatusCondition ; pkmn:id "burn" .'
FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
ELECTRIC = 'ex:electric a pkmn:Type ; pkmn:id "electric" .'
FAIRY = 'ex:fairy a pkmn:Type ; pkmn:id "fairy" .'
CONTACT = 'ex:contact a pkmn:MoveFlag ; pkmn:id "contact" .'
STATIC = 'ex:static a pkmn:Ability ; pkmn:id "static" .'
TARGETS = ATTACK + DEFENSE + HP + CRITICAL + BURN + FIRE + ELECTRIC + FAIRY + CONTACT + STATIC


def effects(*effect_snippets: str) -> str:
    """Effect nodes with their targets. Each snippet comes from `effect`."""
    return TARGETS + "".join(effect_snippets)


def test_shell_smash_as_2_nodes_conforms(validate_fixture):
    raised = "pkmn:statStages 2 ; pkmn:affectsStat ex:attack , ex:defense"
    lowered = "pkmn:statStages -1 ; pkmn:affectsStat ex:defense"
    data = effects(effect(raised), effect(lowered, local_name="second"))
    assert validate_fixture(data).conforms


def test_2_action_properties_fail(validate_fixture):
    properties = "pkmn:inflictsCondition ex:burn ; pkmn:curesCondition ex:burn"
    data = effects(effect(properties, "AbilityEffect"))
    assert validate_fixture(data).rule_ids == {"E2"}


def test_rocky_helmet_without_an_action_edge_conforms(validate_fixture):
    properties = 'pkmn:hpChange -0.1667 ; pkmn:hpBasis "max_hp" ; pkmn:trigger "on_hit"'
    assert validate_fixture(effects(effect(properties, "ItemEffect"))).conforms


def test_filter_on_a_move_flag_conforms(validate_fixture):
    data = effects(effect("pkmn:triggeredBy ex:contact , ex:fire", "AbilityEffect"))
    assert validate_fixture(data).conforms


@pytest.mark.parametrize("effect_class", EFFECT_CLASSES)
def test_effect_triggered_by_an_ability_fails(validate_fixture, effect_class):
    data = effects(effect("pkmn:triggeredBy ex:static", effect_class))
    assert validate_fixture(data).rule_ids == {"E3"}


def test_fluffy_as_2_nodes_conforms(validate_fixture):
    contact = 'pkmn:damageMultiplier 0.5 ; pkmn:triggeredBy ex:contact ; pkmn:trigger "on_hit"'
    fire = 'pkmn:damageMultiplier 2.0 ; pkmn:triggeredBy ex:fire ; pkmn:trigger "on_hit"'
    data = effects(
        effect(contact, "AbilityEffect"), effect(fire, "AbilityEffect", local_name="second")
    )
    assert validate_fixture(data).conforms


def test_stat_multiplier_with_its_stat_conforms(validate_fixture):
    properties = "pkmn:statMultiplier 1.5 ; pkmn:affectsStat ex:attack"
    assert validate_fixture(effects(effect(properties, "ItemEffect"))).conforms


def test_critical_hit_ratio_can_be_raised(validate_fixture):
    properties = "pkmn:statStages 2 ; pkmn:affectsStat ex:critical"
    assert validate_fixture(effects(effect(properties))).conforms


@pytest.mark.parametrize(
    "properties",
    [
        "pkmn:statMultiplier 1.5",
        "pkmn:powerMultiplier 1.2 ; pkmn:damageMultiplier 1.3",
        "pkmn:powerMultiplier 1.2 ; pkmn:affectsStat ex:attack",
        "pkmn:damageMultiplier 1.2 ; pkmn:affectsStat ex:attack",
        "pkmn:damageMultiplier 0.0",
        "pkmn:powerMultiplier 1.2 , 1.5",
        "pkmn:statStages 1 ; pkmn:affectsStat ex:hp",
        "pkmn:blocksStatDrop ex:hp",
        "pkmn:blocksStatDrop ex:critical",
    ],
    ids=[
        "stat-multiplier-without-stat",
        "2-multipliers",
        "power-with-stat",
        "damage-with-stat",
        "zero-multiplier",
        "repeated-multiplier",
        "affects-hp",
        "blocks-hp-drop",
        "blocks-critical-drop",
    ],
)
def test_wrong_multiplier_or_stat_target_fails(validate_fixture, properties):
    data = effects(effect(properties, "ItemEffect"))
    assert validate_fixture(data).rule_ids == {"E4"}


@pytest.mark.parametrize(
    "properties",
    [
        'pkmn:hpChange 0.25 ; pkmn:hpBasis "max_hp" ; pkmn:restoresStat ex:hp',
        "pkmn:amount 10 ; pkmn:restoresStat ex:hp",
        'pkmn:hpChange -0.5 ; pkmn:hpBasis "current_hp"',
        'pkmn:hpChange 1.0 ; pkmn:hpBasis "damage_dealt" ; pkmn:restoresStat ex:hp',
    ],
    ids=["share", "amount", "loss", "whole"],
)
def test_valid_hp_effect_conforms(validate_fixture, properties):
    assert validate_fixture(effects(effect(properties, "ItemEffect"))).conforms


@pytest.mark.parametrize(
    "properties",
    [
        "pkmn:hpChange 0.25",
        'pkmn:hpBasis "max_hp"',
        'pkmn:hpChange 1.5 ; pkmn:hpBasis "max_hp"',
        'pkmn:hpChange 0.0 ; pkmn:hpBasis "max_hp"',
        'pkmn:hpChange 0.5 ; pkmn:hpBasis "base_hp"',
        "pkmn:restoresStat ex:hp",
        'pkmn:amount 10 ; pkmn:hpChange 0.25 ; pkmn:hpBasis "max_hp" ; pkmn:restoresStat ex:hp',
        'pkmn:hpChange -0.25 ; pkmn:hpBasis "max_hp" ; pkmn:restoresStat ex:hp',
        'pkmn:amount 10 ; pkmn:hpChange -0.25 ; pkmn:hpBasis "max_hp" ; pkmn:restoresStat ex:hp',
    ],
    ids=[
        "no-basis",
        "basis-alone",
        "above-1",
        "zero",
        "unknown-basis",
        "restore-without-value",
        "amount-and-share",
        "loss-that-restores",
        "loss-with-amount",
    ],
)
def test_wrong_hp_effect_fails(validate_fixture, properties):
    data = effects(effect(properties, "ItemEffect"))
    assert validate_fixture(data).rule_ids == {"E5"}


def test_pixilate_with_target_move_conforms(validate_fixture):
    properties = 'pkmn:target "move" ; pkmn:setsType ex:fairy ; pkmn:powerMultiplier 1.2'
    assert validate_fixture(effects(effect(properties, "AbilityEffect"))).conforms


def test_target_move_without_a_change_to_the_move_fails(validate_fixture):
    data = effects(effect('pkmn:target "move" ; pkmn:statStages 1', "AbilityEffect"))
    assert validate_fixture(data).rule_ids == {"E6"}


def test_volt_absorb_as_immunity_and_reward_nodes_conforms(validate_fixture):
    immunity = 'pkmn:immuneTo ex:electric ; pkmn:trigger "passive"'
    reward = (
        'pkmn:hpChange 0.25 ; pkmn:hpBasis "max_hp" ; pkmn:restoresStat ex:hp ; '
        'pkmn:trigger "on_hit" ; pkmn:triggeredBy ex:electric'
    )
    data = effects(
        effect(immunity, "AbilityEffect"), effect(reward, "AbilityEffect", local_name="second")
    )
    assert validate_fixture(data).conforms


@pytest.mark.parametrize(
    "properties",
    [
        'pkmn:immuneTo ex:burn ; pkmn:trigger "passive"',
        "pkmn:immuneTo ex:electric",
        'pkmn:immuneTo ex:contact ; pkmn:trigger "on_hit"',
    ],
    ids=["condition-target", "no-trigger", "active-trigger"],
)
def test_wrong_immunity_fails(validate_fixture, properties):
    data = effects(effect(properties, "AbilityEffect"))
    assert validate_fixture(data).rule_ids == {"E7"}
