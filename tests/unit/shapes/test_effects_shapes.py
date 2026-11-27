import pytest

from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    IN_FIRST_GENERATION,
    build_generation,
)

EFFECT_CLASSES = ["MoveEffect", "AbilityEffect", "ItemEffect"]


def effect(
    properties: str, effect_class: str = "MoveEffect", scope: str = IN_FIRST_GENERATION
) -> str:
    pairs = " ; ".join(pair for pair in (properties, scope) if pair)
    return GENERATION_NODE + f'ex:effect a pkmn:{effect_class} ; pkmn:id "effect" ; {pairs} .'


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
