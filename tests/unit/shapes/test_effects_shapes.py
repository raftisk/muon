import pytest

EFFECT_CLASSES = ["MoveEffect", "AbilityEffect", "ItemEffect"]


def effect(properties: str, effect_class: str = "MoveEffect") -> str:
    return f'ex:effect a pkmn:{effect_class} ; pkmn:id "effect" ; {properties} .'


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
    data = 'ex:oran a pkmn:Berry ; pkmn:id "oran" ; pkmn:effectText "Restores 10 HP." .'
    assert validate_fixture(data).conforms


def test_effect_text_on_type_fails(validate_fixture):
    data = 'ex:fire a pkmn:Type ; pkmn:id "fire" ; pkmn:effectText "Hot." .'
    assert validate_fixture(data).rule_ids == {"X3"}


def test_effect_text_on_effect_node_fails(validate_fixture):
    assert validate_fixture(effect('pkmn:effectText "Hot."')).rule_ids == {"X3"}
