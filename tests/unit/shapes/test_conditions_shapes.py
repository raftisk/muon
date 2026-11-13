import pytest

CONDITION_CLASSES = ["VolatileCondition", "Weather"]


def condition(properties: str, condition_class: str) -> str:
    values = f" ; {properties}" if properties else ""
    return f'ex:condition a pkmn:{condition_class} ; pkmn:id "condition"{values} .'


@pytest.mark.parametrize("condition_class", CONDITION_CLASSES)
@pytest.mark.parametrize(
    "properties", ["", "pkmn:minTurns 4 ; pkmn:maxTurns 5", "pkmn:minTurns 5 ; pkmn:maxTurns 5"]
)
def test_condition_durations_conform(validate_fixture, condition_class, properties):
    assert validate_fixture(condition(properties, condition_class)).conforms


@pytest.mark.parametrize("condition_class", CONDITION_CLASSES)
@pytest.mark.parametrize(
    "properties",
    [
        "pkmn:minTurns 4",
        "pkmn:maxTurns 5",
        "pkmn:minTurns 0 ; pkmn:maxTurns 5",
        "pkmn:minTurns 5 ; pkmn:maxTurns 4",
        "pkmn:minTurns 4 , 5 ; pkmn:maxTurns 5",
    ],
)
def test_wrong_condition_durations_fail(validate_fixture, condition_class, properties):
    assert validate_fixture(condition(properties, condition_class)).rule_ids == {"B1"}
