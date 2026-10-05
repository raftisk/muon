FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" ; '
GRASS = 'ex:grass a pkmn:Type ; pkmn:id "grass" .'
ATTACK = 'ex:attack a pkmn:Stat ; pkmn:id "attack" .'
DEFENSE = 'ex:defense a pkmn:Stat ; pkmn:id "defense" .'
HP = 'pkmn:hp a pkmn:Stat ; pkmn:id "hp" .'
STATS = ATTACK + DEFENSE + HP


def nature(edges: str = "") -> str:
    return f'ex:adamant a pkmn:Nature ; pkmn:id "adamant" {edges}.'


def test_type_pair_with_one_property_conforms(validate_fixture):
    assert validate_fixture(FIRE + "pkmn:superEffectiveAgainst ex:grass ." + GRASS).conforms


def test_type_pair_with_two_properties_fails_k1(validate_fixture):
    data = FIRE + "pkmn:superEffectiveAgainst ex:grass ; pkmn:noEffectAgainst ex:grass ." + GRASS
    report = validate_fixture(data)
    assert report.rule_ids == {"K1"}
    assert any("#fire" in message and "#grass" in message for message in report.messages)


def test_type_pair_with_three_properties_fails_k1(validate_fixture):
    data = (
        FIRE
        + "pkmn:superEffectiveAgainst ex:grass ; pkmn:notVeryEffectiveAgainst ex:grass ;"
        + " pkmn:noEffectAgainst ex:grass ."
        + GRASS
    )
    assert validate_fixture(data).rule_ids == {"K1"}


def test_type_with_different_properties_for_different_defenders_conforms(validate_fixture):
    other = 'ex:water a pkmn:Type ; pkmn:id "water" .'
    data = (
        FIRE
        + "pkmn:superEffectiveAgainst ex:grass ; pkmn:notVeryEffectiveAgainst ex:water ."
        + GRASS
        + other
    )
    assert validate_fixture(data).conforms


def test_complete_nature_conforms(validate_fixture):
    data = STATS + nature("; pkmn:boostsStat ex:attack ; pkmn:reducesStat ex:defense ")
    assert validate_fixture(data).conforms


def test_neutral_nature_conforms(validate_fixture):
    assert validate_fixture(nature()).conforms


def test_nature_with_only_boost_fails_n1(validate_fixture):
    assert validate_fixture(STATS + nature("; pkmn:boostsStat ex:attack ")).rule_ids == {"N1"}


def test_nature_with_only_reduction_fails_n1(validate_fixture):
    assert validate_fixture(STATS + nature("; pkmn:reducesStat ex:defense ")).rule_ids == {"N1"}


def test_nature_with_same_stat_fails_n1(validate_fixture):
    data = STATS + nature("; pkmn:boostsStat ex:attack ; pkmn:reducesStat ex:attack ")
    assert validate_fixture(data).rule_ids == {"N1"}


def test_nature_boosting_hp_fails_n1(validate_fixture):
    data = STATS + nature("; pkmn:boostsStat pkmn:hp ; pkmn:reducesStat ex:defense ")
    assert validate_fixture(data).rule_ids == {"N1"}
