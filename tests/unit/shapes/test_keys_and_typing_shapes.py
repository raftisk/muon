from tests.unit.shapes.games_fixtures import GENERATION_NODE, build_version_group

FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'


def test_valid_slug_conforms(validate_fixture):
    assert validate_fixture(FIRE).conforms


def test_composite_reified_key_conforms(validate_fixture):
    data = (
        GENERATION_NODE
        + build_version_group("sv", "scarlet-violet")
        + 'ex:entry a pkmn:LearnsetEntry ; pkmn:id "diglett:earthquake:level:40:scarlet-violet" ; '
        "pkmn:inVersionGroup ex:sv ."
    )
    assert validate_fixture(data).conforms


def test_uppercase_id_fails(validate_fixture):
    report = validate_fixture('ex:fire a pkmn:Type ; pkmn:id "Fire" .')
    assert report.rule_ids == {"L2"}


def test_underscore_id_fails(validate_fixture):
    report = validate_fixture('ex:ball a pkmn:Type ; pkmn:id "shadow_ball" .')
    assert report.rule_ids == {"L2"}


def test_missing_id_fails(validate_fixture):
    assert validate_fixture("ex:fire a pkmn:Type .").rule_ids == {"L2"}


def test_two_ids_fail(validate_fixture):
    data = 'ex:fire a pkmn:Type ; pkmn:id "fire" , "fire-type" .'
    assert validate_fixture(data).rule_ids == {"L2"}
