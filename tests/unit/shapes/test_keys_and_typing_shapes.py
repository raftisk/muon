FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
FLAMETHROWER = 'ex:flamethrower a pkmn:Move ; pkmn:id "flamethrower" ; pkmn:hasType ex:fire .'


def test_valid_slug_conforms(validate_fixture):
    assert validate_fixture(FIRE).conforms


def test_composite_reified_key_conforms(validate_fixture):
    data = 'ex:entry a pkmn:LearnsetEntry ; pkmn:id "diglett:earthquake:level:40:scarlet-violet" .'
    assert validate_fixture(data).conforms


def test_uppercase_id_fails(validate_fixture):
    report = validate_fixture('ex:fire a pkmn:Type ; pkmn:id "Fire" .')
    assert report.rule_ids == {"L2"}


def test_underscore_id_fails(validate_fixture):
    report = validate_fixture('ex:ball a pkmn:Move ; pkmn:id "shadow_ball" .')
    assert report.rule_ids == {"L2"}


def test_missing_id_fails(validate_fixture):
    assert validate_fixture("ex:fire a pkmn:Type .").rule_ids == {"L2"}


def test_two_ids_fail(validate_fixture):
    data = 'ex:fire a pkmn:Type ; pkmn:id "fire" , "fire-type" .'
    assert validate_fixture(data).rule_ids == {"L2"}


def test_edge_to_range_class_conforms(validate_fixture):
    assert validate_fixture(FIRE + FLAMETHROWER).conforms


def test_edge_to_wrong_class_fails_range(validate_fixture):
    data = FLAMETHROWER + 'ex:fire a pkmn:Ability ; pkmn:id "fire" .'
    report = validate_fixture(data)
    assert report.rule_ids == {"T1"}
    assert any("hasType" in message for message in report.messages)


def test_edge_from_wrong_class_fails_domain(validate_fixture):
    data = FIRE + 'ex:static a pkmn:Ability ; pkmn:id "static" ; pkmn:hasType ex:fire .'
    assert validate_fixture(data).rule_ids == {"T1"}


def test_subclass_of_range_class_conforms(validate_fixture):
    data = (
        'ex:diglett a pkmn:PokemonSpecies ; pkmn:id "diglett" ; mo:foundIn ex:kanto .'
        'ex:kanto a mo:Region ; pkmn:id "kanto" .'
    )
    assert validate_fixture(data).conforms


def test_undeclared_property_is_not_checked(validate_fixture):
    data = FIRE + 'ex:note-target a pkmn:Type ; pkmn:id "note" ; ex:related ex:fire .'
    assert validate_fixture(data).conforms
