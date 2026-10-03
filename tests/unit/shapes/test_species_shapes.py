FIRE = 'ex:fire a pkmn:Type ; pkmn:id "fire" .'
FLYING = 'ex:flying a pkmn:Type ; pkmn:id "flying" .'
TYPES = FIRE + FLYING

CHARMANDER = """
ex:charmander a pkmn:PokemonSpecies ; pkmn:id "charmander" ; pkmn:pokedexNumber 4 ;
    pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.125 ;
    pkmn:growthRate "medium_slow" .
"""


def species(extra: str = "", dex: str = "pkmn:pokedexNumber 4 ;") -> str:
    return (
        'ex:mon a pkmn:PokemonSpecies ; pkmn:id "mon" ; '
        f"{dex} pkmn:primaryType ex:fire ; {extra} pkmn:isGenderless false ; "
        "pkmn:femaleRate 0.5 ."
    )


def test_single_type_species_conforms(validate_fixture):
    assert validate_fixture(TYPES + CHARMANDER).conforms


def test_two_type_species_conforms(validate_fixture):
    data = TYPES + species("pkmn:secondaryType ex:flying ;")
    assert validate_fixture(data).conforms


def test_two_primary_types_fail(validate_fixture):
    data = TYPES + species("pkmn:primaryType ex:flying ;")
    assert validate_fixture(data).rule_ids == {"S2"}


def test_two_secondary_types_fail(validate_fixture):
    data = TYPES + FIRE.replace("fire", "ice") + species("pkmn:secondaryType ex:flying , ex:ice ;")
    assert "S2" in validate_fixture(data).rule_ids


def test_same_type_twice_fails(validate_fixture):
    data = TYPES + species("pkmn:secondaryType ex:fire ;")
    assert validate_fixture(data).rule_ids == {"S2"}


def test_secondary_without_primary_fails(validate_fixture):
    data = TYPES + (
        'ex:mon a pkmn:PokemonSpecies ; pkmn:id "mon" ; pkmn:pokedexNumber 4 ; '
        "pkmn:secondaryType ex:flying ; pkmn:isGenderless false ; pkmn:femaleRate 0.5 ."
    )
    assert validate_fixture(data).rule_ids == {"S2"}


def test_missing_dex_number_fails(validate_fixture):
    assert validate_fixture(TYPES + species(dex="")).rule_ids == {"S1"}


def test_zero_dex_number_fails(validate_fixture):
    data = TYPES + species(dex="pkmn:pokedexNumber 0 ;")
    assert validate_fixture(data).rule_ids == {"S1"}


def test_form_is_validated_like_a_species(validate_fixture):
    data = (
        TYPES
        + CHARMANDER
        + (
            'ex:charmander-form a pkmn:PokemonSpecies ; pkmn:id "charmander-form" ; '
            "pkmn:primaryType ex:fire ; pkmn:isGenderless false ; pkmn:femaleRate 0.5 . "
            "ex:charmander pkmn:hasForm ex:charmander-form ."
        )
    )
    assert validate_fixture(data).rule_ids == {"S1"}


def test_female_rate_above_one_fails(validate_fixture):
    data = TYPES + species().replace("0.5", "1.5")
    assert validate_fixture(data).rule_ids == {"S3"}


def test_genderless_with_female_rate_fails(validate_fixture):
    data = TYPES + species().replace("isGenderless false", "isGenderless true")
    assert validate_fixture(data).rule_ids == {"S3"}


def test_genderless_without_female_rate_conforms(validate_fixture):
    data = TYPES + (
        'ex:magnemite a pkmn:PokemonSpecies ; pkmn:id "magnemite" ; pkmn:pokedexNumber 81 ; '
        "pkmn:primaryType ex:fire ; pkmn:isGenderless true ."
    )
    assert validate_fixture(data).conforms


def test_gendered_without_female_rate_fails(validate_fixture):
    data = TYPES + species().replace("pkmn:femaleRate 0.5", "")
    assert validate_fixture(data).rule_ids == {"S3"}


def test_unknown_growth_rate_fails(validate_fixture):
    data = TYPES + species('pkmn:growthRate "glacial" ;')
    assert validate_fixture(data).rule_ids == {"S3"}


def test_ev_amount_in_range_conforms(validate_fixture):
    data = "[] rdf:predicate pkmn:yieldsEv ; rdf:subject ex:a ; rdf:object ex:b ; pkmn:amount 2 ."
    assert validate_fixture(data).conforms


def test_ev_amount_above_three_fails(validate_fixture):
    data = "[] rdf:predicate pkmn:yieldsEv ; rdf:subject ex:a ; rdf:object ex:b ; pkmn:amount 5 ."
    assert validate_fixture(data).rule_ids == {"S4"}


def test_ev_amount_missing_fails(validate_fixture):
    data = "[] rdf:predicate pkmn:yieldsEv ; rdf:subject ex:a ; rdf:object ex:b ."
    assert validate_fixture(data).rule_ids == {"S4"}
