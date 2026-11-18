from tests.unit.shapes.games_fixtures import (
    GENERATION_NODE,
    build_version,
    build_version_group,
)

REFERENCES = (
    GENERATION_NODE
    + build_version_group("rb", "red-blue")
    + build_version("red", "red", "rb")
    + 'ex:cave a mo:Location ; pkmn:id "cave" .'
)


def item(properties: str = "", class_name: str = "pkmn:Item", generation: bool = True) -> str:
    link = "pkmn:introducedIn ex:gen1 ; " if generation else ""
    head = f'ex:potion a {class_name} ; pkmn:id "potion" ; '
    return f'{head}{link}{properties} pkmn:category "Medicine" .'


def test_complete_item_conforms(validate_fixture):
    data = REFERENCES + item("pkmn:buyPrice 200 ; pkmn:sellPrice 100 ; pkmn:flingPower 30 ;")
    assert validate_fixture(data).conforms


def test_item_with_only_a_generation_conforms(validate_fixture):
    assert validate_fixture(REFERENCES + item()).conforms


def test_item_without_a_generation_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item(generation=False)).rule_ids == {"I1"}


def test_machine_without_a_generation_conforms(validate_fixture):
    data = REFERENCES + (
        'ex:tm01 a pkmn:Machine ; pkmn:id "tm01-red-blue" ; pkmn:machineKind "TM" ; '
        "pkmn:machineNumber 1 ; pkmn:inVersionGroup ex:rb ; pkmn:teachesMove ex:move ."
        'ex:move a pkmn:Move ; pkmn:id "move" .'
    )
    assert "I1" not in validate_fixture(data).rule_ids


def test_negative_price_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item("pkmn:buyPrice -1 ;")).rule_ids == {"I1"}


def test_zero_price_conforms(validate_fixture):
    assert validate_fixture(REFERENCES + item("pkmn:sellPrice 0 ;")).conforms


def test_two_sell_prices_fail(validate_fixture):
    data = REFERENCES + item("pkmn:sellPrice 10, 20 ;")
    assert validate_fixture(data).rule_ids == {"I1"}


def test_zero_fling_power_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item("pkmn:flingPower 0 ;")).rule_ids == {"I1"}


def test_price_that_is_not_an_integer_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item('pkmn:buyPrice "cheap" ;')).rule_ids == {"I1"}


FLAVOR_EDGE = "[] rdf:predicate pkmn:hasFlavor ; rdf:subject ex:oran ; rdf:object ex:spicy"


def test_flavor_potency_conforms(validate_fixture):
    assert validate_fixture(f"{FLAVOR_EDGE} ; pkmn:potency 10 .").conforms


def test_flavor_potency_of_zero_fails(validate_fixture):
    assert validate_fixture(f"{FLAVOR_EDGE} ; pkmn:potency 0 .").rule_ids == {"I2"}


def test_flavor_potency_missing_fails(validate_fixture):
    assert validate_fixture(f"{FLAVOR_EDGE} .").rule_ids == {"I2"}


FLAVORS = (
    'ex:spicy a pkmn:Flavor ; pkmn:id "spicy" . ex:dry a pkmn:Flavor ; pkmn:id "dry" . '
    'ex:attack a pkmn:Stat ; pkmn:id "attack" . ex:defense a pkmn:Stat ; pkmn:id "defense" .'
)


def nature(flavors: str) -> str:
    return (
        'ex:lonely a pkmn:Nature ; pkmn:id "lonely" ; pkmn:boostsStat ex:attack ; '
        f"pkmn:reducesStat ex:defense ; {flavors} ."
    )


def test_nature_with_2_flavors_conforms(validate_fixture):
    data = FLAVORS + nature("pkmn:likesFlavor ex:spicy ; pkmn:dislikesFlavor ex:dry")
    assert validate_fixture(data).conforms


def test_nature_with_no_flavors_conforms(validate_fixture):
    assert validate_fixture(FLAVORS + nature("")).conforms is True


def test_nature_with_one_flavor_fails(validate_fixture):
    data = FLAVORS + nature("pkmn:likesFlavor ex:spicy")
    assert validate_fixture(data).rule_ids == {"N2"}


def test_nature_with_the_same_flavor_twice_fails(validate_fixture):
    data = FLAVORS + nature("pkmn:likesFlavor ex:spicy ; pkmn:dislikesFlavor ex:spicy")
    assert validate_fixture(data).rule_ids == {"N2"}


def item_find(properties: str = "pkmn:atLocation ex:cave ; pkmn:inVersion ex:red") -> str:
    return f'ex:find a pkmn:ItemFind ; pkmn:id "potion:cave:red" ; {properties} .'


def test_item_find_conforms(validate_fixture):
    assert validate_fixture(REFERENCES + item_find()).conforms


def test_hidden_item_find_conforms(validate_fixture):
    data = REFERENCES + item_find(
        "pkmn:atLocation ex:cave ; pkmn:inVersion ex:red ; pkmn:isHidden true"
    )
    assert validate_fixture(data).conforms


def test_item_find_without_a_location_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item_find("pkmn:inVersion ex:red")).rule_ids == {"C4"}


def test_item_find_without_a_version_fails(validate_fixture):
    assert validate_fixture(REFERENCES + item_find("pkmn:atLocation ex:cave")).rule_ids == {"C4"}


def test_item_find_with_a_non_boolean_hidden_fails(validate_fixture):
    data = REFERENCES + item_find(
        'pkmn:atLocation ex:cave ; pkmn:inVersion ex:red ; pkmn:isHidden "yes"'
    )
    assert validate_fixture(data).rule_ids == {"C4"}


ADJACENCY = "[] rdf:predicate mo:adjacentTo ; rdf:subject ex:a ; rdf:object ex:b"


def test_adjacency_direction_conforms(validate_fixture):
    assert validate_fixture(f'{ADJACENCY} ; pkmn:direction "north" .').conforms


def test_adjacency_with_an_unknown_direction_fails(validate_fixture):
    assert validate_fixture(f'{ADJACENCY} ; pkmn:direction "up" .').rule_ids == {"C5"}


def test_adjacency_without_a_direction_fails(validate_fixture):
    assert validate_fixture(f"{ADJACENCY} .").rule_ids == {"C5"}
