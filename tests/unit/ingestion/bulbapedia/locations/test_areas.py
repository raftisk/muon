import pytest

from muon.ingestion.bulbapedia.locations.areas import (
    AreaRef,
    build_area_id,
    find_area,
    is_area_heading,
    is_game_heading,
    read_area_key,
    read_areas,
)
from muon.ingestion.bulbapedia.locations.identity import (
    LOCATION_INFOBOXES,
    LocationIndex,
    build_location_index,
)
from muon.ingestion.bulbapedia.pages import FETCH_HINT, PageStore
from muon.ingestion.bulbapedia.wikitext import split_sections

INDEX = LocationIndex(
    locations=frozenset({"Hau'oli Outskirts", "Mount Coronet"}),
    non_locations=frozenset(),
    redirects={},
    foreign_titles=frozenset(),
)
POKEMON = "pokémon"


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("B1F (Feebas lake)", "B1F"),
        ("B1F", "B1F"),
        ("Northern 1F room 1 (Eterna City - Celestic Town side)", "Northern 1F room 1"),
        ("4F room 1", "4F room 1"),
    ],
)
def test_area_key_drops_a_trailing_parenthetical(heading: str, expected: str) -> None:
    assert read_area_key(heading) == expected


@pytest.mark.parametrize(
    "heading",
    [
        "Generation IV",
        "Pokémon Sun and Moon",
        "Gold, Silver and Crystal",
        "Let's Go, Pikachu! and Let's Go, Eevee!",
        "Pokémon Omega Ruby and Pokémon Alpha Sapphire Special Demo Version",
        "Black 2 and White 2",
        "Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!",
        "Pokémon XD: Gale of Darkness",
        "Visible encounters",
        "Hidden items",
        "Prairie biome",
    ],
)
def test_non_area_headings(heading: str) -> None:
    assert not is_area_heading(heading)


@pytest.mark.parametrize("heading", ["1F", "B1F", "Entrance", "Exterior", "Northern half"])
def test_area_headings(heading: str) -> None:
    assert is_area_heading(heading)


def test_game_heading_needs_every_part_to_be_a_game() -> None:
    assert is_game_heading("Sun and Moon")
    assert not is_game_heading("Sun and Moon Cave")
    assert not is_game_heading("")


def test_area_is_the_deepest_area_heading() -> None:
    path = ("In the games", "Pokémon", "Generation IV", "2F", "Visible encounters")

    assert find_area(path, POKEMON, "Mount Coronet", INDEX) == AreaRef(
        key="2f", name="2F", location_title="Mount Coronet"
    )


def test_heading_that_names_a_pinned_location_points_at_it() -> None:
    path = ("Items", "Hau'oli Outskirts")

    area = find_area(path, "items", "Hau'oli City", INDEX)

    assert area == AreaRef(key="", name="Hau'oli Outskirts", location_title="Hau'oli Outskirts")


def test_section_with_only_non_area_headings_belongs_to_the_page() -> None:
    assert find_area(("Items", "Hidden items"), "items", "Cave", INDEX) is None
    assert find_area(("Pokémon", "Generation III"), POKEMON, "Cave", INDEX) is None
    assert find_area(("Pokémon",), POKEMON, "Cave", INDEX) is None


def test_section_outside_the_core_games_is_not_read() -> None:
    assert find_area(("In animation", "Pokémon", "Ash's team"), POKEMON, "Cave", INDEX) is None
    assert find_area(("Trivia", "Items", "1F"), "items", "Cave", INDEX) is None


def test_areas_are_unique_in_page_order_and_named_by_the_first_heading() -> None:
    wikitext = (
        "==Items==\n===B1F (Feebas lake)===\n===1F===\n"
        "==Pokémon==\n===Generation IV===\n====1F====\n====B1F====\n====4F====\n====4F room 1====\n"
    )

    areas = read_areas(split_sections(wikitext), "Cave", INDEX)

    assert [(area.key, area.name) for area in areas] == [
        ("b1f", "B1F"),
        ("1f", "1F"),
        ("4f", "4F"),
        ("4f-room-1", "4F room 1"),
    ]


def test_area_id_joins_the_location_id_and_the_key() -> None:
    assert build_area_id("mount-coronet", "4f-room-1") == "mount-coronet-4f-room-1"


PAGE_TYPE = "location"
pages = pytest.mark.pages


def read_cached_areas(title: str) -> tuple[AreaRef, ...]:
    store = PageStore()
    manifest = store.read_manifest()
    pins = manifest.list_pins(PAGE_TYPE)
    if manifest.find_pin(PAGE_TYPE, title) is None:
        pytest.skip(f"{title!r} is not pinned: {FETCH_HINT.format(page_type=PAGE_TYPE)}")
    foreign = [
        pin.title
        for name in ("species", "item", "move", "ability")
        for pin in manifest.list_pins(name)
    ]
    index = build_location_index(
        [store.read_page(PAGE_TYPE, pin) for pin in pins], foreign, LOCATION_INFOBOXES
    )
    page = store.read_page(PAGE_TYPE, manifest.find_pin(PAGE_TYPE, title))
    return read_areas(split_sections(page.wikitext), page.title, index)


@pages
def test_mt_moon_has_one_area_per_floor() -> None:
    keys = [area.key for area in read_cached_areas("Mt. Moon")]

    assert {"1f", "b1f", "b2f"} <= set(keys)
    assert len(keys) == len(set(keys))


@pages
def test_mount_coronet_keeps_distinct_keys_apart_and_merges_repeats() -> None:
    keys = [area.key for area in read_cached_areas("Mount Coronet")]

    assert keys.count("b1f") == 1
    assert keys.count("2f") == 1
    assert "4f" in keys
    assert "4f-room-1" in keys
