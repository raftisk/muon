"""Golden tests of the location page parser on the pinned reference pages. They skip on a
clone without the page cache."""

import pytest

from muon.ingestion.bulbapedia.locations.identity import (
    LOCATION_INFOBOXES,
    LocationIndex,
    build_location_index,
)
from muon.ingestion.bulbapedia.locations.page import (
    InfoboxKind,
    LocationClass,
    LocationPageExtract,
    Neighbour,
    parse_location_page,
)
from muon.ingestion.bulbapedia.pages import FETCH_HINT, PageStore

pytestmark = pytest.mark.pages

PAGE_TYPE = "location"
FOREIGN_PAGE_TYPES = ("species", "item", "move", "ability")


@pytest.fixture(scope="module")
def store() -> PageStore:
    return PageStore()


@pytest.fixture(scope="module")
def index(store: PageStore) -> LocationIndex:
    manifest = store.read_manifest()
    pins = manifest.list_pins(PAGE_TYPE)
    if not pins:
        pytest.skip(f"No location pins: {FETCH_HINT.format(page_type=PAGE_TYPE)}")
    foreign = [pin.title for name in FOREIGN_PAGE_TYPES for pin in manifest.list_pins(name)]
    pages = [store.read_page(PAGE_TYPE, pin) for pin in pins]
    return build_location_index(pages, foreign, LOCATION_INFOBOXES)


def parse(store: PageStore, index: LocationIndex, title: str) -> LocationPageExtract:
    pin = store.read_manifest().find_pin(PAGE_TYPE, title)
    if pin is None:
        pytest.skip(f"{title!r} is not pinned: {FETCH_HINT.format(page_type=PAGE_TYPE)}")
    result = parse_location_page(store.read_page(PAGE_TYPE, pin), index)
    assert isinstance(result, LocationPageExtract)
    return result


def test_sootopolis_city_is_a_settlement_in_hoenn(store: PageStore, index: LocationIndex) -> None:
    result = parse(store, index, "Sootopolis City")

    assert result.kind is InfoboxKind.TOWN
    assert result.leaf_class is LocationClass.SETTLEMENT
    assert result.parent_titles == ("Hoenn",)
    assert result.generation == "Generation III"
    assert Neighbour(title="Cave of Origin", direction="north") in result.neighbours
    assert Neighbour(title="Hoenn Route 126", direction="south") in result.neighbours
    assert {"Ruby-Sapphire", "Emerald", "Omega Ruby-Alpha Sapphire"} <= set(result.version_groups)


def test_kanto_is_an_abox_region(store: PageStore, index: LocationIndex) -> None:
    result = parse(store, index, "Kanto")

    assert result.is_abox_region
    assert result.leaf_class is None


def test_obsidian_fieldlands_is_a_geo_feature_in_hisui(
    store: PageStore, index: LocationIndex
) -> None:
    result = parse(store, index, "Obsidian Fieldlands")

    assert result.leaf_class is LocationClass.GEO_FEATURE
    assert "Hisui" in result.parent_titles
    assert result.version_groups == ("Legends: Arceus",)


def test_mount_coronet_has_the_short_name_as_alias(store: PageStore, index: LocationIndex) -> None:
    result = parse(store, index, "Mount Coronet")

    assert result.aliases == ("Mt. Coronet",)
    assert result.leaf_class is LocationClass.GEO_FEATURE
    assert {"Sinnoh", "Hisui"} <= set(result.parent_titles)
    assert result.generation == "Generation IV"
