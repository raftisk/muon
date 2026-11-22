"""Golden tests of the location page parser on the pinned reference pages. They skip on a
clone without the page cache."""

import pytest

from muon.ingestion.bulbapedia.locations.identity import (
    LocationIndex,
)
from muon.ingestion.bulbapedia.locations.page import (
    InfoboxKind,
    LocationClass,
    LocationPageExtract,
    Neighbour,
    parse_location_page,
)
from muon.ingestion.bulbapedia.pages import PageStore
from tests.unit.ingestion.bulbapedia.page_cache import (
    read_cached_location_index,
    read_cached_location_page,
)

pytestmark = pytest.mark.pages


@pytest.fixture(scope="module")
def store() -> PageStore:
    return PageStore()


@pytest.fixture(scope="module")
def index(store: PageStore) -> LocationIndex:
    return read_cached_location_index(store)


def parse(store: PageStore, index: LocationIndex, title: str) -> LocationPageExtract:
    result = parse_location_page(read_cached_location_page(store, title), index)
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
