"""Tests that read the gitignored page cache skip when a page file is missing, as on a fresh
clone."""

from collections.abc import Iterable

import pytest

from muon.ingestion.bulbapedia.locations.identity import LocationIndex
from muon.ingestion.bulbapedia.pages import FETCH_HINT, Page, PagePin, PageStore
from muon.ingestion.bulbapedia.species.run import LOCATION_PAGE_TYPE, read_location_index


def skip_unless_cached(store: PageStore, page_type: str, pins: Iterable[PagePin]) -> None:
    pin_list = list(pins)
    is_cached = bool(pin_list) and all(
        store.build_page_path(pin.page_id).exists() for pin in pin_list
    )
    if not is_cached:
        pytest.skip(f"page cache is incomplete: {FETCH_HINT.format(page_type=page_type)}")


def read_cached_location_index(store: PageStore) -> LocationIndex:
    skip_unless_cached(
        store, LOCATION_PAGE_TYPE, store.read_manifest().list_pins(LOCATION_PAGE_TYPE)
    )
    return read_location_index(store)


def read_cached_location_page(store: PageStore, title: str) -> Page:
    pin = store.read_manifest().find_pin(LOCATION_PAGE_TYPE, title)
    if pin is None:
        pytest.skip(f"{title!r} is not pinned: {FETCH_HINT.format(page_type=LOCATION_PAGE_TYPE)}")
    skip_unless_cached(store, LOCATION_PAGE_TYPE, [pin])
    return store.read_page(LOCATION_PAGE_TYPE, pin)
