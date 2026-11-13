"""Fixtures for the move golden tests, which read the gitignored page cache.

A test that needs a page the cache lacks skips with the fetch hint, as on a fresh clone.
"""

from collections.abc import Callable

import pytest

from muon.ingestion.bulbapedia.moves.page import MovePageExtract, parse_move_page
from muon.ingestion.bulbapedia.pages import FETCH_HINT, Page, PageStore

MOVE = "move"


@pytest.fixture(scope="session")
def read_pinned_page() -> Callable[[str], Page]:
    store = PageStore()
    manifest = store.read_manifest()

    def read(title: str) -> Page:
        pin = manifest.find_pin(MOVE, title)
        if pin is None or not store.build_page_path(pin.page_id).exists():
            hint = FETCH_HINT.format(page_type=MOVE)
            pytest.skip(f"{title!r} is not in the page cache: {hint}")
        return store.read_page(MOVE, pin)

    return read


@pytest.fixture(scope="session")
def read_extract(read_pinned_page: Callable[[str], Page]) -> Callable[[str], MovePageExtract]:
    """Parse the pinned page of a move name, e.g. "Shadow Ball", into its extract."""

    def read(name: str) -> MovePageExtract:
        extract = parse_move_page(read_pinned_page(f"{name} (move)"))
        assert isinstance(extract, MovePageExtract), f"{name} is out of scope"
        return extract

    return read
