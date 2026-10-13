"""Fixtures for the golden tests, which read the gitignored page cache.

A test that needs a page the cache lacks skips with the fetch hint, as on a fresh clone.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.pages import FETCH_HINT, Page, PageStore
from muon.ingestion.bulbapedia.species import forms
from muon.ingestion.bulbapedia.species.extract import SpeciesPageExtract
from muon.ingestion.bulbapedia.species.page import parse_species_page
from tests.unit.ingestion.bulbapedia.species.synthetic_pages import build_store, default_wikitexts

SPECIES = "species"

PILOT_TITLES = tuple(
    f"{name} (Pokémon)"
    for name in (
        "Diglett",
        "Meowth",
        "Tauros",
        "Rotom",
        "Charizard",
        "Darmanitan",
        "Eevee",
        "Tyrogue",
        "Kyurem",
        "Unown",
        "Nidoran♀",
        "Nidoran♂",
        "Abra",
        "Pikachu",
        "Groudon",
        "Rayquaza",
        "Gourgeist",
    )
)

# Species the pilot evolutions point at, so the species index can resolve them.
INDEX_EXTRA_TITLES = tuple(
    f"{name} (Pokémon)"
    for name in (
        "Dugtrio",
        "Persian",
        "Perrserker",
        "Kadabra",
        "Hitmonlee",
        "Hitmonchan",
        "Hitmontop",
        "Nidorina",
        "Nidorino",
        "Burmy",
        "Wormadam",
        "Mothim",
    )
)


@pytest.fixture(scope="session")
def read_pinned_page() -> Callable[[str], Page]:
    store = PageStore()
    manifest = store.read_manifest()

    def read(title: str) -> Page:
        pin = manifest.find_pin(SPECIES, title)
        if pin is None or not store.build_page_path(pin.page_id).exists():
            hint = FETCH_HINT.format(page_type=SPECIES)
            pytest.skip(f"{title!r} is not in the page cache: {hint}")
        return store.read_page(SPECIES, pin)

    return read


@pytest.fixture(scope="session")
def pilot_extracts(read_pinned_page: Callable[[str], Page]) -> dict[str, SpeciesPageExtract]:
    """The extract of every pilot page and index extra, by page title."""
    extracts: dict[str, SpeciesPageExtract] = {}
    for title in (*PILOT_TITLES, *INDEX_EXTRA_TITLES):
        extract = parse_species_page(read_pinned_page(title))
        assert extract is not None, f"{title} has no Pokémon Infobox"
        extracts[title] = extract
    return extracts


@pytest.fixture
def no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed overrides name real pages, which the synthetic store lacks."""
    monkeypatch.setattr(forms, "FORM_OVERRIDES", {})


@pytest.fixture
def store(tmp_path: Path, no_overrides: None) -> PageStore:
    """The synthetic store of `synthetic_pages`, with the overrides switched off."""
    return build_store(tmp_path / "data", default_wikitexts())
