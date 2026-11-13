"""Fixtures for the move golden tests, which read the gitignored page cache.

A test that needs a page the cache lacks skips with the fetch hint, as on a fresh clone.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.batching import UNIVERSE, build_abox_index
from muon.ingestion.bulbapedia.moves.batches import MoveChunkInputs
from muon.ingestion.bulbapedia.moves.page import MovePageExtract, parse_move_page
from muon.ingestion.bulbapedia.moves.run import SPECIES_PAGE_TYPE, build_species_resolver
from muon.ingestion.bulbapedia.pages import FETCH_HINT, Page, PageStore
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

MOVE = "move"
ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"


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


@pytest.fixture(scope="session")
def chunk_inputs() -> Callable[[Page], MoveChunkInputs]:
    """Build the pass 2 inputs of a pinned page: the real A-Box and the species pins."""
    model = load_ontology_model(ONTOLOGY_DIR, UNIVERSE)
    abox = build_abox_index(read_abox(model, ONTOLOGY_DIR))
    species_pins = PageStore().read_manifest().list_pins(SPECIES_PAGE_TYPE)
    resolve_species = build_species_resolver(species_pins)

    def build(page: Page) -> MoveChunkInputs:
        return MoveChunkInputs(
            abox=abox, resolve_species=resolve_species, source_url=page.permalink
        )

    return build
