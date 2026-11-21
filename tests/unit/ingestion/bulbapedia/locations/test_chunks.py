from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.batching import build_abox_index
from muon.ingestion.bulbapedia.chunking import ChunkAspect
from muon.ingestion.bulbapedia.locations.batches import (
    CLASS_IRIS,
    STRUCTURE,
    LocationBuildInputs,
    build_location_batch,
)
from muon.ingestion.bulbapedia.locations.chunks import (
    LEAD_SECTION,
    read_location_aspect,
    read_location_chunks,
)
from muon.ingestion.bulbapedia.locations.identity import (
    LOCATION_INFOBOXES,
    build_location_index,
)
from muon.ingestion.bulbapedia.locations.page import LocationPageExtract, parse_location_page
from muon.ingestion.bulbapedia.locations.run import build_group_versions
from muon.ingestion.bulbapedia.pages import FETCH_HINT, PageStore
from muon.ingestion.bulbapedia.wikitext import mask_comments, split_sections
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
LOREM = "This sentence is long enough to stand as a paragraph of its own in a chunk."


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ((), ChunkAspect.GEOGRAPHY),
        (("Description",), ChunkAspect.GEOGRAPHY),
        (("Geography", "Places of interest", "Gym"), ChunkAspect.GEOGRAPHY),
        (("In the games", "Geography"), ChunkAspect.GEOGRAPHY),
        (("In the core series games", "Layout", "Entrance"), ChunkAspect.GEOGRAPHY),
        (("Route description",), ChunkAspect.GEOGRAPHY),
        (("Trivia",), ChunkAspect.TRIVIA),
        (("Trivia", "Glitches"), ChunkAspect.TRIVIA),
        (("Origin",), ChunkAspect.ORIGIN),
        (("Trivia", "Origin"), ChunkAspect.ORIGIN),
        (("In the games",), None),
        (("In the games", "Items"), None),
        (("Pokémon",), None),
        (("In animation", "Geography"), None),
        (("Related articles",), None),
    ],
)
def test_aspect_of_a_section_path(path: tuple[str, ...], expected: ChunkAspect | None) -> None:
    assert read_location_aspect(path) == expected


def test_chunks_follow_page_order_and_name_the_lead() -> None:
    wikitext = (
        f"{{{{Town infobox|region=Hoenn}}}}\n{LOREM}\n"
        f"==In the games==\n===Geography===\n{LOREM} Geography.\n"
        f"===Items===\n{LOREM} Items.\n"
        f"==Trivia==\n{LOREM} Trivia.\n===Origin===\n{LOREM} Origin.\n"
        f"==Related articles==\n{LOREM} Related.\n"
    )

    reading = read_location_chunks(split_sections(wikitext), "sootopolis-city")

    assert [(c.aspect.value, c.section_path) for c in reading.chunks] == [
        ("geography", LEAD_SECTION),
        ("geography", "In the games > Geography"),
        ("trivia", "Trivia"),
        ("origin", "Trivia > Origin"),
    ]
    assert {c.subject_id for c in reading.chunks} == {"sootopolis-city"}
    assert reading.dropped_templates == {"town infobox": 1}


pages = pytest.mark.pages


def build_golden(title: str):
    store = PageStore()
    manifest = store.read_manifest()
    pin = manifest.find_pin("location", title)
    if pin is None:
        pytest.skip(f"{title!r} is not pinned: {FETCH_HINT.format(page_type='location')}")
    foreign = [
        p.title for n in ("species", "item", "move", "ability") for p in manifest.list_pins(n)
    ]
    all_pages = [store.read_page("location", p) for p in manifest.list_pins("location")]
    index = build_location_index(all_pages, foreign, LOCATION_INFOBOXES)
    page = store.read_page("location", pin)
    extract = parse_location_page(page, index)
    assert isinstance(extract, LocationPageExtract)
    reading = read_location_chunks(split_sections(mask_comments(page.wikitext)), "x")
    return extract, reading, index


@pages
def test_pallet_town_has_lead_and_places_of_interest_chunks_and_no_structure() -> None:
    extract, reading, index = build_golden("Pallet Town")
    paths = [c.section_path for c in reading.chunks]

    assert paths.count(LEAD_SECTION) >= 2
    assert any("Places of interest" in path for path in paths)
    seed = read_abox(load_ontology_model(ONTOLOGY_DIR, "pkmn"), ONTOLOGY_DIR)
    inputs = LocationBuildInputs(
        build_abox_index(seed),
        index,
        {"Pallet Town": CLASS_IRIS[extract.leaf_class]},
        build_group_versions(seed),
    )
    batch = build_location_batch(extract, inputs)
    assert STRUCTURE not in {node.class_iri for node in batch.nodes}
    assert "Professor Oak's Lab" not in {node.name for node in batch.nodes}


@pages
def test_kanto_page_gives_chunks() -> None:
    extract, reading, _ = build_golden("Kanto")

    assert extract.is_abox_region
    assert {c.aspect for c in reading.chunks} >= {ChunkAspect.GEOGRAPHY}
    assert len(reading.chunks) > 5
