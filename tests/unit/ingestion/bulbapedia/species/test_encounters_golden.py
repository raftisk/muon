"""Encounters of the pinned golden pages against spec scenario 1 and the golden page notes.

Location verdicts need the location pins. Without them the tests skip, as on a fresh clone.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import FOUND_IN, LOCATION, UNIVERSE, build_abox_index
from muon.ingestion.bulbapedia.pages import Page, PageStore
from muon.ingestion.bulbapedia.species.batches import (
    ENCOUNTER,
    ENCOUNTER_KIND,
    HAS_ENCOUNTER,
    METHOD,
    VERSION,
    build_encounter_batch,
)
from muon.ingestion.bulbapedia.species.encounters import (
    METHOD_KINDS,
    EncounterInputs,
    EncounterReading,
    read_encounters,
)
from muon.ingestion.bulbapedia.species.page import parse_species_page
from muon.ingestion.bulbapedia.species.run import LOCATION_PAGE_TYPE, read_location_index
from muon.ingestion.runner import WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
DIGLETT = "Diglett (Pokémon)"
GASTLY = "Gastly (Pokémon)"
VULPIX = "Vulpix (Pokémon)"
MAKUHITA = "Makuhita (Pokémon)"
GOLDEN_TITLES = (DIGLETT, GASTLY, VULPIX, MAKUHITA)


@dataclass(frozen=True)
class GoldenPage:
    reading: EncounterReading
    batch: GraphBatch

    @property
    def encounters(self) -> dict[str, NodeRecord]:
        return {node.id: node for node in self.batch.owner_nodes}


@pytest.fixture(scope="module")
def golden(read_pinned_page: Callable[[str], Page]) -> dict[str, GoldenPage]:
    store = PageStore()
    if not store.read_manifest().list_pins(LOCATION_PAGE_TYPE):
        pytest.skip("no location pins: run `muon fetch bulbapedia location --from-export FILE`")
    seed = read_abox(load_ontology_model(ONTOLOGY_DIR, UNIVERSE), ONTOLOGY_DIR)
    abox = build_abox_index(seed)
    inputs = EncounterInputs(
        read_location_index(store),
        frozenset(node.name for node in seed.nodes if node.class_iri == VERSION),
    )
    pages: dict[str, GoldenPage] = {}
    for title in GOLDEN_TITLES:
        page = read_pinned_page(title)
        extract = parse_species_page(page)
        assert extract is not None
        reading = read_encounters(page, extract, inputs)
        pages[title] = GoldenPage(reading, build_encounter_batch(extract, reading.facts, abox))
    return pages


def test_diglett_generation_1_rows(golden: dict[str, GoldenPage]) -> None:
    diglett = golden[DIGLETT]

    assert {f"diglett:digletts-cave:{v}" for v in ("red", "blue", "yellow")} <= set(
        diglett.encounters
    )
    assert diglett.reading.counts.skipped_versions["Blue (Japan)"] == 1


def test_diglett_heartgold_and_soulsilver_parts(golden: dict[str, GoldenPage]) -> None:
    encounters = golden[DIGLETT].encounters

    for version in ("heartgold", "soulsilver"):
        assert f"diglett:johto-route-48:{version}" in encounters
        assert f"diglett:digletts-cave:{version}" in encounters
        assert f"diglett:johto-safari-zone:{version}" in encounters
        rock_smash = encounters[f"diglett:vermilion-city:{version}:rock-smash"]
        assert rock_smash.properties == {ENCOUNTER_KIND: "wild", METHOD: "rock-smash"}


def test_diglett_pal_park_dream_world_and_none_rows(golden: dict[str, GoldenPage]) -> None:
    diglett = golden[DIGLETT]

    for version in ("diamond", "pearl", "platinum", "heartgold", "soulsilver"):
        encounter = diglett.encounters[f"diglett:pal-park:{version}"]
        assert encounter.properties == {ENCOUNTER_KIND: "migration"}
    assert diglett.reading.counts.skipped_versions["Dream World"] == 1
    assert diglett.reading.counts.rows["none"] == 8


def test_diglett_kinds(golden: dict[str, GoldenPage]) -> None:
    encounters = golden[DIGLETT].encounters

    for encounter_id, encounter in encounters.items():
        if ":pal-park:" in encounter_id:
            continue
        expected = METHOD_KINDS.get(str(encounter.properties.get(METHOD)), "wild")
        assert encounter.properties[ENCOUNTER_KIND] == expected
    for place in ("bridge-field", "rolling-fields"):
        raid = encounters[f"diglett:{place}:sword:max-raid"]
        assert raid.properties[ENCOUNTER_KIND] == "raid"


def test_diglett_acquisitions_give_nothing(golden: dict[str, GoldenPage]) -> None:
    acquisitions = golden[DIGLETT].reading.counts.acquisitions

    assert acquisitions["breeding"] == 1
    assert acquisitions["bank"] == 2
    assert acquisitions["trade"] == 2


def test_diglett_sun_and_moon_row_gives_the_alolan_form(golden: dict[str, GoldenPage]) -> None:
    encounters = golden[DIGLETT].encounters
    places = ("alola-route-5", "alola-route-7", "digletts-tunnel", "lush-jungle")
    places += ("seaward-cave", "ten-carat-hill", "verdant-cavern")

    for place in places:
        assert f"diglett-alola:{place}:sun" in encounters
        assert f"diglett:{place}:sun" not in encounters


def test_diglett_sword_and_shield_row(golden: dict[str, GoldenPage]) -> None:
    encounters = golden[DIGLETT].encounters

    for place in ("galar-route-4", "galar-mine", "rolling-fields", "giants-mirror"):
        assert f"diglett:{place}:sword" in encounters
    assert "diglett:bridge-field:sword:max-raid" in encounters
    assert "diglett:bridge-field:sword" not in encounters


def test_diglett_area_zero_and_scarlet_violet_rows(golden: dict[str, GoldenPage]) -> None:
    diglett = golden[DIGLETT]
    places = ("south-province-area-two", "south-province-area-six", "inlet-grotto")
    places += ("alfornada-cavern", "east-province-area-three", "asado-desert")

    for version in ("scarlet", "violet"):
        assert f"diglett-alola:coastal-biome:{version}" in diglett.encounters
        assert f"diglett-alola:torchlit-labyrinth:{version}" in diglett.encounters
        for place in places:
            assert f"diglett:{place}:{version}" in diglett.encounters
    assert diglett.reading.counts.not_a_location["Type"] == 1


def test_diglett_found_in_and_claims(golden: dict[str, GoldenPage]) -> None:
    batch = golden[DIGLETT].batch
    found_in = [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == FOUND_IN]
    pairs = {(n.id.split(":")[0], n.id.split(":")[1]) for n in batch.owner_nodes}

    assert sorted(found_in) == sorted(pairs)
    assert len(found_in) == len(set(found_in))
    anchors = {(claim.anchor.id, claim.property_iri) for claim in batch.claims}
    assert anchors == {
        (species, rule)
        for species in ("diglett", "diglett-alola")
        for rule in (HAS_ENCOUNTER, FOUND_IN)
    }


def test_gastly_area_prefix_is_dropped_when_the_part_names_locations(
    golden: dict[str, GoldenPage],
) -> None:
    encounters = golden[GASTLY].encounters

    assert "gastly:scarlet-bog:legends-arceus:mass-outbreak" in encounters
    assert not any(":crimson-mirelands:" in encounter_id for encounter_id in encounters)


def test_vulpix_form_labels_on_separate_parts(golden: dict[str, GoldenPage]) -> None:
    vulpix = golden[VULPIX]

    assert "vulpix-alola:tapu-village:sun" in vulpix.encounters
    assert "vulpix:galar-route-3:sword" in vulpix.encounters
    assert not any(
        encounter_id.startswith("vulpix-alola:") and ":sword" in encounter_id
        for encounter_id in vulpix.encounters
    )
    assert vulpix.reading.counts.unmatched_form_labels["Kantonian Form"] > 0


def test_makuhita_shadow_encounters_are_snag(golden: dict[str, GoldenPage]) -> None:
    encounters = golden[MAKUHITA].encounters

    assert encounters["makuhita:phenac-city:colosseum:snag"].properties == {
        ENCOUNTER_KIND: "snag",
        METHOD: "snag",
    }
    assert "makuhita:onbs:xd:snag" in encounters


def test_every_golden_encounter_batch_compiles(golden: dict[str, GoldenPage]) -> None:
    materializer = LabelMaterializer(load_ontology_model(ONTOLOGY_DIR, UNIVERSE))

    for page in golden.values():
        assert {node.class_iri for node in page.batch.owner_nodes} == {ENCOUNTER}
        assert any(node.class_iri == LOCATION for node in page.batch.stub_nodes)
        assert build_statements(page.batch, WriteMode.UPSERT, materializer)
