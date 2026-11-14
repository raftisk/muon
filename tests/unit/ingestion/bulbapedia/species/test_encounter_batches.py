import pytest

from muon.ingestion.batch import NodeRef
from muon.ingestion.bulbapedia.batching import (
    FOUND_IN,
    LOCATION,
    UNIVERSE,
    AboxIndex,
    build_abox_index,
    build_encounter_id,
)
from muon.ingestion.bulbapedia.species.batches import (
    ENCOUNTER,
    ENCOUNTER_KIND,
    ENCOUNTER_SCOPE,
    HAS_ENCOUNTER,
    METHOD,
    POKEMON_SPECIES,
    VERSION,
    build_encounter_batch,
    build_encounter_name,
)
from muon.ingestion.bulbapedia.species.extract import EncounterFact, SpeciesPageExtract
from muon.ingestion.bulbapedia.species.page import parse_species_page
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model
from tests.unit.ingestion.bulbapedia.species.synthetic_pages import (
    ALPHA,
    ONTOLOGY_DIR,
    build_page,
    species_wikitext,
)

ALOLA_ID = "alpha-alola"


@pytest.fixture(scope="module")
def abox() -> AboxIndex:
    return build_abox_index(read_abox(load_ontology_model(ONTOLOGY_DIR, UNIVERSE), ONTOLOGY_DIR))


@pytest.fixture(scope="module")
def extract() -> SpeciesPageExtract:
    parsed = parse_species_page(build_page(ALPHA, species_wikitext("Alpha", 1)))
    assert parsed is not None
    base = parsed.forms[0]
    alola = base.model_copy(update={"id": ALOLA_ID, "name": "Alolan Alpha", "is_default": False})
    return parsed.model_copy(update={"forms": (base, alola)})


def fact(
    location: str = "Vermilion City",
    version: str = "HeartGold",
    method: str | None = "rock-smash",
    species_id: str = "alpha",
) -> EncounterFact:
    return EncounterFact(
        species_id=species_id, location_title=location, version=version, method=method, kind="wild"
    )


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("rock-smash", "diglett:vermilion-city:heartgold:rock-smash"),
        (None, "diglett:vermilion-city:heartgold"),
    ],
)
def test_encounter_id(method: str | None, expected: str) -> None:
    assert build_encounter_id("diglett", "vermilion-city", "heartgold", method) == expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("rock-smash", "Diglett in Vermilion City (HeartGold, rock-smash)"),
        (None, "Diglett in Vermilion City (HeartGold)"),
    ],
)
def test_encounter_name(method: str | None, expected: str) -> None:
    assert build_encounter_name("Diglett", "Vermilion City", "HeartGold", method) == expected


def test_fact_gives_an_encounter_with_its_edges(
    extract: SpeciesPageExtract, abox: AboxIndex
) -> None:
    batch = build_encounter_batch(extract, [fact()], abox)

    encounter = next(node for node in batch.nodes if node.class_iri == ENCOUNTER)
    assert encounter.id == "alpha:vermilion-city:heartgold:rock-smash"
    assert encounter.name == "Alpha in Vermilion City (HeartGold, rock-smash)"
    assert encounter.properties == {ENCOUNTER_KIND: "wild", METHOD: "rock-smash"}
    assert batch.scope == ENCOUNTER_SCOPE
    assert {(e.property_iri.rsplit("#")[1], e.start.id, e.end.id) for e in batch.edges} == {
        ("hasEncounter", "alpha", encounter.id),
        ("atLocation", encounter.id, "vermilion-city"),
        ("inVersion", encounter.id, "heartgold"),
        ("foundIn", "alpha", "vermilion-city"),
    }


def test_encounter_with_no_method_carries_only_its_kind(
    extract: SpeciesPageExtract, abox: AboxIndex
) -> None:
    batch = build_encounter_batch(extract, [fact(method=None)], abox)

    encounter = next(node for node in batch.nodes if node.class_iri == ENCOUNTER)
    assert encounter.properties == {ENCOUNTER_KIND: "wild"}


def test_repeated_facts_collapse(extract: SpeciesPageExtract, abox: AboxIndex) -> None:
    batch = build_encounter_batch(extract, [fact(), fact()], abox)

    assert sum(node.class_iri == ENCOUNTER for node in batch.nodes) == 1


def test_found_in_once_per_species_and_location(
    extract: SpeciesPageExtract, abox: AboxIndex
) -> None:
    facts = [fact(), fact(version="SoulSilver"), fact(method="surf"), fact(species_id=ALOLA_ID)]
    batch = build_encounter_batch(extract, facts, abox)

    found_in = [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == FOUND_IN]
    assert found_in == [("alpha", "vermilion-city"), (ALOLA_ID, "vermilion-city")]
    assert sum(node.class_iri == ENCOUNTER for node in batch.nodes) == 4


def test_claims_sit_at_every_form_anchor(extract: SpeciesPageExtract, abox: AboxIndex) -> None:
    batch = build_encounter_batch(extract, [fact()], abox)

    claims = {
        (c.anchor.id, c.property_iri, c.other_class_iri, c.owns_other_end) for c in batch.claims
    }
    assert claims == {
        (species_id, rule, other, owns)
        for species_id in ("alpha", ALOLA_ID)
        for rule, other, owns in ((HAS_ENCOUNTER, ENCOUNTER, True), (FOUND_IN, LOCATION, False))
    }


def test_page_with_no_fact_gives_anchors_and_claims_only(
    extract: SpeciesPageExtract, abox: AboxIndex
) -> None:
    batch = build_encounter_batch(extract, [], abox)

    assert {node.node_ref for node in batch.nodes} == {
        NodeRef(class_iri=POKEMON_SPECIES, id="alpha"),
        NodeRef(class_iri=POKEMON_SPECIES, id=ALOLA_ID),
    }
    assert batch.edges == ()
    assert len(batch.claims) == 4


def test_version_stub_carries_the_abox_id(extract: SpeciesPageExtract, abox: AboxIndex) -> None:
    batch = build_encounter_batch(extract, [fact(version="Let's Go Pikachu", method=None)], abox)

    versions = [node for node in batch.nodes if node.class_iri == VERSION]
    assert [(node.id, node.name) for node in versions] == [("lets-go-pikachu", "Let's Go Pikachu")]
    encounter = next(node for node in batch.nodes if node.class_iri == ENCOUNTER)
    assert encounter.id == "alpha:vermilion-city:lets-go-pikachu"


def test_location_stub_takes_its_id_from_the_title(
    extract: SpeciesPageExtract, abox: AboxIndex
) -> None:
    batch = build_encounter_batch(extract, [fact(location="Diglett's Cave")], abox)

    locations = [node for node in batch.nodes if node.class_iri == LOCATION]
    assert [(node.id, node.name, node.properties) for node in locations] == [
        ("digletts-cave", "Diglett's Cave", {})
    ]
