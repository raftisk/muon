from pathlib import Path

import pytest

from muon.config import DEFAULT_UNIVERSES_DIR
from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    DESCRIBES,
    GENERATION,
    INTRODUCED_IN,
    ITEM,
    LOCATION,
    MENTIONED_IN,
    MOVE,
    REGION,
    TEXT_CHUNK,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.locations.areas import AreaRef
from muon.ingestion.bulbapedia.locations.batches import (
    ADJACENT_TO,
    APPEARS_IN,
    AT_LOCATION,
    DIRECTION,
    GEO_FEATURE,
    HAS_FIND,
    IN_VERSION,
    ITEM_FIND,
    LOCATED_IN,
    LOCATION_AREA,
    LOCATION_CHUNK_SCOPE,
    LOCATION_FIND_SCOPE,
    REQUIRES,
    ROLE,
    ROUTE,
    SETTLEMENT,
    STRUCTURE,
    VERSION,
    LocationBuildInputs,
    build_find_batch,
    build_find_id,
    build_location_batch,
    build_location_chunk_batch,
    list_unresolved_mentions,
)
from muon.ingestion.bulbapedia.locations.chunks import LocationChunkReading
from muon.ingestion.bulbapedia.locations.finds import FindFact
from muon.ingestion.bulbapedia.locations.identity import LocationIndex
from muon.ingestion.bulbapedia.locations.page import (
    FieldRequirement,
    InfoboxKind,
    LocationClass,
    LocationPageExtract,
    Neighbour,
)
from muon.ingestion.bulbapedia.locations.run import build_group_versions
from muon.ingestion.runner import WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import build_plugin_materializer
from muon.ontology.model import load_ontology_model
from muon.plugins import discover_universes

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
INDEX = LocationIndex(
    locations=frozenset({"Sootopolis City", "Route 126", "Cave of Origin", "Mount Coronet"}),
    non_locations=frozenset({"Badge"}),
    redirects={"Mt. Coronet": "Mount Coronet"},
    foreign_titles=frozenset({"Cyndaquil"}),
)
CLASSES = {
    "Sootopolis City": SETTLEMENT,
    "Route 126": ROUTE,
    "Cave of Origin": GEO_FEATURE,
    "Mount Coronet": GEO_FEATURE,
}


@pytest.fixture(scope="module")
def seed() -> GraphBatch:
    return read_abox(load_ontology_model(ONTOLOGY_DIR, "pkmn"), ONTOLOGY_DIR)


@pytest.fixture(scope="module")
def abox(seed: GraphBatch) -> AboxIndex:
    return build_abox_index(seed)


@pytest.fixture(scope="module")
def inputs(seed: GraphBatch, abox: AboxIndex) -> LocationBuildInputs:
    return LocationBuildInputs(abox, INDEX, CLASSES, build_group_versions(seed))


def build_extract(**values: object) -> LocationPageExtract:
    base: dict[str, object] = {
        "page_id": 1,
        "title": "Sootopolis City",
        "revision_id": 2,
        "kind": InfoboxKind.TOWN,
        "leaf_class": LocationClass.SETTLEMENT,
        "is_abox_region": False,
        "version_groups": ("Emerald",),
        "generation": "Generation III",
    }
    return LocationPageExtract.model_validate(base | values)


def list_edges(batch: GraphBatch, property_iri: str) -> list[tuple[str, str, str]]:
    return [
        (e.start.id, e.end.id, e.end.class_iri.rsplit("#", 1)[1])
        for e in batch.edges
        if e.property_iri == property_iri
    ]


def test_settlement_owns_itself_and_references_what_it_names(inputs: LocationBuildInputs) -> None:
    extract = build_extract(
        parent_titles=("Hoenn",),
        neighbours=(
            Neighbour(title="Cave of Origin", direction="north"),
            Neighbour(title="Route 126", direction="south"),
        ),
    )

    batch = build_location_batch(extract, inputs)

    assert [(n.class_iri, n.id, n.name) for n in batch.owner_nodes] == [
        (SETTLEMENT, "sootopolis-city", "Sootopolis City")
    ]
    assert batch.scope == {SETTLEMENT, LOCATION_AREA}
    assert list_edges(batch, LOCATED_IN) == [("sootopolis-city", "hoenn", "Region")]
    assert list_edges(batch, ADJACENT_TO) == [
        ("sootopolis-city", "cave-of-origin", "GeoFeature"),
        ("sootopolis-city", "route-126", "Route"),
    ]
    directions = [e.properties[DIRECTION] for e in batch.edges if e.property_iri == ADJACENT_TO]
    assert directions == ["north", "south"]
    assert {(n.class_iri, n.id) for n in batch.stub_nodes} >= {
        (ROUTE, "route-126"),
        (GEO_FEATURE, "cave-of-origin"),
        (REGION, "hoenn"),
    }


def test_versions_and_generation_come_from_the_abox(inputs: LocationBuildInputs) -> None:
    extract = build_extract(version_groups=("Red-Blue", "Yellow"))

    batch = build_location_batch(extract, inputs)

    assert [v for _, v, _ in list_edges(batch, APPEARS_IN)] == ["blue", "red", "yellow"]
    assert list_edges(batch, INTRODUCED_IN) == [("sootopolis-city", "generation-iii", "Generation")]
    assert {n.class_iri for n in batch.nodes if n.id == "red"} == {VERSION}
    assert GENERATION in {n.class_iri for n in batch.nodes}


def test_route_requires_the_field_moves_and_items(inputs: LocationBuildInputs) -> None:
    extract = build_extract(
        title="Route 126",
        kind=InfoboxKind.ROUTE,
        leaf_class=LocationClass.ROUTE,
        field_requirements=(
            FieldRequirement(role="fieldMove", target="Rock Smash"),
            FieldRequirement(role="fieldItem", target="Bicycle"),
        ),
    )

    batch = build_location_batch(extract, inputs)

    edges = [e for e in batch.edges if e.property_iri == REQUIRES]
    assert {(e.end.class_iri, e.end.id, e.properties[ROLE]) for e in edges} == {
        (MOVE, "rock-smash", "fieldMove"),
        (ITEM, "bicycle", "fieldItem"),
    }


def test_areas_are_owned_records_inside_the_location(inputs: LocationBuildInputs) -> None:
    extract = build_extract(
        title="Mount Coronet",
        leaf_class=LocationClass.GEO_FEATURE,
        kind=InfoboxKind.INFOBOX_LOCATION,
        areas=(
            AreaRef(key="2f", name="2F", location_title="Mount Coronet"),
            AreaRef(key="4f-room-1", name="4F room 1", location_title="Mount Coronet"),
        ),
    )

    batch = build_location_batch(extract, inputs)

    areas = [n for n in batch.owner_nodes if n.class_iri == LOCATION_AREA]
    assert [(a.id, a.name) for a in areas] == [
        ("mount-coronet-2f", "Mount Coronet 2F"),
        ("mount-coronet-4f-room-1", "Mount Coronet 4F room 1"),
    ]
    assert list_edges(batch, LOCATED_IN) == [
        ("mount-coronet-2f", "mount-coronet", "GeoFeature"),
        ("mount-coronet-4f-room-1", "mount-coronet", "GeoFeature"),
    ]


def test_owner_claims_cover_the_edges_the_page_writes(inputs: LocationBuildInputs) -> None:
    batch = build_location_batch(build_extract(), inputs)

    claims = {(c.property_iri.rsplit("#", 1)[1], c.direction.value) for c in batch.claims}

    assert claims == {
        ("locatedIn", "out"),
        ("adjacentTo", "out"),
        ("appearsIn", "out"),
        ("introducedIn", "out"),
        ("requires", "out"),
        ("locatedIn", "in"),
    }
    assert {c.anchor.id for c in batch.claims} == {"sootopolis-city"}
    area_claim = next(c for c in batch.claims if c.direction.value == "in")
    assert area_claim.owns_other_end
    assert area_claim.other_class_iri == LOCATION_AREA


def test_abox_region_page_gives_a_reference_and_no_claims(inputs: LocationBuildInputs) -> None:
    extract = build_extract(
        title="Kanto", kind=InfoboxKind.REGION, leaf_class=None, is_abox_region=True
    )

    batch = build_location_batch(extract, inputs)

    assert batch.scope == frozenset()
    assert [(n.class_iri, n.id) for n in batch.nodes] == [(REGION, "kanto")]
    assert batch.edges == ()
    assert batch.claims == ()


def test_references_resolve_redirects_and_drop_pages_that_are_not_places(
    inputs: LocationBuildInputs,
) -> None:
    extract = build_extract(
        parent_titles=("Mt. Coronet", "Badge", "Cyndaquil", "Unknown Cove"),
    )

    batch = build_location_batch(extract, inputs)

    assert list_edges(batch, LOCATED_IN) == [
        ("sootopolis-city", "mount-coronet", "GeoFeature"),
        ("sootopolis-city", "unknown-cove", "Location"),
    ]
    assert LOCATION in {n.class_iri for n in batch.stub_nodes}


def test_a_target_named_in_2_directions_gives_1_edge_with_the_first(
    inputs: LocationBuildInputs,
) -> None:
    extract = build_extract(
        neighbours=(
            Neighbour(title="Route 126", direction="south"),
            Neighbour(title="Route 126", direction="west"),
        )
    )

    batch = build_location_batch(extract, inputs)

    edges = [e for e in batch.edges if e.property_iri == ADJACENT_TO]
    assert [e.properties[DIRECTION] for e in edges] == ["south"]


def test_aliases_are_a_property_only_when_the_page_has_some(inputs: LocationBuildInputs) -> None:
    plain = build_location_batch(build_extract(), inputs).owner_nodes[0]
    aliased = build_location_batch(build_extract(aliases=("Rune City",)), inputs).owner_nodes[0]

    assert ALIASES not in plain.properties
    assert aliased.properties[ALIASES] == ("Rune City",)


def test_a_name_the_abox_lacks_fails_the_page(inputs: LocationBuildInputs) -> None:
    with pytest.raises(BatchBuildError, match="Generation XIV"):
        build_location_batch(build_extract(generation="Generation XIV"), inputs)
    with pytest.raises(BatchBuildError, match="Nowhere"):
        build_location_batch(build_extract(version_groups=("Nowhere",)), inputs)


def test_batches_compile_into_upsert_statements_that_stub_other_pages(
    inputs: LocationBuildInputs,
) -> None:
    model = load_ontology_model(ONTOLOGY_DIR, "pkmn")
    manifest = discover_universes(DEFAULT_UNIVERSES_DIR, ONTOLOGY_DIR).get("pkmn").manifest
    materializer = build_plugin_materializer(model, manifest)
    extract = build_extract(
        parent_titles=("Hoenn",), neighbours=(Neighbour(title="Route 126", direction="south"),)
    )

    statements = build_statements(
        build_location_batch(extract, inputs), WriteMode.UPSERT, materializer
    )

    owners = [row for s in statements for row in s.parameters.get("nodes", [])]
    stubs = [row for s in statements for row in s.parameters.get("stubs", [])]
    assert [row["id"] for row in owners] == ["sootopolis-city"]
    assert {row["id"] for row in stubs} >= {"route-126", "hoenn"}


def test_structure_class_has_its_own_iri() -> None:
    assert STRUCTURE.endswith("#Structure")


def build_reading(*facts: ChunkFact) -> LocationChunkReading:
    return LocationChunkReading(
        chunks=facts, dropped_templates={}, removed_tables=0, math_formulas=0, long_chunks=()
    )


def build_geography(text: str, *mentions: MentionRef) -> ChunkFact:
    return ChunkFact(
        subject_id="sootopolis-city",
        aspect=ChunkAspect.GEOGRAPHY,
        section_path="Lead",
        text=text,
        mentions=mentions,
    )


def test_chunks_describe_the_location_of_the_page(inputs: LocationBuildInputs) -> None:
    reading = build_reading(build_geography("A city."), build_geography("A crater."))

    batch = build_location_chunk_batch(build_extract(), reading, inputs)

    assert batch.scope == LOCATION_CHUNK_SCOPE
    assert [n.id for n in batch.owner_nodes] == [
        "bulbapedia:settlement:sootopolis-city:geography:1",
        "bulbapedia:settlement:sootopolis-city:geography:2",
    ]
    assert {(e.end.class_iri, e.end.id) for e in batch.edges if e.property_iri == DESCRIBES} == {
        (SETTLEMENT, "sootopolis-city")
    }
    assert [(n.class_iri, n.id) for n in batch.stub_nodes] == [(SETTLEMENT, "sootopolis-city")]


def test_chunk_claims_anchor_at_the_location_and_own_its_chunks(
    inputs: LocationBuildInputs,
) -> None:
    batch = build_location_chunk_batch(build_extract(), build_reading(), inputs)

    describes = [c for c in batch.claims if c.property_iri == DESCRIBES]

    assert [(c.anchor.id, c.direction.value, c.owns_other_end) for c in describes] == [
        ("sootopolis-city", "in", True)
    ]
    assert batch.owner_nodes == ()


def test_a_chunk_of_a_region_page_describes_the_region(inputs: LocationBuildInputs) -> None:
    extract = build_extract(
        title="Kanto", kind=InfoboxKind.REGION, leaf_class=None, is_abox_region=True
    )

    batch = build_location_chunk_batch(extract, build_reading(build_geography("A region.")), inputs)

    assert [n.id for n in batch.owner_nodes] == ["bulbapedia:region:kanto:geography:1"]
    assert [(n.class_iri, n.id) for n in batch.stub_nodes] == [(REGION, "kanto")]


def test_a_resolved_mention_links_in_and_the_rest_are_listed(inputs: LocationBuildInputs) -> None:
    fact = build_geography(
        "Fire and nothing.",
        MentionRef(template="t", target="Fire"),
        MentionRef(template="p", target="Ghost"),
    )

    batch = build_location_chunk_batch(build_extract(), build_reading(fact), inputs)

    assert [
        (e.start.id, e.end.class_iri) for e in batch.edges if e.property_iri == MENTIONED_IN
    ] == [("fire", TEXT_CHUNK)]
    assert list_unresolved_mentions(build_reading(fact), inputs) == (
        MentionRef(template="p", target="Ghost"),
    )


FOUND_IN = "https://muon.dev/ns/core#foundIn"


def build_fact(item: str = "Moon Stone", version: str = "Gold", **values: object) -> FindFact:
    base: dict[str, object] = {
        "item_name": item,
        "area": None,
        "version": version,
        "method": None,
        "is_hidden": False,
    }
    return FindFact(**(base | values))  # type: ignore[arg-type]


def test_find_id_joins_place_version_item_method_and_hidden() -> None:
    assert build_find_id("cave", "gold", "moon-stone", None, False) == "cave:gold:moon-stone"
    assert (
        build_find_id("cave-1f", "red", "potion", "rock-smash", True)
        == "cave-1f:red:potion:rock-smash:hidden"
    )


def test_a_find_is_at_the_location_in_a_version_and_belongs_to_its_item(
    inputs: LocationBuildInputs,
) -> None:
    owners = {"moon stone": ("moon-stone", "Moon Stone")}
    with_items = LocationBuildInputs(
        inputs.abox, inputs.index, inputs.classes, inputs.group_versions, item_owners=owners
    )

    build = build_find_batch(build_extract(), [build_fact(is_hidden=True)], with_items)

    batch = build.batch
    assert batch.scope == LOCATION_FIND_SCOPE
    find = batch.owner_nodes[0]
    assert (find.class_iri, find.id) == (ITEM_FIND, "sootopolis-city:gold:moon-stone:hidden")
    assert find.properties == {"https://muon.dev/ns/pkmn#isHidden": True}
    assert find.name == "Moon Stone at Sootopolis City (Gold, hidden)"
    assert list_edges(batch, AT_LOCATION) == [(find.id, "sootopolis-city", "Settlement")]
    assert list_edges(batch, IN_VERSION) == [(find.id, "gold", "Version")]
    assert [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == HAS_FIND] == [
        ("moon-stone", find.id)
    ]
    assert [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == FOUND_IN] == [
        ("moon-stone", "sootopolis-city")
    ]
    assert build.item_stub_names == ()


def test_a_find_in_an_area_is_at_the_area_and_the_shortcut_goes_to_the_page(
    inputs: LocationBuildInputs,
) -> None:
    area = AreaRef(key="b1f", name="B1F", location_title="Sootopolis City")

    build = build_find_batch(build_extract(), [build_fact(area=area, method="rock-smash")], inputs)

    find = build.batch.owner_nodes[0]
    assert find.id == "sootopolis-city-b1f:gold:moon-stone:rock-smash"
    assert find.properties == {"https://muon.dev/ns/pkmn#method": "rock-smash"}
    assert list_edges(build.batch, AT_LOCATION) == [
        (find.id, "sootopolis-city-b1f", "LocationArea")
    ]
    assert [e.end.id for e in build.batch.edges if e.property_iri == FOUND_IN] == [
        "sootopolis-city"
    ]
    anchors = {(c.anchor.id, c.property_iri.rsplit("#", 1)[1]) for c in build.batch.claims}
    assert anchors == {
        ("sootopolis-city", "atLocation"),
        ("sootopolis-city-b1f", "atLocation"),
        ("sootopolis-city", "foundIn"),
    }
    assert all(c.owns_other_end for c in build.batch.claims if c.property_iri == AT_LOCATION)
    assert not any(c.owns_other_end for c in build.batch.claims if c.property_iri == FOUND_IN)


def test_facts_with_the_same_id_give_1_find_and_an_unowned_item_is_listed(
    inputs: LocationBuildInputs,
) -> None:
    facts = [build_fact(), build_fact(), build_fact(item="Potion")]

    build = build_find_batch(build_extract(), facts, inputs)

    assert len(build.batch.owner_nodes) == 2
    assert build.duplicates == 1
    assert build.item_stub_names == ("Moon Stone", "Potion")


def test_a_page_with_no_finds_still_claims_its_location(inputs: LocationBuildInputs) -> None:
    build = build_find_batch(build_extract(), [], inputs)

    assert build.batch.owner_nodes == ()
    assert {c.anchor.id for c in build.batch.claims} == {"sootopolis-city"}
