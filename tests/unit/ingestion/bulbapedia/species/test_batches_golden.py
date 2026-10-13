"""Batches of the pinned pilot pages against the spec scenarios and the Diglett target."""

from collections import Counter
from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.species.batches import (
    CAN_LEARN,
    EVOLUTION,
    EVOLUTION_SCOPE,
    EVOLVES_TO,
    FORM_CHANGE,
    HELD_ITEM,
    INTRODUCED_IN,
    LEARNSET_ENTRY,
    MACHINE,
    PKMN,
    POKEMON_SPECIES,
    REQUIRES,
    SPECIES_SCOPE,
    VIA_MACHINE,
    AboxIndex,
    EvolutionBuild,
    SpeciesIndex,
    build_abox_index,
    build_evolution_batch,
    build_species_batch,
    build_species_index,
)
from muon.ingestion.bulbapedia.species.extract import NoteKind, SpeciesPageExtract
from muon.ingestion.runner import WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model
from tests.unit.ingestion.bulbapedia.species.claim_summary import (
    ENTRY_CLAIM_PATTERNS,
    EVOLUTION_ANCHOR_PATTERNS,
    REQUIRES_CLAIM_PATTERNS,
    SPECIES_CLAIM_PATTERNS,
    claims_by_anchor,
)
from tests.unit.ingestion.test_species_batch import build_diglett_batch

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
Extracts = dict[str, SpeciesPageExtract]


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def abox(model: OntologyModel) -> AboxIndex:
    return build_abox_index(read_abox(model, ONTOLOGY_DIR))


@pytest.fixture(scope="module")
def species_index(pilot_extracts: Extracts) -> SpeciesIndex:
    return build_species_index(pilot_extracts.values())


@pytest.fixture(scope="module")
def species_batches(pilot_extracts: Extracts, abox: AboxIndex) -> dict[str, GraphBatch]:
    return {title: build_species_batch(extract, abox) for title, extract in pilot_extracts.items()}


@pytest.fixture(scope="module")
def evolution_builds(
    pilot_extracts: Extracts, abox: AboxIndex, species_index: SpeciesIndex
) -> dict[str, EvolutionBuild]:
    return {
        title: build_evolution_batch(extract, abox, species_index)
        for title, extract in pilot_extracts.items()
    }


def title_of(name: str) -> str:
    return f"{name} (Pokémon)"


def entries_by_method(batch: GraphBatch, species_id: str) -> Counter[str]:
    return Counter(
        str(node.properties[PKMN + "method"])
        for node in batch.owner_nodes
        if node.class_iri == LEARNSET_ENTRY and node.id.startswith(f"{species_id}:")
    )


def edge_keys(batch: GraphBatch, property_iri: str) -> set[tuple[str, str]]:
    return {(e.start.id, e.end.id) for e in batch.edges if e.property_iri == property_iri}


def test_diglett_species_batch_contains_the_hand_written_target(
    species_batches: dict[str, GraphBatch],
) -> None:
    parsed = species_batches[title_of("Diglett")]
    target = build_diglett_batch()
    parsed_nodes = {node.node_ref: node for node in parsed.nodes}
    parsed_edges = {(e.property_iri, e.start, e.end): e for e in parsed.edges}

    assert parsed.scope == target.scope == SPECIES_SCOPE
    assert parsed.universe == target.universe
    for expected in target.nodes:
        actual = parsed_nodes[expected.node_ref]
        assert actual.name == expected.name, expected.id
        assert {k: actual.properties[k] for k in expected.properties} == dict(
            expected.properties
        ), expected.id
        assert (actual.class_iri in parsed.scope) == (expected.class_iri in target.scope)
    for expected_edge in target.edges:
        actual_edge = parsed_edges[
            (expected_edge.property_iri, expected_edge.start, expected_edge.end)
        ]
        assert dict(actual_edge.properties) == dict(expected_edge.properties), expected_edge


def test_diglett_generations(species_batches: dict[str, GraphBatch]) -> None:
    batch = species_batches[title_of("Diglett")]

    assert edge_keys(batch, INTRODUCED_IN) == {
        ("diglett", "generation-i"),
        ("diglett-alola", "generation-vii"),
    }


def test_diglett_learnset_entries(species_batches: dict[str, GraphBatch]) -> None:
    batch = species_batches[title_of("Diglett")]
    nodes = {n.id: n for n in batch.nodes}

    assert dict(entries_by_method(batch, "diglett")) == {
        "level": 13,
        "machine": 40,
        "egg": 5,
        "special": 1,
    }
    assert dict(entries_by_method(batch, "diglett-alola")) == {
        "level": 13,
        "machine": 42,
        "egg": 7,
    }
    assert "diglett:earthquake:level:40:scarlet-violet" in nodes
    assert "diglett:earthquake:machine:scarlet-violet" in nodes
    machine_entry = "diglett:earthquake:machine:scarlet-violet"
    assert (machine_entry, "tm149-scarlet-violet") in edge_keys(batch, VIA_MACHINE)
    assert nodes["tm149-scarlet-violet"].class_iri == MACHINE


def test_diglett_can_learn_has_one_edge_per_distinct_move(
    species_batches: dict[str, GraphBatch],
) -> None:
    batch = species_batches[title_of("Diglett")]
    entry_moves: dict[str, set[str]] = {"diglett": set(), "diglett-alola": set()}
    for edge in batch.edges:
        if edge.property_iri == PKMN + "teachesMove":
            species_id = edge.start.id.split(":")[0]
            entry_moves[species_id].add(edge.end.id)

    can_learn = edge_keys(batch, CAN_LEARN)

    for species_id, moves in entry_moves.items():
        assert {end for start, end in can_learn if start == species_id} == moves
    assert len(can_learn) == sum(len(moves) for moves in entry_moves.values())


def test_diglett_evolution_batch(evolution_builds: dict[str, EvolutionBuild]) -> None:
    build = evolution_builds[title_of("Diglett")]

    assert build.batch is not None
    assert build.batch.scope == EVOLUTION_SCOPE
    owners = {node.id: node for node in build.batch.owner_nodes}
    assert set(owners) == {"diglett:dugtrio", "diglett-alola:dugtrio-alola"}
    for node in owners.values():
        assert dict(node.properties) == {PKMN + "evoTrigger": "level", PKMN + "minLevel": 26}
    assert edge_keys(build.batch, EVOLVES_TO) == {
        ("diglett", "dugtrio"),
        ("diglett-alola", "dugtrio-alola"),
    }
    assert edge_keys(build.batch, PKMN + "fromSpecies") == {
        ("diglett:dugtrio", "diglett"),
        ("diglett-alola:dugtrio-alola", "diglett-alola"),
    }
    assert {n.class_iri for n in build.batch.stub_nodes} == {POKEMON_SPECIES}
    assert (build.structured, build.total, build.notes) == (2, 2, ())


def test_mega_form_changes_sit_in_the_species_batch_with_their_held_items(
    species_batches: dict[str, GraphBatch],
) -> None:
    batch = species_batches[title_of("Charizard")]
    changes = {n.id: n for n in batch.owner_nodes if n.class_iri == FORM_CHANGE}

    assert set(changes) == {"charizard:charizard-mega-x", "charizard:charizard-mega-y"}
    for node in changes.values():
        assert dict(node.properties) == {PKMN + "formKind": "mega", PKMN + "isBattleOnly": True}
    assert edge_keys(batch, REQUIRES) == {
        ("charizard:charizard-mega-x", "charizardite-x"),
        ("charizard:charizard-mega-y", "charizardite-y"),
    }
    stubs = {(n.class_iri, n.id) for n in batch.stub_nodes}
    assert (HELD_ITEM, "charizardite-x") in stubs
    assert (HELD_ITEM, "charizardite-y") in stubs
    assert edge_keys(batch, PKMN + "hasForm") == {
        ("charizard", "charizard-mega-x"),
        ("charizard", "charizard-mega-y"),
    }


def test_primal_form_change(species_batches: dict[str, GraphBatch]) -> None:
    batch = species_batches[title_of("Groudon")]
    change = next(n for n in batch.owner_nodes if n.class_iri == FORM_CHANGE)

    assert change.id == "groudon:groudon-primal"
    assert dict(change.properties) == {
        PKMN + "formKind": "primal",
        PKMN + "isBattleOnly": True,
        PKMN + "trigger": "switch_in",
    }
    assert edge_keys(batch, REQUIRES) == {("groudon:groudon-primal", "red-orb")}


def test_tyrogue_evolutions_carry_only_their_condition(
    evolution_builds: dict[str, EvolutionBuild],
) -> None:
    build = evolution_builds[title_of("Tyrogue")]

    assert build.batch is not None
    owners = {node.id: node for node in build.batch.owner_nodes}
    assert set(owners) == {"tyrogue:hitmonlee", "tyrogue:hitmonchan", "tyrogue:hitmontop"}
    assert {dict(n.properties)[PKMN + "condition"] for n in owners.values()} == {
        "Level 20 (Attack > Defense)",
        "Level 20 (Attack < Defense)",
        "Level 20 (Attack = Defense)",
    }
    assert all(set(n.properties) == {PKMN + "condition"} for n in owners.values())
    assert len(edge_keys(build.batch, EVOLVES_TO)) == 3
    assert (build.structured, build.total) == (0, 3)


def test_abra_has_no_entries_and_evolves_at_level_16(
    species_batches: dict[str, GraphBatch], evolution_builds: dict[str, EvolutionBuild]
) -> None:
    species = species_batches[title_of("Abra")]
    build = evolution_builds[title_of("Abra")]

    assert not [n for n in species.nodes if n.class_iri == LEARNSET_ENTRY]
    assert not edge_keys(species, CAN_LEARN)
    assert build.batch is not None
    kadabra = next(n for n in build.batch.owner_nodes if n.id == "abra:kadabra")
    assert kadabra.properties[PKMN + "minLevel"] == 16


def test_a_cosmetic_form_reference_falls_back_to_the_base_species_and_is_noted(
    evolution_builds: dict[str, EvolutionBuild],
) -> None:
    build = evolution_builds[title_of("Burmy")]

    assert build.batch is not None
    assert "burmy:mothim" in {n.id for n in build.batch.owner_nodes}
    fallbacks = [n for n in build.notes if n.kind is NoteKind.REFERENCE_FALLBACK]
    assert any("Sandy" in n.detail and "burmy" in n.detail for n in fallbacks)


def test_every_pilot_batch_compiles_in_upsert_mode(
    model: OntologyModel,
    species_batches: dict[str, GraphBatch],
    evolution_builds: dict[str, EvolutionBuild],
) -> None:
    materializer = LabelMaterializer(model)
    batches: list[GraphBatch] = list(species_batches.values())
    batches.extend(b.batch for b in evolution_builds.values() if b.batch is not None)

    for batch in batches:
        assert build_statements(batch, WriteMode.UPSERT, materializer)

    assert len(batches) > len(species_batches)


def test_every_species_stub_of_the_pilot_evolutions_has_an_owner_in_the_run(
    species_batches: dict[str, GraphBatch], evolution_builds: dict[str, EvolutionBuild]
) -> None:
    owners = {
        node.node_ref
        for batch in species_batches.values()
        for node in batch.owner_nodes
        if node.class_iri == POKEMON_SPECIES
    }
    stubs: list[NodeRecord] = [
        stub
        for build in evolution_builds.values()
        if build.batch is not None
        for stub in build.batch.stub_nodes
        if stub.class_iri == POKEMON_SPECIES
    ]

    assert stubs
    assert {stub.node_ref for stub in stubs} <= owners
    assert EVOLUTION in {
        node.class_iri
        for build in evolution_builds.values()
        if build.batch is not None
        for node in build.batch.owner_nodes
    }


def test_diglett_species_batch_claims_the_9_patterns_per_species_and_viamachine_per_entry(
    species_batches: dict[str, GraphBatch],
) -> None:
    batch = species_batches[title_of("Diglett")]
    entry_ids = {n.id for n in batch.owner_nodes if n.class_iri == LEARNSET_ENTRY}

    claims = claims_by_anchor(batch)

    assert claims["diglett"] == claims["diglett-alola"] == SPECIES_CLAIM_PATTERNS
    assert entry_ids
    assert all(claims[entry_id] == ENTRY_CLAIM_PATTERNS for entry_id in entry_ids)
    assert len(batch.claims) == 2 * len(SPECIES_CLAIM_PATTERNS) + len(entry_ids)


def test_diglett_evolution_batch_claims_2_patterns_per_form_and_requires_per_evolution(
    evolution_builds: dict[str, EvolutionBuild],
) -> None:
    batch = evolution_builds[title_of("Diglett")].batch

    assert batch is not None
    assert claims_by_anchor(batch) == {
        "diglett": EVOLUTION_ANCHOR_PATTERNS,
        "diglett-alola": EVOLUTION_ANCHOR_PATTERNS,
        "diglett:dugtrio": REQUIRES_CLAIM_PATTERNS,
        "diglett-alola:dugtrio-alola": REQUIRES_CLAIM_PATTERNS,
    }


def test_charizard_species_batch_claims_requires_at_each_mega_form_change(
    species_batches: dict[str, GraphBatch],
) -> None:
    claims = claims_by_anchor(species_batches[title_of("Charizard")])

    assert claims["charizard:charizard-mega-x"] == REQUIRES_CLAIM_PATTERNS
    assert claims["charizard:charizard-mega-y"] == REQUIRES_CLAIM_PATTERNS


def test_every_pilot_claim_anchors_at_a_node_of_its_batch(
    species_batches: dict[str, GraphBatch], evolution_builds: dict[str, EvolutionBuild]
) -> None:
    batches: list[GraphBatch] = list(species_batches.values())
    batches.extend(b.batch for b in evolution_builds.values() if b.batch is not None)

    for batch in batches:
        refs = {node.node_ref for node in batch.nodes}
        assert batch.claims
        assert {claim.anchor for claim in batch.claims} <= refs
