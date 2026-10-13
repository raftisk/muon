"""Scenario 4 on pinned pages: what each prune statement would delete and keep.

A test builds a page's batch, makes an edited copy of it and reads the claim rows of the
Runner. A node or an edge that the edited batch no longer lists must be missing from the
`kept` list of its claim (and, for an owned node, from the owner ids); everything the
scenario keeps must still be there. No test writes to a database.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from muon.ingestion.batch import (
    ClaimDirection,
    EdgeClaim,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
)
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.species.batches import (
    CAN_LEARN,
    CHUNK_OWNER_CLAIMS,
    DESCRIBES,
    EVOLVES_TO,
    FROM_WORK,
    HAS_LEARNSET_ENTRY,
    HAS_TYPE,
    MENTIONED_IN,
    SLOT,
    TEXT,
    TEXT_CHUNK,
    TYPE,
    AboxIndex,
    ChunkInputs,
    SpeciesIndex,
    build_abox_index,
    build_chunk_batch,
    build_evolution_batch,
    build_owner_claims,
    build_species_batch,
    build_species_index,
    hash_chunk_text,
)
from muon.ingestion.bulbapedia.species.extract import SpeciesPageExtract
from muon.ingestion.bulbapedia.species.page import parse_species_page
from muon.ingestion.runner import build_claim_rows, build_ids_by_label
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel, load_ontology_model
from tests.unit.ingestion.bulbapedia.species.claim_summary import claims_by_anchor
from tests.unit.ingestion.bulbapedia.species.test_chunk_batches import (
    ANCHOR_PATTERNS,
    CHUNK_PATTERNS,
)

pytestmark = pytest.mark.pages

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
DIGLETT = "Diglett (Pokémon)"
CHARIZARD = "Charizard (Pokémon)"
TITLES = tuple(
    f"{name} (Pokémon)"
    for name in ("Diglett", "Dugtrio", "Charizard", "Onix", "Klefki", "Meltan", "Wiglett")
)
LEVEL_ENTRY = "diglett:earthquake:level:40:scarlet-violet"
MACHINE_ENTRY = "diglett:earthquake:machine:scarlet-violet"
DIGLETT_CHUNK_COUNT = 48
NEW_POKEDEX_TEXT = "A text that no earlier Pokédex entry has."
Row = dict[str, Any]


@pytest.fixture(scope="module")
def model() -> OntologyModel:
    return load_ontology_model(ONTOLOGY_DIR, "pkmn")


@pytest.fixture(scope="module")
def materializer(model: OntologyModel) -> LabelMaterializer:
    return LabelMaterializer(model)


@pytest.fixture(scope="module")
def extracts(read_pinned_page: Callable[[str], Page]) -> dict[str, SpeciesPageExtract]:
    parsed: dict[str, SpeciesPageExtract] = {}
    for title in TITLES:
        extract = parse_species_page(read_pinned_page(title))
        assert extract is not None, f"{title} has no Pokémon Infobox"
        parsed[title] = extract
    return parsed


@pytest.fixture(scope="module")
def abox(model: OntologyModel) -> AboxIndex:
    return build_abox_index(read_abox(model, ONTOLOGY_DIR))


@pytest.fixture(scope="module")
def species_index(extracts: dict[str, SpeciesPageExtract]) -> SpeciesIndex:
    return build_species_index(extracts.values())


@pytest.fixture(scope="module")
def diglett_species(extracts: dict[str, SpeciesPageExtract], abox: AboxIndex) -> GraphBatch:
    return build_species_batch(extracts[DIGLETT], abox)


@pytest.fixture(scope="module")
def charizard_species(extracts: dict[str, SpeciesPageExtract], abox: AboxIndex) -> GraphBatch:
    return build_species_batch(extracts[CHARIZARD], abox)


@pytest.fixture(scope="module")
def diglett_evolutions(
    extracts: dict[str, SpeciesPageExtract], abox: AboxIndex, species_index: SpeciesIndex
) -> GraphBatch:
    batch = build_evolution_batch(extracts[DIGLETT], abox, species_index).batch
    assert batch is not None
    return batch


@pytest.fixture(scope="module")
def diglett_chunks(
    extracts: dict[str, SpeciesPageExtract],
    abox: AboxIndex,
    species_index: SpeciesIndex,
    read_pinned_page: Callable[[str], Page],
) -> GraphBatch:
    inputs = ChunkInputs(abox, species_index, read_pinned_page(DIGLETT).permalink)
    batch = build_chunk_batch(extracts[DIGLETT], inputs).batch
    assert batch is not None
    return batch


def edit_batch(
    batch: GraphBatch,
    *,
    drop_nodes: Iterable[NodeRef] = (),
    drop_edges: Iterable[EdgeRecord] = (),
    add_nodes: Iterable[NodeRecord] = (),
    add_edges: Iterable[EdgeRecord] = (),
    add_claims: Iterable[EdgeClaim] = (),
) -> GraphBatch:
    """Return a copy without the dropped nodes, the edges touching them, the claims anchored
    at them and the dropped edges, plus the added nodes, edges and claims."""
    gone = set(drop_nodes)
    gone_edges = {(e.property_iri, e.start, e.end) for e in drop_edges}
    added = tuple(add_nodes)
    kept_edges = tuple(
        e
        for e in batch.edges
        if e.start not in gone
        and e.end not in gone
        and (e.property_iri, e.start, e.end) not in gone_edges
    )
    return GraphBatch(
        universe=batch.universe,
        scope=batch.scope,
        nodes=tuple(n for n in batch.nodes if n.node_ref not in gone) + added,
        edges=kept_edges + tuple(add_edges),
        claims=tuple(c for c in batch.claims if c.anchor not in gone) + tuple(add_claims),
    )


def find_node(batch: GraphBatch, node_id: str) -> NodeRecord:
    return next(node for node in batch.nodes if node.id == node_id)


def find_edges(batch: GraphBatch, property_iri: str, **ends: str) -> list[EdgeRecord]:
    return [
        e
        for e in batch.edges
        if e.property_iri == property_iri
        and e.start.id == ends.get("start", e.start.id)
        and e.end.id == ends.get("end", e.end.id)
    ]


def read_row(
    batch: GraphBatch,
    materializer: LabelMaterializer,
    direction: ClaimDirection,
    anchor_id: str,
    relationship_type: str,
) -> Row:
    rows = build_claim_rows(batch, materializer, direction)
    (row,) = [r for r in rows if r["anchor_id"] == anchor_id and r["type"] == relationship_type]
    return row


def kept_ids(row: Row) -> set[str]:
    return {kept["id"] for kept in row["kept"]}


def owner_ids(batch: GraphBatch, materializer: LabelMaterializer, label: str) -> set[str]:
    return set(build_ids_by_label(batch, materializer)[label])


def test_a_dropped_learnset_entry_leaves_its_kept_list_and_the_owner_ids(
    diglett_species: GraphBatch, materializer: LabelMaterializer
) -> None:
    entry = find_node(diglett_species, LEVEL_ENTRY)
    edited = edit_batch(diglett_species, drop_nodes=[entry.node_ref])
    args = (ClaimDirection.OUT, "diglett", "HAS_LEARNSET_ENTRY")

    before = read_row(diglett_species, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert LEVEL_ENTRY in kept_ids(before)
    assert LEVEL_ENTRY not in kept_ids(after)
    assert MACHINE_ENTRY in kept_ids(after)
    assert after["owns_other_end"] is True
    assert LEVEL_ENTRY not in owner_ids(edited, materializer, "LearnsetEntry")
    assert MACHINE_ENTRY in owner_ids(edited, materializer, "LearnsetEntry")


def test_a_dropped_can_learn_edge_leaves_the_kept_list_and_deletes_no_move(
    diglett_species: GraphBatch, materializer: LabelMaterializer
) -> None:
    (edge,) = find_edges(diglett_species, CAN_LEARN, start="diglett", end="earthquake")
    edited = edit_batch(diglett_species, drop_edges=[edge])
    args = (ClaimDirection.OUT, "diglett", "CAN_LEARN")

    before = read_row(diglett_species, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert "earthquake" in kept_ids(before)
    assert kept_ids(after) == kept_ids(before) - {"earthquake"}
    assert after["owns_other_end"] is False


def test_a_changed_type_replaces_the_kept_type(
    diglett_species: GraphBatch, materializer: LabelMaterializer
) -> None:
    (edge,) = find_edges(diglett_species, HAS_TYPE, start="diglett")
    rock = NodeRecord(class_iri=TYPE, id="rock", name="Rock")
    rock_edge = EdgeRecord(
        property_iri=HAS_TYPE, start=edge.start, end=rock.node_ref, properties={SLOT: 1}
    )
    edited = edit_batch(diglett_species, drop_edges=[edge], add_nodes=[rock], add_edges=[rock_edge])
    args = (ClaimDirection.OUT, "diglett", "HAS_TYPE")

    before = read_row(diglett_species, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert kept_ids(before) == {"ground"}
    assert kept_ids(after) == {"rock"}
    assert after["owns_other_end"] is False


def test_a_dropped_form_leaves_has_form_and_the_species_owner_ids(
    diglett_species: GraphBatch, materializer: LabelMaterializer
) -> None:
    alola = find_node(diglett_species, "diglett-alola")
    edited = edit_batch(diglett_species, drop_nodes=[alola.node_ref])
    args = (ClaimDirection.OUT, "diglett", "HAS_FORM")
    form_entries = {n.id for n in edited.owner_nodes if n.id.startswith("diglett-alola:")}

    before = read_row(diglett_species, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert kept_ids(before) == {"diglett-alola"}
    assert kept_ids(after) == set()
    assert after["owns_other_end"] is True
    assert owner_ids(edited, materializer, "PokemonSpecies") == {"diglett"}
    # Its learnset entries stay in the batch, without their species edge (spec Not included).
    assert form_entries
    assert form_entries <= owner_ids(edited, materializer, "LearnsetEntry")
    assert not [
        e
        for e in edited.edges
        if e.property_iri == HAS_LEARNSET_ENTRY and e.start.id == "diglett-alola"
    ]
    assert not [
        r
        for r in build_claim_rows(edited, materializer, ClaimDirection.OUT)
        if r["anchor_id"] == "diglett-alola"
    ]


def test_a_dropped_form_change_leaves_the_from_species_kept_list_and_the_owner_ids(
    charizard_species: GraphBatch, materializer: LabelMaterializer
) -> None:
    change = find_node(charizard_species, "charizard:charizard-mega-x")
    edited = edit_batch(charizard_species, drop_nodes=[change.node_ref])
    args = (ClaimDirection.IN, "charizard", "FROM_SPECIES")

    before = read_row(charizard_species, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert kept_ids(before) == {"charizard:charizard-mega-x", "charizard:charizard-mega-y"}
    assert kept_ids(after) == {"charizard:charizard-mega-y"}
    assert after["owns_other_end"] is True
    assert owner_ids(edited, materializer, "FormChange") == {"charizard:charizard-mega-y"}


def test_a_dropped_evolution_leaves_both_claims_and_deletes_no_species(
    diglett_evolutions: GraphBatch, materializer: LabelMaterializer
) -> None:
    evolution = find_node(diglett_evolutions, "diglett:dugtrio")
    edited = edit_batch(
        diglett_evolutions,
        drop_nodes=[evolution.node_ref],
        drop_edges=find_edges(diglett_evolutions, EVOLVES_TO, start="diglett"),
    )

    evolves_before = read_row(
        diglett_evolutions, materializer, ClaimDirection.OUT, "diglett", "EVOLVES_TO"
    )
    evolves_after = read_row(edited, materializer, ClaimDirection.OUT, "diglett", "EVOLVES_TO")
    from_before = read_row(
        diglett_evolutions, materializer, ClaimDirection.IN, "diglett", "FROM_SPECIES"
    )
    from_after = read_row(edited, materializer, ClaimDirection.IN, "diglett", "FROM_SPECIES")
    alola_after = read_row(edited, materializer, ClaimDirection.IN, "diglett-alola", "FROM_SPECIES")

    assert kept_ids(evolves_before) == {"dugtrio"}
    assert kept_ids(evolves_after) == set()
    assert evolves_after["owns_other_end"] is False
    assert kept_ids(from_before) == {"diglett:dugtrio"}
    assert kept_ids(from_after) == set()
    assert from_after["owns_other_end"] is True
    assert kept_ids(alola_after) == {"diglett-alola:dugtrio-alola"}
    assert owner_ids(edited, materializer, "Evolution") == {"diglett-alola:dugtrio-alola"}


def test_a_batch_with_no_evolution_owner_still_prunes_every_old_evolution(
    diglett_evolutions: GraphBatch, materializer: LabelMaterializer
) -> None:
    owners = [n.node_ref for n in diglett_evolutions.owner_nodes]
    edited = edit_batch(
        diglett_evolutions,
        drop_nodes=owners,
        drop_edges=find_edges(diglett_evolutions, EVOLVES_TO),
    )

    rows = build_claim_rows(edited, materializer, ClaimDirection.OUT) + build_claim_rows(
        edited, materializer, ClaimDirection.IN
    )

    assert owners and not edited.owner_nodes
    assert {(r["anchor_id"], r["type"]) for r in rows} == {
        ("diglett", "EVOLVES_TO"),
        ("diglett", "FROM_SPECIES"),
        ("diglett-alola", "EVOLVES_TO"),
        ("diglett-alola", "FROM_SPECIES"),
    }
    assert all(r["kept"] == [] for r in rows)
    assert owner_ids(edited, materializer, "Evolution") == set()


def test_a_replaced_pokedex_text_leaves_the_old_chunk_and_adds_the_new_one(
    diglett_chunks: GraphBatch, materializer: LabelMaterializer
) -> None:
    old = next(n for n in diglett_chunks.owner_nodes if ":pokedex:" in n.id)
    new_id = f"{old.id.rsplit(':', 1)[0]}:{hash_chunk_text(NEW_POKEDEX_TEXT)}"
    new = NodeRecord(
        class_iri=TEXT_CHUNK,
        id=new_id,
        name=old.name,
        properties={**old.properties, TEXT: NEW_POKEDEX_TEXT},
    )
    moved = [
        e.model_copy(update={"start": new.node_ref})
        for e in diglett_chunks.edges
        if e.start == old.node_ref
    ] + [
        e.model_copy(update={"end": new.node_ref})
        for e in diglett_chunks.edges
        if e.end == old.node_ref and e.property_iri == MENTIONED_IN
    ]
    edited = edit_batch(
        diglett_chunks,
        drop_nodes=[old.node_ref],
        add_nodes=[new],
        add_edges=moved,
        add_claims=build_owner_claims([new], CHUNK_OWNER_CLAIMS),
    )
    subject = next(
        e.end.id
        for e in diglett_chunks.edges
        if e.start == old.node_ref and e.property_iri == DESCRIBES
    )
    args = (ClaimDirection.IN, subject, "DESCRIBES")

    before = read_row(diglett_chunks, materializer, *args)
    after = read_row(edited, materializer, *args)
    new_from_work = read_row(edited, materializer, ClaimDirection.OUT, new_id, "FROM_WORK")

    assert old.id in kept_ids(before)
    assert old.id not in kept_ids(after)
    assert new_id in kept_ids(after)
    assert kept_ids(after) == (kept_ids(before) - {old.id}) | {new_id}
    assert after["owns_other_end"] is True
    assert old.id not in owner_ids(edited, materializer, "TextChunk")
    assert new_id in owner_ids(edited, materializer, "TextChunk")
    assert kept_ids(new_from_work) == {
        e.end.id
        for e in diglett_chunks.edges
        if e.start == old.node_ref and e.property_iri == FROM_WORK
    }


def test_a_dropped_mention_leaves_the_kept_list_and_deletes_no_entity(
    diglett_chunks: GraphBatch, materializer: LabelMaterializer
) -> None:
    mention = next(e for e in diglett_chunks.edges if e.property_iri == MENTIONED_IN)
    edited = edit_batch(diglett_chunks, drop_edges=[mention])
    args = (ClaimDirection.IN, mention.end.id, "MENTIONED_IN")

    before = read_row(diglett_chunks, materializer, *args)
    after = read_row(edited, materializer, *args)

    assert mention.start.id in kept_ids(before)
    assert kept_ids(after) == kept_ids(before) - {mention.start.id}
    assert after["owns_other_end"] is False
    assert after["other_label"] is None


def test_the_diglett_chunk_batch_claims_describes_at_both_forms_and_2_patterns_per_chunk(
    diglett_chunks: GraphBatch,
) -> None:
    chunk_ids = {n.id for n in diglett_chunks.owner_nodes}

    claims = claims_by_anchor(diglett_chunks)

    assert len(find_edges(diglett_chunks, DESCRIBES)) == len(chunk_ids) == DIGLETT_CHUNK_COUNT
    assert claims["diglett"] == claims["diglett-alola"] == ANCHOR_PATTERNS
    assert all(claims[chunk_id] == CHUNK_PATTERNS for chunk_id in chunk_ids)
    assert len(diglett_chunks.claims) == 2 + 2 * len(chunk_ids)
