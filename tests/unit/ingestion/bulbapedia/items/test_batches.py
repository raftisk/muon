from pathlib import Path

import pytest

from muon.config import DEFAULT_UNIVERSES_DIR
from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    BERRY,
    DESCRIBES,
    FLAVOR,
    FROM_WORK,
    GENERATION,
    HAS_FLAVOR,
    HELD_ITEM,
    INTRODUCED_IN,
    ITEM,
    MENTIONED_IN,
    SOURCE,
    SOURCE_URL,
    TEXT_CHUNK,
    AboxIndex,
    BatchBuildError,
    build_abox_index,
    hash_chunk_text,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.items.batches import (
    BUY_PRICE,
    CATEGORY,
    FLING_POWER,
    ITEM_CHUNK_SCOPE,
    ITEM_SCOPE,
    SELL_PRICE,
    ItemChunkInputs,
    build_item_batch,
    build_item_chunk_batch,
)
from muon.ingestion.bulbapedia.items.chunks import ItemChunkReading
from muon.ingestion.bulbapedia.items.page import ItemClass, ItemPageExtract
from muon.ingestion.bulbapedia.pages import build_permalink
from muon.ingestion.runner import WriteMode, build_statements
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import build_plugin_materializer
from muon.ontology.model import load_ontology_model
from muon.plugins import discover_universes

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
FLAVOR_TEXT = "Heals the holder."


@pytest.fixture(scope="module")
def seed() -> GraphBatch:
    return read_abox(load_ontology_model(ONTOLOGY_DIR, "pkmn"), ONTOLOGY_DIR)


@pytest.fixture(scope="module")
def abox(seed: GraphBatch) -> AboxIndex:
    return build_abox_index(seed)


def build_extract(**values: object) -> ItemPageExtract:
    base: dict[str, object] = {
        "title": "Oran Berry",
        "page_id": 1,
        "revision_id": 2,
        "item_id": "oran-berry",
        "name": "Oran Berry",
        "item_class": ItemClass.BERRY,
        "generation_label": "Generation III",
        "category": "Berries",
        "sell_price": 20,
        "fling_power": 10,
        "flavor_potencies": {"Spicy": 10, "Bitter": 10},
        "aliases": ("オレンのみ",),
    }
    return ItemPageExtract.model_validate({**base, **values})


def build_reading(*facts: ChunkFact) -> ItemChunkReading:
    return ItemChunkReading(
        chunks=facts,
        dropped_templates={},
        removed_tables=0,
        math_formulas=0,
        unknown_codes=(),
        long_chunks=(),
    )


def build_fact(aspect: ChunkAspect, text: str, *mentions: MentionRef) -> ChunkFact:
    is_flavor = aspect is ChunkAspect.FLAVOR
    return ChunkFact(
        subject_id="oran-berry",
        aspect=aspect,
        section_path="Description" if is_flavor else "Effect",
        text=text,
        mentions=mentions,
        versions=("Ruby-Sapphire", "Scarlet-Violet") if is_flavor else (),
    )


def resolve_species(target: str) -> NodeRecord | None:
    return None


def build_chunks(abox: AboxIndex, *facts: ChunkFact, **values: object) -> GraphBatch:
    inputs = ItemChunkInputs(abox=abox, resolver=resolve_species)
    return build_item_chunk_batch(build_extract(**values), build_reading(*facts), inputs)


def get_node(batch: GraphBatch, class_iri: str, node_id: str) -> NodeRecord:
    return next(n for n in batch.nodes if n.class_iri == class_iri and n.id == node_id)


def test_owner_record_carries_its_values(abox: AboxIndex) -> None:
    batch = build_item_batch(build_extract(), abox)

    owner = get_node(batch, BERRY, "oran-berry")
    assert batch.scope == ITEM_SCOPE
    assert owner.name == "Oran Berry"
    assert owner.properties == {
        CATEGORY: "Berries",
        SELL_PRICE: 20,
        FLING_POWER: 10,
        ALIASES: ("オレンのみ",),
        SOURCE_URL: build_permalink("Oran Berry", 2),
    }


def test_values_without_a_value_are_absent(abox: AboxIndex) -> None:
    extract = build_extract(
        category=None, sell_price=None, fling_power=None, aliases=(), flavor_potencies={}
    )

    owner = get_node(build_item_batch(extract, abox), BERRY, "oran-berry")

    assert set(owner.properties) == {SOURCE_URL}


@pytest.mark.parametrize(
    ("item_class", "class_iri"),
    [(ItemClass.BERRY, BERRY), (ItemClass.HELD_ITEM, HELD_ITEM), (ItemClass.ITEM, ITEM)],
)
def test_owner_has_the_class_of_the_extract(
    abox: AboxIndex, item_class: ItemClass, class_iri: str
) -> None:
    extract = build_extract(item_class=item_class, flavor_potencies={})

    batch = build_item_batch(extract, abox)

    assert get_node(batch, class_iri, "oran-berry")


@pytest.mark.parametrize("item_class", [ItemClass.BERRY, ItemClass.HELD_ITEM, ItemClass.ITEM])
def test_owner_carries_the_page_permalink(abox: AboxIndex, item_class: ItemClass) -> None:
    extract = build_extract(item_class=item_class, flavor_potencies={})

    batch = build_item_batch(extract, abox)

    (owner,) = batch.owner_nodes
    assert owner.properties[SOURCE_URL] == build_permalink("Oran Berry", 2)


def test_buy_price_is_written_when_present(abox: AboxIndex) -> None:
    extract = build_extract(buy_price=10000, flavor_potencies={})

    owner = get_node(build_item_batch(extract, abox), BERRY, "oran-berry")

    assert owner.properties[BUY_PRICE] == 10000


def test_generation_and_flavor_edges(abox: AboxIndex) -> None:
    batch = build_item_batch(build_extract(), abox)

    introduced = [e for e in batch.edges if e.property_iri == INTRODUCED_IN]
    flavors = {e.end.id: e.properties for e in batch.edges if e.property_iri == HAS_FLAVOR}
    assert [(e.end.class_iri, e.end.id) for e in introduced] == [(GENERATION, "generation-iii")]
    assert flavors == {"spicy": {"potency": 10}, "bitter": {"potency": 10}}
    assert get_node(batch, FLAVOR, "spicy").name == "Spicy"


def test_claims_cover_the_generation_and_flavor_edges(abox: AboxIndex) -> None:
    batch = build_item_batch(build_extract(), abox)

    assert sorted((claim.property_iri, claim.owns_other_end) for claim in batch.claims) == [
        (HAS_FLAVOR, False),
        (INTRODUCED_IN, False),
    ]


def test_unknown_generation_fails_the_page(abox: AboxIndex) -> None:
    with pytest.raises(BatchBuildError, match="Generation XI"):
        build_item_batch(build_extract(generation_label="Generation XI"), abox)


def test_unknown_flavor_fails_the_page(abox: AboxIndex) -> None:
    with pytest.raises(BatchBuildError, match="Salty"):
        build_item_batch(build_extract(flavor_potencies={"Salty": 5}), abox)


def test_chunk_batch_has_an_item_stub_and_a_describes_claim(abox: AboxIndex) -> None:
    batch = build_chunks(abox, build_fact(ChunkAspect.EFFECT, "Restores HP."))

    chunk = get_node(batch, TEXT_CHUNK, "bulbapedia:item:oran-berry:effect:1")
    assert batch.scope == ITEM_CHUNK_SCOPE
    assert chunk.properties[SOURCE] == "bulbapedia"
    assert get_node(batch, ITEM, "oran-berry").name == "Oran Berry"
    describes = [c for c in batch.claims if c.property_iri == DESCRIBES]
    assert [(c.anchor.class_iri, c.owns_other_end) for c in describes] == [(ITEM, True)]


def test_chunk_batch_with_no_chunk_still_claims_describes(abox: AboxIndex) -> None:
    batch = build_chunks(abox)

    assert [claim.property_iri for claim in batch.claims] == [DESCRIBES]


def test_flavor_chunk_has_a_hash_id_no_source_and_from_work_edges(abox: AboxIndex) -> None:
    batch = build_chunks(abox, build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT))

    chunk_id = f"bulbapedia:item:oran-berry:flavor:{hash_chunk_text(FLAVOR_TEXT)}"
    chunk = get_node(batch, TEXT_CHUNK, chunk_id)
    assert SOURCE not in chunk.properties
    assert sorted(e.end.id for e in batch.edges if e.property_iri == FROM_WORK) == [
        "ruby-sapphire",
        "scarlet-violet",
    ]


def test_unknown_flavor_version_fails_the_page(abox: AboxIndex) -> None:
    fact = build_fact(ChunkAspect.FLAVOR, FLAVOR_TEXT).model_copy(update={"versions": ("Nope",)})

    with pytest.raises(BatchBuildError, match="Nope"):
        build_chunks(abox, fact)


def test_a_resolved_mention_links_in(abox: AboxIndex) -> None:
    fact = build_fact(ChunkAspect.EFFECT, "Like Tackle.", MentionRef(template="m", target="Tackle"))

    batch = build_chunks(abox, fact)

    assert [(e.start.id, e.end.id) for e in batch.edges if e.property_iri == MENTIONED_IN] == [
        ("tackle", "bulbapedia:item:oran-berry:effect:1")
    ]


def test_statements_of_a_berry_owner_use_the_item_key_label(abox: AboxIndex) -> None:
    model = load_ontology_model(ONTOLOGY_DIR, "pkmn")
    manifest = discover_universes(DEFAULT_UNIVERSES_DIR, ONTOLOGY_DIR).get("pkmn").manifest
    materializer = build_plugin_materializer(model, manifest)
    batch = build_chunks(abox, build_fact(ChunkAspect.EFFECT, "Restores HP."))

    statements = build_statements(batch, WriteMode.UPSERT, materializer)

    rows = [row for statement in statements for row in statement.parameters.get("stubs", [])]
    claim_rows = [row for statement in statements for row in statement.parameters.get("claims", [])]
    assert [row["key_labels"][0] for row in rows] == ["Item"]
    describes = [row for row in claim_rows if row["type"] == "DESCRIBES"]
    assert [row["anchor_label"] for row in describes] == ["Item"]
