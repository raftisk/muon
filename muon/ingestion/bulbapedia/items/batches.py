"""1 item page extract to its pass 1 and pass 2 batches.

Pass 1 owns the item (scope {`Item`, `HeldItem`, `Berry`}) with the class of its extract.
The Generation it was introduced in and the Flavors of a berry are A-Box stubs, resolved by
label so a missing name fails the page. The batch claims `introducedIn` and `hasFlavor` out
of the owner, so an upsert prunes a flavor the page no longer lists.

Pass 2 owns the TextChunks of the page (scope {`TextChunk`}). Every chunk `describes` the
item, a flavor chunk links `fromWork` to its VersionGroups, and a mention links in through
`mentionedIn`. The Item stub anchors the `describes` claim, so a page with no chunk left
still removes the chunks it had. The stub merges on the `Item` key label, so it finds a
`Berry` or `HeldItem` owner.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from muon.ingestion.batch import GraphBatch, NodeRecord, PropertyValue
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    BERRY,
    DESCRIBES,
    FLAVOR,
    FROM_WORK,
    GENERATION,
    HAS_FLAVOR,
    HELD_ITEM,
    IN,
    INTRODUCED_IN,
    ITEM,
    MENTIONED_IN,
    OUT,
    PKMN,
    TEXT_CHUNK,
    VERSION_GROUP,
    AboxIndex,
    BatchBuildError,
    ClaimRule,
    MentionContext,
    Parts,
    SpeciesResolver,
    assemble_batch,
    build_chunk_record,
    build_claims,
    build_edge,
    build_mention_parts,
    build_owner_claims,
    build_stub,
    build_work_parts,
    drop_none,
    join_parts,
    list_owner_records,
    number_chunks,
    require_abox,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect
from muon.ingestion.bulbapedia.items.chunks import ItemChunkReading
from muon.ingestion.bulbapedia.items.page import (
    FLAVOR_ROW_TEMPLATE,
    GENERATION_KEY,
    INFOBOX_TEMPLATE,
    ItemClass,
    ItemPageExtract,
)
from muon.ingestion.bulbapedia.pages import build_permalink

CATEGORY = PKMN + "category"
BUY_PRICE = PKMN + "buyPrice"
SELL_PRICE = PKMN + "sellPrice"
FLING_POWER = PKMN + "flingPower"
POTENCY = "potency"
GAME_CODE_FIELD = "game code"

ITEM_CLASS_IRIS: Mapping[ItemClass, str] = {
    ItemClass.BERRY: BERRY,
    ItemClass.HELD_ITEM: HELD_ITEM,
    ItemClass.ITEM: ITEM,
}
ITEM_SCOPE = frozenset({ITEM, HELD_ITEM, BERRY})
ITEM_OWNER_CLAIM_RULES = (
    ClaimRule(INTRODUCED_IN, OUT, GENERATION, owns_other_end=False),
    ClaimRule(HAS_FLAVOR, OUT, FLAVOR, owns_other_end=False),
)
ITEM_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    class_iri: ITEM_OWNER_CLAIM_RULES for class_iri in ITEM_SCOPE
}

ITEM_CHUNK_SCOPE = frozenset({TEXT_CHUNK})
ITEM_CHUNK_ANCHOR_CLAIMS = (ClaimRule(DESCRIBES, IN, TEXT_CHUNK, owns_other_end=True),)
ITEM_CHUNK_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (
        ClaimRule(FROM_WORK, OUT, VERSION_GROUP, owns_other_end=False),
        ClaimRule(MENTIONED_IN, IN, None, owns_other_end=False),
    ),
}


@dataclass(frozen=True)
class ItemChunkInputs:
    """What pass 2 resolves a mention against."""

    abox: AboxIndex
    resolver: SpeciesResolver


@dataclass(frozen=True)
class ItemContext:
    """What `require_abox` needs to name a failed item page."""

    page_title: str
    abox: AboxIndex


def build_item_record(extract: ItemPageExtract) -> NodeRecord:
    """Return the item owner record, without the properties that have no value."""
    properties: dict[str, PropertyValue | None] = {
        CATEGORY: extract.category,
        BUY_PRICE: extract.buy_price,
        SELL_PRICE: extract.sell_price,
        FLING_POWER: extract.fling_power,
        ALIASES: extract.aliases or None,
    }
    return NodeRecord(
        class_iri=ITEM_CLASS_IRIS[extract.item_class],
        id=extract.item_id,
        name=extract.name,
        properties=drop_none(properties),
    )


def build_item_batch(extract: ItemPageExtract, abox: AboxIndex) -> GraphBatch:
    """Build the pass 1 batch of 1 page. A name the A-Box lacks raises `BatchBuildError`."""
    context = ItemContext(extract.title, abox)
    item = build_item_record(extract)
    generation = require_abox(
        context, GENERATION, extract.generation_label, (INFOBOX_TEMPLATE, GENERATION_KEY)
    )
    flavors = [
        (require_abox(context, FLAVOR, label, (FLAVOR_ROW_TEMPLATE, label)), potency)
        for label, potency in extract.flavor_potencies.items()
    ]
    parts = Parts(
        nodes=(item, generation, *(flavor for flavor, _ in flavors)),
        edges=(
            build_edge(INTRODUCED_IN, item, generation),
            *(
                build_edge(HAS_FLAVOR, item, flavor, {POTENCY: potency})
                for flavor, potency in flavors
            ),
        ),
    )
    owners = list_owner_records(parts, ITEM_SCOPE)
    return assemble_batch(ITEM_SCOPE, parts, build_owner_claims(owners, ITEM_OWNER_CLAIMS))


def build_item_chunk_batch(
    extract: ItemPageExtract, reading: ItemChunkReading, inputs: ItemChunkInputs
) -> GraphBatch:
    """Build the pass 2 batch of 1 page.

    Raises `ChunkBuildError` for a chunk id collision and `BatchBuildError` for a flavor
    label the A-Box lacks; `check_game_codes` of the move batches keeps the latter from
    happening. A mention that resolves to nothing gives no `mentionedIn` edge.
    """
    item = build_stub(ITEM, extract.name)
    source_url = build_permalink(extract.title, extract.revision_id)
    mentions = MentionContext(inputs.abox, inputs.resolver)
    parts: list[Parts] = [Parts(nodes=(item,))]
    for fact, label in number_chunks(extract.title, reading.chunks):
        chunk = build_chunk_record(fact, label, item, source_url)
        parts.append(Parts((chunk,), (build_edge(DESCRIBES, chunk, item),)))
        if fact.aspect is ChunkAspect.FLAVOR:
            work_parts, missing = build_work_parts(fact, chunk, inputs.abox, VERSION_GROUP)
            if missing:
                raise BatchBuildError(extract.title, fact.section_path, GAME_CODE_FIELD, missing[0])
            parts.append(work_parts)
        mention_parts, _ = build_mention_parts(fact, chunk, item, mentions)
        parts.append(mention_parts)
    joined = join_parts(parts)
    claims = (
        *build_claims([item.node_ref], ITEM_CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(list_owner_records(joined, ITEM_CHUNK_SCOPE), ITEM_CHUNK_OWNER_CLAIMS),
    )
    return assemble_batch(ITEM_CHUNK_SCOPE, joined, claims)
