"""1 location page extract to its pass 1 batch.

Pass 1 owns the Location of the page and its LocationAreas. The scope of a batch holds the
class of the page and `LocationArea` only: every other node is a reference, so the Runner
never rewrites the labels of a place that another page owns. A reference to a place gets the
class that place's own page has, taken from `LocationBuildInputs.classes`; a place no page
owns gets `mo:Location`. A page whose title is an A-Box Region (L5) gives a batch with an
empty scope and the Region as a reference, so a run never sets a property, label or edge on a
seed Region.

The owner claims cover `locatedIn`, `adjacentTo`, `appearsIn`, `introducedIn` and `requires`
out of the Location and `locatedIn` in from its areas, so an upsert prunes whatever the page
no longer lists.

Pass 2 owns the TextChunks of the page (scope {`TextChunk`}). Every chunk `describes` the
Location, and a mention links in through `mentionedIn`. The Location is a reference of the
class of its page, so a page with no chunk left still removes the chunks it had.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from muon.ingestion.batch import GraphBatch, NodeRecord, PropertyValue, build_node_id
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    CORE,
    DESCRIBES,
    GENERATION,
    IN,
    INTRODUCED_IN,
    ITEM,
    LOCATION,
    MENTIONED_IN,
    MOVE,
    OUT,
    PKMN,
    REGION,
    TEXT_CHUNK,
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
    drop_none,
    join_parts,
    number_chunks,
    require_abox,
    resolve_mention,
)
from muon.ingestion.bulbapedia.chunking import MentionRef
from muon.ingestion.bulbapedia.locations.areas import build_area_id
from muon.ingestion.bulbapedia.locations.chunks import LocationChunkReading
from muon.ingestion.bulbapedia.locations.identity import (
    LocationIndex,
    LocationVerdict,
    classify_location,
    normalize_title,
)
from muon.ingestion.bulbapedia.locations.page import (
    FIELD_MOVE_ROLE,
    LocationClass,
    LocationPageExtract,
)
from muon.ingestion.bulbapedia.pages import build_permalink

SETTLEMENT = CORE + "Settlement"
ROUTE = CORE + "Route"
GEO_FEATURE = CORE + "GeoFeature"
STRUCTURE = CORE + "Structure"
LOCATION_AREA = PKMN + "LocationArea"
VERSION = PKMN + "Version"
LOCATED_IN = CORE + "locatedIn"
ADJACENT_TO = CORE + "adjacentTo"
APPEARS_IN = CORE + "appearsIn"
REQUIRES = PKMN + "requires"
DIRECTION = "direction"
ROLE = "role"
INFOBOX_FIELD = "infobox"
VERSION_GROUP_FIELD = "version group"
GENERATION_FIELD = "generation"
REGION_FIELD = "title"
PAGE_PATH = "location page"

CLASS_IRIS: Mapping[LocationClass, str] = {
    LocationClass.SETTLEMENT: SETTLEMENT,
    LocationClass.ROUTE: ROUTE,
    LocationClass.GEO_FEATURE: GEO_FEATURE,
    LocationClass.STRUCTURE: STRUCTURE,
    LocationClass.LOCATION: LOCATION,
}
# The classes a Location page can have.
LOCATION_FAMILY = frozenset(CLASS_IRIS.values())
# Every class a pass 1 batch can own; a batch holds the class of its page and the area class.
LOCATION_SCOPE = LOCATION_FAMILY | {LOCATION_AREA}

LOCATION_OWNER_RULES = (
    ClaimRule(LOCATED_IN, OUT, LOCATION, owns_other_end=False),
    ClaimRule(ADJACENT_TO, OUT, LOCATION, owns_other_end=False),
    ClaimRule(APPEARS_IN, OUT, VERSION, owns_other_end=False),
    ClaimRule(INTRODUCED_IN, OUT, GENERATION, owns_other_end=False),
    ClaimRule(REQUIRES, OUT, None, owns_other_end=False),
    ClaimRule(LOCATED_IN, IN, LOCATION_AREA, owns_other_end=True),
)
LOCATION_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    class_iri: LOCATION_OWNER_RULES for class_iri in LOCATION_FAMILY
}


LOCATION_CHUNK_SCOPE = frozenset({TEXT_CHUNK})
LOCATION_CHUNK_ANCHOR_CLAIMS = (ClaimRule(DESCRIBES, IN, TEXT_CHUNK, owns_other_end=True),)
LOCATION_CHUNK_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (ClaimRule(MENTIONED_IN, IN, None, owns_other_end=False),),
}


def resolve_no_species(target: str) -> NodeRecord | None:
    return None


@dataclass(frozen=True)
class LocationBuildInputs:
    """What every page of a run resolves against.

    `classes` maps the normalized title of each owned page to its class IRI, `group_versions`
    a VersionGroup label to the labels of its Versions and `resolver` a `{{p}}` mention to a
    species stub.
    """

    abox: AboxIndex
    index: LocationIndex
    classes: Mapping[str, str]
    group_versions: Mapping[str, tuple[str, ...]]
    resolver: SpeciesResolver = resolve_no_species


@dataclass(frozen=True)
class LocationContext:
    """What `require_abox` needs to name a failed location page."""

    page_title: str
    abox: AboxIndex


def build_location_id(title: str) -> str:
    return build_node_id(normalize_title(title))


def build_location_record(extract: LocationPageExtract) -> NodeRecord:
    """Return the Location owner record, with `aliases` only when the page has some."""
    if extract.leaf_class is None:
        raise ValueError(f"{extract.title}: an A-Box Region page has no owner record")
    properties: dict[str, PropertyValue | None] = {ALIASES: extract.aliases or None}
    return NodeRecord(
        class_iri=CLASS_IRIS[extract.leaf_class],
        id=build_location_id(extract.title),
        name=extract.title,
        properties=drop_none(properties),
    )


def build_reference(title: str, inputs: LocationBuildInputs) -> NodeRecord | None:
    """Return the reference node of a place a page names, or `None` when the title is not a
    place.

    An A-Box Region is a Region. A place with a page of its own has that page's class, and
    a place with no owner is a `mo:Location`.
    """
    region_id = inputs.abox.ids.get((REGION, title))
    if region_id is not None:
        return NodeRecord(class_iri=REGION, id=region_id, name=title)
    test = classify_location(inputs.index, title)
    if test.verdict is LocationVerdict.NOT_A_LOCATION:
        return None
    return build_stub(inputs.classes.get(test.title, LOCATION), test.title)


def build_version_nodes(
    extract: LocationPageExtract, inputs: LocationBuildInputs, context: LocationContext
) -> tuple[NodeRecord, ...]:
    """Return the Versions of every version group of the page, without repeats."""
    nodes: dict[str, NodeRecord] = {}
    for group in extract.version_groups:
        labels = inputs.group_versions.get(group)
        if labels is None:
            raise BatchBuildError(extract.title, PAGE_PATH, VERSION_GROUP_FIELD, group)
        for label in labels:
            version = require_abox(context, VERSION, label, (PAGE_PATH, VERSION_GROUP_FIELD))
            nodes[version.id] = version
    return tuple(nodes.values())


def build_region_batch(extract: LocationPageExtract, context: LocationContext) -> GraphBatch:
    region = require_abox(
        context, REGION, normalize_title(extract.title), (PAGE_PATH, REGION_FIELD)
    )
    return assemble_batch(frozenset(), Parts(nodes=(region,)))


def build_location_batch(extract: LocationPageExtract, inputs: LocationBuildInputs) -> GraphBatch:
    """Build the pass 1 batch of 1 page. A name the A-Box lacks raises `BatchBuildError`."""
    context = LocationContext(extract.title, inputs.abox)
    if extract.is_abox_region:
        return build_region_batch(extract, context)
    owner = build_location_record(extract)
    parts: list[Parts] = [Parts(nodes=(owner,))]
    for title in extract.parent_titles:
        parent = build_reference(title, inputs)
        if parent is not None:
            parts.append(Parts((parent,), (build_edge(LOCATED_IN, owner, parent),)))
    neighbour_directions = {n.title: n.direction for n in reversed(extract.neighbours)}
    for title, direction in reversed(neighbour_directions.items()):
        neighbour = build_reference(title, inputs)
        if neighbour is not None:
            edge = build_edge(ADJACENT_TO, owner, neighbour, {DIRECTION: direction})
            parts.append(Parts((neighbour,), (edge,)))
    versions = build_version_nodes(extract, inputs, context)
    parts.append(Parts(versions, tuple(build_edge(APPEARS_IN, owner, v) for v in versions)))
    if extract.generation is not None:
        generation = require_abox(
            context, GENERATION, extract.generation, (INFOBOX_FIELD, GENERATION_FIELD)
        )
        parts.append(Parts((generation,), (build_edge(INTRODUCED_IN, owner, generation),)))
    for requirement in extract.field_requirements:
        class_iri = MOVE if requirement.role == FIELD_MOVE_ROLE else ITEM
        target = build_stub(class_iri, requirement.target)
        edge = build_edge(REQUIRES, owner, target, {ROLE: requirement.role})
        parts.append(Parts((target,), (edge,)))
    for area in extract.areas:
        record = NodeRecord(
            class_iri=LOCATION_AREA,
            id=build_area_id(owner.id, area.key),
            name=f"{extract.title} {area.name}",
        )
        parts.append(Parts((record,), (build_edge(LOCATED_IN, record, owner),)))
    claims = build_owner_claims([owner], LOCATION_OWNER_CLAIMS)
    scope = frozenset({owner.class_iri, LOCATION_AREA})
    return assemble_batch(scope, join_parts(parts), claims)


def build_chunk_subject(extract: LocationPageExtract, inputs: LocationBuildInputs) -> NodeRecord:
    """Return the reference of the Location a chunk describes: a Region for an A-Box Region
    page, else a node of the class of the page."""
    if extract.is_abox_region:
        context = LocationContext(extract.title, inputs.abox)
        return require_abox(
            context, REGION, normalize_title(extract.title), (PAGE_PATH, REGION_FIELD)
        )
    if extract.leaf_class is None:
        raise ValueError(f"{extract.title}: a page that is not an A-Box Region needs a class")
    return build_stub(CLASS_IRIS[extract.leaf_class], normalize_title(extract.title))


def build_location_chunk_batch(
    extract: LocationPageExtract, reading: LocationChunkReading, inputs: LocationBuildInputs
) -> GraphBatch:
    """Build the pass 2 batch of 1 page.

    Raises `ChunkBuildError` for a chunk id collision. A mention that resolves to nothing
    gives no `mentionedIn` edge.
    """
    subject = build_chunk_subject(extract, inputs)
    source_url = build_permalink(extract.title, extract.revision_id)
    mentions = MentionContext(inputs.abox, inputs.resolver)
    parts: list[Parts] = [Parts(nodes=(subject,))]
    for fact, label in number_chunks(extract.title, reading.chunks):
        chunk = build_chunk_record(fact, label, subject, source_url)
        parts.append(Parts((chunk,), (build_edge(DESCRIBES, chunk, subject),)))
        mention_parts, _ = build_mention_parts(fact, chunk, subject, mentions)
        parts.append(mention_parts)
    joined = join_parts(parts)
    claims = (
        *build_claims([subject.node_ref], LOCATION_CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(
            [node for node in joined.nodes if node.class_iri in LOCATION_CHUNK_SCOPE],
            LOCATION_CHUNK_OWNER_CLAIMS,
        ),
    )
    return assemble_batch(LOCATION_CHUNK_SCOPE, joined, claims)


def list_unresolved_mentions(
    reading: LocationChunkReading, inputs: LocationBuildInputs
) -> tuple[MentionRef, ...]:
    """Return the mentions of the chunks that resolve to nothing, in page order."""
    context = MentionContext(inputs.abox, inputs.resolver)
    return tuple(
        mention
        for fact in reading.chunks
        for mention in fact.mentions
        if resolve_mention(mention, context) is None
    )
