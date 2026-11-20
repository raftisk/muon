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

Pass 3 owns the ItemFinds of the page (scope {`ItemFind`}). A find is at the Location or one of
its areas in 1 Version, and its Item `hasFind` it. The Location and each area claim their
finds, so a re-run removes the finds a page no longer lists, and the Location claims the
`foundIn` shortcut of the Items it holds.

Pass 4 owns the Encounters of the page (scope {`Encounter`}). An encounter is at the Location
or one of its areas in 1 Version, and its species `hasEncounter` it. The id starts with the id
of the place, so the claims at the Location and at each area filter on the id prefix and a
species page that owns the coarse Encounters of the same species never loses them.

Pass 2 owns the TextChunks of the page (scope {`TextChunk`}). Every chunk `describes` the
Location, and a mention links in through `mentionedIn`. The Location is a reference of the
class of its page, so a page with no chunk left still removes the chunks it had.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from muon.ingestion.batch import GraphBatch, NodeRecord, PropertyValue, build_node_id
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    CORE,
    DESCRIBES,
    FOUND_IN,
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
from muon.ingestion.bulbapedia.locations.areas import AreaRef, build_area_id
from muon.ingestion.bulbapedia.locations.chunks import LocationChunkReading
from muon.ingestion.bulbapedia.locations.encounters import EncounterFact
from muon.ingestion.bulbapedia.locations.finds import FindFact
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


ITEM_FIND = PKMN + "ItemFind"
AT_LOCATION = PKMN + "atLocation"
IN_VERSION = PKMN + "inVersion"
HAS_FIND = PKMN + "hasFind"
IS_HIDDEN = PKMN + "isHidden"
METHOD = PKMN + "method"
FIND_ID_SEPARATOR = ":"
HIDDEN_SEGMENT = "hidden"
FIND_NAME_TEMPLATE = "{item} at {place} ({version}{suffix})"

ENCOUNTER = PKMN + "Encounter"
HAS_ENCOUNTER = PKMN + "hasEncounter"
ENCOUNTER_KIND = PKMN + "encounterKind"
MIN_LEVEL = PKMN + "minLevel"
MAX_LEVEL = PKMN + "maxLevel"
CHANCE = PKMN + "chance"
CONDITIONS = PKMN + "conditions"
ENCOUNTER_NAME_TEMPLATE = "{species} at {place} ({details})"

LOCATION_ENCOUNTER_SCOPE = frozenset({ENCOUNTER})
LOCATION_ENCOUNTER_ANCHOR_CLAIMS = (
    ClaimRule(AT_LOCATION, IN, ENCOUNTER, owns_other_end=True, is_id_prefixed=True),
)
LOCATION_FIND_SCOPE = frozenset({ITEM_FIND})
LOCATION_FIND_ANCHOR_CLAIMS = (ClaimRule(AT_LOCATION, IN, ITEM_FIND, owns_other_end=True),)
LOCATION_FIND_SHORTCUT_CLAIMS = (ClaimRule(FOUND_IN, IN, ITEM, owns_other_end=False),)
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
    a VersionGroup label to the labels of its Versions, `resolver` a `{{p}}` mention to a
    species stub and `item_owners` the lowercase name of each Item owner to its id and name.
    """

    abox: AboxIndex
    index: LocationIndex
    classes: Mapping[str, str]
    group_versions: Mapping[str, tuple[str, ...]]
    resolver: SpeciesResolver = resolve_no_species
    item_owners: Mapping[str, tuple[str, str]] = field(default_factory=dict)


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

    A redirect is followed to its target. An A-Box Region is a Region. A place with a page of
    its own has that page's class, and a place with no owner is a `mo:Location`.
    """
    test = classify_location(inputs.index, title)
    if test.verdict is LocationVerdict.NOT_A_LOCATION:
        return None
    region_id = inputs.abox.ids.get((REGION, test.title))
    if region_id is not None:
        return NodeRecord(class_iri=REGION, id=region_id, name=test.title)
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


@dataclass(frozen=True)
class FindBuild:
    """The pass 3 batch of a page, the facts that collapsed into another and the Item names
    no Item owner has."""

    batch: GraphBatch
    duplicates: int
    item_stub_names: tuple[str, ...]


def build_find_id(
    at_id: str, version_id: str, item_id: str, method: str | None, is_hidden: bool
) -> str:
    """Return the ItemFind id `{place}:{version}:{item}[:{method}][:hidden]`."""
    segments = (at_id, version_id, item_id, *((method,) if method else ()))
    return FIND_ID_SEPARATOR.join((*segments, *((HIDDEN_SEGMENT,) if is_hidden else ())))


def build_find_item(name: str, inputs: LocationBuildInputs) -> tuple[NodeRecord, bool]:
    """Return the Item reference of a name and whether no Item owner has it.

    The reference has the id and name of the owner. An unowned name gets the id of the name.
    """
    owner = inputs.item_owners.get(name.lower())
    if owner is None:
        return build_stub(ITEM, name), True
    return NodeRecord(class_iri=ITEM, id=owner[0], name=owner[1]), False


def build_place(
    area: AreaRef | None, owner: NodeRecord, extract: LocationPageExtract
) -> NodeRecord:
    """Return the reference of the place a find or an encounter is at: the page Location or
    one of its areas.

    A heading that names another pinned location gives the page Location. The named location
    claims every find and encounter at it, so a row of this page there would be deleted by
    that page's run.
    """
    if area is None or not area.key:
        return owner
    return build_area_reference(area, owner, extract)


def build_area_reference(
    area: AreaRef, owner: NodeRecord, extract: LocationPageExtract
) -> NodeRecord:
    return NodeRecord(
        class_iri=LOCATION_AREA,
        id=build_area_id(owner.id, area.key),
        name=f"{extract.title} {area.name}",
    )


def list_anchors(
    owner: NodeRecord, extract: LocationPageExtract, places: Sequence[NodeRecord]
) -> list[NodeRecord]:
    """Return the Location, every area of the page and every place a row is at, once each:
    the nodes a pass claims its rows at, so that a place that lost its rows loses them in the
    graph too."""
    anchors = {owner.node_ref: owner}
    for area in extract.areas:
        reference = build_area_reference(area, owner, extract)
        anchors[reference.node_ref] = reference
    for place in places:
        anchors[place.node_ref] = place
    return list(anchors.values())


def build_find_batch(
    extract: LocationPageExtract, facts: Sequence[FindFact], inputs: LocationBuildInputs
) -> FindBuild:
    """Build the pass 3 batch of 1 page. A Version label the A-Box lacks raises
    `BatchBuildError`; facts with the same id give 1 find. An A-Box Region page gives a batch
    with the Region as its only node and no claim, like its pass 1 batch."""
    context = LocationContext(extract.title, inputs.abox)
    owner = build_chunk_subject(extract, inputs)
    if extract.is_abox_region:
        return FindBuild(assemble_batch(frozenset(), Parts(nodes=(owner,))), 0, ())
    parts: list[Parts] = [Parts(nodes=(owner,))]
    places: list[NodeRecord] = []
    finds: set[str] = set()
    duplicates = 0
    unowned: set[str] = set()
    for fact in facts:
        place = build_place(fact.area, owner, extract)
        version = require_abox(context, VERSION, fact.version, (PAGE_PATH, VERSION_GROUP_FIELD))
        item, is_unowned = build_find_item(fact.item_name, inputs)
        if is_unowned:
            unowned.add(fact.item_name)
        find_id = build_find_id(place.id, version.id, item.id, fact.method, fact.is_hidden)
        if find_id in finds:
            duplicates += 1
            continue
        finds.add(find_id)
        properties: dict[str, PropertyValue | None] = {
            IS_HIDDEN: True if fact.is_hidden else None,
            METHOD: fact.method,
        }
        suffix = "".join(
            f", {part}"
            for part in (fact.method, HIDDEN_SEGMENT if fact.is_hidden else None)
            if part
        )
        find = NodeRecord(
            class_iri=ITEM_FIND,
            id=find_id,
            name=FIND_NAME_TEMPLATE.format(
                item=item.name, place=place.name, version=version.name, suffix=suffix
            ),
            properties=drop_none(properties),
        )
        places.append(place)
        edges = (
            build_edge(AT_LOCATION, find, place),
            build_edge(IN_VERSION, find, version),
            build_edge(HAS_FIND, item, find),
            build_edge(FOUND_IN, item, owner),
        )
        parts.append(Parts((find, place, version, item), edges))
    claims = (
        *build_claims(
            [a.node_ref for a in list_anchors(owner, extract, places)], LOCATION_FIND_ANCHOR_CLAIMS
        ),
        *build_claims([owner.node_ref], LOCATION_FIND_SHORTCUT_CLAIMS),
    )
    anchor_parts = Parts(tuple(list_anchors(owner, extract, places)))
    batch = assemble_batch(LOCATION_FIND_SCOPE, join_parts([*parts, anchor_parts]), claims)
    return FindBuild(batch, duplicates, tuple(sorted(unowned)))


@dataclass(frozen=True)
class EncounterBuild:
    """The pass 4 batch of a page, the encounters that merged into another, the identical rows
    that collapsed and the species names no species page has."""

    batch: GraphBatch
    merges: int
    duplicates: int
    unresolved_species: tuple[str, ...]


def build_location_encounter_id(
    at_id: str,
    version_id: str,
    species_id: str,
    method: str | None,
    conditions: Sequence[str],
) -> str:
    """Return the Encounter id `{place}:{version}:{species}[:{method}][:{condition} ...]`, the
    conditions sorted."""
    segments = (at_id, version_id, species_id, *((method,) if method else ()), *sorted(conditions))
    return FIND_ID_SEPARATOR.join(segments)


def merge_encounter_facts(first: EncounterFact, second: EncounterFact) -> EncounterFact:
    """Return the union of the level ranges of 2 facts with 1 id and no chance: the rows differ,
    and a chance of either would be wrong for both."""
    levels = [
        level
        for fact in (first, second)
        for level in (fact.min_level, fact.max_level)
        if level is not None
    ]
    return EncounterFact(
        species_title=first.species_title,
        area=first.area,
        version=first.version,
        kind=first.kind,
        method=first.method,
        min_level=min(levels) if levels else None,
        max_level=max(levels) if levels else None,
        chance=None,
        conditions=first.conditions,
    )


def build_encounter_record(
    fact: EncounterFact,
    encounter_id: str,
    species: NodeRecord,
    place: NodeRecord,
    version: NodeRecord,
) -> NodeRecord:
    properties: dict[str, PropertyValue | None] = {
        ENCOUNTER_KIND: fact.kind,
        METHOD: fact.method,
        MIN_LEVEL: fact.min_level,
        MAX_LEVEL: fact.max_level,
        CHANCE: fact.chance,
        CONDITIONS: tuple(sorted(fact.conditions)) or None,
    }
    details = ", ".join(
        part for part in (version.name, fact.method, *sorted(fact.conditions)) if part
    )
    name = ENCOUNTER_NAME_TEMPLATE.format(species=species.name, place=place.name, details=details)
    return NodeRecord(
        class_iri=ENCOUNTER, id=encounter_id, name=name, properties=drop_none(properties)
    )


def build_encounter_batch(
    extract: LocationPageExtract, facts: Sequence[EncounterFact], inputs: LocationBuildInputs
) -> EncounterBuild:
    """Build the pass 4 batch of 1 page. A Version label the A-Box lacks raises
    `BatchBuildError`. A species no species page has gives no encounter and is listed. Facts
    with the same id and the same values give 1 encounter; facts with the same id and other
    values merge. An A-Box Region page gives a batch with the Region as its only node."""
    context = LocationContext(extract.title, inputs.abox)
    owner = build_chunk_subject(extract, inputs)
    if extract.is_abox_region:
        return EncounterBuild(assemble_batch(frozenset(), Parts(nodes=(owner,))), 0, 0, ())
    kept: dict[str, tuple[EncounterFact, NodeRecord, NodeRecord, NodeRecord]] = {}
    places: list[NodeRecord] = []
    duplicates = merges = 0
    unresolved: set[str] = set()
    for fact in facts:
        species = inputs.resolver(fact.species_title)
        if species is None:
            unresolved.add(fact.species_title)
            continue
        place = build_place(fact.area, owner, extract)
        version = require_abox(context, VERSION, fact.version, (PAGE_PATH, VERSION_GROUP_FIELD))
        encounter_id = build_location_encounter_id(
            place.id, version.id, species.id, fact.method, fact.conditions
        )
        earlier = kept.get(encounter_id)
        if earlier is None:
            kept[encounter_id] = (fact, species, place, version)
            places.append(place)
        elif earlier[0] == fact:
            duplicates += 1
        else:
            merges += 1
            kept[encounter_id] = (merge_encounter_facts(earlier[0], fact), *earlier[1:])
    parts: list[Parts] = [Parts(nodes=(owner,))]
    for encounter_id, (fact, species, place, version) in kept.items():
        encounter = build_encounter_record(fact, encounter_id, species, place, version)
        edges = (
            build_edge(HAS_ENCOUNTER, species, encounter),
            build_edge(AT_LOCATION, encounter, place),
            build_edge(IN_VERSION, encounter, version),
        )
        parts.append(Parts((encounter, species, place, version), edges))
    anchors = list_anchors(owner, extract, places)
    claims = build_claims([a.node_ref for a in anchors], LOCATION_ENCOUNTER_ANCHOR_CLAIMS)
    batch = assemble_batch(
        LOCATION_ENCOUNTER_SCOPE, join_parts([*parts, Parts(tuple(anchors))]), claims
    )
    return EncounterBuild(batch, merges, duplicates, tuple(sorted(unresolved)))
