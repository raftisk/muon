"""Batch builders: a species page extract becomes its pass-1, pass-2 and pass-3 `GraphBatch`.

Pass 1 (`build_species_batch`) owns the species nodes, their learnset entries and the Mega
and Primal form changes, which all belong to 1 page. Pass 2 (`build_evolution_batch`) owns
the evolutions whose from-species is a node of the page; they point at species of other
pages, so they wait until every species batch is written. Names that the hand-written
A-Box owns resolve through `AboxIndex`, and a miss fails the page. Evolution references
resolve through `SpeciesIndex`, and a miss drops the evolution; neither creates a stub for
an id nothing owns.

Pass 3 (`build_chunk_batch`) owns the text chunks of the page. A chunk describes a species
node of the page and is mentioned by the entities its typed templates name, so the batch
holds species, move, ability, type and version stubs. A mention that resolves to nothing is
noted and never fails the page.

Each batch also claims the edge patterns its page owns, so a re-ingest deletes what the
page no longer gives. The claim tables below say which patterns each pass claims. The
edges that a node id fixes (`teachesMove`, `inVersionGroup`, `toSpecies`, `describes`
of a chunk and the `fromSpecies` of a change) are not claimed, since they cannot go
stale. Passes 2 and 3 anchor their claims at a stub of every form of the page, and a
page that gives no evolution or no chunk still gets that batch, with the anchors, their
claims and no owner record, so a re-ingest deletes what the page gave before.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from muon.ingestion.batch import (
    ClaimDirection,
    EdgeClaim,
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    NodeRef,
    PropertyValue,
    build_node_id,
)
from muon.ingestion.bulbapedia.species.evolution import EVOLUTION_PATH
from muon.ingestion.bulbapedia.species.extract import (
    BaseStats,
    ChunkAspect,
    ChunkFact,
    EvolutionFact,
    FormChangeFact,
    FormFacts,
    LearnMethod,
    LearnsetRow,
    MentionRef,
    NoteKind,
    PageNote,
    Requirement,
    RequirementRole,
    SpeciesPageExtract,
    SpeciesRef,
)
from muon.ingestion.bulbapedia.species.learnset import LEARNSET_PATH
from muon.ingestion.bulbapedia.species.page import INFOBOX_LABEL
from muon.ingestion.bulbapedia.wikitext import PATH_SEPARATOR
from muon.ontology.model import extract_local_name

UNIVERSE = "pkmn"
PKMN = "https://muon.dev/ns/pkmn#"
CORE = "https://muon.dev/ns/core#"

POKEMON_SPECIES = PKMN + "PokemonSpecies"
LEARNSET_ENTRY = PKMN + "LearnsetEntry"
FORM_CHANGE = PKMN + "FormChange"
EVOLUTION = PKMN + "Evolution"
TYPE = PKMN + "Type"
EGG_GROUP = PKMN + "EggGroup"
STAT = PKMN + "Stat"
GENERATION = PKMN + "Generation"
VERSION_GROUP = PKMN + "VersionGroup"
ABILITY = PKMN + "Ability"
MOVE = PKMN + "Move"
MACHINE = PKMN + "Machine"
HELD_ITEM = PKMN + "HeldItem"
ITEM = PKMN + "Item"
REGION = CORE + "Region"
TEXT_CHUNK = CORE + "TextChunk"
VERSION = PKMN + "Version"

HAS_TYPE = PKMN + "hasType"
HAS_ABILITY = PKMN + "hasAbility"
IN_EGG_GROUP = PKMN + "inEggGroup"
YIELDS_EV = PKMN + "yieldsEv"
HAS_FORM = PKMN + "hasForm"
INTRODUCED_IN = PKMN + "introducedIn"
HAS_LEARNSET_ENTRY = PKMN + "hasLearnsetEntry"
TEACHES_MOVE = PKMN + "teachesMove"
IN_VERSION_GROUP = PKMN + "inVersionGroup"
VIA_MACHINE = PKMN + "viaMachine"
CAN_LEARN = PKMN + "canLearn"
FROM_SPECIES = PKMN + "fromSpecies"
TO_SPECIES = PKMN + "toSpecies"
REQUIRES = PKMN + "requires"
EVOLVES_TO = PKMN + "evolvesTo"
DESCRIBES = CORE + "describes"
MENTIONED_IN = CORE + "mentionedIn"
FROM_WORK = CORE + "fromWork"

POKEDEX_NUMBER = PKMN + "pokedexNumber"
IS_DEFAULT_FORM = PKMN + "isDefaultForm"
HEIGHT = PKMN + "height"
WEIGHT = PKMN + "weight"
FEMALE_RATE = PKMN + "femaleRate"
IS_GENDERLESS = PKMN + "isGenderless"
CATCH_RATE = PKMN + "catchRate"
BASE_EXP_YIELD = PKMN + "baseExpYield"
GROWTH_RATE = PKMN + "growthRate"
EGG_CYCLES = PKMN + "eggCycles"
BASE_FRIENDSHIP = PKMN + "baseFriendship"
COLOR = PKMN + "color"
SHAPE = PKMN + "shape"
GENUS = PKMN + "genus"
IS_LEGENDARY = PKMN + "isLegendary"
IS_MYTHICAL = PKMN + "isMythical"
IS_STARTER = PKMN + "isStarter"
BASE_HP = PKMN + "baseHp"
BASE_ATTACK = PKMN + "baseAttack"
BASE_DEFENSE = PKMN + "baseDefense"
BASE_SP_ATTACK = PKMN + "baseSpAttack"
BASE_SP_DEFENSE = PKMN + "baseSpDefense"
BASE_SPEED = PKMN + "baseSpeed"
ALIASES = CORE + "aliases"
METHOD = PKMN + "method"
LEVEL = PKMN + "level"
FORM_KIND = PKMN + "formKind"
IS_BATTLE_ONLY = PKMN + "isBattleOnly"
TRIGGER = PKMN + "trigger"
EVO_TRIGGER = PKMN + "evoTrigger"
MIN_LEVEL = PKMN + "minLevel"
TIME_OF_DAY = PKMN + "timeOfDay"
GENDER = PKMN + "gender"
CONDITION = PKMN + "condition"
TEXT = CORE + "text"
ASPECT = CORE + "aspect"
SOURCE = CORE + "source"
SECTION = CORE + "section"
SOURCE_URL = CORE + "sourceUrl"

SPECIES_SCOPE = frozenset({POKEMON_SPECIES, LEARNSET_ENTRY, FORM_CHANGE})
EVOLUTION_SCOPE = frozenset({EVOLUTION})
CHUNK_SCOPE = frozenset({TEXT_CHUNK})
CHUNK_SOURCE = "bulbapedia"
CHUNK_ID_TEMPLATE = "{source}:{label}:{subject}:{aspect}:{n}"
CHUNK_NAME_TEMPLATE = "{subject} {aspect} {n}"
TEXT_HASH_LENGTH = 8
SPECIES_MENTION = "p"
MENTION_CLASSES: Mapping[str, str] = {"m": MOVE, "a": ABILITY, "type": TYPE, "t": TYPE}
MENTION_NOUNS: Mapping[str, str] = {
    SPECIES_MENTION: "species",
    "m": "move",
    "a": "ability",
    "type": "A-Box Type",
    "t": "A-Box Type",
}
ROW_DROPPED_SUFFIX = "; row dropped"
SNAPSHOT_VERSION_GROUP = "Scarlet-Violet"
GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERALS = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))

SLOT = "slot"
IS_HIDDEN = "isHidden"
AMOUNT = "amount"
REGION_PROPERTY = "region"
ROLE = "role"

EV_STAT_NAMES: Mapping[str, str] = {
    "evhp": "HP",
    "evat": "Attack",
    "evde": "Defense",
    "evsa": "Special Attack",
    "evsd": "Special Defense",
    "evsp": "Speed",
}
REQUIREMENT_CLASSES: Mapping[RequirementRole, str] = {
    RequirementRole.USE: ITEM,
    RequirementRole.HOLD: HELD_ITEM,
    RequirementRole.KNOWN_MOVE: MOVE,
    RequirementRole.KNOWN_MOVE_TYPE: TYPE,
    RequirementRole.PARTY_SPECIES: POKEMON_SPECIES,
    RequirementRole.PARTY_TYPE: TYPE,
    RequirementRole.TRADED_FOR: POKEMON_SPECIES,
}
A_BOX_REQUIREMENT_CLASSES = frozenset({TYPE})
HOLD_ROLE = RequirementRole.HOLD.value
INFOBOX_SECTION = INFOBOX_LABEL
LEARNSET_SECTION = PATH_SEPARATOR.join(LEARNSET_PATH)
EVOLUTION_SECTION = PATH_SEPARATOR.join(EVOLUTION_PATH)


@dataclass(frozen=True)
class ClaimRule:
    """An edge pattern that a batch claims at every anchor of a class."""

    property_iri: str
    direction: ClaimDirection
    other_class_iri: str | None
    owns_other_end: bool


OUT = ClaimDirection.OUT
IN = ClaimDirection.IN

SPECIES_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    POKEMON_SPECIES: (
        ClaimRule(CAN_LEARN, OUT, MOVE, False),
        ClaimRule(HAS_ABILITY, OUT, ABILITY, False),
        ClaimRule(HAS_TYPE, OUT, TYPE, False),
        ClaimRule(IN_EGG_GROUP, OUT, EGG_GROUP, False),
        ClaimRule(INTRODUCED_IN, OUT, GENERATION, False),
        ClaimRule(YIELDS_EV, OUT, STAT, False),
        ClaimRule(HAS_LEARNSET_ENTRY, OUT, LEARNSET_ENTRY, True),
        ClaimRule(HAS_FORM, OUT, POKEMON_SPECIES, True),
        ClaimRule(FROM_SPECIES, IN, FORM_CHANGE, True),
    ),
    LEARNSET_ENTRY: (ClaimRule(VIA_MACHINE, OUT, MACHINE, False),),
    FORM_CHANGE: (ClaimRule(REQUIRES, OUT, None, False),),
}
EVOLUTION_ANCHOR_CLAIMS: tuple[ClaimRule, ...] = (
    ClaimRule(EVOLVES_TO, OUT, POKEMON_SPECIES, False),
    ClaimRule(FROM_SPECIES, IN, EVOLUTION, True),
)
EVOLUTION_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    EVOLUTION: (ClaimRule(REQUIRES, OUT, None, False),),
}
CHUNK_ANCHOR_CLAIMS: tuple[ClaimRule, ...] = (ClaimRule(DESCRIBES, IN, TEXT_CHUNK, True),)
CHUNK_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (
        ClaimRule(FROM_WORK, OUT, VERSION, False),
        ClaimRule(MENTIONED_IN, IN, None, False),
    ),
}


class BatchBuildError(Exception):
    """A page names something the A-Box lacks. The message names all 4 fields."""

    def __init__(self, page_title: str, section_path: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: no A-Box node for {field} in {section_path}: {value!r}")
        self.page_title = page_title
        self.section_path = section_path
        self.field = field
        self.value = value


@dataclass(frozen=True)
class AboxIndex:
    """The ids of the hand-written A-Box nodes by class and label, and the generation of
    each region."""

    ids: Mapping[tuple[str, str], str]
    region_generations: Mapping[str, str]


@dataclass(frozen=True)
class SpeciesIndex:
    """Every species node id of a run with its name, and each base species' id by name."""

    names: Mapping[str, str]
    ids_by_name: Mapping[str, str]


@dataclass(frozen=True)
class Parts:
    nodes: tuple[NodeRecord, ...] = ()
    edges: tuple[EdgeRecord, ...] = ()


@dataclass(frozen=True)
class PageContext:
    """What a builder looks names up in: the A-Box and, for pass 2, every species of the run."""

    page_title: str
    abox: AboxIndex
    species: SpeciesIndex = SpeciesIndex(names={}, ids_by_name={})


@dataclass(frozen=True)
class EvolutionBuild:
    """The pass-2 batch of a page (with no owner record when it owns no evolution), what was
    dropped or substituted, and how many of its evolutions a rule structured."""

    batch: GraphBatch
    notes: tuple[PageNote, ...]
    structured: int
    total: int


class AboxMissError(Exception):
    def __init__(self, class_iri: str, label: str) -> None:
        super().__init__(f"No A-Box node of class {class_iri} labelled {label!r}")
        self.class_iri = class_iri
        self.label = label


def build_abox_index(abox: GraphBatch) -> AboxIndex:
    """Index the nodes of `read_abox()` by (class, label) and map each region to the label
    of the generation its `introducedIn` edge names."""
    ids = {(node.class_iri, node.name): node.id for node in abox.nodes}
    names = {node.node_ref: node.name for node in abox.nodes}
    region_generations = {
        names[edge.start]: names[edge.end]
        for edge in abox.edges
        if edge.property_iri == INTRODUCED_IN
        and edge.start.class_iri == REGION
        and edge.end.class_iri == GENERATION
    }
    return AboxIndex(ids=ids, region_generations=region_generations)


def resolve_abox(index: AboxIndex, class_iri: str, label: str) -> NodeRecord:
    """Return the stub record of an A-Box node. It carries the A-Box id and name."""
    node_id = index.ids.get((class_iri, label))
    if node_id is None:
        raise AboxMissError(class_iri, label)
    return NodeRecord(class_iri=class_iri, id=node_id, name=label)


def require_abox(
    context: PageContext, class_iri: str, label: str, location: tuple[str, str]
) -> NodeRecord:
    """Resolve an A-Box node for a page; a miss fails the page. `location` is the section
    path and the field that named it."""
    try:
        return resolve_abox(context.abox, class_iri, label)
    except AboxMissError as error:
        section_path, field_name = location
        raise BatchBuildError(context.page_title, section_path, field_name, label) from error


def build_species_index(extracts: Iterable[SpeciesPageExtract]) -> SpeciesIndex:
    names: dict[str, str] = {}
    ids_by_name: dict[str, str] = {}
    for extract in extracts:
        for form in extract.forms:
            names[form.id] = form.name
        base = extract.forms[0]
        ids_by_name[base.name] = base.id
    return SpeciesIndex(names=names, ids_by_name=ids_by_name)


def resolve_species(index: SpeciesIndex, ref: SpeciesRef) -> tuple[str, bool] | None:
    """Return the id a species reference names and whether the base id stood in for it.

    `{base id}-{code}` is used when that node exists. A code that names no node is a
    cosmetic form, so the base id stands in. A name no page owns gives `None`.
    """
    base_id = index.ids_by_name.get(ref.name)
    if base_id is None:
        return None
    if not ref.code:
        return base_id, False
    form_id = f"{base_id}-{build_node_id(ref.code)}"
    if form_id in index.names:
        return form_id, False
    return base_id, True


def to_roman(number: int) -> str:
    numeral = ""
    for value, symbol in ROMAN_NUMERALS:
        count, number = divmod(number, value)
        numeral += symbol * count
    return numeral


def build_entry_id(species_id: str, move_id: str, method: LearnMethod, level: int | None) -> str:
    """Return `{species}:{move}:{method}[:{level}]:scarlet-violet`; only level rows have a level."""
    parts = [species_id, move_id, method.value]
    if method is LearnMethod.LEVEL and level is not None:
        parts.append(str(level))
    parts.append(build_node_id(SNAPSHOT_VERSION_GROUP))
    return ":".join(parts)


def build_entry_name(
    species_name: str, move_name: str, method: LearnMethod, level: int | None
) -> str:
    """Return e.g. "Diglett learns Earthquake (level 40, Scarlet-Violet)"."""
    how = method.value if level is None else f"{method.value} {level}"
    return f"{species_name} learns {move_name} ({how}, {SNAPSHOT_VERSION_GROUP})"


def build_machine_stub(tm_number: str) -> NodeRecord:
    """Return the Machine stub of a TM number: id `tm149-scarlet-violet`, name `TM149`."""
    machine_id = f"{tm_number.lower()}-{build_node_id(SNAPSHOT_VERSION_GROUP)}"
    return NodeRecord(class_iri=MACHINE, id=machine_id, name=tm_number)


def build_change_id(from_id: str, to_id: str) -> str:
    return f"{from_id}:{to_id}"


def build_change_name(from_name: str, to_name: str) -> str:
    return f"{from_name} to {to_name}"


def build_stub(class_iri: str, name: str) -> NodeRecord:
    """Return a stub for a node another page owns: id from the name, name as shown."""
    return NodeRecord(class_iri=class_iri, id=build_node_id(name), name=name)


def build_edge(
    property_iri: str,
    start: NodeRecord,
    end: NodeRecord,
    properties: Mapping[str, PropertyValue] | None = None,
) -> EdgeRecord:
    return EdgeRecord(
        property_iri=property_iri,
        start=start.node_ref,
        end=end.node_ref,
        properties=properties or {},
    )


def drop_none(values: Mapping[str, PropertyValue | None]) -> dict[str, PropertyValue]:
    return {key: value for key, value in values.items() if value is not None}


def read_base_stat_values(stats: BaseStats) -> dict[str, PropertyValue]:
    return {
        BASE_HP: stats.hp,
        BASE_ATTACK: stats.attack,
        BASE_DEFENSE: stats.defense,
        BASE_SP_ATTACK: stats.sp_attack,
        BASE_SP_DEFENSE: stats.sp_defense,
        BASE_SPEED: stats.speed,
    }


def build_species_record(form: FormFacts) -> NodeRecord:
    properties = drop_none(
        {
            POKEDEX_NUMBER: form.pokedex_number,
            IS_DEFAULT_FORM: form.is_default,
            HEIGHT: form.height,
            WEIGHT: form.weight,
            FEMALE_RATE: form.female_rate,
            IS_GENDERLESS: form.is_genderless,
            CATCH_RATE: form.catch_rate,
            BASE_EXP_YIELD: form.base_exp_yield,
            GROWTH_RATE: form.growth_rate,
            EGG_CYCLES: form.egg_cycles,
            BASE_FRIENDSHIP: form.base_friendship,
            COLOR: form.color,
            SHAPE: form.shape,
            GENUS: form.genus,
            IS_LEGENDARY: form.is_legendary,
            IS_MYTHICAL: form.is_mythical,
            IS_STARTER: form.is_starter,
            ALIASES: form.aliases or None,
            **read_base_stat_values(form.base_stats),
        }
    )
    return NodeRecord(class_iri=POKEMON_SPECIES, id=form.id, name=form.name, properties=properties)


def build_species_edges(form: FormFacts, species: NodeRecord, context: PageContext) -> Parts:
    """Build the A-Box, ability and generation edges of 1 species node."""
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    for slot, type_name in enumerate(form.types, start=1):
        type_stub = require_abox(context, TYPE, type_name, (INFOBOX_SECTION, f"type{slot}"))
        nodes.append(type_stub)
        edges.append(build_edge(HAS_TYPE, species, type_stub, {SLOT: slot}))
    for ability in form.abilities:
        ability_stub = build_stub(ABILITY, ability.name)
        nodes.append(ability_stub)
        edges.append(build_edge(HAS_ABILITY, species, ability_stub, {IS_HIDDEN: ability.is_hidden}))
    for group_name in form.egg_groups:
        group = require_abox(context, EGG_GROUP, group_name, (INFOBOX_SECTION, "egggroup"))
        nodes.append(group)
        edges.append(build_edge(IN_EGG_GROUP, species, group))
    for key, amount in form.ev_yield.items():
        stat = require_abox(context, STAT, EV_STAT_NAMES[key], (INFOBOX_SECTION, key))
        nodes.append(stat)
        edges.append(build_edge(YIELDS_EV, species, stat, {AMOUNT: amount}))
    return Parts(tuple(nodes), tuple(edges))


def read_generation_label(form: FormFacts, context: PageContext) -> str | None:
    """Return the label of the generation a node was introduced in: the base node's own, or
    the generation of a regional form's region."""
    if form.generation is not None:
        return GENERATION_LABEL.format(numeral=to_roman(form.generation))
    if form.region is not None:
        label = context.abox.region_generations.get(form.region)
        if label is None:
            raise BatchBuildError(context.page_title, INFOBOX_SECTION, REGION_PROPERTY, form.region)
        return label
    return None


def build_generation_parts(form: FormFacts, species: NodeRecord, context: PageContext) -> Parts:
    label = read_generation_label(form, context)
    if label is None:
        return Parts()
    generation = require_abox(context, GENERATION, label, (INFOBOX_SECTION, "generation"))
    return Parts((generation,), (build_edge(INTRODUCED_IN, species, generation),))


def build_form_edge(base: NodeRecord, form: FormFacts, species: NodeRecord) -> Parts:
    properties: dict[str, PropertyValue] = {}
    if form.region is not None:
        properties[REGION_PROPERTY] = form.region
    return Parts(edges=(build_edge(HAS_FORM, base, species, properties),))


def build_learnset_parts(
    extract: SpeciesPageExtract, records: Mapping[str, NodeRecord], context: PageContext
) -> Parts:
    """Build the learnset entries, their links and the `canLearn` shortcuts."""
    names = {form.id: form.name for form in extract.forms}
    version_group = require_abox(
        context, VERSION_GROUP, SNAPSHOT_VERSION_GROUP, (LEARNSET_SECTION, "version group")
    )
    nodes: list[NodeRecord] = [version_group] if extract.learnset else []
    edges: dict[tuple[str, str, str], EdgeRecord] = {}
    for row in extract.learnset:
        species = records[row.form_id]
        move = build_stub(MOVE, row.move)
        entry = build_entry_record(row, names[row.form_id], move)
        nodes.extend([move, entry])
        edges[(HAS_LEARNSET_ENTRY, species.id, entry.id)] = build_edge(
            HAS_LEARNSET_ENTRY, species, entry
        )
        edges[(TEACHES_MOVE, entry.id, move.id)] = build_edge(TEACHES_MOVE, entry, move)
        edges[(IN_VERSION_GROUP, entry.id, version_group.id)] = build_edge(
            IN_VERSION_GROUP, entry, version_group
        )
        edges[(CAN_LEARN, species.id, move.id)] = build_edge(CAN_LEARN, species, move)
        if row.machine is not None:
            machine = build_machine_stub(row.machine)
            nodes.append(machine)
            edges[(VIA_MACHINE, entry.id, machine.id)] = build_edge(VIA_MACHINE, entry, machine)
    return Parts(tuple(nodes), tuple(edges.values()))


def build_entry_record(row: LearnsetRow, species_name: str, move: NodeRecord) -> NodeRecord:
    properties: dict[str, PropertyValue] = {METHOD: row.method.value}
    if row.method is LearnMethod.LEVEL and row.level is not None:
        properties[LEVEL] = row.level
    level = row.level if row.method is LearnMethod.LEVEL else None
    return NodeRecord(
        class_iri=LEARNSET_ENTRY,
        id=build_entry_id(row.form_id, move.id, row.method, level),
        name=build_entry_name(species_name, move.name, row.method, level),
        properties=properties,
    )


def build_form_change_parts(change: FormChangeFact, records: Mapping[str, NodeRecord]) -> Parts:
    source = records[change.from_id]
    target = records[change.to_id]
    properties = drop_none(
        {
            FORM_KIND: change.form_kind,
            IS_BATTLE_ONLY: change.is_battle_only,
            TRIGGER: change.trigger,
        }
    )
    record = NodeRecord(
        class_iri=FORM_CHANGE,
        id=build_change_id(change.from_id, change.to_id),
        name=build_change_name(source.name, target.name),
        properties=properties,
    )
    item = build_stub(HELD_ITEM, change.held_item)
    return Parts(
        nodes=(record, item),
        edges=(
            build_edge(FROM_SPECIES, record, source),
            build_edge(TO_SPECIES, record, target),
            build_edge(REQUIRES, record, item, {ROLE: HOLD_ROLE}),
        ),
    )


def join_parts(parts: Iterable[Parts]) -> Parts:
    collected = list(parts)
    return Parts(
        nodes=tuple(node for part in collected for node in part.nodes),
        edges=tuple(edge for part in collected for edge in part.edges),
    )


def build_claims(anchors: Iterable[NodeRef], rules: Sequence[ClaimRule]) -> tuple[EdgeClaim, ...]:
    """Return every rule as a claim at every anchor, anchor by anchor."""
    return tuple(
        EdgeClaim(
            anchor=anchor,
            property_iri=rule.property_iri,
            direction=rule.direction,
            other_class_iri=rule.other_class_iri,
            owns_other_end=rule.owns_other_end,
        )
        for anchor in anchors
        for rule in rules
    )


def build_owner_claims(
    nodes: Iterable[NodeRecord], rules_by_class: Mapping[str, tuple[ClaimRule, ...]]
) -> tuple[EdgeClaim, ...]:
    """Claim, at each owner record, the rules of its class. A class with no rules claims nothing."""
    return tuple(
        claim
        for node in nodes
        for claim in build_claims([node.node_ref], rules_by_class.get(node.class_iri, ()))
    )


def list_owner_records(parts: Parts, scope: frozenset[str]) -> list[NodeRecord]:
    return [node for node in parts.nodes if node.class_iri in scope]


def assemble_batch(
    scope: frozenset[str], parts: Parts, claims: Iterable[EdgeClaim] = ()
) -> GraphBatch:
    """Build a sorted batch. A node or edge repeated across parts is kept once, and an owner
    record wins over a stub of the same key. A repeated claim is kept once."""
    nodes: dict[NodeRef, NodeRecord] = {}
    for node in parts.nodes:
        kept = nodes.get(node.node_ref)
        if kept is None or (node.class_iri in scope and kept.class_iri not in scope):
            nodes[node.node_ref] = node
    edges = {(e.property_iri, e.start, e.end): e for e in reversed(parts.edges)}
    claims_by_key = {claim.sort_key: claim for claim in reversed(tuple(claims))}
    return GraphBatch(
        universe=UNIVERSE,
        scope=scope,
        nodes=tuple(sorted(nodes.values(), key=lambda node: (node.class_iri, node.id))),
        edges=tuple(
            sorted(
                edges.values(),
                key=lambda e: (
                    e.property_iri,
                    e.start.class_iri,
                    e.start.id,
                    e.end.class_iri,
                    e.end.id,
                ),
            )
        ),
        claims=tuple(claims_by_key[key] for key in sorted(claims_by_key)),
    )


def build_species_batch(extract: SpeciesPageExtract, abox: AboxIndex) -> GraphBatch:
    """Build the pass-1 batch of a page. Raises `BatchBuildError` for a name the A-Box lacks."""
    context = PageContext(extract.title, abox)
    records = {form.id: build_species_record(form) for form in extract.forms}
    base = records[extract.forms[0].id]
    parts: list[Parts] = [Parts(nodes=tuple(records.values()))]
    for form in extract.forms:
        species = records[form.id]
        parts.append(build_species_edges(form, species, context))
        parts.append(build_generation_parts(form, species, context))
        if not form.is_default:
            parts.append(build_form_edge(base, form, species))
    parts.append(build_learnset_parts(extract, records, context))
    parts.extend(build_form_change_parts(change, records) for change in extract.form_changes)
    joined = join_parts(parts)
    owner_claims = build_owner_claims(
        list_owner_records(joined, SPECIES_SCOPE), SPECIES_OWNER_CLAIMS
    )
    return assemble_batch(SPECIES_SCOPE, joined, owner_claims)


def resolve_ends(
    fact: EvolutionFact, context: PageContext
) -> tuple[str, str, list[PageNote]] | None:
    """Resolve the source and target of an evolution to species ids.

    Returns `None` when either is unknown. A base id that stands in for a cosmetic form is
    noted.
    """
    notes: list[PageNote] = []
    ids: list[str] = []
    for ref in (fact.source, fact.target):
        resolved = resolve_species(context.species, ref)
        if resolved is None:
            return None
        ids.append(resolved[0])
        if resolved[1]:
            notes.append(build_fallback_note(fact, ref, resolved[0]))
    return ids[0], ids[1], notes


def build_fallback_note(fact: EvolutionFact, ref: SpeciesRef, species_id: str) -> PageNote:
    return PageNote(
        kind=NoteKind.REFERENCE_FALLBACK,
        section_path=fact.section_path,
        detail=f"{ref.name} with code {ref.code!r} resolves to {species_id}",
    )


def build_drop_note(fact: EvolutionFact, reason: str) -> PageNote:
    return PageNote(
        kind=NoteKind.DROPPED_VALUE,
        section_path=fact.section_path,
        detail=f"evolution {fact.source.name} to {fact.target.name} dropped: {reason}",
    )


def build_species_stub(context: PageContext, species_id: str) -> NodeRecord:
    return NodeRecord(
        class_iri=POKEMON_SPECIES, id=species_id, name=context.species.names[species_id]
    )


def build_page_anchors(extract: SpeciesPageExtract, context: PageContext) -> Parts:
    """Return a species stub of every form of the page, the anchors of passes 2 and 3."""
    return Parts(nodes=tuple(build_species_stub(context, form.id) for form in extract.forms))


def build_requirement_parts(
    requirement: Requirement, evolution: NodeRecord, context: PageContext
) -> Parts | None:
    """Build the `requires` edge of 1 requirement. A species no page owns gives `None`."""
    class_iri = REQUIREMENT_CLASSES[requirement.role]
    if requirement.species is not None:
        resolved = resolve_species(context.species, requirement.species)
        if resolved is None:
            return None
        stub = build_species_stub(context, resolved[0])
    elif class_iri in A_BOX_REQUIREMENT_CLASSES:
        location = (EVOLUTION_SECTION, requirement.role.value)
        stub = require_abox(context, class_iri, requirement.name, location)
    else:
        stub = build_stub(class_iri, requirement.name)
    edge = build_edge(REQUIRES, evolution, stub, {ROLE: requirement.role.value})
    return Parts((stub,), (edge,))


def build_evolution_parts(
    fact: EvolutionFact, ends: tuple[str, str], context: PageContext
) -> Parts | None:
    """Build the evolution node with its species and requirement edges, or `None` when a
    required species is unknown."""
    source_id, target_id = ends
    source = build_species_stub(context, source_id)
    target = build_species_stub(context, target_id)
    evolution = NodeRecord(
        class_iri=EVOLUTION,
        id=build_change_id(source_id, target_id),
        name=build_change_name(source.name, target.name),
        properties=drop_none(
            {
                EVO_TRIGGER: fact.trigger,
                MIN_LEVEL: fact.min_level,
                TIME_OF_DAY: fact.time_of_day,
                GENDER: fact.gender,
                CONDITION: fact.condition,
            }
        ),
    )
    requirement_parts = [
        build_requirement_parts(requirement, evolution, context)
        for requirement in fact.requirements
    ]
    if any(part is None for part in requirement_parts):
        return None
    return join_parts(
        [
            Parts(
                nodes=(evolution, source, target),
                edges=(
                    build_edge(FROM_SPECIES, evolution, source),
                    build_edge(TO_SPECIES, evolution, target),
                    build_edge(EVOLVES_TO, source, target),
                ),
            ),
            *(part for part in requirement_parts if part is not None),
        ]
    )


def is_same_evolution(first: EvolutionFact, second: EvolutionFact) -> bool:
    """Whether 2 facts give the same method. Their species references and sections do not
    count: boxes of cosmetic forms resolve to the same ids and differ only there."""
    return (
        first.trigger,
        first.min_level,
        first.time_of_day,
        first.gender,
        first.requirements,
        first.condition,
    ) == (
        second.trigger,
        second.min_level,
        second.time_of_day,
        second.gender,
        second.requirements,
        second.condition,
    )


def build_evolution_batch(
    extract: SpeciesPageExtract, abox: AboxIndex, index: SpeciesIndex
) -> EvolutionBuild:
    """Build the pass-2 batch of the evolutions a page owns.

    A page owns an evolution when its source resolves to a node of the page; the boxes for
    other pages' species are cross-checks and are skipped here. An evolution with an
    unknown species is dropped and noted.
    """
    context = PageContext(extract.title, abox, index)
    own_ids = {form.id for form in extract.forms}
    parts: list[Parts] = []
    notes: list[PageNote] = []
    seen: dict[str, EvolutionFact] = {}
    for fact in extract.evolutions:
        ends = resolve_ends(fact, context)
        if ends is not None and ends[0] not in own_ids:
            continue
        if ends is None:
            notes.append(build_drop_note(fact, "a species resolves to no id"))
            continue
        evolution_id = build_change_id(ends[0], ends[1])
        if evolution_id in seen:
            if not is_same_evolution(seen[evolution_id], fact):
                notes.append(build_drop_note(fact, "another box gave this pair a different method"))
            continue
        built = build_evolution_parts(fact, (ends[0], ends[1]), context)
        if built is None:
            notes.append(build_drop_note(fact, "a required species resolves to no id"))
            continue
        seen[evolution_id] = fact
        parts.append(built)
        notes.extend(ends[2])
    structured = sum(fact.condition is None for fact in seen.values())
    anchors = build_page_anchors(extract, context)
    joined = join_parts([*parts, anchors])
    claims = (
        *build_claims([anchor.node_ref for anchor in anchors.nodes], EVOLUTION_ANCHOR_CLAIMS),
        *build_owner_claims(list_owner_records(joined, EVOLUTION_SCOPE), EVOLUTION_OWNER_CLAIMS),
    )
    batch = assemble_batch(EVOLUTION_SCOPE, joined, claims)
    return EvolutionBuild(batch=batch, notes=tuple(notes), structured=structured, total=len(seen))


class ChunkBuildError(Exception):
    """2 different Pokédex texts of 1 subject share a chunk id. The message holds both."""

    def __init__(self, page_title: str, digest: str, first: str, second: str) -> None:
        super().__init__(
            f"{page_title}: Pokédex texts share the hash {digest}: {first!r}, {second!r}"
        )
        self.page_title = page_title
        self.digest = digest


@dataclass(frozen=True)
class ChunkInputs:
    abox: AboxIndex
    species: SpeciesIndex
    source_url: str


@dataclass(frozen=True)
class ChunkBuild:
    """The pass-3 batch of a page (with no owner record when it holds no chunk) and the notes."""

    batch: GraphBatch
    notes: tuple[PageNote, ...]


def hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:TEXT_HASH_LENGTH]


def number_chunks(page_title: str, facts: Sequence[ChunkFact]) -> tuple[tuple[ChunkFact, str], ...]:
    """Pair each fact with the last segment of its id.

    The segment counts per (subject, aspect) from 1 in page order, and for `pokedex` it is
    the hash of the text. Raises `ChunkBuildError` when 2 texts of 1 subject share a hash.
    """
    counts: Counter[tuple[str, ChunkAspect]] = Counter()
    texts_by_hash: dict[tuple[str, str], str] = {}
    numbered: list[tuple[ChunkFact, str]] = []
    for fact in facts:
        if fact.aspect is ChunkAspect.POKEDEX:
            digest = hash_chunk_text(fact.text)
            earlier = texts_by_hash.setdefault((fact.subject_id, digest), fact.text)
            if earlier != fact.text:
                raise ChunkBuildError(page_title, digest, earlier, fact.text)
            numbered.append((fact, digest))
            continue
        counts[(fact.subject_id, fact.aspect)] += 1
        numbered.append((fact, str(counts[(fact.subject_id, fact.aspect)])))
    return tuple(numbered)


def build_chunk_record(
    fact: ChunkFact, label: str, subject: NodeRecord, source_url: str
) -> NodeRecord:
    """Build the chunk node. Pokédex chunks name their origin through `fromWork` and carry
    no `source`."""
    chunk_id = CHUNK_ID_TEMPLATE.format(
        source=CHUNK_SOURCE,
        label=extract_local_name(subject.class_iri).lower(),
        subject=subject.id,
        aspect=fact.aspect.value,
        n=label,
    )
    properties: dict[str, PropertyValue] = {
        TEXT: fact.text,
        ASPECT: fact.aspect.value,
        SECTION: fact.section_path,
        SOURCE_URL: source_url,
    }
    if fact.aspect is not ChunkAspect.POKEDEX:
        properties[SOURCE] = CHUNK_SOURCE
    name = CHUNK_NAME_TEMPLATE.format(subject=subject.name, aspect=fact.aspect.value, n=label)
    return NodeRecord(class_iri=TEXT_CHUNK, id=chunk_id, name=name, properties=properties)


def resolve_mention(mention: MentionRef, context: PageContext) -> NodeRecord | None:
    """Return the stub that a mention names, or `None` when nothing owns the target.

    A species goes through the species index and a type through the A-Box. A move or an
    ability is trusted as written, since its page may not be in the run yet.
    """
    if mention.template == SPECIES_MENTION:
        resolved = resolve_species(context.species, SpeciesRef(name=mention.target, code=""))
        return None if resolved is None else build_species_stub(context, resolved[0])
    class_iri = MENTION_CLASSES[mention.template]
    if class_iri == TYPE:
        try:
            return resolve_abox(context.abox, class_iri, mention.target)
        except AboxMissError:
            return None
    return build_stub(class_iri, mention.target) if build_node_id(mention.target) else None


def build_unresolved_note(mention: MentionRef, section_path: str) -> PageNote:
    noun = MENTION_NOUNS[mention.template]
    return PageNote(
        kind=NoteKind.UNRESOLVED_MENTION,
        section_path=section_path,
        detail="{{" + f"{mention.template}|{mention.target}" + "}} names no " + noun,
    )


def build_mention_parts(
    fact: ChunkFact, chunk: NodeRecord, subject: NodeRecord, context: PageContext
) -> tuple[Parts, tuple[PageNote, ...]]:
    """Build the `mentionedIn` edges of a chunk. The subject never mentions itself."""
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    notes: list[PageNote] = []
    for mention in fact.mentions:
        entity = resolve_mention(mention, context)
        if entity is None:
            notes.append(build_unresolved_note(mention, fact.section_path))
        elif entity.node_ref != subject.node_ref:
            nodes.append(entity)
            edges.append(build_edge(MENTIONED_IN, entity, chunk))
    return Parts(tuple(nodes), tuple(edges)), tuple(notes)


def build_version_parts(
    fact: ChunkFact, chunk: NodeRecord, context: PageContext
) -> tuple[Parts, tuple[PageNote, ...]]:
    """Build the `fromWork` edges of a Pokédex chunk. A version the A-Box lacks is dropped
    with a note, and the notes say so when no version is left."""
    nodes: list[NodeRecord] = []
    edges: list[EdgeRecord] = []
    missing: list[str] = []
    for name in fact.versions:
        try:
            version = resolve_abox(context.abox, VERSION, name)
        except AboxMissError:
            missing.append(name)
            continue
        nodes.append(version)
        edges.append(build_edge(FROM_WORK, chunk, version))
    suffix = "" if edges else ROW_DROPPED_SUFFIX
    notes = tuple(
        PageNote(
            kind=NoteKind.UNKNOWN_VERSION,
            section_path=fact.section_path,
            detail=f"{name} is no A-Box Version{suffix}",
        )
        for name in missing
    )
    return Parts(tuple(nodes), tuple(edges)), notes


def build_chunk_batch(extract: SpeciesPageExtract, inputs: ChunkInputs) -> ChunkBuild:
    """Build the pass-3 batch of a page. Raises `ChunkBuildError` for a chunk id collision.

    A Pokédex chunk with no version the A-Box knows is dropped.
    """
    context = PageContext(extract.title, inputs.abox, inputs.species)
    parts: list[Parts] = []
    notes: list[PageNote] = []
    for fact, label in number_chunks(extract.title, extract.chunks):
        subject = build_species_stub(context, fact.subject_id)
        chunk = build_chunk_record(fact, label, subject, inputs.source_url)
        chunk_parts = [Parts((chunk, subject), (build_edge(DESCRIBES, chunk, subject),))]
        if fact.aspect is ChunkAspect.POKEDEX:
            version_parts, version_notes = build_version_parts(fact, chunk, context)
            notes.extend(version_notes)
            if not version_parts.edges:
                continue
            chunk_parts.append(version_parts)
        mention_parts, mention_notes = build_mention_parts(fact, chunk, subject, context)
        notes.extend(mention_notes)
        parts.extend([*chunk_parts, mention_parts])
    anchors = build_page_anchors(extract, context)
    joined = join_parts([*parts, anchors])
    claims = (
        *build_claims([anchor.node_ref for anchor in anchors.nodes], CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(list_owner_records(joined, CHUNK_SCOPE), CHUNK_OWNER_CLAIMS),
    )
    return ChunkBuild(batch=assemble_batch(CHUNK_SCOPE, joined, claims), notes=tuple(notes))
