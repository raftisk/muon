"""Batch builders: a species page extract becomes its pass-1 and pass-2 `GraphBatch`.

Pass 1 (`build_species_batch`) owns the species nodes, their learnset entries and the Mega
and Primal form changes, which all belong to 1 page. Pass 2 (`build_evolution_batch`) owns
the evolutions whose from-species is a node of the page; they point at species of other
pages, so they wait until every species batch is written. Names that the hand-written
A-Box owns resolve through `AboxIndex`, and a miss fails the page. Evolution references
resolve through `SpeciesIndex`, and a miss drops the evolution; neither creates a stub for
an id nothing owns.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from muon.ingestion.batch import (
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
    EvolutionFact,
    FormChangeFact,
    FormFacts,
    LearnMethod,
    LearnsetRow,
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

SPECIES_SCOPE = frozenset({POKEMON_SPECIES, LEARNSET_ENTRY, FORM_CHANGE})
EVOLUTION_SCOPE = frozenset({EVOLUTION})
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
    """The pass-2 batch of a page (`None` when it owns no evolution), what was dropped or
    substituted, and how many of its evolutions a rule structured."""

    batch: GraphBatch | None
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


def assemble_batch(scope: frozenset[str], parts: Parts) -> GraphBatch:
    """Build a sorted batch. A node or edge repeated across parts is kept once, and an owner
    record wins over a stub of the same key."""
    nodes: dict[NodeRef, NodeRecord] = {}
    for node in parts.nodes:
        kept = nodes.get(node.node_ref)
        if kept is None or (node.class_iri in scope and kept.class_iri not in scope):
            nodes[node.node_ref] = node
    edges = {(e.property_iri, e.start, e.end): e for e in reversed(parts.edges)}
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
    return assemble_batch(SPECIES_SCOPE, join_parts(parts))


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
    if not parts:
        return EvolutionBuild(batch=None, notes=tuple(notes), structured=0, total=0)
    structured = sum(fact.condition is None for fact in seen.values())
    batch = assemble_batch(EVOLUTION_SCOPE, join_parts(parts))
    return EvolutionBuild(batch=batch, notes=tuple(notes), structured=structured, total=len(seen))
