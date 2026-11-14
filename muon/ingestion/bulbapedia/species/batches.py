"""Batch builders: a species page extract becomes its pass 1 to pass 4 `GraphBatch`.

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

Pass 4 (`build_encounter_batch`) owns the encounters of the page: 1 Encounter node per
species, location, version and method, with its `atLocation` and `inVersion` edges, and 1
`foundIn` edge per species and location. Locations are stubs the location page run owns;
versions resolve through the A-Box.

Each batch also claims the edge patterns its page owns, so a re-ingest deletes what the
page no longer gives. The claim tables below say which patterns each pass claims. The
edges that a node id fixes (`teachesMove`, `inVersionGroup`, `toSpecies`, `describes`
of a chunk and the `fromSpecies` of a change) are not claimed, since they cannot go
stale. Passes 2 and 3 anchor their claims at a stub of every form of the page, and a
page that gives no evolution or no chunk still gets that batch, with the anchors, their
claims and no owner record, so a re-ingest deletes what the page gave before. Pass 4
anchors its claims the same way.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from muon.ingestion.batch import (
    EdgeRecord,
    GraphBatch,
    NodeRecord,
    PropertyValue,
    build_node_id,
)
from muon.ingestion.bulbapedia.batching import (
    ABILITY,
    ALIASES,
    DESCRIBES,
    FOUND_IN,
    FROM_WORK,
    GENERATION,
    GENERATION_LABEL,
    HAS_TYPE,
    IN,
    IN_VERSION_GROUP,
    INTRODUCED_IN,
    LOCATION,
    MACHINE,
    MENTIONED_IN,
    MOVE,
    OUT,
    PKMN,
    TEACHES_MOVE,
    TEXT_CHUNK,
    TYPE,
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
    build_encounter_id,
    build_machine_id,
    build_mention_parts,
    build_owner_claims,
    build_stub,
    build_work_parts,
    describe_unresolved,
    drop_none,
    join_parts,
    list_owner_records,
    number_chunks,
    require_abox,
    resolve_abox,
    to_roman,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact, MentionRef
from muon.ingestion.bulbapedia.species.evolution import EVOLUTION_PATH
from muon.ingestion.bulbapedia.species.extract import (
    BaseStats,
    EncounterFact,
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

POKEMON_SPECIES = PKMN + "PokemonSpecies"
LEARNSET_ENTRY = PKMN + "LearnsetEntry"
FORM_CHANGE = PKMN + "FormChange"
EVOLUTION = PKMN + "Evolution"
EGG_GROUP = PKMN + "EggGroup"
STAT = PKMN + "Stat"
HELD_ITEM = PKMN + "HeldItem"
ITEM = PKMN + "Item"
VERSION = PKMN + "Version"
ENCOUNTER = PKMN + "Encounter"

HAS_ABILITY = PKMN + "hasAbility"
IN_EGG_GROUP = PKMN + "inEggGroup"
YIELDS_EV = PKMN + "yieldsEv"
HAS_FORM = PKMN + "hasForm"
HAS_LEARNSET_ENTRY = PKMN + "hasLearnsetEntry"
VIA_MACHINE = PKMN + "viaMachine"
CAN_LEARN = PKMN + "canLearn"
FROM_SPECIES = PKMN + "fromSpecies"
TO_SPECIES = PKMN + "toSpecies"
REQUIRES = PKMN + "requires"
EVOLVES_TO = PKMN + "evolvesTo"
HAS_ENCOUNTER = PKMN + "hasEncounter"
AT_LOCATION = PKMN + "atLocation"
IN_VERSION = PKMN + "inVersion"

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
ENCOUNTER_KIND = PKMN + "encounterKind"

SPECIES_SCOPE = frozenset({POKEMON_SPECIES, LEARNSET_ENTRY, FORM_CHANGE})
EVOLUTION_SCOPE = frozenset({EVOLUTION})
CHUNK_SCOPE = frozenset({TEXT_CHUNK})
ENCOUNTER_SCOPE = frozenset({ENCOUNTER})
ROW_DROPPED_SUFFIX = "; row dropped"
SNAPSHOT_VERSION_GROUP = "Scarlet-Violet"
MACHINE_KIND_LENGTH = 2  # TM, HM and TR

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
# `atLocation` and `inVersion` are not claimed: the Encounter id fixes both ends.
ENCOUNTER_ANCHOR_CLAIMS: tuple[ClaimRule, ...] = (
    ClaimRule(HAS_ENCOUNTER, OUT, ENCOUNTER, True),
    ClaimRule(FOUND_IN, OUT, LOCATION, False),
)
CHUNK_OWNER_CLAIMS: Mapping[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (
        ClaimRule(FROM_WORK, OUT, VERSION, False),
        ClaimRule(MENTIONED_IN, IN, None, False),
    ),
}


@dataclass(frozen=True)
class SpeciesIndex:
    """Every species node id of a run with its name, and each base species' id by name."""

    names: Mapping[str, str]
    ids_by_name: Mapping[str, str]


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
    kind, number_text = tm_number[:MACHINE_KIND_LENGTH], tm_number[MACHINE_KIND_LENGTH:]
    machine_id = build_machine_id(kind, number_text, build_node_id(SNAPSHOT_VERSION_GROUP))
    return NodeRecord(class_iri=MACHINE, id=machine_id, name=tm_number)


def build_change_id(from_id: str, to_id: str) -> str:
    return f"{from_id}:{to_id}"


def build_change_name(from_name: str, to_name: str) -> str:
    return f"{from_name} to {to_name}"


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


def build_unresolved_note(mention: MentionRef, section_path: str) -> PageNote:
    return PageNote(
        kind=NoteKind.UNRESOLVED_MENTION,
        section_path=section_path,
        detail=describe_unresolved(mention),
    )


def build_version_notes(fact: ChunkFact, missing: Sequence[str], has_edges: bool) -> list[PageNote]:
    """Note each version of a Pokédex chunk the A-Box lacks, and say so when the row drops."""
    suffix = "" if has_edges else ROW_DROPPED_SUFFIX
    return [
        PageNote(
            kind=NoteKind.UNKNOWN_VERSION,
            section_path=fact.section_path,
            detail=f"{name} is no A-Box Version{suffix}",
        )
        for name in missing
    ]


def build_species_resolver(context: PageContext) -> SpeciesResolver:
    """Resolve a `{{p}}` target to the stub of its base species through the species index."""

    def resolve(target: str) -> NodeRecord | None:
        resolved = resolve_species(context.species, SpeciesRef(name=target, code=""))
        return None if resolved is None else build_species_stub(context, resolved[0])

    return resolve


def build_chunk_batch(extract: SpeciesPageExtract, inputs: ChunkInputs) -> ChunkBuild:
    """Build the pass-3 batch of a page. Raises `ChunkBuildError` for a chunk id collision.

    A Pokédex chunk with no version the A-Box knows is dropped.
    """
    context = PageContext(extract.title, inputs.abox, inputs.species)
    mentions = MentionContext(inputs.abox, build_species_resolver(context))
    parts: list[Parts] = []
    notes: list[PageNote] = []
    for fact, label in number_chunks(extract.title, extract.chunks):
        subject = build_species_stub(context, fact.subject_id)
        chunk = build_chunk_record(fact, label, subject, inputs.source_url)
        chunk_parts = [Parts((chunk, subject), (build_edge(DESCRIBES, chunk, subject),))]
        if fact.aspect is ChunkAspect.POKEDEX:
            version_parts, missing = build_work_parts(fact, chunk, inputs.abox, VERSION)
            notes.extend(build_version_notes(fact, missing, bool(version_parts.edges)))
            if not version_parts.edges:
                continue
            chunk_parts.append(version_parts)
        mention_parts, unresolved = build_mention_parts(fact, chunk, subject, mentions)
        notes.extend(build_unresolved_note(mention, fact.section_path) for mention in unresolved)
        parts.extend([*chunk_parts, mention_parts])
    anchors = build_page_anchors(extract, context)
    joined = join_parts([*parts, anchors])
    claims = (
        *build_claims([anchor.node_ref for anchor in anchors.nodes], CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(list_owner_records(joined, CHUNK_SCOPE), CHUNK_OWNER_CLAIMS),
    )
    return ChunkBuild(batch=assemble_batch(CHUNK_SCOPE, joined, claims), notes=tuple(notes))


def build_encounter_name(
    species_name: str, location_name: str, version: str, method: str | None
) -> str:
    """Return e.g. `Diglett in Vermilion City (HeartGold, rock-smash)`."""
    details = ", ".join((version, *((method,) if method else ())))
    return f"{species_name} in {location_name} ({details})"


def build_encounter_parts(fact: EncounterFact, species: NodeRecord, abox: AboxIndex) -> Parts:
    location = build_stub(LOCATION, fact.location_title)
    version = resolve_abox(abox, VERSION, fact.version)
    encounter = NodeRecord(
        class_iri=ENCOUNTER,
        id=build_encounter_id(species.id, location.id, version.id, fact.method),
        name=build_encounter_name(species.name, location.name, version.name, fact.method),
        properties=drop_none({ENCOUNTER_KIND: fact.kind, METHOD: fact.method}),
    )
    return Parts(
        nodes=(encounter, location, version),
        edges=(
            build_edge(HAS_ENCOUNTER, species, encounter),
            build_edge(AT_LOCATION, encounter, location),
            build_edge(IN_VERSION, encounter, version),
            build_edge(FOUND_IN, species, location),
        ),
    )


def build_encounter_batch(
    extract: SpeciesPageExtract, facts: Sequence[EncounterFact], abox: AboxIndex
) -> GraphBatch:
    """Build the pass 4 batch of a page. A page with no fact gives the anchors and their
    claims only. Every fact version is an A-Box Version: the run checks the table rows
    before any page is read."""
    context = PageContext(extract.title, abox, build_species_index([extract]))
    anchors = build_page_anchors(extract, context)
    species = {anchor.id: anchor for anchor in anchors.nodes}
    joined = join_parts(
        [*(build_encounter_parts(fact, species[fact.species_id], abox) for fact in facts), anchors]
    )
    claims = build_claims([anchor.node_ref for anchor in anchors.nodes], ENCOUNTER_ANCHOR_CLAIMS)
    return assemble_batch(ENCOUNTER_SCOPE, joined, claims)
