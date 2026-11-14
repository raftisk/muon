"""1 move page extract to its pass 1 and pass 2 batches.

Pass 1 owns the Move and its Machines (scope {`Move`, `Machine`}). The Type, MoveDamageClass,
MoveFlags, Generation and VersionGroups the move points at are A-Box stubs, resolved by
label so a missing name fails the page. The batch claims `hasType`, `hasDamageClass`,
`hasFlag` and `introducedIn` out of the Move and `teachesMove` in from a Machine, owning the
Machine, so an upsert prunes edges and machines the page no longer names. `inVersionGroup`
is not claimed: the Machine id fixes its version group.

Pass 2 owns the TextChunks of the page (scope {`TextChunk`}). Every chunk `describes` the
Move, a flavor chunk links `fromWork` to its VersionGroups, and a mention links in through
`mentionedIn`. The Move stub anchors the `describes` claim, so a page with no chunk left
still removes the chunks it had.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from muon.ingestion.batch import GraphBatch, NodeRecord, PropertyValue
from muon.ingestion.bulbapedia.batching import (
    ALIASES,
    CORE,
    DESCRIBES,
    FROM_WORK,
    GENERATION,
    HAS_TYPE,
    IN,
    IN_VERSION_GROUP,
    INTRODUCED_IN,
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
    to_roman,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect, ChunkFact
from muon.ingestion.bulbapedia.moves.chunks import GAME_CODES
from muon.ingestion.bulbapedia.moves.page import (
    DAMAGE_CLASS_KEY,
    GENERATION_KEY,
    INFOBOX_LABEL,
    MACHINE_FIELDS,
    TYPE_KEY,
    MachineFact,
    MoveNote,
    MoveNoteKind,
    MovePageExtract,
)

MOVE_DAMAGE_CLASS = PKMN + "MoveDamageClass"
MOVE_FLAG = PKMN + "MoveFlag"
HAS_DAMAGE_CLASS = PKMN + "hasDamageClass"
HAS_FLAG = PKMN + "hasFlag"
PART_OF = CORE + "partOf"

POWER = PKMN + "power"
ACCURACY = PKMN + "accuracy"
PP = PKMN + "pp"
PRIORITY = PKMN + "priority"
TARGET = PKMN + "target"
CRIT_STAGE = PKMN + "critStage"
MIN_HITS = PKMN + "minHits"
MAX_HITS = PKMN + "maxHits"
MIN_TURNS = PKMN + "minTurns"
MAX_TURNS = PKMN + "maxTurns"
IS_PROTECTABLE = PKMN + "isProtectable"
IS_MIRRORABLE = PKMN + "isMirrorable"
EFFECT_TEXT = PKMN + "effectText"
MACHINE_KIND = PKMN + "machineKind"
MACHINE_NUMBER = PKMN + "machineNumber"

MOVE_SCOPE = frozenset({MOVE, MACHINE})
FLAGS_FIELD = "flags"
GENERATION_FIELD_PATTERN = re.compile(r"^(?:tm|hm|tr)(\d)$")

GAME_CODE_FIELD = "game code"

MOVE_CHUNK_SCOPE = frozenset({TEXT_CHUNK})
MOVE_CHUNK_ANCHOR_CLAIMS = (ClaimRule(DESCRIBES, IN, TEXT_CHUNK, owns_other_end=True),)
MOVE_CHUNK_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (
        ClaimRule(FROM_WORK, OUT, VERSION_GROUP, owns_other_end=False),
        ClaimRule(MENTIONED_IN, IN, None, owns_other_end=False),
    ),
}

MOVE_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    MOVE: (
        ClaimRule(HAS_TYPE, OUT, TYPE, owns_other_end=False),
        ClaimRule(HAS_DAMAGE_CLASS, OUT, MOVE_DAMAGE_CLASS, owns_other_end=False),
        ClaimRule(HAS_FLAG, OUT, MOVE_FLAG, owns_other_end=False),
        ClaimRule(INTRODUCED_IN, OUT, GENERATION, owns_other_end=False),
        ClaimRule(TEACHES_MOVE, IN, MACHINE, owns_other_end=True),
    ),
}


class MachineFieldError(Exception):
    """`MACHINE_FIELDS` names a version group the A-Box lacks or puts in another generation."""


class GameCodeError(Exception):
    """`GAME_CODES` names a version group the A-Box lacks."""


@dataclass(frozen=True)
class MoveChunkInputs:
    """What pass 2 resolves against, and the permalink of the page its chunks cite."""

    abox: AboxIndex
    resolve_species: SpeciesResolver
    source_url: str


@dataclass(frozen=True)
class MoveChunkBuild:
    batch: GraphBatch
    notes: tuple[MoveNote, ...]


@dataclass(frozen=True)
class MoveContext:
    """What `require_abox` needs to name a failed move page."""

    page_title: str
    abox: AboxIndex


def build_move_record(extract: MovePageExtract, effect_text: str | None) -> NodeRecord:
    """Return the Move owner record, without the properties that have no value."""
    properties: dict[str, PropertyValue | None] = {
        POWER: extract.power,
        ACCURACY: extract.accuracy,
        PP: extract.pp,
        PRIORITY: extract.priority,
        TARGET: extract.target,
        CRIT_STAGE: extract.crit_stage,
        MIN_HITS: extract.min_hits,
        MAX_HITS: extract.max_hits,
        MIN_TURNS: extract.min_turns,
        MAX_TURNS: extract.max_turns,
        IS_PROTECTABLE: extract.is_protectable,
        IS_MIRRORABLE: extract.is_mirrorable,
        EFFECT_TEXT: effect_text,
        ALIASES: extract.aliases or None,
    }
    return NodeRecord(
        class_iri=MOVE, id=extract.id, name=extract.name, properties=drop_none(properties)
    )


def build_move_edges(extract: MovePageExtract, move: NodeRecord, context: MoveContext) -> Parts:
    """Build the A-Box stubs and the type, damage class, flag and generation edges."""
    targets = [
        (HAS_TYPE, require_abox(context, TYPE, extract.type_name, (INFOBOX_LABEL, TYPE_KEY))),
        (
            HAS_DAMAGE_CLASS,
            require_abox(
                context,
                MOVE_DAMAGE_CLASS,
                extract.damage_class_name,
                (INFOBOX_LABEL, DAMAGE_CLASS_KEY),
            ),
        ),
        (
            INTRODUCED_IN,
            require_abox(
                context, GENERATION, extract.generation_label, (INFOBOX_LABEL, GENERATION_KEY)
            ),
        ),
        *(
            (HAS_FLAG, require_abox(context, MOVE_FLAG, flag, (INFOBOX_LABEL, FLAGS_FIELD)))
            for flag in extract.flags
        ),
    ]
    return Parts(
        nodes=tuple(stub for _, stub in targets),
        edges=tuple(build_edge(property_iri, move, stub) for property_iri, stub in targets),
    )


def build_machine_parts(extract: MovePageExtract, move: NodeRecord, context: MoveContext) -> Parts:
    """Build 1 Machine owner per machine fact, with `teachesMove` and `inVersionGroup`."""
    return join_parts(build_machine(fact, move, context) for fact in extract.machines)


def build_machine(fact: MachineFact, move: NodeRecord, context: MoveContext) -> Parts:
    group = require_abox(
        context, VERSION_GROUP, fact.version_group_label, (INFOBOX_LABEL, fact.field)
    )
    machine = NodeRecord(
        class_iri=MACHINE,
        id=build_machine_id(fact.kind, fact.number_text, group.id),
        name=f"{fact.kind}{fact.number_text}",
        properties={MACHINE_KIND: fact.kind, MACHINE_NUMBER: int(fact.number_text)},
    )
    return Parts(
        nodes=(machine, group),
        edges=(
            build_edge(TEACHES_MOVE, machine, move),
            build_edge(IN_VERSION_GROUP, machine, group),
        ),
    )


def build_move_batch(
    extract: MovePageExtract, abox: AboxIndex, effect_text: str | None
) -> GraphBatch:
    """Build the pass 1 batch of 1 page. A name the A-Box lacks raises `BatchBuildError`."""
    context = MoveContext(extract.title, abox)
    move = build_move_record(extract, effect_text)
    parts = join_parts(
        [
            Parts(nodes=(move,)),
            build_move_edges(extract, move, context),
            build_machine_parts(extract, move, context),
        ]
    )
    owners = list_owner_records(parts, MOVE_SCOPE)
    return assemble_batch(MOVE_SCOPE, parts, build_owner_claims(owners, MOVE_OWNER_CLAIMS))


def check_machine_fields(seed: GraphBatch) -> None:
    """Fail unless every `MACHINE_FIELDS` label is an A-Box VersionGroup and the groups of
    a generation field (`tm4`, `tr8`) are `partOf` that generation."""
    groups = {node.name: node for node in seed.nodes if node.class_iri == VERSION_GROUP}
    names = {node.node_ref: node.name for node in seed.nodes}
    generations: defaultdict[str, set[str]] = defaultdict(set)
    for edge in seed.edges:
        if edge.property_iri == PART_OF and edge.start.class_iri == VERSION_GROUP:
            generations[names[edge.start]].add(names[edge.end])
    problems: list[str] = []
    for field, spec in MACHINE_FIELDS.items():
        match = GENERATION_FIELD_PATTERN.match(field)
        expected = None if match is None else f"Generation {to_roman(int(match.group(1)))}"
        problems.extend(
            f"{field} names {label!r}, which is no A-Box VersionGroup"
            for label in spec.version_group_labels
            if label not in groups
        )
        problems.extend(
            f"{field} names {label!r}, which is not part of {expected}"
            for label in spec.version_group_labels
            if expected is not None and label in groups and expected not in generations[label]
        )
    if problems:
        raise MachineFieldError("; ".join(problems))


def build_flavor_parts(
    extract: MovePageExtract, fact: ChunkFact, chunk: NodeRecord, abox: AboxIndex
) -> Parts:
    """Build the `fromWork` edges of a flavor chunk. A label the A-Box lacks fails the page;
    `check_game_codes` keeps that from happening."""
    parts, missing = build_work_parts(fact, chunk, abox, VERSION_GROUP)
    if missing:
        raise BatchBuildError(extract.title, fact.section_path, GAME_CODE_FIELD, missing[0])
    return parts


def build_move_chunk_batch(extract: MovePageExtract, inputs: MoveChunkInputs) -> MoveChunkBuild:
    """Build the pass 2 batch of 1 page.

    Raises `ChunkBuildError` for a chunk id collision and `BatchBuildError` for a flavor
    label the A-Box lacks. A mention that resolves to nothing becomes a note.
    """
    move = build_stub(MOVE, extract.name)
    mentions = MentionContext(inputs.abox, inputs.resolve_species)
    parts: list[Parts] = [Parts(nodes=(move,))]
    notes: list[MoveNote] = []
    for fact, label in number_chunks(extract.title, extract.chunks):
        chunk = build_chunk_record(fact, label, move, inputs.source_url)
        parts.append(Parts((chunk,), (build_edge(DESCRIBES, chunk, move),)))
        if fact.aspect is ChunkAspect.FLAVOR:
            parts.append(build_flavor_parts(extract, fact, chunk, inputs.abox))
        mention_parts, unresolved = build_mention_parts(fact, chunk, move, mentions)
        parts.append(mention_parts)
        notes.extend(
            MoveNote(
                kind=MoveNoteKind.UNRESOLVED_MENTION,
                field=fact.section_path,
                detail=describe_unresolved(mention),
            )
            for mention in unresolved
        )
    joined = join_parts(parts)
    claims = (
        *build_claims([move.node_ref], MOVE_CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(list_owner_records(joined, MOVE_CHUNK_SCOPE), MOVE_CHUNK_OWNER_CLAIMS),
    )
    return MoveChunkBuild(
        batch=assemble_batch(MOVE_CHUNK_SCOPE, joined, claims), notes=tuple(notes)
    )


def check_game_codes(seed: GraphBatch) -> None:
    """Fail unless every `GAME_CODES` label is an A-Box VersionGroup."""
    groups = {node.name for node in seed.nodes if node.class_iri == VERSION_GROUP}
    problems = [
        f"{code} names {label!r}, which is no A-Box VersionGroup"
        for code, labels in GAME_CODES.items()
        for label in labels
        if label not in groups
    ]
    if problems:
        raise GameCodeError("; ".join(problems))
