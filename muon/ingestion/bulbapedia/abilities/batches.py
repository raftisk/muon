"""1 ability page extract to its pass 1 and pass 2 batches.

Pass 1 owns the Ability (scope {`Ability`}). The Generation it was introduced in is an
A-Box stub, resolved by label so a missing name fails the page. The batch claims
`introducedIn` out of the Ability. No claim covers `hasAbility`: the species pages own it,
so an ability upsert never removes it.

Pass 2 owns the TextChunks of the page (scope {`TextChunk`}). Every chunk `describes` the
Ability, a flavor chunk links `fromWork` to its VersionGroups, and a mention links in
through `mentionedIn`. The Ability stub anchors the `describes` claim, so a page with no
chunk left still removes the chunks it had.

This module knows the pkmn universe, as every ingestion module of a source does.
"""

from dataclasses import dataclass

from muon.ingestion.batch import GraphBatch, NodeRecord, PropertyValue
from muon.ingestion.bulbapedia.abilities.chunks import (
    FLAVOR_SECTION_PATH,
    NO_ABILITY_VERSION_GROUPS,
    SUP_CODES,
)
from muon.ingestion.bulbapedia.abilities.page import (
    GENERATION_KEY,
    INFOBOX_LABEL,
    AbilityNote,
    AbilityNoteKind,
    AbilityPageExtract,
)
from muon.ingestion.bulbapedia.batching import (
    ABILITY,
    ALIASES,
    DESCRIBES,
    FROM_WORK,
    GENERATION,
    IN,
    INTRODUCED_IN,
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
    describe_unresolved,
    drop_none,
    join_parts,
    list_owner_records,
    number_chunks,
    require_abox,
)
from muon.ingestion.bulbapedia.chunking import ChunkAspect

EFFECT_TEXT = PKMN + "effectText"
SUP_CODE_FIELD = "sup code"

ABILITY_SCOPE = frozenset({ABILITY})
ABILITY_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    ABILITY: (ClaimRule(INTRODUCED_IN, OUT, GENERATION, owns_other_end=False),),
}

ABILITY_CHUNK_SCOPE = frozenset({TEXT_CHUNK})
ABILITY_CHUNK_ANCHOR_CLAIMS = (ClaimRule(DESCRIBES, IN, TEXT_CHUNK, owns_other_end=True),)
ABILITY_CHUNK_OWNER_CLAIMS: dict[str, tuple[ClaimRule, ...]] = {
    TEXT_CHUNK: (
        ClaimRule(FROM_WORK, OUT, VERSION_GROUP, owns_other_end=False),
        ClaimRule(MENTIONED_IN, IN, None, owns_other_end=False),
    ),
}


class SupCodeError(Exception):
    """`SUP_CODES` or `NO_ABILITY_VERSION_GROUPS` names a version group the A-Box lacks."""


@dataclass(frozen=True)
class AbilityContext:
    """What `require_abox` needs to name a failed ability page."""

    page_title: str
    abox: AboxIndex


@dataclass(frozen=True)
class AbilityChunkInputs:
    """What pass 2 resolves against, and the permalink of the page its chunks cite."""

    abox: AboxIndex
    resolve_species: SpeciesResolver
    source_url: str


@dataclass(frozen=True)
class AbilityChunkBuild:
    batch: GraphBatch
    notes: tuple[AbilityNote, ...]


def build_ability_record(extract: AbilityPageExtract, effect_text: str | None) -> NodeRecord:
    """Return the Ability owner record, without the properties that have no value."""
    properties: dict[str, PropertyValue | None] = {
        EFFECT_TEXT: effect_text,
        ALIASES: extract.aliases or None,
    }
    return NodeRecord(
        class_iri=ABILITY, id=extract.id, name=extract.name, properties=drop_none(properties)
    )


def build_ability_batch(
    extract: AbilityPageExtract, abox: AboxIndex, effect_text: str | None
) -> GraphBatch:
    """Build the pass 1 batch of 1 page. A generation the A-Box lacks raises
    `BatchBuildError`."""
    context = AbilityContext(extract.title, abox)
    ability = build_ability_record(extract, effect_text)
    generation = require_abox(
        context, GENERATION, extract.generation_label, (INFOBOX_LABEL, GENERATION_KEY)
    )
    parts = Parts(
        nodes=(ability, generation), edges=(build_edge(INTRODUCED_IN, ability, generation),)
    )
    owners = list_owner_records(parts, ABILITY_SCOPE)
    return assemble_batch(ABILITY_SCOPE, parts, build_owner_claims(owners, ABILITY_OWNER_CLAIMS))


def build_ability_chunk_batch(
    extract: AbilityPageExtract, inputs: AbilityChunkInputs
) -> AbilityChunkBuild:
    """Build the pass 2 batch of 1 page.

    Raises `ChunkBuildError` for a chunk id collision and `BatchBuildError` for a flavor
    label the A-Box lacks; `check_sup_codes` keeps the latter from happening. A mention
    that resolves to nothing becomes a note.
    """
    ability = build_stub(ABILITY, extract.name)
    mentions = MentionContext(inputs.abox, inputs.resolve_species)
    parts: list[Parts] = [Parts(nodes=(ability,))]
    notes: list[AbilityNote] = []
    for fact, label in number_chunks(extract.title, extract.chunks):
        chunk = build_chunk_record(fact, label, ability, inputs.source_url)
        parts.append(Parts((chunk,), (build_edge(DESCRIBES, chunk, ability),)))
        if fact.aspect is ChunkAspect.FLAVOR:
            work_parts, missing = build_work_parts(fact, chunk, inputs.abox, VERSION_GROUP)
            if missing:
                raise BatchBuildError(
                    extract.title, FLAVOR_SECTION_PATH, SUP_CODE_FIELD, missing[0]
                )
            parts.append(work_parts)
        mention_parts, unresolved = build_mention_parts(fact, chunk, ability, mentions)
        parts.append(mention_parts)
        notes.extend(
            AbilityNote(
                kind=AbilityNoteKind.UNRESOLVED_MENTION,
                field=fact.section_path,
                detail=describe_unresolved(mention),
            )
            for mention in unresolved
        )
    joined = join_parts(parts)
    claims = (
        *build_claims([ability.node_ref], ABILITY_CHUNK_ANCHOR_CLAIMS),
        *build_owner_claims(
            list_owner_records(joined, ABILITY_CHUNK_SCOPE), ABILITY_CHUNK_OWNER_CLAIMS
        ),
    )
    return AbilityChunkBuild(
        batch=assemble_batch(ABILITY_CHUNK_SCOPE, joined, claims), notes=tuple(notes)
    )


def check_sup_codes(seed: GraphBatch) -> None:
    """Fail unless every `SUP_CODES` and `NO_ABILITY_VERSION_GROUPS` label is an A-Box
    VersionGroup."""
    groups = {node.name for node in seed.nodes if node.class_iri == VERSION_GROUP}
    problems = [
        f"{code} names {label!r}, which is no A-Box VersionGroup"
        for code, labels in SUP_CODES.items()
        for label in labels
        if label not in groups
    ]
    problems.extend(
        f"NO_ABILITY_VERSION_GROUPS names {label!r}, which is no A-Box VersionGroup"
        for label in NO_ABILITY_VERSION_GROUPS
        if label not in groups
    )
    if problems:
        raise SupCodeError("; ".join(problems))
