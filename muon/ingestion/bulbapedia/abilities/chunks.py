"""Text chunks of an ability page: Effect, Trivia and the infobox `desc` rows become
`ChunkFact`s.

Map:
- Aspect: Effect and its subsections give `effect` chunks, except a subsection whose title
  names Pokémon Champions or contests; Trivia gives `trivia` chunks (`read_ability_aspect`).
  Both follow the shared prose rules of `chunking.py`.
- Desc rows: a row names a generation. A row with no `sup` mark is 1 text shown in every
  version group of that generation in `ABILITY_VERSION_GROUPS`. A row with `sup` marks
  splits at `<br>`, and each part is shown in exactly the version groups its marks name
  (`SUP_CODES`). A part marked only for a side game gives nothing (`read_desc_parts`,
  `resolve_part_labels`).
- Flavor: parts with the same text give 1 chunk with the union of their version groups
  (`merge_flavor_rows`).

Every chunk describes the ability of the page. The facts hold Bulbapedia terms only;
`batches.py` maps them onto ids and IRIs. This module knows the pkmn universe, as every
ingestion module of a source does.
"""

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard

import mwparserfromhell
from mwparserfromhell.nodes import Node, Template

from muon.ingestion.batch import GraphBatch
from muon.ingestion.bulbapedia.abilities.page import (
    AbilityNote,
    AbilityNoteKind,
    DescRow,
)
from muon.ingestion.bulbapedia.batching import CORE, VERSION_GROUP
from muon.ingestion.bulbapedia.chunking import (
    MENTION_TEMPLATES,
    ChunkAspect,
    ChunkFact,
    FlavorRow,
    MentionRef,
    RenderedBlock,
    find_long_chunks,
    merge_flavor_rows,
    normalize_chunk_text,
    read_prose_section,
)
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    normalize_template_name,
    read_plain_text,
)

EFFECT_TITLE = "effect"
TRIVIA_TITLE = "trivia"
EXCLUDED_EFFECT_WORDS = ("champions", "contest")
SUP_TEMPLATE_PATTERN = re.compile(r"^sup/\w+$")
SUP_CODE_PARAM = "1"
ROW_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
PART_OF = CORE + "partOf"
# Desc rows sit in the infobox, and the lead section has no title.
FLAVOR_SECTION_PATH = "Infobox"

# Version groups whose games have no abilities. A `sup` mark can still name one.
NO_ABILITY_VERSION_GROUPS = (
    "Let's Go Pikachu-Let's Go Eevee",
    "Legends: Arceus",
    "Legends: Z-A",
)

# `sup` codes of `desc` rows mapped to A-Box VersionGroup labels. `PBR` (Battle
# Revolution) is a side game: a part marked only `PBR` gives no chunk.
SUP_CODES: Mapping[str, tuple[str, ...]] = {
    "RSEFRLG": ("Ruby-Sapphire", "Emerald", "FireRed-LeafGreen"),
    "RS": ("Ruby-Sapphire",),
    "Colo": ("Colosseum",),
    "XD": ("XD",),
    "DPP": ("Diamond-Pearl", "Platinum"),
    "HGSS": ("HeartGold-SoulSilver",),
    "DPPtHGSS": ("Diamond-Pearl", "Platinum", "HeartGold-SoulSilver"),
    "PBR": (),
    "XY": ("X-Y",),
    "ORAS": ("Omega Ruby-Alpha Sapphire",),
    "SM": ("Sun-Moon",),
    "USUM": ("Ultra Sun-Ultra Moon",),
    "SwShLA": ("Sword-Shield", "Legends: Arceus"),
    "BDSP": ("Brilliant Diamond and Shining Pearl",),
    "SwShBDSP": ("Sword-Shield", "Brilliant Diamond and Shining Pearl"),
    "LA": ("Legends: Arceus",),
}


@dataclass(frozen=True)
class DescPart:
    """1 text of a desc row, with its `sup` marks removed from the wikitext."""

    wikitext: str
    codes: tuple[str, ...]


@dataclass(frozen=True)
class UnknownSupCode:
    generation_label: str
    code: str


@dataclass(frozen=True)
class AbilityChunkReading:
    """The chunks of an ability page, what its cleanup removed or kept, and what a reviewer
    should see: `sup` codes no rule reads and chunks above `LONG_CHUNK_LENGTH`."""

    chunks: tuple[ChunkFact, ...]
    dropped_templates: Mapping[str, int]
    removed_tables: int
    math_formulas: int
    unknown_codes: tuple[UnknownSupCode, ...]
    long_chunks: tuple[ChunkFact, ...]


def read_ability_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
    """Return the aspect of the prose section at `path`, or `None` when it gives no chunk.

    The same rule as `read_move_aspect`, kept apart until a 3rd page type needs it.
    """
    titles = tuple(title.lower() for title in path)
    if titles[:1] == (TRIVIA_TITLE,):
        return ChunkAspect.TRIVIA
    if titles[:1] != (EFFECT_TITLE,):
        return None
    if any(word in title for title in titles for word in EXCLUDED_EFFECT_WORDS):
        return None
    return ChunkAspect.EFFECT


def build_ability_version_groups(seed: GraphBatch) -> Mapping[str, tuple[str, ...]]:
    """Map each Generation label to its A-Box VersionGroup labels, in seed order, without
    `NO_ABILITY_VERSION_GROUPS`. This is the default of a desc row with no `sup` mark."""
    names = {node.node_ref: node.name for node in seed.nodes}
    generations = {
        names[edge.start]: names[edge.end]
        for edge in seed.edges
        if edge.property_iri == PART_OF and edge.start.class_iri == VERSION_GROUP
    }
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for node in seed.nodes:
        if node.class_iri == VERSION_GROUP and node.name not in NO_ABILITY_VERSION_GROUPS:
            groups[generations[node.name]].append(node.name)
    return {generation: tuple(labels) for generation, labels in groups.items()}


def is_sup_template(node: Node) -> TypeGuard[Template]:
    return isinstance(node, Template) and bool(
        SUP_TEMPLATE_PATTERN.match(normalize_template_name(str(node.name)))
    )


def read_part(raw: str) -> DescPart:
    code = mwparserfromhell.parse(raw)
    marks = [node for node in code.filter_templates(recursive=False) if is_sup_template(node)]
    codes = tuple(
        str(mark.get(SUP_CODE_PARAM).value).strip() for mark in marks if mark.has(SUP_CODE_PARAM)
    )
    for mark in marks:
        code.remove(mark)
    return DescPart(wikitext=str(code), codes=codes)


def read_desc_parts(raw: str) -> tuple[DescPart, ...]:
    """Return the texts of a desc row. A row with no `sup` mark is 1 text, even on 2 lines;
    a row with one splits at `<br>`."""
    whole = read_part(raw)
    if not whole.codes:
        return (whole,)
    return tuple(read_part(piece) for piece in ROW_BREAK_PATTERN.split(raw) if piece.strip())


def resolve_part_labels(
    part: DescPart, generation_label: str, version_groups: Mapping[str, tuple[str, ...]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the version group labels of a part and its codes that `SUP_CODES` lacks.

    No code gives the generation default. An unknown code also gives the generation
    default, so a new code loses no text before it gets a rule.
    """
    unknown = tuple(code for code in part.codes if code not in SUP_CODES)
    if not part.codes or unknown:
        return version_groups.get(generation_label, ()), unknown
    labels = (label for code in part.codes for label in SUP_CODES[code])
    return tuple(dict.fromkeys(labels)), ()


def render_desc_part(wikitext: str) -> RenderedBlock:
    rendered = read_plain_text(wikitext, MENTION_TEMPLATES)
    return RenderedBlock(
        is_bullet=False,
        text=normalize_chunk_text(rendered.text),
        mentions=tuple(
            MentionRef(template=item.name, target=item.target) for item in rendered.captured
        ),
        dropped=rendered.dropped,
    )


@dataclass(frozen=True)
class FlavorReading:
    rows: tuple[FlavorRow, ...]
    unknown_codes: tuple[UnknownSupCode, ...]
    dropped: tuple[str, ...]


def read_flavor_rows(
    desc_rows: Sequence[DescRow], version_groups: Mapping[str, tuple[str, ...]]
) -> FlavorReading:
    """Read every desc row part with its version groups. A part with no version group or no
    text gives nothing."""
    rows: list[FlavorRow] = []
    unknown: list[UnknownSupCode] = []
    dropped: list[str] = []
    for desc_row in desc_rows:
        for part in read_desc_parts(desc_row.wikitext):
            labels, codes = resolve_part_labels(part, desc_row.generation_label, version_groups)
            unknown.extend(UnknownSupCode(desc_row.generation_label, code) for code in codes)
            block = render_desc_part(part.wikitext)
            if not labels or not block.text:
                continue
            dropped.extend(block.dropped)
            rows.append(FlavorRow(section_path=FLAVOR_SECTION_PATH, block=block, labels=labels))
    return FlavorReading(tuple(rows), tuple(unknown), tuple(dropped))


def read_ability_chunks(
    sections: Sequence[Section],
    desc_rows: Sequence[DescRow],
    ability_id: str,
    version_groups: Mapping[str, tuple[str, ...]],
) -> AbilityChunkReading:
    """Read the chunks of an ability page: prose chunks in page order, then flavor chunks."""
    prose: list[ChunkFact] = []
    dropped: Counter[str] = Counter()
    removed_tables = 0
    math_formulas = 0
    for section in sections:
        aspect = read_ability_aspect(section.path)
        if aspect is None:
            continue
        reading = read_prose_section(section, aspect, ability_id)
        prose.extend(reading.chunks)
        dropped.update(reading.dropped)
        removed_tables += reading.removed_tables
        math_formulas += reading.math_formulas
    flavor = read_flavor_rows(desc_rows, version_groups)
    dropped.update(flavor.dropped)
    chunks = (*prose, *merge_flavor_rows(flavor.rows, ability_id))
    return AbilityChunkReading(
        chunks=chunks,
        dropped_templates=dict(sorted(dropped.items())),
        removed_tables=removed_tables,
        math_formulas=math_formulas,
        unknown_codes=flavor.unknown_codes,
        long_chunks=find_long_chunks(chunks),
    )


def build_chunk_notes(reading: AbilityChunkReading) -> tuple[AbilityNote, ...]:
    unknown = (
        AbilityNote(
            kind=AbilityNoteKind.UNKNOWN_SUP_CODE,
            field=FLAVOR_SECTION_PATH,
            detail=f"{code.code} in a {code.generation_label} desc row has no rule",
        )
        for code in reading.unknown_codes
    )
    long = (
        AbilityNote(
            kind=AbilityNoteKind.LONG_CHUNK,
            field=chunk.section_path,
            detail=f"{len(chunk.text)} characters",
        )
        for chunk in reading.long_chunks
    )
    return (*unknown, *long)
