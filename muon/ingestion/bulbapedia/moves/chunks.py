"""Text chunks of a move page: Effect, Trivia and Description become `ChunkFact`s.

Map:
- Aspect: Effect and its subsections give `effect` chunks, except the Pokémon Champions and
  contest subsections; Trivia gives `trivia` chunks (`read_move_aspect`). Both follow the
  shared prose rules of `chunking.py`.
- Description: each `movedescentry` row names its games through `{{gameabbrevN}}` codes
  and holds the in-game text. Only main-series codes count (`GAME_CODES`); a side-series
  row gives nothing (`read_description_rows`).
- Captions: a row with 2 texts labels them with `<small>(X)</small>` or `{{tt|*|X}}`. A TM
  or HM text is dropped, a "move" caption is removed and any other caption stays as a
  lead-in. A row with no caption is 1 text, even on 2 lines (`read_row_text`).
- Flavor: rows with the same text give 1 chunk with the union of their version groups
  (`merge_flavor_rows`).

Every chunk describes the move of the page. The facts hold Bulbapedia terms only;
`batches.py` maps them onto ids and IRIs. This module knows the pkmn universe, as every
ingestion module of a source does.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard

import mwparserfromhell
from mwparserfromhell.nodes import Node, Tag, Template
from mwparserfromhell.wikicode import Wikicode

from muon.ingestion.bulbapedia.chunking import (
    MENTION_TEMPLATES,
    ChunkAspect,
    ChunkFact,
    MentionRef,
    RenderedBlock,
    find_long_chunks,
    normalize_chunk_text,
    read_prose_section,
    read_tag_name,
    strip_comments,
)
from muon.ingestion.bulbapedia.wikitext import (
    PATH_SEPARATOR,
    Section,
    normalize_template_name,
    read_plain_text,
    to_plain_text,
)

EFFECT_TITLE = "effect"
TRIVIA_TITLE = "trivia"
DESCRIPTION_TITLE = "description"
EXCLUDED_EFFECT_WORDS = ("champions", "contest")
ROW_TEMPLATE = "movedescentry"
GAMES_PARAM = "1"
TEXT_PARAM = "2"
GAME_CODE_TEMPLATE_PATTERN = re.compile(r"^gameabbrev[2-9]$")
ROW_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
TOOLTIP_TEMPLATE = "tt"
TOOLTIP_MARK = "*"
TOOLTIP_MARK_PARAM = "1"
TOOLTIP_NOTE_PARAM = "2"
SMALL_TAG = "small"
CAPTION_BRACKETS = "()"
TM_HM_CAPTION_PATTERN = re.compile(r"^(TM|HM)\b")
MOVE_CAPTION = "move"
CAPTION_LAYOUT = "{caption}: {text}"
PART_JOINER = " "

# Bulbapedia game codes of `{{gameabbrev2}}` to `{{gameabbrev9}}`, mapped to A-Box
# VersionGroup labels. A cell such as `SMUSUMPE` chains several codes.
GAME_CODES: Mapping[str, tuple[str, ...]] = {
    "GS": ("Gold-Silver",),
    "C": ("Crystal",),
    "RS": ("Ruby-Sapphire",),
    "E": ("Emerald",),
    "FRLG": ("FireRed-LeafGreen",),
    "DP": ("Diamond-Pearl",),
    "Pt": ("Platinum",),
    "HGSS": ("HeartGold-SoulSilver",),
    "BW": ("Black-White",),
    "B2W2": ("Black 2-White 2",),
    "XY": ("X-Y",),
    "ORAS": ("Omega Ruby-Alpha Sapphire",),
    "SM": ("Sun-Moon",),
    "USUM": ("Ultra Sun-Ultra Moon",),
    "PE": ("Let's Go Pikachu-Let's Go Eevee",),
    "SwSh": ("Sword-Shield",),
    "BDSP": ("Brilliant Diamond and Shining Pearl",),
    "LA": ("Legends: Arceus",),
    "SV": ("Scarlet-Violet",),
    "ZA": ("Legends: Z-A",),
}
CODES_LONGEST_FIRST = tuple(sorted(GAME_CODES, key=len, reverse=True))


@dataclass(frozen=True)
class RowPart:
    """1 text of a Description row, with its caption removed from the wikitext."""

    caption: str | None
    wikitext: str


@dataclass(frozen=True)
class FlavorRow:
    section_path: str
    block: RenderedBlock
    labels: tuple[str, ...]


@dataclass(frozen=True)
class UnknownGameCode:
    """A code cell of a row that `GAME_CODES` could not read to its end."""

    section_path: str
    cell: str


@dataclass(frozen=True)
class DescriptionReading:
    rows: tuple[FlavorRow, ...]
    unknown_codes: tuple[UnknownGameCode, ...]
    dropped: tuple[str, ...]


@dataclass(frozen=True)
class MoveChunkReading:
    """The chunks of a move page, what its cleanup removed or kept, and what a reviewer
    should see: game code cells no rule reads and chunks above `LONG_CHUNK_LENGTH`."""

    chunks: tuple[ChunkFact, ...]
    dropped_templates: Mapping[str, int]
    removed_tables: int
    math_formulas: int
    unknown_codes: tuple[UnknownGameCode, ...]
    long_chunks: tuple[ChunkFact, ...]


def read_move_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
    """Return the aspect of the prose section at `path`, or `None` when it gives no prose chunk.

    Titles compare in lowercase. Description gives flavor chunks through its own reader.
    """
    titles = tuple(title.lower() for title in path)
    if titles[:1] == (TRIVIA_TITLE,):
        return ChunkAspect.TRIVIA
    if titles[:1] != (EFFECT_TITLE,):
        return None
    if any(word in title for title in titles for word in EXCLUDED_EFFECT_WORDS):
        return None
    return ChunkAspect.EFFECT


def split_game_codes(cell: str) -> tuple[tuple[str, ...], str]:
    """Return the labels of the codes in `cell`, in order, and the part no code matches.

    At each position the longest code wins, so `BWB2W2` reads as `BW` and `B2W2`.
    """
    labels: list[str] = []
    position = 0
    while position < len(cell):
        code = next((code for code in CODES_LONGEST_FIRST if cell.startswith(code, position)), None)
        if code is None:
            return tuple(labels), cell[position:]
        labels.extend(GAME_CODES[code])
        position += len(code)
    return tuple(labels), ""


def read_row_games(template: Template) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the version group labels of a row and its code cells that no rule reads.

    Only `{{gameabbrev2}}` to `{{gameabbrev9}}` count; side-series codes give nothing.
    """
    if not template.has(GAMES_PARAM):
        return (), ()
    labels: list[str] = []
    unread: list[str] = []
    for game in template.get(GAMES_PARAM).value.filter_templates(recursive=True):
        name = normalize_template_name(str(game.name))
        if not GAME_CODE_TEMPLATE_PATTERN.match(name) or not game.has(GAMES_PARAM):
            continue
        cell = str(game.get(GAMES_PARAM).value).strip()
        found, rest = split_game_codes(cell)
        labels.extend(found)
        if rest:
            unread.append(cell)
    return tuple(dict.fromkeys(labels)), tuple(unread)


def is_caption_tooltip(node: Node) -> TypeGuard[Template]:
    return (
        isinstance(node, Template)
        and normalize_template_name(str(node.name)) == TOOLTIP_TEMPLATE
        and node.has(TOOLTIP_MARK_PARAM)
        and str(node.get(TOOLTIP_MARK_PARAM).value).strip() == TOOLTIP_MARK
        and node.has(TOOLTIP_NOTE_PARAM)
    )


def find_caption_node(code: Wikicode) -> Tag | Template | None:
    for node in code.filter(recursive=False):
        if isinstance(node, Tag) and read_tag_name(node) == SMALL_TAG:
            return node
        if is_caption_tooltip(node):
            return node
    return None


def read_caption(part: str) -> RowPart:
    """Return the text of a row part and its caption: the 1st `<small>` element or the note
    of the 1st `{{tt|*|X}}`, without its brackets. The caption leaves the wikitext."""
    code = mwparserfromhell.parse(part)
    node = find_caption_node(code)
    if node is None:
        return RowPart(caption=None, wikitext=part)
    raw = node.contents if isinstance(node, Tag) else node.get(TOOLTIP_NOTE_PARAM).value
    caption = to_plain_text(str(raw)).strip().strip(CAPTION_BRACKETS).strip()
    code.remove(node)
    return RowPart(caption=caption or None, wikitext=str(code))


def render_part(part: RowPart) -> RenderedBlock:
    rendered = read_plain_text(part.wikitext, MENTION_TEMPLATES)
    text = normalize_chunk_text(rendered.text)
    if part.caption is not None and part.caption != MOVE_CAPTION and text:
        text = CAPTION_LAYOUT.format(caption=part.caption, text=text)
    return RenderedBlock(
        is_bullet=False,
        text=text,
        mentions=tuple(
            MentionRef(template=item.name, target=item.target) for item in rendered.captured
        ),
        dropped=rendered.dropped,
    )


def read_row_text(raw: str) -> RenderedBlock | None:
    """Return the text of a row, or `None` when nothing is left.

    The row is cut at `<br>` only when at least 1 part has a caption. A part captioned TM
    or HM is dropped, a part captioned "move" keeps its text alone, any other caption leads
    its text in, and the parts join with a space. A row with no caption is 1 text.
    """
    parts = [read_caption(piece) for piece in ROW_BREAK_PATTERN.split(raw)]
    if not any(part.caption for part in parts):
        parts = [RowPart(caption=None, wikitext=raw)]
    kept = [
        render_part(part)
        for part in parts
        if part.caption is None or not TM_HM_CAPTION_PATTERN.match(part.caption)
    ]
    text = normalize_chunk_text(PART_JOINER.join(block.text for block in kept))
    if not text:
        return None
    return RenderedBlock(
        is_bullet=False,
        text=text,
        mentions=tuple(mention for block in kept for mention in block.mentions),
        dropped=tuple(name for block in kept for name in block.dropped),
    )


def read_description_rows(sections: Sequence[Section]) -> DescriptionReading:
    """Read the `movedescentry` rows of every Description section, in page order.

    A row with no main-series game or no text gives nothing.
    """
    rows: list[FlavorRow] = []
    unknown: list[UnknownGameCode] = []
    dropped: list[str] = []
    for section in sections:
        if tuple(title.lower() for title in section.path[:1]) != (DESCRIPTION_TITLE,):
            continue
        section_path = PATH_SEPARATOR.join(section.path)
        code = mwparserfromhell.parse(strip_comments(section.text))
        for template in code.filter_templates(recursive=False):
            if normalize_template_name(str(template.name)) != ROW_TEMPLATE:
                continue
            labels, unread = read_row_games(template)
            unknown.extend(UnknownGameCode(section_path, cell) for cell in unread)
            if not labels or not template.has(TEXT_PARAM):
                continue
            block = read_row_text(str(template.get(TEXT_PARAM).value))
            if block is None:
                continue
            dropped.extend(block.dropped)
            rows.append(FlavorRow(section_path=section_path, block=block, labels=labels))
    return DescriptionReading(tuple(rows), tuple(unknown), tuple(dropped))


def merge_flavor_rows(rows: Sequence[FlavorRow], move_id: str) -> tuple[ChunkFact, ...]:
    """Return 1 chunk per text, in first-seen order, with the ordered union of the version
    groups of its rows and the section and mentions of its first row."""
    first_rows: dict[str, FlavorRow] = {}
    labels: dict[str, dict[str, None]] = {}
    for row in rows:
        first_rows.setdefault(row.block.text, row)
        labels.setdefault(row.block.text, {}).update(dict.fromkeys(row.labels))
    return tuple(
        ChunkFact(
            subject_id=move_id,
            aspect=ChunkAspect.FLAVOR,
            section_path=row.section_path,
            text=text,
            mentions=row.block.mentions,
            versions=tuple(labels[text]),
        )
        for text, row in first_rows.items()
    )


def read_move_chunks(sections: Sequence[Section], move_id: str) -> MoveChunkReading:
    """Read the chunks of a move page: prose chunks in page order, then flavor chunks."""
    prose: list[ChunkFact] = []
    dropped: Counter[str] = Counter()
    removed_tables = 0
    math_formulas = 0
    for section in sections:
        aspect = read_move_aspect(section.path)
        if aspect is None:
            continue
        reading = read_prose_section(section, aspect, move_id)
        prose.extend(reading.chunks)
        dropped.update(reading.dropped)
        removed_tables += reading.removed_tables
        math_formulas += reading.math_formulas
    description = read_description_rows(sections)
    dropped.update(description.dropped)
    chunks = (*prose, *merge_flavor_rows(description.rows, move_id))
    return MoveChunkReading(
        chunks=chunks,
        dropped_templates=dict(sorted(dropped.items())),
        removed_tables=removed_tables,
        math_formulas=math_formulas,
        unknown_codes=description.unknown_codes,
        long_chunks=find_long_chunks(chunks),
    )
