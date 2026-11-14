"""Prose chunking rules shared by every Bulbapedia page type.

Map:
- Models: `ChunkAspect`, `MentionRef` and `ChunkFact` hold a chunk in Bulbapedia terms.
- Blocks: a prose section is cut on blank lines into paragraphs and bullet items, after
  comments, tables and galleries are removed and layout tags unwrapped (`split_blocks`).
- Rendering: each block becomes plain text with the typed templates it mentions
  (`render_block`), through the display rules of `wikitext.py`.
- Merging: a prose paragraph under `MIN_PARAGRAPH_LENGTH` joins the next one, or the one
  before when none follows (`merge_short_paragraphs`).

A page type decides which sections are read, as which aspect and about which subject;
this module knows nothing of page types or of the ontology.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Tag
from pydantic import BaseModel, ConfigDict

from muon.ingestion.bulbapedia.wikitext import (
    COMMENT_PATTERN,
    PATH_SEPARATOR,
    Section,
    read_plain_text,
)

MIN_PARAGRAPH_LENGTH = 100
# Above this many characters a chunk is reported, never split: the longest chunk of the
# pinned pages has 3,895, and the embedding model takes about 8,000 tokens.
LONG_CHUNK_LENGTH = 4000
TABLE_TAG = "table"
MATH_TAG = "math"
MENTION_TEMPLATES = frozenset({"p", "m", "a", "type", "t"})
REMOVED_TAGS = frozenset({TABLE_TAG, "gallery"})
UNWRAPPED_TAGS = frozenset({"div", "center"})
SOFT_HYPHEN = "­"
BULLET_MARK = "*"
BULLET_ITEM_DEPTH = 1
LINE_BREAK = "\n"
LINE_JOINER = " "
BLANK_LINES_PATTERN = re.compile(r"\n\s*\n")
# A line that holds only comments is removed with its line break, as MediaWiki does, so a
# comment between 2 lines of a paragraph does not split it.
COMMENT_LINE_PATTERN = re.compile(
    r"^[ \t]*(?:<!--(?:(?!-->).)*(?:-->|\Z)[ \t]*)+(?:\n|\Z)", re.MULTILINE | re.DOTALL
)


class ChunkAspect(StrEnum):
    BIOLOGY = "biology"
    ORIGIN = "origin"
    TRIVIA = "trivia"
    POKEDEX = "pokedex"
    EFFECT = "effect"
    FLAVOR = "flavor"


class ChunkModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class MentionRef(ChunkModel):
    """A typed template that rendered into a chunk: its name and its raw 1st parameter."""

    template: str
    target: str


class ChunkFact(ChunkModel):
    """1 chunk in Bulbapedia terms. `subject_id` is the id of the entity it describes.

    `versions` holds the game names a text is shown in, in page order: the Version names of
    a Pokédex entry or the VersionGroup labels of a flavor text. It is empty for every
    other aspect.
    """

    subject_id: str
    aspect: ChunkAspect
    section_path: str
    text: str
    mentions: tuple[MentionRef, ...]
    versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Block:
    """A paragraph or a bullet item as raw wikitext lines. A bullet keeps its sub-bullets."""

    is_bullet: bool
    lines: tuple[str, ...]


@dataclass(frozen=True)
class RenderedBlock:
    is_bullet: bool
    text: str
    mentions: tuple[MentionRef, ...]
    dropped: tuple[str, ...]


@dataclass(frozen=True)
class ProseReading:
    """The chunks of 1 prose section, the names of the templates its render dropped, and how
    many tables it removed and `<math>` formulas it kept as LaTeX source."""

    chunks: tuple[ChunkFact, ...]
    dropped: tuple[str, ...]
    removed_tables: int = 0
    math_formulas: int = 0


def read_tag_name(tag: Tag) -> str:
    return str(tag.tag).strip().lower()


def remove_tags(
    wikitext: str, names: frozenset[str], unwrapped: frozenset[str] = frozenset()
) -> str:
    """Return `wikitext` without the elements named in `names`, nested ones included.

    A wiki table is the element `table`. Removing the node keeps a table with blank lines
    inside from splitting into text blocks. The elements named in `unwrapped` keep their
    contents and lose their tags, so a layout `<div>` around a removed table or around
    text with blank lines leaves no stray tag in either block.
    """
    code = mwparserfromhell.parse(wikitext)
    for tag in code.filter_tags(recursive=True, matches=lambda node: read_tag_name(node) in names):
        if code.contains(tag):
            code.remove(tag)
    for tag in code.filter_tags(
        recursive=True, matches=lambda node: read_tag_name(node) in unwrapped
    ):
        if tag.contents is None:
            code.remove(tag)
        else:
            code.replace(tag, tag.contents)
    return str(code)


def cut_group(lines: Sequence[str]) -> list[Block]:
    """Cut the non-blank lines between 2 blank lines into bullet items and prose paragraphs.

    A line with exactly 1 `*` opens an item and the lines after it with 2 or more belong
    to it. A deeper line with no open item opens one. Other runs of lines are paragraphs.
    """
    blocks: list[Block] = []
    for line in lines:
        previous = blocks[-1] if blocks else None
        if line.startswith(BULLET_MARK):
            depth = len(line) - len(line.lstrip(BULLET_MARK))
            if previous is not None and previous.is_bullet and depth > BULLET_ITEM_DEPTH:
                blocks[-1] = Block(True, (*previous.lines, line))
            else:
                blocks.append(Block(True, (line,)))
        elif previous is not None and not previous.is_bullet:
            blocks[-1] = Block(False, (*previous.lines, line))
        else:
            blocks.append(Block(False, (line,)))
    return blocks


def strip_comments(wikitext: str) -> str:
    return COMMENT_PATTERN.sub("", COMMENT_LINE_PATTERN.sub("", wikitext))


def count_tags(wikitext: str, name: str) -> int:
    """Count the elements named `name` in `wikitext`, nested ones included."""
    code = mwparserfromhell.parse(wikitext)
    return len(code.filter_tags(recursive=True, matches=lambda node: read_tag_name(node) == name))


def join_template_lines(wikitext: str) -> str:
    """Put each top-level template that spans lines on 1 line.

    A notice such as `{{MoveResearch|\n* question}}` holds bullet lines, which the block
    cut would otherwise read as bullet items of the page and leave with stray braces.
    """
    code = mwparserfromhell.parse(wikitext)
    for template in code.filter_templates(recursive=False):
        written = str(template)
        if LINE_BREAK in written:
            code.replace(template, LINE_JOINER.join(written.splitlines()))
    return str(code)


def split_blocks(wikitext: str) -> tuple[Block, ...]:
    """Cut section wikitext into paragraphs and bullet items, in page order.

    Comments, tables and galleries are removed first, layout `<div>` tags are unwrapped and
    every template is put on 1 line.
    """
    text = join_template_lines(remove_tags(strip_comments(wikitext), REMOVED_TAGS, UNWRAPPED_TAGS))
    blocks: list[Block] = []
    for group in BLANK_LINES_PATTERN.split(text):
        lines = [line.strip() for line in group.split(LINE_BREAK) if line.strip()]
        blocks.extend(cut_group(lines))
    return tuple(blocks)


def normalize_chunk_text(text: str) -> str:
    """Remove soft hyphens and turn every run of whitespace, line breaks included, into 1 space."""
    return LINE_JOINER.join(text.replace(SOFT_HYPHEN, "").split())


def render_block(block: Block) -> RenderedBlock:
    lines = [line.lstrip(BULLET_MARK) for line in block.lines] if block.is_bullet else block.lines
    rendered = read_plain_text(LINE_BREAK.join(lines), MENTION_TEMPLATES)
    return RenderedBlock(
        is_bullet=block.is_bullet,
        text=normalize_chunk_text(rendered.text),
        mentions=tuple(
            MentionRef(template=item.name, target=item.target) for item in rendered.captured
        ),
        dropped=rendered.dropped,
    )


def join_paragraphs(first: RenderedBlock, second: RenderedBlock) -> RenderedBlock:
    return RenderedBlock(
        is_bullet=False,
        text=f"{first.text}{LINE_JOINER}{second.text}",
        mentions=(*first.mentions, *second.mentions),
        dropped=(*first.dropped, *second.dropped),
    )


def merge_short_paragraphs(blocks: Sequence[RenderedBlock]) -> tuple[RenderedBlock, ...]:
    """Merge each prose paragraph under `MIN_PARAGRAPH_LENGTH` into the next prose paragraph.

    Merging repeats while the result stays short. Bullet items never merge and are skipped
    when looking for the next paragraph. A short paragraph with no paragraph after it joins
    the last one before it, and stays alone when the section has no other paragraph.
    """
    merged: list[RenderedBlock] = []
    pending: RenderedBlock | None = None
    for block in blocks:
        if block.is_bullet:
            merged.append(block)
            continue
        current = block if pending is None else join_paragraphs(pending, block)
        pending = current if len(current.text) < MIN_PARAGRAPH_LENGTH else None
        if pending is None:
            merged.append(current)
    if pending is None:
        return tuple(merged)
    last_prose = next((i for i in reversed(range(len(merged))) if not merged[i].is_bullet), None)
    if last_prose is None:
        merged.append(pending)
    else:
        merged[last_prose] = join_paragraphs(merged[last_prose], pending)
    return tuple(merged)


def find_long_chunks(chunks: Sequence[ChunkFact]) -> tuple[ChunkFact, ...]:
    return tuple(chunk for chunk in chunks if len(chunk.text) > LONG_CHUNK_LENGTH)


def read_prose_section(section: Section, aspect: ChunkAspect, subject_id: str) -> ProseReading:
    """Read the chunks of a prose section about `subject_id`, 1 per merged block."""
    uncommented = strip_comments(section.text)
    rendered = [render_block(block) for block in split_blocks(section.text)]
    kept = merge_short_paragraphs([block for block in rendered if block.text])
    chunks = tuple(
        ChunkFact(
            subject_id=subject_id,
            aspect=aspect,
            section_path=PATH_SEPARATOR.join(section.path),
            text=block.text,
            mentions=block.mentions,
        )
        for block in kept
    )
    return ProseReading(
        chunks=chunks,
        dropped=tuple(name for b in rendered for name in b.dropped),
        removed_tables=count_tags(uncommented, TABLE_TAG),
        math_formulas=count_tags(uncommented, MATH_TAG),
    )
