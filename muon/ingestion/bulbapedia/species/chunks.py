"""Text chunks of a species page: prose sections and Pokédex rows become `ChunkFact`s.

Map:
- Aspect: the path of a section decides whether it is read and as what (`read_aspect`).
- Blocks: a prose section is cut on blank lines into paragraphs and bullet items, each
  rendered to plain text with the typed templates it mentions (`split_blocks`,
  `render_block`).
- Merging: a prose paragraph under `MIN_PARAGRAPH_LENGTH` joins the next one, or the one
  before when none follows (`merge_short_paragraphs`).
- Subjects: a chunk describes the base form unless an explicit marker names another form
  of the page (`match_marker`, `pick_biology_subject`). Nothing infers a form from prose.

The facts hold Bulbapedia terms only. `batches.py` maps them onto ids and IRIs.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mwparserfromhell

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.species.extract import (
    ChunkAspect,
    ChunkFact,
    MentionRef,
    NoteKind,
    PageNote,
)
from muon.ingestion.bulbapedia.species.forms import ResolvedForm, format_form_names
from muon.ingestion.bulbapedia.wikitext import (
    COMMENT_PATTERN,
    PATH_SEPARATOR,
    Section,
    read_plain_text,
)

MIN_PARAGRAPH_LENGTH = 100
MENTION_TEMPLATES = frozenset({"p", "m", "a", "type", "t"})
BIOLOGY_TITLE = "biology"
TRIVIA_TITLE = "trivia"
FORMS_TITLE = "forms"
POKEDEX_PATH = ("game data", "pokédex entries")
ORIGIN_PATHS = (("trivia", "origin"), ("trivia", "concept and development"))
EXCLUDED_PATHS = (("biology", "evolution"), ("trivia", "merchandise"))
REMOVED_TAGS = frozenset({"table", "gallery"})
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
class ChunkReading:
    chunks: tuple[ChunkFact, ...]
    notes: tuple[PageNote, ...]
    dropped_templates: Mapping[str, int]


def has_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def read_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
    """Return the aspect of the section at `path`, or `None` when it gives no chunk.

    Titles compare in lowercase. The lead (empty path) and every section outside Biology,
    Trivia and the Pokédex entries give nothing, as do Biology > Evolution and
    Trivia > Merchandise with their subsections.
    """
    titles = tuple(title.lower() for title in path)
    if any(has_prefix(titles, excluded) for excluded in EXCLUDED_PATHS):
        return None
    if any(has_prefix(titles, origin) for origin in ORIGIN_PATHS):
        return ChunkAspect.ORIGIN
    if has_prefix(titles, POKEDEX_PATH):
        return ChunkAspect.POKEDEX
    if titles[:1] == (BIOLOGY_TITLE,):
        return ChunkAspect.BIOLOGY
    if titles[:1] == (TRIVIA_TITLE,):
        return ChunkAspect.TRIVIA
    return None


def remove_tags(wikitext: str, names: frozenset[str]) -> str:
    """Return `wikitext` without the elements named in `names`, nested ones included.

    A wiki table is the element `table`. Removing the node keeps a table with blank lines
    inside from splitting into text blocks.
    """
    code = mwparserfromhell.parse(wikitext)
    doomed = code.filter_tags(
        recursive=True, matches=lambda node: str(node.tag).strip().lower() in names
    )
    for tag in doomed:
        if code.contains(tag):
            code.remove(tag)
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


def split_blocks(wikitext: str) -> tuple[Block, ...]:
    """Cut section wikitext into paragraphs and bullet items, in page order.

    Comments, tables and galleries are removed first.
    """
    uncommented = COMMENT_PATTERN.sub("", COMMENT_LINE_PATTERN.sub("", wikitext))
    text = remove_tags(uncommented, REMOVED_TAGS)
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


def match_marker(marker: str, forms: Sequence[ResolvedForm]) -> str | None:
    """Return the id of the form that `marker` names, or `None`.

    Comparison is on `build_node_id()` slugs. The marker first has to equal a name of
    exactly 1 form (`format_form_names`). When it equals none, it has to sit as a whole
    hyphen-separated run in the node name of exactly 1 form. Several matches name nothing.
    """
    marker_slug = build_node_id(marker)
    if not marker_slug:
        return None
    base_name = next(form.name for form in forms if form.is_default)
    equal = [
        form.id
        for form in forms
        if marker_slug in {build_node_id(name) for name in format_form_names(form, base_name)}
    ]
    if equal:
        return equal[0] if len(equal) == 1 else None
    run = f"-{marker_slug}-"
    contained = [form.id for form in forms if run in f"-{build_node_id(form.name)}-"]
    return contained[0] if len(contained) == 1 else None


def pick_biology_subject(
    path: tuple[str, ...], forms: Sequence[ResolvedForm]
) -> tuple[str, PageNote | None]:
    """Return the form id that the Biology section at `path` describes, with a note when a
    title below Biology looks like a form label and matches none.

    The markers are the titles below Biology, deepest first, without "Forms".
    """
    default_id = forms[0].id
    markers = [title for title in reversed(path[1:]) if title.lower() != FORMS_TITLE]
    if not markers:
        return default_id, None
    for marker in markers:
        form_id = match_marker(marker, forms)
        if form_id is not None:
            return form_id, None
    note = PageNote(
        kind=NoteKind.UNMATCHED_FORM_LABEL,
        section_path=PATH_SEPARATOR.join(path),
        detail=f"{markers[0]} matches no form node",
    )
    return default_id, note


def read_prose_section(
    section: Section, aspect: ChunkAspect, forms: Sequence[ResolvedForm]
) -> tuple[list[ChunkFact], list[PageNote], list[str]]:
    """Read the chunks, notes and dropped template names of a Biology, Origin or Trivia section."""
    notes: list[PageNote] = []
    subject_id = forms[0].id
    if aspect is ChunkAspect.BIOLOGY:
        subject_id, note = pick_biology_subject(section.path, forms)
        if note is not None:
            notes.append(note)
    rendered = [render_block(block) for block in split_blocks(section.text)]
    dropped = [name for block in rendered for name in block.dropped]
    kept = merge_short_paragraphs([block for block in rendered if block.text])
    chunks = [
        ChunkFact(
            subject_id=subject_id,
            aspect=aspect,
            section_path=PATH_SEPARATOR.join(section.path),
            text=block.text,
            mentions=block.mentions,
        )
        for block in kept
    ]
    return chunks, notes, dropped


def read_chunks(sections: Sequence[Section], forms: Sequence[ResolvedForm]) -> ChunkReading:
    """Read the chunks of a page. `forms` lists the base form first.

    Pokédex sections give nothing yet.
    """
    chunks: list[ChunkFact] = []
    notes: list[PageNote] = []
    dropped: Counter[str] = Counter()
    for section in sections:
        aspect = read_aspect(section.path)
        if aspect is None or aspect is ChunkAspect.POKEDEX:
            continue
        section_chunks, section_notes, section_dropped = read_prose_section(section, aspect, forms)
        chunks.extend(section_chunks)
        notes.extend(section_notes)
        dropped.update(section_dropped)
    return ChunkReading(tuple(chunks), tuple(notes), dict(dropped))
