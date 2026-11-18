"""Text chunks of an item page: Effect, Trivia and Description become `ChunkFact`s.

Map:
- Aspect: Effect and its subsections give `effect` chunks, except the Pokémon Champions,
  contest and ingredient subsections; Trivia gives `trivia` chunks (`read_item_aspect`). Both
  follow the shared prose rules of `chunking.py`.
- Description: `descriptions.py` reads the `movedescentry` rows, the same template move pages
  use. Rows with the same text give 1 chunk with the union of their version groups
  (`merge_flavor_rows`).

The sections come from `rebase_core_sections`, so "Effect > Held item" and "Description >
Tag information" read like the sections of a move page. Every chunk describes the item of the
page. The facts hold Bulbapedia terms only; `batches.py` maps them onto ids and IRIs. This
module knows the pkmn universe, as every ingestion module of a source does.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from muon.ingestion.bulbapedia.chunking import (
    ChunkAspect,
    ChunkFact,
    find_long_chunks,
    merge_flavor_rows,
    read_prose_section,
)
from muon.ingestion.bulbapedia.descriptions import UnknownGameCode, read_description_rows
from muon.ingestion.bulbapedia.wikitext import Section

EFFECT_TITLE = "effect"
TRIVIA_TITLE = "trivia"
EXCLUDED_EFFECT_WORDS = ("champions", "contest", "ingredient")


@dataclass(frozen=True)
class ItemChunkReading:
    """The chunks of an item page, what its cleanup removed or kept, and what a reviewer
    should see: game code cells no rule reads and chunks above `LONG_CHUNK_LENGTH`."""

    chunks: tuple[ChunkFact, ...]
    dropped_templates: Mapping[str, int]
    removed_tables: int
    math_formulas: int
    unknown_codes: tuple[UnknownGameCode, ...]
    long_chunks: tuple[ChunkFact, ...]


def read_item_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
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


def read_item_chunks(sections: Sequence[Section], item_id: str) -> ItemChunkReading:
    """Read the chunks of an item page: prose chunks in page order, then flavor chunks."""
    prose: list[ChunkFact] = []
    dropped: Counter[str] = Counter()
    removed_tables = 0
    math_formulas = 0
    for section in sections:
        aspect = read_item_aspect(section.path)
        if aspect is None:
            continue
        reading = read_prose_section(section, aspect, item_id)
        prose.extend(reading.chunks)
        dropped.update(reading.dropped)
        removed_tables += reading.removed_tables
        math_formulas += reading.math_formulas
    description = read_description_rows(sections)
    dropped.update(description.dropped)
    chunks = (*prose, *merge_flavor_rows(description.rows, item_id))
    return ItemChunkReading(
        chunks=chunks,
        dropped_templates=dict(sorted(dropped.items())),
        removed_tables=removed_tables,
        math_formulas=math_formulas,
        unknown_codes=description.unknown_codes,
        long_chunks=find_long_chunks(chunks),
    )
