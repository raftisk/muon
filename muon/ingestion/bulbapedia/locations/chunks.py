"""Text chunks of a location page: the lead, the geography sections, Trivia and Origin.

Map:
- Aspect: the lead and Description, Geography, Route description, Places of interest and
  Layout, with their subsections, give `geography` chunks; Trivia gives `trivia` and Origin,
  at the top or inside Trivia, gives `origin` (`read_location_aspect`). Every other section
  gives nothing.
- Sections: Description and Geography may sit below "In the games" or "In the core series
  games", so those titles are skipped when the path is matched.
- Chunks: the shared prose rules of `chunking.py` cut each section; the lead path is named
  "Lead" in the chunk.

Every chunk describes the Location of the page. The facts hold Bulbapedia terms only;
`batches.py` maps them onto ids and IRIs. This module knows the pkmn universe, as every
ingestion module of a source does.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from muon.ingestion.bulbapedia.chunking import (
    ChunkAspect,
    ChunkFact,
    find_long_chunks,
    read_prose_section,
)
from muon.ingestion.bulbapedia.locations.areas import CORE_SECTION_TITLES
from muon.ingestion.bulbapedia.wikitext import Section

GEOGRAPHY_TITLES = frozenset(
    {"description", "geography", "route description", "places of interest", "layout"}
)
TRIVIA_TITLE = "trivia"
ORIGIN_TITLE = "origin"
LEAD_SECTION = "Lead"


@dataclass(frozen=True)
class LocationChunkReading:
    """The chunks of a location page, what their cleanup removed or kept, and the chunks
    above `LONG_CHUNK_LENGTH` that a reviewer should see."""

    chunks: tuple[ChunkFact, ...]
    dropped_templates: Mapping[str, int]
    removed_tables: int
    math_formulas: int
    long_chunks: tuple[ChunkFact, ...]


def read_location_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
    """Return the aspect of the prose section at `path`, or `None` when it gives no chunk.

    The lead has the empty path. Titles compare in lowercase.
    """
    if not path:
        return ChunkAspect.GEOGRAPHY
    titles = [title.lower() for title in path]
    while titles and titles[0] in CORE_SECTION_TITLES:
        titles.pop(0)
    if not titles:
        return None
    if ORIGIN_TITLE in titles and titles[0] in {ORIGIN_TITLE, TRIVIA_TITLE}:
        return ChunkAspect.ORIGIN
    if titles[0] == TRIVIA_TITLE:
        return ChunkAspect.TRIVIA
    if titles[0] in GEOGRAPHY_TITLES:
        return ChunkAspect.GEOGRAPHY
    return None


def read_location_chunks(sections: Sequence[Section], location_id: str) -> LocationChunkReading:
    """Read the chunks of a location page in page order."""
    chunks: list[ChunkFact] = []
    dropped: Counter[str] = Counter()
    removed_tables = 0
    math_formulas = 0
    for section in sections:
        aspect = read_location_aspect(section.path)
        if aspect is None:
            continue
        reading = read_prose_section(section, aspect, location_id)
        chunks.extend(
            fact.model_copy(update={"section_path": LEAD_SECTION}) if not section.path else fact
            for fact in reading.chunks
        )
        dropped.update(reading.dropped)
        removed_tables += reading.removed_tables
        math_formulas += reading.math_formulas
    return LocationChunkReading(
        chunks=tuple(chunks),
        dropped_templates=dict(sorted(dropped.items())),
        removed_tables=removed_tables,
        math_formulas=math_formulas,
        long_chunks=find_long_chunks(chunks),
    )
