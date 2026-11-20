"""Location areas: the headings below a Pokémon or Items section that name a floor, a sector
or a room (L13).

Map:
- roots: only a section titled `Pokémon` or `Items` that sits at the top of the page or directly
  below "In the games" or "In the core series games" is read, so the anime, manga and
  spin-off sections stay out;
- non-areas: `is_area_heading` separates the headings that name a floor from the headings that
  name a generation, a game, an encounter section or an item find section;
- areas: `find_area` gives the deepest area heading of a section path, `read_areas` the unique
  areas of a page in page order.

A heading that names a pinned location points at that location instead of an area. An area
key is the slug of the heading without a trailing parenthetical, so "B1F (Feebas lake)" and
"B1F" are 1 area. `finds.py` and `encounters.py` call `find_area` for every row they read, so
the 3 passes agree on the area of a fact.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.locations.identity import (
    LocationIndex,
    LocationVerdict,
    classify_location,
    normalize_title,
)
from muon.ingestion.bulbapedia.wikitext import TRAILING_NOTE_PATTERN, Section

ROOT_TITLES = frozenset({"pokémon", "items"})
CORE_SECTION_TITLES = frozenset({"in the games", "in the core series games"})
AREA_ID_SEPARATOR = "-"
GAME_PREFIXES = ("pokémon: ", "pokémon ")
GAME_SUFFIXES = (" special demo version", " versions", " version")
GENERATION_PATTERN = re.compile(r"^generation [ivx]+$")
BIOME_SUFFIX = " biome"
GAME_SEPARATOR_PATTERN = re.compile(r",|\band\b|&")
APOSTROPHE_COMMA = "let's go,"
APOSTROPHE_JOINED = "let's go"

GAME_NAMES = frozenset(
    {
        "red",
        "green",
        "blue",
        "yellow",
        "gold",
        "silver",
        "crystal",
        "ruby",
        "sapphire",
        "emerald",
        "firered",
        "leafgreen",
        "diamond",
        "pearl",
        "platinum",
        "heartgold",
        "soulsilver",
        "black",
        "white",
        "black 2",
        "white 2",
        "x",
        "y",
        "omega ruby",
        "alpha sapphire",
        "sun",
        "moon",
        "ultra sun",
        "ultra moon",
        "let's go pikachu!",
        "let's go eevee!",
        "let's go pikachu",
        "let's go eevee",
        "sword",
        "shield",
        "brilliant diamond",
        "shining pearl",
        "legends: arceus",
        "scarlet",
        "violet",
        "legends: z-a",
        "colosseum",
        "xd",
        "xd: gale of darkness",
    }
)

# Headings of the encounter sections (L18) and the item find sections (L15), plus the row
# groups that the pages of 2026-11-20 put at the same level as an area. The report lists
# every area key, so a heading missing here shows up as a wrong area.
NON_AREA_TITLES = frozenset(
    {
        "visible encounters",
        "hidden encounters",
        "wanderers",
        "rare spawns",
        "special encounters",
        "pokémon dens",
        "curry encounters",
        "regina's pokémon",
        "on hand",
        "given away",
        "gives away",
        "gives to new trainers",
        "temporary",
        "first battle",
        "second battle",
        "hidden items",
        "rock smash",
        "found in smashed rocks",
        "fishing",
        "berry tree",
        "berry plant",
        "berry pile",
        "gifts",
        "100-watt gifts",
        "zygarde cube",
        "totem stickers",
        "fake items",
        "natural objects",
        "dropped after ambush encounters",
        "phenomenon",
        "overworld items",
        "regular items",
        "random berries",
        "ore deposits",
        "hotel guest items",
        "gimmighoul coins",
        "flower bed",
        "exchange service corner",
        "pokémon center café",
        "hidden grotto",
        "hidden grottoes",
    }
)


@dataclass(frozen=True)
class AreaRef:
    """The place an item find or an encounter row belongs to.

    `key` is the area key, or empty when the heading names a location: the row then belongs
    to `location_title` itself. `name` is the heading without its parenthetical, or the title
    of that location, and `location_title` the title of the Location the row belongs to: the
    page for an area, the named location otherwise.
    """

    key: str
    name: str
    location_title: str


def read_area_key(heading: str) -> str:
    """Return the heading without a trailing parenthetical, the form the area name keeps."""
    return TRAILING_NOTE_PATTERN.sub("", heading).strip()


def strip_game_prefix(name: str) -> str:
    for prefix in GAME_PREFIXES:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def is_game_heading(heading: str) -> bool:
    """Whether `heading` names a game or a pair of games ("Pokémon Sun and Moon", "Gold,
    Silver and Crystal")."""
    text = heading.lower().replace(APOSTROPHE_COMMA, APOSTROPHE_JOINED)
    for suffix in GAME_SUFFIXES:
        text = text.removesuffix(suffix)
    stripped = (strip_game_prefix(part.strip()) for part in GAME_SEPARATOR_PATTERN.split(text))
    parts = [part for part in stripped if part]
    return bool(parts) and all(part in GAME_NAMES for part in parts)


def is_area_heading(heading: str) -> bool:
    """Whether `heading` can name an area: not a generation, a game, an encounter section, an
    item find section or a biome."""
    text = heading.strip().lower()
    return not (
        GENERATION_PATTERN.match(text)
        or text in NON_AREA_TITLES
        or text.endswith(BIOME_SUFFIX)
        or is_game_heading(text)
    )


def find_area(
    path: tuple[str, ...], section_title: str, page_title: str, index: LocationIndex
) -> AreaRef | None:
    """Return the area of a section path below the `section_title` section, or `None` when
    the rows belong to the page Location.

    `section_title` is `pokémon` or `items`. The headings below its last occurrence in `path`
    are read from the deepest up; the first one that can name an area wins. A heading that
    names a pinned location gives that location with an empty key. A path whose section does
    not sit at the top of the page or below a core games section gives `None`. `page_title`
    is the title of the page the path belongs to.
    """
    lowered = [title.lower() for title in path]
    if section_title not in lowered:
        return None
    position = len(lowered) - 1 - lowered[::-1].index(section_title)
    if not set(lowered[:position]) <= CORE_SECTION_TITLES:
        return None
    for heading in reversed(path[position + 1 :]):
        if not is_area_heading(heading):
            continue
        test = classify_location(index, heading)
        if test.verdict is LocationVerdict.LOCATION:
            return AreaRef(key="", name=test.title, location_title=test.title)
        name = read_area_key(heading)
        return AreaRef(
            key=build_node_id(name), name=name, location_title=normalize_title(page_title)
        )
    return None


def read_areas(
    sections: Sequence[Section], page_title: str, index: LocationIndex
) -> tuple[AreaRef, ...]:
    """Return the unique areas of a page in page order, each named by its first heading.

    A heading that names the page itself or another location gives no area.
    """
    areas: dict[str, AreaRef] = {}
    for section in sections:
        for root in sorted(ROOT_TITLES):
            area = find_area(section.path, root, page_title, index)
            if area is not None and area.key and area.key not in areas:
                areas[area.key] = area
    return tuple(areas.values())


def build_area_id(location_id: str, key: str) -> str:
    """Return the node id of an area: the id of its location, a hyphen and the area key."""
    return f"{location_id}{AREA_ID_SEPARATOR}{key}"
