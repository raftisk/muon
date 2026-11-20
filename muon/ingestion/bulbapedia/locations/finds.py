"""Item finds of a location page: the `Itemlist` rows of the Items sections (L15, L16).

Map:
- Rows: `Itemlist` holds 1 item and `Itemlist/N` holds N; each game flag set to `yes` makes 1
  find of that item in that Version, so a row of 5 items and 2 flags gives 10 facts. The
  unnamed parameter of an item is the name of its sprite ("Bike BDSP"); the item is the page
  that `display=` (`display2=` and so on) links to, and the sprite name when it has none.
- Headings: the headings below Items say how the item lies there; `FIND_HEADINGS` maps a
  heading to a hidden find or a find method, and `SKIPPED_FIND_HEADINGS` names the rows that
  are no finds (Zygarde Cubes, Totem Stickers, fake items).
- Areas: `areas.find_area` gives the area of the section, the same one the encounters use.
- Skips: a TM, HM or TR row (`machine_find`), a flag no table row names
  (`unknown_version_code`), a row with no item name and a skipped heading are counted and
  left out.

The facts hold Bulbapedia terms only; `batches.py` maps them onto ids and IRIs. This module
knows the pkmn universe, as every ingestion module of a source does.
"""

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Template

from muon.ingestion.bulbapedia.locations.areas import AreaRef, find_area, read_headings_below
from muon.ingestion.bulbapedia.locations.identity import LocationIndex
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    normalize_template_name,
    read_params,
    to_plain_text,
)

ITEMS_TITLE = "items"
ITEMLIST_NAME = "itemlist"
ITEMLIST_PREFIX = "itemlist/"
YES_VALUE = "yes"
FIND_METHOD_ROCK_SMASH = "rock-smash"
FIND_METHOD_FISHING = "fishing"
FIND_METHOD_BERRY_TREE = "berry-tree"
FIND_METHOD_GIFT = "gift"
MACHINE_ITEM_PATTERN = re.compile(r"^(TM|HM|TR)(\d| )")
MACHINE_TEMPLATES = frozenset({"tm", "hm", "tr"})
QUANTITY_PATTERN = re.compile(r"\s*[×x]\s*\d+\s*$")
ITEM_TITLE_SUFFIX = " (item)"
DISPLAY_PARAM = "display"
IGNORED_PARAM_PATTERN = re.compile(r"^(display\d*|sprite|size)$")

# Game flag of an `Itemlist` row to the A-Box Version label. `JR` and `JG`, the Japanese Red
# and Green, name no Version and are counted as unknown codes. `check_version_codes` checks
# every label against the A-Box before any page is read.
VERSION_CODES: Mapping[str, str] = {
    "R": "Red",
    "B": "Blue",
    "Y": "Yellow",
    "G": "Gold",
    "S": "Silver",
    "C": "Crystal",
    "Ru": "Ruby",
    "Sa": "Sapphire",
    "E": "Emerald",
    "FR": "FireRed",
    "LG": "LeafGreen",
    "D": "Diamond",
    "P": "Pearl",
    "Pt": "Platinum",
    "HG": "HeartGold",
    "SS": "SoulSilver",
    "Bl": "Black",
    "W": "White",
    "B2": "Black 2",
    "W2": "White 2",
    "X": "X",
    "y": "Y",
    "OR": "Omega Ruby",
    "AS": "Alpha Sapphire",
    "Su": "Sun",
    "M": "Moon",
    "US": "Ultra Sun",
    "UM": "Ultra Moon",
    "LP": "Let's Go Pikachu",
    "LE": "Let's Go Eevee",
    "Sw": "Sword",
    "Sh": "Shield",
    "BD": "Brilliant Diamond",
    "SP": "Shining Pearl",
    "LA": "Legends: Arceus",
    "Sc": "Scarlet",
    "V": "Violet",
    "ZA": "Legends: Z-A",
    "Colo": "Colosseum",
    "XD": "XD",
}


class FindSkipReason(StrEnum):
    SKIPPED_HEADING = "skipped_heading"
    MACHINE_FIND = "machine_find"
    UNKNOWN_VERSION_CODE = "unknown_version_code"
    NO_ITEM_NAME = "no_item_name"


@dataclass(frozen=True)
class FindKind:
    """How an item lies in a place: hidden, or found by a method."""

    is_hidden: bool = False
    method: str | None = None


# Headings are compared in lowercase.
FIND_HEADINGS: Mapping[str, FindKind] = {
    "hidden items": FindKind(is_hidden=True),
    "rock smash": FindKind(method=FIND_METHOD_ROCK_SMASH),
    "found in smashed rocks": FindKind(method=FIND_METHOD_ROCK_SMASH),
    "fishing": FindKind(method=FIND_METHOD_FISHING),
    "berry tree": FindKind(method=FIND_METHOD_BERRY_TREE),
    "berry plant": FindKind(method=FIND_METHOD_BERRY_TREE),
    "berry pile": FindKind(method=FIND_METHOD_BERRY_TREE),
    "gifts": FindKind(method=FIND_METHOD_GIFT),
    "100-watt gifts": FindKind(method=FIND_METHOD_GIFT),
}
SKIPPED_FIND_HEADINGS = frozenset({"zygarde cube", "totem stickers", "fake items"})


@dataclass(frozen=True)
class FindFact:
    """1 item found at 1 place in 1 Version. `area` is `None` for the page Location."""

    item_name: str
    area: AreaRef | None
    version: str
    method: str | None
    is_hidden: bool


@dataclass(frozen=True)
class FindReading:
    """The facts of a page and the rows it skipped, counted per reason."""

    facts: tuple[FindFact, ...]
    skips: Mapping[str, int]


class FindTableError(Exception):
    """`VERSION_CODES` names a Version the A-Box lacks."""


def check_version_codes(versions: Collection[str]) -> None:
    """Raise `FindTableError` when `VERSION_CODES` names a label outside `versions`, the
    Version labels of the A-Box."""
    missing = sorted(set(VERSION_CODES.values()) - set(versions))
    if missing:
        raise FindTableError(f"Item find version codes name versions the A-Box lacks: {missing}")


def read_find_headings(headings: Sequence[str]) -> tuple[FindKind | None, bool]:
    """Return the find kind the headings give and whether one of them skips the rows.

    The deepest heading in `FIND_HEADINGS` wins; a heading in `SKIPPED_FIND_HEADINGS` anywhere
    skips the rows.
    """
    lowered = [heading.strip().lower() for heading in headings]
    if any(heading in SKIPPED_FIND_HEADINGS for heading in lowered):
        return None, True
    kind = next((FIND_HEADINGS[h] for h in reversed(lowered) if h in FIND_HEADINGS), None)
    return kind, False


def read_display_name(value: str) -> str | None:
    """Return the item a `display=` value names: the 1st page it links to without an item
    suffix, else its plain text without a quantity such as `×2`. A TM, HM or TR template gives
    the first word of the machine page name, which `MACHINE_ITEM_PATTERN` matches."""
    code = mwparserfromhell.parse(value)
    for template in code.filter_templates(recursive=False):
        if normalize_template_name(template.name.strip_code()) in MACHINE_TEMPLATES:
            return f"{normalize_template_name(template.name.strip_code()).upper()} "
    for link in code.filter_wikilinks():
        title = str(link.title).split("#", 1)[0].strip()
        if title:
            return title.removesuffix(ITEM_TITLE_SUFFIX)
    text = QUANTITY_PATTERN.sub("", to_plain_text(value)).strip()
    return text or None


def read_row_items(template: Template, count: int) -> tuple[str, ...]:
    """Return the item names of a row, 1 per unnamed parameter up to `count`; an empty string
    stands for a row with no item name. The item of a named `display` replaces its sprite name
    unless the sprite name is empty."""
    params = read_params(template)
    positional = [str(param.value).strip() for param in template.params if not param.showkey]
    names: list[str] = []
    for position, sprite in enumerate(positional[:count], start=1):
        display_key = DISPLAY_PARAM if position == 1 else f"{DISPLAY_PARAM}{position}"
        display = read_display_name(params.get(display_key, "")) if sprite else None
        names.append(display or to_plain_text(sprite))
    return tuple(names)


def read_row_count(template: Template) -> int | None:
    """Return how many items an `Itemlist` or `Itemlist/N` row holds, or `None` for another
    template."""
    name = normalize_template_name(template.name.strip_code())
    if name == ITEMLIST_NAME:
        return 1
    if name.startswith(ITEMLIST_PREFIX) and name.removeprefix(ITEMLIST_PREFIX).isdigit():
        return int(name.removeprefix(ITEMLIST_PREFIX))
    return None


def read_row_flags(template: Template) -> tuple[list[str], list[str]]:
    """Return the Version labels the row's game flags name and the flags no table names."""
    versions: list[str] = []
    unknown: list[str] = []
    for key, value in read_params(template).items():
        if key.isdigit() or IGNORED_PARAM_PATTERN.match(key) or value.lower() != YES_VALUE:
            continue
        if key in VERSION_CODES:
            versions.append(VERSION_CODES[key])
        else:
            unknown.append(key)
    return versions, unknown


def read_section_rows(
    section: Section, kind: FindKind | None, area: AreaRef | None, skips: Counter[str]
) -> list[FindFact]:
    """Read the `Itemlist` rows of 1 section into facts, counting what it skips."""
    facts: list[FindFact] = []
    for template in mwparserfromhell.parse(section.text).filter_templates():
        count = read_row_count(template)
        if count is None:
            continue
        versions, unknown = read_row_flags(template)
        skips[FindSkipReason.UNKNOWN_VERSION_CODE] += len(unknown)
        for item_name in read_row_items(template, count):
            if not item_name:
                skips[FindSkipReason.NO_ITEM_NAME] += 1
            elif MACHINE_ITEM_PATTERN.match(item_name):
                skips[FindSkipReason.MACHINE_FIND] += len(versions)
            else:
                facts.extend(
                    FindFact(
                        item_name=item_name,
                        area=area,
                        version=version,
                        method=kind.method if kind else None,
                        is_hidden=kind.is_hidden if kind else False,
                    )
                    for version in versions
                )
    return facts


def read_item_finds(
    sections: Sequence[Section], page_title: str, index: LocationIndex
) -> FindReading:
    """Read the item finds of a page in page order, 1 fact per item, place and Version."""
    facts: list[FindFact] = []
    skips: Counter[str] = Counter()
    for section in sections:
        headings = read_headings_below(section.path, ITEMS_TITLE)
        if headings is None:
            continue
        kind, is_skipped = read_find_headings(headings)
        if is_skipped:
            rows = read_section_rows(section, None, None, Counter())
            skips[FindSkipReason.SKIPPED_HEADING] += len(rows)
            continue
        area = find_area(section.path, ITEMS_TITLE, page_title, index)
        facts.extend(read_section_rows(section, kind, area, skips))
    return FindReading(tuple(facts), {reason.value: skips[reason] for reason in FindSkipReason})
