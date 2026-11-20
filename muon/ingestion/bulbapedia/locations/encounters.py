"""Encounters of a location page: the `catch/entry*` rows of the Pokémon sections (L18 to L20).

Map:
- Sections: the headings below Pokémon give a default kind and method (`SECTION_KINDS`); a
  section of curry encounters or trades is skipped and counted (`SKIPPED_SECTIONS`).
- Rows: a row of a variant in `encounter_templates.CATCH_TEMPLATES` is read by position
  (`read_catch_row`); the other variants have their own readers (`read_sv_row`,
  `read_sv_special_row`, `read_raid_row`, `read_arceus_row`).
- Cells: `read_level_range` reads `22`, `20-24` and `21, 23`, `read_rate` a percentage, and
  `read_method_cell` maps the method cell onto the existing encounter kinds and methods.
- Facts: 1 fact per game flag set to `yes` and per rate cell that is not `0%`. A per-time or
  per-season rate cell gives a `time-*` or `season-*` condition, and the weather columns of
  `catch/entry8` give `weather-*` conditions.
- Skips: a trade, a side game table, a row of another form, a species-less row and a value no
  table maps are counted and left out; `merge_facts` in `batches.py` merges equal ids.

The facts hold Bulbapedia terms only; `batches.py` maps them onto ids and IRIs. This module
knows the pkmn universe, as every ingestion module of a source does.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Template

from muon.ingestion.bulbapedia.locations.areas import AreaRef, find_area, read_headings_below
from muon.ingestion.bulbapedia.locations.encounter_templates import (
    CATCH_TEMPLATES,
    FOOTER_PREFIX,
    HEADER_PREFIX,
    LA_TEMPLATE,
    LA_VERSIONS,
    RAID_TEMPLATE,
    RAID_VERSIONS,
    SEASON_CONDITIONS,
    SIDE_GAME_HEADERS,
    SV_SPECIAL_TEMPLATE,
    SV_TEMPLATE,
    SV_VERSIONS,
    TIME_CONDITIONS,
    WEATHER_CONDITION,
    WEATHER_PARAMETERS,
    ZA_TEMPLATE,
    ZA_VERSIONS,
    CatchTemplate,
    RateLayout,
)
from muon.ingestion.bulbapedia.locations.identity import LocationIndex
from muon.ingestion.bulbapedia.wikitext import Section, normalize_template_name, to_plain_text

POKEMON_TITLE = "pokémon"
ENTRY_PREFIX = "catch/entry"
YES_VALUE = "yes"
ALL_PARAMETER = "all"
FORM_PARAMETER = "form"
BASE_FORMS = frozenset({"", "normal form"})
SPECIES_POSITION = 2
FIRST_VERSION_POSITION = 3
NO_RATE_MARKS = frozenset({"-", "–", "—"})
PRESENT_MARKS = frozenset({"✔", "yes"})
LEVEL_PATTERN = re.compile(r"\d+")
RATE_PATTERN = re.compile(r"^(\d{1,3})\s*%?$")
MAX_CHANCE = 100
UNMAPPED_SAMPLE_LIMIT = 40
SV_UNMAPPED_GROUND = "catch/entry9 without land, overland or watersurface"

KIND_WILD = "wild"
KIND_RAID = "raid"
KIND_STATIC = "static"
KIND_GIFT = "gift"
KIND_SNAG = "snag"
SV_TIME_PARAMETERS = (
    ("morning", TIME_CONDITIONS[0]),
    ("day", TIME_CONDITIONS[1]),
    ("evening", "time-evening"),
    ("night", TIME_CONDITIONS[2]),
)
SV_METHOD_PARAMETERS: Mapping[str, str | None] = {
    "land": None,
    "overland": None,
    "watersurface": "surf",
}
ARCEUS_TIME_POSITIONS = (
    (5, "time-morning"),
    (6, "time-day"),
    (7, "time-evening"),
    (8, "time-night"),
)
ZA_TIME_POSITIONS = ((5, "time-day"), (6, "time-night"))
ZA_ALL_POSITION = 4


class EncounterSkipReason(StrEnum):
    SKIPPED_SECTION = "skipped_section"
    SIDE_GAME = "side_game"
    TRADE = "trade"
    FORM_ROW = "form_row"
    NO_SPECIES = "no_species"
    UNMAPPED_VALUE = "unmapped_value"
    UNKNOWN_TEMPLATE = "unknown_template"


@dataclass(frozen=True)
class KindMethod:
    """The kind and method an encounter has; `method` is `None` for none."""

    kind: str
    method: str | None = None


# Headings are compared in lowercase. A heading in no table gives a plain wild table.
SECTION_KINDS: Mapping[str, KindMethod] = {
    "visible encounters": KindMethod(KIND_WILD),
    "hidden encounters": KindMethod(KIND_WILD),
    "wanderers": KindMethod(KIND_WILD, "wanderer"),
    "rare spawns": KindMethod(KIND_WILD),
    "special encounters": KindMethod(KIND_STATIC),
    "pokémon dens": KindMethod(KIND_RAID, "max-raid"),
}
SKIPPED_SECTIONS = frozenset({"curry encounters"})
TRADE_WORD = "trade"
DEFAULT_KIND = KindMethod(KIND_WILD)

# A method cell that says only where the grass is: the section gives kind and method.
PLAIN_CELLS = frozenset(
    {
        "grass",
        "cave",
        "dark",
        "brown grass",
        "tall grass",
        "sand",
        "building",
        "yellow flowers",
        "red flowers",
        "purple flowers",
        "long grass",
        "dust",
        "dark tall grass",
        "swamp",
        "walking",
        "inside",
        "ceiling",
    }
)
METHOD_CELLS: Mapping[str, KindMethod] = {
    "surf": KindMethod(KIND_WILD, "surf"),
    "fish old": KindMethod(KIND_WILD, "old-rod"),
    "fish good": KindMethod(KIND_WILD, "good-rod"),
    "fish super": KindMethod(KIND_WILD, "super-rod"),
    "fish": KindMethod(KIND_WILD, "fishing"),
    "fishing": KindMethod(KIND_WILD, "fishing"),
    "rock smash": KindMethod(KIND_WILD, "rock-smash"),
    "headbutt": KindMethod(KIND_WILD, "headbutt"),
    "swarm": KindMethod(KIND_WILD, "mass-outbreak"),
    "poké radar": KindMethod(KIND_WILD, "poke-radar"),
    "horde encounter": KindMethod(KIND_WILD, "horde"),
    "grotto": KindMethod(KIND_WILD, "hidden-grotto"),
    "wanderer": KindMethod(KIND_WILD, "wanderer"),
    "gift": KindMethod(KIND_GIFT, "gift"),
    "starter": KindMethod(KIND_GIFT, "gift"),
    "egg": KindMethod(KIND_GIFT, "gift"),
    "helix": KindMethod(KIND_GIFT, "gift"),
    "dome": KindMethod(KIND_GIFT, "gift"),
    "amber": KindMethod(KIND_GIFT, "gift"),
    "skull": KindMethod(KIND_GIFT, "gift"),
    "armor": KindMethod(KIND_GIFT, "gift"),
    "root": KindMethod(KIND_GIFT, "gift"),
    "claw": KindMethod(KIND_GIFT, "gift"),
    "cover": KindMethod(KIND_GIFT, "gift"),
    "plume": KindMethod(KIND_GIFT, "gift"),
    "fixed": KindMethod(KIND_STATIC, "static"),
    "special": KindMethod(KIND_STATIC, "static"),
    "raid": KindMethod(KIND_RAID, "max-raid"),
    "shadow": KindMethod(KIND_SNAG, "snag"),
}
TRADE_CELLS = frozenset({"trade"})


@dataclass(frozen=True)
class EncounterFact:
    """1 species met at 1 place in 1 Version. `area` is `None` for the page Location."""

    species_title: str
    area: AreaRef | None
    version: str
    kind: str
    method: str | None
    min_level: int | None
    max_level: int | None
    chance: int | None
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class RateEntry:
    """1 rate of a row: the conditions it holds under and its chance, `None` when the cell is
    no percentage."""

    conditions: tuple[str, ...]
    chance: int | None
    versions: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EncounterReading:
    """The facts of a page, the rows it skipped per reason and the values no table maps."""

    facts: tuple[EncounterFact, ...]
    skips: Mapping[str, int]
    unmapped: Mapping[str, int]


@dataclass
class RowContext:
    """What a row is read in: the section default, the area and the counters."""

    default: KindMethod
    area: AreaRef | None
    skips: Counter[str]
    unmapped: Counter[str]


def read_level_range(cell: str) -> tuple[int, int] | None:
    """Return the lowest and highest level of a level cell (`22`, `20-24`, `21,23`), or `None`
    when the cell holds no number."""
    levels = [int(number) for number in LEVEL_PATTERN.findall(to_plain_text(cell))]
    return (min(levels), max(levels)) if levels else None


def read_rate(cell: str) -> int | None:
    """Return the percentage of a rate cell, or `None` when the cell is no whole percentage
    from 0 to 100."""
    matched = RATE_PATTERN.match(to_plain_text(cell).strip())
    if matched is None or int(matched.group(1)) > MAX_CHANCE:
        return None
    return int(matched.group(1))


def read_method_cell(cell: str, default: KindMethod) -> KindMethod | EncounterSkipReason:
    """Return the kind and method of a method cell, or the reason the row is skipped. A cell
    that names only the ground gives the section default."""
    value = to_plain_text(cell).strip().lower()
    if value in PLAIN_CELLS:
        return default
    if value in TRADE_CELLS:
        return EncounterSkipReason.TRADE
    return METHOD_CELLS.get(value, EncounterSkipReason.UNMAPPED_VALUE)


def read_section_kind(headings: Sequence[str]) -> KindMethod | EncounterSkipReason:
    """Return the default kind and method of the headings below Pokémon, or the reason the
    section is skipped. The deepest heading in `SECTION_KINDS` wins."""
    lowered = [heading.strip().lower() for heading in headings]
    if any(heading in SKIPPED_SECTIONS for heading in lowered):
        return EncounterSkipReason.SKIPPED_SECTION
    if any(TRADE_WORD in heading for heading in lowered):
        return EncounterSkipReason.TRADE
    return next((SECTION_KINDS[h] for h in reversed(lowered) if h in SECTION_KINDS), DEFAULT_KIND)


def read_positional(template: Template) -> list[str]:
    return [str(param.value).strip() for param in template.params if not param.showkey]


def read_named(template: Template) -> dict[str, str]:
    return {
        str(param.name).strip().lower(): str(param.value).strip()
        for param in template.params
        if param.showkey
    }


def read_cell(positions: Sequence[str], position: int) -> str:
    return positions[position - 1] if len(positions) >= position else ""


def read_flagged_versions(positions: Sequence[str], labels: Sequence[str]) -> tuple[str, ...]:
    """Return the labels of the game columns whose flag is `yes`."""
    return tuple(
        label
        for offset, label in enumerate(labels)
        if read_cell(positions, FIRST_VERSION_POSITION + offset).lower() == YES_VALUE
    )


def read_entry_rates(
    template: CatchTemplate, positions: Sequence[str], named: Mapping[str, str]
) -> list[RateEntry]:
    """Return the rates of a row in the layout of its template."""
    if ALL_PARAMETER in named:
        return drop_absent_entry(named[ALL_PARAMETER])
    cells = [read_cell(positions, template.rate_position + offset) for offset in range(4)]
    layout = template.rate_layout
    if layout is RateLayout.SINGLE:
        return drop_absent_entry(cells[0])
    if layout is RateLayout.PER_VERSION:
        return [
            RateEntry((), read_rate(cell), (label,))
            for cell, label in zip(cells, template.version_labels, strict=False)
            if cell.strip() not in NO_RATE_MARKS and read_rate(cell) != 0
        ]
    names = TIME_CONDITIONS if layout is RateLayout.PER_TIME else SEASON_CONDITIONS
    return split_by_condition(cells[: len(names)], names)


def drop_absent_entry(cell: str) -> list[RateEntry]:
    """Return the 1 entry of a rate cell, or none when the cell says there is no encounter
    (`0%` or a dash). An empty or unreadable cell gives an entry with no chance."""
    rate = read_rate(cell)
    if cell.strip() in NO_RATE_MARKS or rate == 0:
        return []
    return [RateEntry((), rate)]


def split_by_condition(cells: Sequence[str], names: Sequence[str]) -> list[RateEntry]:
    """Return 1 entry per condition whose rate is not `0%`; equal rates in every cell, or no
    rate at all, give 1 entry with no condition."""
    rates = [read_rate(cell) for cell in cells]
    if len(set(cells)) == 1 and rates[0] != 0:
        return [RateEntry((), rates[0])]
    return [
        RateEntry((name,), rate)
        for name, cell, rate in zip(names, cells, rates, strict=True)
        if cell.strip() not in NO_RATE_MARKS and rate != 0
    ]


def read_weather_entries(named: Mapping[str, str]) -> list[RateEntry]:
    """Return 1 entry per weather column of a `catch/entry8` row that holds a rate."""
    entries: list[RateEntry] = []
    for name in WEATHER_PARAMETERS:
        cell = named.get(name, "").strip()
        if cell.lower() in {*NO_RATE_MARKS, "", "na"} or read_rate(cell) == 0:
            continue
        entries.append(RateEntry((WEATHER_CONDITION.format(name=name),), read_rate(cell)))
    return entries


def build_facts(
    species: str,
    versions: Sequence[str],
    kind_method: KindMethod,
    levels: tuple[int, int] | None,
    entries: Sequence[RateEntry],
    context: RowContext,
) -> list[EncounterFact]:
    """Return the facts of a row: 1 per rate entry and per game the entry holds for."""
    facts: list[EncounterFact] = []
    for entry in entries:
        for version in versions:
            if entry.versions is not None and version not in entry.versions:
                continue
            facts.append(
                EncounterFact(
                    species_title=species,
                    area=context.area,
                    version=version,
                    kind=kind_method.kind,
                    method=kind_method.method,
                    min_level=levels[0] if levels else None,
                    max_level=levels[1] if levels else None,
                    chance=entry.chance,
                    conditions=entry.conditions,
                )
            )
    return facts


def read_row_species(
    positions: Sequence[str], named: Mapping[str, str], context: RowContext
) -> str | None:
    """Return the species name of a row, or `None` after counting why the row is skipped."""
    species = to_plain_text(read_cell(positions, SPECIES_POSITION))
    if not species:
        context.skips[EncounterSkipReason.NO_SPECIES] += 1
        return None
    if named.get(FORM_PARAMETER, "").lower() not in BASE_FORMS:
        context.skips[EncounterSkipReason.FORM_ROW] += 1
        return None
    return species


def read_cell_kind(cell: str, context: RowContext) -> KindMethod | None:
    """Return the kind and method of a method cell, or `None` after counting the skip."""
    reading = read_method_cell(cell, context.default)
    if isinstance(reading, EncounterSkipReason):
        context.skips[reading] += 1
        if reading is EncounterSkipReason.UNMAPPED_VALUE:
            context.unmapped[to_plain_text(cell).strip()] += 1
        return None
    return reading


def read_catch_row(
    template: Template, spec: CatchTemplate, context: RowContext
) -> list[EncounterFact]:
    """Read a row of a variant of the shared layout."""
    positions, named = read_positional(template), read_named(template)
    species = read_row_species(positions, named, context)
    kind_method = read_cell_kind(read_cell(positions, spec.method_position), context)
    if species is None or kind_method is None:
        return []
    levels = read_level_range(read_cell(positions, spec.level_position))
    weather = read_weather_entries(named)
    entries = weather or read_entry_rates(spec, positions, named)
    versions = read_flagged_versions(positions, spec.version_labels)
    return build_facts(species, versions, kind_method, levels, entries, context)


def read_sv_row(template: Template, context: RowContext) -> list[EncounterFact]:
    """Read a Scarlet and Violet row: 2 game flags, the level, the rate, then named flags for
    the ground and the time of day."""
    positions, named = read_positional(template), read_named(template)
    species = read_row_species(positions, named, context)
    if species is None:
        return []
    ground = [key for key in SV_METHOD_PARAMETERS if named.get(key, "").lower() == YES_VALUE]
    if not ground:
        context.skips[EncounterSkipReason.UNMAPPED_VALUE] += 1
        context.unmapped[SV_UNMAPPED_GROUND] += 1
        return []
    method = SV_METHOD_PARAMETERS[ground[0]] or context.default.method
    kind_method = KindMethod(context.default.kind, method)
    levels = read_level_range(read_cell(positions, 5))
    chance = read_rate(named.get(ALL_PARAMETER, read_cell(positions, 6)))
    present = tuple(
        condition
        for key, condition in SV_TIME_PARAMETERS
        if named.get(key, "").lower() == YES_VALUE
    )
    conditions = present if 0 < len(present) < len(SV_TIME_PARAMETERS) else ()
    versions = read_flagged_versions(positions, SV_VERSIONS)
    entries = [] if chance == 0 else [RateEntry(conditions, chance)]
    return build_facts(species, versions, kind_method, levels, entries, context)


def read_sv_special_row(template: Template, context: RowContext) -> list[EncounterFact]:
    """Read a Scarlet and Violet fixed encounter: 2 game flags, a place note and the level."""
    positions, named = read_positional(template), read_named(template)
    species = read_row_species(positions, named, context)
    if species is None:
        return []
    levels = read_level_range(read_cell(positions, 6))
    versions = read_flagged_versions(positions, SV_VERSIONS)
    entries = [RateEntry((), None)]
    return build_facts(
        species, versions, KindMethod(KIND_STATIC, "static"), levels, entries, context
    )


def read_raid_row(template: Template, context: RowContext) -> list[EncounterFact]:
    """Read a Sword and Shield den row: 2 game flags and star columns that no field keeps."""
    positions, named = read_positional(template), read_named(template)
    species = read_row_species(positions, named, context)
    if species is None:
        return []
    versions = read_flagged_versions(positions, RAID_VERSIONS)
    return build_facts(
        species, versions, KindMethod(KIND_RAID, "max-raid"), None, [RateEntry((), None)], context
    )


def read_marked_conditions(
    positions: Sequence[str],
    named: Mapping[str, str],
    marks: Sequence[tuple[int, str]],
    all_position: int | None,
) -> tuple[str, ...]:
    """Return the time conditions of the ✔ cells, or none when the row holds for all times:
    a named `all`, or a ✔ in the cell at `all_position` when the template has one."""
    if ALL_PARAMETER in named:
        return ()
    if all_position is not None and read_cell(positions, all_position) in PRESENT_MARKS:
        return ()
    present = tuple(
        condition
        for position, condition in marks
        if read_cell(positions, position) in PRESENT_MARKS
    )
    return present if len(present) < len(marks) else ()


def read_arceus_row(template: Template, context: RowContext, is_za: bool) -> list[EncounterFact]:
    """Read a Legends: Arceus or Legends: Z-A row: a level cell, then ✔ cells for the time of
    day. The rows hold no rate. Only the first level cell is read."""
    positions, named = read_positional(template), read_named(template)
    species = read_row_species(positions, named, context)
    if species is None:
        return []
    marks = ZA_TIME_POSITIONS if is_za else ARCEUS_TIME_POSITIONS
    conditions = read_marked_conditions(positions, named, marks, ZA_ALL_POSITION if is_za else None)
    levels = read_level_range(read_cell(positions, 3))
    versions = ZA_VERSIONS if is_za else LA_VERSIONS
    return build_facts(
        species, versions, context.default, levels, [RateEntry(conditions, None)], context
    )


def read_row(template: Template, context: RowContext) -> list[EncounterFact]:
    """Read 1 `catch/entry*` row with the reader of its variant."""
    name = normalize_template_name(template.name.strip_code())
    spec = CATCH_TEMPLATES.get(name)
    if spec is not None:
        return read_catch_row(template, spec, context)
    if name == SV_TEMPLATE:
        return read_sv_row(template, context)
    if name == SV_SPECIAL_TEMPLATE:
        return read_sv_special_row(template, context)
    if name == RAID_TEMPLATE:
        return read_raid_row(template, context)
    if name in (LA_TEMPLATE, ZA_TEMPLATE):
        return read_arceus_row(template, context, is_za=name == ZA_TEMPLATE)
    context.skips[EncounterSkipReason.UNKNOWN_TEMPLATE] += 1
    context.unmapped[name] += 1
    return []


def read_section_rows(
    section: Section,
    default: KindMethod,
    area: AreaRef | None,
    skips: Counter[str],
    unmapped: Counter[str],
) -> list[EncounterFact]:
    """Read the rows of 1 section in order, skipping the tables of a side game header."""
    context = RowContext(default, area, skips, unmapped)
    facts: list[EncounterFact] = []
    is_side_game = False
    for template in mwparserfromhell.parse(section.text).filter_templates(recursive=False):
        name = normalize_template_name(template.name.strip_code())
        if name.startswith(HEADER_PREFIX):
            is_side_game = name in SIDE_GAME_HEADERS
        elif name.startswith(FOOTER_PREFIX):
            is_side_game = False
        elif name.startswith(ENTRY_PREFIX):
            if is_side_game:
                skips[EncounterSkipReason.SIDE_GAME] += 1
            else:
                facts.extend(read_row(template, context))
    return facts


def read_encounters(
    sections: Sequence[Section], page_title: str, index: LocationIndex
) -> EncounterReading:
    """Read the encounters of a page in page order."""
    facts: list[EncounterFact] = []
    skips: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    for section in sections:
        headings = read_headings_below(section.path, POKEMON_TITLE)
        if headings is None or ENTRY_PREFIX not in section.text.lower():
            continue
        default = read_section_kind(headings)
        if isinstance(default, EncounterSkipReason):
            skips[default] += section.text.lower().count(ENTRY_PREFIX)
            continue
        area = find_area(section.path, POKEMON_TITLE, page_title, index)
        page_area = area if area is not None and area.key else None
        facts.extend(read_section_rows(section, default, page_area, skips, unmapped))
    return EncounterReading(
        facts=tuple(facts),
        skips={reason.value: skips[reason] for reason in EncounterSkipReason},
        unmapped=dict(
            sorted(unmapped.items(), key=lambda pair: (-pair[1], pair[0]))[:UNMAPPED_SAMPLE_LIMIT]
        ),
    )
