"""Encounters of a species page: the Game locations table becomes `EncounterFact`s.

Map:
- Rows: the `Availability/Entry1` and `Entry2` templates of the main table, the own text
  of the Game locations section. `/None` and `NA` rows give nothing (`read_rows`).
- Versions: `v` and `v2` name A-Box versions; DLC rows map to their base versions and
  pseudo-versions are skipped (`resolve_versions`). A Pal Park row gives its own place.
- Parts: an area splits at `<br>` (`split_parts`). A part whose 1st link is an acquisition
  with no place, or whose text starts with "Unobtainable", gives nothing.
- Links: every wikilink and location template of a part is read as a `LinkRef`
  (`read_links`). A link is a method marker, an acquisition or a location candidate.
- Locations: `locations.py` tests every candidate. A bold `'''[[X]]:'''` prefix counts
  only when no other candidate of the part is a location.
- Methods: every location of a part gets every method of the part; no method means kind
  `wild` (`METHOD_KINDS`). A `(...)` group with no marker is counted as unread.
- Forms: a `<small>` caption names the form through the chunk marker rule.

The facts hold Bulbapedia terms only: titles and version labels. `batches.py` maps them
onto ids. Every outcome a reviewer should see is counted in `EncounterCounts`.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from mwparserfromhell.nodes import Node, Tag, Template, Text, Wikilink
from mwparserfromhell.wikicode import Wikicode

from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.runs import ReportModel
from muon.ingestion.bulbapedia.species.chunks import match_marker
from muon.ingestion.bulbapedia.species.extract import EncounterFact, SpeciesPageExtract
from muon.ingestion.bulbapedia.species.forms import ResolvedForm
from muon.ingestion.bulbapedia.species.locations import (
    LocationIndex,
    LocationVerdict,
    classify_location,
    normalize_title,
)
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    normalize_template_name,
    read_small_captions,
    split_sections,
    to_plain_text,
)

GAME_LOCATIONS_TITLE = "game locations"
ENTRY_TEMPLATES = frozenset({"availability/entry1", "availability/entry2"})
NONE_TEMPLATES = frozenset({"availability/entry1/none", "availability/entry2/none"})
NA_TEMPLATE = "availability/na"
VERSION_PARAMS = ("v", "v2")
AREA_PARAM = "area"
JAPAN_PARAM = "ex"
BREAK_TAG = "br"
BOLD_TAG = "b"
PREFIX_MARK = ":"
ANCHOR_MARK = "#"
DENS_SUFFIX = "/Dens"  # the Max Raid dens of a Wild Area place have a subpage
OPEN_GROUP = "("
CLOSE_GROUP = ")"
CAPTION_PARENS = "() "

ROW_ENTRY = "entry"
ROW_NONE = "none"
ROW_NA = "na"

# DLC rows: the area belongs to the base game, which has the Version.
VERSION_ROWS: Mapping[str, tuple[str, ...]] = {
    "Expansion Pass": ("Sword", "Shield"),
    "Sword Expansion Pass": ("Sword",),
    "Shield Expansion Pass": ("Shield",),
    "The Hidden Treasure of Area Zero": ("Scarlet", "Violet"),
    "The Hidden Treasure of Area Zero (Scarlet)": ("Scarlet",),
    "The Hidden Treasure of Area Zero (Violet)": ("Violet",),
    "Mega Dimension": ("Legends: Z-A",),
}
# A website, a pedometer accessory and a mobile side game: no place inside a game.
SKIPPED_VERSIONS = frozenset({"Dream World", "Pokéwalker", "Masters EX"})


@dataclass(frozen=True)
class PalParkRow:
    """A row whose `location_param` names the place and whose versions the table gives."""

    version: str
    location_param: str
    versions: tuple[str, ...]
    kind: str


PAL_PARK_ROW = PalParkRow(
    version="Pal Park",
    location_param="link",
    versions=("Diamond", "Pearl", "Platinum", "HeartGold", "SoulSilver"),
    kind="migration",
)


@dataclass(frozen=True)
class LocationTemplate:
    """A template that links a place: `pattern` takes its positional parameters by number."""

    params: tuple[str, ...]
    pattern: str


ROUTE = LocationTemplate(params=("1", "2"), pattern="{2} Route {1}")
LOCATION_TEMPLATES: Mapping[str, LocationTemplate] = {
    "rt": ROUTE,
    "rtn": ROUTE,
    "safari": LocationTemplate(params=("1",), pattern="{1} Safari Zone"),
    "ka": LocationTemplate(params=("1",), pattern="{1} (Kanto)"),
    "ho": LocationTemplate(params=("1",), pattern="{1} (Hoenn)"),
    "si": LocationTemplate(params=("1",), pattern="{1} (Sinnoh)"),
    "kal": LocationTemplate(params=("1",), pattern="{1} (Kalos)"),
    "al": LocationTemplate(params=("1",), pattern="{1} (Alola)"),
    "pal": LocationTemplate(params=("1",), pattern="{1} (Paldea)"),
    "fb": LocationTemplate(params=("1", "2"), pattern="{1} {2}"),
    "obp": LocationTemplate(params=("1", "2"), pattern="{1} ({2})"),
}
SECTION_LINK_TEMPLATE = "dl"  # {{DL|page|section|display}} links page#section
POKEMON_LINK_TEMPLATE = "pkmn2"  # {{pkmn2|Roaming}} links "Roaming Pokémon"
POKEMON_LINK_PATTERN = "{1} Pokémon"
MOVE_TEMPLATE = "m"
SHADOW_TEMPLATE = "color2"
SHADOW_COLOR = "{{shadow color}}"
SNAG = "snag"

# Links whose page tells how to get a species but names no place.
ACQUISITION_LINKS: Mapping[str, str] = {
    "Evolution": "evolution",
    "Trade": "trade",
    "In-game trade": "in-game-trade",
    "Pokémon breeding": "breeding",
    "Poké Transfer": "transfer",
    "Pokémon Bank": "bank",
    "Pokémon HOME": "home",
    "Unova Link": "unova-link",
    "Poké Pelago": "poke-pelago",
    "Event Pokémon": "event",
}
IN_EVENTS_ANCHOR = "In events"
EVENT_KIND = "event"
UNOBTAINABLE_PREFIX = "Unobtainable"
UNOBTAINABLE_KIND = "unobtainable"

METHOD_LINKS: Mapping[str, str] = {
    "Super Rod": "super-rod",
    "Good Rod": "good-rod",
    "Old Rod": "old-rod",
    "Fishing": "fishing",
    "Headbutt tree": "headbutt",
    "SOS Battle": "sos",
    "Horde Encounter": "horde",
    "Hidden Grotto": "hidden-grotto",
    "Poké Radar": "poke-radar",
    "Mass outbreak": "mass-outbreak",
    "Wanderer": "wanderer",
    "Max Raid Battle": "max-raid",
    "Dynamax Adventure": "dynamax-adventure",
    "Tera Raid Battle": "tera-raid",
    "DexNav": "dexnav",
    "Roaming Pokémon": "roaming",
    "Gift Pokémon": "gift",
}
METHOD_MOVES: Mapping[str, str] = {
    "Surf": "surf",
    "Surfing": "surf",
    "Rock Smash": "rock-smash",
}
EVENT_LIST_PREFIX = "List of in-game event Pokémon"
EVENT_LIST_LABEL = "event list: "
STATIC_DISPLAY_PATTERN = re.compile(r"(only )?(one|two|three|four)|one of each form|weekly")
GIFT_DISPLAYS = frozenset({"received", "reward", "buy"})
STATIC = "static"
GIFT = "gift"

METHOD_KINDS: Mapping[str, str] = {
    "surf": "wild",
    "old-rod": "wild",
    "good-rod": "wild",
    "super-rod": "wild",
    "fishing": "wild",
    "rock-smash": "wild",
    "headbutt": "wild",
    "sos": "wild",
    "horde": "wild",
    "hidden-grotto": "wild",
    "poke-radar": "wild",
    "mass-outbreak": "wild",
    "wanderer": "wild",
    "dexnav": "wild",
    "roaming": "wild",
    "max-raid": "raid",
    "dynamax-adventure": "raid",
    "tera-raid": "raid",
    STATIC: STATIC,
    GIFT: GIFT,
    SNAG: SNAG,
}
NO_METHOD_KIND = "wild"
ENCOUNTER_KINDS = ("wild", "raid", STATIC, GIFT, SNAG, PAL_PARK_ROW.kind)
NAMESPACE_PREFIXES = ("File:", "Image:", "Category:", ":Category:", "Template:")


class EncounterTableError(Exception):
    """A table row names a version the A-Box lacks."""


@dataclass(frozen=True)
class EncounterCounts:
    """The outcomes of a page, each counted by the name, title or text a reviewer reads."""

    rows: Counter[str] = field(default_factory=Counter)
    skipped_versions: Counter[str] = field(default_factory=Counter)
    unknown_versions: Counter[str] = field(default_factory=Counter)
    acquisitions: Counter[str] = field(default_factory=Counter)
    candidates: Counter[str] = field(default_factory=Counter)
    locations: Counter[str] = field(default_factory=Counter)
    not_a_location: Counter[str] = field(default_factory=Counter)
    unknown_location: Counter[str] = field(default_factory=Counter)
    unread_markers: Counter[str] = field(default_factory=Counter)
    unmatched_form_labels: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class EncounterReading:
    facts: tuple[EncounterFact, ...]
    counts: EncounterCounts


@dataclass(frozen=True)
class EncounterInputs:
    """What a page is read against: the location pins and the A-Box Version labels."""

    locations: LocationIndex
    versions: frozenset[str]


@dataclass(frozen=True)
class LinkRef:
    """1 link of a part: the page it names, the words it shows and the template behind it."""

    title: str
    display: str
    anchor: str = ""
    template: str = ""


@dataclass(frozen=True)
class PartLinks:
    """The links of 1 part in page order, split by what they mean."""

    first: LinkRef | None
    candidates: tuple[str, ...]
    methods: tuple[str, ...]
    prefix: str | None
    unread: tuple[str, ...]


def check_version_rows(versions: frozenset[str]) -> None:
    """Raise `EncounterTableError` when a table names a version the A-Box lacks."""
    named = {name for row in VERSION_ROWS.values() for name in row} | set(PAL_PARK_ROW.versions)
    missing = sorted(named - versions)
    if missing:
        raise EncounterTableError(f"Encounter tables name versions the A-Box lacks: {missing}")


def to_resolved_forms(extract: SpeciesPageExtract) -> tuple[ResolvedForm, ...]:
    """Rebuild the form nodes of a page with their infobox names, for `match_marker`."""
    infobox_names = {c.form_id: c.infobox_name for c in extract.candidates if c.form_id}
    return tuple(
        ResolvedForm(
            id=form.id,
            name=form.name,
            index=form.form_index,
            infobox_name=infobox_names.get(form.id, ""),
            region=form.region,
            is_default=form.is_default,
        )
        for form in extract.forms
    )


def find_main_table(sections: Sequence[Section]) -> Section | None:
    """Return the Game locations section, whose own text ends at "In side games"."""
    return next(
        (s for s in sections if s.path and s.path[-1].lower() == GAME_LOCATIONS_TITLE), None
    )


def read_rows(section: Section) -> list[tuple[str, Template]]:
    """Return each row template of the main table with its kind: entry, none or na."""
    rows: list[tuple[str, Template]] = []
    for template in section.parse().filter_templates(recursive=False):
        name = normalize_template_name(str(template.name))
        if name in ENTRY_TEMPLATES:
            rows.append((ROW_ENTRY, template))
        elif name in NONE_TEMPLATES:
            rows.append((ROW_NONE, template))
        elif name == NA_TEMPLATE:
            rows.append((ROW_NA, template))
    return rows


def read_param(template: Template, name: str) -> str:
    return str(template.get(name).value).strip() if template.has(name) else ""


def resolve_versions(
    row: Template, versions: frozenset[str]
) -> tuple[tuple[str, ...], str | None, str | None]:
    """Return the A-Box versions of an entry row, the name of a skipped pseudo-version and
    the first value no rule knows. A row with a skip or an unknown value gives no version."""
    values = [read_param(row, param) for param in VERSION_PARAMS if read_param(row, param)]
    japan = read_param(row, JAPAN_PARAM)
    if japan:
        return (), f"{values[0]} {japan}", None
    resolved: list[str] = []
    for value in values:
        if value in SKIPPED_VERSIONS:
            return (), value, None
        if value in VERSION_ROWS:
            resolved.extend(VERSION_ROWS[value])
        elif value in versions:
            resolved.append(value)
        else:
            return (), None, value
    return tuple(resolved), None, None


def split_parts(area: Wikicode) -> list[list[Node]]:
    """Cut an area at every `<br>` tag, keeping the nodes of each part in order."""
    parts: list[list[Node]] = [[]]
    for node in area.nodes:
        if isinstance(node, Tag) and str(node.tag).strip().lower() == BREAK_TAG:
            parts.append([])
        else:
            parts[-1].append(node)
    return [part for part in parts if "".join(str(node) for node in part).strip()]


def read_template_link(template: Template) -> LinkRef | None:
    """Return the link a template shows, or `None` for a template that links no page."""
    name = normalize_template_name(str(template.name))
    params = [str(param.value).strip() for param in template.params]
    if name in LOCATION_TEMPLATES:
        rule = LOCATION_TEMPLATES[name]
        if len(params) < len(rule.params):
            return None
        title = rule.pattern.format(None, *params)
        return LinkRef(title=title, display=title, template=name)
    if name == SECTION_LINK_TEMPLATE and len(params) >= 2:
        display = params[2] if len(params) > 2 else params[1]
        return LinkRef(title=params[0], display=display, anchor=params[1], template=name)
    if name == POKEMON_LINK_TEMPLATE and params:
        display = params[1] if len(params) > 1 else params[0]
        title = POKEMON_LINK_PATTERN.format(None, params[0])
        return LinkRef(title=title, display=display, template=name)
    if name == MOVE_TEMPLATE and params:
        return LinkRef(title=params[0], display=params[0], template=name)
    if name == SHADOW_TEMPLATE and params and params[0] == SHADOW_COLOR:
        return LinkRef(title=params[1] if len(params) > 1 else "", display="", template=name)
    return None


def read_wikilink(link: Wikilink) -> LinkRef | None:
    """Return a wikilink as a `LinkRef`, or `None` for a file or category link."""
    target = str(link.title).strip()
    if target.startswith(NAMESPACE_PREFIXES):
        return None
    title, _, anchor = target.partition(ANCHOR_MARK)
    display = str(link.text).strip() if link.text is not None else target
    return LinkRef(
        title=title.strip().removesuffix(DENS_SUFFIX), display=display, anchor=anchor.strip()
    )


def read_bold_prefix(tag: Tag) -> LinkRef | None:
    """Return the link of a `'''[[X]]:'''` prefix, or `None` for any other bold text."""
    if str(tag.tag).strip().lower() != BOLD_TAG or tag.contents is None:
        return None
    links = tag.contents.filter_wikilinks(recursive=False)
    if len(links) != 1 or not tag.contents.strip_code().strip().endswith(PREFIX_MARK):
        return None
    return read_wikilink(links[0])


def read_event_method(link: LinkRef) -> str | None:
    """Return the method an in-game event list link shows by its display, or `None`."""
    display = link.display.strip().lower()
    if STATIC_DISPLAY_PATTERN.fullmatch(display):
        return STATIC
    if display in GIFT_DISPLAYS:
        return GIFT
    return None


def read_link_method(link: LinkRef) -> str | None:
    """Return the method a marker link gives, or `None` for a link that is no marker."""
    if link.template == MOVE_TEMPLATE:
        return METHOD_MOVES.get(normalize_title(link.title))
    if link.template == SHADOW_TEMPLATE:
        return SNAG
    title = normalize_title(link.title)
    if title.startswith(EVENT_LIST_PREFIX):
        return read_event_method(link)
    return METHOD_LINKS.get(title)


def is_marker(link: LinkRef) -> bool:
    """Whether a link is a method, acquisition or event list link rather than a place."""
    title = normalize_title(link.title)
    return (
        link.template in {MOVE_TEMPLATE, SHADOW_TEMPLATE}
        or title in METHOD_LINKS
        or title.startswith(EVENT_LIST_PREFIX)
        or read_acquisition(link) is not None
    )


def read_acquisition(link: LinkRef) -> str | None:
    """Return the acquisition kind of a link with no place, or `None`."""
    if link.anchor == IN_EVENTS_ANCHOR:
        return EVENT_KIND
    return ACQUISITION_LINKS.get(normalize_title(link.title)) if link.title else None


def read_node_link(node: Node) -> LinkRef | None:
    if isinstance(node, Wikilink):
        return read_wikilink(node)
    if isinstance(node, Template):
        return read_template_link(node)
    return None


def count_group_depth(text: str, depth: int) -> int:
    return max(depth + text.count(OPEN_GROUP) - text.count(CLOSE_GROUP), 0)


def read_groups(part: Sequence[Node]) -> list[list[Node]]:
    """Return the nodes of each top-level `(...)` group of a part. Parentheses inside links
    and templates are not counted; the text of a group's opening node is part of it."""
    groups: list[list[Node]] = []
    depth = 0
    for node in part:
        if isinstance(node, Text):
            before = depth
            depth = count_group_depth(str(node), depth)
            if before == 0 and OPEN_GROUP in str(node):
                groups.append([node])
            elif before > 0:
                groups[-1].append(node)
        elif depth > 0:
            groups[-1].append(node)
    return groups


def read_unread_groups(part: Sequence[Node]) -> tuple[str, ...]:
    """Return the text of each group from which no marker reads a method (D7 of the plan)."""
    unread: list[str] = []
    for group in read_groups(part):
        links = [link for node in group if (link := read_node_link(node)) is not None]
        if any(read_link_method(link) is not None for link in links):
            continue
        text = to_plain_text("".join(str(node) for node in group))
        start, end = text.find(OPEN_GROUP), text.rfind(CLOSE_GROUP)
        cut = text[start : end + 1] if 0 <= start < end else text[max(start, 0) :]
        unread.append(" ".join(cut.split()))
    return tuple(unread)


def read_part_links(part: Sequence[Node]) -> PartLinks:
    """Read the links of a part: the 1st link, the place candidates, the methods in page
    order without repeats, the bold prefix and the unread groups."""
    links: list[LinkRef] = []
    prefix: str | None = None
    for node in part:
        if isinstance(node, Tag):
            bold = read_bold_prefix(node)
            if bold is not None and prefix is None and not links:
                prefix = bold.title
            continue
        link = read_node_link(node)
        if link is not None:
            links.append(link)
    methods = [method for link in links if (method := read_link_method(link)) is not None]
    unread_events = [
        f"{EVENT_LIST_LABEL}{link.display}"
        for link in links
        if normalize_title(link.title).startswith(EVENT_LIST_PREFIX)
        and read_event_method(link) is None
    ]
    return PartLinks(
        first=links[0] if links else None,
        candidates=tuple(
            link.title
            for link in links
            if link.title and link.template != POKEMON_LINK_TEMPLATE and not is_marker(link)
        ),
        methods=tuple(dict.fromkeys(methods)),
        prefix=prefix,
        unread=(*read_unread_groups(part), *unread_events),
    )


def read_form_id(
    part: Sequence[Node], forms: Sequence[ResolvedForm], counts: EncounterCounts
) -> str:
    """Return the form a `<small>` caption names, or the base form. An unmatched caption
    is counted by its label."""
    base_id = next(form.id for form in forms if form.is_default)
    captions = read_small_captions("".join(str(node) for node in part))
    if not captions:
        return base_id
    label = captions[-1].strip(CAPTION_PARENS)
    form_id = match_marker(label, forms)
    if form_id is None:
        counts.unmatched_form_labels[label] += 1
        return base_id
    return form_id


def classify_candidates(
    titles: Iterable[str], index: LocationIndex, counts: EncounterCounts
) -> list[str]:
    """Return the location titles among `titles`, in order without repeats, counting each
    verdict."""
    locations: list[str] = []
    for candidate in titles:
        result = classify_location(index, candidate)
        counts.candidates[normalize_title(candidate)] += 1
        if result.verdict is LocationVerdict.LOCATION:
            counts.locations[result.title] += 1
            locations.append(result.title)
        elif result.verdict is LocationVerdict.NOT_A_LOCATION:
            counts.not_a_location[result.title] += 1
        else:
            counts.unknown_location[result.title] += 1
    return list(dict.fromkeys(locations))


@dataclass(frozen=True)
class PartContext:
    versions: tuple[str, ...]
    forms: tuple[ResolvedForm, ...]
    index: LocationIndex


def read_part(
    part: Sequence[Node], context: PartContext, counts: EncounterCounts
) -> list[EncounterFact]:
    """Read 1 part into its facts: 1 per location, version and method of the part."""
    text = to_plain_text("".join(str(node) for node in part)).strip()
    if text.startswith(UNOBTAINABLE_PREFIX):
        counts.acquisitions[UNOBTAINABLE_KIND] += 1
        return []
    links = read_part_links(part)
    acquisition = None if links.first is None else read_acquisition(links.first)
    if acquisition is not None:
        counts.acquisitions[acquisition] += 1
        return []
    counts.unread_markers.update(links.unread)
    locations = classify_candidates(links.candidates, context.index, counts)
    if not locations and links.prefix is not None:
        locations = classify_candidates([links.prefix], context.index, counts)
    if not locations:
        return []
    species_id = read_form_id(part, context.forms, counts)
    methods: Sequence[str | None] = links.methods or (None,)
    return [
        EncounterFact(
            species_id=species_id,
            location_title=location,
            version=version,
            method=method,
            kind=NO_METHOD_KIND if method is None else METHOD_KINDS[method],
        )
        for location in locations
        for version in context.versions
        for method in methods
    ]


def read_pal_park_row(
    row: Template, extract: SpeciesPageExtract, index: LocationIndex, counts: EncounterCounts
) -> list[EncounterFact]:
    """Read a Pal Park row: its `link` names the place and the table gives the versions."""
    locations = classify_candidates([read_param(row, PAL_PARK_ROW.location_param)], index, counts)
    base_id = extract.forms[0].id
    return [
        EncounterFact(
            species_id=base_id,
            location_title=location,
            version=version,
            method=None,
            kind=PAL_PARK_ROW.kind,
        )
        for location in locations
        for version in PAL_PARK_ROW.versions
    ]


def read_entry_row(
    row: Template,
    extract: SpeciesPageExtract,
    inputs: EncounterInputs,
    counts: EncounterCounts,
) -> list[EncounterFact]:
    if read_param(row, VERSION_PARAMS[0]) == PAL_PARK_ROW.version:
        return read_pal_park_row(row, extract, inputs.locations, counts)
    versions, skipped, unknown = resolve_versions(row, inputs.versions)
    if skipped is not None:
        counts.skipped_versions[skipped] += 1
        return []
    if unknown is not None:
        counts.unknown_versions[unknown] += 1
        return []
    if not row.has(AREA_PARAM):
        return []
    context = PartContext(versions, to_resolved_forms(extract), inputs.locations)
    area = row.get(AREA_PARAM).value
    return [fact for part in split_parts(area) for fact in read_part(part, context, counts)]


def read_encounters(
    page: Page, extract: SpeciesPageExtract, inputs: EncounterInputs
) -> EncounterReading:
    """Read the main Game locations table of a page into facts, in row order, without
    repeats. A page with no such section gives no fact."""
    counts = EncounterCounts()
    section = find_main_table(split_sections(page.wikitext))
    if section is None:
        return EncounterReading(facts=(), counts=counts)
    facts: list[EncounterFact] = []
    for kind, row in read_rows(section):
        counts.rows[kind] += 1
        if kind == ROW_ENTRY:
            facts.extend(read_entry_row(row, extract, inputs, counts))
    return EncounterReading(facts=tuple(dict.fromkeys(facts)), counts=counts)


class EncounterReport(ReportModel):
    """The encounter outcomes of the requested pages. `candidate_occurrences` and
    `unknown_location_occurrences` count every link, not every title."""

    facts: int
    rows: Mapping[str, int]
    skipped_versions: Mapping[str, int]
    unknown_versions: Mapping[str, int]
    acquisitions: Mapping[str, int]
    candidate_occurrences: int
    unknown_location_occurrences: int
    locations: Mapping[str, int]
    not_a_location: Mapping[str, int]
    unknown_location: Mapping[str, int]
    unread_markers: Mapping[str, int]
    unmatched_form_labels: Mapping[str, int]


def sum_counts(readings: Iterable[EncounterReading]) -> EncounterCounts:
    totals = EncounterCounts()
    for reading in readings:
        for name, counter in vars(reading.counts).items():
            getattr(totals, name).update(counter)
    return totals


def sort_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def build_encounter_report(readings: Sequence[EncounterReading]) -> EncounterReport:
    totals = sum_counts(readings)
    return EncounterReport(
        facts=sum(len(reading.facts) for reading in readings),
        rows=sort_counter(totals.rows),
        skipped_versions=sort_counter(totals.skipped_versions),
        unknown_versions=sort_counter(totals.unknown_versions),
        acquisitions=sort_counter(totals.acquisitions),
        candidate_occurrences=totals.candidates.total(),
        unknown_location_occurrences=totals.unknown_location.total(),
        locations=sort_counter(totals.locations),
        not_a_location=sort_counter(totals.not_a_location),
        unknown_location=sort_counter(totals.unknown_location),
        unread_markers=sort_counter(totals.unread_markers),
        unmatched_form_labels=sort_counter(totals.unmatched_form_labels),
    )
