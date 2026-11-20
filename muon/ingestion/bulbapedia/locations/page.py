"""1 pinned location page to 1 extract, or to the reason it is out of scope.

Map:
- models: the extract, its notes, neighbours and field requirements, and the skip record;
- scope: a page with a location infobox in the lead and 1 core game category is read; a
  redirect, a page with another infobox and a side-game place are skipped
  (`parse_location_page`);
- class: the infobox kind, the L4 title table, the page categories and the `type=` value give
  the leaf class (`read_location_class`);
- containment: the links of `location=`, else the `region` values, give the parent titles
  (`read_parent_titles`);
- adjacency: `north`, `south`, `east`, `west` and their `2` and `alt` variants give the
  neighbours (`read_neighbours`);
- versions and generation: the core game categories give the version groups
  (`read_version_groups`) and `generation=` the first generation (`read_generation`);
- field moves: a route infobox field with the value `yes` gives a required move or item
  (`read_field_requirements`);
- places: the links of the Places of interest sections name the building pages that the next
  export must pin (`read_place_titles`).

The `id` is the slug of the page title, built by `batches.py`. Names stay in Bulbapedia terms;
`batches.py` maps them onto the ontology. This module knows the pkmn universe, as every
ingestion module of a source does.
"""

import re
from collections.abc import Collection, Mapping, Sequence
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Template, Wikilink
from pydantic import BaseModel, ConfigDict

from muon.ingestion.bulbapedia.locations.areas import AreaRef, read_areas
from muon.ingestion.bulbapedia.locations.identity import (
    LOCATION_INFOBOXES,
    LocationIndex,
    list_redirect_titles,
    normalize_title,
    read_redirect_target,
)
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    mask_comments,
    normalize_template_name,
    read_categories,
    read_params,
    split_sections,
)

GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERALS: Mapping[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
}
GENERATION_TOKEN_PATTERN = re.compile(r"\b(?:([IVX]+)|(\d+))\b")
DIRECTION_FIELD_PATTERN = re.compile(r"^(north|south|east|west)\d*(?:alt)?$")
ROUTE_TITLE_PATTERN = re.compile(r"^Route (\d+)$")
ROUTE_TEMPLATES = frozenset({"rt", "rtn"})
KANTO_TEMPLATE = "ka"
MAIN_TEMPLATE = "main"
PLACES_SECTION_TITLE = "places of interest"
BREADCRUMB_SEPARATOR = " → "
KANTO_TITLE_SUFFIX = " (Kanto)"
ROUTE_TITLE = "{region} Route {number}"
IGNORED_LINK_PREFIXES = ("file:", "image:", "category:")
ANCHOR_MARK = "#"
PLAIN_TITLE_PADDING = "' *†‡\n"
LOCATION_FIELD = "location"
REGION_FIELDS = ("region", "region2", "region3")
GENERATION_FIELD = "generation"
TYPE_FIELD = "type"
ROUTE_REGION_FIELD = "region{direction}"
YES_VALUE = "yes"
FIELD_MOVE_ROLE = "fieldMove"
FIELD_ITEM_ROLE = "fieldItem"

# Move and item names of the route infobox fields; the names must match the pkmn Move and
# Item nodes, which the batches resolve by id.
FIELD_MOVES: Mapping[str, str] = {
    "surf": "Surf",
    "cut": "Cut",
    "strength": "Strength",
    "rocksmash": "Rock Smash",
    "waterfall": "Waterfall",
    "dive": "Dive",
    "rockclimb": "Rock Climb",
    "whirlpool": "Whirlpool",
    "flash": "Flash",
    "defog": "Defog",
}
FIELD_ITEMS: Mapping[str, str] = {
    "bike": "Bicycle",
    "mach": "Mach Bike",
    "acro": "Acro Bike",
}

# Core game category to the A-Box VersionGroup labels it stands for; "Red, Blue and Yellow"
# is 2 groups. `check_category_table` checks the labels against the A-Box before any page is
# read.
CORE_GAME_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "Red, Blue and Yellow locations": ("Red-Blue", "Yellow"),
    "Gold, Silver and Crystal locations": ("Gold-Silver", "Crystal"),
    "Ruby and Sapphire locations": ("Ruby-Sapphire",),
    "FireRed and LeafGreen locations": ("FireRed-LeafGreen",),
    "Emerald locations": ("Emerald",),
    "Diamond and Pearl locations": ("Diamond-Pearl",),
    "Platinum locations": ("Platinum",),
    "HeartGold and SoulSilver locations": ("HeartGold-SoulSilver",),
    "Black and White locations": ("Black-White",),
    "Black 2 and White 2 locations": ("Black 2-White 2",),
    "X and Y locations": ("X-Y",),
    "Omega Ruby and Alpha Sapphire locations": ("Omega Ruby-Alpha Sapphire",),
    "Sun and Moon locations": ("Sun-Moon",),
    "Ultra Sun and Ultra Moon locations": ("Ultra Sun-Ultra Moon",),
    "Let's Go, Pikachu! and Let's Go, Eevee! locations": ("Let's Go Pikachu-Let's Go Eevee",),
    "Sword and Shield locations": ("Sword-Shield",),
    "Brilliant Diamond and Shining Pearl locations": ("Brilliant Diamond and Shining Pearl",),
    "Legends: Arceus locations": ("Legends: Arceus",),
    "Scarlet and Violet locations": ("Scarlet-Violet",),
    "Legends: Z-A locations": ("Legends: Z-A",),
    "Colosseum locations": ("Colosseum",),
    "XD locations": ("XD",),
}

# Places of the core series games that carry no core game category. A page of these titles
# is in scope and takes the version groups listed, which the categories would have given.
SCOPE_OVERRIDES: Mapping[str, tuple[str, ...]] = {
    "One Island": ("FireRed-LeafGreen",),
    "Two Island": ("FireRed-LeafGreen",),
    "Three Island": ("FireRed-LeafGreen",),
    "Four Island": ("FireRed-LeafGreen",),
    "Five Island": ("FireRed-LeafGreen",),
    "Six Island": ("FireRed-LeafGreen",),
    "Seven Island": ("FireRed-LeafGreen",),
    "Hammerlocke Stadium": ("Sword-Shield",),
    "Wyndon Stadium": ("Sword-Shield",),
    "Naranja Academy": ("Scarlet-Violet",),
    "Uva Academy": ("Scarlet-Violet",),
}

# The A-Box Regions. A page of these titles gives chunks only (L5). Must stay in sync with
# `individuals/pkmn.ttl`; `batches.py` checks the titles against the A-Box at run start.
ABOX_REGION_TITLES: frozenset[str] = frozenset(
    {
        "Kanto",
        "Johto",
        "Hoenn",
        "Orre",
        "Sinnoh",
        "Unova",
        "Kalos",
        "Alola",
        "Galar",
        "Hisui",
        "Paldea",
    }
)

# Sub-regions the A-Box has no Region for (L4). Each is a GeoFeature whatever its infobox.
GEO_FEATURE_TITLES: frozenset[str] = frozenset(
    {
        "Kitakami",
        "Sevii Islands",
        "Crown Tundra",
        "Isle of Armor",
        "Wild Area",
        "Terarium",
        "Obsidian Fieldlands",
        "Crimson Mirelands",
        "Cobalt Coastlands",
        "Coronet Highlands",
        "Alabaster Icelands",
    }
)


class SkipReason(StrEnum):
    REDIRECT = "redirect"
    NO_LOCATION_INFOBOX = "no_location_infobox"
    NOT_CORE_SERIES = "not_core_series"


class InfoboxKind(StrEnum):
    INFOBOX_LOCATION = "infobox location"
    ROUTE = "route infobox"
    TOWN = "town infobox"
    REGION = "regioninfobox"
    GYM = "gyminfobox"


class LocationClass(StrEnum):
    SETTLEMENT = "settlement"
    ROUTE = "route"
    GEO_FEATURE = "geo_feature"
    STRUCTURE = "structure"
    LOCATION = "location"


class LocationNoteKind(StrEnum):
    NO_LEAF_CLASS = "no_leaf_class"
    CLASS_CONFLICT = "class_conflict"
    LIMITED_FIELD_MOVE = "limited_field_move"


# First class in this order wins when the categories of a page give several (L6).
CLASS_BY_CATEGORY: Mapping[LocationClass, frozenset[str]] = {
    LocationClass.SETTLEMENT: frozenset({"Cities", "Towns"}),
    LocationClass.ROUTE: frozenset({"Routes", "Paths"}),
    LocationClass.GEO_FEATURE: frozenset(
        {
            "Caves",
            "Mountains",
            "Forests",
            "Islands",
            "Lakes",
            "Beaches",
            "Volcanoes",
            "Deserts",
            "Plains",
            "Underwater locations",
            "Parks",
            "Farms",
            "Meadows",
            "Wild zones",
            "Cobalt Coastlands",
            "Obsidian Fieldlands",
            "Crimson Mirelands",
            "Alabaster Icelands",
            "Coronet Highlands",
            "Wild Area",
            "Crown Tundra",
            "Isle of Armor",
        }
    ),
    LocationClass.STRUCTURE: frozenset(
        {
            "Towers",
            "Ruins",
            "Monuments",
            "Bridges",
            "Abandoned buildings",
            "Research centers",
            "Villainous team bases",
            "Apparel shops",
            "Restaurants",
            "Shops",
        }
    ),
}
CLASS_BY_TYPE: Mapping[str, LocationClass] = {
    "building": LocationClass.STRUCTURE,
    "cave": LocationClass.GEO_FEATURE,
    "forest": LocationClass.GEO_FEATURE,
    "mountain": LocationClass.GEO_FEATURE,
    "lake": LocationClass.GEO_FEATURE,
    "volcano": LocationClass.GEO_FEATURE,
    "ocean": LocationClass.GEO_FEATURE,
    "sand": LocationClass.GEO_FEATURE,
    "snow": LocationClass.GEO_FEATURE,
}
CLASS_BY_KIND: Mapping[InfoboxKind, LocationClass] = {
    InfoboxKind.TOWN: LocationClass.SETTLEMENT,
    InfoboxKind.ROUTE: LocationClass.ROUTE,
    InfoboxKind.GYM: LocationClass.STRUCTURE,
}


class LocationModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class LocationNote(LocationModel):
    """Something the parser read, left out or changed that a reviewer should see."""

    kind: LocationNoteKind
    field: str
    detail: str


class Neighbour(LocationModel):
    """1 page named by a direction field of the infobox."""

    title: str
    direction: str


class FieldRequirement(LocationModel):
    """A move or an item needed to explore the location fully. `role` is `fieldMove` or
    `fieldItem` and `target` the name of the Move or Item."""

    role: str
    target: str


class SkippedPage(LocationModel):
    title: str
    page_id: int
    reason: SkipReason


class InfoboxFields(LocationModel):
    """The kind and the parameters of the location infobox, parameter names in lowercase."""

    kind: InfoboxKind
    params: Mapping[str, str]


class LocationPageExtract(LocationModel):
    """The facts of 1 in-scope location page, in Bulbapedia names."""

    page_id: int
    title: str
    revision_id: int
    kind: InfoboxKind
    leaf_class: LocationClass | None
    is_abox_region: bool
    parent_titles: tuple[str, ...] = ()
    neighbours: tuple[Neighbour, ...] = ()
    version_groups: tuple[str, ...] = ()
    generation: str | None = None
    field_requirements: tuple[FieldRequirement, ...] = ()
    areas: tuple[AreaRef, ...] = ()
    place_titles: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    notes: tuple[LocationNote, ...] = ()


class LocationTableError(Exception):
    """A parser table names a version group the A-Box lacks."""


def check_category_table(version_groups: Collection[str]) -> None:
    """Raise `LocationTableError` when `CORE_GAME_CATEGORIES` names a label outside
    `version_groups`, the VersionGroup labels of the A-Box."""
    named = {label for labels in CORE_GAME_CATEGORIES.values() for label in labels}
    named |= {label for labels in SCOPE_OVERRIDES.values() for label in labels}
    missing = sorted(named - set(version_groups))
    if missing:
        raise LocationTableError(
            f"Location category table names version groups the A-Box lacks: {missing}"
        )


def read_infobox(lead: Section) -> InfoboxFields | None:
    """Return the first location infobox among the top-level templates of the lead, or `None`.

    The lead must have its comments masked.
    """
    for template in lead.parse().filter_templates(recursive=False):
        name = normalize_template_name(template.name.strip_code())
        if name in LOCATION_INFOBOXES:
            params = {key.lower(): value for key, value in read_params(template).items()}
            return InfoboxFields(kind=InfoboxKind(name), params=params)
    return None


def read_link_title(link: Wikilink) -> str | None:
    """Return the normalized page title of a link without its anchor, or `None` for a file or
    category link."""
    target = str(link.title).strip()
    if target.lower().startswith(IGNORED_LINK_PREFIXES):
        return None
    title = normalize_title(target.split(ANCHOR_MARK, 1)[0])
    return title or None


def read_template_title(template: Template) -> str | None:
    """Return the page a route (`rt`, `rtn`) or Kanto (`ka`) template names, or `None`."""
    name = normalize_template_name(template.name.strip_code())
    arguments = [str(param.value).strip() for param in template.params if not param.showkey]
    if name in ROUTE_TEMPLATES and len(arguments) >= 2:
        return ROUTE_TITLE.format(region=arguments[1], number=arguments[0])
    if name == KANTO_TEMPLATE and arguments:
        return normalize_title(arguments[0]) + KANTO_TITLE_SUFFIX
    return None


def read_link_titles(value: str) -> tuple[str, ...]:
    """Return the page titles named by the links, `rt` or `rtn` templates and `ka` templates
    of `value`, in page order and without repeats."""
    titles: list[str] = []
    for node in mwparserfromhell.parse(value).filter():
        if isinstance(node, Wikilink):
            title = read_link_title(node)
        elif isinstance(node, Template):
            title = read_template_title(node)
        else:
            continue
        if title is not None and title not in titles:
            titles.append(title)
    return tuple(titles)


def read_main_title(value: str) -> str:
    """Return the page a `{{main}}` argument names: the text before an anchor or a
    " → " breadcrumb, normalized, or an empty string."""
    target = value.split(ANCHOR_MARK, 1)[0].split(BREADCRUMB_SEPARATOR, 1)[0]
    return normalize_title(target) if target.strip() else ""


def read_place_titles(sections: Sequence[Section]) -> tuple[str, ...]:
    """Return the titles linked from the Places of interest sections, in page order.

    A link is a `[[X]]` link or a `{{main|X}}` template of the section or of a subsection.
    The titles are export candidates: the scope rule of a later run skips those that are not
    places.
    """
    titles: dict[str, None] = {}
    for section in sections:
        if PLACES_SECTION_TITLE not in (title.lower() for title in section.path):
            continue
        for node in section.parse().filter():
            found: tuple[str, ...]
            if isinstance(node, Wikilink):
                title = read_link_title(node)
                found = (title,) if title else ()
            elif isinstance(node, Template) and (
                normalize_template_name(node.name.strip_code()) == MAIN_TEMPLATE
            ):
                found = tuple(
                    title
                    for param in node.params
                    if not param.showkey and (title := read_main_title(str(param.value)))
                )
            else:
                continue
            titles.update(dict.fromkeys(found))
    return tuple(titles)


def read_plain_title(value: str, region: str | None = None) -> str | None:
    """Return the page title written as plain text in `value`, or `None` when it is empty.

    `Route 111` with a region gives `Hoenn Route 111`, the title of the route page.
    """
    text = mwparserfromhell.parse(value).strip_code().strip(PLAIN_TITLE_PADDING).strip()
    if not text:
        return None
    route = ROUTE_TITLE_PATTERN.match(text)
    if route is not None and region:
        return ROUTE_TITLE.format(region=region, number=route.group(1))
    return normalize_title(text)


def read_parent_titles(infobox: InfoboxFields) -> tuple[str, ...]:
    """Return the titles `location=` links to, else the `region` values.

    A region infobox names the place of its league in `location=`, so it reads the region
    values only.
    """
    params = infobox.params
    if infobox.kind is not InfoboxKind.REGION:
        linked = read_link_titles(params.get(LOCATION_FIELD, ""))
        if linked:
            return linked
    regions = (read_plain_title(params.get(field, "")) for field in REGION_FIELDS)
    return tuple(dict.fromkeys(title for title in regions if title is not None))


def read_neighbours(infobox: InfoboxFields) -> tuple[Neighbour, ...]:
    """Return 1 neighbour per target of each direction field, in field order.

    A field value is a link, a route template or plain text; `regionnorth` and its siblings
    only name the region of a plain `Route 8`, falling back to the `region` of the page.
    """
    params = infobox.params
    neighbours: dict[Neighbour, None] = {}
    for field, value in params.items():
        matched = DIRECTION_FIELD_PATTERN.match(field)
        if matched is None or not value:
            continue
        direction = matched.group(1)
        region_value = params.get(ROUTE_REGION_FIELD.format(direction=direction))
        region_value = region_value or params.get(REGION_FIELDS[0], "")
        titles = read_link_titles(value)
        if not titles:
            plain = read_plain_title(value, read_plain_title(region_value))
            titles = (plain,) if plain is not None else ()
        for title in titles:
            neighbours[Neighbour(title=title, direction=direction)] = None
    return tuple(neighbours)


def read_version_groups(categories: Collection[str]) -> tuple[str, ...]:
    """Return the version group labels of the core game categories, in table order."""
    return tuple(
        label
        for category, labels in CORE_GAME_CATEGORIES.items()
        if category in categories
        for label in labels
    )


def read_generation(infobox: InfoboxFields) -> str | None:
    """Return the label of the lowest generation `generation=` names, or `None`.

    The value is a number (`3`) or a list of `{{gen|IV}}` templates.
    """
    value = infobox.params.get(GENERATION_FIELD, "")
    numbers: list[int] = []
    for roman, digits in GENERATION_TOKEN_PATTERN.findall(value):
        number = ROMAN_NUMERALS.get(roman) if roman else int(digits)
        if number is not None:
            numbers.append(number)
    if not numbers:
        return None
    numeral = next(name for name, number in ROMAN_NUMERALS.items() if number == min(numbers))
    return GENERATION_LABEL.format(numeral=numeral)


def read_field_requirements(
    infobox: InfoboxFields,
) -> tuple[tuple[FieldRequirement, ...], tuple[LocationNote, ...]]:
    """Return the required moves and items of a route infobox, and a note for each field with
    a value other than a plain `yes` (`cut={{sup/4|HGSS}}`)."""
    if infobox.kind is not InfoboxKind.ROUTE:
        return (), ()
    requirements: list[FieldRequirement] = []
    notes: list[LocationNote] = []
    for field, value in infobox.params.items():
        if field in FIELD_MOVES:
            role, target = FIELD_MOVE_ROLE, FIELD_MOVES[field]
        elif field in FIELD_ITEMS:
            role, target = FIELD_ITEM_ROLE, FIELD_ITEMS[field]
        else:
            continue
        if not value:
            continue
        if value.lower() == YES_VALUE:
            requirements.append(FieldRequirement(role=role, target=target))
        else:
            notes.append(
                LocationNote(kind=LocationNoteKind.LIMITED_FIELD_MOVE, field=field, detail=value)
            )
    return tuple(requirements), tuple(notes)


def read_location_class(
    kind: InfoboxKind, categories: Collection[str], type_value: str, title: str
) -> tuple[LocationClass | None, tuple[LocationNote, ...]]:
    """Return the leaf class of a page and the notes about it, first rule that applies.

    A town, route or gym infobox fixes the class. Otherwise a title of `GEO_FEATURE_TITLES`
    gives a GeoFeature, a title of `ABOX_REGION_TITLES` gives `None` (the A-Box owns the
    Region), then the categories give the class, then `type=`. A page that fits no rule is a
    plain Location with a `no_leaf_class` note; a page whose categories give 2 classes takes
    the first in table order with a `class_conflict` note.
    """
    if kind in CLASS_BY_KIND:
        return CLASS_BY_KIND[kind], ()
    if kind is InfoboxKind.REGION and title in ABOX_REGION_TITLES:
        return None, ()
    if title in GEO_FEATURE_TITLES:
        return LocationClass.GEO_FEATURE, ()
    by_category = [
        location_class
        for location_class, names in CLASS_BY_CATEGORY.items()
        if names & set(categories)
    ]
    if len(by_category) > 1:
        detail = ", ".join(location_class.value for location_class in by_category)
        note = LocationNote(kind=LocationNoteKind.CLASS_CONFLICT, field="categories", detail=detail)
        return by_category[0], (note,)
    if by_category:
        return by_category[0], ()
    by_type = CLASS_BY_TYPE.get(type_value.lower())
    if by_type is not None:
        return by_type, ()
    note = LocationNote(kind=LocationNoteKind.NO_LEAF_CLASS, field=TYPE_FIELD, detail=type_value)
    return LocationClass.LOCATION, (note,)


def parse_location_page(page: Page, index: LocationIndex) -> LocationPageExtract | SkippedPage:
    """Parse 1 pinned location page into an extract or the reason it is out of scope."""

    def skip(reason: SkipReason) -> SkippedPage:
        return SkippedPage(title=page.title, page_id=page.page_id, reason=reason)

    if read_redirect_target(page.wikitext) is not None:
        return skip(SkipReason.REDIRECT)
    masked = mask_comments(page.wikitext)
    sections = split_sections(masked)
    infobox = read_infobox(sections[0])
    if infobox is None:
        return skip(SkipReason.NO_LOCATION_INFOBOX)
    categories = read_categories(masked)
    title = normalize_title(page.title)
    version_groups = read_version_groups(categories) or SCOPE_OVERRIDES.get(title, ())
    if not version_groups:
        return skip(SkipReason.NOT_CORE_SERIES)
    leaf_class, class_notes = read_location_class(
        infobox.kind, categories, infobox.params.get(TYPE_FIELD, ""), title
    )
    requirements, requirement_notes = read_field_requirements(infobox)
    own_titles = {title, *list_redirect_titles(index, title)}
    return LocationPageExtract(
        page_id=page.page_id,
        title=page.title,
        revision_id=page.revision_id,
        kind=infobox.kind,
        leaf_class=leaf_class,
        is_abox_region=infobox.kind is InfoboxKind.REGION and title in ABOX_REGION_TITLES,
        parent_titles=tuple(t for t in read_parent_titles(infobox) if t not in own_titles),
        neighbours=tuple(n for n in read_neighbours(infobox) if n.title not in own_titles),
        version_groups=version_groups,
        generation=read_generation(infobox),
        field_requirements=requirements,
        areas=read_areas(sections, page.title, index),
        place_titles=read_place_titles(sections),
        aliases=list_redirect_titles(index, title),
        notes=(*class_notes, *requirement_notes),
    )
