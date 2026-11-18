"""1 pinned item page to 1 extract, or to the reason it is out of scope.

Map:
- models: the extract, its notes and bag rows, and the skip record;
- scope: a page with an `ItemInfobox/head` and 1 main-series bag row is read; a TM, HM or TR
  page belongs to the move pages (`parse_item_page`);
- infobox: the bag rows give the `category` (`read_category`), `ItemInfobox/Fling` the fling
  power and `ItemInfobox/FlavorRow` the berry flavors (`read_flavor_potencies`);
- class: the lead's type phrase and the page categories give `Berry`, `HeldItem` or `Item`
  (`read_item_class`);
- prices: the Scarlet/Violet row of the Price section (`read_sv_prices`);
- sections: Price, Effect, Description and Acquisition sit 1 level below "In the core series
  games" or "In the games"; `rebase_core_sections` drops that title from the path so the
  shared readers see them like the sections of a move page.

The `id` is the slug of the page title without " (item)". Names stay in Bulbapedia terms;
`batches.py` maps them onto the ontology. This module knows the pkmn universe, as every
ingestion module of a source does.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import StrEnum

import mwparserfromhell
from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.wikitext import (
    AliasRules,
    Section,
    find_templates,
    mask_comments,
    normalize_template_name,
    read_aliases,
    read_categories,
    read_params,
    split_sections,
    to_plain_text,
)

INFOBOX_TEMPLATE = "ItemInfobox/head"
INFOBOX_PREFIX = "iteminfobox/"
BAG_ROW_TEMPLATES = ("ItemInfobox/BagRow", "ItemInfobox/BagRowBottom")
FLING_TEMPLATES = ("ItemInfobox/Fling", "ItemInfobox/FlingEffect")
FLAVOR_ROW_TEMPLATE = "ItemInfobox/FlavorRow"
PRICE_ROW_TEMPLATE = "ItemPrice"
GAME_CODE_TEMPLATE = "gameabbrev9"
GAME_CODE_PARAM = "1"
GENERATION_KEY = "gen"
GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVX]+$")
MACHINE_TITLE_PATTERN = re.compile(r"^(TM|HM|TR)\d+$")
ITEM_TITLE_SUFFIX = " (item)"
CORE_SECTION_TITLES = ("In the core series games", "In the games")
PRICE_PATH = ("Price",)
BAG_GENERATION_PARAM = "1"
BAG_POCKET_PARAM = "2"
BAG_GAME_PARAM = "game"
PRICE_GAME_PARAM = "1"
PRICE_BUY_PARAM = "2"
PRICE_SELL_PARAM = "3"
FLING_POWER_PARAM = "1"
FLAVOR_COLUMNS = ("Spicy", "Dry", "Sweet", "Bitter", "Sour")
FIRST_FLAVOR_PARAM = 2
SV_CODE = "SV"
NOT_AVAILABLE = "N/A"
HELD_ITEM_CATEGORY_SUFFIX = "held items"
BERRY_LINK = "berry"
HELD_ITEM_LINK = "held item"
POCKET_ALIASES: Mapping[str, str] = {"Medicines": "Medicine"}
PRICE_PATTERN = re.compile(r"^\$?\s*(\d[\d,]*)$")
LEADING_INTEGER_PATTERN = re.compile(r"^\s*(\d+)")
TYPE_PHRASE_START_PATTERN = re.compile(r"\bis (?:an?|the) ")
PHRASE_TOKEN_PATTERN = re.compile(r"\[\[.*?\]\]|(introduced in|exclusive to|[.,;\n])")
ALIAS_RULES = AliasRules(template_name="Langtable", infobox_keys=(), should_drop_notes=False)


class SkipReason(StrEnum):
    NO_ITEM_INFOBOX = "no_item_infobox"
    NO_MAIN_SERIES_BAG_ROW = "no_main_series_bag_row"
    MACHINE_PAGE = "machine_page"


class ItemClass(StrEnum):
    BERRY = "Berry"
    HELD_ITEM = "HeldItem"
    ITEM = "Item"


class ItemNoteKind(StrEnum):
    UNREAD_PRICE = "unread_price"


class ItemModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ItemNote(ItemModel):
    """Something the parser read, left out or changed that a reviewer should see."""

    kind: ItemNoteKind
    field: str
    detail: str


class BagRow(ItemModel):
    """1 main-series `ItemInfobox/BagRow` row: its generation numeral, pocket and game code."""

    generation: str
    pocket: str
    game: str | None = None


class SkippedPage(ItemModel):
    title: str
    page_id: int
    item_id: str
    reason: SkipReason


class ItemPageExtract(ItemModel):
    """The facts of 1 in-scope item page, in Bulbapedia names."""

    title: str
    page_id: int
    revision_id: int
    item_id: str
    name: str
    item_class: ItemClass
    generation_label: str
    category: str | None = None
    buy_price: int | None = None
    sell_price: int | None = None
    fling_power: int | None = None
    flavor_potencies: Mapping[str, int] = {}
    aliases: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    sections: tuple[Section, ...] = ()
    notes: tuple[ItemNote, ...] = ()


class ItemPageParseError(Exception):
    """A page broke a rule that has no other path. The message names page, field and value."""

    def __init__(self, page_title: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: cannot read {field} in {INFOBOX_TEMPLATE}: {value!r}")
        self.page_title = page_title
        self.field = field
        self.value = value


def rebase_core_sections(sections: Sequence[Section]) -> tuple[Section, ...]:
    """Drop the "In the core series games" or "In the games" title from the section paths.

    The section of that title holds no prose the readers need, so it leaves the list. Every
    other section keeps its path, and its level moves up by 1 under the dropped title.
    """
    rebased: list[Section] = []
    for section in sections:
        if section.path[:1] not in [(title,) for title in CORE_SECTION_TITLES]:
            rebased.append(section)
        elif len(section.path) > 1:
            rebased.append(replace(section, level=section.level - 1, path=section.path[1:]))
    return tuple(rebased)


def read_infobox(lead: Section) -> dict[str, str] | None:
    """Return the parameters of the first `ItemInfobox/head` of the lead, or `None`.

    The lead must have its comments masked.
    """
    infoboxes = find_templates(lead.parse(), [INFOBOX_TEMPLATE])
    if not infoboxes:
        return None
    return read_params(infoboxes[0])


def read_bag_rows(lead: Section) -> tuple[BagRow, ...]:
    """Return the bag rows of the lead whose generation is a Roman numeral, in page order.

    A Pokopia row has no generation and is left out.
    """
    rows: list[BagRow] = []
    for template in find_templates(lead.parse(), BAG_ROW_TEMPLATES):
        params = read_params(template)
        generation = params.get(BAG_GENERATION_PARAM, "")
        if ROMAN_NUMERAL_PATTERN.match(generation) is None:
            continue
        rows.append(
            BagRow(
                generation=generation,
                pocket=to_plain_text(params.get(BAG_POCKET_PARAM, "")),
                game=params.get(BAG_GAME_PARAM),
            )
        )
    return tuple(rows)


def read_category(rows: Sequence[BagRow]) -> str | None:
    """Return the pocket of the last row with each word capitalized, or `None` with no pocket."""
    if not rows or not rows[-1].pocket:
        return None
    words = (word[:1].upper() + word[1:] for word in rows[-1].pocket.split())
    pocket = " ".join(words)
    return POCKET_ALIASES.get(pocket, pocket)


def read_type_phrase(lead: Section) -> str | None:
    """Return the words after the lead's first "is a" up to the end of the noun phrase.

    The phrase ends at "introduced in", "exclusive to", a period, a comma, a semicolon or a
    line end outside a link, so "type of [[held item]]" and "type of consumable [[held item]]"
    come back whole. The infobox templates are removed first.
    """
    code = lead.parse()
    for template in code.filter_templates(recursive=False):
        if normalize_template_name(str(template.name)).startswith(INFOBOX_PREFIX):
            code.remove(template)
    text = str(code)
    start = TYPE_PHRASE_START_PATTERN.search(text)
    if start is None:
        return None
    rest = text[start.end() :]
    end = next(
        (token.start() for token in PHRASE_TOKEN_PATTERN.finditer(rest) if token.group(1)),
        len(rest),
    )
    return rest[:end].strip() or None


def read_item_class(type_phrase: str | None, categories: Sequence[str]) -> ItemClass:
    """Return `Berry` when the 1st link of the phrase is Berry, `HeldItem` when the phrase
    links "held item" or a category ends in "held items", else `Item`.

    Hyper berries link a food type before Berry and stay `Item`.
    """
    links = [
        str(link.title).strip().lower()
        for link in mwparserfromhell.parse(type_phrase or "").filter_wikilinks()
    ]
    if links and links[0] == BERRY_LINK:
        return ItemClass.BERRY
    is_held_category = any(
        category.lower().endswith(HELD_ITEM_CATEGORY_SUFFIX) for category in categories
    )
    if HELD_ITEM_LINK in links or is_held_category:
        return ItemClass.HELD_ITEM
    return ItemClass.ITEM


def read_price_cell(raw: str) -> int | None:
    """Return the amount of a price cell, or `None` for `N/A` and any text that is not 1 price."""
    text = to_plain_text(raw)
    if not text or text == NOT_AVAILABLE:
        return None
    matched = PRICE_PATTERN.match(text)
    if matched is None:
        return None
    return int(matched.group(1).replace(",", ""))


def has_sv_code(cell: str) -> bool:
    games = find_templates(mwparserfromhell.parse(cell), [GAME_CODE_TEMPLATE])
    return any(
        SV_CODE in str(game.get(GAME_CODE_PARAM).value)
        for game in games
        if game.has(GAME_CODE_PARAM)
    )


def read_sv_prices(
    sections: Sequence[Section],
) -> tuple[int | None, int | None, tuple[ItemNote, ...]]:
    """Return the buy price, the sell price and the notes of the Scarlet/Violet price row.

    The row is the 1st `ItemPrice` of the Price section whose game cell holds a
    `gameabbrev9` code containing `SV`. A cell that is not exactly 1 price leaves its value
    absent and adds an `unread_price` note; `N/A` leaves it absent with no note.
    """
    for section in sections:
        if section.path != PRICE_PATH:
            continue
        for template in find_templates(section.parse(), [PRICE_ROW_TEMPLATE]):
            params = read_params(template)
            if not has_sv_code(params.get(PRICE_GAME_PARAM, "")):
                continue
            notes: list[ItemNote] = []
            values: list[int | None] = []
            for field, param in (("buyPrice", PRICE_BUY_PARAM), ("sellPrice", PRICE_SELL_PARAM)):
                raw = params.get(param, "")
                value = read_price_cell(raw)
                if value is None and to_plain_text(raw) not in ("", NOT_AVAILABLE):
                    notes.append(ItemNote(kind=ItemNoteKind.UNREAD_PRICE, field=field, detail=raw))
                values.append(value)
            return values[0], values[1], tuple(notes)
    return None, None, ()


def read_fling_power(lead: Section) -> int | None:
    """Return the leading integer of the 1st fling row's 1st parameter, or `None`."""
    for template in find_templates(lead.parse(), FLING_TEMPLATES):
        matched = LEADING_INTEGER_PATTERN.match(read_params(template).get(FLING_POWER_PARAM, ""))
        return int(matched.group(1)) if matched else None
    return None


def read_flavor_potencies(lead: Section) -> Mapping[str, int]:
    """Return the potency of each flavor of the last `ItemInfobox/FlavorRow`, zeros left out.

    `ItemInfobox/ZAFlavorRow` is another template and is not read.
    """
    rows = find_templates(lead.parse(), [FLAVOR_ROW_TEMPLATE])
    if not rows:
        return {}
    params = read_params(rows[-1])
    potencies: dict[str, int] = {}
    for offset, flavor in enumerate(FLAVOR_COLUMNS):
        matched = LEADING_INTEGER_PATTERN.match(params.get(str(FIRST_FLAVOR_PARAM + offset), ""))
        if matched and int(matched.group(1)) > 0:
            potencies[flavor] = int(matched.group(1))
    return potencies


def read_generation_label(page_title: str, infobox: Mapping[str, str]) -> str:
    numeral = infobox.get(GENERATION_KEY, "").strip()
    if ROMAN_NUMERAL_PATTERN.match(numeral) is None:
        raise ItemPageParseError(page_title, GENERATION_KEY, numeral)
    return GENERATION_LABEL.format(numeral=numeral)


def parse_item_page(page: Page) -> ItemPageExtract | SkippedPage:
    """Parse 1 pinned item page. Raises `ItemPageParseError` for a `gen` no rule reads."""
    name = page.title.removesuffix(ITEM_TITLE_SUFFIX)
    item_id = build_node_id(name)

    def skip(reason: SkipReason) -> SkippedPage:
        return SkippedPage(title=page.title, page_id=page.page_id, item_id=item_id, reason=reason)

    if MACHINE_TITLE_PATTERN.match(page.title):
        return skip(SkipReason.MACHINE_PAGE)
    masked = mask_comments(page.wikitext)
    sections = split_sections(masked)
    lead = sections[0]
    infobox = read_infobox(lead)
    if infobox is None:
        return skip(SkipReason.NO_ITEM_INFOBOX)
    bag_rows = read_bag_rows(lead)
    if not bag_rows:
        return skip(SkipReason.NO_MAIN_SERIES_BAG_ROW)
    categories = read_categories(masked)
    rebased = rebase_core_sections(sections)
    buy_price, sell_price, notes = read_sv_prices(rebased)
    return ItemPageExtract(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        item_id=item_id,
        name=name,
        item_class=read_item_class(read_type_phrase(lead), sorted(categories)),
        generation_label=read_generation_label(page.title, infobox),
        category=read_category(bag_rows),
        buy_price=buy_price,
        sell_price=sell_price,
        fling_power=read_fling_power(lead),
        flavor_potencies=read_flavor_potencies(lead),
        aliases=read_aliases(split_sections(page.wikitext), infobox, name, ALIAS_RULES),
        categories=tuple(sorted(categories)),
        sections=rebased,
        notes=notes,
    )
