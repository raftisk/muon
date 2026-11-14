"""1 pinned ability page to 1 extract, or to the reason it is out of scope.

Map:
- models: the extract, its notes, its `desc` rows and holder rows, and the skip record;
- scope: a page with an `AbilityInfobox/header` is read, every other page is skipped;
- values: the infobox `name` and `gen`, the other-language names and the page categories,
  with the "Abilities " ones counted apart for the effect review;
- rows: the `AbilityInfobox/desc` rows of the lead, raw, for `chunks.py`, and the species
  of every `Ability/entry` row of the "Pokémon with X" sections, for `crosscheck.py`.

The `id` is the slug of the page title without " (Ability)", the id the species pages
already give their Ability stubs. Names stay in Bulbapedia terms; `batches.py` maps them
onto the ontology. This module knows the pkmn universe, as every ingestion module of a
source does.
"""

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.chunking import ChunkFact
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.wikitext import (
    AliasRules,
    Section,
    find_templates,
    mask_comments,
    read_aliases,
    read_categories,
    read_params,
    split_sections,
)

INFOBOX_TEMPLATE = "abilityinfobox/header"
INFOBOX_LABEL = "AbilityInfobox/header"
DESC_TEMPLATE = "abilityinfobox/desc"
ENTRY_TEMPLATE = "ability/entry"
TITLE_SUFFIX = " (Ability)"
GENERATION_KEY = "gen"
GENERATION_LABEL = "Generation {numeral}"
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVX]+$")
DESC_GENERATION_PARAM = "1"
DESC_TEXT_PARAM = "2"
ENTRY_SPECIES_PARAM = "2"
HOLDER_SECTION_PREFIX = "pokémon with"
ABILITY_CATEGORY_PREFIX = "Abilities "
ABILITY_ALIAS_RULES = AliasRules(
    template_name="langtable", infobox_keys=("jpname",), should_drop_notes=True
)


class SkipReason(StrEnum):
    NO_INFOBOX = "no_infobox"


class AbilityNoteKind(StrEnum):
    NO_POKEAPI_MATCH = "no_pokeapi_match"
    UNKNOWN_SUP_CODE = "unknown_sup_code"
    UNRESOLVED_MENTION = "unresolved_mention"
    LONG_CHUNK = "long_chunk"


class AbilityModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class AbilityNote(AbilityModel):
    """Something the parser read, left out or changed that a reviewer should see."""

    kind: AbilityNoteKind
    field: str
    detail: str


class DescRow(AbilityModel):
    """1 `AbilityInfobox/desc` row: the generation it names and its raw text."""

    generation_label: str
    wikitext: str


class HolderEntry(AbilityModel):
    """1 `Ability/entry` row of "Pokémon with X": the species name as the row writes it."""

    species_name: str


class SkippedPage(AbilityModel):
    title: str
    page_id: int
    ability_id: str | None
    reason: SkipReason


class AbilityPageExtract(AbilityModel):
    """The facts of 1 in-scope ability page, in Bulbapedia names."""

    title: str
    page_id: int
    revision_id: int
    id: str
    name: str
    generation_label: str
    aliases: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    ability_categories: tuple[str, ...] = ()
    desc_rows: tuple[DescRow, ...] = ()
    holders: tuple[HolderEntry, ...] = ()
    chunks: tuple[ChunkFact, ...] = ()
    dropped_templates: Mapping[str, int] = {}
    removed_tables: int = 0
    math_formulas: int = 0
    notes: tuple[AbilityNote, ...] = ()


class AbilityPageParseError(Exception):
    """A page broke a rule that has no other path. The message names page, field and value."""

    def __init__(self, page_title: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: cannot read {field} in {INFOBOX_LABEL}: {value!r}")
        self.page_title = page_title
        self.field = field
        self.value = value


def read_ability_infobox(lead: Section) -> dict[str, str] | None:
    """Return the parameters of the first `AbilityInfobox/header` of the lead, or `None`.

    The lead must have its comments masked.
    """
    infoboxes = find_templates(lead.parse(), [INFOBOX_TEMPLATE])
    if not infoboxes:
        return None
    return read_params(infoboxes[0])


def read_generation_label(page_title: str, infobox: Mapping[str, str]) -> str:
    numeral = infobox.get(GENERATION_KEY, "").strip()
    if ROMAN_NUMERAL_PATTERN.match(numeral) is None:
        raise AbilityPageParseError(page_title, GENERATION_KEY, numeral)
    return GENERATION_LABEL.format(numeral=numeral)


def read_desc_rows(page_title: str, lead: Section) -> tuple[DescRow, ...]:
    """Return every `AbilityInfobox/desc` row of the lead in page order.

    `AbilityInfobox/desc/Champions` has another template name and is not read.
    """
    rows: list[DescRow] = []
    for template in find_templates(lead.parse(), [DESC_TEMPLATE]):
        params = read_params(template)
        numeral = params.get(DESC_GENERATION_PARAM, "")
        if ROMAN_NUMERAL_PATTERN.match(numeral) is None:
            raise AbilityPageParseError(page_title, f"{DESC_TEMPLATE} generation", numeral)
        rows.append(
            DescRow(
                generation_label=GENERATION_LABEL.format(numeral=numeral),
                wikitext=str(template.get(DESC_TEXT_PARAM).value)
                if template.has(DESC_TEXT_PARAM)
                else "",
            )
        )
    return tuple(rows)


def read_holders(sections: Sequence[Section]) -> tuple[HolderEntry, ...]:
    """Return the species of every `Ability/entry` row of the "Pokémon with X" sections.

    Named parameters (`form`, `formlink`, notes) leave the positions of the others alone.
    """
    holders: list[HolderEntry] = []
    for section in sections:
        if not section.path or not section.path[0].lower().startswith(HOLDER_SECTION_PREFIX):
            continue
        for template in find_templates(section.parse(), [ENTRY_TEMPLATE]):
            if template.has(ENTRY_SPECIES_PARAM):
                name = str(template.get(ENTRY_SPECIES_PARAM).value).strip()
                holders.append(HolderEntry(species_name=name))
    return tuple(holders)


def parse_ability_page(page: Page) -> AbilityPageExtract | SkippedPage:
    """Parse 1 pinned ability page. Raises `AbilityPageParseError` for a value no rule reads."""
    masked = mask_comments(page.wikitext)
    sections = split_sections(masked)
    name = page.title.removesuffix(TITLE_SUFFIX)
    ability_id = build_node_id(name)
    infobox = read_ability_infobox(sections[0])
    if infobox is None:
        return SkippedPage(
            title=page.title,
            page_id=page.page_id,
            ability_id=ability_id,
            reason=SkipReason.NO_INFOBOX,
        )
    categories = read_categories(masked)
    return AbilityPageExtract(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        id=ability_id,
        name=name,
        generation_label=read_generation_label(page.title, infobox),
        aliases=read_aliases(split_sections(page.wikitext), infobox, name, ABILITY_ALIAS_RULES),
        categories=tuple(sorted(categories)),
        ability_categories=tuple(
            sorted(
                category for category in categories if category.startswith(ABILITY_CATEGORY_PREFIX)
            )
        ),
        desc_rows=read_desc_rows(page.title, sections[0]),
        holders=read_holders(sections),
    )
