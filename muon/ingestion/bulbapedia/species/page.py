"""Species page facts: infobox values, abilities, base stats, genus, aliases and categories.

Every value is read per node of a `FormSet`. A form takes its own value where the page
gives one and the base node's value otherwise. A fact the rules cannot place becomes a
`Residue` item, and a rule with no residue path (a missing `ndex`, an unknown `gendercode`)
raises `PageParseError`, which fails the page.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mwparserfromhell
from mwparserfromhell.nodes import Tag, Template

from muon.ingestion.bulbapedia.species.extract import (
    AbilitySlot,
    BaseStats,
    FormFacts,
    NoteKind,
    PageNote,
    Residue,
    ResidueKind,
)
from muon.ingestion.bulbapedia.species.forms import (
    DEFAULT_FORM_INDEX,
    NAME_KEY,
    FormSet,
    ResolvedForm,
    flatten_text,
    match_subsection,
    rank_mega_forms,
    read_region,
)
from muon.ingestion.bulbapedia.wikitext import (
    COMMENT_PATTERN,
    Section,
    find_sections,
    find_templates,
    read_params,
    read_small_captions,
    render_node,
    to_plain_text,
)

INFOBOX_TEMPLATE = "pokémon infobox"
INFOBOX_LABEL = "Pokémon Infobox"
STATS_PATH = ("Game data", "Stats", "Base stats")
LANGUAGES_PATH = ("In other languages",)
STATS_TEMPLATES = ("stats", "basestats", "base stats", "basestats with rby")
OTHER_LANGUAGES_TEMPLATE = "other languages"
PATH_SEPARATOR = " > "
CATEGORY_PATTERN = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)

GENDER_RATES: Mapping[int, float] = {
    0: 0.0,
    31: 0.125,
    63: 0.25,
    127: 0.5,
    191: 0.75,
    223: 0.875,
    254: 1.0,
}
GENDERLESS_CODE = 255
GROWTH_RATES: Mapping[int, str] = {
    600_000: "erratic",
    800_000: "fast",
    1_000_000: "medium_fast",
    1_059_860: "medium_slow",
    1_250_000: "slow",
    1_640_000: "fluctuating",
}
BODY_SHAPES: Mapping[str, str] = {
    "01": "ball",
    "02": "squiggle",
    "03": "fish",
    "04": "arms",
    "05": "blob",
    "06": "upright",
    "07": "legs",
    "08": "quadruped",
    "09": "wings",
    "10": "tentacles",
    "11": "heads",
    "12": "humanoid",
    "13": "bug_wings",
    "14": "armor",
}
CATEGORY_FLAGS: Mapping[str, str] = {
    "Legendary Pokémon": "is_legendary",
    "Mythical Pokémon": "is_mythical",
    "First partner Pokémon": "is_starter",
}
STAT_PARAMS: Mapping[str, str] = {
    "HP": "hp",
    "Attack": "attack",
    "Defense": "defense",
    "SpAtk": "sp_attack",
    "SpDef": "sp_defense",
    "Speed": "speed",
}
EV_PARAMS = ("evhp", "evat", "evde", "evsa", "evsd", "evsp")
MAX_EV_AMOUNT = 3

NDEX_KEY = "ndex"
GENERATION_KEY = "generation"
GENDER_KEY = "gendercode"
CATCH_RATE_KEY = "catchrate"
EXP_YIELD_KEY = "expyield"
GROWTH_KEY = "lv100exp"
EGG_CYCLES_KEY = "eggcycles"
FRIENDSHIP_KEY = "friendship"
COLOR_KEY = "color"
SHAPE_KEY = "body"
FORM_COLORS_KEY = "formcolors"
FORM_SHAPES_KEY = "formbody"
EV_FORMS_KEY = "evforms"
GENUS_KEY = "category"
TYPE_KEYS = ("type1", "type2")
FORM_TYPE_KEYS = ("form{index}type1", "form{index}type2")
EGG_GROUP_KEYS = ("egggroup1", "egggroup2")
HEIGHT_KEY = "height-m"
WEIGHT_KEY = "weight-kg"
ALIAS_INFOBOX_KEYS = ("jname", "tmname")

BASE_ABILITY_KEYS = ("ability1", "ability2")
INDEXED_ABILITY_PATTERN = re.compile(r"^ability(\d+)-(\d+)$")
MEGA_ABILITY_KEYS = ("abilitym", "abilitym2")
HIDDEN_ABILITY_SLOTS = (("abilityd", "abilitydcaption"), ("abilityd2", "abilityd2caption"))
ABILITY_NOTE_KEYS = ("ability1note", "ability2note")

GENUS_SUFFIX = " Pokémon"
GENUS_CAPTION_TEMPLATE = "tt"
CAPTION_PADDING = "() "
LINE_BREAK_PATTERN = re.compile(r"<(?:br|hr)\s*/?>", re.IGNORECASE)
ROMANIZATION_PATTERN = re.compile(r"''(.+?)(?:''|$)")
ALIAS_PADDING = "' *†‡"  # quote debris and the footnote markers of `{{tt|*|note}}`
MEANING_SUFFIX = "meaning"
LANGUAGE_KEY_PATTERN = re.compile(r"^[A-Za-z_]+$")
EXCLUDED_LANGUAGE_KEYS = frozenset({"type"})
TEMPLATE_OPENERS = ("{{", "[[")
TEMPLATE_CLOSERS = ("}}", "]]")
TEMPLATE_BRACES = 2

GENERATION_RANGE_PATTERN = re.compile(r"^Generations? ([IVX]+)(?: ?(?:-|to) ?([IVX]+)| onwards?)?$")
VERSION_RANGE_PATTERN = re.compile(r"^Version [\d.]+(\+)?$")
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}
SNAPSHOT_GENERATION = 9
EV_FOOTNOTE_MARK = "*"
OPEN_ENDED_GENERATION = 99


class PageParseError(Exception):
    """A page broke a rule that has no residue path. The message names all 4 fields."""

    def __init__(self, page_title: str, section_path: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: cannot read {field} in {section_path}: {value!r}")
        self.page_title = page_title
        self.section_path = section_path
        self.field = field
        self.value = value


def read_infobox(lead: Section) -> dict[str, str] | None:
    """Return the parameters of `Pokémon Infobox` with comments removed, or `None` if absent."""
    infoboxes = find_templates(lead.parse(), [INFOBOX_TEMPLATE])
    if not infoboxes:
        return None
    return {
        name: COMMENT_PATTERN.sub("", value).strip()
        for name, value in read_params(infoboxes[0]).items()
    }


def read_categories(wikitext: str) -> frozenset[str]:
    return frozenset(name.strip() for name in CATEGORY_PATTERN.findall(wikitext))


def text_of(infobox: Mapping[str, str], key: str) -> str:
    return flatten_text(infobox.get(key, ""))


def read_int(page_title: str, infobox: Mapping[str, str], key: str) -> int:
    value = text_of(infobox, key)
    if not value.isdigit():
        raise PageParseError(page_title, INFOBOX_LABEL, key, value)
    return int(value)


def read_roman(numeral: str) -> int:
    total = 0
    for current, following in zip(numeral, [*numeral[1:], ""], strict=False):
        value = ROMAN_VALUES[current]
        total += -value if following and ROMAN_VALUES[following] > value else value
    return total


def parse_generation_range(title: str) -> tuple[int, int | None] | None:
    """Return the first generation and the last (`None` when open-ended) a title spans.

    "Generations I-V" gives (1, 5), "Generation VI onward" gives (6, None) and
    "Generation V" gives (5, 5). A game version title ("Version 1.0.1+") is a span inside
    Generation IX, open-ended with a "+". Any other title gives `None`.
    """
    version = VERSION_RANGE_PATTERN.match(title)
    if version:
        return SNAPSHOT_GENERATION, None if version.group(1) else SNAPSHOT_GENERATION
    match = GENERATION_RANGE_PATTERN.match(title)
    if match is None:
        return None
    first = read_roman(match.group(1))
    if match.group(2):
        return first, read_roman(match.group(2))
    is_open = title.endswith(("onward", "onwards"))
    return first, None if is_open else first


@dataclass(frozen=True)
class StatsCandidate:
    span: tuple[int, int | None] | None
    stats: BaseStats


def covers_snapshot(span: tuple[int, int | None]) -> bool:
    first, last = span
    return first <= SNAPSHOT_GENERATION and (last is None or last >= SNAPSHOT_GENERATION)


def pick_stats(options: Sequence[StatsCandidate]) -> BaseStats:
    """Pick the stats of Generation IX (the latest span when several cover it), else the
    unranged ones, else the latest range."""
    covering = [o for o in options if o.span is not None and covers_snapshot(o.span)]
    if covering:
        return max(covering, key=lambda option: latest_generation(option.span)).stats
    for option in options:
        if option.span is None:
            return option.stats
    return max(options, key=lambda option: latest_generation(option.span)).stats


def latest_generation(span: tuple[int, int | None] | None) -> int:
    if span is None:
        return 0
    return span[1] if span[1] is not None else OPEN_ENDED_GENERATION


def list_stats_titles(sections: Sequence[Section]) -> tuple[str, ...]:
    """Return the form subsection titles under Base stats, generation ranges left out."""
    depth = len(STATS_PATH) + 1
    return tuple(
        section.title
        for section in find_sections(sections, STATS_PATH)
        if len(section.path) == depth and parse_generation_range(section.title) is None
    )


def read_stats_template(section: Section, page_title: str) -> BaseStats | None:
    templates = find_templates(section.parse(), STATS_TEMPLATES)
    if not templates:
        return None
    params = read_params(templates[0])
    path = PATH_SEPARATOR.join(section.path)
    values: dict[str, int] = {}
    for param, field in STAT_PARAMS.items():
        value = flatten_text(params.get(param, ""))
        if not value.isdigit():
            raise PageParseError(page_title, path, param, value)
        values[field] = int(value)
    return BaseStats(**values)


def locate_stats_owner(
    section: Section, forms: FormSet, default_id: str
) -> tuple[tuple[str, ...], tuple[int, int | None] | None]:
    """Return the node ids a Base stats section belongs to and the generation span it names."""
    depth = len(section.path) - len(STATS_PATH)
    span = parse_generation_range(section.title)
    if depth == 0:
        return (default_id,), None
    if depth == 1 and span is not None:
        return (default_id,), span
    if depth == 1:
        return forms.subsections.get(section.title, ()), None
    if depth == 2 and span is not None:
        return forms.subsections.get(section.path[-2], ()), span
    return (), None


def read_base_stats(
    sections: Sequence[Section], forms: FormSet, page_title: str
) -> dict[str, BaseStats]:
    """Return the base stats per node id for the nodes that have a subsection of their own.

    The base node must have stats: `PageParseError` otherwise.
    """
    default_id = forms.forms[0].id
    options: dict[str, list[StatsCandidate]] = defaultdict(list)
    for section in find_sections(sections, STATS_PATH):
        owners, span = locate_stats_owner(section, forms, default_id)
        stats = read_stats_template(section, page_title) if owners else None
        if stats is None:
            continue
        for owner in owners:
            options[owner].append(StatsCandidate(span, stats))
    if default_id not in options:
        raise PageParseError(page_title, PATH_SEPARATOR.join(STATS_PATH), "base stats", "")
    return {owner: pick_stats(candidates) for owner, candidates in options.items()}


def reduce_ability_name(value: str) -> str:
    """Return the text before the first template or tag, e.g. "Gorilla Tactics{{!}}...""."""
    parts: list[str] = []
    for node in mwparserfromhell.parse(value).nodes:
        if isinstance(node, Template | Tag):
            break
        parts.append(render_node(node))
    return " ".join("".join(parts).split())


def match_caption(caption: str, forms: Sequence[ResolvedForm]) -> tuple[str, ...]:
    """Return the node ids a caption names: form names, a list of them, or a regional form."""
    ids = match_subsection(caption, tuple(forms))
    if ids:
        return ids
    region = read_region(caption)
    return tuple(form.id for form in forms if region and form.region == region)


def caption_ids(captions: Sequence[str], forms: Sequence[ResolvedForm]) -> tuple[str, ...]:
    """Return the node ids named by `captions`, one per line, "(Alolan Form)" included."""
    ids: list[str] = []
    for caption in captions:
        for line in caption.split("\n"):
            ids.extend(match_caption(line.strip(CAPTION_PADDING), forms))
    return tuple(dict.fromkeys(ids))


def build_mega_ids(forms: FormSet) -> dict[int, str]:
    """Return the node id of each Mega form by its position among the infobox Mega forms."""
    ranks = rank_mega_forms(forms.candidates)
    return {
        ranks[result.index]: result.form_id
        for result in forms.candidates
        if result.index in ranks and result.form_id
    }


@dataclass(frozen=True)
class AbilityReading:
    """The abilities of a page per node id, what was left as residue and what to report."""

    abilities: Mapping[str, tuple[AbilitySlot, ...]]
    residue: tuple[Residue, ...]
    notes: tuple[PageNote, ...]


def read_visible_abilities(infobox: Mapping[str, str], forms: FormSet) -> dict[str, list[str]]:
    """Read ability1 and ability2, abilityN-1 and N-2, abilitym and abilitym2 into node ids."""
    default_id = forms.forms[0].id
    ids_by_index = {form.index: form.id for form in forms.forms}
    visible: dict[str, list[str]] = defaultdict(list)
    for key in BASE_ABILITY_KEYS:
        name = reduce_ability_name(infobox.get(key, ""))
        if name:
            visible[default_id].append(name)
    indexed = sorted(
        (int(match.group(1)), int(match.group(2)), key)
        for key in infobox
        if (match := INDEXED_ABILITY_PATTERN.match(key))
    )
    for index, _, key in indexed:
        name = reduce_ability_name(infobox[key])
        captioned = caption_ids(read_small_captions(infobox[key]), forms.forms)
        owners = captioned or ((ids_by_index[index],) if index in ids_by_index else ())
        for owner in owners if name else ():
            visible[owner].append(name)
    mega_ids = build_mega_ids(forms)
    for rank, key in enumerate(MEGA_ABILITY_KEYS):
        name = reduce_ability_name(infobox.get(key, ""))
        if name and rank in mega_ids:
            visible[mega_ids[rank]].append(name)
    return visible


def read_noted_forms(
    infobox: Mapping[str, str], forms: FormSet
) -> tuple[dict[str, str], tuple[Residue, ...]]:
    """Find the forms an ability note names. Their non-hidden abilities become residue."""
    default_id = forms.forms[0].id
    noted: dict[str, str] = {}
    residue: list[Residue] = []
    for key in ABILITY_NOTE_KEYS:
        raw = infobox.get(key, "")
        for form_id in caption_ids(read_small_captions(raw), forms.forms):
            if form_id == default_id or form_id in noted:
                continue
            noted[form_id] = key
            residue.append(
                Residue(
                    kind=ResidueKind.ABILITIES,
                    section_path=f"{INFOBOX_LABEL}{PATH_SEPARATOR}{key}",
                    form_ids=(form_id,),
                    wikitext=raw,
                    text=to_plain_text(raw),
                )
            )
    return noted, tuple(residue)


def read_hidden_abilities(infobox: Mapping[str, str], forms: FormSet) -> dict[str, list[str]]:
    """Assign abilityd and abilityd2 by caption. An unnamed slot covers the nodes no caption
    names. Mega forms take no hidden ability."""
    mega_ids = set(build_mega_ids(forms).values())
    eligible = [form.id for form in forms.forms if form.id not in mega_ids]
    slots: list[tuple[str, list[str]]] = []
    for key, caption_key in HIDDEN_ABILITY_SLOTS:
        name = reduce_ability_name(infobox.get(key, ""))
        if not name:
            continue
        captions = to_plain_text(infobox.get(caption_key, ""))
        named = [i for i in caption_ids([captions], forms.forms) if i in eligible]
        slots.append((name, named))
    claimed = {form_id for _, named in slots for form_id in named}
    hidden: dict[str, list[str]] = defaultdict(list)
    for name, named in slots:
        for form_id in named or [i for i in eligible if i not in claimed]:
            hidden[form_id].append(name)
    return hidden


def merge_ability_slots(
    form_id: str, visible: Sequence[str], hidden: Sequence[str]
) -> tuple[tuple[AbilitySlot, ...], tuple[PageNote, ...]]:
    """Keep 1 slot per ability name. A name that is both visible and hidden stays visible."""
    slots: list[AbilitySlot] = []
    notes: list[PageNote] = []
    seen: set[str] = set()
    for name, is_hidden in [*((n, False) for n in visible), *((n, True) for n in hidden)]:
        if name in seen:
            notes.append(
                PageNote(
                    kind=NoteKind.DUPLICATE_ABILITY,
                    section_path=INFOBOX_LABEL,
                    detail=f"{form_id} lists {name} twice",
                )
            )
            continue
        seen.add(name)
        slots.append(AbilitySlot(name=name, is_hidden=is_hidden))
    return tuple(slots), tuple(notes)


def read_abilities(infobox: Mapping[str, str], forms: FormSet) -> AbilityReading:
    """Read the abilities of every node by the layout of `Template:Pokémon Infobox`.

    A node with no ability fields of its own takes the base node's non-hidden abilities,
    unless an ability note names it, in which case it gets residue instead.
    """
    default_id = forms.forms[0].id
    visible = read_visible_abilities(infobox, forms)
    noted, residue = read_noted_forms(infobox, forms)
    hidden = read_hidden_abilities(infobox, forms)
    abilities: dict[str, tuple[AbilitySlot, ...]] = {}
    notes: list[PageNote] = []
    for form in forms.forms:
        if form.id in noted:
            own: list[str] = []
        else:
            own = visible.get(form.id) or visible.get(default_id, [])
        slots, slot_notes = merge_ability_slots(form.id, own, hidden.get(form.id, []))
        abilities[form.id] = slots
        notes.extend(slot_notes)
    return AbilityReading(abilities, residue, tuple(notes))


@dataclass(frozen=True)
class GenusItem:
    text: str
    caption: str | None


def read_genus_items(raw: str) -> list[GenusItem]:
    """Split a `category` value into its `{{tt|text|caption}}` items, or 1 plain item."""
    items = [
        GenusItem(to_plain_text(str(t.get(1).value)), to_plain_text(str(t.get(2).value)))
        for t in find_templates(mwparserfromhell.parse(raw), [GENUS_CAPTION_TEMPLATE])
        if t.has(1) and t.has(2)
    ]
    if items:
        return items
    return [GenusItem(to_plain_text(raw).removesuffix(GENUS_SUFFIX).strip(), None)]


def read_genus(
    infobox: Mapping[str, str], forms: FormSet
) -> tuple[dict[str, str | None], tuple[Residue, ...]]:
    """Read `category` per node. A caption naming forms sends its text to those forms and any
    other caption gives the base value. Once a page names forms, a node no caption covers is
    residue; otherwise every node takes the base value."""
    raw = infobox.get(GENUS_KEY, "").strip()
    default_id = forms.forms[0].id
    if not raw:
        return {form.id: None for form in forms.forms}, ()
    assigned: dict[str, str] = {}
    names_forms = False
    for item in read_genus_items(raw):
        ids = caption_ids([item.caption], forms.forms) if item.caption else ()
        names_forms = names_forms or bool(ids)
        for form_id in ids or (default_id,):
            assigned.setdefault(form_id, item.text)
    genus: dict[str, str | None] = {}
    residue: list[Residue] = []
    for form in forms.forms:
        base_value = None if names_forms else assigned.get(default_id)
        genus[form.id] = assigned.get(form.id, base_value)
        if genus[form.id] is None:
            residue.append(
                Residue(
                    kind=ResidueKind.GENUS,
                    section_path=f"{INFOBOX_LABEL}{PATH_SEPARATOR}{GENUS_KEY}",
                    form_ids=(form.id,),
                    wikitext=raw,
                    text=to_plain_text(raw),
                )
            )
    return genus, tuple(residue)


def split_top_level(body: str) -> list[str]:
    """Split `body` on the `|` that sit outside `{{ }}` and `[[ ]]`."""
    pieces: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(body):
        pair = body[index : index + TEMPLATE_BRACES]
        if pair in TEMPLATE_OPENERS:
            depth += 1
            index += TEMPLATE_BRACES
        elif pair in TEMPLATE_CLOSERS:
            depth = max(0, depth - 1)
            index += TEMPLATE_BRACES
        else:
            if body[index] == "|" and depth == 0:
                pieces.append(body[start:index])
                start = index + 1
            index += 1
    pieces.append(body[start:])
    return pieces


def read_language_values(sections: Sequence[Section]) -> list[str]:
    """Return the raw value of each language parameter of `Other languages`, in order.

    The template is split on its own top-level pipes: an unclosed `''` in 1 value makes
    `mwparserfromhell` fold the parameters after it into that value.
    """
    values: list[str] = []
    for section in find_sections(sections, LANGUAGES_PATH):
        for template in find_templates(section.parse(), [OTHER_LANGUAGES_TEMPLATE]):
            body = str(template)[TEMPLATE_BRACES:-TEMPLATE_BRACES]
            for piece in split_top_level(body)[1:]:
                key, separator, value = piece.partition("=")
                key = key.strip()
                if separator and is_language_key(key):
                    values.append(value.strip())
    return values


def is_language_key(key: str) -> bool:
    return (
        LANGUAGE_KEY_PATTERN.match(key) is not None
        and not key.endswith(MEANING_SUFFIX)
        and key not in EXCLUDED_LANGUAGE_KEYS
    )


def split_alias_names(raw: str) -> list[str]:
    """Return the names and romanizations in 1 language value, in reading order."""
    names: list[str] = []
    for line in LINE_BREAK_PATTERN.split(raw):
        names.append(ROMANIZATION_PATTERN.sub("", line))
        names.extend(ROMANIZATION_PATTERN.findall(line))
    cleaned = (to_plain_text(name).strip(ALIAS_PADDING) for name in names)
    return [name for name in cleaned if name]


def read_aliases(
    sections: Sequence[Section], infobox: Mapping[str, str], english_name: str
) -> tuple[str, ...]:
    """Return the other-language names, then `jname` and `tmname`, without repeats or the
    English name."""
    raw_values = [
        *read_language_values(sections),
        *(infobox.get(key, "") for key in ALIAS_INFOBOX_KEYS),
    ]
    names = (name for raw in raw_values for name in split_alias_names(raw))
    return tuple(name for name in dict.fromkeys(names) if name != english_name)


@dataclass(frozen=True)
class FactInputs:
    page_title: str
    infobox: Mapping[str, str]
    forms: FormSet
    sections: tuple[Section, ...]
    categories: frozenset[str]


@dataclass(frozen=True)
class PageValues:
    """The values every node of a page shares, read and checked once."""

    pokedex_number: int
    generation: int
    female_rate: float | None
    is_genderless: bool
    catch_rate: int
    base_exp_yield: int
    growth_rate: str
    egg_cycles: int
    base_friendship: int
    color: str | None
    shape: str | None
    egg_groups: tuple[str, ...]


def read_page_values(inputs: FactInputs) -> PageValues:
    infobox, title = inputs.infobox, inputs.page_title
    gender_code = read_int(title, infobox, GENDER_KEY)
    if gender_code != GENDERLESS_CODE and gender_code not in GENDER_RATES:
        raise PageParseError(title, INFOBOX_LABEL, GENDER_KEY, str(gender_code))
    growth_total = text_of(infobox, GROWTH_KEY).replace(",", "")
    growth_rate = GROWTH_RATES.get(int(growth_total)) if growth_total.isdigit() else None
    if growth_rate is None:
        raise PageParseError(title, INFOBOX_LABEL, GROWTH_KEY, growth_total)
    shape_code = text_of(infobox, SHAPE_KEY)
    if shape_code and shape_code not in BODY_SHAPES:
        raise PageParseError(title, INFOBOX_LABEL, SHAPE_KEY, shape_code)
    return PageValues(
        pokedex_number=read_int(title, infobox, NDEX_KEY),
        generation=read_int(title, infobox, GENERATION_KEY),
        female_rate=GENDER_RATES.get(gender_code),
        is_genderless=gender_code == GENDERLESS_CODE,
        catch_rate=read_int(title, infobox, CATCH_RATE_KEY),
        base_exp_yield=read_int(title, infobox, EXP_YIELD_KEY),
        growth_rate=growth_rate,
        egg_cycles=read_int(title, infobox, EGG_CYCLES_KEY),
        base_friendship=read_int(title, infobox, FRIENDSHIP_KEY),
        color=text_of(infobox, COLOR_KEY) or None,
        shape=BODY_SHAPES.get(shape_code),
        egg_groups=tuple(g for g in (text_of(infobox, k) for k in EGG_GROUP_KEYS) if g),
    )


def read_types(inputs: FactInputs, form: ResolvedForm) -> tuple[str, ...]:
    """Return the own types of a form, else the base types. `type1` is required."""
    infobox = inputs.infobox
    if form.index != DEFAULT_FORM_INDEX:
        own = [text_of(infobox, key.format(index=form.index)) for key in FORM_TYPE_KEYS]
        if own[0]:
            return tuple(name for name in own if name)
    base = [text_of(infobox, key) for key in TYPE_KEYS]
    if not base[0]:
        raise PageParseError(inputs.page_title, INFOBOX_LABEL, TYPE_KEYS[0], "")
    return tuple(name for name in base if name)


def read_measure(
    inputs: FactInputs, form: ResolvedForm, key: str
) -> tuple[float | None, PageNote | None]:
    """Read `height-m` or `weight-kg` for a form. A non-numeric value is left out and noted."""
    own_key = key if form.index == DEFAULT_FORM_INDEX else f"{key}{form.index}"
    has_own = bool(text_of(inputs.infobox, own_key))
    value = text_of(inputs.infobox, own_key if has_own else key)
    try:
        return float(value), None
    except ValueError:
        note = PageNote(
            kind=NoteKind.DROPPED_VALUE,
            section_path=f"{INFOBOX_LABEL}{PATH_SEPARATOR}{own_key if has_own else key}",
            detail=f"{form.id}: {value!r} is not a number",
        )
        return None, note


def read_ev_yield(inputs: FactInputs, form: ResolvedForm) -> dict[str, int]:
    """Read the EV yield of a form: its own keys when `evforms` covers it, else the base's."""
    infobox = inputs.infobox
    covered = text_of(infobox, EV_FORMS_KEY)
    is_own = covered.isdigit() and form.index <= int(covered)
    suffix = str(form.index) if is_own and form.index != DEFAULT_FORM_INDEX else ""
    yields: dict[str, int] = {}
    for key in EV_PARAMS:
        value = text_of(infobox, f"{key}{suffix}").removesuffix(EV_FOOTNOTE_MARK)
        if not value or value == "0":
            continue
        if not value.isdigit() or int(value) > MAX_EV_AMOUNT:
            raise PageParseError(inputs.page_title, INFOBOX_LABEL, f"{key}{suffix}", value)
        yields[key] = int(value)
    return yields


@dataclass(frozen=True)
class FormSources:
    """What `read_form_facts` reads once per page and then looks up per node."""

    values: PageValues
    abilities: Mapping[str, tuple[AbilitySlot, ...]]
    genus: Mapping[str, str | None]
    base_stats: Mapping[str, BaseStats]
    aliases: tuple[str, ...]


def build_form_facts(
    inputs: FactInputs, form: ResolvedForm, sources: FormSources
) -> tuple[FormFacts, tuple[PageNote, ...]]:
    values = sources.values
    height, height_note = read_measure(inputs, form, HEIGHT_KEY)
    weight, weight_note = read_measure(inputs, form, WEIGHT_KEY)
    is_default = form.is_default
    flags = {field: name in inputs.categories for name, field in CATEGORY_FLAGS.items()}
    copies_look = not text_of(inputs.infobox, FORM_COLORS_KEY)
    copies_shape = not text_of(inputs.infobox, FORM_SHAPES_KEY)
    facts = FormFacts(
        id=form.id,
        name=form.name,
        form_index=form.index,
        is_default=is_default,
        region=form.region,
        generation=values.generation if is_default else None,
        pokedex_number=values.pokedex_number,
        types=read_types(inputs, form),
        abilities=sources.abilities[form.id],
        egg_groups=values.egg_groups,
        height=height,
        weight=weight,
        female_rate=values.female_rate,
        is_genderless=values.is_genderless,
        catch_rate=values.catch_rate,
        base_exp_yield=values.base_exp_yield,
        growth_rate=values.growth_rate,
        egg_cycles=values.egg_cycles,
        base_friendship=values.base_friendship,
        color=values.color if is_default or copies_look else None,
        shape=values.shape if is_default or copies_shape else None,
        genus=sources.genus[form.id],
        is_legendary=flags["is_legendary"],
        is_mythical=flags["is_mythical"],
        is_starter=flags["is_starter"],
        base_stats=sources.base_stats.get(form.id) or sources.base_stats[inputs.forms.forms[0].id],
        ev_yield=read_ev_yield(inputs, form),
        aliases=sources.aliases if is_default else (),
    )
    notes = tuple(note for note in (height_note, weight_note) if note)
    return facts, notes


def read_form_facts(
    inputs: FactInputs,
) -> tuple[tuple[FormFacts, ...], tuple[Residue, ...], tuple[PageNote, ...]]:
    """Read the facts of every node. A page that breaks a required rule raises `PageParseError`."""
    abilities = read_abilities(inputs.infobox, inputs.forms)
    genus, genus_residue = read_genus(inputs.infobox, inputs.forms)
    english_name = text_of(inputs.infobox, NAME_KEY)
    sources = FormSources(
        values=read_page_values(inputs),
        abilities=abilities.abilities,
        genus=genus,
        base_stats=read_base_stats(inputs.sections, inputs.forms, inputs.page_title),
        aliases=read_aliases(inputs.sections, inputs.infobox, english_name),
    )
    facts: list[FormFacts] = []
    notes: list[PageNote] = [*abilities.notes]
    for form in inputs.forms.forms:
        form_facts, form_notes = build_form_facts(inputs, form, sources)
        facts.append(form_facts)
        notes.extend(form_notes)
    return tuple(facts), (*abilities.residue, *genus_residue), tuple(notes)
