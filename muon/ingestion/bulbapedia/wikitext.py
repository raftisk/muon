"""Wikitext helpers shared by every Bulbapedia page parser.

A page is cut into sections on its heading lines before any template parsing, so an
unbalanced `{{` or table in 1 section cannot hide the headings after it. Templates are
matched on normalized names, and values render to plain text through `TEXT_TEMPLATES`. One
render pass also reports the typed templates it showed and the ones it dropped. This module
knows nothing of page types or of the ontology.
"""

import html
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain

import mwparserfromhell
from mwparserfromhell.nodes import (
    ExternalLink,
    Heading,
    HTMLEntity,
    Node,
    Tag,
    Template,
    Text,
    Wikilink,
)
from mwparserfromhell.wikicode import Wikicode

HEADING_PATTERN = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
COMMENT_PATTERN = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
NON_NEWLINE_PATTERN = re.compile(r"[^\n]")
MASK_CHARACTER = " "
PATH_SEPARATOR = " > "
LEAD_LEVEL = 1
LEAD_TITLE = ""
PLAIN_TEXT_SEPARATOR = "\n"
CATEGORY_PATTERN = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)
LANGUAGES_PATH = ("In other languages",)
LINE_BREAK_PATTERN = re.compile(r"<(?:br|hr)\s*/?>", re.IGNORECASE)
ROMANIZATION_PATTERN = re.compile(r"''(.+?)(?:''|$)")
TRAILING_NOTE_PATTERN = re.compile(r"\s*\([^()]*\)\s*$")
ALIAS_PADDING = "' *†‡"  # quote debris and the footnote markers of `{{tt|*|note}}`
MEANING_SUFFIX = "meaning"
LANGUAGE_KEY_PATTERN = re.compile(r"^[A-Za-z_]+$")
EXCLUDED_LANGUAGE_KEYS = frozenset({"type"})
TEMPLATE_OPENERS = ("{{", "[[")
TEMPLATE_CLOSERS = ("}}", "]]")
TEMPLATE_BRACES = 2
LINE_BREAK_TAG = "br"
SMALL_TAG = "small"
DROPPED_TAGS = frozenset({"ref", "sup", SMALL_TAG})
NON_TEXT_LINK_PREFIXES = ("file:", "image:")
TARGET_PARAM = "1"


POKEMON_PREFIX = "Pokémon "
TYPE_SUFFIX = "-type"
EGG_GROUP_SUFFIX = " Egg Group"
# `{{2v2|Black|White|2}}` shows "Black 2 and White 2": the 3rd parameter follows both names.
VERSION_PAIR_LAYOUT = "{0} {2} and {1} {2}"
BLACK_2_WHITE_2 = "Pokémon Black 2 and White 2"


@dataclass(frozen=True)
class TextRule:
    """How a template renders as text: the first of `params` that is present, else `literal`.

    `prefix` and `suffix` wrap the value of the last parameter in `params` only. An earlier
    parameter is display text and shows as written, and `literal` is never wrapped. `codes`
    maps a value of the last parameter that is a code, such as `{{aniseries|DP}}`, to the
    text the template shows for it; prefix and suffix do not wrap a mapped code.
    `layout` replaces all of that: a format string over the values of `params`, where `{0}`
    is the 1st name in `params`, `{1}` the 2nd and so on, and an absent parameter is empty.
    """

    params: tuple[str, ...] = ()
    literal: str = ""
    prefix: str = ""
    suffix: str = ""
    layout: str = ""
    codes: Mapping[str, str] = field(default_factory=dict)


FIRST_PARAM = TextRule(params=("1",))
SECOND_PARAM = TextRule(params=("2",))
SECOND_OR_FIRST_PARAM = TextRule(params=("2", "1"))
THIRD_OR_FIRST_PARAM = TextRule(params=("3", "1"))
THIRD_OR_SECOND_PARAM = TextRule(params=("3", "2"))
FOURTH_OR_SECOND_PARAM = TextRule(params=("4", "2"))
FIRST_TWO_PARAMS = ("1", "2")
FIRST_THREE_PARAMS = ("1", "2", "3")
FIRST_FIVE_PARAMS = ("1", "2", "3", "4", "5")
TCG = "TCG"


# `{{aniseries|AG}}` and `{{aniseries|RS}}` show the same series; OS and GS are both
# parts of the original series and show their own titles.
ANIME_SERIES_CODES: Mapping[str, str] = {
    "OS": "Pokémon the Series: The Beginning",
    "GS": "Pokémon the Series: Gold and Silver",
    "RS": "Pokémon the Series: Ruby and Sapphire",
    "AG": "Pokémon the Series: Ruby and Sapphire",
    "DP": "Pokémon the Series: Diamond and Pearl",
    "BW": "Pokémon the Series: Black & White",
    "XY": "Pokémon the Series: XY",
    "SM": "Pokémon the Series: Sun & Moon",
    "JN": "Pokémon Journeys: The Series",
    "PTS": "Pokémon the Series",
    "HZ": "Pokémon Horizons: The Series",
}
GAME_QUARTET_CODES: Mapping[str, str] = {
    "BWB2W2": "Pokémon Black, White, Black 2, and White 2",
    "SMUSUM": "Pokémon Sun, Moon, Ultra Sun, and Ultra Moon",
}


def show_param_or(literal: str) -> TextRule:
    """The rule of a template that shows `literal`, or its 1st parameter in its place."""
    return TextRule(params=("1",), literal=literal)


# Rendering checked on Special:ExpandTemplates on 2026-11-13. A name absent from the table
# is dropped: boxes, notices, hatnotes, superscripts, images and layout clears.

TEXT_TEMPLATES: Mapping[str, TextRule] = {
    "tt": FIRST_PARAM,
    "color2": THIRD_OR_SECOND_PARAM,
    "mcolor": THIRD_OR_FIRST_PARAM,
    "pcolor": THIRD_OR_FIRST_PARAM,
    "acolor": THIRD_OR_FIRST_PARAM,
    "tcolor": THIRD_OR_FIRST_PARAM,
    "p": SECOND_OR_FIRST_PARAM,
    "m": SECOND_OR_FIRST_PARAM,
    "a": SECOND_OR_FIRST_PARAM,
    "type": TextRule(params=("2", "1"), suffix=TYPE_SUFFIX),
    "t": SECOND_OR_FIRST_PARAM,
    "gen": FIRST_PARAM,
    "rf": FIRST_PARAM,
    "pkmn": FIRST_PARAM,
    "obp": THIRD_OR_FIRST_PARAM,
    "wp": SECOND_OR_FIRST_PARAM,
    "g": TextRule(params=("1",), prefix=POKEMON_PREFIX),
    "game": TextRule(params=("3", "1"), prefix=POKEMON_PREFIX),
    "dl": THIRD_OR_SECOND_PARAM,
    "stat": SECOND_OR_FIRST_PARAM,
    "pkmn2": SECOND_OR_FIRST_PARAM,
    "egg": FIRST_PARAM,
    "egg3": TextRule(params=("1",), suffix=EGG_GROUP_SUFFIX),
    "shiny": TextRule(literal="Shiny"),
    "game3": SECOND_PARAM,
    "2v2": TextRule(params=("1", "2", "3"), layout=VERSION_PAIR_LAYOUT),
    "b2w2": TextRule(params=("1",), literal=BLACK_2_WHITE_2),
    "cat": SECOND_OR_FIRST_PARAM,
    "form": SECOND_OR_FIRST_PARAM,
    "tcg": SECOND_OR_FIRST_PARAM,
    "scpkmn": TextRule(literal="Pokémon"),
    "!": TextRule(literal="|"),
    "'": TextRule(literal="'"),
    "status": SECOND_OR_FIRST_PARAM,
    "weather": SECOND_OR_FIRST_PARAM,
    "pmin": SECOND_OR_FIRST_PARAM,
    "ga": SECOND_OR_FIRST_PARAM,
    "tc": SECOND_OR_FIRST_PARAM,
    "b": SECOND_OR_FIRST_PARAM,
    "i": SECOND_OR_FIRST_PARAM,
    "an": SECOND_OR_FIRST_PARAM,
    "jwp": SECOND_OR_FIRST_PARAM,
    "vg": SECOND_OR_FIRST_PARAM,
    "go": SECOND_OR_FIRST_PARAM,
    "adv": SECOND_OR_FIRST_PARAM,
    "hi": SECOND_OR_FIRST_PARAM,
    "jpn": SECOND_OR_FIRST_PARAM,
    "fd": SECOND_OR_FIRST_PARAM,
    "poko": SECOND_OR_FIRST_PARAM,
    "sbw": SECOND_OR_FIRST_PARAM,
    "si": SECOND_OR_FIRST_PARAM,
    "un": SECOND_OR_FIRST_PARAM,
    "ka": SECOND_OR_FIRST_PARAM,
    "mas": SECOND_OR_FIRST_PARAM,
    "mov": SECOND_OR_FIRST_PARAM,
    "n": SECOND_OR_FIRST_PARAM,
    "smw": SECOND_OR_FIRST_PARAM,
    "wk": SECOND_OR_FIRST_PARAM,
    "few": SECOND_OR_FIRST_PARAM,
    "ci": SECOND_OR_FIRST_PARAM,
    "ho": SECOND_OR_FIRST_PARAM,
    "jo": SECOND_OR_FIRST_PARAM,
    "kal": SECOND_OR_FIRST_PARAM,
    "al": SECOND_OR_FIRST_PARAM,
    "book": SECOND_OR_FIRST_PARAM,
    "ma": SECOND_OR_FIRST_PARAM,
    "so": SECOND_OR_FIRST_PARAM,
    "ssb": SECOND_OR_FIRST_PARAM,
    "unite": SECOND_OR_FIRST_PARAM,
    "wikipedia": SECOND_OR_FIRST_PARAM,
    "template:wp": SECOND_OR_FIRST_PARAM,
    "gdis": THIRD_OR_FIRST_PARAM,
    "tp": SECOND_PARAM,
    "file": SECOND_PARAM,
    "lang": SECOND_PARAM,
    "ap": FIRST_PARAM,
    "j": FIRST_PARAM,
    "mdc": FIRST_PARAM,
    "rtn": FIRST_PARAM,
    "v2": FIRST_PARAM,
    "tcg id": FOURTH_OR_SECOND_PARAM,
    "rt": TextRule(params=("3", "1"), prefix="Route "),
    "me": TextRule(params=("1",), prefix="Mega "),
    "badge": TextRule(params=("1",), suffix=" Badge"),
    "egg2": TextRule(params=("1",), suffix=" Group"),
    "ct": TextRule(params=("1",), suffix=TYPE_SUFFIX),
    "player": show_param_or("player"),
    "tera": show_param_or("Tera"),
    "ash": show_param_or("Ash"),
    "ashfr": show_param_or("his friends"),
    "mtr": show_param_or("Meowth"),
    "season": show_param_or("season"),
    "spin-off games": show_param_or("spin-off games"),
    "lgpe": show_param_or("Pokémon: Let's Go, Pikachu! and Let's Go, Eevee!"),
    "safari": TextRule(literal="Safari Zone"),
    "abilities": TextRule(literal="Abilities"),
    "berries": TextRule(literal="Berries"),
    "gary": TextRule(literal="Gary"),
    "todd": TextRule(literal="Todd"),
    "trt": TextRule(literal="Team Rocket"),
    "trading card game": TextRule(literal=TCG),
    "ptcg": TextRule(literal=TCG),
    "trading figure game": TextRule(literal="TFG"),
    "pokémon xd": TextRule(literal="Pokémon XD"),
    "xd": TextRule(literal="XD"),
    "detpikmov": TextRule(literal="POKÉMON Detective Pikachu"),
    "stab": TextRule(literal="STAB"),
    "ev": TextRule(literal="EV"),
    "iv": TextRule(literal="IV"),
    "pp": TextRule(literal="PP"),
    "scball": TextRule(literal="Poké Ball"),
    "pdollar": TextRule(literal="$"),
    "v": TextRule(params=FIRST_TWO_PARAMS, layout="{0} Version{1}"),
    "2t": TextRule(params=FIRST_TWO_PARAMS, layout="{0}/{1}"),
    "frac": TextRule(params=FIRST_TWO_PARAMS, layout="{0}/{1}"),
    "sk": TextRule(params=FIRST_TWO_PARAMS, layout="{0} {1}"),
    "sync": TextRule(params=FIRST_TWO_PARAMS, layout="{0} & {1}"),
    "tr": TextRule(params=FIRST_TWO_PARAMS, layout="TR{0} ({1})"),
    "ruby": TextRule(params=FIRST_TWO_PARAMS, layout="{0}({1})"),
    "3v2": TextRule(params=FIRST_THREE_PARAMS, layout="{0}, {1}, and {2}"),
    "game2": TextRule(params=FIRST_THREE_PARAMS, layout="Pokémon {0}, {1}, and {2}"),
    "4v": TextRule(params=FIRST_THREE_PARAMS, layout="{0}, {1}, {0} {2}, and {1} {2}"),
    "game5": TextRule(params=FIRST_FIVE_PARAMS, layout="Pokémon {0}, {1}, {2}, {3}, and {4}"),
    "aniseries": TextRule(
        params=("2", "1"), literal="Pokémon animated series", codes=ANIME_SERIES_CODES
    ),
    "game4": TextRule(params=("1",), codes=GAME_QUARTET_CODES),
}


@dataclass(frozen=True)
class CapturedTemplate:
    """A template that a render pass showed: its normalized name and raw 1st parameter."""

    name: str
    target: str


@dataclass(frozen=True)
class PlainText:
    """The result of 1 render pass, with the templates and drops in page order.

    `dropped` keeps repeats. A node inside a dropped tag or a file link is never visited,
    so it appears in neither field.
    """

    text: str = ""
    captured: tuple[CapturedTemplate, ...] = ()
    dropped: tuple[str, ...] = ()


@dataclass(frozen=True)
class AliasRules:
    """Where a page type keeps its other-language names.

    `template_name` is the language table template of the "In other languages" section and
    `infobox_keys` the infobox fields that also hold a name. With `should_drop_notes`, a
    comment inside a language value is blanked before the value is split, and a trailing
    parenthetical source note such as "(games, XY019-present)" is cut from each name.
    """

    template_name: str
    infobox_keys: tuple[str, ...]
    should_drop_notes: bool


@dataclass(frozen=True)
class Section:
    """1 heading and the text under it, up to the next heading of any level.

    `path` holds the normalized titles of the open ancestors and of the section itself.
    `text` is raw wikitext without the heading line.
    """

    level: int
    title: str
    path: tuple[str, ...]
    text: str

    def parse(self) -> Wikicode:
        return mwparserfromhell.parse(self.text)


def normalize_template_name(name: str) -> str:
    """Return the form in which template names compare: pages mix `Evobox-2` and `evobox-2`."""
    return name.strip().lower().replace("_", " ")


def join_plain_texts(parts: Iterable[PlainText]) -> PlainText:
    kept = tuple(parts)
    return PlainText(
        "".join(part.text for part in kept),
        tuple(chain.from_iterable(part.captured for part in kept)),
        tuple(chain.from_iterable(part.dropped for part in kept)),
    )


def render_code(code: Wikicode, captured_names: frozenset[str]) -> PlainText:
    return join_plain_texts(render_fragment(node, captured_names) for node in code.nodes)


def render_node(node: Node) -> str:
    return render_fragment(node, frozenset()).text


def render_fragment(node: Node, captured_names: frozenset[str]) -> PlainText:
    """Render 1 node, reporting the templates of `captured_names` that it showed."""
    if isinstance(node, Text):
        return PlainText(str(node.value))
    if isinstance(node, Template):
        return render_template(node, captured_names)
    if isinstance(node, Wikilink):
        return render_link(node, captured_names)
    if isinstance(node, Tag):
        return render_tag(node, captured_names)
    if isinstance(node, HTMLEntity):
        return PlainText(str(node.normalize()))
    if isinstance(node, ExternalLink):
        return render_code(node.title if node.title is not None else node.url, captured_names)
    if isinstance(node, Heading):
        return render_code(node.title, captured_names)
    return PlainText()


def render_template(template: Template, captured_names: frozenset[str]) -> PlainText:
    name = normalize_template_name(str(template.name))
    rule = TEXT_TEMPLATES.get(name)
    if rule is None:
        return PlainText(dropped=(name,))
    shown = render_rule(template, rule, captured_names)
    if name not in captured_names or not template.has(TARGET_PARAM, ignore_empty=True):
        return shown
    target = str(template.get(TARGET_PARAM).value).strip()
    return PlainText(shown.text, (CapturedTemplate(name, target), *shown.captured), shown.dropped)


def render_param(template: Template, param: str, captured_names: frozenset[str]) -> PlainText:
    """Render the value of `param`, stripped; an absent or empty parameter gives nothing."""
    if not template.has(param, ignore_empty=True):
        return PlainText()
    value = render_code(template.get(param).value, captured_names)
    return PlainText(value.text.strip(), value.captured, value.dropped)


def render_rule(template: Template, rule: TextRule, captured_names: frozenset[str]) -> PlainText:
    if rule.layout:
        values = [render_param(template, param, captured_names) for param in rule.params]
        joined = join_plain_texts(values)
        text = rule.layout.format(*(value.text for value in values)).strip()
        return PlainText(text, joined.captured, joined.dropped)
    for param in rule.params:
        if not template.has(param, ignore_empty=True):
            continue
        value = render_param(template, param, captured_names)
        if param != rule.params[-1]:
            return value
        if value.text in rule.codes:
            return PlainText(rule.codes[value.text], value.captured, value.dropped)
        text = f"{rule.prefix}{value.text}{rule.suffix}"
        return PlainText(text, value.captured, value.dropped)
    return PlainText(rule.literal)


def render_link(link: Wikilink, captured_names: frozenset[str]) -> PlainText:
    title = str(link.title).strip()
    if title.lower().startswith(NON_TEXT_LINK_PREFIXES):
        return PlainText()
    return render_code(link.text if link.text is not None else link.title, captured_names)


def render_tag(tag: Tag, captured_names: frozenset[str]) -> PlainText:
    name = str(tag.tag).strip().lower()
    if name == LINE_BREAK_TAG:
        return PlainText(PLAIN_TEXT_SEPARATOR)
    if name in DROPPED_TAGS or tag.contents is None:
        return PlainText()
    return render_code(tag.contents, captured_names)


def collapse_whitespace(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.split(PLAIN_TEXT_SEPARATOR))
    return PLAIN_TEXT_SEPARATOR.join(lines).strip()


def read_plain_text(value: str, captured_names: frozenset[str] = frozenset()) -> PlainText:
    """Return `value` as plain text with the templates the render showed and dropped.

    `PLAIN_TEXT_SEPARATOR` stands where `value` had a `<br>`. Comments, `<ref>`, `<sup>` and
    `<small>` contents and every template outside `TEXT_TEMPLATES` are removed. `captured`
    lists the templates of `captured_names` that appear in the text, each with its raw 1st
    parameter. A template inside a removed part is not listed. Read `<small>` captions with
    `read_small_captions` first.
    """
    rendered = render_code(mwparserfromhell.parse(value), captured_names)
    return PlainText(collapse_whitespace(rendered.text), rendered.captured, rendered.dropped)


def to_plain_text(value: str) -> str:
    """Return `read_plain_text(value).text`."""
    return read_plain_text(value).text


def read_small_captions(value: str) -> tuple[str, ...]:
    """Return the plain text of each `<small>` element in `value`, in order."""
    code = mwparserfromhell.parse(value)
    return tuple(
        to_plain_text(str(tag.contents))
        for tag in code.filter_tags(recursive=True)
        if str(tag.tag).strip().lower() == SMALL_TAG and tag.contents is not None
    )


def normalize_heading(raw: str) -> str:
    """Return a heading title as text: links and text templates rendered, entities decoded."""
    return " ".join(html.unescape(to_plain_text(raw)).split())


def mask_comments(wikitext: str) -> str:
    """Return `wikitext` with every comment blanked, so offsets and line breaks stay the same."""
    return COMMENT_PATTERN.sub(
        lambda comment: NON_NEWLINE_PATTERN.sub(MASK_CHARACTER, comment.group()), wikitext
    )


def split_sections(wikitext: str) -> tuple[Section, ...]:
    """Cut `wikitext` on its heading lines: the lead first, then 1 section per heading.

    The lead has an empty title and an empty path. A heading closes every open section of
    its level or deeper, so the path of a section never holds a sibling. A heading line
    inside a comment is not a heading, and a comment after a heading is ignored.
    """
    headings = list(HEADING_PATTERN.finditer(mask_comments(wikitext)))
    lead_end = headings[0].start() if headings else len(wikitext)
    sections = [Section(LEAD_LEVEL, LEAD_TITLE, (), wikitext[:lead_end])]
    open_headings: list[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        level = len(heading.group(1))
        title = normalize_heading(wikitext[heading.start(2) : heading.end(2)])
        while open_headings and open_headings[-1][0] >= level:
            open_headings.pop()
        open_headings.append((level, title))
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(wikitext)
        path = tuple(open_title for _, open_title in open_headings)
        sections.append(Section(level, title, path, wikitext[heading.end() : body_end]))
    return tuple(sections)


def find_sections(sections: Collection[Section], prefix: tuple[str, ...]) -> tuple[Section, ...]:
    """Return the sections whose path starts with `prefix`, in page order."""
    return tuple(section for section in sections if section.path[: len(prefix)] == prefix)


def find_templates(code: Wikicode, names: Collection[str]) -> list[Template]:
    """Return the templates in `code`, nested ones included, whose normalized name is in `names`."""
    wanted = {normalize_template_name(name) for name in names}
    return [
        template
        for template in code.filter_templates(recursive=True)
        if normalize_template_name(str(template.name)) in wanted
    ]


def read_params(template: Template) -> dict[str, str]:
    """Return the parameters of `template` as stripped names to raw values; the last repeat wins."""
    return {str(param.name).strip(): str(param.value).strip() for param in template.params}


def read_categories(wikitext: str) -> frozenset[str]:
    return frozenset(name.strip() for name in CATEGORY_PATTERN.findall(wikitext))


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


def read_language_values(sections: Sequence[Section], template_name: str) -> list[str]:
    """Return the raw value of each language parameter of the language table, in order.

    The template is split on its own top-level pipes: an unclosed `''` in 1 value makes
    `mwparserfromhell` fold the parameters after it into that value.
    """
    values: list[str] = []
    for section in find_sections(sections, LANGUAGES_PATH):
        for template in find_templates(section.parse(), [template_name]):
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


def split_alias_names(raw: str, should_drop_notes: bool) -> list[str]:
    """Return the names and romanizations in 1 language value, in reading order."""
    if should_drop_notes:
        raw = COMMENT_PATTERN.sub("", raw)
    names: list[str] = []
    for raw_line in LINE_BREAK_PATTERN.split(raw):
        # A source note can hold italic titles, as in `(prior to ''[[Pokémon the Series]]'')`,
        # that would otherwise read as romanizations.
        line = TRAILING_NOTE_PATTERN.sub("", raw_line) if should_drop_notes else raw_line
        names.append(ROMANIZATION_PATTERN.sub("", line))
        names.extend(ROMANIZATION_PATTERN.findall(line))
    cleaned = (to_plain_text(name).strip(ALIAS_PADDING) for name in names)
    if should_drop_notes:
        cleaned = (TRAILING_NOTE_PATTERN.sub("", name).strip(ALIAS_PADDING) for name in cleaned)
    return [name for name in cleaned if name]


def read_aliases(
    sections: Sequence[Section],
    infobox: Mapping[str, str],
    english_name: str,
    rules: AliasRules,
) -> tuple[str, ...]:
    """Return the other-language names, then the infobox names, without repeats or the
    English name."""
    raw_values = [
        *read_language_values(sections, rules.template_name),
        *(infobox.get(key, "") for key in rules.infobox_keys),
    ]
    names = (name for raw in raw_values for name in split_alias_names(raw, rules.should_drop_notes))
    return tuple(name for name in dict.fromkeys(names) if name != english_name)
