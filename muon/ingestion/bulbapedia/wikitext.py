"""Wikitext helpers shared by every Bulbapedia page parser.

A page is cut into sections on its heading lines before any template parsing, so an
unbalanced `{{` or table in 1 section cannot hide the headings after it. Templates are
matched on normalized names, and values render to plain text through `TEXT_TEMPLATES`. One
render pass also reports the typed templates it showed and the ones it dropped. This module
knows nothing of page types or of the ontology.
"""

import html
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
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
    parameter is display text and shows as written, and `literal` is never wrapped.
    `layout` replaces all of that: a format string over the values of `params`, where `{0}`
    is the 1st name in `params`, `{1}` the 2nd and so on, and an absent parameter is empty.
    """

    params: tuple[str, ...] = ()
    literal: str = ""
    prefix: str = ""
    suffix: str = ""
    layout: str = ""


FIRST_PARAM = TextRule(params=("1",))
SECOND_PARAM = TextRule(params=("2",))
SECOND_OR_FIRST_PARAM = TextRule(params=("2", "1"))
THIRD_OR_FIRST_PARAM = TextRule(params=("3", "1"))
THIRD_OR_SECOND_PARAM = TextRule(params=("3", "2"))

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
        text = f"{rule.prefix}{value.text}{rule.suffix}" if param == rule.params[-1] else value.text
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
