"""Scarlet/Violet learnset rows of a species page.

A row is a Generation IX learnlist template. Every row template carries its game in its
name, so the rows of other games (`levelZA`, `tm8`) never match and no table header has to
be tracked. A row belongs to the form of the subsection that holds it, or to the base
node when the table has no form subsections.
"""

import re
from collections.abc import Mapping, Sequence

from mwparserfromhell.nodes import Template

from muon.ingestion.bulbapedia.species.extract import (
    LearnMethod,
    LearnsetRow,
    NoteKind,
    PageNote,
)
from muon.ingestion.bulbapedia.species.forms import FormSet
from muon.ingestion.bulbapedia.species.page import PATH_SEPARATOR, PageParseError
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    find_sections,
    find_templates,
    normalize_template_name,
    to_plain_text,
)

LEARNSET_PATH = ("Game data", "Learnset")
FORM_SUBSECTION_DEPTH = len(LEARNSET_PATH) + 2
ROW_TEMPLATES: Mapping[str, LearnMethod] = {
    "learnlist/level9": LearnMethod.LEVEL,
    "learnlist/tm9": LearnMethod.MACHINE,
    "learnlist/breed9": LearnMethod.EGG,
    "learnlist/event9": LearnMethod.SPECIAL,
}
LEVEL_MARKERS: Mapping[str, LearnMethod] = {
    "Evo.": LearnMethod.EVOLUTION,
    "Rem.": LearnMethod.REMINDER,
}
FIRST_PARAM = 1
MOVE_PARAM = 2
MACHINE_PATTERN = re.compile(r"^TM\d+$")


def has_row_templates(section: Section) -> bool:
    return bool(find_templates(section.parse(), ROW_TEMPLATES))


def list_learnset_titles(sections: Sequence[Section]) -> tuple[str, ...]:
    """Return the titles of the form subsections that hold Scarlet/Violet rows.

    A form subsection sits 1 level under a table such as "By leveling up". The titles repeat
    once per table.
    """
    return tuple(
        section.title
        for section in find_sections(sections, LEARNSET_PATH)
        if len(section.path) == FORM_SUBSECTION_DEPTH and has_row_templates(section)
    )


def locate_learnset_forms(section: Section, forms: FormSet) -> tuple[str, ...]:
    """Return the node ids a learnset section's rows belong to; none when its title is unmatched."""
    if len(section.path) == FORM_SUBSECTION_DEPTH:
        return forms.subsections.get(section.title, ())
    if len(section.path) < FORM_SUBSECTION_DEPTH:
        return (forms.forms[0].id,)
    return ()


def read_level(raw: str) -> tuple[LearnMethod, int | None] | None:
    """Read the level cell of a `level9` row: a number, or a marker such as `Evo.`."""
    text = to_plain_text(raw)
    if text.isdigit():
        return LearnMethod.LEVEL, int(text)
    marker = LEVEL_MARKERS.get(text)
    return (marker, None) if marker else None


def read_param(template: Template, index: int) -> str:
    return to_plain_text(str(template.get(index).value)) if template.has(index) else ""


def read_row_fields(
    template: Template, method: LearnMethod, page_title: str
) -> tuple[LearnMethod, int | None, str | None]:
    """Return the method, level and TM number of 1 row; an unreadable cell fails the page."""
    path = PATH_SEPARATOR.join(LEARNSET_PATH)
    first = read_param(template, FIRST_PARAM)
    if method is LearnMethod.LEVEL:
        read = read_level(first)
        if read is None:
            raise PageParseError(page_title, path, "level", first)
        return read[0], read[1], None
    if method is LearnMethod.MACHINE:
        if not MACHINE_PATTERN.match(first):
            raise PageParseError(page_title, path, "machine", first)
        return method, None, first
    return method, None, None


def read_row(template: Template, form_id: str, page_title: str) -> LearnsetRow:
    path = PATH_SEPARATOR.join(LEARNSET_PATH)
    name = normalize_template_name(str(template.name))
    method, level, machine = read_row_fields(template, ROW_TEMPLATES[name], page_title)
    move = read_param(template, MOVE_PARAM)
    if not move:
        raise PageParseError(page_title, path, "move", "")
    return LearnsetRow(form_id=form_id, move=move, method=method, level=level, machine=machine)


def read_learnset(
    sections: Sequence[Section], forms: FormSet, page_title: str
) -> tuple[tuple[LearnsetRow, ...], tuple[PageNote, ...]]:
    """Read the Scarlet/Violet rows of every node.

    Rows under a subsection title that names no form are dropped; `FormSet` already notes
    the title. A page with no Scarlet/Violet row at all gets a `missing_learnset` note.
    """
    rows: list[LearnsetRow] = []
    has_rows = False
    for section in find_sections(sections, LEARNSET_PATH):
        form_ids = locate_learnset_forms(section, forms)
        templates = find_templates(section.parse(), ROW_TEMPLATES)
        has_rows = has_rows or bool(templates)
        rows.extend(
            read_row(template, form_id, page_title)
            for template in templates
            for form_id in form_ids
        )
    notes: tuple[PageNote, ...] = ()
    if not has_rows:
        notes = (
            PageNote(
                kind=NoteKind.MISSING_LEARNSET,
                section_path=PATH_SEPARATOR.join(LEARNSET_PATH),
                detail="no Scarlet/Violet learnlist row",
            ),
        )
    return tuple(dict.fromkeys(rows)), notes
