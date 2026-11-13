"""Text chunks of a species page: prose sections and Pokédex rows become `ChunkFact`s.

Map:
- Aspect: the path of a section decides whether it is read and as what (`read_aspect`).
- Prose: Biology, Origin and Trivia sections follow the shared rules of `chunking.py`
  (`read_species_prose`).
- Pokédex: each `Dex/Entry` row gives a text with the main-series versions that show it.
  `Dex/Gen/N` returns the subject to the base form and `Dex/Form` switches it. Rows with
  the same subject and text merge into 1 chunk (`read_dex_rows`, `merge_dex_rows`).
- Subjects: a chunk describes the base form unless an explicit marker names another form
  of the page (`match_marker`, `pick_biology_subject`). Nothing infers a form from prose.

The facts hold Bulbapedia terms only. `batches.py` maps them onto ids and IRIs.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mwparserfromhell
from mwparserfromhell.nodes import Template

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.chunking import (
    MENTION_TEMPLATES,
    ChunkAspect,
    ChunkFact,
    MentionRef,
    normalize_chunk_text,
    read_prose_section,
)
from muon.ingestion.bulbapedia.species.extract import NoteKind, PageNote
from muon.ingestion.bulbapedia.species.forms import ResolvedForm, format_form_names
from muon.ingestion.bulbapedia.wikitext import (
    PATH_SEPARATOR,
    Section,
    normalize_template_name,
    read_plain_text,
    to_plain_text,
)

BIOLOGY_TITLE = "biology"
TRIVIA_TITLE = "trivia"
FORMS_TITLE = "forms"
POKEDEX_PATH = ("game data", "pokédex entries")
ORIGIN_PATHS = (("trivia", "origin"), ("trivia", "concept and development"))
EXCLUDED_PATHS = (("biology", "evolution"), ("trivia", "merchandise"))
DEX_ENTRY_TEMPLATES = frozenset({"dex/entry1", "dex/entry2", "dex/entry3"})
DEX_GEN_PREFIX = "dex/gen/"
DEX_FORM_TEMPLATE = "dex/form"
ENTRY_PARAM = "entry"
FORM_LABEL_PARAM = "1"
VERSION_PARAMS = ("v", "v2", "v3")
# Versions of games outside the main series: their rows give no chunk and no note.
SIDE_GAME_VERSIONS = frozenset(
    {"stadium", "stadium 2", "colosseum", "xd", "battle revolution", "pokopia", "champions"}
)


@dataclass(frozen=True)
class DexRow:
    """1 Pokédex entry row: its text, mentions and the main-series versions that show it."""

    subject_id: str
    section_path: str
    text: str
    mentions: tuple[MentionRef, ...]
    dropped: tuple[str, ...]
    versions: tuple[str, ...]


@dataclass(frozen=True)
class DexSubject:
    """Who the entries describe and under which section path, until the next `Dex/Gen`."""

    subject_id: str
    section_path: str


@dataclass(frozen=True)
class ChunkReading:
    chunks: tuple[ChunkFact, ...]
    notes: tuple[PageNote, ...]
    dropped_templates: Mapping[str, int]


def has_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def read_aspect(path: tuple[str, ...]) -> ChunkAspect | None:
    """Return the aspect of the section at `path`, or `None` when it gives no chunk.

    Titles compare in lowercase. The lead (empty path) and every section outside Biology,
    Trivia and the Pokédex entries give nothing, as do Biology > Evolution and
    Trivia > Merchandise with their subsections.
    """
    titles = tuple(title.lower() for title in path)
    if any(has_prefix(titles, excluded) for excluded in EXCLUDED_PATHS):
        return None
    if any(has_prefix(titles, origin) for origin in ORIGIN_PATHS):
        return ChunkAspect.ORIGIN
    if has_prefix(titles, POKEDEX_PATH):
        return ChunkAspect.POKEDEX
    if titles[:1] == (BIOLOGY_TITLE,):
        return ChunkAspect.BIOLOGY
    if titles[:1] == (TRIVIA_TITLE,):
        return ChunkAspect.TRIVIA
    return None


def match_marker(marker: str, forms: Sequence[ResolvedForm]) -> str | None:
    """Return the id of the form that `marker` names, or `None`.

    Comparison is on `build_node_id()` slugs. The marker first has to equal a name of
    exactly 1 form (`format_form_names`). When it equals none, it has to sit as a whole
    hyphen-separated run in the node name of exactly 1 form. Several matches name nothing.
    """
    marker_slug = build_node_id(marker)
    if not marker_slug:
        return None
    base_name = next(form.name for form in forms if form.is_default)
    equal = [
        form.id
        for form in forms
        if marker_slug in {build_node_id(name) for name in format_form_names(form, base_name)}
    ]
    if equal:
        return equal[0] if len(equal) == 1 else None
    run = f"-{marker_slug}-"
    contained = [form.id for form in forms if run in f"-{build_node_id(form.name)}-"]
    return contained[0] if len(contained) == 1 else None


def pick_biology_subject(
    path: tuple[str, ...], forms: Sequence[ResolvedForm]
) -> tuple[str, PageNote | None]:
    """Return the form id that the Biology section at `path` describes, with a note when a
    title below Biology looks like a form label and matches none.

    The markers are the titles below Biology, deepest first, without "Forms".
    """
    default_id = forms[0].id
    markers = [title for title in reversed(path[1:]) if title.lower() != FORMS_TITLE]
    if not markers:
        return default_id, None
    for marker in markers:
        form_id = match_marker(marker, forms)
        if form_id is not None:
            return form_id, None
    note = PageNote(
        kind=NoteKind.UNMATCHED_FORM_LABEL,
        section_path=PATH_SEPARATOR.join(path),
        detail=f"{markers[0]} matches no form node",
    )
    return default_id, note


def read_species_prose(
    section: Section, aspect: ChunkAspect, forms: Sequence[ResolvedForm]
) -> tuple[list[ChunkFact], list[PageNote], list[str]]:
    """Read the chunks, notes and dropped template names of a Biology, Origin or Trivia section."""
    notes: list[PageNote] = []
    subject_id = forms[0].id
    if aspect is ChunkAspect.BIOLOGY:
        subject_id, note = pick_biology_subject(section.path, forms)
        if note is not None:
            notes.append(note)
    reading = read_prose_section(section, aspect, subject_id)
    return list(reading.chunks), notes, list(reading.dropped)


def read_dex_versions(template: Template) -> tuple[str, ...]:
    """Return the main-series versions of an entry row, in the order the row lists them."""
    names = (
        to_plain_text(str(template.get(param).value))
        for param in VERSION_PARAMS
        if template.has(param, ignore_empty=True)
    )
    return tuple(name for name in names if name.lower() not in SIDE_GAME_VERSIONS)


def read_dex_row(template: Template, subject: DexSubject) -> DexRow | None:
    """Return the row of a `Dex/Entry` template, or `None` when it shows no chunk.

    A row gives nothing when it has no entry text or only side-game versions.
    """
    versions = read_dex_versions(template)
    if not versions or not template.has(ENTRY_PARAM, ignore_empty=True):
        return None
    rendered = read_plain_text(str(template.get(ENTRY_PARAM).value), MENTION_TEMPLATES)
    text = normalize_chunk_text(rendered.text)
    if not text:
        return None
    return DexRow(
        subject_id=subject.subject_id,
        section_path=subject.section_path,
        text=text,
        mentions=tuple(
            MentionRef(template=item.name, target=item.target) for item in rendered.captured
        ),
        dropped=rendered.dropped,
        versions=versions,
    )


def switch_dex_form(
    template: Template, section_path: str, forms: Sequence[ResolvedForm]
) -> tuple[DexSubject, PageNote | None]:
    """Return the subject that a `Dex/Form` template starts, with a note when its label
    names no form. The label stays in the section path either way."""
    default = DexSubject(forms[0].id, section_path)
    if not template.has(FORM_LABEL_PARAM, ignore_empty=True):
        return default, None
    label = to_plain_text(str(template.get(FORM_LABEL_PARAM).value))
    labelled_path = f"{section_path}{PATH_SEPARATOR}{label}"
    form_id = match_marker(label, forms)
    if form_id is not None:
        return DexSubject(form_id, labelled_path), None
    note = PageNote(
        kind=NoteKind.UNMATCHED_FORM_LABEL,
        section_path=labelled_path,
        detail=f"{label} matches no form node",
    )
    return DexSubject(forms[0].id, labelled_path), note


def read_dex_rows(
    section: Section, forms: Sequence[ResolvedForm]
) -> tuple[tuple[DexRow, ...], tuple[PageNote, ...]]:
    """Read the entry rows of a Pokédex section, walking its templates in page order.

    `Dex/Gen/N` returns the subject to the base form, and `Dex/Form` switches it.
    """
    section_path = PATH_SEPARATOR.join(section.path)
    default = DexSubject(forms[0].id, section_path)
    subject = default
    rows: list[DexRow] = []
    notes: list[PageNote] = []
    for template in mwparserfromhell.parse(section.text).filter_templates(recursive=True):
        name = normalize_template_name(str(template.name))
        if name.startswith(DEX_GEN_PREFIX):
            subject = default
        elif name == DEX_FORM_TEMPLATE:
            subject, note = switch_dex_form(template, section_path, forms)
            if note is not None:
                notes.append(note)
        elif name in DEX_ENTRY_TEMPLATES:
            row = read_dex_row(template, subject)
            if row is not None:
                rows.append(row)
    return tuple(rows), tuple(notes)


def merge_dex_rows(rows: Sequence[DexRow]) -> tuple[ChunkFact, ...]:
    """Return 1 chunk per (subject, text), in first-seen order.

    Its versions are the ordered union over the rows, and its section and mentions are
    those of the first row.
    """
    first_rows: dict[tuple[str, str], DexRow] = {}
    versions: dict[tuple[str, str], dict[str, None]] = {}
    for row in rows:
        key = (row.subject_id, row.text)
        first_rows.setdefault(key, row)
        versions.setdefault(key, {}).update(dict.fromkeys(row.versions))
    return tuple(
        ChunkFact(
            subject_id=row.subject_id,
            aspect=ChunkAspect.POKEDEX,
            section_path=row.section_path,
            text=row.text,
            mentions=row.mentions,
            versions=tuple(versions[key]),
        )
        for key, row in first_rows.items()
    )


def read_chunks(sections: Sequence[Section], forms: Sequence[ResolvedForm]) -> ChunkReading:
    """Read the chunks of a page. `forms` lists the base form first.

    Prose chunks come in page order, then the Pokédex chunks of all Pokédex sections.
    """
    chunks: list[ChunkFact] = []
    notes: list[PageNote] = []
    dex_rows: list[DexRow] = []
    dropped: Counter[str] = Counter()
    for section in sections:
        aspect = read_aspect(section.path)
        if aspect is None:
            continue
        if aspect is ChunkAspect.POKEDEX:
            rows, dex_notes = read_dex_rows(section, forms)
            dex_rows.extend(rows)
            notes.extend(dex_notes)
            dropped.update(name for row in rows for name in row.dropped)
            continue
        section_chunks, section_notes, section_dropped = read_species_prose(section, aspect, forms)
        chunks.extend(section_chunks)
        notes.extend(section_notes)
        dropped.update(section_dropped)
    chunks.extend(merge_dex_rows(dex_rows))
    return ChunkReading(tuple(chunks), tuple(notes), dict(dropped))
