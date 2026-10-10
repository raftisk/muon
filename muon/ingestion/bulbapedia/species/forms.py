"""Species forms: which infobox forms become nodes, their ids and names, and which
subsection titles belong to which node.

Form candidates come from the infobox only. A candidate becomes a node when the page gives
it its own types or abilities, or names it in a Base stats or Learnset subsection title;
every other candidate is cosmetic and stays on the base node. Ids come from sprite codes,
which infobox images and Evobox `no` fields share, so references agree across pages.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.species.extract import (
    FormCandidateResult,
    FormOutcome,
    NoteKind,
    PageNote,
)
from muon.ingestion.bulbapedia.wikitext import to_plain_text

GENDER_SYMBOLS: Mapping[str, str] = {"♀": " f", "♂": " m"}
DEFAULT_FORM_INDEX = 1
NAME_KEY = "name"
FORM_KEY_PATTERN = re.compile(r"^form(\d+)$")
FORM_NAME_KEY = "form{index}"
IMAGE_KEY = "image"
FORM_IMAGE_KEY = "image{index}"
FORM_TYPE_KEY = "form{index}type1"
FORM_ABILITY_KEYS = ("ability{index}-1", "ability{index}-2")
MEGA_ABILITY_KEYS = ("abilitym", "abilitym2")
MEGA_PATTERN = re.compile(r"\bMega\b")
GIGANTAMAX_MARKER = "Gigantamax"
PRIMAL_FORM_NAME = "Primal Reversion"
LEADING_DIGITS = re.compile(r"^\d+")
IMAGE_EXTENSION = re.compile(r"\.(?:png|jpe?g|gif|svg)$", re.IGNORECASE)
CODE_PADDING = "- "
REGIONAL_ADJECTIVES: Mapping[str, str] = {
    "Alolan": "Alola",
    "Galarian": "Galar",
    "Hisuian": "Hisui",
    "Paldean": "Paldea",
}
ADJECTIVE_ALTERNATIVES = "|".join(REGIONAL_ADJECTIVES)
REGIONAL_NAME_PATTERN = re.compile(rf"^({ADJECTIVE_ALTERNATIVES}) Form(?: \((.+)\))?$")
COMBINED_NAME_PATTERN = re.compile(rf"^({ADJECTIVE_ALTERNATIVES}) Form/(.+)$")
TITLE_LIST_SEPARATOR = re.compile(r", and |,? and |, |/")
FORM_SUFFIX = " Form"
STATS_GROUP = "Base stats"
LEARNSET_GROUP = "Learnset"


class OverrideError(Exception):
    """An entry of `FORM_OVERRIDES` matches nothing on the page or in the pinned titles."""


@dataclass(frozen=True)
class FormOverride:
    """Replaces 1 infobox form's id, name or node outcome. A field left `None` keeps the rule."""

    form_id: str | None = None
    name: str | None = None
    is_node: bool | None = None


@dataclass(frozen=True)
class PageOverride:
    """The overrides of 1 page: forms by infobox index, and the form ids a subsection maps to."""

    forms: Mapping[int, FormOverride] = field(default_factory=dict)
    subsections: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


# Starts empty. An entry is added only when the full run shows a page the rules cannot read
# and no rule change fixes it.
FORM_OVERRIDES: Mapping[str, PageOverride] = {}
NO_OVERRIDE = PageOverride()


@dataclass(frozen=True)
class FormCandidate:
    """1 form named in the infobox. Candidate 1 is the default form."""

    index: int
    infobox_name: str
    image: str | None


@dataclass(frozen=True)
class ResolvedForm:
    id: str
    name: str
    index: int
    infobox_name: str
    region: str | None
    is_default: bool


@dataclass(frozen=True)
class FormSet:
    """The nodes of a page, what became of every candidate and where each subsection goes.

    `forms` lists the base node first. `subsections` maps a Base stats or Learnset
    subsection title to the ids of the nodes it covers.
    """

    forms: tuple[ResolvedForm, ...]
    candidates: tuple[FormCandidateResult, ...]
    subsections: Mapping[str, tuple[str, ...]]
    notes: tuple[PageNote, ...]


@dataclass(frozen=True)
class FormInputs:
    page_title: str
    infobox: Mapping[str, str]
    stats_titles: tuple[str, ...]
    learnset_titles: tuple[str, ...]


def build_species_id(name: str) -> str:
    """Return the node id of a species name: ♀ and ♂ become " f" and " m" before slugging."""
    spelled = name
    for symbol, suffix in GENDER_SYMBOLS.items():
        spelled = spelled.replace(symbol, suffix)
    return build_node_id(spelled)


def flatten_text(raw: str) -> str:
    """Return `raw` as 1 line of plain text."""
    return " ".join(to_plain_text(raw).split())


def read_sprite_code(file_or_number: str, base_name: str) -> str:
    """Return the form code of a sprite file or Evobox `no` value, `""` for the default form.

    The leading dex digits, the base name and an image extension are removed, so
    `0050Diglett-Alola.png` and `0051Dugtrio-Alola` give `Alola`, and `0050` gives `""`.
    """
    stem = IMAGE_EXTENSION.sub("", file_or_number.strip())
    rest = LEADING_DIGITS.sub("", stem)
    if rest.lower().startswith(base_name.lower()):
        rest = rest[len(base_name) :]
    return rest.strip(CODE_PADDING)


def read_region(infobox_name: str) -> str | None:
    """Return the Region name of an `{Adjective} Form` name, optionally with a parenthetical."""
    match = REGIONAL_NAME_PATTERN.match(infobox_name)
    return REGIONAL_ADJECTIVES[match.group(1)] if match else None


def build_form_name(base_name: str, infobox_name: str) -> str:
    """Return the node name of a non-default form: "Alolan Diglett", "Darmanitan (Zen Mode)"."""
    regional = REGIONAL_NAME_PATTERN.match(infobox_name)
    if regional:
        adjective, qualifier = regional.groups()
        name = f"{adjective} {base_name}"
        return f"{name} ({qualifier})" if qualifier else name
    combined = COMBINED_NAME_PATTERN.match(infobox_name)
    if combined:
        return f"{combined.group(1)} {base_name} ({combined.group(2)})"
    if infobox_name == PRIMAL_FORM_NAME:
        return f"Primal {base_name}"
    if base_name in infobox_name:
        return infobox_name
    return f"{base_name} ({infobox_name})"


def is_mega_form(infobox_name: str) -> bool:
    return MEGA_PATTERN.search(infobox_name) is not None


def read_form_candidates(infobox: Mapping[str, str]) -> tuple[FormCandidate, ...]:
    """Return the default candidate and every `formN` that has a name, in index order."""
    indices = {DEFAULT_FORM_INDEX}
    for key, value in infobox.items():
        match = FORM_KEY_PATTERN.match(key)
        if match and flatten_text(value):
            indices.add(int(match.group(1)))
    return tuple(build_candidate(infobox, index) for index in sorted(indices))


def build_candidate(infobox: Mapping[str, str], index: int) -> FormCandidate:
    name = flatten_text(infobox.get(FORM_NAME_KEY.format(index=index), ""))
    image_keys = [FORM_IMAGE_KEY.format(index=index)]
    if index == DEFAULT_FORM_INDEX:
        image_keys.insert(0, IMAGE_KEY)
    images = (flatten_text(infobox.get(key, "")) for key in image_keys)
    return FormCandidate(index, name, next((image for image in images if image), None))


def check_override_titles(pinned_titles: Collection[str]) -> None:
    """Raise `OverrideError` when `FORM_OVERRIDES` names a page that is not pinned."""
    unknown = sorted(set(FORM_OVERRIDES) - set(pinned_titles))
    if unknown:
        raise OverrideError(f"FORM_OVERRIDES names pages that are not pinned: {unknown}")


def check_override(inputs: FormInputs, override: PageOverride, indices: Collection[int]) -> None:
    stale_indices = sorted(set(override.forms) - set(indices))
    if stale_indices:
        raise OverrideError(
            f"FORM_OVERRIDES for {inputs.page_title!r} names form indexes the infobox lacks: "
            f"{stale_indices}"
        )
    titles = {*inputs.stats_titles, *inputs.learnset_titles}
    stale_titles = sorted(set(override.subsections) - titles)
    if stale_titles:
        raise OverrideError(
            f"FORM_OVERRIDES for {inputs.page_title!r} names subsections the page lacks: "
            f"{stale_titles}"
        )


def build_resolved_form(
    candidate: FormCandidate, base_name: str, override: FormOverride | None
) -> ResolvedForm:
    if candidate.index == DEFAULT_FORM_INDEX:
        return ResolvedForm(
            build_species_id(base_name),
            base_name,
            candidate.index,
            candidate.infobox_name,
            None,
            True,
        )
    form_id = override.form_id if override and override.form_id else None
    name = override.name if override and override.name else None
    return ResolvedForm(
        form_id or build_form_id(candidate, base_name),
        name or build_form_name(base_name, candidate.infobox_name),
        candidate.index,
        candidate.infobox_name,
        read_region(candidate.infobox_name),
        False,
    )


def build_form_id(candidate: FormCandidate, base_name: str) -> str:
    """Return `{base id}-{code}`: the slug of the sprite code, else of the infobox name."""
    code = read_sprite_code(candidate.image, base_name) if candidate.image else ""
    suffix = build_node_id(code) or build_node_id(candidate.infobox_name)
    return f"{build_species_id(base_name)}-{suffix}"


def format_form_names(form: ResolvedForm, base_name: str) -> frozenset[str]:
    """Return the titles that name `form`: its node name, its infobox name with or without
    " Form", "{base} ({infobox name})" and "{infobox name} {base}"."""
    names = {form.name}
    if form.infobox_name:
        short = form.infobox_name.removesuffix(FORM_SUFFIX)
        names.update(
            {
                form.infobox_name,
                short,
                f"{short} {base_name}",
                f"{base_name} ({form.infobox_name})",
                f"{base_name} ({short})",
            }
        )
    return frozenset(names)


def match_exact(title: str, forms: tuple[ResolvedForm, ...], base_name: str) -> tuple[str, ...]:
    return tuple(form.id for form in forms if title in format_form_names(form, base_name))


def match_list(title: str, forms: tuple[ResolvedForm, ...], base_name: str) -> tuple[str, ...]:
    """Map "Heat, Wash, and Mow Rotom" to the forms each part names, or to nothing."""
    parts = TITLE_LIST_SEPARATOR.split(title.removesuffix(f" {base_name}"))
    if len(parts) < 2:
        return ()
    ids: list[str] = []
    for part in parts:
        spellings = (part, f"{part} {base_name}", f"{base_name} {part}", f"{base_name} ({part})")
        found = [form.id for form in forms if format_form_names(form, base_name) & set(spellings)]
        if not found:
            return ()
        ids.extend(found)
    return tuple(dict.fromkeys(ids))


def match_regional(title: str, forms: tuple[ResolvedForm, ...], base_name: str) -> tuple[str, ...]:
    """Map "Paldean Tauros" to every form with that region."""
    for adjective, region in REGIONAL_ADJECTIVES.items():
        if title == f"{adjective} {base_name}":
            return tuple(form.id for form in forms if form.region == region)
    return ()


def match_subsection(title: str, forms: tuple[ResolvedForm, ...]) -> tuple[str, ...]:
    """Return the ids of the forms a Base stats or Learnset subsection title names.

    The first rule that matches wins: a node name, an infobox name or "{base} ({infobox
    name})"; a list of such names; a regional title shared by several forms. `forms` holds
    the default form, whose name is the base name.
    """
    base_name = next(form.name for form in forms if form.is_default)
    for rule in (match_exact, match_list, match_regional):
        ids = rule(title, forms, base_name)
        if ids:
            return ids
    return ()


def has_value(infobox: Mapping[str, str], key: str) -> bool:
    return bool(flatten_text(infobox.get(key, "")))


def has_own_data(infobox: Mapping[str, str], index: int, mega_rank: int | None) -> bool:
    """Whether the infobox gives form `index` its own types or abilities.

    `mega_rank` is the position of the form among the Mega forms, which take `abilitym`
    and `abilitym2` in infobox order, or `None` for a form that is not Mega.
    """
    keys = [key.format(index=index) for key in (FORM_TYPE_KEY, *FORM_ABILITY_KEYS)]
    if mega_rank is not None and mega_rank < len(MEGA_ABILITY_KEYS):
        keys.append(MEGA_ABILITY_KEYS[mega_rank])
    return any(has_value(infobox, key) for key in keys)


class NamedCandidate(Protocol):
    """A `FormCandidate` or a `FormCandidateResult`."""

    @property
    def index(self) -> int: ...

    @property
    def infobox_name(self) -> str: ...


def rank_mega_forms(candidates: Iterable[NamedCandidate]) -> Mapping[int, int]:
    """Return the position of each non-default Mega candidate among the Mega forms."""
    mega = [
        candidate.index
        for candidate in candidates
        if candidate.index != DEFAULT_FORM_INDEX and is_mega_form(candidate.infobox_name)
    ]
    return {index: rank for rank, index in enumerate(mega)}


def is_gigantamax(candidate: FormCandidate) -> bool:
    return GIGANTAMAX_MARKER in candidate.infobox_name


@dataclass(frozen=True)
class Decision:
    """The outcome of 1 candidate, and its form when it is a node."""

    result: FormCandidateResult
    form: ResolvedForm | None


def decide_candidate(
    candidate: FormCandidate,
    draft: ResolvedForm,
    is_data_backed: bool,
    override: FormOverride | None,
) -> Decision:
    """Apply the node rule, then the override. Gigantamax never becomes a node by rule."""
    if draft.is_default:
        outcome, is_node = FormOutcome.NODE, True
    elif override is not None:
        rule_says_node = is_data_backed and not is_gigantamax(candidate)
        is_node = rule_says_node if override.is_node is None else override.is_node
        outcome = FormOutcome.OVERRIDDEN
    elif is_gigantamax(candidate):
        outcome, is_node = FormOutcome.SKIPPED, False
    elif is_data_backed:
        outcome, is_node = FormOutcome.NODE, True
    else:
        outcome, is_node = FormOutcome.COSMETIC, False
    result = FormCandidateResult(
        index=candidate.index,
        infobox_name=candidate.infobox_name,
        outcome=outcome,
        form_id=draft.id if is_node else None,
    )
    return Decision(result, draft if is_node else None)


def match_titles(
    titles: Collection[str], matchable: tuple[ResolvedForm, ...], override: PageOverride
) -> dict[str, tuple[str, ...]]:
    """Map each title to the form ids it names; an override replaces the rules for its title."""
    return {
        title: override.subsections.get(title) or match_subsection(title, matchable)
        for title in titles
    }


def build_override_notes(
    override: PageOverride, results: Iterable[FormCandidateResult]
) -> tuple[PageNote, ...]:
    by_index = {result.index: result for result in results}
    form_notes = [
        PageNote(
            kind=NoteKind.OVERRIDE_APPLIED,
            section_path=by_index[index].infobox_name or FORM_NAME_KEY.format(index=index),
            detail=f"infobox form {index} overridden",
        )
        for index in sorted(override.forms)
    ]
    title_notes = [
        PageNote(
            kind=NoteKind.OVERRIDE_APPLIED,
            section_path=title,
            detail=f"subsection mapped to {list(ids)}",
        )
        for title, ids in sorted(override.subsections.items())
    ]
    return (*form_notes, *title_notes)


def build_unmatched_notes(
    group: str, titles: Collection[str], matched: Collection[str]
) -> list[PageNote]:
    return [
        PageNote(
            kind=NoteKind.UNMATCHED_SUBSECTION,
            section_path=title,
            detail=f"no form matches this {group} subsection",
        )
        for title in dict.fromkeys(titles)
        if title not in matched
    ]


def resolve_forms(inputs: FormInputs) -> FormSet:
    """Decide the nodes of a page and map its subsection titles onto them.

    Raises `OverrideError` when an override names a form index, subsection or form id the
    page lacks.
    """
    override = FORM_OVERRIDES.get(inputs.page_title, NO_OVERRIDE)
    candidates = read_form_candidates(inputs.infobox)
    check_override(inputs, override, [candidate.index for candidate in candidates])
    base_name = flatten_text(inputs.infobox[NAME_KEY])
    drafts = {
        candidate.index: build_resolved_form(
            candidate, base_name, override.forms.get(candidate.index)
        )
        for candidate in candidates
    }
    matchable = tuple(
        drafts[candidate.index]
        for candidate in candidates
        if not is_gigantamax(candidate) or candidate.index in override.forms
    )
    titles = dict.fromkeys([*inputs.stats_titles, *inputs.learnset_titles])
    matches = match_titles(titles, matchable, override)
    named = {form_id for ids in matches.values() for form_id in ids}
    ranks = rank_mega_forms(candidates)
    decisions = [
        decide_candidate(
            candidate,
            drafts[candidate.index],
            has_own_data(inputs.infobox, candidate.index, ranks.get(candidate.index))
            or drafts[candidate.index].id in named,
            override.forms.get(candidate.index),
        )
        for candidate in candidates
    ]
    return assemble_form_set(inputs, override, decisions, matches)


def assemble_form_set(
    inputs: FormInputs,
    override: PageOverride,
    decisions: list[Decision],
    matches: Mapping[str, tuple[str, ...]],
) -> FormSet:
    forms = tuple(decision.form for decision in decisions if decision.form is not None)
    node_ids = {form.id for form in forms}
    for title, ids in override.subsections.items():
        stale = [form_id for form_id in ids if form_id not in node_ids]
        if stale:
            raise OverrideError(
                f"FORM_OVERRIDES for {inputs.page_title!r} maps {title!r} to unknown forms: {stale}"
            )
    subsections = {
        title: kept
        for title, ids in matches.items()
        if (kept := tuple(form_id for form_id in ids if form_id in node_ids))
    }
    results = tuple(decision.result for decision in decisions)
    notes = (
        *build_override_notes(override, results),
        *build_unmatched_notes(STATS_GROUP, inputs.stats_titles, subsections),
        *build_unmatched_notes(LEARNSET_GROUP, inputs.learnset_titles, subsections),
    )
    return FormSet(forms, results, subsections, notes)
