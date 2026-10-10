"""Evolutions of a species page: Evobox stages and the rules that read their methods.

An Evobox lists stages (`no1`, `name1`, ...) and the method from each stage to the next
(`evo1`, or `evo1a` and `evo1b` for a branch). Each edge becomes an `EvolutionFact`: its
method is read by `METHOD_RULES` when exactly 1 Scarlet/Violet alternative remains and
matches a rule, and kept as `condition` text plus a residue item otherwise.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Template
from mwparserfromhell.wikicode import Wikicode

from muon.ingestion.bulbapedia.species.extract import (
    EvolutionFact,
    Requirement,
    RequirementRole,
    Residue,
    ResidueKind,
    SpeciesRef,
)
from muon.ingestion.bulbapedia.species.forms import flatten_text, read_sprite_code
from muon.ingestion.bulbapedia.wikitext import (
    Section,
    find_sections,
    find_templates,
    normalize_template_name,
    read_params,
    to_plain_text,
)

EVOLUTION_PATH = ("Game data", "Evolution data")
PATH_SEPARATOR = " > "
EVOBOX_PREFIX = "evobox"
STAGE_TEMPLATES = (
    "evobox-1",
    "evobox-2",
    "evobox-3",
    "evobox/1branch2",
    "evobox/1branch3",
    "evobox/2branch1",
    "evobox/2branch2",
)
STAGE_NUMBER_PATTERN = re.compile(r"^no(\d+)([a-z]?)$")
METHOD_PARAM_PATTERN = re.compile(r"^evo(\d+)([a-z]?)$")
STAGE_NAME_KEY = "name{stage}{variant}"
STAGE_NUMBER_KEY = "no{stage}{variant}"
SINGLE_VARIANT = ""

ALTERNATIVE_SEPARATOR_PATTERN = re.compile(
    r"<big><big>→</big></big>|<br\s*/?>\s*or\s*<br\s*/?>", re.IGNORECASE
)
OTHER_GAME_PATTERN = re.compile(r"\{\{gameicon\|([^}|]+)\}\}\s*only", re.IGNORECASE)
SNAPSHOT_GAME_ICONS = frozenset({"sc", "v", "sv"})
GAME_NOTE_PATTERN = re.compile(r"\([^()]*\{\{gameicon[^()]*\)", re.IGNORECASE)
SMALL_TAG_PATTERN = re.compile(r"</?small\s*>", re.IGNORECASE)
REGION_LIMIT_PATTERN = re.compile(r"\bin (?:Hisui|Alola)\b")
REGION_QUALIFIER_PATTERN = re.compile(r"\s*\boutside of [A-Z][a-z]+\b")
DECORATION_TOKENS = frozenset({"+", "→"})

BAG_TEMPLATE = "bag/s"
LINK_TEMPLATE = "color2"
MOVE_TEMPLATES = ("mcolor",)
SPECIES_TEMPLATES = ("pcolor", "p")
TYPE_TEMPLATES = ("type", "tcolor")
LINK_TARGET_PARAM = 2
FIRST_PARAM = 1
NON_ITEM_LINKS = frozenset({"trade", "level", "time", "gender", "friendship", "hp", "party"})

TIMES_OF_DAY: Mapping[str, str] = {
    "day": "day",
    "night": "night",
    "evening": "dusk",
    "dusk": "dusk",
}
GENDERS: Mapping[str, str] = {"male": "male", "female": "female"}
TIME_WORDS = "|".join(TIMES_OF_DAY)
GENDER_WORDS = "|".join(GENDERS)
TIME_CLAUSE = rf"(?:at|during the) (?P<time>{TIME_WORDS})"
LEVEL_OR_UP = r"Level (?:(?P<level>\d+)|up)"


@dataclass(frozen=True)
class EvoboxEdge:
    """1 evolution arrow of an Evobox. `method` is the raw wikitext of its `evo` parameter."""

    source: SpeciesRef
    target: SpeciesRef
    method: str


def read_stages(params: Mapping[str, str]) -> dict[int, dict[str, SpeciesRef]]:
    """Return the species of each stage by variant letter; a single species has variant `""`."""
    stages: dict[int, dict[str, SpeciesRef]] = {}
    for key in params:
        match = STAGE_NUMBER_PATTERN.match(key)
        if match is None:
            continue
        stage, variant = int(match.group(1)), match.group(2)
        name = flatten_text(params.get(STAGE_NAME_KEY.format(stage=stage, variant=variant), ""))
        if name:
            code = read_sprite_code(flatten_text(params[key]), name)
            stages.setdefault(stage, {})[variant] = SpeciesRef(name=name, code=code)
    return stages


def pick_variant(variants: Mapping[str, SpeciesRef], variant: str) -> SpeciesRef | None:
    """Pick the species of variant `variant`, else the stage's single species."""
    return variants.get(variant) or variants.get(SINGLE_VARIANT)


def read_evobox_edges(template: Template) -> tuple[EvoboxEdge, ...]:
    """Read the arrows of 1 Evobox stage template.

    `evo{N}{x}` runs from stage N to stage N+1. Variant `x` selects the same letter on a
    stage that branches and the single species on one that does not.
    """
    params = read_params(template)
    stages = read_stages(params)
    edges: list[EvoboxEdge] = []
    for key in sorted(params):
        match = METHOD_PARAM_PATTERN.match(key)
        if match is None or not params[key].strip():
            continue
        stage, variant = int(match.group(1)), match.group(2)
        source = pick_variant(stages.get(stage, {}), variant)
        target = pick_variant(stages.get(stage + 1, {}), variant)
        if source and target:
            edges.append(EvoboxEdge(source, target, params[key].strip()))
    return tuple(edges)


def split_alternatives(method: str) -> list[str]:
    parts = ALTERNATIVE_SEPARATOR_PATTERN.split(method)
    return [part.strip() for part in parts if part.strip()]


def read_method_text(alternative: str) -> str:
    """Return an alternative as 1 line of plain text.

    `<small>` contents stay (they hold "at night" and "for Karrablast"), a parenthetical
    that names a game is dropped, the item icons' `+` and the `→` are dropped and an
    "outside of {region}" qualifier is removed.
    """
    without_notes = GAME_NOTE_PATTERN.sub("", alternative)
    plain = to_plain_text(SMALL_TAG_PATTERN.sub("", without_notes))
    words = [word for word in plain.split() if word not in DECORATION_TOKENS]
    return REGION_QUALIFIER_PATTERN.sub("", " ".join(words)).strip()


def is_other_game_only(alternative: str) -> bool:
    match = OTHER_GAME_PATTERN.search(alternative)
    return match is not None and match.group(1).strip().lower() not in SNAPSHOT_GAME_ICONS


def select_method(method: str) -> str | None:
    """Return the Scarlet/Violet alternative of `method`, or `None` unless exactly 1 remains.

    An alternative is dropped when it is marked for another game only, or limited to a
    region that Scarlet/Violet lacks ("in Hisui").
    """
    kept = [
        alternative
        for alternative in split_alternatives(method)
        if not is_other_game_only(alternative)
        and not REGION_LIMIT_PATTERN.search(read_method_text(alternative))
    ]
    return kept[0] if len(kept) == 1 else None


class RefKind(StrEnum):
    ITEM = "item"
    MOVE = "move"
    SPECIES = "species"
    TYPE = "type"


LINK_SUFFIXES: tuple[tuple[str, RefKind], ...] = (
    (" (move)", RefKind.MOVE),
    (" (Pokémon)", RefKind.SPECIES),
    (" (type)", RefKind.TYPE),
)


@dataclass(frozen=True)
class MethodRefs:
    """The typed names an alternative mentions, from its typed templates."""

    items: tuple[str, ...] = ()
    moves: tuple[str, ...] = ()
    species: tuple[str, ...] = ()
    types: tuple[str, ...] = ()


def read_link_ref(template: Template) -> tuple[RefKind, str] | None:
    """Classify a `color2` link by its target's suffix; an unsuffixed target is an item."""
    if not template.has(LINK_TARGET_PARAM):
        return None
    target = flatten_text(str(template.get(LINK_TARGET_PARAM).value))
    for suffix, kind in LINK_SUFFIXES:
        if target.endswith(suffix):
            return kind, target.removesuffix(suffix)
    return None if target.lower() in NON_ITEM_LINKS else (RefKind.ITEM, target)


def read_first_params(code: Wikicode, names: tuple[str, ...]) -> list[str]:
    return [
        flatten_text(str(template.get(FIRST_PARAM).value))
        for template in find_templates(code, names)
        if template.has(FIRST_PARAM)
    ]


def read_method_refs(alternative: str) -> MethodRefs:
    """Read the items, moves, species and types of an alternative from its typed templates."""
    code = mwparserfromhell.parse(alternative)
    found: dict[RefKind, list[str]] = {
        RefKind.ITEM: read_first_params(code, (BAG_TEMPLATE,)),
        RefKind.MOVE: read_first_params(code, MOVE_TEMPLATES),
        RefKind.SPECIES: read_first_params(code, SPECIES_TEMPLATES),
        RefKind.TYPE: read_first_params(code, TYPE_TEMPLATES),
    }
    for template in find_templates(code, (LINK_TEMPLATE,)):
        link = read_link_ref(template)
        if link:
            found[link[0]].append(link[1])
    return MethodRefs(
        items=tuple(dict.fromkeys(found[RefKind.ITEM])),
        moves=tuple(dict.fromkeys(found[RefKind.MOVE])),
        species=tuple(dict.fromkeys(found[RefKind.SPECIES])),
        types=tuple(dict.fromkeys(found[RefKind.TYPE])),
    )


@dataclass(frozen=True)
class MethodFacts:
    trigger: str
    min_level: int | None = None
    time_of_day: str | None = None
    gender: str | None = None
    requirements: tuple[Requirement, ...] = ()


@dataclass(frozen=True)
class MethodRule:
    pattern: re.Pattern[str]
    build: Callable[[re.Match[str], MethodRefs], MethodFacts | None]


def read_group(match: re.Match[str], name: str) -> str | None:
    return match.groupdict().get(name)


def read_level(match: re.Match[str]) -> int | None:
    level = read_group(match, "level")
    return int(level) if level else None


def read_time(match: re.Match[str]) -> str | None:
    word = read_group(match, "time")
    return TIMES_OF_DAY[word.lower()] if word else None


def read_gender(match: re.Match[str]) -> str | None:
    word = read_group(match, "gender") or read_group(match, "gender_note")
    return GENDERS[word.lower()] if word else None


def requirement(role: RequirementRole, name: str, is_species: bool = False) -> Requirement:
    species = SpeciesRef(name=name, code="") if is_species else None
    return Requirement(role=role, name=name, species=species)


def build_level(match: re.Match[str], refs: MethodRefs) -> MethodFacts:
    return MethodFacts("level", min_level=read_level(match), time_of_day=read_time(match))


def build_level_gender(match: re.Match[str], refs: MethodRefs) -> MethodFacts:
    return MethodFacts("level", min_level=read_level(match), gender=read_gender(match))


def build_item(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    item = match.group("item")
    if item not in refs.items:
        return None
    return MethodFacts(
        "item",
        gender=read_gender(match),
        requirements=(requirement(RequirementRole.USE, item),),
    )


def build_trade(match: re.Match[str], refs: MethodRefs) -> MethodFacts:
    return MethodFacts("trade")


def build_trade_holding(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    item = match.group("item")
    if item not in refs.items:
        return None
    return MethodFacts("trade", requirements=(requirement(RequirementRole.HOLD, item),))


def build_trade_for(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    species = match.group("species")
    if species not in refs.species:
        return None
    role = RequirementRole.TRADED_FOR
    return MethodFacts("trade", requirements=(requirement(role, species, is_species=True),))


def build_friendship(match: re.Match[str], refs: MethodRefs) -> MethodFacts:
    return MethodFacts("friendship", time_of_day=read_time(match))


def build_level_holding(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    item = match.group("item")
    if item not in refs.items:
        return None
    return MethodFacts(
        "level",
        time_of_day=read_time(match),
        requirements=(requirement(RequirementRole.HOLD, item),),
    )


def build_level_knowing(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    move = match.group("move")
    if move not in refs.moves:
        return None
    return MethodFacts("level", requirements=(requirement(RequirementRole.KNOWN_MOVE, move),))


def build_friendship_knowing_type(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    type_name = match.group("type")
    if type_name not in refs.types:
        return None
    role = RequirementRole.KNOWN_MOVE_TYPE
    return MethodFacts("friendship", requirements=(requirement(role, type_name),))


def build_party_species(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    species = match.group("species")
    if species not in refs.species:
        return None
    role = RequirementRole.PARTY_SPECIES
    return MethodFacts(
        "level",
        min_level=read_level(match),
        requirements=(requirement(role, species, is_species=True),),
    )


def build_party_type(match: re.Match[str], refs: MethodRefs) -> MethodFacts | None:
    type_name = match.group("type")
    if type_name not in refs.types:
        return None
    return MethodFacts(
        "level",
        min_level=read_level(match),
        requirements=(requirement(RequirementRole.PARTY_TYPE, type_name),),
    )


def compile_rule(pattern: str, build: Callable[..., MethodFacts | None]) -> MethodRule:
    return MethodRule(re.compile(f"^{pattern}$", re.IGNORECASE), build)


# FR-031 order. Each pattern covers the whole plain text, so a method with an extra
# condition falls through to residue.
METHOD_RULES: tuple[MethodRule, ...] = (
    compile_rule(rf"Level (?P<level>\d+)(?: {TIME_CLAUSE})?", build_level),
    compile_rule(
        rf"Level (?P<level>\d+) (?:\((?P<gender_note>{GENDER_WORDS})\)|(?P<gender>{GENDER_WORDS}))",
        build_level_gender,
    ),
    compile_rule(
        rf"(?P<item>.+?)(?: (?:\((?P<gender_note>{GENDER_WORDS})\)|(?P<gender>{GENDER_WORDS})))?",
        build_item,
    ),
    compile_rule(r"Trade", build_trade),
    compile_rule(r"Trade (?:while )?holding (?P<item>.+)", build_trade_holding),
    compile_rule(r"Trade for (?P<species>.+)", build_trade_for),
    compile_rule(rf"Level up with high friendship(?: {TIME_CLAUSE})?", build_friendship),
    compile_rule(rf"Level up holding (?P<item>.+?)(?: {TIME_CLAUSE})?", build_level_holding),
    compile_rule(r"Level up knowing (?P<move>.+)", build_level_knowing),
    compile_rule(
        r"Level up with high friendship knowing an? (?P<type>[\w-]+?)-type move",
        build_friendship_knowing_type,
    ),
    compile_rule(rf"{LEVEL_OR_UP} with (?P<species>.+) in (?:the )?party", build_party_species),
    compile_rule(
        rf"{LEVEL_OR_UP} with an? (?P<type>[\w-]+?)-type Pokémon in (?:the )?party",
        build_party_type,
    ),
)


def apply_method_rules(alternative: str) -> MethodFacts | None:
    """Read 1 alternative with `METHOD_RULES`; `None` when no rule matches it."""
    refs = read_method_refs(alternative)
    text = read_method_text(alternative)
    for rule in METHOD_RULES:
        match = rule.pattern.match(text)
        facts = rule.build(match, refs) if match else None
        if facts is not None:
            return facts
    return None


def build_evolution(edge: EvoboxEdge, section_path: str) -> tuple[EvolutionFact, Residue | None]:
    """Turn 1 arrow into a fact. A method no rule reads keeps its cleaned text as `condition`."""
    selected = select_method(edge.method)
    facts = apply_method_rules(selected) if selected else None
    if facts is not None:
        return (
            EvolutionFact(
                source=edge.source,
                target=edge.target,
                trigger=facts.trigger,
                min_level=facts.min_level,
                time_of_day=facts.time_of_day,
                gender=facts.gender,
                requirements=facts.requirements,
                condition=None,
                section_path=section_path,
            ),
            None,
        )
    text = " or ".join(read_method_text(part) for part in split_alternatives(edge.method))
    fact = EvolutionFact(
        source=edge.source,
        target=edge.target,
        trigger=None,
        min_level=None,
        time_of_day=None,
        gender=None,
        requirements=(),
        condition=text,
        section_path=section_path,
    )
    residue = Residue(
        kind=ResidueKind.EVOLUTION,
        section_path=section_path,
        form_ids=(),
        wikitext=edge.method,
        text=text,
    )
    return fact, residue


def has_unread_evobox(section: Section) -> bool:
    """Whether the section holds an Evobox-family template that no reader handles."""
    return any(
        name.startswith(EVOBOX_PREFIX) and name not in STAGE_TEMPLATES
        for name in (
            normalize_template_name(str(template.name))
            for template in section.parse().filter_templates(recursive=True)
        )
    )


def read_evolutions(
    sections: Sequence[Section],
) -> tuple[tuple[EvolutionFact, ...], tuple[Residue, ...]]:
    """Read every Evobox stage template under Evolution data.

    A section that also holds an Evobox template no reader handles (`Evobox/setup`, a
    species template such as `Evobox/Applin`) gives 1 residue item holding the section.
    """
    evolutions: list[EvolutionFact] = []
    residue: list[Residue] = []
    for section in find_sections(sections, EVOLUTION_PATH):
        path = PATH_SEPARATOR.join(section.path)
        for template in find_templates(section.parse(), STAGE_TEMPLATES):
            for edge in read_evobox_edges(template):
                fact, item = build_evolution(edge, path)
                evolutions.append(fact)
                if item is not None:
                    residue.append(item)
        if has_unread_evobox(section):
            residue.append(
                Residue(
                    kind=ResidueKind.EVOLUTION,
                    section_path=path,
                    form_ids=(),
                    wikitext=section.text.strip(),
                    text=to_plain_text(section.text),
                )
            )
    return tuple(dict.fromkeys(evolutions)), tuple(residue)
