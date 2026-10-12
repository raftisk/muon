"""The typed extract of 1 species page, in Bulbapedia terms.

Names are the page's own words and values are parsed, but nothing here is an IRI or a node
id of the A-Box: `batches.py` maps the extract onto the ontology. A fact that no rule
could read is a `Residue` item, left for the LLM extractor.
"""

from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PageParseError(Exception):
    """A page broke a rule that has no residue path. The message names all 4 fields."""

    def __init__(self, page_title: str, section_path: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: cannot read {field} in {section_path}: {value!r}")
        self.page_title = page_title
        self.section_path = section_path
        self.field = field
        self.value = value


class ExtractModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ResidueKind(StrEnum):
    ABILITIES = "abilities"
    GENUS = "genus"
    EVOLUTION = "evolution"
    FORM_CHANGE = "form_change"


class Residue(ExtractModel):
    """A fact left for the LLM extractor. `form_ids` names the forms it concerns."""

    kind: ResidueKind
    section_path: str
    form_ids: tuple[str, ...]
    wikitext: str
    text: str


class NoteKind(StrEnum):
    UNMATCHED_SUBSECTION = "unmatched_subsection"
    DROPPED_VALUE = "dropped_value"
    DUPLICATE_ABILITY = "duplicate_ability"
    OVERRIDE_APPLIED = "override_applied"
    REFERENCE_FALLBACK = "reference_fallback"
    MISSING_LEARNSET = "missing_learnset"
    UNRESOLVED_MENTION = "unresolved_mention"
    UNMATCHED_FORM_LABEL = "unmatched_form_label"
    UNKNOWN_VERSION = "unknown_version"


class PageNote(ExtractModel):
    """Something the parser left out or changed that a reviewer should see."""

    kind: NoteKind
    section_path: str
    detail: str


class FormOutcome(StrEnum):
    NODE = "node"
    COSMETIC = "cosmetic"
    SKIPPED = "skipped"
    OVERRIDDEN = "overridden"


class FormCandidateResult(ExtractModel):
    """What became of 1 infobox form. `form_id` is set when the form is a node."""

    index: int
    infobox_name: str
    outcome: FormOutcome
    form_id: str | None


class AbilitySlot(ExtractModel):
    name: str
    is_hidden: bool


class BaseStats(ExtractModel):
    hp: int
    attack: int
    defense: int
    sp_attack: int
    sp_defense: int
    speed: int


class FormFacts(ExtractModel):
    """The facts of 1 species node. `ev_yield` maps an infobox stat key to its amount."""

    id: str
    name: str
    form_index: int
    is_default: bool
    region: str | None
    generation: int | None
    pokedex_number: int
    types: tuple[str, ...]
    abilities: tuple[AbilitySlot, ...]
    egg_groups: tuple[str, ...]
    height: float | None
    weight: float | None
    female_rate: float | None
    is_genderless: bool
    catch_rate: int
    base_exp_yield: int
    growth_rate: str
    egg_cycles: int
    base_friendship: int
    color: str | None
    shape: str | None
    genus: str | None
    is_legendary: bool
    is_mythical: bool
    is_starter: bool
    base_stats: BaseStats
    ev_yield: Mapping[str, int]
    aliases: tuple[str, ...]


class LearnMethod(StrEnum):
    LEVEL = "level"
    EVOLUTION = "evolution"
    REMINDER = "reminder"
    MACHINE = "machine"
    EGG = "egg"
    SPECIAL = "special"


class LearnsetRow(ExtractModel):
    form_id: str
    move: str
    method: LearnMethod
    level: int | None
    machine: str | None


class SpeciesRef(ExtractModel):
    """A species as an Evobox names it. `code` is the sprite code, empty for the default form."""

    name: str
    code: str


class RequirementRole(StrEnum):
    USE = "use"
    HOLD = "hold"
    KNOWN_MOVE = "knownMove"
    KNOWN_MOVE_TYPE = "knownMoveType"
    PARTY_SPECIES = "partySpecies"
    PARTY_TYPE = "partyType"
    TRADED_FOR = "tradedFor"


class Requirement(ExtractModel):
    """What an evolution needs. `species` is set for the 3 roles that name a species."""

    role: RequirementRole
    name: str
    species: SpeciesRef | None


class EvolutionFact(ExtractModel):
    """1 edge of an Evobox. `condition` holds the method text when no rule read it."""

    source: SpeciesRef
    target: SpeciesRef
    trigger: str | None
    min_level: int | None
    time_of_day: str | None
    gender: str | None
    requirements: tuple[Requirement, ...]
    condition: str | None
    section_path: str


class FormChangeFact(ExtractModel):
    from_id: str
    to_id: str
    form_kind: str
    is_battle_only: bool
    trigger: str | None
    held_item: str


class ChunkAspect(StrEnum):
    BIOLOGY = "biology"
    ORIGIN = "origin"
    TRIVIA = "trivia"
    POKEDEX = "pokedex"


class MentionRef(ExtractModel):
    """A typed template that rendered into a chunk: its name and its raw 1st parameter."""

    template: str
    target: str


class ChunkFact(ExtractModel):
    """1 chunk in Bulbapedia terms. `subject_id` is a form id of the page.

    `versions` holds the game names of a Pokédex entry, in page order, and is empty for
    every other aspect.
    """

    subject_id: str
    aspect: ChunkAspect
    section_path: str
    text: str
    mentions: tuple[MentionRef, ...]
    versions: tuple[str, ...] = ()


class SpeciesPageExtract(ExtractModel):
    """Everything 1 pinned page yields. `forms` lists the base node first.

    `dropped_templates` counts the templates left out of chunk text, by normalized name.
    """

    title: str
    page_id: int
    revision_id: int
    forms: tuple[FormFacts, ...]
    candidates: tuple[FormCandidateResult, ...]
    learnset: tuple[LearnsetRow, ...]
    form_changes: tuple[FormChangeFact, ...]
    evolutions: tuple[EvolutionFact, ...]
    residue: tuple[Residue, ...]
    notes: tuple[PageNote, ...]
    chunks: tuple[ChunkFact, ...] = ()
    dropped_templates: Mapping[str, int] = Field(default_factory=dict)
