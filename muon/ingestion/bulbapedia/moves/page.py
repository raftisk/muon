"""1 pinned move page to 1 extract, or to the reason it is out of scope.

Map of the module:
- models: the extract, its notes and machine facts, and the skip record;
- scope: Z-Moves, Max Moves, G-Max Moves, Shadow moves and pages without a `MoveInfobox`
  are skipped, every other pinned move page is read;
- values: the infobox shows the latest game, so a value changed in Pokémon Champions is
  read back to its Scarlet/Violet number, and every other `{{tt}}` keeps its 1st parameter;
- machines: 1 fact per (kind, number, version group) from `MACHINE_FIELDS`.

Names stay in Bulbapedia terms; `batches.py` maps them onto the ontology. This module knows
the pkmn universe, as every ingestion module of a source does.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell
from mwparserfromhell.nodes import Template
from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.chunking import ChunkFact
from muon.ingestion.bulbapedia.moves.chunks import MoveChunkReading, read_move_chunks
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

INFOBOX_TEMPLATE = "moveinfobox"
INFOBOX_LABEL = "MoveInfobox"
TT_TEMPLATE = "tt"
FOOTNOTE_MARK = "*"
VALUE_PARAM = "1"
NOTE_PARAM = "2"
SHADOW_TYPE = "Shadow"
YES_VALUES = frozenset({"yes", "true"})
NO_VALUES = frozenset({"no", "false"})
NO_VALUE_MARKERS = frozenset({"", "—", "N/A", "Varies"})
UNICODE_MINUS = "−"
INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
DIGITS_PATTERN = re.compile(r"^\d+$")
CHAMPIONS_PATTERN = re.compile(r"(\d+) prior to (?:Pokémon )?Champions")
MACHINE_FIELD_PATTERN = re.compile(r"^(?:tm|hm|tr)[a-z0-9]+$")
MACHINE_NUMBER_MARK = "#"
MACHINE_KIND_LENGTH = 2  # tm, hm and tr

NAME_KEY = "name"
TYPE_KEY = "type"
DAMAGE_CLASS_KEY = "damagecategory"
PP_KEY = "basepp"
POWER_KEY = "power"
ACCURACY_KEY = "accuracy"
PRIORITY_KEY = "priority"
GENERATION_KEY = "gen"
TARGET_KEY = "target"
PROTECT_KEY = "protect"
MIRROR_MOVE_KEY = "mirrormove"
DEFAULT_PRIORITY = 0
GENERATION_LABEL = "Generation {numeral}"

SCOPE_CATEGORIES: Mapping[str, str] = {
    "Z-Moves": "z_move",
    "Max Moves": "max_move",
    "G-Max Moves": "g_max_move",
}

TARGET_CODES: Mapping[str, str] = {
    "anyadjacent": "any_adjacent",
    "self": "self",
    "adjacentfoes": "all_adjacent_foes",
    "any": "any",
    "alladjacent": "all_adjacent",
    "allfield": "entire_field",
    "allyfield": "user_side",
    "team": "user_and_allies",
    "random": "random_foe",
    "varies": "varies",
    "all": "all",
    "foes": "foe_side",
    "adjacentally": "adjacent_ally",
    "selfadjacentally": "user_or_adjacent_ally",
    "adjacentfoe": "adjacent_foe",
    "allies": "all_allies",
}

MOVE_ALIAS_RULES = AliasRules(
    template_name="langtable", infobox_keys=("jname",), should_drop_notes=True
)


class SkipReason(StrEnum):
    Z_MOVE = "z_move"
    MAX_MOVE = "max_move"
    G_MAX_MOVE = "g_max_move"
    SHADOW = "shadow"
    NO_INFOBOX = "no_infobox"


class MoveNoteKind(StrEnum):
    CHAMPIONS_VALUE = "champions_value"
    MISSING_BOOLEAN = "missing_boolean"
    NUMBER_WITHOUT_FLAG = "number_without_flag"
    UNKNOWN_FIELD = "unknown_field"
    NO_POKEAPI_MATCH = "no_pokeapi_match"
    OVERRIDE_APPLIED = "override_applied"
    TABLE_ROW_APPLIED = "table_row_applied"
    UNKNOWN_GAME_CODE = "unknown_game_code"
    UNRESOLVED_MENTION = "unresolved_mention"
    LONG_CHUNK = "long_chunk"


class MoveModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class MoveNote(MoveModel):
    """Something the parser read, left out or changed that a reviewer should see."""

    kind: MoveNoteKind
    field: str
    detail: str


class MachineFact(MoveModel):
    """1 machine the infobox names: kind (TM, HM, TR), number as written, version group."""

    kind: str
    number_text: str
    version_group_label: str
    field: str


class SkippedPage(MoveModel):
    title: str
    page_id: int
    move_id: str | None
    reason: SkipReason


class MovePageExtract(MoveModel):
    """The facts of 1 in-scope move page, in Bulbapedia names."""

    title: str
    page_id: int
    revision_id: int
    id: str
    name: str
    type_name: str
    damage_class_name: str
    generation_label: str
    pp: int
    priority: int
    target: str
    power: int | None = None
    accuracy: int | None = None
    is_protectable: bool | None = None
    is_mirrorable: bool | None = None
    aliases: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()
    crit_stage: int | None = None
    min_hits: int | None = None
    max_hits: int | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    machines: tuple[MachineFact, ...] = ()
    chunks: tuple[ChunkFact, ...] = ()
    dropped_templates: Mapping[str, int] = {}
    removed_tables: int = 0
    math_formulas: int = 0
    notes: tuple[MoveNote, ...] = ()


class MovePageParseError(Exception):
    """A page broke a rule that has no other path. The message names page, field and value."""

    def __init__(self, page_title: str, field: str, value: str) -> None:
        super().__init__(f"{page_title}: cannot read {field} in {INFOBOX_LABEL}: {value!r}")
        self.page_title = page_title
        self.field = field
        self.value = value


@dataclass(frozen=True)
class MachineField:
    """An infobox machine field: its kind and the version groups (A-Box labels) it covers."""

    kind: str
    version_group_labels: tuple[str, ...]


GENERATION_III_GROUPS = ("Ruby-Sapphire", "Emerald", "FireRed-LeafGreen", "Colosseum", "XD")
GENERATION_IV_GROUPS = ("Diamond-Pearl", "Platinum", "HeartGold-SoulSilver")
GENERATION_VI_GROUPS = ("X-Y", "Omega Ruby-Alpha Sapphire")

MACHINE_FIELDS: Mapping[str, MachineField] = {
    "tm1": MachineField("TM", ("Red-Blue", "Yellow")),
    "hm1": MachineField("HM", ("Red-Blue", "Yellow")),
    "tm2": MachineField("TM", ("Gold-Silver", "Crystal")),
    "hm2": MachineField("HM", ("Gold-Silver", "Crystal")),
    "tm3": MachineField("TM", GENERATION_III_GROUPS),
    "hm3": MachineField("HM", GENERATION_III_GROUPS),
    "hmrse": MachineField("HM", ("Ruby-Sapphire", "Emerald")),
    "tm4": MachineField("TM", GENERATION_IV_GROUPS),
    "hm4": MachineField("HM", GENERATION_IV_GROUPS),
    "hmdppt": MachineField("HM", ("Diamond-Pearl", "Platinum")),
    "hmhs": MachineField("HM", ("HeartGold-SoulSilver",)),
    "tm5": MachineField("TM", ("Black-White", "Black 2-White 2")),
    "hm5": MachineField("HM", ("Black-White", "Black 2-White 2")),
    "tm6": MachineField("TM", GENERATION_VI_GROUPS),
    "hm6": MachineField("HM", GENERATION_VI_GROUPS),
    "tmxy": MachineField("TM", ("X-Y",)),
    "tmoras": MachineField("TM", ("Omega Ruby-Alpha Sapphire",)),
    "hmoras": MachineField("HM", ("Omega Ruby-Alpha Sapphire",)),
    "tm7": MachineField("TM", ("Sun-Moon", "Ultra Sun-Ultra Moon")),
    "tmpe": MachineField("TM", ("Let's Go Pikachu-Let's Go Eevee",)),
    "tm8": MachineField("TM", ("Sword-Shield",)),
    "tr8": MachineField("TR", ("Sword-Shield",)),
    "tmbdsp": MachineField("TM", ("Brilliant Diamond and Shining Pearl",)),
    "tm9": MachineField("TM", ("Scarlet-Violet",)),
    "tmza": MachineField("TM", ("Legends: Z-A",)),
}


FLAG_FIELDS: Mapping[str, str] = {
    "touches": "contact",
    "sound": "sound",
    "magiccoat": "reflectable",
    "snatch": "snatchable",
}

FLAG_CATEGORIES: Mapping[str, str] = {
    "Punching moves": "punch",
    "Biting moves": "bite",
    "Ball and bomb moves": "ballistic",
    "Powder and spore moves": "powder",
    "Pulse moves": "pulse",
    "Wind moves": "wind",
    "Slicing moves": "slicing",
    "Dance moves": "dance",
    "Moves that switch the user out": "pivot",
    "Multistrike moves": "multihit",
    "One-hit knockout moves": "ohko",
    "Moves with a charging turn": "two-turn",
    "Moves with a semi-invulnerable turn": "two-turn",
    "Moves that have recoil": "recoil",
    "HP-draining moves": "drain",
    "Status moves that heal the user immediately": "healing",
}

MULTISTRIKE_CATEGORY = "Multistrike moves"


@dataclass(frozen=True)
class FlagOverride:
    added: frozenset[str] = frozenset()
    removed: frozenset[str] = frozenset()


TWO_TURN = FlagOverride(added=frozenset({"two-turn"}))
HEALING = FlagOverride(added=frozenset({"healing"}))
EXPLOSIVE = FlagOverride(added=frozenset({"explosive"}))

FLAG_OVERRIDES: Mapping[str, FlagOverride] = {
    "dig": TWO_TURN,  # Underground on the 1st turn, attacks on the 2nd
    "dive": TWO_TURN,  # Submerged on the 1st turn, attacks on the 2nd
    "sky-attack": TWO_TURN,  # Charging status on the 1st turn
    "freeze-shock": TWO_TURN,  # Charging status, skipped with a Power Herb
    "ice-burn": TWO_TURN,  # Charging status, skipped with a Power Herb
    "dark-void": FlagOverride(removed=frozenset({"drain"})),  # puts foes to sleep, drains no HP
    "floral-healing": HEALING,  # restores 1/2 of the target's maximum HP
    "heal-pulse": HEALING,  # restores 50% of the target's maximum HP
    "jungle-healing": HEALING,  # restores 25% to the user and its allies
    "life-dew": HEALING,  # restores 25% to the user and its allies
    "swallow": HEALING,  # restores HP by the Stockpile count
    "wish": HEALING,  # heals the Pokémon in the user's position at the end of next turn
    "healing-wish": HEALING,  # restores the HP of the Pokémon sent out next
    "lunar-dance": HEALING,  # restores the HP and PP of the Pokémon sent out next
    "explosion": EXPLOSIVE,  # cannot be used while a Pokémon with Damp is on the field
    "self-destruct": EXPLOSIVE,  # cannot be used while a Pokémon with Damp is on the field
    "misty-explosion": EXPLOSIVE,  # fails while a Pokémon with Damp is on the field
    "mind-blown": EXPLOSIVE,  # cannot be used while a Pokémon with Damp is on the field
}

HIGH_CRIT_STAGE = 1
PERFECT_CRIT_STAGE = 3

CRIT_CATEGORIES: Mapping[str, int] = {
    "Moves with a high critical-hit ratio": HIGH_CRIT_STAGE,
    "Moves with a perfect critical hit chance": PERFECT_CRIT_STAGE,
}

CRIT_OVERRIDES: Mapping[str, int | None] = {
    "aeroblast": HIGH_CRIT_STAGE,  # "has an increased critical hit ratio"
    "attack-order": HIGH_CRIT_STAGE,  # "has an increased critical hit ratio"
    "cross-chop": HIGH_CRIT_STAGE,  # "has an increased critical hit ratio"
    "leaf-blade": HIGH_CRIT_STAGE,  # "has an increased critical hit ratio"
    "spacial-rend": HIGH_CRIT_STAGE,  # increased ratio, unchanged from Generation VIII on
    "stone-edge": HIGH_CRIT_STAGE,  # "has an increased critical hit ratio"
    "frost-breath": PERFECT_CRIT_STAGE,  # "will always result in a critical hit if possible"
    "storm-throw": PERFECT_CRIT_STAGE,  # "will always result in a critical hit if possible"
    "dragon-claw": None,  # "has no secondary effect"
    "plasma-fists": None,  # the Effect section names no critical-hit effect
    "zippy-zap": None,  # from Generation VIII raises evasion instead of a sure critical hit
    "dire-claw": None,  # Scarlet/Violet: "does not have an increased critical-hit ratio"
}

TWO_TO_FIVE = (2, 5)
TWICE = (2, 2)
THREE_TIMES = (3, 3)

MOVE_HITS: Mapping[str, tuple[int, int]] = {
    "arm-thrust": TWO_TO_FIVE,  # "2-5 times per use"
    "barrage": TWO_TO_FIVE,  # "2-5 times per use"
    "beat-up": (1, 6),  # 1 strike per healthy party member, the user included
    "bone-rush": TWO_TO_FIVE,  # "2-5 times per use"
    "bonemerang": TWICE,  # "hitting the target twice per use"
    "bullet-seed": TWO_TO_FIVE,  # "2-5 times per use"
    "comet-punch": TWO_TO_FIVE,  # "2-5 times per use"
    "double-hit": TWICE,  # "hitting twice per use"
    "double-iron-bash": TWICE,  # "hitting the target twice per use"
    "double-kick": TWICE,  # "hitting the target twice per use"
    "double-slap": TWO_TO_FIVE,  # "2-5 times per use"
    "dragon-darts": TWICE,  # "deals damage twice"
    "dual-chop": TWICE,  # "hitting twice per use"
    "dual-wingbeat": TWICE,  # "hitting twice per use"
    "fury-attack": TWO_TO_FIVE,  # "2-5 times per use"
    "fury-swipes": TWO_TO_FIVE,  # "2-5 times per use"
    "gear-grind": TWICE,  # "hitting twice per use"
    "icicle-spear": TWO_TO_FIVE,  # "2-5 times per use"
    "pin-missile": TWO_TO_FIVE,  # "2-5 times per use"
    "population-bomb": (1, 10),  # up to 10, each strike checks accuracy and a miss ends it
    "rock-blast": TWO_TO_FIVE,  # "2-5 times per use"
    "scale-shot": TWO_TO_FIVE,  # "2-5 times per use"
    "spike-cannon": TWO_TO_FIVE,  # "2-5 times per use"
    "surging-strikes": THREE_TIMES,  # "hitting its target three times per use"
    "tachyon-cutter": TWICE,  # "hitting twice per use"
    "tail-slap": TWO_TO_FIVE,  # "2-5 times per use"
    "triple-axel": (1, 3),  # up to 3, each strike checks accuracy and a miss ends it
    "triple-dive": THREE_TIMES,  # "hitting the target three times per use"
    "triple-kick": (1, 3),  # up to 3, each strike checks accuracy and a miss ends it
    "twin-beam": TWICE,  # "hitting twice per use"
    "twineedle": TWICE,  # "hitting the target twice per use"
    "water-shuriken": TWO_TO_FIVE,  # "2-5 times per use"
}

RAMPAGE = (2, 3)
ROLLING = (5, 5)

MOVE_TURNS: Mapping[str, tuple[int, int]] = {
    "uproar": (3, 3),  # Generation V on: "exactly 3 turns"
    "thrash": RAMPAGE,  # Rampaging "for 2-3 turns"
    "petal-dance": RAMPAGE,  # Rampaging "for 2-3 turns"
    "outrage": RAMPAGE,  # Rampaging "for 2-3 turns"
    "raging-fury": RAMPAGE,  # Rampaging "for 2-3 turns"
    "rollout": ROLLING,  # "inflicts damage over five turns"
    "ice-ball": ROLLING,  # "inflicts damage over five turns"
}

PARSER_TABLES: Mapping[str, Mapping[str, object]] = {
    "FLAG_OVERRIDES": FLAG_OVERRIDES,
    "CRIT_OVERRIDES": CRIT_OVERRIDES,
    "MOVE_HITS": MOVE_HITS,
    "MOVE_TURNS": MOVE_TURNS,
}


class TableError(Exception):
    """A parser table disagrees with the pinned pages."""


@dataclass(frozen=True)
class NumberReading:
    value: int | None
    notes: tuple[MoveNote, ...] = ()


def read_move_infobox(lead: Section) -> dict[str, str] | None:
    """Return the parameters of the first `MoveInfobox` of the lead, or `None` if absent.

    The lead must have its comments masked: Doodle keeps a comment after its `gen` value.
    """
    infoboxes = find_templates(lead.parse(), [INFOBOX_TEMPLATE])
    if not infoboxes:
        return None
    return read_params(infoboxes[0])


def read_skip_reason(infobox: Mapping[str, str], categories: frozenset[str]) -> SkipReason | None:
    for category, reason in SCOPE_CATEGORIES.items():
        if category in categories:
            return SkipReason(reason)
    if infobox.get(TYPE_KEY, "").strip() == SHADOW_TYPE:
        return SkipReason.SHADOW
    return None


def is_tt(template: Template) -> bool:
    return normalize_template_name(str(template.name)) == TT_TEMPLATE


def is_footnote(template: Template) -> bool:
    return is_tt(template) and str(template.get(VALUE_PARAM).value).strip() == FOOTNOTE_MARK


def split_tt_value(raw: str) -> tuple[str, str]:
    """Return the value and the `{{tt}}` note of an infobox value, footnotes removed.

    A value that is 1 `{{tt|value|note}}` gives its 2 parameters; any other value gives
    itself and an empty note.
    """
    code = mwparserfromhell.parse(raw)
    for template in code.filter_templates(recursive=False):
        if template.has(VALUE_PARAM) and is_footnote(template):
            code.remove(template)
    nodes = [node for node in code.nodes if str(node).strip()]
    if len(nodes) == 1 and isinstance(nodes[0], Template) and is_tt(nodes[0]):
        template = nodes[0]
        value = str(template.get(VALUE_PARAM).value) if template.has(VALUE_PARAM) else ""
        note = str(template.get(NOTE_PARAM).value) if template.has(NOTE_PARAM) else ""
        return value, note
    return str(code), ""


def read_number(page_title: str, infobox: Mapping[str, str], key: str) -> NumberReading:
    """Read an infobox number per D6: the Scarlet/Violet value, `None` for a no-value marker."""
    value, note = split_tt_value(infobox.get(key, ""))
    notes: tuple[MoveNote, ...] = ()
    champions = CHAMPIONS_PATTERN.search(note)
    if champions is not None:
        notes = (MoveNote(kind=MoveNoteKind.CHAMPIONS_VALUE, field=key, detail=note.strip()),)
        value = champions.group(1)
    text = to_plain_text(value).replace(UNICODE_MINUS, "-").strip()
    if text in NO_VALUE_MARKERS:
        return NumberReading(None, notes)
    if INTEGER_PATTERN.match(text) is None:
        raise MovePageParseError(page_title, key, infobox.get(key, ""))
    return NumberReading(int(text), notes)


def read_required(page_title: str, infobox: Mapping[str, str], key: str) -> str:
    text = to_plain_text(infobox.get(key, "")).strip()
    if not text:
        raise MovePageParseError(page_title, key, infobox.get(key, ""))
    return text


def read_target(page_title: str, infobox: Mapping[str, str]) -> str:
    code = read_required(page_title, infobox, TARGET_KEY)
    target = TARGET_CODES.get(code)
    if target is None:
        raise MovePageParseError(page_title, TARGET_KEY, code)
    return target


def read_boolean(infobox: Mapping[str, str], key: str) -> tuple[bool | None, tuple[MoveNote, ...]]:
    """`yes` is true and `no` false. A missing or other value gives no value and a note."""
    value = infobox.get(key, "").strip().lower()
    if value in YES_VALUES:
        return True, ()
    if value in NO_VALUES:
        return False, ()
    detail = f"{key} is {value!r}" if value else f"no {key} field"
    return None, (MoveNote(kind=MoveNoteKind.MISSING_BOOLEAN, field=key, detail=detail),)


def read_flags(
    infobox: Mapping[str, str], categories: frozenset[str], move_id: str
) -> tuple[frozenset[str], tuple[MoveNote, ...]]:
    """Return the flag ids from the infobox fields and the categories, then `FLAG_OVERRIDES`."""
    from_fields = {
        flag
        for key, flag in FLAG_FIELDS.items()
        if infobox.get(key, "").strip().lower() in YES_VALUES
    }
    from_categories = {flag for category, flag in FLAG_CATEGORIES.items() if category in categories}
    flags = frozenset(from_fields | from_categories)
    override = FLAG_OVERRIDES.get(move_id)
    if override is None:
        return flags, ()
    detail = f"added {sorted(override.added)}, removed {sorted(override.removed)}"
    note = MoveNote(kind=MoveNoteKind.OVERRIDE_APPLIED, field="flags", detail=detail)
    return (flags | override.added) - override.removed, (note,)


def read_crit_stage(
    categories: frozenset[str], move_id: str
) -> tuple[int | None, tuple[MoveNote, ...]]:
    """Return the stage of the crit categories, or the `CRIT_OVERRIDES` row of the move."""
    stages = [stage for category, stage in CRIT_CATEGORIES.items() if category in categories]
    stage = max(stages) if stages else None
    if move_id not in CRIT_OVERRIDES:
        return stage, ()
    override = CRIT_OVERRIDES[move_id]
    detail = f"{stage} to {override}"
    return override, (
        MoveNote(kind=MoveNoteKind.OVERRIDE_APPLIED, field="crit_stage", detail=detail),
    )


def read_range(table: Mapping[str, tuple[int, int]], move_id: str) -> tuple[int | None, int | None]:
    low_high = table.get(move_id)
    if low_high is None:
        return None, None
    return low_high


def build_table_note(
    table_name: str, table: Mapping[str, tuple[int, int]], move_id: str
) -> tuple[MoveNote, ...]:
    if move_id not in table:
        return ()
    low, high = table[move_id]
    detail = f"{table_name} {low} to {high}"
    return (MoveNote(kind=MoveNoteKind.TABLE_ROW_APPLIED, field=table_name, detail=detail),)


def check_tables(extracts: Iterable[MovePageExtract]) -> None:
    """Fail when a table row names no in-scope move, or `MOVE_HITS` misses a multistrike move."""
    in_scope = list(extracts)
    move_ids = {extract.id for extract in in_scope}
    problems = [
        f"{table_name} names {move_id}, which is no in-scope move"
        for table_name, table in PARSER_TABLES.items()
        for move_id in sorted(set(table) - move_ids)
    ]
    multistrike = {extract.id for extract in in_scope if MULTISTRIKE_CATEGORY in extract.categories}
    problems.extend(
        f"MOVE_HITS lacks {move_id}, a member of {MULTISTRIKE_CATEGORY!r}"
        for move_id in sorted(multistrike - set(MOVE_HITS))
    )
    problems.extend(
        f"MOVE_HITS names {move_id}, which is not in {MULTISTRIKE_CATEGORY!r}"
        for move_id in sorted((set(MOVE_HITS) & move_ids) - multistrike)
    )
    if problems:
        raise TableError("; ".join(problems))


def build_number_key(field: str) -> str:
    """Return the number field of a machine field: `tm4` gives `tm#4`, `hmrse` gives `hm#rse`."""
    return field[:MACHINE_KIND_LENGTH] + MACHINE_NUMBER_MARK + field[MACHINE_KIND_LENGTH:]


def read_machines(
    page_title: str, infobox: Mapping[str, str]
) -> tuple[tuple[MachineFact, ...], tuple[MoveNote, ...]]:
    """Return 1 fact per version group of every machine field set to `yes` with a number."""
    machines: list[MachineFact] = []
    notes: list[MoveNote] = []
    for field, spec in MACHINE_FIELDS.items():
        number_key = build_number_key(field)
        is_set = infobox.get(field, "").strip().lower() in YES_VALUES
        number_text = infobox.get(number_key, "").strip()
        if not is_set:
            if number_text:
                detail = f"{number_key}={number_text} without {field}=yes"
                notes.append(
                    MoveNote(kind=MoveNoteKind.NUMBER_WITHOUT_FLAG, field=number_key, detail=detail)
                )
            continue
        if DIGITS_PATTERN.match(number_text) is None:
            raise MovePageParseError(page_title, number_key, number_text)
        machines.extend(
            MachineFact(
                kind=spec.kind, number_text=number_text, version_group_label=label, field=field
            )
            for label in spec.version_group_labels
        )
    for field in sorted(infobox):
        if MACHINE_FIELD_PATTERN.match(field) and field not in MACHINE_FIELDS:
            detail = f"{field}={infobox[field].strip()}"
            notes.append(MoveNote(kind=MoveNoteKind.UNKNOWN_FIELD, field=field, detail=detail))
    return tuple(machines), tuple(notes)


def build_chunk_notes(reading: MoveChunkReading) -> tuple[MoveNote, ...]:
    unknown = (
        MoveNote(
            kind=MoveNoteKind.UNKNOWN_GAME_CODE,
            field=code.section_path,
            detail=f"{code.cell} has a game code no rule reads",
        )
        for code in reading.unknown_codes
    )
    long = (
        MoveNote(
            kind=MoveNoteKind.LONG_CHUNK,
            field=chunk.section_path,
            detail=f"{len(chunk.text)} characters",
        )
        for chunk in reading.long_chunks
    )
    return (*unknown, *long)


def parse_move_page(page: Page) -> MovePageExtract | SkippedPage:
    """Parse 1 pinned move page. Raises `MovePageParseError` for a value no rule can read."""
    masked = mask_comments(page.wikitext)
    infobox = read_move_infobox(split_sections(masked)[0])
    if infobox is None:
        return SkippedPage(
            title=page.title, page_id=page.page_id, move_id=None, reason=SkipReason.NO_INFOBOX
        )
    name = read_required(page.title, infobox, NAME_KEY)
    categories = read_categories(masked)
    reason = read_skip_reason(infobox, categories)
    if reason is not None:
        return SkippedPage(
            title=page.title, page_id=page.page_id, move_id=build_node_id(name), reason=reason
        )
    readings = {key: read_number(page.title, infobox, key) for key in (POWER_KEY, ACCURACY_KEY)}
    pp = read_number(page.title, infobox, PP_KEY)
    if pp.value is None:
        raise MovePageParseError(page.title, PP_KEY, infobox.get(PP_KEY, ""))
    priority = read_number(page.title, infobox, PRIORITY_KEY)
    is_protectable, protect_notes = read_boolean(infobox, PROTECT_KEY)
    is_mirrorable, mirror_notes = read_boolean(infobox, MIRROR_MOVE_KEY)
    machines, machine_notes = read_machines(page.title, infobox)
    numeral = read_required(page.title, infobox, GENERATION_KEY)
    move_id = build_node_id(name)
    flags, flag_notes = read_flags(infobox, categories, move_id)
    crit_stage, crit_notes = read_crit_stage(categories, move_id)
    min_hits, max_hits = read_range(MOVE_HITS, move_id)
    min_turns, max_turns = read_range(MOVE_TURNS, move_id)
    sections = split_sections(page.wikitext)
    chunk_reading = read_move_chunks(sections, move_id)
    return MovePageExtract(
        title=page.title,
        page_id=page.page_id,
        revision_id=page.revision_id,
        id=move_id,
        name=name,
        type_name=read_required(page.title, infobox, TYPE_KEY),
        damage_class_name=read_required(page.title, infobox, DAMAGE_CLASS_KEY),
        generation_label=GENERATION_LABEL.format(numeral=numeral),
        pp=pp.value,
        priority=DEFAULT_PRIORITY if priority.value is None else priority.value,
        target=read_target(page.title, infobox),
        power=readings[POWER_KEY].value,
        accuracy=readings[ACCURACY_KEY].value,
        is_protectable=is_protectable,
        is_mirrorable=is_mirrorable,
        aliases=read_aliases(sections, infobox, name, MOVE_ALIAS_RULES),
        categories=tuple(sorted(categories)),
        flags=tuple(sorted(flags)),
        crit_stage=crit_stage,
        min_hits=min_hits,
        max_hits=max_hits,
        min_turns=min_turns,
        max_turns=max_turns,
        machines=machines,
        chunks=chunk_reading.chunks,
        dropped_templates=chunk_reading.dropped_templates,
        removed_tables=chunk_reading.removed_tables,
        math_formulas=chunk_reading.math_formulas,
        notes=(
            *readings[POWER_KEY].notes,
            *readings[ACCURACY_KEY].notes,
            *pp.notes,
            *priority.notes,
            *protect_notes,
            *mirror_notes,
            *machine_notes,
            *flag_notes,
            *crit_notes,
            *build_table_note("MOVE_HITS", MOVE_HITS, move_id),
            *build_table_note("MOVE_TURNS", MOVE_TURNS, move_id),
            *build_chunk_notes(chunk_reading),
        ),
    )
