"""What the move parser reads from the pinned PokeAPI snapshot.

PokeAPI owns 1 move fact, `effectText` (P1). Every other value only feeds `cross_check()`,
which reports where PokeAPI and Bulbapedia differ and never changes a batch (P7). Moves are
keyed by their Bulbapedia id; only regular moves (PokeAPI id below 10000) enter the index.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from muon.ingestion.batch import build_node_id
from muon.ingestion.bulbapedia.moves.page import MachineFact, MovePageExtract
from muon.ingestion.pokeapi.snapshot import SnapshotStore

ENGLISH_LANGUAGE_ID = "9"
REGULAR_MOVE_ID_LIMIT = 10000
POKEAPI_ID_EXCEPTIONS: Mapping[str, str] = {"vise-grip": "vice-grip"}
BULBAPEDIA_IDS: Mapping[str, str] = {value: key for key, value in POKEAPI_ID_EXCEPTIONS.items()}
CRIT_RATE_STAGES: Mapping[str, int] = {"1": 1, "6": 3}
VERSION_GROUP_MAP: Mapping[str, str] = {
    "brilliant-diamond-shining-pearl": "brilliant-diamond-and-shining-pearl",
    "legends-za": "legends-z-a",
    "the-isle-of-armor": "sword-shield",
    "the-crown-tundra": "sword-shield",
    "the-teal-mask": "scarlet-violet",
    "the-indigo-disk": "scarlet-violet",
}
SKIPPED_VERSION_GROUPS = frozenset({"red-green-japan", "blue-japan", "mega-dimension", "champions"})
MACHINE_ITEM_PATTERN = re.compile(r"^(tm|hm|tr)(\d+)$")

MOVES_FILE = "moves.csv"
META_FILE = "move_meta.csv"
EFFECT_PROSE_FILE = "move_effect_prose.csv"
MACHINES_FILE = "machines.csv"
ITEMS_FILE = "items.csv"
VERSION_GROUPS_FILE = "version_groups.csv"
TYPES_FILE = "types.csv"
DAMAGE_CLASSES_FILE = "move_damage_classes.csv"

ID_COLUMN = "id"
IDENTIFIER_COLUMN = "identifier"
NO_VALUE = ""


@dataclass(frozen=True, order=True)
class MachineKey:
    kind: str
    number: int
    version_group_id: str


@dataclass(frozen=True)
class PokeapiMove:
    """1 PokeAPI move in ontology terms. `has_meta` is false for a move without meta row."""

    identifier: str
    power: int | None
    accuracy: int | None
    pp: int | None
    priority: int
    type_id: str
    damage_class_id: str
    has_meta: bool
    crit_stage: int | None
    min_hits: int | None
    max_hits: int | None
    effect_text: str | None
    machines: frozenset[MachineKey]


@dataclass(frozen=True)
class PokeapiIndex:
    """PokeAPI moves by Bulbapedia id, and the machine rows of skipped version groups."""

    moves: Mapping[str, PokeapiMove]
    skipped_machine_count: int


class CrossCheckDifference(BaseModel):
    """1 value on which Bulbapedia and PokeAPI disagree. An empty string means no value."""

    model_config = ConfigDict(frozen=True)

    move_id: str
    field: str
    bulbapedia_value: str
    pokeapi_value: str


@dataclass(frozen=True)
class MachineRows:
    keys_by_move: Mapping[str, frozenset[MachineKey]]
    skipped_count: int


def read_optional_int(text: str) -> int | None:
    return None if text == NO_VALUE else int(text)


def read_identifiers(store: SnapshotStore, name: str) -> dict[str, str]:
    return {row[ID_COLUMN]: row[IDENTIFIER_COLUMN] for row in store.read_table(name)}


def read_effect_texts(store: SnapshotStore) -> dict[str, str]:
    """Return the English short effect of every effect id, as written."""
    return {
        row["move_effect_id"]: row["short_effect"]
        for row in store.read_table(EFFECT_PROSE_FILE)
        if row["local_language_id"] == ENGLISH_LANGUAGE_ID and row["short_effect"]
    }


def read_machine_rows(store: SnapshotStore) -> MachineRows:
    """Return the machines of each PokeAPI move id, numbered from the item identifier.

    `machine_number` is no help: HMs and TRs count on from 100. DLC version groups fold
    into their base group, and the groups of `SKIPPED_VERSION_GROUPS` are counted only.
    """
    items = read_identifiers(store, ITEMS_FILE)
    version_groups = read_identifiers(store, VERSION_GROUPS_FILE)
    keys: defaultdict[str, set[MachineKey]] = defaultdict(set)
    skipped = 0
    for row in store.read_table(MACHINES_FILE):
        group = version_groups[row["version_group_id"]]
        if group in SKIPPED_VERSION_GROUPS:
            skipped += 1
            continue
        item = MACHINE_ITEM_PATTERN.match(items[row["item_id"]])
        if item is None:
            continue
        version_group_id = VERSION_GROUP_MAP.get(group, group)
        keys[row["move_id"]].add(
            MachineKey(item.group(1).upper(), int(item.group(2)), version_group_id)
        )
    return MachineRows(
        keys_by_move={move_id: frozenset(found) for move_id, found in keys.items()},
        skipped_count=skipped,
    )


def build_pokeapi_index(store: SnapshotStore) -> PokeapiIndex:
    """Read the pinned snapshot into an index of regular moves by Bulbapedia id."""
    types = read_identifiers(store, TYPES_FILE)
    damage_classes = read_identifiers(store, DAMAGE_CLASSES_FILE)
    effect_texts = read_effect_texts(store)
    meta = {row["move_id"]: row for row in store.read_table(META_FILE)}
    machines = read_machine_rows(store)
    moves: dict[str, PokeapiMove] = {}
    for row in store.read_table(MOVES_FILE):
        if int(row[ID_COLUMN]) >= REGULAR_MOVE_ID_LIMIT:
            continue
        meta_row = meta.get(row[ID_COLUMN])
        identifier = row[IDENTIFIER_COLUMN]
        moves[BULBAPEDIA_IDS.get(identifier, identifier)] = PokeapiMove(
            identifier=identifier,
            power=read_optional_int(row["power"]),
            accuracy=read_optional_int(row["accuracy"]),
            pp=read_optional_int(row["pp"]),
            priority=int(row["priority"]),
            type_id=types[row["type_id"]],
            damage_class_id=damage_classes[row["damage_class_id"]],
            has_meta=meta_row is not None,
            crit_stage=None if meta_row is None else CRIT_RATE_STAGES.get(meta_row["crit_rate"]),
            min_hits=None if meta_row is None else read_optional_int(meta_row["min_hits"]),
            max_hits=None if meta_row is None else read_optional_int(meta_row["max_hits"]),
            effect_text=effect_texts.get(row["effect_id"]),
            machines=machines.keys_by_move.get(row[ID_COLUMN], frozenset()),
        )
    return PokeapiIndex(moves=moves, skipped_machine_count=machines.skipped_count)


def read_effect_text(index: PokeapiIndex, move_id: str) -> str | None:
    """Return the English short effect of a move, or `None` without a row or English text."""
    move = index.moves.get(move_id)
    return None if move is None else move.effect_text


def format_value(value: object) -> str:
    return NO_VALUE if value is None else str(value)


def format_hits(low: int | None, high: int | None) -> str:
    return NO_VALUE if low is None else f"{low}-{high}"


def format_machine(key: MachineKey) -> str:
    return f"{key.kind}{key.number} {key.version_group_id}"


def build_machine_key(fact: MachineFact, version_group_ids: Mapping[str, str]) -> MachineKey:
    return MachineKey(fact.kind, int(fact.number_text), version_group_ids[fact.version_group_label])


def list_machine_differences(
    extract: MovePageExtract, move: PokeapiMove, version_group_ids: Mapping[str, str]
) -> list[CrossCheckDifference]:
    bulbapedia = {build_machine_key(fact, version_group_ids) for fact in extract.machines}
    only_bulbapedia = [
        (format_machine(key), NO_VALUE) for key in sorted(bulbapedia - move.machines)
    ]
    only_pokeapi = [(NO_VALUE, format_machine(key)) for key in sorted(move.machines - bulbapedia)]
    return [
        CrossCheckDifference(
            move_id=extract.id, field="machine", bulbapedia_value=ours, pokeapi_value=theirs
        )
        for ours, theirs in (*only_bulbapedia, *only_pokeapi)
    ]


def cross_check(
    extract: MovePageExtract, index: PokeapiIndex, version_group_ids: Mapping[str, str]
) -> tuple[CrossCheckDifference, ...]:
    """Compare 1 extract with its PokeAPI move. A move without a row gives no difference.

    `version_group_ids` maps A-Box version group labels to node ids. Crit stage and hits
    are compared only for a move with a meta row.
    """
    move = index.moves.get(extract.id)
    if move is None:
        return ()
    pairs = {
        "power": (format_value(extract.power), format_value(move.power)),
        "accuracy": (format_value(extract.accuracy), format_value(move.accuracy)),
        "pp": (str(extract.pp), format_value(move.pp)),
        "priority": (str(extract.priority), str(move.priority)),
        "type": (build_node_id(extract.type_name), move.type_id),
        "damage_class": (build_node_id(extract.damage_class_name), move.damage_class_id),
    }
    if move.has_meta:
        pairs["crit_stage"] = (format_value(extract.crit_stage), format_value(move.crit_stage))
        pairs["hits"] = (
            format_hits(extract.min_hits, extract.max_hits),
            format_hits(move.min_hits, move.max_hits),
        )
    differences = [
        CrossCheckDifference(
            move_id=extract.id, field=field, bulbapedia_value=ours, pokeapi_value=theirs
        )
        for field, (ours, theirs) in pairs.items()
        if ours != theirs
    ]
    return (*differences, *list_machine_differences(extract, move, version_group_ids))


def list_unmatched_pokeapi_moves(index: PokeapiIndex, move_ids: Iterable[str]) -> tuple[str, ...]:
    """Return the PokeAPI identifiers of the indexed moves no in-scope page matches."""
    matched = set(move_ids)
    return tuple(
        sorted(move.identifier for move_id, move in index.moves.items() if move_id not in matched)
    )
