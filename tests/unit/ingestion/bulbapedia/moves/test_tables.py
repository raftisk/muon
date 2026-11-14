from collections.abc import Callable
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.batching import UNIVERSE
from muon.ingestion.bulbapedia.moves import batches as move_batches
from muon.ingestion.bulbapedia.moves.batches import check_type_overrides
from muon.ingestion.bulbapedia.moves.page import (
    CRIT_OVERRIDES,
    FLAG_OVERRIDES,
    MOVE_HITS,
    MOVE_TURNS,
    MULTISTRIKE_CATEGORY,
    TYPE_OVERRIDES,
    MoveNoteKind,
    MovePageExtract,
    TableError,
    check_tables,
    read_crit_stage,
    read_flags,
    read_range,
    read_type,
)
from muon.ingestion.seed import read_abox
from muon.ontology.model import load_ontology_model

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
HIGH_CRIT = "Moves with a high critical-hit ratio"
PERFECT_CRIT = "Moves with a perfect critical hit chance"


def build_extract(move_id: str, categories: tuple[str, ...] = ()) -> MovePageExtract:
    return MovePageExtract(
        title=f"{move_id} (move)",
        page_id=1,
        revision_id=1,
        id=move_id,
        name=move_id,
        type_name="Normal",
        damage_class_name="Physical",
        generation_label="Generation I",
        pp=10,
        priority=0,
        target="any_adjacent",
        categories=categories,
    )


def build_table_extracts() -> list[MovePageExtract]:
    """1 extract per table row, the `MOVE_HITS` rows in the multistrike category."""
    ids = set(FLAG_OVERRIDES) | set(CRIT_OVERRIDES) | set(MOVE_TURNS) | set(TYPE_OVERRIDES)
    return [build_extract(move_id) for move_id in sorted(ids - set(MOVE_HITS))] + [
        build_extract(move_id, (MULTISTRIKE_CATEGORY,)) for move_id in sorted(MOVE_HITS)
    ]


def test_flags_come_from_fields_and_categories() -> None:
    infobox = {"touches": "yes", "sound": "no", "magiccoat": "Yes", "snatch": "no"}

    flags, notes = read_flags(infobox, frozenset({"Punching moves", "Wind moves"}), "testmove")

    assert flags == {"contact", "reflectable", "punch", "wind"}
    assert notes == ()


def test_both_two_turn_categories_give_1_flag() -> None:
    categories = frozenset({"Moves with a charging turn", "Moves with a semi-invulnerable turn"})

    flags, _ = read_flags({}, categories, "fly")

    assert flags == {"two-turn"}


def test_an_override_adds_and_removes_flags_with_a_note() -> None:
    added, added_notes = read_flags({}, frozenset(), "dig")
    removed, _ = read_flags({}, frozenset({"HP-draining moves"}), "dark-void")

    assert added == {"two-turn"}
    assert [note.kind for note in added_notes] == [MoveNoteKind.OVERRIDE_APPLIED]
    assert removed == frozenset()


@pytest.mark.parametrize(
    ("categories", "move_id", "expected"),
    [
        (frozenset(), "tackle", None),
        (frozenset({HIGH_CRIT}), "slash", 1),
        (frozenset({PERFECT_CRIT}), "wicked-blow", 3),
        (frozenset(), "stone-edge", 1),
        (frozenset(), "frost-breath", 3),
        (frozenset({HIGH_CRIT}), "dragon-claw", None),
        (frozenset({PERFECT_CRIT}), "zippy-zap", None),
    ],
)
def test_crit_stage(categories: frozenset[str], move_id: str, expected: int | None) -> None:
    stage, notes = read_crit_stage(categories, move_id)

    assert stage == expected
    assert bool(notes) == (move_id in CRIT_OVERRIDES)


def test_range_reads_a_row_or_nothing() -> None:
    assert read_range(MOVE_HITS, "population-bomb") == (1, 10)
    assert read_range(MOVE_HITS, "tackle") == (None, None)


def test_tables_pass_when_every_row_is_in_scope() -> None:
    check_tables(build_table_extracts())


def test_a_row_without_an_in_scope_move_fails() -> None:
    extracts = [extract for extract in build_table_extracts() if extract.id != "wish"]

    with pytest.raises(TableError, match="FLAG_OVERRIDES names wish"):
        check_tables(extracts)


def test_a_multistrike_move_without_a_hits_row_fails() -> None:
    extracts = [*build_table_extracts(), build_extract("new-strike", (MULTISTRIKE_CATEGORY,))]

    with pytest.raises(TableError, match="MOVE_HITS lacks new-strike"):
        check_tables(extracts)


def test_a_hits_row_outside_the_multistrike_category_fails() -> None:
    extracts = [
        build_extract(extract.id) if extract.id == "beat-up" else extract
        for extract in build_table_extracts()
    ]

    with pytest.raises(TableError, match="MOVE_HITS names beat-up"):
        check_tables(extracts)


@pytest.mark.pages
@pytest.mark.parametrize(
    ("name", "flags", "hits"),
    [
        ("Bullet Seed", {"ballistic", "multihit"}, (2, 5)),
        ("Brave Bird", {"contact", "recoil"}, (None, None)),
        ("Population Bomb", {"contact", "multihit", "slicing"}, (1, 10)),
        ("Triple Kick", {"contact", "multihit"}, (1, 3)),
        ("Beat Up", {"multihit"}, (1, 6)),
        ("Shadow Ball", {"ballistic"}, (None, None)),
    ],
)
def test_pinned_flags_and_hits(
    read_extract: Callable[[str], MovePageExtract],
    name: str,
    flags: set[str],
    hits: tuple[int | None, int | None],
) -> None:
    extract = read_extract(name)

    assert set(extract.flags) == flags
    assert (extract.min_hits, extract.max_hits) == hits


@pytest.mark.pages
@pytest.mark.parametrize("name", ["Dig", "Dive", "Sky Attack", "Freeze Shock", "Ice Burn"])
def test_two_turn_overrides(read_extract: Callable[[str], MovePageExtract], name: str) -> None:
    extract = read_extract(name)

    assert "two-turn" in extract.flags
    assert MoveNoteKind.OVERRIDE_APPLIED in [note.kind for note in extract.notes]


@pytest.mark.pages
def test_dark_void_drains_nothing(read_extract: Callable[[str], MovePageExtract]) -> None:
    assert "drain" not in read_extract("Dark Void").flags


@pytest.mark.pages
@pytest.mark.parametrize("name", ["Explosion", "Self-Destruct", "Misty Explosion", "Mind Blown"])
def test_explosive_moves(read_extract: Callable[[str], MovePageExtract], name: str) -> None:
    assert "explosive" in read_extract(name).flags


@pytest.mark.pages
@pytest.mark.parametrize("name", ["Swallow", "Wish", "Heal Pulse"])
def test_healing_overrides(read_extract: Callable[[str], MovePageExtract], name: str) -> None:
    assert "healing" in read_extract(name).flags


@pytest.mark.pages
def test_outrage_locks_its_user_in(read_extract: Callable[[str], MovePageExtract]) -> None:
    extract = read_extract("Outrage")

    assert (extract.min_turns, extract.max_turns) == (2, 3)
    assert MoveNoteKind.TABLE_ROW_APPLIED in [note.kind for note in extract.notes]


@pytest.mark.pages
@pytest.mark.parametrize(
    ("name", "stage"), [("Stone Edge", 1), ("Aqua Cutter", 1), ("Frost Breath", 3)]
)
def test_pinned_crit_stages(
    read_extract: Callable[[str], MovePageExtract], name: str, stage: int
) -> None:
    assert read_extract(name).crit_stage == stage


def test_type_comes_from_the_infobox_without_a_row() -> None:
    assert read_type("Tackle (move)", {"type": "Normal"}, "tackle") == ("Normal", ())


def test_a_type_row_wins_over_the_infobox_with_a_note() -> None:
    type_name, notes = read_type("Growth (move)", {"type": "Grass"}, "growth")

    assert type_name == "Normal"
    assert [(note.kind, note.detail) for note in notes] == [
        (MoveNoteKind.TABLE_ROW_APPLIED, "TYPE_OVERRIDES Grass to Normal")
    ]


def test_a_type_row_without_an_in_scope_move_fails() -> None:
    extracts = [e for e in build_table_extracts() if e.id != "snap-trap"]

    with pytest.raises(TableError, match="TYPE_OVERRIDES names snap-trap"):
        check_tables(extracts)


def test_type_rows_name_abox_types() -> None:
    model = load_ontology_model(ONTOLOGY_DIR, UNIVERSE)

    check_type_overrides(read_abox(model, ONTOLOGY_DIR))


def test_a_type_row_outside_the_abox_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(move_batches, "TYPE_OVERRIDES", {"growth": "Shadow"})
    model = load_ontology_model(ONTOLOGY_DIR, UNIVERSE)

    with pytest.raises(TableError, match="growth 'Shadow', which is no A-Box Type"):
        check_type_overrides(read_abox(model, ONTOLOGY_DIR))


@pytest.mark.pages
@pytest.mark.parametrize(("name", "expected"), [("Growth", "Normal"), ("Snap Trap", "Grass")])
def test_pinned_type_overrides(
    read_extract: Callable[[str], MovePageExtract], name: str, expected: str
) -> None:
    assert read_extract(name).type_name == expected
