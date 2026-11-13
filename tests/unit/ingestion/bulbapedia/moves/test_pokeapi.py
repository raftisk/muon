from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.moves.page import MachineFact, MovePageExtract
from muon.ingestion.bulbapedia.moves.pokeapi import (
    MachineKey,
    PokeapiIndex,
    build_pokeapi_index,
    cross_check,
    list_unmatched_pokeapi_moves,
    read_effect_text,
)
from muon.ingestion.pokeapi.snapshot import (
    REPOSITORY,
    SnapshotError,
    SnapshotManifest,
    SnapshotStore,
    compute_sha256,
)

MOVES_HEADER = "id,identifier,generation_id,type_id,power,pp,accuracy,priority,target_id,damage_class_id,effect_id,effect_chance,contest_type_id,contest_effect_id,super_contest_effect_id"  # noqa: E501
META_HEADER = "move_id,meta_category_id,meta_ailment_id,min_hits,max_hits,min_turns,max_turns,drain,healing,crit_rate,ailment_chance,flinch_chance,stat_chance"  # noqa: E501

TABLES: Mapping[str, tuple[str, ...]] = {
    "moves.csv": (
        MOVES_HEADER,
        "11,vice-grip,1,1,55,30,100,0,10,2,1,,5,1,5",
        "247,shadow-ball,2,8,80,15,100,0,10,3,73,20,4,22,17",
        "251,beat-up,2,17,,10,100,0,10,2,155,,4,14,7",
        "400,stone-edge,4,6,100,5,80,0,10,2,44,,,,",
        "827,dire-claw,8,4,80,15,100,0,10,2,,,,,",
        "10001,shadow-rush,3,10002,55,,100,0,10,2,10001,,,,",
    ),
    "move_meta.csv": (
        META_HEADER,
        "11,0,0,,,,,0,0,0,0,0,0",
        "247,6,0,,,,,0,0,0,0,0,20",
        "251,0,0,6,6,,,0,0,0,0,0,0",
        "400,0,0,,,,,0,0,1,0,0,0",
    ),
    "move_effect_prose.csv": (
        "move_effect_id,local_language_id,short_effect,effect",
        "1,9,Inflicts regular damage with no additional effect.,long",
        "73,5,Peut baisser la Défense Spéciale.,long",
        "73,9,Has a chance to lower the target’s Special Defense by one stage.,long",
        "155,5,Frappe avec chaque Pokémon.,long",
        '44,9,"Has an increased chance for a critical hit.",long',
    ),
    "machines.csv": (
        "machine_number,version_group_id,item_id,move_id",
        "30,3,334,247",
        "133,20,1221,247",
        "133,21,1221,247",
        "30,32,334,247",
        "105,20,1300,11",
    ),
    "items.csv": (
        "id,identifier,category_id,cost,fling_power,fling_effect_id",
        "334,tm30,37,3000,,",
        "1221,tr33,37,0,,",
        "1300,hm05,37,0,,",
    ),
    "version_groups.csv": (
        "id,identifier,generation_id,order",
        "3,gold-silver,2,3",
        "20,sword-shield,8,20",
        "21,the-isle-of-armor,8,21",
        "32,champions,9,32",
    ),
    "types.csv": (
        "id,identifier,generation_id,damage_class_id",
        "1,normal,1,2",
        "4,poison,1,2",
        "6,rock,1,2",
        "8,ghost,1,2",
        "17,dark,2,3",
    ),
    "move_damage_classes.csv": ("id,identifier", "1,status", "2,physical", "3,special"),
}

VERSION_GROUP_IDS = {"Gold-Silver": "gold-silver", "Sword-Shield": "sword-shield"}


def write_snapshot(root: Path, tables: Mapping[str, tuple[str, ...]]) -> SnapshotStore:
    store = SnapshotStore(root)
    contents = {name: ("\n".join(lines) + "\n").encode() for name, lines in tables.items()}
    for name, content in contents.items():
        path = store.build_file_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    store.write_manifest(
        SnapshotManifest(
            repository=REPOSITORY,
            commit="a" * 40,
            files={name: compute_sha256(content) for name, content in contents.items()},
        )
    )
    return store


@pytest.fixture
def index(tmp_path: Path) -> PokeapiIndex:
    return build_pokeapi_index(write_snapshot(tmp_path, TABLES))


def build_extract(move_id: str, **values: object) -> MovePageExtract:
    base: dict[str, object] = {
        "title": f"{move_id} (move)",
        "page_id": 1,
        "revision_id": 1,
        "id": move_id,
        "name": move_id,
        "type_name": "Ghost",
        "damage_class_name": "Special",
        "generation_label": "Generation II",
        "pp": 15,
        "priority": 0,
        "target": "any_adjacent",
        "power": 80,
        "accuracy": 100,
    }
    return MovePageExtract.model_validate({**base, **values})


def test_effect_text_is_the_english_short_effect(index: PokeapiIndex) -> None:
    expected = "Has a chance to lower the target’s Special Defense by one stage."

    assert read_effect_text(index, "shadow-ball") == expected


def test_vise_grip_reads_the_vice_grip_row(index: PokeapiIndex) -> None:
    assert index.moves["vise-grip"].identifier == "vice-grip"
    assert (
        read_effect_text(index, "vise-grip") == "Inflicts regular damage with no additional effect."
    )


@pytest.mark.parametrize("move_id", ["nihil-light", "beat-up", "dire-claw"])
def test_no_row_or_no_english_text_gives_no_effect_text(index: PokeapiIndex, move_id: str) -> None:
    assert read_effect_text(index, move_id) is None


def test_only_regular_moves_enter_the_index(index: PokeapiIndex) -> None:
    assert "shadow-rush" not in index.moves


def test_machines_number_from_the_item_and_fold_dlc_groups(index: PokeapiIndex) -> None:
    assert index.moves["shadow-ball"].machines == {
        MachineKey("TM", 30, "gold-silver"),
        MachineKey("TR", 33, "sword-shield"),
    }
    assert index.moves["vise-grip"].machines == {MachineKey("HM", 5, "sword-shield")}
    assert index.skipped_machine_count == 1


def test_matching_values_give_no_difference(index: PokeapiIndex) -> None:
    extract = build_extract(
        "shadow-ball",
        machines=(
            MachineFact(
                kind="TM", number_text="30", version_group_label="Gold-Silver", field="tm2"
            ),
            MachineFact(
                kind="TR", number_text="33", version_group_label="Sword-Shield", field="tr8"
            ),
        ),
    )

    assert cross_check(extract, index, VERSION_GROUP_IDS) == ()


@pytest.mark.parametrize(
    ("values", "field", "ours", "theirs"),
    [
        ({"power": 90}, "power", "90", "80"),
        ({"accuracy": None}, "accuracy", "", "100"),
        ({"pp": 10}, "pp", "10", "15"),
        ({"priority": 1}, "priority", "1", "0"),
        ({"type_name": "Dark"}, "type", "dark", "ghost"),
        ({"damage_class_name": "Physical"}, "damage_class", "physical", "special"),
        ({"crit_stage": 3}, "crit_stage", "3", ""),
        ({"min_hits": 2, "max_hits": 5}, "hits", "2-5", ""),
    ],
)
def test_each_field_reports_its_difference(
    index: PokeapiIndex, values: dict[str, object], field: str, ours: str, theirs: str
) -> None:
    extract = build_extract("shadow-ball", **values)

    differences = [
        d for d in cross_check(extract, index, VERSION_GROUP_IDS) if d.field != "machine"
    ]

    assert [(d.field, d.bulbapedia_value, d.pokeapi_value) for d in differences] == [
        (field, ours, theirs)
    ]


def test_crit_rate_1_is_stage_1(index: PokeapiIndex) -> None:
    extract = build_extract(
        "stone-edge",
        type_name="Rock",
        damage_class_name="Physical",
        power=100,
        accuracy=80,
        pp=5,
        crit_stage=1,
    )

    assert cross_check(extract, index, VERSION_GROUP_IDS) == ()


def test_a_move_without_meta_compares_no_crit_or_hits(index: PokeapiIndex) -> None:
    extract = build_extract(
        "dire-claw", type_name="Poison", damage_class_name="Physical", crit_stage=1
    )

    assert cross_check(extract, index, VERSION_GROUP_IDS) == ()


def test_machines_only_one_source_has(index: PokeapiIndex) -> None:
    extract = build_extract(
        "shadow-ball",
        machines=(
            MachineFact(
                kind="TM", number_text="30", version_group_label="Gold-Silver", field="tm2"
            ),
            MachineFact(
                kind="TM", number_text="31", version_group_label="Sword-Shield", field="tm8"
            ),
        ),
    )

    differences = cross_check(extract, index, VERSION_GROUP_IDS)

    assert [(d.field, d.bulbapedia_value, d.pokeapi_value) for d in differences] == [
        ("machine", "TM31 sword-shield", ""),
        ("machine", "", "TR33 sword-shield"),
    ]


def test_a_move_without_a_row_gives_no_difference(index: PokeapiIndex) -> None:
    assert cross_check(build_extract("nihil-light"), index, VERSION_GROUP_IDS) == ()


def test_unmatched_pokeapi_moves_use_pokeapi_identifiers(index: PokeapiIndex) -> None:
    unmatched = list_unmatched_pokeapi_moves(index, ["shadow-ball", "beat-up", "stone-edge"])

    assert unmatched == ("dire-claw", "vice-grip")


@pytest.fixture(scope="session")
def pinned_index() -> PokeapiIndex:
    try:
        return build_pokeapi_index(SnapshotStore())
    except SnapshotError as error:
        pytest.skip(f"the PokeAPI snapshot is not on disk: {error}")


@pytest.mark.pages
def test_pinned_effect_texts(pinned_index: PokeapiIndex) -> None:
    expected = "Has a chance to lower the target’s Special Defense by one stage."

    assert read_effect_text(pinned_index, "shadow-ball") == expected
    assert read_effect_text(pinned_index, "vise-grip") is not None
    assert read_effect_text(pinned_index, "nihil-light") is None
    assert read_effect_text(pinned_index, "dire-claw") is None


@pytest.mark.pages
@pytest.mark.parametrize(
    ("name", "ours", "theirs"), [("Beat Up", "1-6", "6-6"), ("Triple Kick", "1-3", "3-3")]
)
def test_pinned_hit_differences(
    read_extract: Callable[[str], MovePageExtract],
    pinned_index: PokeapiIndex,
    name: str,
    ours: str,
    theirs: str,
) -> None:
    extract = read_extract(name)
    version_group_ids = {
        fact.version_group_label: fact.version_group_label.lower() for fact in extract.machines
    }
    hits = [
        (d.bulbapedia_value, d.pokeapi_value)
        for d in cross_check(
            extract.model_copy(update={"machines": ()}), pinned_index, version_group_ids
        )
        if d.field == "hits"
    ]

    assert hits == [(ours, theirs)]
