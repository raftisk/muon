"""The move parse run on a synthetic page store and a tmp PokeAPI snapshot.

The parser tables name real moves, which a 3-page store lacks, so every test but the table
test empties them.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.moves import page as move_page
from muon.ingestion.bulbapedia.moves import run as move_run
from muon.ingestion.bulbapedia.moves.batches import EFFECT_TEXT
from muon.ingestion.bulbapedia.moves.page import MoveNoteKind, TableError
from muon.ingestion.bulbapedia.moves.run import (
    PAGE_OUTPUT_DIR,
    REPORT_FILE_NAME,
    MachineConflictError,
    MoveParseRequest,
    MoveParseRun,
    MoveRunInputs,
    PageStatus,
    ParseRunError,
    parse_move_pages,
    serialize_output,
    write_outputs,
)
from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotStore
from tests.unit.ingestion.bulbapedia.moves.test_pokeapi import TABLES, write_snapshot

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
TIMESTAMP = datetime(2026, 11, 13, tzinfo=UTC)
SHADOW_BALL = "Shadow Ball (move)"
Z_MOVE = "Never-Ending Nightmare (move)"
NO_INFOBOX = "Excavate (move)"
BROKEN = "Broken (move)"
TWIN = "Twin Ball (move)"
PAGE_IDS = {SHADOW_BALL: 3858, Z_MOVE: 3901, NO_INFOBOX: 3902, BROKEN: 3903, TWIN: 3904}


def move_wikitext(name: str, type_name: str = "Ghost", extra: str = "") -> str:
    return f"""{{{{MoveInfobox
|name={name}
|type={type_name}
|damagecategory=Special
|basepp=15
|power=80
|accuracy=100
|gen=II
|target=anyadjacent
|protect=yes
|mirrormove=yes
|tm2=yes
|tm#2=30
}}}}
'''{name}''' is a move.
{extra}
"""


def default_wikitexts() -> dict[str, str]:
    return {
        SHADOW_BALL: move_wikitext("Shadow Ball", extra="[[Category:Ball and bomb moves]]"),
        Z_MOVE: move_wikitext("Never-Ending Nightmare", extra="[[Category:Z-Moves]]"),
        NO_INFOBOX: "'''Excavate''' is a move in Pokémon Mystery Dungeon.",
    }


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS[title],
        revision_id=PAGE_IDS[title] * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type="move",
        wikitext=wikitext,
    )


def build_store(root: Path, wikitexts: dict[str, str]) -> PageStore:
    store = PageStore(root)
    pages = [build_page(title, wikitext) for title, wikitext in wikitexts.items()]
    for page in pages:
        store.write_page(page)
    store.write_manifest(Manifest().replace_pins("move", [page.pin for page in pages]))
    return store


@pytest.fixture
def no_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(move_page, "PARSER_TABLES", {})
    monkeypatch.setattr(move_page, "MOVE_HITS", {})


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotStore:
    return write_snapshot(tmp_path / "pokeapi", TABLES)


def run_parse(
    store: PageStore, snapshot: SnapshotStore, titles: tuple[str, ...] = ()
) -> MoveParseRun:
    return parse_move_pages(MoveRunInputs(MoveParseRequest(titles), store, snapshot, ONTOLOGY_DIR))


@pytest.mark.usefixtures("no_tables")
def test_every_pin_gets_a_result_and_outputs_are_written(
    tmp_path: Path, snapshot: SnapshotStore
) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    run = run_parse(store, snapshot)
    write_outputs(run, tmp_path / "parsed", is_full_run=True)

    assert {r.title: (r.status, r.reason) for r in run.report.pages} == {
        SHADOW_BALL: (PageStatus.PARSED, None),
        Z_MOVE: (PageStatus.SKIPPED, "z_move"),
        NO_INFOBOX: (PageStatus.SKIPPED, "no_infobox"),
    }
    assert (run.report.pages_parsed, run.report.pages_skipped, run.report.pages_failed) == (1, 2, 0)
    assert run.report.skipped_per_reason["z_move"] == 1
    assert run.report.moves_per_flag == {"ballistic": 1}
    assert run.report.batch_counts.owner_nodes == {"Move": 1, "Machine": 2}
    assert run.report.snapshot_commit == "a" * 40
    assert sorted(path.name for path in (tmp_path / "parsed" / PAGE_OUTPUT_DIR).iterdir()) == [
        "3858.json"
    ]
    assert (tmp_path / "parsed" / REPORT_FILE_NAME).exists()


@pytest.mark.usefixtures("no_tables")
def test_a_rerun_is_byte_identical(tmp_path: Path, snapshot: SnapshotStore) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    write_outputs(run_parse(store, snapshot), tmp_path / "first", is_full_run=True)
    write_outputs(run_parse(store, snapshot), tmp_path / "second", is_full_run=True)

    first = sorted(p.relative_to(tmp_path / "first") for p in (tmp_path / "first").rglob("*.json"))
    assert first == sorted(
        p.relative_to(tmp_path / "second") for p in (tmp_path / "second").rglob("*.json")
    )
    for relative in first:
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()


@pytest.mark.usefixtures("no_tables")
def test_a_title_run_writes_that_page_only(tmp_path: Path, snapshot: SnapshotStore) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    run = run_parse(store, snapshot, (Z_MOVE,))

    assert [r.title for r in run.report.pages] == [Z_MOVE]
    assert run.outputs == ()


@pytest.mark.usefixtures("no_tables")
def test_an_unknown_title_fails_before_output(tmp_path: Path, snapshot: SnapshotStore) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    with pytest.raises(ParseRunError, match="Tackle"):
        run_parse(store, snapshot, ("Tackle (move)",))


@pytest.mark.usefixtures("no_tables")
def test_a_full_run_deletes_stale_outputs(tmp_path: Path, snapshot: SnapshotStore) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())
    stale = tmp_path / "parsed" / PAGE_OUTPUT_DIR / "999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    write_outputs(run_parse(store, snapshot, (SHADOW_BALL,)), tmp_path / "parsed", False)
    assert stale.exists()
    write_outputs(run_parse(store, snapshot), tmp_path / "parsed", is_full_run=True)

    assert not stale.exists()


@pytest.mark.usefixtures("no_tables")
def test_the_cross_check_changes_no_batch(
    tmp_path: Path, snapshot: SnapshotStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())
    checked = run_parse(store, snapshot)
    monkeypatch.setattr(move_run, "cross_check", lambda *_: ())

    unchecked = run_parse(store, snapshot)

    assert checked.report.cross_check
    assert unchecked.report.cross_check == ()
    assert [serialize_output(o) for o in checked.outputs] == [
        serialize_output(o) for o in unchecked.outputs
    ]


@pytest.mark.usefixtures("no_tables")
def test_effect_text_comes_from_the_snapshot(tmp_path: Path, snapshot: SnapshotStore) -> None:
    twin_wikitext = move_wikitext("Twin Ball").replace("tm#2=30", "tm#2=31")
    store = build_store(tmp_path / "data", {**default_wikitexts(), TWIN: twin_wikitext})

    run = run_parse(store, snapshot)

    moves = {node.id: node for o in run.outputs for node in o.batch.owner_nodes}
    twin = next(o for o in run.outputs if o.page.title == TWIN)
    assert EFFECT_TEXT in moves["shadow-ball"].properties
    assert EFFECT_TEXT not in moves["twin-ball"].properties
    assert [(n.kind, n.detail) for n in twin.extract.notes] == [
        (MoveNoteKind.NO_POKEAPI_MATCH, "no PokeAPI move")
    ]
    assert run.report.moves_without_pokeapi_row == ("twin-ball",)


@pytest.mark.usefixtures("no_tables")
def test_a_page_the_a_box_cannot_resolve_fails_alone(
    tmp_path: Path, snapshot: SnapshotStore
) -> None:
    wikitexts = {**default_wikitexts(), BROKEN: move_wikitext("Broken", type_name="Cosmic")}
    store = build_store(tmp_path / "data", wikitexts)

    run = run_parse(store, snapshot)

    failed = [r for r in run.report.pages if r.status is PageStatus.FAILED]
    assert [r.title for r in failed] == [BROKEN]
    assert "Cosmic" in (failed[0].reason or "")
    assert [o.page.title for o in run.outputs] == [SHADOW_BALL]


@pytest.mark.usefixtures("no_tables")
def test_2_pages_with_the_same_machine_stop_the_run(
    tmp_path: Path, snapshot: SnapshotStore
) -> None:
    wikitexts = {**default_wikitexts(), TWIN: move_wikitext("Twin Ball")}
    store = build_store(tmp_path / "data", wikitexts)

    with pytest.raises(
        MachineConflictError, match=re.escape(f"tm30-gold-silver from {SHADOW_BALL} and {TWIN}")
    ):
        run_parse(store, snapshot)


@pytest.mark.usefixtures("no_tables")
def test_a_missing_snapshot_stops_the_run(tmp_path: Path) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    with pytest.raises(SnapshotError, match="--update"):
        run_parse(store, SnapshotStore(tmp_path / "empty"))


def test_tables_naming_moves_out_of_scope_stop_the_run(
    tmp_path: Path, snapshot: SnapshotStore
) -> None:
    store = build_store(tmp_path / "data", default_wikitexts())

    with pytest.raises(TableError, match="FLAG_OVERRIDES names dig"):
        run_parse(store, snapshot)
