import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.abilities.run import (
    PAGE_OUTPUT_DIR,
    REPORT_FILE_NAME,
    AbilityParseRequest,
    AbilityParseRun,
    AbilityRunInputs,
    ParseRunError,
    parse_ability_pages,
    write_outputs,
)
from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.bulbapedia.runs import PageStatus
from muon.ingestion.pokeapi.snapshot import SnapshotError, SnapshotStore
from tests.unit.ingestion.bulbapedia.abilities.test_crosscheck import write_species_output
from tests.unit.ingestion.bulbapedia.abilities.test_page import build_ability_wikitext
from tests.unit.ingestion.bulbapedia.abilities.test_pokeapi import TABLES
from tests.unit.ingestion.bulbapedia.moves.test_pokeapi import write_snapshot

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
TIMESTAMP = datetime(2026, 11, 14, tzinfo=UTC)
TEST_GUARD = "Test Guard (Ability)"
NO_INFOBOX = "Stench Guard (Ability)"
PAGE_IDS = {TEST_GUARD: 501, NO_INFOBOX: 502}


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS[title],
        revision_id=PAGE_IDS[title] * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type="ability",
        wikitext=wikitext,
    )


def build_store(root: Path) -> PageStore:
    store = PageStore(root)
    pages = [
        build_page(TEST_GUARD, build_ability_wikitext()),
        build_page(NO_INFOBOX, "'''Stench Guard''' is an Ability.\n"),
    ]
    for page in pages:
        store.write_page(page)
    store.write_manifest(Manifest().replace_pins("ability", [page.pin for page in pages]))
    return store


def run_parse(tmp_path: Path, titles: tuple[str, ...] = ()) -> AbilityParseRun:
    species_dir = tmp_path / "species"
    write_species_output(species_dir, "Pikachu (Pokémon)", ("pikachu", "test-guard"))
    inputs = AbilityRunInputs(
        AbilityParseRequest(titles),
        build_store(tmp_path / "data"),
        write_snapshot(tmp_path / "pokeapi", TABLES),
        ONTOLOGY_DIR,
        species_dir,
    )
    return parse_ability_pages(inputs)


def test_every_pin_gets_a_result_and_the_report_counts_them(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert [(page.title, page.status) for page in run.report.pages] == [
        (NO_INFOBOX, PageStatus.SKIPPED),
        (TEST_GUARD, PageStatus.PARSED),
    ]
    assert run.report.skipped_per_reason == {"no_infobox": 1}
    assert run.report.chunks_per_aspect == {"effect": 1, "flavor": 3}
    # No species page is pinned, so the {{p|Pikachu}} mention resolves to nothing.
    assert [note.kind.value for note in run.report.notes] == [
        "no_pokeapi_match",
        "unresolved_mention",
    ]
    assert run.report.abilities_per_category == {"Abilities that guard": 1}


def test_the_cross_check_reports_the_holders_the_species_output_lacks(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert not run.report.species_output_missing
    assert [(d.species_id, d.side.value) for d in run.report.holder_differences] == [
        ("raichu", "table_only")
    ]
    assert run.report.unowned_species_stubs == ()


def test_a_missing_species_output_skips_the_cross_check(tmp_path: Path) -> None:
    inputs = AbilityRunInputs(
        AbilityParseRequest(),
        build_store(tmp_path / "data"),
        write_snapshot(tmp_path / "pokeapi", TABLES),
        ONTOLOGY_DIR,
        tmp_path / "no-species",
    )

    report = parse_ability_pages(inputs).report

    assert report.species_output_missing
    assert report.holder_differences == ()


def test_two_runs_give_identical_files_and_a_full_run_deletes_stale_ones(tmp_path: Path) -> None:
    root = tmp_path / "parsed"
    stale = root / PAGE_OUTPUT_DIR / "999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    write_outputs(run_parse(tmp_path), root, is_full_run=True)
    first = {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()}
    report = (root / REPORT_FILE_NAME).read_bytes()
    write_outputs(run_parse(tmp_path), root, is_full_run=True)

    assert not stale.exists()
    assert list(first) == [f"{PAGE_IDS[TEST_GUARD]}.json"]
    assert {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()} == first
    assert (root / REPORT_FILE_NAME).read_bytes() == report
    assert json.loads(first[f"{PAGE_IDS[TEST_GUARD]}.json"])["extract"]["id"] == "test-guard"


def test_a_title_that_is_not_pinned_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ParseRunError, match="Overgrow"):
        run_parse(tmp_path, ("Overgrow (Ability)",))


def test_a_missing_prose_file_stops_the_run_with_the_fetch_hint(tmp_path: Path) -> None:
    snapshot = write_snapshot(tmp_path / "pokeapi", TABLES)
    snapshot.build_file_path("ability_prose.csv").unlink()
    inputs = AbilityRunInputs(
        AbilityParseRequest(), build_store(tmp_path / "data"), snapshot, ONTOLOGY_DIR
    )

    with pytest.raises(SnapshotError, match="muon fetch pokeapi"):
        parse_ability_pages(inputs)


def test_a_missing_snapshot_needs_no_page_to_fail(tmp_path: Path) -> None:
    inputs = AbilityRunInputs(
        AbilityParseRequest(),
        build_store(tmp_path / "data"),
        SnapshotStore(tmp_path / "empty"),
        ONTOLOGY_DIR,
    )

    with pytest.raises(SnapshotError, match="--update"):
        parse_ability_pages(inputs)
