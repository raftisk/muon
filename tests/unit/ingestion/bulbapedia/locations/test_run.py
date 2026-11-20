import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.locations.run import (
    CANDIDATES_FILE_NAME,
    COVERAGE_FILE_NAME,
    PAGE_OUTPUT_DIR,
    REPORT_FILE_NAME,
    IdOverlapError,
    LocationParseRequest,
    LocationParseRun,
    LocationRunInputs,
    ParseRunError,
    parse_location_pages,
    write_outputs,
)
from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.bulbapedia.runs import PageStatus
from muon.ingestion.pokeapi.snapshot import SnapshotManifest, SnapshotStore, compute_sha256

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
TIMESTAMP = datetime(2026, 11, 20, tzinfo=UTC)
GAME = "[[Category:Emerald locations]]"
SOOTOPOLIS = "Sootopolis City"
ROUTE = "Hoenn Route 126"
KANTO = "Kanto"
BADGE = "Badge"
MT_CORONET = "Mt. Coronet"
CYNDAQUIL = "Cyndaquil (Pokémon)"
POOCHYENA = "Poochyena (Pokémon)"
PAGE_IDS = {SOOTOPOLIS: 701, ROUTE: 702, KANTO: 703, BADGE: 704, MT_CORONET: 705}

SOOTOPOLIS_WIKITEXT = (
    "{{Town infobox|region=Hoenn|generation=3|north=Cave of Origin|south=Route 126"
    "|regionsouth=Hoenn}}\n==Places of interest==\n===Gym===\n{{main|Sootopolis Gym}}\n"
    "==Items==\n===1F===\n{{Itemlist|Moon Stone|Floor|Sw=yes|Sh=yes}}\n"
    "===Hidden items===\n{{Itemlist|Potion|Hidden|Sw=yes}}\n"
    "{{Itemlist|Potion|Again|Sw=yes}}\n"
    "==Pokémon==\n===1F===\n"
    "{{catch/entry3|261|Poochyena|yes|no|yes|Grass|3-4|15%}}\n"
    "{{catch/entry3|999|Ghostmon|yes|no|no|Grass|3|5%}}\n" + GAME
)
ROUTE_WIKITEXT = (
    "{{Route infobox|region=Hoenn|generation=3|north=Sootopolis City|surf=yes}}\n" + GAME
)
KANTO_WIKITEXT = "{{RegionInfobox|name=Kanto|region=Kanto}}\n[[Category:Emerald locations]]\n"


def build_page(title: str, wikitext: str, page_type: str = "location") -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS.get(title, 800 + len(title)),
        revision_id=PAGE_IDS.get(title, 800 + len(title)) * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type=page_type,
        wikitext=wikitext,
    )


def build_store(root: Path) -> PageStore:
    store = PageStore(root)
    pages = [
        build_page(SOOTOPOLIS, SOOTOPOLIS_WIKITEXT),
        build_page(ROUTE, ROUTE_WIKITEXT),
        build_page(KANTO, KANTO_WIKITEXT),
        build_page(BADGE, "{{ItemInfobox/head|name=Badge}}\n" + GAME),
        build_page(MT_CORONET, "#REDIRECT [[Mount Coronet]]"),
    ]
    species = [
        build_page(CYNDAQUIL, "{{Pokémon Infobox|name=Cyndaquil}}", "species"),
        build_page(POOCHYENA, "{{Pokémon Infobox|name=Poochyena}}", "species"),
    ]
    for page in [*pages, *species]:
        store.write_page(page)
    manifest = Manifest().replace_pins("location", [page.pin for page in pages])
    store.write_manifest(manifest.replace_pins("species", [page.pin for page in species]))
    return store


def build_snapshot(root: Path, text: str | None) -> SnapshotStore:
    """A PokeAPI snapshot whose `locations.csv` holds `text`, or no snapshot at all."""
    store = SnapshotStore(root)
    if text is not None:
        content = text.encode()
        store.build_file_path("locations.csv").parent.mkdir(parents=True)
        store.build_file_path("locations.csv").write_bytes(content)
        store.write_manifest(
            SnapshotManifest(
                repository="x/y", commit="a" * 40, files={"locations.csv": compute_sha256(content)}
            )
        )
    return store


def run_parse(
    tmp_path: Path, titles: tuple[str, ...] = (), locations_csv: str | None = None
) -> LocationParseRun:
    inputs = LocationRunInputs(
        LocationParseRequest(titles),
        build_store(tmp_path / "data"),
        ONTOLOGY_DIR,
        tmp_path / "out",
        build_snapshot(tmp_path / "pokeapi", locations_csv),
    )
    return parse_location_pages(inputs)


def test_every_pin_gets_a_result_and_the_report_counts_them(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert {page.title: page.status for page in report.pages} == {
        SOOTOPOLIS: PageStatus.PARSED,
        ROUTE: PageStatus.PARSED,
        KANTO: PageStatus.PARSED,
        BADGE: PageStatus.SKIPPED,
        MT_CORONET: PageStatus.SKIPPED,
    }
    assert report.skipped_per_reason == {
        "redirect": 1,
        "no_location_infobox": 1,
        "not_core_series": 0,
    }
    assert report.locations_per_class == {"abox_region": 1, "route": 1, "settlement": 1}
    assert report.areas == 1
    assert report.duplicate_ids == ()


def test_a_page_names_its_neighbour_with_the_class_of_its_page(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    sootopolis = next(o for o in run.outputs if o.page.title == SOOTOPOLIS)
    route_edge = next(e for e in sootopolis.batch.edges if e.end.id == "hoenn-route-126")
    assert route_edge.end.class_iri.endswith("#Route")
    assert {o.page.title for o in run.outputs} == {SOOTOPOLIS, ROUTE, KANTO}


def test_references_no_page_owns_are_listed_and_become_candidates(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert run.report.unowned_stub_titles == ("Cave of Origin",)
    assert run.candidates == (
        "Badge",
        "Cave of Origin",
        "Hoenn",
        "Hoenn Route 126",
        "Kanto",
        "Mt. Coronet",
        "Sootopolis City",
        "Sootopolis Gym",
    )
    assert run.report.candidate_count == 8


def test_a_title_that_is_not_pinned_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ParseRunError, match="Overgrow"):
        run_parse(tmp_path, ("Overgrow",))


def test_a_title_run_builds_only_the_requested_page_and_keeps_every_candidate(
    tmp_path: Path,
) -> None:
    run = run_parse(tmp_path, (SOOTOPOLIS,))

    assert [output.page.title for output in run.outputs] == [SOOTOPOLIS]
    assert len(run.candidates) == 8
    route_edge = next(e for e in run.outputs[0].batch.edges if e.end.id == "hoenn-route-126")
    assert route_edge.end.class_iri.endswith("#Route")


def test_outputs_hold_no_page_text(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert all(output.page.wikitext == "" for output in run.outputs)


def test_two_runs_give_identical_files_and_a_full_run_deletes_stale_ones(tmp_path: Path) -> None:
    root = tmp_path / "files"
    stale = root / PAGE_OUTPUT_DIR / "999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    write_outputs(run_parse(tmp_path), root, is_full_run=True)
    first = {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()}
    report = (root / REPORT_FILE_NAME).read_bytes()
    candidates = (root / CANDIDATES_FILE_NAME).read_bytes()
    write_outputs(run_parse(tmp_path), root, is_full_run=True)

    assert not stale.exists()
    assert sorted(first) == ["701.json", "702.json", "703.json"]
    assert {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()} == first
    assert (root / REPORT_FILE_NAME).read_bytes() == report
    assert (root / CANDIDATES_FILE_NAME).read_bytes() == candidates
    assert json.loads(first["701.json"])["extract"]["title"] == SOOTOPOLIS


def write_item_output(root: Path, item_id: str, name: str) -> None:
    folder = root / "item"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{item_id}.json").write_text(
        json.dumps({"extract": {"item_id": item_id, "name": name, "x": 1}})
    )


def test_finds_are_counted_and_items_without_an_owner_are_listed(tmp_path: Path) -> None:
    write_item_output(tmp_path / "out", "moon-stone", "Moon Stone")

    report = run_parse(tmp_path).report

    assert report.finds_per_method == {"none": 3}
    assert report.hidden_finds == 1
    assert report.duplicate_finds == 1
    assert report.item_stub_names == ("Potion",)
    assert not report.is_item_source_missing
    assert report.find_counts.owner_nodes == {"ItemFind": 3}


def test_a_missing_item_output_is_named_and_the_run_goes_on(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert report.is_item_source_missing
    assert report.pages_parsed == 3


LOCATIONS_CSV = (
    "id,region_id,identifier\n"
    "1,3,sootopolis-city\n"
    "2,3,hoenn-route-126\n"
    "3,3,cave-of-origin\n"
    "4,5,undella-town\n"
    "5,,hyrule-castle\n"
    "6,11,pinball-table\n"
)


def test_pokeapi_locations_with_no_owner_are_listed(tmp_path: Path) -> None:
    report = run_parse(tmp_path, locations_csv=LOCATIONS_CSV).report

    assert report.pokeapi_unmatched == ("cave-of-origin", "undella-town")
    assert not report.is_pokeapi_source_missing


def test_a_missing_pokeapi_file_is_named_and_the_run_goes_on(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert report.is_pokeapi_source_missing
    assert report.pokeapi_unmatched == ()
    assert report.pages_parsed == 3


def test_encounters_are_counted_and_the_unknown_species_is_listed(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert report.encounters_per_kind == {"wild/none": 2}
    assert report.unresolved_species == ("Ghostmon",)
    assert report.encounter_counts.owner_nodes == {"Encounter": 2}


def test_coverage_lists_each_location_and_version_with_its_species(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert [(r.location_id, r.version_id, r.species_ids) for r in run.coverage] == [
        ("sootopolis-city", "emerald", ("poochyena",)),
        ("sootopolis-city", "ruby", ("poochyena",)),
    ]
    assert run.report.coverage_pairs == 2


def test_the_coverage_file_is_written_with_the_run(tmp_path: Path) -> None:
    root = tmp_path / "files"

    write_outputs(run_parse(tmp_path), root, is_full_run=True)

    rows = json.loads((root / COVERAGE_FILE_NAME).read_text())
    assert rows[0] == {
        "location_id": "sootopolis-city",
        "version_id": "emerald",
        "species_ids": ["poochyena"],
    }


def test_a_location_id_equal_to_a_species_id_stops_the_run(tmp_path: Path) -> None:
    store = build_store(tmp_path / "data")
    clash = build_page("Poochyena", "{{Infobox location|type=cave}}\n" + GAME, "location")
    store.write_page(clash)
    manifest = store.read_manifest()
    pins = [*manifest.list_pins("location"), clash.pin]
    store.write_manifest(manifest.replace_pins("location", pins))
    inputs = LocationRunInputs(
        LocationParseRequest(),
        store,
        ONTOLOGY_DIR,
        tmp_path / "out",
        build_snapshot(tmp_path / "pk", None),
    )

    with pytest.raises(IdOverlapError, match="poochyena"):
        parse_location_pages(inputs)
