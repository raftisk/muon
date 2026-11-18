import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.items.run import (
    PAGE_OUTPUT_DIR,
    REPORT_FILE_NAME,
    ItemParseRequest,
    ItemParseRun,
    ItemRunInputs,
    ParseRunError,
    parse_item_pages,
    read_item_stub_ids,
    write_outputs,
)
from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.bulbapedia.runs import PageStatus

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
TIMESTAMP = datetime(2026, 11, 18, tzinfo=UTC)
ORAN = "Oran Berry"
SLUDGE = "Black Sludge"
BAD_GEN = "Broken Orb"
NO_BOX = "Plain Page"
MACHINE = "TM01"
PAGE_IDS = {ORAN: 601, SLUDGE: 602, BAD_GEN: 603, NO_BOX: 604, MACHINE: 605}

ORAN_WIKITEXT = """{{ItemInfobox/head
|name=Oran Berry
|gen=III
}}
{{ItemInfobox/BagRowBottom|IX|Berries}}
{{ItemInfobox/Fling|10}}
{{ItemInfobox/FlavorRow|IV/VIII|10|10|0|10|10|20}}
|}

The '''Oran Berry''' is a type of [[Berry]] introduced in [[Generation III]].

==In the core series games==
===Price===
{{ItemPrice/head|Berries}}
{{ItemPrice|{{gameabbrev9|SV}}|N/A|{{pdollar}}20}}
===Effect===
Restores a little HP for the {{p|Pikachu}} that holds it, which is long enough to be a chunk.
===Description===
{{movedescentry|{{gameabbrev9|SV}}|A berry.}}
==Trivia==
* A fact about the berry.
"""
SLUDGE_WIKITEXT = ORAN_WIKITEXT.replace("Oran Berry", "Black Sludge").replace(
    "[[Berry]]", "[[held item]]"
)
BAD_GEN_WIKITEXT = ORAN_WIKITEXT.replace("|gen=III", "|gen=ZZ")
BOX = "{{ItemInfobox/head\n|name=X\n|gen=III\n}}\n{{ItemInfobox/BagRowBottom|IX|Items}}\n"


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS[title],
        revision_id=PAGE_IDS[title] * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type="item",
        wikitext=wikitext,
    )


def build_store(root: Path) -> PageStore:
    store = PageStore(root)
    pages = [
        build_page(ORAN, ORAN_WIKITEXT),
        build_page(SLUDGE, SLUDGE_WIKITEXT),
        build_page(BAD_GEN, BAD_GEN_WIKITEXT),
        build_page(NO_BOX, "Just prose.\n"),
        build_page(MACHINE, BOX),
    ]
    for page in pages:
        store.write_page(page)
    store.write_manifest(Manifest().replace_pins("item", [page.pin for page in pages]))
    return store


def write_stub_output(root: Path, page_type: str, stubs: dict[str, str], scope: list[str]) -> None:
    pkmn = "https://muon.dev/ns/pkmn#"
    nodes = [
        {"class_iri": pkmn + cls, "id": item_id, "name": item_id} for item_id, cls in stubs.items()
    ]
    folder = root / page_type
    folder.mkdir(parents=True, exist_ok=True)
    batch = {"nodes": nodes, "scope": [pkmn + name for name in scope], "edges": []}
    (folder / "1.json").write_text(json.dumps({"title": "T", "species_batch": batch, "x": 1}))


def run_parse(
    tmp_path: Path, titles: tuple[str, ...] = (), with_stubs: bool = True
) -> ItemParseRun:
    output_root = tmp_path / "parsed"
    if with_stubs:
        write_stub_output(output_root, "species", {"oran-berry": "Berry", "moon-stone": "Item"}, [])
        write_stub_output(output_root, "move", {}, [])
        write_stub_output(output_root, "ability", {}, [])
    inputs = ItemRunInputs(
        ItemParseRequest(titles), build_store(tmp_path / "data"), ONTOLOGY_DIR, output_root
    )
    return parse_item_pages(inputs)


def test_every_pin_gets_a_result_and_the_report_counts_them(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert {page.title: page.status for page in report.pages} == {
        ORAN: PageStatus.PARSED,
        SLUDGE: PageStatus.PARSED,
        BAD_GEN: PageStatus.FAILED,
        NO_BOX: PageStatus.SKIPPED,
        MACHINE: PageStatus.SKIPPED,
    }
    assert (report.pages_parsed, report.pages_skipped, report.pages_failed) == (2, 2, 1)
    assert report.skipped_per_reason == {
        "no_item_infobox": 1,
        "no_main_series_bag_row": 0,
        "machine_page": 1,
    }
    assert "ZZ" in next(page.reason or "" for page in report.pages if page.title == BAD_GEN)


def test_the_report_counts_classes_values_and_chunks(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert report.items_per_class == {"Berry": 1, "HeldItem": 1}
    assert report.items_per_category == {"Berries": 2}
    assert report.flavor_edges == 8
    assert report.absent_fling_power == 0
    assert report.chunks_per_aspect == {"effect": 2, "flavor": 2, "trivia": 2}
    # No species page is pinned, so the {{p|Pikachu}} mention resolves to nothing.
    assert report.mentions == 0


def test_stubs_that_no_page_owns_are_listed(tmp_path: Path) -> None:
    report = run_parse(tmp_path).report

    assert report.unowned_stub_ids == ("moon-stone",)
    assert report.missing_stub_sources == ()


def test_missing_output_folders_are_named_and_the_run_goes_on(tmp_path: Path) -> None:
    report = run_parse(tmp_path, with_stubs=False).report

    assert report.missing_stub_sources == ("species", "move", "ability")
    assert report.unowned_stub_ids == ()
    assert report.pages_parsed == 2


def test_stub_ids_leave_out_owners_and_other_classes(tmp_path: Path) -> None:
    pkmn = "https://muon.dev/ns/pkmn#"
    root = tmp_path / "parsed"
    folder = root / "species"
    folder.mkdir(parents=True)
    batch = {
        "nodes": [
            {"class_iri": pkmn + "Berry", "id": "owner", "name": "O"},
            {"class_iri": pkmn + "HeldItem", "id": "held", "name": "H"},
            {"class_iri": pkmn + "Move", "id": "move", "name": "M"},
        ],
        "scope": [pkmn + "Berry"],
    }
    (folder / "1.json").write_text(json.dumps({"species_batch": batch}))

    stubs, missing = read_item_stub_ids(root)

    assert stubs == {"held": "HeldItem"}
    assert missing == ("move", "ability")


def test_a_title_that_is_not_pinned_stops_the_run(tmp_path: Path) -> None:
    with pytest.raises(ParseRunError, match="Overgrow"):
        run_parse(tmp_path, ("Overgrow",))


def test_a_title_run_builds_only_the_requested_page(tmp_path: Path) -> None:
    run = run_parse(tmp_path, (ORAN,))

    assert [output.page.title for output in run.outputs] == [ORAN]
    assert [page.title for page in run.report.pages] == [ORAN]


def test_outputs_hold_no_page_text_and_the_batches_have_the_item_class(tmp_path: Path) -> None:
    run = run_parse(tmp_path)

    assert all(output.page.wikitext == "" for output in run.outputs)
    assert all(output.extract.sections == () for output in run.outputs)
    assert {node.id for output in run.outputs for node in output.batch.owner_nodes} == {
        "oran-berry",
        "black-sludge",
    }


def test_two_runs_give_identical_files_and_a_full_run_deletes_stale_ones(tmp_path: Path) -> None:
    root = tmp_path / "out"
    stale = root / PAGE_OUTPUT_DIR / "999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")

    write_outputs(run_parse(tmp_path), root, is_full_run=True)
    first = {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()}
    report = (root / REPORT_FILE_NAME).read_bytes()
    write_outputs(run_parse(tmp_path), root, is_full_run=True)

    assert not stale.exists()
    assert sorted(first) == [f"{PAGE_IDS[ORAN]}.json", f"{PAGE_IDS[SLUDGE]}.json"]
    assert {path.name: path.read_bytes() for path in (root / PAGE_OUTPUT_DIR).iterdir()} == first
    assert (root / REPORT_FILE_NAME).read_bytes() == report
    assert json.loads(first[f"{PAGE_IDS[ORAN]}.json"])["extract"]["item_id"] == "oran-berry"
