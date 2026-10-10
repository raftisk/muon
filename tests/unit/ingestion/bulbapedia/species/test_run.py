import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.bulbapedia.species import forms
from muon.ingestion.bulbapedia.species.extract import NoteKind, ResidueKind
from muon.ingestion.bulbapedia.species.forms import OverrideError, PageOverride
from muon.ingestion.bulbapedia.species.run import (
    PAGE_OUTPUT_DIR,
    REPORT_FILE_NAME,
    PageStatus,
    ParseRequest,
    ParseRun,
    ParseRunError,
    RunInputs,
    parse_species_pages,
    write_outputs,
)

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
PAGE_TYPE = "species"
TIMESTAMP = datetime(2026, 10, 10, tzinfo=UTC)
ALPHA = "Alpha (Pokémon)"
BETA = "Beta (Pokémon)"
GAMMA = "Gamma (Pokémon)"
DELTA = "Delta (Pokémon)"
PAGE_IDS = {ALPHA: 101, BETA: 102, GAMMA: 103, DELTA: 104}
LEVEL_16 = "{{Bag/s|Rare Candy|SV}}<br>{{color2|000|Level|Level 16}}"
STATS = "{{Stats|HP=10|Attack=55|Defense=25|SpAtk=35|SpDef=45|Speed=95}}"


def species_wikitext(
    name: str, ndex: int, evolution: str = "", egg_group: str = "Field", extra: str = ""
) -> str:
    return f"""{{{{Pokémon Infobox
|name={name}
|ndex={ndex:04d}
|type1=Ground
|egggroup1={egg_group}
|eggcycles=20
|catchrate=255
|expyield=53
|lv100exp=1,000,000
|gendercode=127
|friendship=70
|color=Brown
|body=05
|generation=1
|height-m=0.2
|weight-kg=0.8
|category=Mole
|evsp=1
}}}}
==Game data==
===Stats===
====Base stats====
{STATS}
===Learnset===
====By leveling up====
{{{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}}}
===Evolution data===
{evolution}
{extra}
"""


def evolution_box(first: str, second: str, method: str = LEVEL_16) -> str:
    return f"{{{{Evobox-2|no1=0001|name1={first}|evo1={method}|no2=0002|name2={second}}}}}"


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS[title],
        revision_id=PAGE_IDS[title] * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type=PAGE_TYPE,
        wikitext=wikitext,
    )


def build_store(root: Path, wikitexts: dict[str, str]) -> PageStore:
    store = PageStore(root)
    pages = [build_page(title, wikitext) for title, wikitext in wikitexts.items()]
    for page in pages:
        store.write_page(page)
    store.write_manifest(Manifest().replace_pins(PAGE_TYPE, [page.pin for page in pages]))
    return store


def default_wikitexts() -> dict[str, str]:
    return {
        ALPHA: species_wikitext(
            "Alpha",
            1,
            evolution_box("Alpha", "Beta") + "\n{{Evobox/Setup|1Alpha|Alpha}}",
        ),
        BETA: species_wikitext("Beta", 2, evolution_box("Alpha", "Beta")),
        GAMMA: species_wikitext("Gamma", 3, egg_group="Imaginary"),
        DELTA: "{{New Pokémon Infobox|name=Delta}}\n==Game data==\n",
    }


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The committed overrides name real pages, which the synthetic store lacks."""
    monkeypatch.setattr(forms, "FORM_OVERRIDES", {})


@pytest.fixture
def store(tmp_path: Path) -> PageStore:
    return build_store(tmp_path / "data", default_wikitexts())


def run_parse(store: PageStore, *titles: str) -> ParseRun:
    return parse_species_pages(RunInputs(ParseRequest(titles), store, ONTOLOGY_DIR))


def test_every_pinned_page_gets_a_result(store: PageStore) -> None:
    run = run_parse(store)

    statuses = {result.title: result.status for result in run.report.pages}
    assert statuses == {
        ALPHA: PageStatus.PARSED,
        BETA: PageStatus.PARSED,
        GAMMA: PageStatus.FAILED,
        DELTA: PageStatus.SKIPPED,
    }
    assert (run.report.pages_parsed, run.report.pages_skipped, run.report.pages_failed) == (2, 1, 1)
    assert [output.page.title for output in run.outputs] == [ALPHA, BETA]


def test_a_failed_page_names_the_value_and_a_skipped_page_says_why(store: PageStore) -> None:
    run = run_parse(store)
    reasons = {result.title: result.reason for result in run.report.pages}

    assert "Imaginary" in (reasons[GAMMA] or "")
    assert GAMMA in (reasons[GAMMA] or "")
    assert reasons[DELTA] == "no Pokémon Infobox"
    assert reasons[ALPHA] is None


def test_a_page_that_breaks_a_parse_rule_fails_alone(tmp_path: Path) -> None:
    wikitexts = default_wikitexts()
    wikitexts[GAMMA] = species_wikitext("Gamma", 3).replace("gendercode=127", "gendercode=100")
    store = build_store(tmp_path / "data", wikitexts)

    run = run_parse(store)

    failed = [r for r in run.report.pages if r.status is PageStatus.FAILED]
    assert [r.title for r in failed] == [GAMMA]
    assert "gendercode" in (failed[0].reason or "")
    assert [output.page.title for output in run.outputs] == [ALPHA, BETA]


def test_the_evolution_batch_resolves_species_of_other_pages(store: PageStore) -> None:
    run = run_parse(store)
    alpha = next(o for o in run.outputs if o.page.title == ALPHA)
    beta = next(o for o in run.outputs if o.page.title == BETA)

    assert alpha.evolution_batch is not None
    assert [n.id for n in alpha.evolution_batch.owner_nodes] == ["alpha:beta"]
    assert beta.evolution_batch is None
    assert run.report.evolution_species_stub_misses == 0
    assert run.report.abox_stub_misses == 0
    assert (run.report.evolutions_total, run.report.evolutions_structured) == (1, 1)
    assert run.report.evolution_rule_share == 1.0
    assert run.report.unowned_evolution_pairs == ()


def test_a_title_run_keeps_the_requested_page_and_still_indexes_every_page(
    store: PageStore,
) -> None:
    run = run_parse(store, ALPHA)

    assert [o.page.title for o in run.outputs] == [ALPHA]
    assert [r.title for r in run.report.pages] == [ALPHA]
    assert run.outputs[0].evolution_batch is not None
    assert run.report.evolution_species_stub_misses == 0


def test_an_unpinned_title_stops_the_run_before_any_page_is_read(
    tmp_path: Path, store: PageStore
) -> None:
    with pytest.raises(ParseRunError, match="Ghost"):
        run_parse(store, "Ghost (Pokémon)")

    assert not (tmp_path / "parsed").exists()


def test_a_stale_override_stops_the_run(store: PageStore, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forms, "FORM_OVERRIDES", {"Ghost (Pokémon)": PageOverride()})

    with pytest.raises(OverrideError, match="Ghost"):
        run_parse(store)


def test_residue_items_are_numbered_per_kind_in_page_order(store: PageStore) -> None:
    run = run_parse(store)

    alpha = next(o for o in run.outputs if o.page.title == ALPHA)
    assert [(r.id, r.kind) for r in alpha.residue] == [("101:evolution:1", ResidueKind.EVOLUTION)]
    item = alpha.residue[0]
    assert (item.page_title, item.revision_id) == (ALPHA, 1010)
    assert "Evobox/Setup" in item.wikitext
    assert [r.id for r in run.report.residue] == ["101:evolution:1"]


def test_the_report_lists_notes_candidates_and_pages_without_a_learnset(
    tmp_path: Path,
) -> None:
    wikitexts = default_wikitexts()
    wikitexts[BETA] = species_wikitext("Beta", 2).replace("learnlist/level9", "learnlist/levelZA")
    store = build_store(tmp_path / "data", wikitexts)

    run = run_parse(store)

    assert run.report.pages_without_learnset == (BETA,)
    assert [(n.page_title, n.kind) for n in run.report.notes] == [(BETA, NoteKind.MISSING_LEARNSET)]
    assert [(e.page_title, e.candidate.form_id) for e in run.report.form_candidates] == [
        (ALPHA, "alpha"),
        (BETA, "beta"),
    ]


def test_the_report_counts_batches_by_class_and_property(store: PageStore) -> None:
    run = run_parse(store)

    species = run.report.species_counts
    assert species.owner_nodes == {"PokemonSpecies": 2, "LearnsetEntry": 2}
    assert species.stubs["Type"] == 2  # 1 Ground stub in each page batch
    assert species.edges["hasType"] == 2
    assert run.report.evolution_counts.owner_nodes == {"Evolution": 1}


def test_an_evolution_a_page_shows_for_another_species_is_unowned_when_nobody_emits_it(
    tmp_path: Path,
) -> None:
    wikitexts = default_wikitexts()
    wikitexts[ALPHA] = species_wikitext("Alpha", 1)
    store = build_store(tmp_path / "data", wikitexts)

    run = run_parse(store)

    assert [(p.page_title, p.evolution_id) for p in run.report.unowned_evolution_pairs] == [
        (BETA, "alpha:beta")
    ]


def test_mega_forms_with_a_stone_are_counted_next_to_their_form_changes(tmp_path: Path) -> None:
    mega_infobox = (
        "|form2=Mega Alpha\n|image2=0001Alpha-Mega.png\n|form2type1=Ground\n"
        "|abilitym=Tough Claws\n|mega=Alphanite\n"
    )
    wikitext = species_wikitext("Alpha", 1).replace("|name=Alpha\n", f"|name=Alpha\n{mega_infobox}")
    store = build_store(tmp_path / "data", {ALPHA: wikitext})

    run = run_parse(store)

    assert (run.report.mega_forms_with_stone, run.report.mega_form_changes) == (1, 1)


def test_outputs_are_written_with_sorted_keys_and_sorted_scope(
    tmp_path: Path, store: PageStore
) -> None:
    root = tmp_path / "parsed"
    write_outputs(run_parse(store), root, is_full_run=True)

    text = (root / PAGE_OUTPUT_DIR / "101.json").read_text(encoding="utf-8")
    data = json.loads(text)
    assert text == json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    assert data["title"] == ALPHA
    assert data["species_batch"]["scope"] == sorted(data["species_batch"]["scope"])
    assert data["evolution_batch"]["scope"] == ["https://muon.dev/ns/pkmn#Evolution"]
    assert data["residue"][0]["id"] == "101:evolution:1"
    assert json.loads((root / PAGE_OUTPUT_DIR / "102.json").read_text())["evolution_batch"] is None
    report = json.loads((root / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert report["pages_failed"] == 1


def test_only_the_parsed_pages_get_an_output_file(tmp_path: Path, store: PageStore) -> None:
    root = tmp_path / "parsed"
    write_outputs(run_parse(store), root, is_full_run=True)

    assert sorted(path.name for path in (root / PAGE_OUTPUT_DIR).iterdir()) == [
        "101.json",
        "102.json",
    ]


def test_a_second_run_writes_identical_files(tmp_path: Path, store: PageStore) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_outputs(run_parse(store), first, is_full_run=True)
    write_outputs(run_parse(store), second, is_full_run=True)

    files = sorted(path.relative_to(first) for path in first.rglob("*.json"))
    assert files
    for relative in files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_a_title_run_writes_one_output_and_keeps_the_others(
    tmp_path: Path, store: PageStore
) -> None:
    root = tmp_path / "parsed"
    write_outputs(run_parse(store), root, is_full_run=True)
    beta_before = (root / PAGE_OUTPUT_DIR / "102.json").read_bytes()
    (root / PAGE_OUTPUT_DIR / "101.json").unlink()

    write_outputs(run_parse(store, ALPHA), root, is_full_run=False)

    assert (root / PAGE_OUTPUT_DIR / "101.json").exists()
    assert (root / PAGE_OUTPUT_DIR / "102.json").read_bytes() == beta_before
    report = json.loads((root / REPORT_FILE_NAME).read_text(encoding="utf-8"))
    assert [page["title"] for page in report["pages"]] == [ALPHA]


def test_a_full_run_deletes_the_outputs_of_pages_it_made_none_for(
    tmp_path: Path, store: PageStore
) -> None:
    root = tmp_path / "parsed"
    output_dir = root / PAGE_OUTPUT_DIR
    output_dir.mkdir(parents=True)
    (output_dir / "999.json").write_text("{}", encoding="utf-8")

    write_outputs(run_parse(store), root, is_full_run=True)

    assert not (output_dir / "999.json").exists()
    assert (output_dir / "101.json").exists()


def test_a_title_run_does_not_delete_other_outputs(tmp_path: Path, store: PageStore) -> None:
    root = tmp_path / "parsed"
    output_dir = root / PAGE_OUTPUT_DIR
    output_dir.mkdir(parents=True)
    (output_dir / "999.json").write_text("{}", encoding="utf-8")

    write_outputs(run_parse(store, ALPHA), root, is_full_run=False)

    assert (output_dir / "999.json").exists()
