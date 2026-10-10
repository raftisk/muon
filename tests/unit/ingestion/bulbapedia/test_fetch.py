import json
import re
import shutil
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.api import BulbapediaAPIError
from muon.ingestion.bulbapedia.fetch import FetchError, FetchRequest, fetch_pages
from muon.ingestion.bulbapedia.pages import PageStore
from tests.unit.ingestion.bulbapedia.fake_wiki import FakeWiki, build_status_failure

SPECIES = "species"
ABILITY = "ability"
SPECIES_CATEGORY = "Category:Pokémon"
ABILITY_CATEGORY = "Category:Abilities"
BULBASAUR = "Bulbasaur (Pokémon)"
DIGLETT = "Diglett (Pokémon)"
DUGTRIO = "Dugtrio (Pokémon)"
ALL_SPECIES = [BULBASAUR, DIGLETT, DUGTRIO]
OVERVIEW = "Pokémon (species)"
NATURAL_CURE = "Natural Cure (Ability)"
THICK_FAT = "Thick Fat (Ability)"


@pytest.fixture
def wiki() -> FakeWiki:
    fake = FakeWiki()
    for title in ALL_SPECIES:
        fake.add_page(title, f"{{{{Pokémon Infobox|name={title}}}}}", SPECIES_CATEGORY)
    fake.add_page(OVERVIEW, "An overview page", SPECIES_CATEGORY)
    fake.add_page("Ability", "An overview page", ABILITY_CATEGORY)
    fake.add_page(NATURAL_CURE, "Cures status on switch-out.", ABILITY_CATEGORY)
    fake.add_page(THICK_FAT, "Halves Fire and Ice damage.", ABILITY_CATEGORY)
    return fake


@pytest.fixture
def store(tmp_path: Path) -> PageStore:
    return PageStore(tmp_path)


def latest_revision_id(wiki: FakeWiki, title: str) -> int:
    return wiki.pages[title].revisions[-1].revision_id


def content_titles(wiki: FakeWiki, first_request: int = 0) -> list[str]:
    """Titles of the content-by-title requests from request number `first_request` on."""
    return [
        title
        for request in wiki.requests[first_request:]
        if "content" in request.url.params.get("rvprop", "") and "titles" in request.url.params
        for title in request.url.params["titles"].split("|")
    ]


def content_revision_ids(wiki: FakeWiki, first_request: int = 0) -> list[int]:
    return [
        int(value)
        for request in wiki.requests[first_request:]
        if "revids" in request.url.params
        for value in request.url.params["revids"].split("|")
    ]


def test_first_fetch_pins_every_matching_page(wiki: FakeWiki, store: PageStore) -> None:
    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    manifest = store.read_manifest()
    assert [pin.title for pin in manifest.list_pins(SPECIES)] == ALL_SPECIES
    for pin in manifest.list_pins(SPECIES):
        page = store.read_page(SPECIES, pin)
        assert page.revision_id == latest_revision_id(wiki, pin.title)
    assert OVERVIEW not in {pin.title for pin in manifest.list_pins(SPECIES)}
    assert (report.listed, report.fetched, report.skipped) == (3, 3, 0)
    assert report.added == tuple(ALL_SPECIES)
    assert report.requests == 2
    assert report.elapsed_seconds >= 0


def test_rerun_makes_no_request_and_keeps_the_manifest(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    manifest_bytes = store.manifest_path.read_bytes()

    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    assert report.requests == 0
    assert (report.fetched, report.skipped) == (0, 3)
    assert store.manifest_path.read_bytes() == manifest_bytes


def test_fresh_clone_fetches_the_pinned_revisions(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    pinned_id = latest_revision_id(wiki, DIGLETT)
    manifest_bytes = store.manifest_path.read_bytes()
    shutil.rmtree(store.pages_dir)
    wiki.edit_page(DIGLETT, "a newer revision")
    first_request = len(wiki.requests)

    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    pin = store.read_manifest().find_pin(SPECIES, DIGLETT)
    assert pin is not None
    assert store.read_page(SPECIES, pin).revision_id == pinned_id
    assert store.manifest_path.read_bytes() == manifest_bytes
    assert content_titles(wiki, first_request) == []
    assert report.fetched == 3


def test_only_a_corrupt_page_file_is_fetched_again(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    pin = store.read_manifest().find_pin(SPECIES, DIGLETT)
    assert pin is not None
    store.build_page_path(pin.page_id).write_text("{", encoding="utf-8")
    first_request = len(wiki.requests)

    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    assert content_revision_ids(wiki, first_request) == [pin.revision_id]
    assert report.fetched == 1
    assert store.read_page(SPECIES, pin).title == DIGLETT


def test_update_moves_adds_and_removes_pins(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    bulbasaur_id = wiki.pages[BULBASAUR].page_id
    wiki.edit_page(DIGLETT, "an edit")
    wiki.add_page("Mew (Pokémon)", "a new species", SPECIES_CATEGORY)
    wiki.remove_from_category(SPECIES_CATEGORY, BULBASAUR)
    first_request = len(wiki.requests)

    report = fetch_pages(FetchRequest(SPECIES, is_update=True), store, wiki.build_client())

    assert report.changed == (DIGLETT,)
    assert report.added == ("Mew (Pokémon)",)
    assert report.removed == (BULBASAUR,)
    assert content_revision_ids(wiki, first_request) == [latest_revision_id(wiki, DIGLETT)]
    assert content_titles(wiki, first_request) == ["Mew (Pokémon)"]
    titles = [pin.title for pin in store.read_manifest().list_pins(SPECIES)]
    assert titles == [DIGLETT, DUGTRIO, "Mew (Pokémon)"]
    assert not store.build_page_path(bulbasaur_id).exists()


def test_update_with_no_matching_member_fails(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    manifest_bytes = store.manifest_path.read_bytes()
    for title in ALL_SPECIES:
        wiki.remove_from_category(SPECIES_CATEGORY, title)

    with pytest.raises(FetchError, match="lists no page ending in ' \\(Pokémon\\)'"):
        fetch_pages(FetchRequest(SPECIES, is_update=True), store, wiki.build_client())

    assert store.manifest_path.read_bytes() == manifest_bytes


def test_title_run_pins_only_the_given_pages(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    report = fetch_pages(FetchRequest(ABILITY, titles=(NATURAL_CURE,)), store, wiki.build_client())
    fetch_pages(FetchRequest(ABILITY, titles=(THICK_FAT,)), store, wiki.build_client())

    manifest = store.read_manifest()
    assert [pin.title for pin in manifest.list_pins(ABILITY)] == [NATURAL_CURE, THICK_FAT]
    assert [pin.title for pin in manifest.list_pins(SPECIES)] == ALL_SPECIES
    assert report.added == (NATURAL_CURE,)
    assert content_titles(wiki) == ALL_SPECIES + [NATURAL_CURE, THICK_FAT]


def test_title_run_records_redirects(wiki: FakeWiki, store: PageStore) -> None:
    wiki.add_redirect("Natural Cure", NATURAL_CURE)

    report = fetch_pages(
        FetchRequest(ABILITY, titles=("Natural Cure",)), store, wiki.build_client()
    )

    assert report.redirects == (("Natural Cure", NATURAL_CURE),)
    assert store.read_manifest().find_pin(ABILITY, NATURAL_CURE) is not None


def test_title_run_keeps_a_pinned_revision_unless_updating(
    wiki: FakeWiki, store: PageStore
) -> None:
    fetch_pages(FetchRequest(ABILITY, titles=(NATURAL_CURE,)), store, wiki.build_client())
    pinned_id = latest_revision_id(wiki, NATURAL_CURE)
    wiki.edit_page(NATURAL_CURE, "an edit")

    kept = fetch_pages(FetchRequest(ABILITY, titles=(NATURAL_CURE,)), store, wiki.build_client())
    pin = store.read_manifest().find_pin(ABILITY, NATURAL_CURE)
    assert pin is not None
    assert (pin.revision_id, kept.requests, kept.changed) == (pinned_id, 0, ())

    updated = fetch_pages(
        FetchRequest(ABILITY, titles=(NATURAL_CURE,), is_update=True), store, wiki.build_client()
    )
    pin = store.read_manifest().find_pin(ABILITY, NATURAL_CURE)
    assert pin is not None
    assert pin.revision_id == latest_revision_id(wiki, NATURAL_CURE)
    assert updated.changed == (NATURAL_CURE,)


def test_title_run_with_a_missing_title_changes_nothing(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(ABILITY, titles=(NATURAL_CURE,)), store, wiki.build_client())
    manifest_bytes = store.manifest_path.read_bytes()
    files_before = sorted(store.list_page_files())

    with pytest.raises(BulbapediaAPIError, match="No Such Ability"):
        fetch_pages(FetchRequest(ABILITY, titles=("No Such Ability",)), store, wiki.build_client())

    assert store.manifest_path.read_bytes() == manifest_bytes
    assert sorted(store.list_page_files()) == files_before


def test_failed_first_fetch_resumes_with_the_missing_pages(
    wiki: FakeWiki, store: PageStore
) -> None:
    extra = [f"Species {number} (Pokémon)" for number in range(117)]
    for title in extra:
        wiki.add_page(title, f"text of {title}", SPECIES_CATEGORY)
    wiki.fail_at(3, build_status_failure(403))

    with pytest.raises(BulbapediaAPIError, match="HTTP 403"):
        fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    assert not store.manifest_path.exists()
    first_batch = {page.title for page in store.read_unpinned_pages(frozenset())}
    assert len(first_batch) == 50
    first_request = len(wiki.requests)

    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    assert first_batch.isdisjoint(content_titles(wiki, first_request))
    assert len(content_titles(wiki, first_request)) == 70
    assert len(store.read_manifest().list_pins(SPECIES)) == 120
    assert (report.fetched, report.skipped) == (70, 50)


def test_failed_pinned_fetch_resumes_with_the_missing_pages(
    wiki: FakeWiki, store: PageStore
) -> None:
    for number in range(117):
        wiki.add_page(f"Species {number} (Pokémon)", "text", SPECIES_CATEGORY)
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    shutil.rmtree(store.pages_dir)
    wiki.fail_at(len(wiki.requests) + 2, build_status_failure(403))

    with pytest.raises(BulbapediaAPIError):
        fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    first_request = len(wiki.requests)
    report = fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())

    assert len(content_revision_ids(wiki, first_request)) == 70
    assert report.fetched == 70


def test_pin_that_disagrees_with_the_api_fails(wiki: FakeWiki, store: PageStore) -> None:
    fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["page_types"][SPECIES][0]["sha1"] = "f" * 40
    store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FetchError, match=rf"{re.escape(BULBASAUR)}.*but the manifest pins f{{40}}"):
        fetch_pages(FetchRequest(SPECIES), store, wiki.build_client())


def test_unknown_page_type_fails(wiki: FakeWiki, store: PageStore) -> None:
    with pytest.raises(FetchError, match="known page types: ability, item, species"):
        fetch_pages(FetchRequest("moves"), store, wiki.build_client())
