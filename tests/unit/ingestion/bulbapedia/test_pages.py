import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from muon.ingestion.bulbapedia.pages import (
    Manifest,
    Page,
    PagePin,
    PageStore,
    PageStoreError,
    compute_sha1,
)

SPECIES = "species"
ABILITY = "ability"
TIMESTAMP = datetime(2026, 9, 14, 11, 45, 57, tzinfo=UTC)
ABC_SHA1 = "a9993e364706816aba3e25717850c26c9cd0d89d"


def build_page(title: str, page_id: int, revision_id: int, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=page_id,
        revision_id=revision_id,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type=SPECIES,
        wikitext=wikitext,
    )


DIGLETT = build_page("Diglett (Pokémon)", 2482, 4630045, "{{Pokémon Infobox|name=Diglett}}")
BULBASAUR = build_page("Bulbasaur (Pokémon)", 1, 100, "{{Pokémon Infobox|name=Bulbasaur}}")
NATURAL_CURE = PagePin(title="Natural Cure (Ability)", page_id=4106, revision_id=7, sha1=ABC_SHA1)


@pytest.fixture
def store(tmp_path: Path) -> PageStore:
    return PageStore(tmp_path)


def test_compute_sha1_matches_known_digest() -> None:
    assert compute_sha1("abc") == ABC_SHA1


def test_missing_manifest_is_empty(store: PageStore) -> None:
    assert store.read_manifest() == Manifest()


def test_manifest_round_trip(store: PageStore) -> None:
    manifest = Manifest().replace_pins(SPECIES, [DIGLETT.pin, BULBASAUR.pin])

    store.write_manifest(manifest)

    assert store.read_manifest() == manifest


def test_write_manifest_sorts_page_types_and_pins(store: PageStore) -> None:
    unsorted = Manifest(
        page_types={SPECIES: (DIGLETT.pin, BULBASAUR.pin), ABILITY: (NATURAL_CURE,)}
    )

    store.write_manifest(unsorted)

    written = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert list(written["page_types"]) == [ABILITY, SPECIES]
    assert [pin["title"] for pin in written["page_types"][SPECIES]] == [
        "Bulbasaur (Pokémon)",
        "Diglett (Pokémon)",
    ]


def test_unchanged_rewrite_is_byte_identical(store: PageStore) -> None:
    store.write_manifest(Manifest().replace_pins(SPECIES, [DIGLETT.pin, BULBASAUR.pin]))
    first = store.manifest_path.read_bytes()

    store.write_manifest(store.read_manifest())

    assert store.manifest_path.read_bytes() == first
    assert first.endswith(b"\n")


def test_non_ascii_titles_are_written_as_is(store: PageStore) -> None:
    pins = [
        PagePin(title="Flabébé (Pokémon)", page_id=3, revision_id=3, sha1=ABC_SHA1),
        PagePin(title="Nidoran♀ (Pokémon)", page_id=4, revision_id=4, sha1=ABC_SHA1),
    ]

    store.write_manifest(Manifest().replace_pins(SPECIES, pins))

    text = store.manifest_path.read_text(encoding="utf-8")
    assert "Flabébé (Pokémon)" in text
    assert "Nidoran♀ (Pokémon)" in text


def test_replace_pins_keeps_other_page_types_and_the_original() -> None:
    original = Manifest().replace_pins(ABILITY, [NATURAL_CURE])

    updated = original.replace_pins(SPECIES, [DIGLETT.pin])

    assert updated.list_pins(ABILITY) == (NATURAL_CURE,)
    assert updated.list_pins(SPECIES) == (DIGLETT.pin,)
    assert original.list_pins(SPECIES) == ()


def test_replace_pins_with_no_pins_drops_the_page_type() -> None:
    manifest = Manifest().replace_pins(ABILITY, [NATURAL_CURE]).replace_pins(ABILITY, [])

    assert ABILITY not in manifest.page_types


def test_find_pin_and_list_page_ids() -> None:
    manifest = Manifest().replace_pins(SPECIES, [DIGLETT.pin]).replace_pins(ABILITY, [NATURAL_CURE])

    assert manifest.find_pin(SPECIES, "Diglett (Pokémon)") == DIGLETT.pin
    assert manifest.find_pin(SPECIES, "Dugtrio (Pokémon)") is None
    assert manifest.list_page_ids() == frozenset({2482, 4106})


def test_invalid_manifest_raises(store: PageStore) -> None:
    store.manifest_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(PageStoreError, match="Cannot read manifest"):
        store.read_manifest()


def test_page_write_and_read(store: PageStore) -> None:
    store.write_page(DIGLETT)

    assert store.read_page(SPECIES, DIGLETT.pin) == DIGLETT


def test_read_page_without_file_names_title_and_fetch_hint(store: PageStore) -> None:
    with pytest.raises(PageStoreError, match=r"Diglett.*muon fetch bulbapedia species"):
        store.read_page(SPECIES, DIGLETT.pin)


def test_read_page_with_other_revision_names_both_ids(store: PageStore) -> None:
    store.write_page(DIGLETT)
    older_pin = DIGLETT.pin.model_copy(update={"revision_id": 4623307})

    with pytest.raises(PageStoreError, match=r"Diglett.*4630045.*4623307"):
        store.read_page(SPECIES, older_pin)


def test_read_page_with_other_sha1_names_both_values(store: PageStore) -> None:
    store.write_page(DIGLETT)
    other_pin = DIGLETT.pin.model_copy(update={"sha1": ABC_SHA1})

    with pytest.raises(PageStoreError, match=rf"Diglett.*{DIGLETT.sha1}.*{ABC_SHA1}"):
        store.read_page(SPECIES, other_pin)


def test_read_page_with_edited_wikitext_fails(store: PageStore) -> None:
    store.write_page(DIGLETT)
    path = store.build_page_path(DIGLETT.page_id)
    edited = json.loads(path.read_text(encoding="utf-8")) | {"wikitext": "edited"}
    path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(PageStoreError, match=rf"hashes to {compute_sha1('edited')}"):
        store.read_page(SPECIES, DIGLETT.pin)


def test_read_page_with_other_page_type_fails(store: PageStore) -> None:
    store.write_page(DIGLETT)

    with pytest.raises(PageStoreError, match="page type 'species', expected 'ability'"):
        store.read_page(ABILITY, DIGLETT.pin)


def test_read_page_with_corrupt_file_fails(store: PageStore) -> None:
    store.build_page_path(DIGLETT.page_id).parent.mkdir(parents=True)
    store.build_page_path(DIGLETT.page_id).write_text("[]", encoding="utf-8")

    with pytest.raises(PageStoreError, match="Cannot read page file"):
        store.read_page(SPECIES, DIGLETT.pin)


def test_is_current(store: PageStore) -> None:
    assert not store.is_current(DIGLETT.pin)

    store.write_page(DIGLETT)

    assert store.is_current(DIGLETT.pin)
    assert not store.is_current(DIGLETT.pin.model_copy(update={"revision_id": 1}))
    assert not store.is_current(DIGLETT.pin.model_copy(update={"sha1": ABC_SHA1}))


def test_is_current_is_false_for_edited_or_corrupt_files(store: PageStore) -> None:
    store.write_page(DIGLETT)
    path = store.build_page_path(DIGLETT.page_id)
    edited = json.loads(path.read_text(encoding="utf-8")) | {"wikitext": "edited"}
    path.write_text(json.dumps(edited), encoding="utf-8")

    assert not store.is_current(DIGLETT.pin)

    path.write_text("{", encoding="utf-8")

    assert not store.is_current(DIGLETT.pin)


def test_delete_pages_except(store: PageStore) -> None:
    assert store.delete_pages_except(frozenset()) == ()
    store.write_page(DIGLETT)
    store.write_page(BULBASAUR)

    deleted = store.delete_pages_except(frozenset({DIGLETT.page_id}))

    assert deleted == (BULBASAUR.page_id,)
    assert store.is_current(DIGLETT.pin)
    assert not store.build_page_path(BULBASAUR.page_id).exists()


@pytest.mark.parametrize(
    ("title", "url_title"),
    [
        ("Diglett (Pokémon)", "Diglett_(Pok%C3%A9mon)"),
        ("Farfetch'd (Pokémon)", "Farfetch%27d_(Pok%C3%A9mon)"),
        ("Type: Null (Pokémon)", "Type%3A_Null_(Pok%C3%A9mon)"),
    ],
)
def test_permalink(title: str, url_title: str) -> None:
    page = build_page(title, 1, 4630045, "text")

    assert page.permalink == (
        f"https://bulbapedia.bulbagarden.net/w/index.php?title={url_title}&oldid=4630045"
    )


def test_writes_leave_no_temporary_files(store: PageStore) -> None:
    store.write_manifest(Manifest().replace_pins(SPECIES, [DIGLETT.pin]))
    store.write_page(DIGLETT)
    store.write_manifest(store.read_manifest())

    names = sorted(path.name for path in store.manifest_path.parent.rglob("*") if path.is_file())

    assert names == ["2482.json", "manifest.json"]
