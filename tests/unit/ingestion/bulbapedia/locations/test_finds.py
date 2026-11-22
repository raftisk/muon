import pytest

from muon.ingestion.bulbapedia.locations.finds import (
    FIND_HEADINGS,
    SKIPPED_FIND_HEADINGS,
    VERSION_CODES,
    FindReading,
    FindTableError,
    check_version_codes,
    read_display_name,
    read_item_finds,
)
from muon.ingestion.bulbapedia.locations.identity import (
    LocationIndex,
)
from muon.ingestion.bulbapedia.pages import PageStore
from muon.ingestion.bulbapedia.wikitext import mask_comments, split_sections
from tests.unit.ingestion.bulbapedia.page_cache import (
    read_cached_location_index,
    read_cached_location_page,
)

INDEX = LocationIndex(
    locations=frozenset({"Cave", "Hau'oli Outskirts"}),
    non_locations=frozenset(),
    redirects={},
    foreign_titles=frozenset(),
)


def read_finds(wikitext: str) -> FindReading:
    return read_item_finds(split_sections(wikitext), "Cave", INDEX)


def list_facts(reading: FindReading) -> list[tuple[str, str, str | None, bool]]:
    return [(f.item_name, f.version, f.method, f.is_hidden) for f in reading.facts]


def test_a_row_gives_1_fact_per_game_flag() -> None:
    reading = read_finds("==Items==\n{{Itemlist|Moon Stone|Near the exit|G=yes|S=yes|C=no}}\n")

    assert list_facts(reading) == [
        ("Moon Stone", "Gold", None, False),
        ("Moon Stone", "Silver", None, False),
    ]
    assert all(fact.area is None for fact in reading.facts)


def test_a_row_of_5_items_with_2_flags_gives_10_facts() -> None:
    wikitext = (
        "==Items==\n===Rock Smash===\n"
        "{{Itemlist/5|Red Shard|Blue Shard|Yellow Shard|Green Shard|Heart Scale|Smashed"
        "|HG=yes|SS=yes}}\n"
    )

    reading = read_finds(wikitext)

    assert len(reading.facts) == 10
    assert {f.method for f in reading.facts} == {"rock-smash"}
    assert [f.item_name for f in reading.facts][:2] == ["Red Shard", "Red Shard"]


@pytest.mark.parametrize(
    ("heading", "method", "is_hidden"),
    [
        ("Hidden items", None, True),
        ("Rock Smash", "rock-smash", False),
        ("Found in smashed rocks", "rock-smash", False),
        ("Fishing", "fishing", False),
        ("Berry tree", "berry-tree", False),
        ("Berry plant", "berry-tree", False),
        ("Berry pile", "berry-tree", False),
        ("Gifts", "gift", False),
        ("100-Watt gifts", "gift", False),
        ("Natural objects", None, False),
    ],
)
def test_the_heading_gives_the_find_kind(heading: str, method: str | None, is_hidden: bool) -> None:
    reading = read_finds(f"==Items==\n==={heading}===\n{{{{Itemlist|Potion|Here|Sw=yes}}}}\n")

    assert list_facts(reading) == [("Potion", "Sword", method, is_hidden)]


@pytest.mark.parametrize("heading", ["Zygarde Cube", "Totem Stickers", "Fake items"])
def test_skipped_headings_give_no_facts_and_are_counted(heading: str) -> None:
    reading = read_finds(
        f"==Items==\n==={heading}===\n{{{{Itemlist|Sticker|Here|Su=yes|M=yes}}}}\n"
    )

    assert reading.facts == ()
    assert reading.skips["skipped_heading"] == 2


def test_the_item_is_the_page_display_links_to() -> None:
    wikitext = (
        "==Items==\n"
        "{{Itemlist|Bike BDSP|From a man|D=yes|display=[[Bicycle|Bike]]}}\n"
        "{{Itemlist|Ultra Ball|Two of them|Colo=yes|display=[[Ultra Ball]] ×2}}\n"
        "{{Itemlist/2|Potion|Antidote|Both|Y=yes|display2=[[Antidote (item)|Antidote]]}}\n"
    )

    assert [f.item_name for f in read_finds(wikitext).facts] == [
        "Bicycle",
        "Ultra Ball",
        "Potion",
        "Antidote",
    ]


def test_machine_rows_unknown_flags_and_rows_with_no_item_are_counted() -> None:
    wikitext = (
        "==Items==\n"
        "{{Itemlist|TM Normal|Man|Bl=yes|W=yes|display={{TM|70|Flash Cannon}}}}\n"
        "{{Itemlist|TM01|Man|G=yes}}\n"
        "{{Itemlist|Nugget|Floor|JR=yes|JG=yes|R=yes}}\n"
        "{{Itemlist||sprite=Cell|By the TV|Su=yes|display=[[Zygarde Cube|Cell]]}}\n"
    )

    reading = read_finds(wikitext)

    assert list_facts(reading) == [("Nugget", "Red", None, False)]
    assert reading.skips == {
        "skipped_heading": 0,
        "machine_find": 3,
        "unknown_version_code": 2,
        "no_item_name": 1,
    }


def test_the_area_comes_from_the_headings_below_items() -> None:
    wikitext = (
        "==Items==\n===B1F===\n{{Itemlist|Potion|Here|Sw=yes}}\n"
        "===Hau'oli Outskirts===\n{{Itemlist|Ether|There|Sw=yes}}\n"
        "==Trivia==\n{{Itemlist|Elixir|Not read|Sw=yes}}\n"
    )

    areas = [(f.item_name, f.area) for f in read_finds(wikitext).facts]

    assert [(name, area and (area.key, area.location_title)) for name, area in areas] == [
        ("Potion", ("b1f", "Cave")),
        ("Ether", ("", "Hau'oli Outskirts")),
    ]


def test_every_flag_and_heading_table_row_is_a_real_name() -> None:
    labels = set(VERSION_CODES.values())

    check_version_codes(labels)
    with pytest.raises(FindTableError, match="XD"):
        check_version_codes(labels - {"XD"})
    assert not set(FIND_HEADINGS) & SKIPPED_FIND_HEADINGS
    assert read_display_name("") is None


pages = pytest.mark.pages


def read_cached_finds(title: str) -> FindReading:
    store = PageStore()
    index = read_cached_location_index(store)
    page = read_cached_location_page(store, title)
    return read_item_finds(split_sections(mask_comments(page.wikitext)), page.title, index)


@pages
def test_tohjo_falls_has_moon_stones_and_rock_smash_rows() -> None:
    reading = read_cached_finds("Tohjo Falls")
    facts = list_facts(reading)

    assert ("Moon Stone", "Gold", None, False) in facts
    assert ("Moon Stone", "HeartGold", None, False) in facts
    assert ("Max Ether", "SoulSilver", "rock-smash", False) in facts
    assert ("Heart Scale", "HeartGold", "rock-smash", False) in facts
