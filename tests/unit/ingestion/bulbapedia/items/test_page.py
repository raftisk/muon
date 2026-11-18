from datetime import UTC, datetime

import pytest

from muon.ingestion.bulbapedia.items.page import (
    ItemClass,
    ItemNoteKind,
    ItemPageExtract,
    ItemPageParseError,
    SkippedPage,
    SkipReason,
    parse_item_page,
    read_category,
    read_item_class,
    read_type_phrase,
)
from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.wikitext import split_sections

HEAD = "{{ItemInfobox/head\n|name=X\n|gen=III\n}}\n"
ROW = "{{ItemInfobox/BagRowBottom|IX|Other Items}}\n|}\n"
PRICE_HEAD = "==In the core series games==\n===Price===\n{{ItemPrice/head|items}}\n"


def build_page(wikitext: str, title: str = "Potion") -> Page:
    return Page(
        title=title,
        page_id=1,
        revision_id=2,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        sha1="x",
        page_type="item",
        wikitext=wikitext,
    )


def build_item(
    lead: str = "The '''X''' is a type of [[item]] introduced in [[Generation III]].",
    rows: str = ROW,
    head: str = HEAD,
    body: str = "",
    title: str = "Potion",
) -> str:
    return f"{head}{rows}\n{lead}\n{body}"


def parse_extract(wikitext: str, title: str = "Potion") -> ItemPageExtract:
    result = parse_item_page(build_page(wikitext, title))
    assert isinstance(result, ItemPageExtract)
    return result


def parse_skip(wikitext: str, title: str = "Potion") -> SkippedPage:
    result = parse_item_page(build_page(wikitext, title))
    assert isinstance(result, SkippedPage)
    return result


def test_page_without_an_infobox_is_skipped() -> None:
    assert parse_skip("A page.").reason is SkipReason.NO_ITEM_INFOBOX


def test_page_with_only_a_pokopia_row_is_skipped() -> None:
    wikitext = build_item(rows="{{ItemInfobox/BagRowBottom|2=Materials}}\n")

    assert parse_skip(wikitext).reason is SkipReason.NO_MAIN_SERIES_BAG_ROW


@pytest.mark.parametrize("title", ["TM01", "HM05", "TR99"])
def test_machine_page_is_skipped(title: str) -> None:
    assert parse_skip(build_item(), title).reason is SkipReason.MACHINE_PAGE


@pytest.mark.parametrize("title", ["TM Case", "TMV Pass", "TM00x"])
def test_machine_like_title_stays_in_scope(title: str) -> None:
    assert parse_extract(build_item(), title).title == title


@pytest.mark.parametrize(
    ("title", "item_id", "name"),
    [
        ("Black Sludge", "black-sludge", "Black Sludge"),
        ("Apple (item)", "apple", "Apple"),
        ("Poké Ball (Hisui)", "poke-ball-hisui", "Poké Ball (Hisui)"),
    ],
)
def test_id_and_name_drop_the_item_suffix_only(title: str, item_id: str, name: str) -> None:
    extract = parse_extract(build_item(), title)

    assert (extract.item_id, extract.name) == (item_id, name)


def test_unknown_generation_raises_with_the_value() -> None:
    wikitext = build_item(head="{{ItemInfobox/head\n|name=X\n|gen=Z\n}}\n")

    with pytest.raises(ItemPageParseError, match="'Z'"):
        parse_item_page(build_page(wikitext))


@pytest.mark.parametrize(
    ("lead", "categories", "expected"),
    [
        ("is a type of [[Berry]] introduced in [[Generation III]].", "", ItemClass.BERRY),
        ("is a type of [[held item]] introduced in [[Generation IV]].", "", ItemClass.HELD_ITEM),
        (
            "is a type of consumable [[held item]] introduced in [[Generation V]].",
            "",
            ItemClass.HELD_ITEM,
        ),
        ("is a type of [[item]] introduced in [[Generation I]].", "", ItemClass.ITEM),
        (
            "is a type of [[item]] introduced in [[Generation IX]].",
            "[[Category:Gen IV held items]]",
            ItemClass.HELD_ITEM,
        ),
        (
            "is a type of [[Hyperspace Lumiose]] [[Berry]] introduced in [[Pokémon Legends: Z-A]].",
            "",
            ItemClass.ITEM,
        ),
        ("is a type of [[Berry]] exclusive to the [[Generation II]] games.", "", ItemClass.BERRY),
        ("has no type phrase.", "", ItemClass.ITEM),
    ],
)
def test_item_class(lead: str, categories: str, expected: ItemClass) -> None:
    wikitext = build_item(lead=f"The '''X''' {lead}", body=categories)

    assert parse_extract(wikitext).item_class is expected


def test_type_phrase_keeps_a_link_with_a_period() -> None:
    lead = split_sections("The X is a type of [[Exp. Candy L]] introduced in [[Generation VIII]].")[
        0
    ]

    assert read_type_phrase(lead) == "type of [[Exp. Candy L]]"


def test_item_class_without_a_phrase_reads_the_categories() -> None:
    assert read_item_class(None, ["Gen III held items"]) is ItemClass.HELD_ITEM
    assert read_item_class(None, []) is ItemClass.ITEM


def test_category_is_the_pocket_of_the_last_main_series_row() -> None:
    rows = (
        "{{ItemInfobox/BagRow|III|Items}}\n"
        "{{ItemInfobox/BagRow|VIII|Other Items}}\n"
        "{{ItemInfobox/BagRowBottom|2=Materials}}\n"
    )

    assert parse_extract(build_item(rows=rows)).category == "Other Items"


def test_category_reads_the_legends_z_a_row() -> None:
    rows = (
        "{{ItemInfobox/BagRow|VI|Items}}\n"
        "{{ItemInfobox/BagRowBottom|IX|Mega Stones|game=ZA|group={{gameabbrev9|ZA}}}}\n"
    )

    assert parse_extract(build_item(rows=rows)).category == "Mega Stones"


@pytest.mark.parametrize(
    ("pocket", "expected"),
    [("Medicines", "Medicine"), ("other items", "Other Items"), ("Berries", "Berries")],
)
def test_category_capitalizes_words_and_maps_aliases(pocket: str, expected: str) -> None:
    extract = parse_extract(build_item(rows=f"{{{{ItemInfobox/BagRowBottom|IX|{pocket}}}}}\n"))

    assert extract.category == expected


def test_category_of_no_rows_is_none() -> None:
    assert read_category(()) is None


def price_page(rows: str) -> str:
    return build_item(body=PRICE_HEAD + rows)


def test_scarlet_violet_prices() -> None:
    row = "{{ItemPrice|{{gameabbrev9|SV}}|{{pdollar}}10,000|{{pdollar}}2,500}}\n"

    extract = parse_extract(price_page(row))

    assert (extract.buy_price, extract.sell_price, extract.notes) == (10000, 2500, ())


def test_svza_row_counts_and_n_a_leaves_the_price_absent() -> None:
    row = "{{ItemPrice|{{gameabbrev9|SVZA}}|N/A|{{PDollar}}2,500}}\n"

    extract = parse_extract(price_page(row))

    assert (extract.buy_price, extract.sell_price, extract.notes) == (None, 2500, ())


def test_other_game_rows_give_no_price() -> None:
    row = "{{ItemPrice|{{gameabbrev8|SwSh}}|{{pdollar}}300|{{pdollar}}150}}\n"

    assert parse_extract(price_page(row)).buy_price is None


def test_a_cell_with_2_prices_is_noted_and_left_absent() -> None:
    row = "{{ItemPrice|{{gameabbrev9|SV}}|{{pdollar}}100 / {{pdollar}}200|{{pdollar}}50}}\n"

    extract = parse_extract(price_page(row))

    assert (extract.buy_price, extract.sell_price) == (None, 50)
    assert [(note.kind, note.field) for note in extract.notes] == [
        (ItemNoteKind.UNREAD_PRICE, "buyPrice")
    ]


def test_prices_outside_the_price_section_are_ignored() -> None:
    body = (
        "==In spin-off games==\n===Pokémon Mystery Dungeon series===\n===Price===\n"
        "{{ItemPrice|{{gameabbrev9|SV}}|{{pdollar}}1|{{pdollar}}1}}\n"
    )

    assert parse_extract(build_item(body=body)).sell_price is None


@pytest.mark.parametrize(
    ("fling", "expected"),
    [
        ("{{ItemInfobox/Fling|30|colorscheme=items}}", 30),
        ("{{ItemInfobox/FlingEffect|10|effect}}", 10),
        ("{{ItemInfobox/Fling|30<sup>Gen V</sup>}}", 30),
        ("", None),
    ],
)
def test_fling_power(fling: str, expected: int | None) -> None:
    extract = parse_extract(build_item(rows=ROW + fling + "\n"))

    assert extract.fling_power == expected


def test_flavor_potencies_take_the_last_row_and_drop_zeros() -> None:
    rows = (
        ROW
        + "{{ItemInfobox/FlavorRow|III|10|10|10|10|10|20}}\n"
        + "{{ItemInfobox/FlavorRow|IV/VIII|10|10|0|10|10|20}}\n"
        + "{{ItemInfobox/ZAFlavorRow|IX|0|0|0|0|40|1}}\n"
    )

    extract = parse_extract(build_item(rows=rows))

    assert extract.flavor_potencies == {"Spicy": 10, "Dry": 10, "Bitter": 10, "Sour": 10}


def test_item_without_flavor_rows_has_no_potencies() -> None:
    assert parse_extract(build_item()).flavor_potencies == {}


def test_core_section_titles_are_dropped_from_the_paths() -> None:
    body = (
        "==In the core series games==\n===Effect===\n====Held item====\nText.\n"
        "===Description===\n====Tag information====\nTag.\n"
        "==Trivia==\n* A fact.\n==In other languages==\n"
    )

    paths = [section.path for section in parse_extract(build_item(body=body)).sections]

    assert paths == [
        (),
        ("Effect",),
        ("Effect", "Held item"),
        ("Description",),
        ("Description", "Tag information"),
        ("Trivia",),
        ("In other languages",),
    ]


def test_in_the_games_title_is_dropped_too() -> None:
    body = "==In the games==\n===Effect===\nText.\n"

    paths = [section.path for section in parse_extract(build_item(body=body)).sections]

    assert paths == [(), ("Effect",)]


def test_aliases_and_categories_are_read() -> None:
    body = (
        "==In other languages==\n{{langtable|color={{items color light}}|fr=Potion FR}}\n"
        "[[Category:Healing items]]\n"
    )

    extract = parse_extract(build_item(body=body))

    assert extract.aliases == ("Potion FR",)
    assert extract.categories == ("Healing items",)
