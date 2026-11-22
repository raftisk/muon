import pytest

from muon.ingestion.bulbapedia.locations.encounter_templates import (
    CATCH_TEMPLATES,
    CatchTableError,
    check_catch_templates,
    list_catch_versions,
)
from muon.ingestion.bulbapedia.locations.encounters import (
    EncounterFact,
    EncounterReading,
    EncounterSkipReason,
    KindMethod,
    read_encounters,
    read_level_range,
    read_method_cell,
    read_rate,
    read_section_kind,
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
    locations=frozenset({"Cave"}),
    non_locations=frozenset(),
    redirects={},
    foreign_titles=frozenset(),
)


def read_rows(rows: str, heading: str = "") -> EncounterReading:
    below = f"==={heading}===\n" if heading else ""
    return read_encounters(split_sections(f"==Pokémon==\n{below}{rows}\n"), "Cave", INDEX)


def list_facts(
    reading: EncounterReading,
) -> list[tuple[str, str, str | None, str | None, int | None, tuple[str, ...]]]:
    return [
        (f.species_title, f.version, f.method, f.kind, f.chance, f.conditions)
        for f in reading.facts
    ]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("22", (22, 22)),
        ("20-24", (20, 24)),
        ("21,23", (21, 23)),
        ("56, 58-59", (56, 59)),
        ("28<br>41-44", (28, 44)),
        ("", None),
        ("many", None),
    ],
)
def test_level_cells(cell: str, expected: tuple[int, int] | None) -> None:
    assert read_level_range(cell) == expected


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("50%", 50),
        ("5", 5),
        ("0%", 0),
        ("100%", 100),
        ("0.5%", None),
        ("One", None),
        ("150%", None),
    ],
)
def test_rate_cells(cell: str, expected: int | None) -> None:
    assert read_rate(cell) == expected


def test_method_cells_map_onto_the_existing_kinds() -> None:
    default = KindMethod("wild")

    assert read_method_cell("Grass", default) == default
    assert read_method_cell("Fish Old", default) == KindMethod("wild", "old-rod")
    assert read_method_cell("Gift", default) == KindMethod("gift", "gift")
    assert read_method_cell("Fixed", default) == KindMethod("static", "static")
    assert read_method_cell("Trade", default) is EncounterSkipReason.TRADE
    assert read_method_cell("Hoenn", default) is EncounterSkipReason.UNMAPPED_VALUE


def test_section_headings_give_the_default_kind() -> None:
    assert read_section_kind(["Generation IV", "Wanderers"]) == KindMethod("wild", "wanderer")
    assert read_section_kind(["Pokémon Dens"]) == KindMethod("raid", "max-raid")
    assert read_section_kind(["Special encounters"]) == KindMethod("static")
    assert read_section_kind(["Curry encounters"]) is EncounterSkipReason.SKIPPED_SECTION
    assert read_section_kind(["In-game trades"]) is EncounterSkipReason.TRADE
    assert read_section_kind([]) == KindMethod("wild")


def test_a_gen_1_row_gives_1_fact_per_game_flag() -> None:
    reading = read_rows("{{catch/entry1|016|Pidgey|yes|no|yes|Grass|2-5|50%|type1=Normal}}")

    assert list_facts(reading) == [
        ("Pidgey", "Red", None, "wild", 50, ()),
        ("Pidgey", "Yellow", None, "wild", 50, ()),
    ]
    assert [(f.min_level, f.max_level) for f in reading.facts] == [(2, 5), (2, 5)]


def test_per_time_rates_give_a_condition_each_and_a_zero_gives_none() -> None:
    reading = read_rows("{{catch/entry4|041|Zubat|no|no|yes|Cave|14-15|0%|20%|10%}}")

    assert list_facts(reading) == [
        ("Zubat", "Platinum", None, "wild", 20, ("time-day",)),
        ("Zubat", "Platinum", None, "wild", 10, ("time-night",)),
    ]


def test_equal_per_time_rates_and_all_give_1_fact_with_no_condition() -> None:
    equal = read_rows("{{catch/entryhs|016|Pidgey|yes|no|Grass|2-4|10%|10%|10%}}")
    named = read_rows("{{catch/entryhs|016|Pidgey|yes|no|Grass|2-4|all=30%}}")

    assert list_facts(equal) == [("Pidgey", "HeartGold", None, "wild", 10, ())]
    assert list_facts(named) == [("Pidgey", "HeartGold", None, "wild", 30, ())]


def test_seasons_and_per_game_rates() -> None:
    seasons = read_rows("{{catch/entry5-2|039|Jigglypuff|yes|no|Grass|57|10%|0%|20%|20%}}")
    games = read_rows("{{catch/entry7|165|Ledyba|yes|yes|Grass|2-3|20%|-}}")

    assert [f.conditions for f in seasons.facts] == [
        ("season-spring",),
        ("season-autumn",),
        ("season-winter",),
    ]
    assert list_facts(games) == [("Ledyba", "Sun", None, "wild", 20, ())]


def test_the_method_cell_and_a_section_default_decide_kind_and_method() -> None:
    surf = read_rows("{{catch/entry2|118|Goldeen|yes|yes|yes|Fish Old|10|all=15%}}")
    wanderer = read_rows("{{catch/entry8|010|Caterpie|yes|no|Grass|2-5|15%}}", "Wanderers")

    assert {(f.kind, f.method) for f in surf.facts} == {("wild", "old-rod")}
    assert {(f.kind, f.method) for f in wanderer.facts} == {("wild", "wanderer")}


def test_weather_columns_give_a_weather_condition() -> None:
    reading = read_rows(
        "{{catch/entry8|010|Caterpie|yes|yes|Grass|2-5|15%|rain=20%|snow=na|fog=0%}}"
    )

    assert {f.conditions for f in reading.facts} == {("weather-rain",)}
    assert {f.version for f in reading.facts} == {"Sword", "Shield"}


def test_scarlet_violet_rows_read_the_named_flags() -> None:
    row = (
        "{{Catch/entry9|0187|Hoppip|type1=grass|yes|yes|land=yes|morning=yes|day=yes|2-8|60"
        "|all=60}}"
    )

    reading = read_rows(row)

    assert list_facts(reading) == [
        ("Hoppip", "Scarlet", None, "wild", 60, ("time-morning", "time-day")),
        ("Hoppip", "Violet", None, "wild", 60, ("time-morning", "time-day")),
    ]
    assert reading.facts[0].min_level == 2


def test_scarlet_violet_special_and_arceus_rows() -> None:
    special = read_rows("{{Catch/entry9/special|0417|Pachirisu|yes|no|In a tree|5|Respawns}}")
    arceus = read_rows("{{Catch/entryla|404|Luxio|15-18|30-33|✔|✔|||✔}}")

    assert list_facts(special) == [("Pachirisu", "Scarlet", "static", "static", None, ())]
    assert list_facts(arceus) == [
        ("Luxio", "Legends: Arceus", None, "wild", None, ("time-morning", "time-day"))
    ]
    assert (arceus.facts[0].min_level, arceus.facts[0].max_level) == (15, 18)


def test_trades_forms_side_games_and_unmapped_values_are_counted() -> None:
    rows = (
        "{{catch/entry1|016|Pidgey|yes|no|no|Trade|2|10%}}\n"
        "{{catch/entry7|019A|Rattata|yes|yes|Grass|2-3|20%|30%|form=Alolan Form}}\n"
        "{{catch/header/dn|land}}\n{{catch/entry8|010|Caterpie|yes|yes|Grass|2-5|15%}}\n"
        "{{catch/footer|land}}\n"
        "{{catch/entry1|016|Pidgey|yes|no|no|Rippling|2|10%}}\n"
        "{{catch/entry1|016||yes|no|no|Grass|2|10%}}\n"
    )

    reading = read_rows(rows)

    assert reading.facts == ()
    assert reading.skips["trade"] == 1
    assert reading.skips["form_row"] == 1
    assert reading.skips["side_game"] == 1
    assert reading.skips["unmapped_value"] == 1
    assert reading.unmapped == {"Rippling": 1}


def test_rows_outside_the_pokemon_sections_or_the_core_games_are_not_read() -> None:
    row = "{{catch/entry1|016|Pidgey|yes|no|no|Grass|2|10%}}"

    reading = read_encounters(
        split_sections(f"==In animation==\n===Pokémon===\n{row}\n"), "Cave", INDEX
    )

    assert reading.facts == ()


def test_the_area_comes_from_the_headings_and_a_named_location_stays_at_the_page() -> None:
    row = "{{catch/entry1|016|Pidgey|yes|no|no|Grass|2|10%}}\n"
    wikitext = f"==Pokémon==\n===1F===\n{row}===Cave===\n{row}"

    reading = read_encounters(split_sections(wikitext), "Cave", INDEX)

    assert [f.area.key if f.area else None for f in reading.facts] == ["1f", None]


def test_every_template_names_a_real_version() -> None:
    labels = set(list_catch_versions())

    check_catch_templates(labels)
    with pytest.raises(CatchTableError, match="Sun"):
        check_catch_templates(labels - {"Sun"})
    assert len(CATCH_TEMPLATES) == 16
    assert EncounterFact  # the fact model is part of the reader contract


pages = pytest.mark.pages


def read_cached_encounters(title: str) -> EncounterReading:
    store = PageStore()
    index = read_cached_location_index(store)
    page = read_cached_location_page(store, title)
    return read_encounters(split_sections(mask_comments(page.wikitext)), page.title, index)


@pages
def test_tohjo_falls_has_zubat_in_gold_and_goldeen_with_the_old_rod() -> None:
    facts = list_facts(read_cached_encounters("Tohjo Falls"))

    assert ("Zubat", "Gold", None, "wild", 30, ()) in facts
    assert ("Goldeen", "Gold", "old-rod", "wild", 15, ()) in facts
    assert ("Goldeen", "HeartGold", "old-rod", "wild", 15, ()) in facts


@pages
def test_blush_mountain_has_elekid_in_moon() -> None:
    reading = read_cached_encounters("Blush Mountain")

    assert any(f.species_title == "Elekid" and f.version == "Moon" for f in reading.facts)


@pages
def test_mount_coronet_2f_covers_the_sinnoh_games() -> None:
    reading = read_cached_encounters("Mount Coronet")
    versions = {f.version for f in reading.facts if f.area and f.area.key == "2f"}

    assert {"Diamond", "Pearl", "Platinum", "Brilliant Diamond", "Shining Pearl"} <= versions
