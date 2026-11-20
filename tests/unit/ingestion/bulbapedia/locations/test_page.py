from datetime import UTC, datetime

import pytest

from muon.ingestion.bulbapedia.locations.identity import LocationIndex
from muon.ingestion.bulbapedia.locations.page import (
    CLASS_BY_CATEGORY,
    CLASS_BY_TYPE,
    CORE_GAME_CATEGORIES,
    FieldRequirement,
    InfoboxKind,
    LocationClass,
    LocationNoteKind,
    LocationPageExtract,
    LocationTableError,
    Neighbour,
    SkippedPage,
    SkipReason,
    check_category_table,
    parse_location_page,
    read_field_requirements,
    read_generation,
    read_infobox,
    read_link_titles,
    read_location_class,
    read_neighbours,
    read_parent_titles,
    read_version_groups,
)
from muon.ingestion.bulbapedia.pages import Page, compute_sha1
from muon.ingestion.bulbapedia.wikitext import split_sections

GAME = "[[Category:Hoenn locations]][[Category:Emerald locations]]"
INDEX = LocationIndex(
    locations=frozenset({"Mount Coronet", "Sootopolis City"}),
    non_locations=frozenset(),
    redirects={"Mt. Coronet": "Mount Coronet"},
    foreign_titles=frozenset(),
)


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=1,
        revision_id=2,
        timestamp=datetime(2026, 11, 20, tzinfo=UTC),
        sha1=compute_sha1(wikitext),
        page_type="location",
        wikitext=wikitext,
    )


def parse(title: str, wikitext: str) -> LocationPageExtract | SkippedPage:
    return parse_location_page(build_page(title, wikitext), INDEX)


def extract(title: str, wikitext: str) -> LocationPageExtract:
    result = parse(title, wikitext)
    assert isinstance(result, LocationPageExtract)
    return result


def read_fields(template: str, body: str):
    infobox = read_infobox(split_sections(f"{{{{{template}\n{body}\n}}}}")[0])
    assert infobox is not None
    return infobox


def test_town_infobox_gives_a_settlement_in_its_region() -> None:
    result = extract(
        "Sootopolis City",
        "{{Town infobox|name=Sootopolis City|region=Hoenn|generation=3|north=Cave of Origin}}\n"
        + GAME,
    )

    assert result.kind is InfoboxKind.TOWN
    assert result.leaf_class is LocationClass.SETTLEMENT
    assert result.parent_titles == ("Hoenn",)
    assert result.generation == "Generation III"
    assert result.version_groups == ("Emerald",)
    assert not result.is_abox_region


def test_redirect_page_is_skipped() -> None:
    result = parse("Mt. Coronet", "#REDIRECT [[Mount Coronet]]")

    assert isinstance(result, SkippedPage)
    assert result.reason is SkipReason.REDIRECT


def test_page_with_another_infobox_is_skipped() -> None:
    result = parse("Badge", "{{ItemInfobox/head|name=Badge}}\n" + GAME)

    assert isinstance(result, SkippedPage)
    assert result.reason is SkipReason.NO_LOCATION_INFOBOX


def test_page_without_a_core_game_category_is_skipped() -> None:
    result = parse("Snap Beach", "{{Infobox location|type=land}}\n[[Category:New Snap locations]]")

    assert isinstance(result, SkippedPage)
    assert result.reason is SkipReason.NOT_CORE_SERIES


def test_aliases_are_the_pinned_redirects_to_the_page() -> None:
    result = extract("Mount Coronet", "{{Infobox location|type=mountain}}\n" + GAME)

    assert result.aliases == ("Mt. Coronet",)


def test_parent_equal_to_the_page_gives_nothing() -> None:
    result = extract(
        "Mount Coronet",
        "{{Infobox location|type=mountain|location=Inside [[Mt. Coronet]] and [[Sinnoh]]}}\n"
        + GAME,
    )

    assert result.parent_titles == ("Sinnoh",)


def test_category_table_is_checked_against_the_version_groups() -> None:
    labels = {label for names in CORE_GAME_CATEGORIES.values() for label in names}

    check_category_table(labels)
    with pytest.raises(LocationTableError, match="XD"):
        check_category_table(labels - {"XD"})


def test_red_blue_and_yellow_category_gives_2_groups() -> None:
    assert read_version_groups({"Red, Blue and Yellow locations"}) == ("Red-Blue", "Yellow")


@pytest.mark.parametrize(
    ("kind", "categories", "type_value", "title", "expected"),
    [
        (InfoboxKind.TOWN, set(), "", "Pallet Town", LocationClass.SETTLEMENT),
        (InfoboxKind.ROUTE, {"Caves"}, "", "Route 1", LocationClass.ROUTE),
        (InfoboxKind.GYM, set(), "", "Pewter Gym", LocationClass.STRUCTURE),
        (InfoboxKind.REGION, set(), "", "Kitakami", LocationClass.GEO_FEATURE),
        (InfoboxKind.REGION, set(), "", "Kanto", None),
        (InfoboxKind.INFOBOX_LOCATION, {"Caves"}, "rouge", "Cave", LocationClass.GEO_FEATURE),
        (InfoboxKind.INFOBOX_LOCATION, {"Restaurants"}, "rouge", "Cafe", LocationClass.STRUCTURE),
        (InfoboxKind.INFOBOX_LOCATION, set(), "building", "Lab", LocationClass.STRUCTURE),
        (InfoboxKind.INFOBOX_LOCATION, set(), "volcano", "Peak", LocationClass.GEO_FEATURE),
        (InfoboxKind.INFOBOX_LOCATION, set(), "Cave", "Pit", LocationClass.GEO_FEATURE),
        (
            InfoboxKind.INFOBOX_LOCATION,
            {"Obsidian Fieldlands"},
            "fieldlands",
            "Aipom Hill",
            LocationClass.GEO_FEATURE,
        ),
    ],
)
def test_leaf_class_rules(kind, categories, type_value, title, expected) -> None:
    leaf_class, notes = read_location_class(kind, categories, type_value, title)

    assert leaf_class is expected
    assert notes == ()


def test_title_table_comes_before_the_categories() -> None:
    leaf_class, _ = read_location_class(
        InfoboxKind.INFOBOX_LOCATION, {"Towers"}, "land", "Wild Area"
    )

    assert leaf_class is LocationClass.GEO_FEATURE


def test_colour_scheme_type_gives_a_plain_location_with_a_note() -> None:
    leaf_class, notes = read_location_class(InfoboxKind.INFOBOX_LOCATION, set(), "jaune", "Plaza")

    assert leaf_class is LocationClass.LOCATION
    assert [(n.kind, n.detail) for n in notes] == [(LocationNoteKind.NO_LEAF_CLASS, "jaune")]


def test_categories_with_2_classes_take_the_first_in_table_order() -> None:
    leaf_class, notes = read_location_class(
        InfoboxKind.INFOBOX_LOCATION, {"Ruins", "Caves"}, "", "Sealed Chamber"
    )

    assert leaf_class is LocationClass.GEO_FEATURE
    assert [(n.kind, n.detail) for n in notes] == [
        (LocationNoteKind.CLASS_CONFLICT, "geo_feature, structure")
    ]


def test_class_tables_share_no_category_between_classes() -> None:
    names = [name for group in CLASS_BY_CATEGORY.values() for name in group]

    assert len(names) == len(set(names))
    assert LocationClass.LOCATION not in CLASS_BY_TYPE.values()


def test_link_titles_read_links_route_and_kanto_templates() -> None:
    value = (
        "Between {{rt|207|Sinnoh|Routes 207}} and {{rtn|208|Sinnoh}}, "
        "[[Spear Pillar#Layout|the pillar]], {{ka|Power Plant}} and [[Hoenn]][[File:Map.png]]"
    )

    assert read_link_titles(value) == (
        "Sinnoh Route 207",
        "Sinnoh Route 208",
        "Spear Pillar",
        "Power Plant (Kanto)",
        "Hoenn",
    )


def test_parents_come_from_location_links_else_region_values() -> None:
    linked = read_fields("Infobox location", "|location=North of [[Lilycove City]]|region=Hoenn")
    regional = read_fields(
        "Infobox location", "|location=Center of the region|region=Sinnoh|region2=Hisui"
    )

    assert read_parent_titles(linked) == ("Lilycove City",)
    assert read_parent_titles(regional) == ("Sinnoh", "Hisui")


def test_region_infobox_reads_region_values_not_its_league_location() -> None:
    infobox = read_fields("RegionInfobox", "|location=[[Indigo Plateau]]|region=Kanto")

    assert read_parent_titles(infobox) == ("Kanto",)


def test_neighbours_read_plain_titles_and_the_variants() -> None:
    infobox = read_fields(
        "Route infobox",
        "|region=Hoenn|west=Lavaridge Town|east=Route 111|regioneast=Kanto|north=[[Mt. Chimney]]"
        "|north2=Jagged Pass|southalt=Route 217{{tt|*|only in Sinnoh}}|prev=Route 110",
    )

    assert read_neighbours(infobox) == (
        Neighbour(title="Lavaridge Town", direction="west"),
        Neighbour(title="Kanto Route 111", direction="east"),
        Neighbour(title="Mt. Chimney", direction="north"),
        Neighbour(title="Jagged Pass", direction="north"),
        Neighbour(title="Hoenn Route 217", direction="south"),
    )


def test_region_fields_resolve_a_route_without_becoming_neighbours() -> None:
    infobox = read_fields("Town infobox", "|region=Hoenn|south=Route 126|regionsouth=Johto")

    assert read_neighbours(infobox) == (Neighbour(title="Johto Route 126", direction="south"),)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", "Generation III"),
        ("{{gen|IV}}, {{Gen|VIII}}", "Generation IV"),
        ("{{Gen|VIII}}, {{Gen|II}}", "Generation II"),
        ("", None),
        ("unknown", None),
    ],
)
def test_generation_is_the_lowest_named(value: str, expected: str | None) -> None:
    infobox = read_fields("Infobox location", f"|generation={value}")

    assert read_generation(infobox) == expected


def test_plain_yes_gives_a_requirement_and_any_other_value_a_note() -> None:
    infobox = read_fields(
        "Route infobox", "|surf=yes|cut={{sup/4|HGSS}}|bike=yes|rocksmash=Yes|strength=no|flash="
    )

    requirements, notes = read_field_requirements(infobox)

    assert requirements == (
        FieldRequirement(role="fieldMove", target="Surf"),
        FieldRequirement(role="fieldItem", target="Bicycle"),
        FieldRequirement(role="fieldMove", target="Rock Smash"),
    )
    assert [(n.kind, n.field, n.detail) for n in notes] == [
        (LocationNoteKind.LIMITED_FIELD_MOVE, "cut", "{{sup/4|HGSS}}"),
        (LocationNoteKind.LIMITED_FIELD_MOVE, "strength", "no"),
    ]


def test_field_moves_of_a_town_infobox_are_not_read() -> None:
    infobox = read_fields("Town infobox", "|surf=yes")

    assert read_field_requirements(infobox) == ((), ())


def test_abox_region_page_has_no_class() -> None:
    result = extract(
        "Kanto", "{{RegionInfobox|name=Kanto|region=Kanto}}\n[[Category:Kanto locations]]\n" + GAME
    )

    assert result.is_abox_region
    assert result.leaf_class is None


def test_place_titles_read_links_and_main_templates_of_the_section() -> None:
    wikitext = (
        "{{Town infobox|region=Hoenn}}\n"
        "==Geography==\nSee [[Not a place]].\n"
        "==Places of interest==\n===Gym===\n{{main|Sootopolis Gym}}\n"
        "===Shop===\nA [[Poké Mart]] and {{main|Pokémon Center|Cafe}}.\n"
        "==Trivia==\n[[Elsewhere]]\n" + GAME
    )

    result = extract("Sootopolis City", wikitext)

    assert result.place_titles == ("Sootopolis Gym", "Poké Mart", "Pokémon Center", "Cafe")


def test_place_titles_drop_anchors_and_breadcrumbs_of_main_templates() -> None:
    wikitext = (
        "{{Town infobox|region=Hoenn}}\n==Places of interest==\n"
        "{{main|Player's house#Hoenn}} {{main|Player's house → Johto}} {{main|#Only an anchor}}\n"
        + GAME
    )

    assert extract("Sootopolis City", wikitext).place_titles == ("Player's house",)
