from datetime import UTC, datetime

import pytest

from muon.ingestion.bulbapedia.pages import Page, compute_sha1
from muon.ingestion.bulbapedia.species.locations import (
    LocationIndex,
    LocationVerdict,
    build_location_index,
    classify_location,
    list_candidate_titles,
    normalize_title,
    read_redirect_target,
)

INFOBOX = "location infobox"
INFOBOXES = frozenset({INFOBOX})
TIMESTAMP = datetime(2026, 11, 14, tzinfo=UTC)
SPECIES_PIN = "Cyndaquil (Pokémon)"


def build_page(title: str, wikitext: str, page_id: int) -> Page:
    return Page(
        title=title,
        page_id=page_id,
        revision_id=page_id,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type="location",
        wikitext=wikitext,
    )


PAGES = [
    build_page("Diglett's Cave", "{{Location infobox|name=Diglett's Cave}}\nA cave.", 1),
    build_page("Route 228 (Sinnoh)", "{{location infobox|name=Route 228}}\nA route.", 2),
    build_page("Jade Orb", "{{ItemInfobox|name=Jade Orb}}\nAn orb.", 3),
    build_page("Melanie", "{{Character Infobox|name=Melanie}}\nA person.", 4),
    build_page("Sinnoh Route 228", "#REDIRECT [[Route 228 (Sinnoh)]]", 5),
    build_page("Cyndaquil", "#REDIRECT [[Cyndaquil (Pokémon)]]", 6),
    build_page("Old Cave", "#redirect [[Unpinned Cave#Lower floor]]", 7),
    build_page("Double", "#REDIRECT [[Sinnoh Route 228]]", 8),
]


@pytest.fixture
def index() -> LocationIndex:
    return build_location_index(PAGES, [SPECIES_PIN, "Tackle (move)"], INFOBOXES)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("kalos Route 7", "Kalos Route 7"),
        ("tall_grass", "Tall grass"),
        ("  Diglett's   Cave ", "Diglett's Cave"),
    ],
)
def test_normalize_title(title: str, expected: str) -> None:
    assert normalize_title(title) == expected


@pytest.mark.parametrize(
    ("wikitext", "expected"),
    [
        ("#REDIRECT [[Kalos Route 15]]", "Kalos Route 15"),
        ("#redirect [[kalos Route 15#Area]]", "Kalos Route 15"),
        ("A cave.\n#REDIRECT [[Elsewhere]]", None),
    ],
)
def test_read_redirect_target(wikitext: str, expected: str | None) -> None:
    assert read_redirect_target(wikitext) == expected


def test_index_sorts_the_pages(index: LocationIndex) -> None:
    assert index.locations == {"Diglett's Cave", "Route 228 (Sinnoh)"}
    assert index.non_locations == {"Jade Orb", "Melanie"}
    assert index.redirects["Sinnoh Route 228"] == "Route 228 (Sinnoh)"


def test_infobox_name_with_a_comment_marks_a_location() -> None:
    page = build_page("Hunting Grounds", "{{Location infobox\n<!--|image=x.png-->\n|name=X}}", 9)

    assert build_location_index([page], [], INFOBOXES).locations == {"Hunting Grounds"}


@pytest.mark.parametrize(
    ("candidate", "verdict", "title"),
    [
        ("Diglett's Cave", LocationVerdict.LOCATION, "Diglett's Cave"),
        ("diglett's Cave", LocationVerdict.LOCATION, "Diglett's Cave"),
        ("Sinnoh Route 228", LocationVerdict.LOCATION, "Route 228 (Sinnoh)"),
        ("Jade Orb", LocationVerdict.NOT_A_LOCATION, "Jade Orb"),
        ("Melanie", LocationVerdict.NOT_A_LOCATION, "Melanie"),
        ("Cyndaquil", LocationVerdict.NOT_A_LOCATION, "Cyndaquil"),
        (SPECIES_PIN, LocationVerdict.NOT_A_LOCATION, SPECIES_PIN),
        ("Double", LocationVerdict.NOT_A_LOCATION, "Double"),
        ("Old Cave", LocationVerdict.UNKNOWN_LOCATION, "Unpinned Cave"),
        ("Lake Valor", LocationVerdict.UNKNOWN_LOCATION, "Lake Valor"),
    ],
)
def test_classify_location(
    index: LocationIndex, candidate: str, verdict: LocationVerdict, title: str
) -> None:
    result = classify_location(index, candidate)

    assert (result.verdict, result.title) == (verdict, title)


def test_candidate_titles_hold_pins_and_unknown_titles(index: LocationIndex) -> None:
    titles = list_candidate_titles(index, ["Unpinned Cave", "lake Valor", "Lake Valor"])

    assert titles == (
        "Cyndaquil",
        "Diglett's Cave",
        "Double",
        "Jade Orb",
        "Lake Valor",
        "Melanie",
        "Old Cave",
        "Route 228 (Sinnoh)",
        "Sinnoh Route 228",
        "Unpinned Cave",
    )


def test_candidate_titles_never_hold_a_foreign_pin(index: LocationIndex) -> None:
    assert SPECIES_PIN not in list_candidate_titles(index, [SPECIES_PIN])


def test_empty_index_gives_unknown_for_every_candidate() -> None:
    empty = build_location_index([], [], INFOBOXES)

    for candidate in ("Diglett's Cave", "Jade Orb", "Sinnoh Route 228"):
        assert classify_location(empty, candidate).verdict is LocationVerdict.UNKNOWN_LOCATION
    assert list_candidate_titles(empty, ["Jade Orb"]) == ("Jade Orb",)
