"""Location identity for encounters: which link titles of a species page name a place.

Map:
- Titles: `normalize_title` gives the form in which Bulbapedia titles compare.
- Index: `build_location_index` sorts the pinned pages of the `location` page type into
  location articles (a location infobox in the lead), other articles and redirects, and
  keeps the titles that other page types pin.
- Test: `classify_location` gives every candidate 1 verdict. A redirect is followed 1 step.
- Candidates: `list_candidate_titles` gives the titles the next Special:Export must hold,
  so that 1 export replaces the location pins with a complete set.

A candidate that fails the test gives no node: a Location stub with a wrong id would last
until a purge. This module knows nothing about encounters.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import mwparserfromhell

from muon.ingestion.bulbapedia.pages import Page
from muon.ingestion.bulbapedia.wikitext import normalize_template_name, split_sections

ANCHOR_MARK = "#"
TITLE_SPACE = " "
URL_TITLE_SPACE = "_"
WHITESPACE_PATTERN = re.compile(r"\s+")
REDIRECT_PATTERN = re.compile(r"^\s*#redirect\s*:?\s*\[\[([^\]|]+)", re.IGNORECASE)

# Normalized names of the infobox templates that mark a location article, read off the
# lead of the 1,107 pinned location articles of 2026-11-14 (uses in brackets).
LOCATION_INFOBOXES: frozenset[str] = frozenset(
    {
        "infobox location",  # 582, caves, forests, Pal Park, Max Lair
        "route infobox",  # 206
        "town infobox",  # 100
        "regioninfobox",  # 11, regions and Hyperspace Lumiose
        "gyminfobox",  # 3
    }
)


def normalize_title(title: str) -> str:
    """Return the form in which titles compare: spaces for underscores, whitespace runs
    collapsed and the 1st character uppercase, since Bulbapedia capitalizes every title
    (`kalos Route 7` is the page `Kalos Route 7`)."""
    spaced = WHITESPACE_PATTERN.sub(TITLE_SPACE, title.replace(URL_TITLE_SPACE, TITLE_SPACE))
    stripped = spaced.strip()
    return stripped[:1].upper() + stripped[1:]


def read_redirect_target(wikitext: str) -> str | None:
    """Return the normalized target of a `#REDIRECT [[Target]]` page, without its anchor,
    or `None` for an article."""
    match = REDIRECT_PATTERN.match(wikitext)
    if match is None:
        return None
    return normalize_title(match.group(1).split(ANCHOR_MARK, 1)[0])


def has_location_infobox(page: Page, infobox_names: frozenset[str]) -> bool:
    lead = split_sections(page.wikitext)[0]
    templates = mwparserfromhell.parse(lead.text).filter_templates(recursive=False)
    return any(normalize_template_name(t.name.strip_code()) in infobox_names for t in templates)


@dataclass(frozen=True)
class LocationIndex:
    """The pinned location-type pages by normalized title, and the titles other page types pin."""

    locations: frozenset[str]
    non_locations: frozenset[str]
    redirects: Mapping[str, str]
    foreign_titles: frozenset[str]

    @property
    def pinned_titles(self) -> frozenset[str]:
        return self.locations | self.non_locations | self.redirects.keys()


def build_location_index(
    pages: Iterable[Page], foreign_titles: Iterable[str], infobox_names: frozenset[str]
) -> LocationIndex:
    """Sort the pinned location-type pages. `infobox_names` holds normalized template names."""
    locations: set[str] = set()
    non_locations: set[str] = set()
    redirects: dict[str, str] = {}
    for page in pages:
        title = normalize_title(page.title)
        target = read_redirect_target(page.wikitext)
        if target is not None:
            redirects[title] = target
        elif has_location_infobox(page, infobox_names):
            locations.add(title)
        else:
            non_locations.add(title)
    return LocationIndex(
        locations=frozenset(locations),
        non_locations=frozenset(non_locations),
        redirects=redirects,
        foreign_titles=frozenset(normalize_title(title) for title in foreign_titles),
    )


class LocationVerdict(StrEnum):
    LOCATION = "location"
    NOT_A_LOCATION = "not_a_location"
    UNKNOWN_LOCATION = "unknown_location"


@dataclass(frozen=True)
class LocationTest:
    """The verdict on 1 candidate. `title` is the location title for a location, the title
    the next export must hold for an unknown one, and the normalized candidate otherwise."""

    verdict: LocationVerdict
    title: str


def classify_location(index: LocationIndex, candidate: str) -> LocationTest:
    """Test 1 candidate title against the pins.

    A title that another page type pins is never a location. A pinned redirect is followed
    1 step; a target that is itself a pinned redirect is not a location, and a target that
    nothing pins is unknown, with the target as the title to export.
    """
    title = normalize_title(candidate)
    target = index.redirects.get(title, title)
    if title in index.foreign_titles or target in index.foreign_titles:
        return LocationTest(LocationVerdict.NOT_A_LOCATION, title)
    if target in index.locations:
        return LocationTest(LocationVerdict.LOCATION, target)
    if target in index.pinned_titles:
        return LocationTest(LocationVerdict.NOT_A_LOCATION, title)
    return LocationTest(LocationVerdict.UNKNOWN_LOCATION, target)


def list_candidate_titles(index: LocationIndex, unknown: Iterable[str]) -> tuple[str, ...]:
    """Return every pinned location-type title and every unknown title, minus the titles
    other page types pin, sorted and without duplicates."""
    titles = (index.pinned_titles | {normalize_title(title) for title in unknown}) - (
        index.foreign_titles
    )
    return tuple(sorted(titles))
