"""The catch template table of the location pages: which parameter holds what (L18, L19).

Every `catch/entry*` variant of the core series games is a row of a wild table. Its
parameters are positional: 1 is the dex number and 2 the species name, then 1 flag per game
column (`yes` marks the games that have the row), the method cell, the level cell and the
rate cells. The columns were read off the template sources (the export of 2026-11-20):

- `CATCH_TEMPLATES` holds the variants that share that layout and gives each its game
  labels, the position of its method cell and of its level cell and how its rate cells read.
- `catch/entry8/raid`, `catch/entry9`, `catch/entry9/special`, `catch/entryla` and
  `catch/entryza` have their own layouts and readers in `encounters.py`.

A game label is the label of an A-Box Version; `check_catch_templates` checks them before any
page is read. A change to a template needs a row here and a golden test.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum

TIME_CONDITIONS = ("time-morning", "time-day", "time-night")
SEASON_CONDITIONS = ("season-spring", "season-summer", "season-autumn", "season-winter")
# Named parameters of the weather columns of `catch/entry8`.
WEATHER_PARAMETERS = (
    "clear",
    "sun",
    "cloudy",
    "rain",
    "thunderstorm",
    "snow",
    "blizzard",
    "fog",
    "sandstorm",
)
WEATHER_CONDITION = "weather-{name}"


class RateLayout(StrEnum):
    """How the cells after the level cell read.

    `single` is 1 rate for every game of the row, `per_time` has a morning, a day and a night
    rate, `per_season` has a rate per season and `per_version` has 1 rate for each game column
    in order. A named `all=` rate replaces the cells in every layout.
    """

    SINGLE = "single"
    PER_TIME = "per_time"
    PER_SEASON = "per_season"
    PER_VERSION = "per_version"


@dataclass(frozen=True)
class CatchTemplate:
    """The columns of 1 catch template variant, positions counted from 1."""

    name: str
    version_labels: tuple[str, ...]
    method_position: int
    level_position: int
    rate_layout: RateLayout

    @property
    def rate_position(self) -> int:
        return self.level_position + 1


def build_template(
    name: str,
    version_labels: tuple[str, ...],
    method_position: int,
    rate_layout: RateLayout = RateLayout.SINGLE,
) -> CatchTemplate:
    return CatchTemplate(name, version_labels, method_position, method_position + 1, rate_layout)


CATCH_TEMPLATES: Mapping[str, CatchTemplate] = {
    template.name: template
    for template in (
        build_template("catch/entry1", ("Red", "Blue", "Yellow"), 6),
        build_template("catch/entry2", ("Gold", "Silver", "Crystal"), 6, RateLayout.PER_TIME),
        build_template("catch/entry3", ("Ruby", "Sapphire", "Emerald"), 6),
        build_template("catch/entry3a", ("FireRed", "LeafGreen", "Emerald"), 6),
        build_template("catch/entry4", ("Diamond", "Pearl", "Platinum"), 6, RateLayout.PER_TIME),
        # Position 5 is a 3rd flag that no column shows.
        build_template("catch/entry5", ("Black", "White"), 6, RateLayout.PER_SEASON),
        build_template("catch/entry5-2", ("Black 2", "White 2"), 5, RateLayout.PER_SEASON),
        build_template("catch/entry6", ("X", "Y"), 5),
        build_template("catch/entry7", ("Sun", "Moon"), 5, RateLayout.PER_VERSION),
        build_template("catch/entry8", ("Sword", "Shield"), 5),
        build_template(
            "catch/entrybdsp", ("Brilliant Diamond", "Shining Pearl"), 5, RateLayout.PER_TIME
        ),
        build_template("catch/entryfl", ("FireRed", "LeafGreen"), 5),
        build_template("catch/entryhs", ("HeartGold", "SoulSilver"), 5, RateLayout.PER_TIME),
        build_template("catch/entryoras", ("Omega Ruby", "Alpha Sapphire"), 5),
        build_template("catch/entrype", ("Let's Go Pikachu", "Let's Go Eevee"), 5),
        build_template("catch/entryusum", ("Ultra Sun", "Ultra Moon"), 5, RateLayout.PER_VERSION),
    )
}

# Variants with their own readers, and the labels of their games.
RAID_TEMPLATE = "catch/entry8/raid"
SV_TEMPLATE = "catch/entry9"
SV_SPECIAL_TEMPLATE = "catch/entry9/special"
LA_TEMPLATE = "catch/entryla"
ZA_TEMPLATE = "catch/entryza"
RAID_VERSIONS = ("Sword", "Shield")
SV_VERSIONS = ("Scarlet", "Violet")
LA_VERSIONS = ("Legends: Arceus",)
ZA_VERSIONS = ("Legends: Z-A",)

# Headers whose tables are side games (Dynamax Adventures); their rows are skipped.
SIDE_GAME_HEADERS = frozenset({"catch/header/dn"})
HEADER_PREFIX = "catch/header"
FOOTER_PREFIX = "catch/footer"


class CatchTableError(Exception):
    """A catch template names a version the A-Box lacks."""


def list_catch_versions() -> frozenset[str]:
    labels = {label for template in CATCH_TEMPLATES.values() for label in template.version_labels}
    return frozenset(labels | set(RAID_VERSIONS + SV_VERSIONS + LA_VERSIONS + ZA_VERSIONS))


def check_catch_templates(versions: Collection[str]) -> None:
    """Raise `CatchTableError` when a template names a version label outside `versions`, the
    Version labels of the A-Box."""
    missing = sorted(list_catch_versions() - set(versions))
    if missing:
        raise CatchTableError(f"Catch templates name versions the A-Box lacks: {missing}")
